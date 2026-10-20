"""Embedder ablations on a graded benchmark: run named *arms* over one corpus.

An ablation asks "which change to the embedder moved the metric?". Each **arm**
is one embedding configuration (a query-side embedder, optionally a different
document-side one); every arm ranks the *same* segmented corpus for the *same*
queries and is scored by :func:`~ef.evaluation.evaluate_retrieval`, so the only
thing that varies is the embedder.

Simple use, an A/B over any two embedders::

    bench = read_graded_cases("cases.jsonl", "named_sets.json")
    report = run_arms(corpus, {"A": Arm(emb_a), "B": Arm(emb_b)}, bench)
    print(report)

The instruction-embedding experiment of ef#34 is one instance:
:func:`instruction_arms` builds its four arms (A: plain model, B: plain model +
instruction, the control, C: instruction-tuned model + instruction, C': the same
model without it) and :func:`instruction_verdict` checks its success line.

Ranking is dense and document-level: every segment is embedded once per
*document-side* embedder (``input_type="document"``), a query is embedded with
the arm's embedder (``input_type="query"``), and a document scores the best
cosine of any of its segments (max-pooling). Arms that share a document-side
embedder share its vectors: pass ``doc_vectors`` (any ``MutableMapping``, e.g. a
``dol`` store) to keep them across calls too. Only the query side differs between
an instructed arm and its bare twin, so the corpus is embedded once per model.

The labels format :func:`read_graded_cases` reads is the one ``ir.eval`` writes:
a JSONL of ``{"artifact_id", "labels": {theme: level}}`` cases with an optional
``{"__meta__": {"probes": {theme: query}}}`` header, plus an optional
``{theme: {"distractors": [...], "hard_positives": [...]}}`` JSON of named sets.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ef.evaluation import BenchmarkReport, RetrievalEvalReport, evaluate_retrieval

__all__ = [
    "DFLT_GRADES",
    "DFLT_TOPICAL_INSTRUCTION",
    "Arm",
    "GradedBenchmark",
    "MaxPoolRetriever",
    "instruction_arms",
    "instruction_verdict",
    "read_graded_cases",
    "run_arms",
    "segment_corpus",
]

#: Graded level -> relevance. Levels absent here count as not relevant, so
#: recall is "core/strong recall".
DFLT_GRADES: Mapping[str, float] = {"core": 2.0, "strong": 1.0}

#: The topical instruction for the instruction-tuned arm (Qwen3 / E5-instruct
#: format: ``Instruct: <task>\nQuery:``). It asks for what a document is *about*,
#: which is what a plain model confuses with surface keywords.
DFLT_TOPICAL_INSTRUCTION = (
    "Instruct: Given a topic, retrieve software packages whose main purpose is "
    "that topic, not packages that only mention related words\nQuery:"
)

#: The metrics the control arm must not move by more than the tolerance.
DFLT_CONTROL_METRICS = ("ndcg", "recall", "fp_rate")


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GradedBenchmark:
    """Queries, graded qrels and the named diagnostic sets, keyed by query id.

    >>> b = GradedBenchmark(qrels={'t': {'d1': 2.0, 'd9': 1.0}}, queries={'t': 'x'},
    ...                     distractors={'t': ('d2', 'd8')})
    >>> b.restrict(['d1', 'd2']).qrels, b.restrict(['d1', 'd2']).distractors
    ({'t': {'d1': 2.0}}, {'t': ('d2',)})
    """

    qrels: Mapping[str, Mapping[str, float]]
    queries: Mapping[str, str]
    distractors: Mapping[str, Sequence[str]] = field(default_factory=dict)
    hard_positives: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def restrict(self, doc_ids: Iterable[str]) -> GradedBenchmark:
        """The same benchmark with every judgement limited to ``doc_ids``.

        Use it when some labelled documents could not be put in the corpus: a
        document that cannot be retrieved would otherwise count as a miss for
        every arm alike, diluting the differences.
        """
        keep = set(doc_ids)

        def only(named):
            return {q: tuple(d for d in ds if d in keep) for q, ds in named.items()}

        return GradedBenchmark(
            qrels={q: {d: g for d, g in r.items() if d in keep} for q, r in self.qrels.items()},
            queries=dict(self.queries),
            distractors=only(self.distractors),
            hard_positives=only(self.hard_positives),
        )


def read_graded_cases(
    cases_path: str | Path,
    named_sets_path: str | Path | None = None,
    *,
    grades: Mapping[str, float] = DFLT_GRADES,
    probes: Mapping[str, str] | None = None,
) -> GradedBenchmark:
    """Load an ``ir.eval``-style graded case file (and named sets) as a benchmark.

    Each theme is one query: its text comes from ``probes`` or, failing that, the
    ``__meta__`` header's ``probes``; its qrels are the cases whose level for that
    theme is in ``grades``.

    Args:
        cases_path: JSONL of ``{"artifact_id", "labels": {theme: level}}``.
        named_sets_path: optional JSON ``{theme: {"distractors", "hard_positives"}}``.
        grades: level -> relevance grade; other levels are not relevant.
        probes: ``{theme: query text}``, overriding the header's.

    Raises:
        ValueError: when a labelled theme has no query text.
    """
    meta: dict[str, Any] = {}
    qrels: dict[str, dict[str, float]] = {}
    for line in Path(cases_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "__meta__" in record:
            meta = record["__meta__"]
            continue
        for theme, level in (record.get("labels") or {}).items():
            graded = qrels.setdefault(theme, {})
            if level in grades:
                graded[str(record["artifact_id"])] = float(grades[level])
    queries = {**(meta.get("probes") or {}), **(probes or {})}
    missing = sorted(set(qrels) - set(queries))
    if missing:
        raise ValueError(f"No query text for theme(s) {missing}: pass probes=")
    named: Mapping[str, Mapping[str, Any]] = {}
    if named_sets_path is not None:
        named = json.loads(Path(named_sets_path).read_text(encoding="utf-8"))
    return GradedBenchmark(
        qrels=qrels,
        queries={t: queries[t] for t in qrels},
        distractors={t: tuple(s.get("distractors", ())) for t, s in named.items()},
        hard_positives={t: tuple(s.get("hard_positives", ())) for t, s in named.items()},
    )


# ---------------------------------------------------------------------------
# Arms and the max-pool retriever
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Arm:
    """One embedding configuration: ``embedder`` for queries, ``doc_embedder`` for documents.

    ``doc_embedder`` defaults to ``embedder``. Give an instructed arm its bare
    twin as ``doc_embedder`` when the instruction is query-side only, so the
    corpus vectors are shared rather than recomputed.
    """

    embedder: Any
    doc_embedder: Any = None

    @property
    def documents(self) -> Any:
        return self.embedder if self.doc_embedder is None else self.doc_embedder


def segment_corpus(
    corpus: Mapping[str, str | Sequence[str]], *, segmenter: Any = None
) -> dict[str, list[str]]:
    """``{doc_id: [segment text, ...]}``: strings are segmented, sequences kept.

    ``segmenter`` is anything :func:`ef.as_segmenter` accepts (default: the
    recursive character splitter). Empty segments are dropped; a document left
    with none is represented by its id, so it stays rankable.

    >>> segment_corpus({'a': ['x', ''], 'b': ''})
    {'a': ['x'], 'b': ['b']}
    """
    seg = None
    out: dict[str, list[str]] = {}
    for doc_id, value in corpus.items():
        if isinstance(value, str):
            if seg is None:
                from ef.segmenter_adapters import as_segmenter

                seg = as_segmenter(segmenter)
            texts = [s["text"] for s in seg(value)] if value.strip() else []
        else:
            texts = list(value)
        out[str(doc_id)] = [t for t in texts if t and t.strip()] or [str(doc_id)]
    return out


def _unit_rows(vectors: Any) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class MaxPoolRetriever:
    """Rank documents by the best cosine of any of their segments to the query.

    >>> from ef.embedders import HashingEmbedder
    >>> emb = HashingEmbedder()
    >>> segs = {'fruit': ['red apple', 'ripe pear'], 'car': ['blue car engine']}
    >>> r = MaxPoolRetriever.from_segments(segs, emb)
    >>> r('apple', limit=1)
    ['fruit']
    """

    def __init__(self, doc_ids: Sequence[str], vectors: np.ndarray, query_embedder: Any):
        self.doc_ids = list(doc_ids)  # one per row of vectors (repeated per segment)
        self.vectors = _unit_rows(vectors)
        self.query_embedder = query_embedder

    @classmethod
    def from_segments(
        cls, segments: Mapping[str, Sequence[str]], embedder: Any, *, query_embedder: Any = None
    ) -> MaxPoolRetriever:
        owners = [d for d, texts in segments.items() for _ in texts]
        texts = [t for ts in segments.values() for t in ts]
        vectors = embedder(texts, input_type="document")
        return cls(owners, vectors, query_embedder or embedder)

    def __call__(self, query: str, *, limit: int = 10) -> list[str]:
        q = _unit_rows(self.query_embedder([query], input_type="query"))[0]
        best: dict[str, float] = {}
        for doc_id, score in zip(self.doc_ids, self.vectors @ q):
            if score > best.get(doc_id, -np.inf):
                best[doc_id] = float(score)
        return sorted(best, key=best.__getitem__, reverse=True)[:limit]


def _doc_key(embedder: Any) -> str:
    return getattr(embedder, "model_id", None) or f"object:{id(embedder)}"


def run_arms(
    corpus: Mapping[str, str | Sequence[str]],
    arms: Mapping[str, Arm | Any],
    benchmark: GradedBenchmark,
    *,
    k_values: Sequence[int] = (10, 20),
    segmenter: Any = None,
    doc_vectors: MutableMapping[str, np.ndarray] | None = None,
) -> BenchmarkReport:
    """Score every arm on the same segmented corpus; one row per arm.

    Args:
        corpus: ``{doc_id: text}`` (segmented with ``segmenter``) or
            ``{doc_id: [segment, ...]}`` (used as is).
        arms: ``{name: Arm}``; a bare embedder is taken as ``Arm(embedder)``.
        benchmark: what to score against (:func:`read_graded_cases`).
        k_values: rank cutoffs.
        segmenter: for string documents; see :func:`segment_corpus`.
        doc_vectors: cache of segment matrices keyed by the document-side
            embedder's ``model_id`` (a fresh dict by default).

    >>> from ef.embedders import HashingEmbedder
    >>> bench = GradedBenchmark({'q': {'fruit': 1.0}}, {'q': 'apple'})
    >>> rep = run_arms({'fruit': 'red apple', 'car': 'blue car'},
    ...                {'A': HashingEmbedder()}, bench, k_values=(1,))
    >>> rep.rows[0]['embedder'], rep.rows[0]['recall@1']
    ('A', 1.0)
    """
    segments = segment_corpus(corpus, segmenter=segmenter)
    owners = [d for d, texts in segments.items() for _ in texts]
    texts = [t for ts in segments.values() for t in ts]
    cache = {} if doc_vectors is None else doc_vectors
    rows: list[dict[str, Any]] = []
    reports: dict[str, RetrievalEvalReport] = {}
    for name, arm in arms.items():
        arm = arm if isinstance(arm, Arm) else Arm(arm)
        key = _doc_key(arm.documents)
        if key not in cache:
            cache[key] = np.asarray(arm.documents(texts, input_type="document"))
        retriever = MaxPoolRetriever(owners, cache[key], arm.embedder)
        report = evaluate_retrieval(
            retriever,
            benchmark.qrels,
            benchmark.queries,
            k_values=k_values,
            limit=len(segments),
            distractors=benchmark.distractors,
            hard_positives=benchmark.hard_positives,
        )
        reports[name] = report
        rows.append({"embedder": name, **report.metrics})
    return BenchmarkReport(rows=rows, reports=reports)


# ---------------------------------------------------------------------------
# The ef#34 instruction experiment
# ---------------------------------------------------------------------------


def instruction_arms(
    base: Any,
    tuned: Any = None,
    *,
    instruction: str = DFLT_TOPICAL_INSTRUCTION,
    base_instruction: str | None = None,
) -> dict[str, Arm]:
    """The four arms of the instruction experiment (A, B, C, C').

    - ``A``: ``base`` as is (the plain baseline);
    - ``B``: ``base`` with the instruction on queries: the **control**, expected
      to move little, since a plain model was not trained to read instructions;
    - ``C``: ``tuned`` (an instruction-tuned model) with the instruction;
    - ``C'``: ``tuned`` without it, attributing C's gain to the instruction
      rather than to the bigger model.

    Instructions are query-side only (documents bare), so each model embeds the
    corpus once. ``base`` / ``tuned`` are embedders or :func:`ef.as_embedder`
    specs; without ``tuned`` only A and B are built.

    Args:
        base: the plain embedder (e.g. ``all-MiniLM-L6-v2``).
        tuned: the instruction-tuned embedder (e.g. Qwen3-Embedding-0.6B).
        instruction: the query prefix for C, verbatim (model separator included).
        base_instruction: the query prefix for B; defaults to ``instruction``.

    >>> from ef.embedders import HashingEmbedder
    >>> sorted(instruction_arms(HashingEmbedder(), HashingEmbedder()))
    ['A', 'B', 'C', "C'"]
    """
    from ef.embedder_adapters import as_embedder
    from ef.embedder_wrappers import InstructionEmbedder

    def pair(model, prefix):
        model = as_embedder(model)
        return Arm(model), Arm(InstructionEmbedder(model, query_instruction=prefix), model)

    arms: dict[str, Arm] = {}
    arms["A"], arms["B"] = pair(base, instruction if base_instruction is None else base_instruction)
    if tuned is not None:
        arms["C'"], arms["C"] = pair(tuned, instruction)
    return arms


def instruction_verdict(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    k: int = 20,
    fp_cut: float = 0.5,
    control_tolerance: float = 0.02,
    control_metrics: Sequence[str] = DFLT_CONTROL_METRICS,
) -> dict[str, Any]:
    """Check ef#34's success line from ``{arm: {"fp_rate@20": ..., ...}}``.

    Success = C cuts the named-distractor ``fp_rate@k`` by at least ``fp_cut``
    (relative) vs A, at equal-or-better ``recall@k`` (core/strong recall), and
    the control B moves every ``control_metrics`` at ``k`` by less than
    ``control_tolerance`` (absolute; 0.02 = 2 points). Also reports whether C
    beats its bare twin C' on ``fp_rate@k`` (the gain is the instruction's).

    Accepts :attr:`BenchmarkReport.rows` turned into a mapping, e.g.
    ``{r['embedder']: r for r in report.rows}``.

    >>> m = {'A': {'fp_rate@20': .2, 'recall@20': .4, 'ndcg@20': .7},
    ...      'B': {'fp_rate@20': .2, 'recall@20': .41, 'ndcg@20': .7},
    ...      'C': {'fp_rate@20': .05, 'recall@20': .5, 'ndcg@20': .8},
    ...      "C'": {'fp_rate@20': .15, 'recall@20': .45, 'ndcg@20': .75}}
    >>> v = instruction_verdict(m)
    >>> v['success'], round(v['fp_reduction'], 2), v['instruction_beats_bare']
    (True, 0.75, True)
    """
    a, c = metrics["A"], metrics["C"]
    fp, rec = f"fp_rate@{k}", f"recall@{k}"
    fp_reduction = (1 - c[fp] / a[fp]) if a[fp] > 0 else (0.0 if c[fp] == 0 else -np.inf)
    control_moves = {
        f"{m}@{k}": metrics["B"][f"{m}@{k}"] - a[f"{m}@{k}"]
        for m in control_metrics
        if f"{m}@{k}" in a and f"{m}@{k}" in metrics["B"]
    }
    verdict = {
        "fp_reduction": float(fp_reduction),
        "fp_cut_met": bool(fp_reduction >= fp_cut),
        "recall_delta": float(c[rec] - a[rec]),
        "recall_held": bool(c[rec] >= a[rec]),
        "control_moves": control_moves,
        "control_held": all(abs(d) < control_tolerance for d in control_moves.values()),
    }
    if "C'" in metrics:
        verdict["instruction_beats_bare"] = bool(c[fp] < metrics["C'"][fp])
    verdict["success"] = verdict["fp_cut_met"] and verdict["recall_held"] and verdict["control_held"]
    return verdict
