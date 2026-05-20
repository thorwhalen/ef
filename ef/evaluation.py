"""Evaluation hookpoints — measuring retrieval and RAG quality.

Phase 7 makes an indexed ``ef`` corpus **measurable**. It stays a *facade*: it
computes the standard, deterministic metrics and provides the standard data
shapes, but it never synthesizes an answer and never requires an LLM (the §6
boundary). LLM-judged metrics are reached through a thin Ragas bridge.

Two journeys (``ef_use_cases.md`` §F7–F8), two evaluation conventions
(``ef_design_notes.md`` §6):

- **Retrieval evaluation** — :func:`evaluate_retrieval`, BEIR/MTEB-shaped. The
  inputs are the BEIR triple — a corpus, a set of ``queries`` and ``qrels``
  (graded relevance judgements) — and the primary metric is **NDCG@10**. The
  metric primitives (:func:`ndcg_at_k`, :func:`recall_at_k`, …) are pure
  functions, usable on their own. :func:`read_beir` / :func:`write_beir` move
  the triple to and from the on-disk ``corpus.jsonl`` / ``queries.jsonl`` /
  ``qrels.tsv`` format.

- **RAG evaluation** — :func:`evaluate_rag`, Ragas ``SingleTurnSample``-shaped.
  A sample is ``(user_input, response, retrieved_contexts, reference)``;
  :class:`RagSample` names that shape. :func:`evaluate_rag` computes only
  **deterministic, reference-based lexical metrics** (:func:`exact_match`,
  :func:`token_f1`, :func:`context_recall`, :func:`context_precision`) — no LLM.
  For the LLM-judged metrics (faithfulness, answer relevancy, …) hand the
  samples to :func:`as_ragas_dataset` and run Ragas with your own LLM.

Example — score a retriever against a tiny BEIR-shaped triple:

>>> qrels = {'q1': {'d1': 1.0, 'd2': 1.0}, 'q2': {'d3': 1.0}}
>>> queries = {'q1': 'first query', 'q2': 'second query'}
>>> rankings = {'first query': ['d1', 'd9', 'd2'], 'second query': ['d9', 'd3']}
>>> def retriever(query, *, limit=10):
...     return rankings[query][:limit]
>>> report = evaluate_retrieval(retriever, qrels, queries, k_values=(1, 3))
>>> report.metrics['recall@3']
1.0
>>> round(report.primary_at(1), 3)        # NDCG@1, averaged over the two queries
0.5
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, TypedDict

__all__ = [
    # data shapes
    "Qrels",
    "RagSample",
    # retrieval metric primitives
    "dcg_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "average_precision",
    # retrieval evaluation
    "RetrievalEvalReport",
    "evaluate_retrieval",
    # BEIR interchange
    "read_beir",
    "write_beir",
    # RAG metric primitives
    "exact_match",
    "token_f1",
    "context_recall",
    "context_precision",
    # RAG evaluation
    "RagEvalReport",
    "evaluate_rag",
    "as_ragas_dataset",
]


# ===========================================================================
# Data shapes
# ===========================================================================

#: Graded relevance judgements — ``{query_id: {doc_id: relevance}}``. A
#: ``relevance`` of ``0`` means *judged non-relevant*; ``> 0`` means relevant
#: (graded judgements feed :func:`ndcg_at_k`, binary relevance the rest). This
#: is the in-memory form of a BEIR ``qrels.tsv``.
Qrels = Mapping[str, Mapping[str, float]]


class RagSample(TypedDict, total=False):
    """One end-to-end RAG interaction — the unit :func:`evaluate_rag` scores.

    The field names match Ragas' ``SingleTurnSample`` exactly, so a
    :class:`RagSample` is the interchange shape ``app_ef`` / ``srag`` / ``raglab``
    produce and :func:`as_ragas_dataset` forwards unchanged. ``ef`` itself fills
    only ``retrieved_contexts`` (from :meth:`~ef.source_manager.SourceManager.retrieve`);
    ``response`` is the *caller's* LLM output — ``ef`` does not synthesize it.

    Keys:
        user_input: the user's question.
        response: the generated answer (the caller's LLM produced it).
        retrieved_contexts: the context passages handed to the LLM — what
            :meth:`~ef.source_manager.SourceManager.retrieve` returned, as text.
        reference: the ground-truth answer, if known.
        reference_contexts: the ground-truth context passages, if known.
    """

    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    reference_contexts: list[str]


# ===========================================================================
# Retrieval metric primitives — pure functions over a ranking + judgements
# ===========================================================================
#
# Every metric takes ``ranked`` (doc ids best-first, already de-duplicated) and
# ``relevant`` (a ``{doc_id: relevance}`` mapping) and returns a float in
# ``[0, 1]``. They are deterministic and dependency-free — the testable core
# that :func:`evaluate_retrieval` only aggregates.


def _positives(relevant: Mapping[str, float]) -> set[str]:
    """The doc ids judged relevant — those with a strictly positive grade."""
    return {doc_id for doc_id, grade in relevant.items() if grade > 0}


def dcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Discounted cumulative gain of ``ranked`` at cutoff ``k``.

    ``sum(grade_i / log2(i + 2))`` over the top ``k`` results (``i`` 0-indexed),
    with ``grade_i`` the relevance of the ``i``-th doc (``0`` if unjudged) — the
    linear-gain DCG that BEIR / ``pytrec_eval`` use.

    >>> round(dcg_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 3), 4)
    1.5
    """
    return sum(
        grade / math.log2(i + 2)
        for i, doc_id in enumerate(ranked[:k])
        if (grade := float(relevant.get(doc_id, 0.0)))
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Normalized DCG at cutoff ``k`` — ``ef``'s **primary** retrieval metric.

    :func:`dcg_at_k` divided by the ideal DCG (the DCG of the best possible
    ranking of the judged docs). ``1.0`` is a perfect ranking; ``0.0`` when no
    relevant doc was judged.

    >>> ndcg_at_k(['d1', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)   # perfect order
    1.0
    >>> round(ndcg_at_k(['x', 'd1'], {'d1': 1.0}, 2), 4)     # relevant doc at rank 2
    0.6309
    """
    ideal = sorted((float(g) for g in relevant.values() if g > 0), reverse=True)
    idcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(ideal[:k]))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Fraction of the relevant docs that appear in the top ``k`` results.

    >>> recall_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)
    0.5
    >>> recall_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 3)
    1.0
    """
    positives = _positives(relevant)
    if not positives:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if doc_id in positives)
    return found / len(positives)


def precision_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float], k: int
) -> float:
    """Fraction of the top ``k`` results that are relevant.

    >>> precision_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)
    0.5
    """
    if k <= 0:
        return 0.0
    positives = _positives(relevant)
    found = sum(1 for doc_id in ranked[:k] if doc_id in positives)
    return found / k


def reciprocal_rank(
    ranked: Sequence[str], relevant: Mapping[str, float], k: int | None = None
) -> float:
    """Reciprocal of the rank of the first relevant result (``0`` if none).

    Averaged over queries this is MRR. ``k`` optionally caps how far down the
    ranking to look.

    >>> reciprocal_rank(['x', 'd1', 'd2'], {'d1': 1.0})
    0.5
    >>> reciprocal_rank(['x', 'd1'], {'d1': 1.0}, k=1)       # relevant doc past the cutoff
    0.0
    """
    positives = _positives(relevant)
    cutoff = len(ranked) if k is None else k
    for i, doc_id in enumerate(ranked[:cutoff]):
        if doc_id in positives:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    ranked: Sequence[str], relevant: Mapping[str, float], k: int | None = None
) -> float:
    """Average precision of ``ranked`` — the per-query term of MAP.

    The mean of the precision measured at each relevant result's rank. ``k``
    optionally caps the ranking.

    >>> average_precision(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0})
    0.8333333333333333
    """
    positives = _positives(relevant)
    if not positives:
        return 0.0
    cutoff = len(ranked) if k is None else k
    found = 0
    score = 0.0
    for i, doc_id in enumerate(ranked[:cutoff]):
        if doc_id in positives:
            found += 1
            score += found / (i + 1)
    return score / len(positives)


#: The retrieval metrics :func:`evaluate_retrieval` can compute, each keyed by
#: the short name used in the ``metrics=`` argument and in report keys.
_RETRIEVAL_METRICS: dict[
    str, Callable[[Sequence[str], Mapping[str, float], int], float]
] = {
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": reciprocal_rank,
    "map": average_precision,
}


# ===========================================================================
# Retrieval evaluation
# ===========================================================================


@dataclass(frozen=True, slots=True)
class RetrievalEvalReport:
    """The outcome of :func:`evaluate_retrieval` — metrics over a query set.

    ``metrics`` maps a ``"<name>@<k>"`` key (e.g. ``"ndcg@10"``) to the mean of
    that metric over every scored query; ``per_query`` keeps the same keys per
    query for drill-down. ``n_queries`` is how many queries were actually
    scored (a query with no positive judgement is skipped).

    >>> report = RetrievalEvalReport(
    ...     metrics={'ndcg@10': 0.83}, per_query={}, n_queries=12, k_values=(10,))
    >>> report.primary
    0.83
    """

    metrics: Mapping[str, float]
    per_query: Mapping[str, Mapping[str, float]]
    n_queries: int
    k_values: tuple[int, ...]

    @property
    def primary(self) -> float | None:
        """The headline number — mean NDCG@10, or ``None`` if it was not computed."""
        return self.metrics.get("ndcg@10")

    def primary_at(self, k: int) -> float | None:
        """Mean NDCG at cutoff ``k`` — ``None`` if NDCG@``k`` was not computed."""
        return self.metrics.get(f"ndcg@{k}")

    def __str__(self) -> str:
        head = f"RetrievalEvalReport — {self.n_queries} queries"
        body = "\n".join(f"  {key}: {value:.4f}" for key, value in self.metrics.items())
        return f"{head}\n{body}" if body else head


def _result_doc_id(result: Any) -> str:
    """Extract the *source document id* a single retrieval result points at.

    Handles every shape an ``ef`` retriever yields: a :class:`~ef.source_manager.SearchHit`
    (its ``source_id``), a plain :class:`~ef.segments.Segment` from
    :meth:`~ef.source_manager.SourceManager.retrieve` (``metadata["source"]``,
    falling back to the segment ``id``), or a bare string id.
    """
    source_id = getattr(result, "source_id", None)
    if source_id is not None:
        return str(source_id)
    segment = getattr(result, "segment", None)
    if segment is not None:
        result = segment
    if isinstance(result, Mapping):
        metadata = result.get("metadata") or {}
        if metadata.get("source") is not None:
            return str(metadata["source"])
        if result.get("id") is not None:
            return str(result["id"])
    return str(result)


def _dedup(doc_ids: Iterable[str]) -> list[str]:
    """De-duplicate ``doc_ids`` keeping the first (best-ranked) occurrence.

    One source document is often cut into several segments; a retriever can
    return several of them. For doc-level evaluation only the best rank of each
    source counts.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def _retrieve_ranking(retriever: Any, query: str, limit: int) -> list[Any]:
    """Run ``retriever`` for one ``query`` — accepts a search facade or a callable.

    A :class:`~ef.source_manager.SearchableCorpus` / :class:`~ef.source_manager.SourceManager`
    is queried through its ``.search`` method; any other callable is called as
    ``retriever(query, limit=limit)`` (falling back to ``retriever(query)`` for
    a callable that takes no ``limit``).
    """
    search = getattr(retriever, "search", None)
    if callable(search):
        return list(search(query, limit=limit))
    if callable(retriever):
        try:
            return list(retriever(query, limit=limit))
        except TypeError:
            return list(retriever(query))
    raise TypeError(
        f"Cannot use {retriever!r} as a retriever. Pass a SearchableCorpus, a "
        f"SourceManager, or a callable query -> ranked results."
    )


def evaluate_retrieval(
    retriever: Any,
    qrels: Qrels,
    queries: Mapping[str, str],
    *,
    k_values: Sequence[int] = (1, 5, 10, 100),
    metrics: Sequence[str] = ("ndcg", "recall", "precision", "mrr", "map"),
    limit: int | None = None,
) -> RetrievalEvalReport:
    """Score a retriever against a BEIR-shaped evaluation set.

    For every query judged in ``qrels``, the ``retriever`` is run, its results
    are mapped back to *source document ids* (:func:`_result_doc_id`,
    de-duplicated to the best rank per source), and each requested metric is
    computed at each cutoff in ``k_values``. The report holds the mean of each
    ``"<metric>@<k>"`` over all scored queries — primary metric **NDCG@10**.

    The natural pairing with :func:`~ef.source_manager.ingest`::

        corpus, queries, qrels = read_beir("path/to/beir/dataset")
        index = ingest(corpus, embedder="st:all-MiniLM-L6-v2")
        report = evaluate_retrieval(index, qrels, queries)
        report.primary            # mean NDCG@10

    Args:
        retriever: what answers a query — a :class:`~ef.source_manager.SearchableCorpus`,
            a :class:`~ef.source_manager.SourceManager`, or any callable
            ``query -> ranked results``. Results may be
            :class:`~ef.source_manager.SearchHit`\\ s, :class:`~ef.segments.Segment`\\ s
            or bare doc-id strings.
        qrels: the relevance judgements — ``{query_id: {doc_id: relevance}}``.
            Queries with no positive judgement are skipped.
        queries: the query texts — ``{query_id: query_text}``. A judged query
            absent here cannot be run and is skipped.
        k_values: the rank cutoffs to evaluate at.
        metrics: which metrics to compute — any of ``ndcg`` / ``recall`` /
            ``precision`` / ``mrr`` / ``map``.
        limit: how many results to retrieve per query. ``None`` uses
            ``max(k_values)``; raise it when sources segment into many pieces
            (de-duplication to source level can otherwise shorten the ranking).

    Returns:
        a :class:`RetrievalEvalReport`.

    Raises:
        ValueError: for an unknown metric name, or if no query can be scored.

    >>> qrels = {'q1': {'d1': 1.0}, 'q2': {'d2': 1.0}}
    >>> queries = {'q1': 'alpha', 'q2': 'beta'}
    >>> def retriever(query, *, limit=10):
    ...     return {'alpha': ['d1', 'd2'], 'beta': ['d1', 'd2']}[query][:limit]
    >>> report = evaluate_retrieval(retriever, qrels, queries, k_values=(1, 2))
    >>> report.metrics['recall@2'], report.n_queries
    (1.0, 2)
    """
    unknown = [name for name in metrics if name not in _RETRIEVAL_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown retrieval metric(s) {unknown}. "
            f"Choose from {sorted(_RETRIEVAL_METRICS)}."
        )
    if limit is None:
        limit = max(k_values, default=10)

    scorable = sorted(qid for qid in qrels if qid in queries and _positives(qrels[qid]))
    if not scorable:
        raise ValueError(
            "No query could be scored: every judged query is either absent "
            "from `queries` or has no positively-judged document."
        )

    per_query: dict[str, dict[str, float]] = {}
    for qid in scorable:
        results = _retrieve_ranking(retriever, queries[qid], limit)
        ranking = _dedup(_result_doc_id(result) for result in results)
        relevant = {str(doc): float(grade) for doc, grade in qrels[qid].items()}
        scores: dict[str, float] = {}
        for name in metrics:
            metric_fn = _RETRIEVAL_METRICS[name]
            for k in k_values:
                scores[f"{name}@{k}"] = metric_fn(ranking, relevant, k)
        per_query[qid] = scores

    metric_keys = [f"{name}@{k}" for name in metrics for k in k_values]
    aggregated = {
        key: mean(per_query[qid][key] for qid in per_query) for key in metric_keys
    }
    return RetrievalEvalReport(
        metrics=aggregated,
        per_query=per_query,
        n_queries=len(per_query),
        k_values=tuple(k_values),
    )


# ===========================================================================
# BEIR interchange — read / write corpus.jsonl, queries.jsonl, qrels.tsv
# ===========================================================================

#: The qrels file paths :func:`read_beir` looks for, in order, when the caller
#: does not name one — a bare ``qrels.tsv`` or BEIR's ``qrels/<split>.tsv``.
_DEFAULT_QRELS_PATHS = ("qrels.tsv", "qrels/test.tsv", "qrels/dev.tsv")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield the JSON object on each non-blank line of a ``.jsonl`` file."""
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            line = line.strip()
            if line:
                yield json.loads(line)


def _beir_doc_text(record: Mapping[str, Any]) -> str:
    """The text of one BEIR corpus record — ``title`` prepended to ``text``."""
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()
    return f"{title}\n{text}".strip() if title else text


def read_beir(
    directory: str, *, qrels: str | None = None
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, float]]]:
    """Load a BEIR-format dataset from disk into the in-memory evaluation triple.

    Reads ``corpus.jsonl`` (records ``{"_id", "title", "text"}``),
    ``queries.jsonl`` (records ``{"_id", "text"}``) and a ``qrels`` TSV
    (``query-id``, ``corpus-id``, ``score`` columns). The corpus text is the
    record's ``title`` and ``text`` joined — the BEIR convention.

    Args:
        directory: the dataset directory.
        qrels: the qrels file, relative to ``directory``. ``None`` tries
            :data:`_DEFAULT_QRELS_PATHS` in order.

    Returns:
        a ``(corpus, queries, qrels)`` triple — ``corpus`` and ``queries`` are
        ``{id: text}`` dicts, ``qrels`` is ``{query_id: {doc_id: relevance}}``.
        Feed it straight to :func:`evaluate_retrieval` (after indexing ``corpus``
        with :func:`~ef.source_manager.ingest`).

    Raises:
        FileNotFoundError: if ``corpus.jsonl``, ``queries.jsonl`` or the qrels
            file is missing.
    """
    root = Path(directory)
    corpus = {
        str(rec.get("_id", rec.get("id"))): _beir_doc_text(rec)
        for rec in _read_jsonl(root / "corpus.jsonl")
    }
    queries = {
        str(rec.get("_id", rec.get("id"))): (rec.get("text") or "")
        for rec in _read_jsonl(root / "queries.jsonl")
    }
    qrels_path = _resolve_qrels_path(root, qrels)
    judgements = _read_qrels_tsv(qrels_path)
    return corpus, queries, judgements


def _resolve_qrels_path(root: Path, qrels: str | None) -> Path:
    """Resolve the qrels file under ``root`` — an explicit name or the defaults."""
    if qrels is not None:
        path = root / qrels
        if not path.exists():
            raise FileNotFoundError(f"qrels file not found: {path}")
        return path
    for candidate in _DEFAULT_QRELS_PATHS:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No qrels file under {root} (looked for {list(_DEFAULT_QRELS_PATHS)}). "
        f"Pass qrels='<relative/path.tsv>'."
    )


def _read_qrels_tsv(path: Path) -> dict[str, dict[str, float]]:
    """Parse a qrels TSV — a leading non-numeric header row is skipped."""
    judgements: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as rows:
        for row in rows:
            fields = row.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            query_id, doc_id, raw_score = fields[0], fields[1], fields[2]
            try:
                score = float(raw_score)
            except ValueError:
                continue  # the header row (`query-id  corpus-id  score`)
            judgements.setdefault(query_id, {})[doc_id] = score
    return judgements


def write_beir(
    directory: str,
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    qrels: Qrels,
) -> None:
    """Write an evaluation triple to disk in BEIR format — the inverse of :func:`read_beir`.

    Creates ``directory`` (and parents) and writes ``corpus.jsonl``,
    ``queries.jsonl`` and ``qrels.tsv``. Use it to snapshot an ``ef`` corpus as
    a reusable benchmark (``ef_use_cases.md`` §H4).

    Args:
        directory: the output directory — created if absent.
        corpus: the documents — ``{doc_id: text}``.
        queries: the queries — ``{query_id: query_text}``.
        qrels: the relevance judgements — ``{query_id: {doc_id: relevance}}``.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "corpus.jsonl").open("w", encoding="utf-8") as out:
        for doc_id, text in corpus.items():
            out.write(
                json.dumps({"_id": str(doc_id), "title": "", "text": text}) + "\n"
            )
    with (root / "queries.jsonl").open("w", encoding="utf-8") as out:
        for query_id, text in queries.items():
            out.write(json.dumps({"_id": str(query_id), "text": text}) + "\n")
    with (root / "qrels.tsv").open("w", encoding="utf-8") as out:
        out.write("query-id\tcorpus-id\tscore\n")
        for query_id, judged in qrels.items():
            for doc_id, score in judged.items():
                out.write(f"{query_id}\t{doc_id}\t{score}\n")


# ===========================================================================
# RAG metric primitives — deterministic, reference-based, no LLM
# ===========================================================================

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def _normalize_answer(text: str) -> str:
    """Normalize a free-text answer for lexical comparison (the SQuAD recipe).

    Lower-cases, drops punctuation, removes the articles ``a`` / ``an`` / ``the``
    and collapses whitespace — so ``"The Eiffel Tower."`` and ``"eiffel tower"``
    compare equal.

    >>> _normalize_answer('The Eiffel Tower.')
    'eiffel tower'
    """
    text = text.lower().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(response: str, reference: str) -> float:
    """``1.0`` if ``response`` equals ``reference`` after normalization, else ``0.0``.

    >>> exact_match('The Eiffel Tower', 'eiffel tower')
    1.0
    >>> exact_match('Paris', 'London')
    0.0
    """
    return float(_normalize_answer(response) == _normalize_answer(reference))


def token_f1(response: str, reference: str) -> float:
    """Token-overlap F1 of ``response`` against ``reference`` (the SQuAD F1).

    The harmonic mean of token precision and recall over the normalized token
    bags — a graded credit where :func:`exact_match` is all-or-nothing.

    >>> token_f1('the quick brown fox', 'a quick brown fox')
    1.0
    >>> round(token_f1('quick brown fox', 'quick brown dog'), 4)
    0.6667
    """
    predicted = _normalize_answer(response).split()
    expected = _normalize_answer(reference).split()
    if not predicted or not expected:
        return float(not predicted and not expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def context_recall(reference: str, retrieved_contexts: Sequence[str]) -> float:
    """Lexical recall of the ``reference`` tokens by the retrieved contexts.

    The fraction of the (normalized) reference's distinct tokens that appear
    anywhere in ``retrieved_contexts`` — a deterministic *proxy* for "did
    retrieval surface the information the answer needs". For the LLM-judged
    context recall, use :func:`as_ragas_dataset` and Ragas.

    >>> context_recall('paris france', ['Paris is the capital of France.'])
    1.0
    >>> context_recall('paris france', ['London is in England.'])
    0.0
    """
    expected = set(_normalize_answer(reference).split())
    if not expected:
        return 0.0
    covered: set[str] = set()
    for context in retrieved_contexts:
        covered |= set(_normalize_answer(context).split())
    return len(expected & covered) / len(expected)


def context_precision(reference: str, retrieved_contexts: Sequence[str]) -> float:
    """Fraction of the retrieved contexts that lexically overlap the ``reference``.

    A deterministic *proxy* for retrieval precision — a context counts as useful
    if it shares at least one (normalized) token with the reference answer.

    >>> context_precision('paris', ['Paris is in France.', 'Unrelated text.'])
    0.5
    """
    if not retrieved_contexts:
        return 0.0
    expected = set(_normalize_answer(reference).split())
    if not expected:
        return 0.0
    useful = sum(
        1
        for context in retrieved_contexts
        if expected & set(_normalize_answer(context).split())
    )
    return useful / len(retrieved_contexts)


def _m_exact_match(sample: Mapping[str, Any]) -> float | None:
    reference = sample.get("reference")
    if reference is None:
        return None
    return exact_match(sample.get("response") or "", reference)


def _m_token_f1(sample: Mapping[str, Any]) -> float | None:
    reference = sample.get("reference")
    if reference is None:
        return None
    return token_f1(sample.get("response") or "", reference)


def _m_context_recall(sample: Mapping[str, Any]) -> float | None:
    reference = sample.get("reference")
    if reference is None:
        return None
    return context_recall(reference, sample.get("retrieved_contexts") or [])


def _m_context_precision(sample: Mapping[str, Any]) -> float | None:
    reference = sample.get("reference")
    if reference is None:
        return None
    return context_precision(reference, sample.get("retrieved_contexts") or [])


def _m_non_empty_rate(sample: Mapping[str, Any]) -> float | None:
    return float(bool(sample.get("retrieved_contexts")))


#: The RAG metrics :func:`evaluate_rag` can compute. Each maps a sample to a
#: float, or to ``None`` when the metric does not apply (e.g. no ``reference``)
#: — a ``None`` sample is excluded from that metric's mean.
_RAG_METRICS: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
    "exact_match": _m_exact_match,
    "token_f1": _m_token_f1,
    "context_recall": _m_context_recall,
    "context_precision": _m_context_precision,
    "non_empty_rate": _m_non_empty_rate,
}


# ===========================================================================
# RAG evaluation
# ===========================================================================


@dataclass(frozen=True, slots=True)
class RagEvalReport:
    """The outcome of :func:`evaluate_rag` — lexical RAG metrics over a sample set.

    ``metrics`` maps each metric to its mean over the samples it applied to;
    ``coverage`` records how many samples that was (a metric needing a
    ``reference`` skips samples without one). ``n_samples`` is the total scored.

    >>> report = RagEvalReport(
    ...     metrics={'exact_match': 0.5}, coverage={'exact_match': 4}, n_samples=4)
    >>> report.metrics['exact_match']
    0.5
    """

    metrics: Mapping[str, float]
    coverage: Mapping[str, int]
    n_samples: int

    def __str__(self) -> str:
        head = f"RagEvalReport — {self.n_samples} samples"
        body = "\n".join(
            f"  {key}: {value:.4f} (n={self.coverage.get(key, 0)})"
            for key, value in self.metrics.items()
        )
        return f"{head}\n{body}" if body else head


def evaluate_rag(
    samples: Iterable[RagSample | Mapping[str, Any]],
    *,
    metrics: Sequence[str] = (
        "exact_match",
        "token_f1",
        "context_recall",
        "context_precision",
        "non_empty_rate",
    ),
) -> RagEvalReport:
    """Score RAG samples with deterministic, reference-based lexical metrics.

    Each sample is a :class:`RagSample` — ``(user_input, response,
    retrieved_contexts, reference)``. ``evaluate_rag`` computes only metrics
    that need **no LLM**: :func:`exact_match`, :func:`token_f1`,
    :func:`context_recall`, :func:`context_precision` and ``non_empty_rate``
    (the share of samples that retrieved any context). This keeps ``ef`` a
    facade — it never synthesizes an answer and never calls a model (§6).

    For the LLM-judged metrics — faithfulness, answer relevancy, … — pass the
    same samples to :func:`as_ragas_dataset` and run Ragas with your own LLM.

    Args:
        samples: an iterable of :class:`RagSample`\\ s (or plain mappings of the
            same shape).
        metrics: which metrics to compute — any of ``exact_match`` /
            ``token_f1`` / ``context_recall`` / ``context_precision`` /
            ``non_empty_rate``. The reference-based ones skip samples that carry
            no ``reference``.

    Returns:
        a :class:`RagEvalReport`.

    Raises:
        ValueError: for an unknown metric name, or if ``samples`` is empty.

    >>> samples = [
    ...     {'user_input': 'capital of France?', 'response': 'Paris',
    ...      'reference': 'Paris',
    ...      'retrieved_contexts': ['Paris is the capital of France.']},
    ...     {'user_input': '2 + 2?', 'response': 'five', 'reference': 'four',
    ...      'retrieved_contexts': []},
    ... ]
    >>> report = evaluate_rag(samples)
    >>> report.metrics['exact_match']
    0.5
    >>> report.metrics['non_empty_rate']
    0.5
    """
    unknown = [name for name in metrics if name not in _RAG_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown RAG metric(s) {unknown}. Choose from {sorted(_RAG_METRICS)}. "
            f"For LLM-judged metrics (faithfulness, answer relevancy, ...) use "
            f"as_ragas_dataset() and run Ragas."
        )
    rows = [dict(sample) for sample in samples]
    if not rows:
        raise ValueError("evaluate_rag got no samples.")

    aggregated: dict[str, float] = {}
    coverage: dict[str, int] = {}
    for name in metrics:
        metric_fn = _RAG_METRICS[name]
        values = [value for row in rows if (value := metric_fn(row)) is not None]
        coverage[name] = len(values)
        aggregated[name] = mean(values) if values else 0.0
    return RagEvalReport(metrics=aggregated, coverage=coverage, n_samples=len(rows))


def as_ragas_dataset(samples: Iterable[RagSample | Mapping[str, Any]]) -> Any:
    """Convert :class:`RagSample`\\ s into a Ragas ``EvaluationDataset``.

    The hookpoint for the *LLM-judged* RAG metrics ``ef`` deliberately does not
    compute itself (faithfulness, answer relevancy, …). :class:`RagSample`
    already mirrors Ragas' ``SingleTurnSample`` field-for-field, so this is a
    thin, lossless adapter. Run the evaluation with your own LLM::

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy

        dataset = as_ragas_dataset(samples)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    Args:
        samples: an iterable of :class:`RagSample`\\ s (or plain mappings).

    Returns:
        a ``ragas.EvaluationDataset``.

    Raises:
        ImportError: if the optional ``ragas`` package is not installed.
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "as_ragas_dataset needs the optional `ragas` package for LLM-judged "
            "RAG metrics. Install it with: pip install ragas"
        ) from exc

    turns = []
    for sample in samples:
        sample = dict(sample)
        turns.append(
            SingleTurnSample(
                user_input=sample.get("user_input"),
                response=sample.get("response"),
                retrieved_contexts=sample.get("retrieved_contexts"),
                reference=sample.get("reference"),
                reference_contexts=sample.get("reference_contexts"),
            )
        )
    return EvaluationDataset(samples=turns)
