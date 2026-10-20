"""Embedder ablations (ef#37): graded-case loading, arms, max-pool ranking, verdict.

A tiny synthetic fixture stands in for a real benchmark; the embedders are
deterministic toys, so no model is downloaded.
"""

import json

import numpy as np
import pytest

from ef.ablation import (
    Arm,
    GradedBenchmark,
    MaxPoolRetriever,
    instruction_arms,
    instruction_verdict,
    read_graded_cases,
    run_arms,
    segment_corpus,
)
from ef.embedder_wrappers import InstructionEmbedder
from ef.embedders import FunctionEmbedder

VOCAB = ("graph", "audio", "embed", "topic")


def _bag_embedder(calls=None, *, model_id="bag@4"):
    """Bag-of-keywords toy; ``TOPIC:`` in a text turns it toward the graph axis."""

    def embed(texts):
        texts = list(texts)
        if calls is not None:
            calls.append(texts)
        rows = []
        for t in texts:
            low = t.lower()
            row = [low.count(w) for w in VOCAB]
            if "topic:" in low:
                row[1] = 0  # the "instruction" suppresses the surface-cue axis
            rows.append(row)
        return np.asarray(rows, dtype=np.float32) + 1e-3

    return FunctionEmbedder(embed, model_id=model_id)


@pytest.fixture
def labels(tmp_path):
    cases = [
        {"__meta__": {"probes": {"graphs": "graph audio"}}},
        {"artifact_id": "nx", "labels": {"graphs": "core"}},
        {"artifact_id": "dag", "labels": {"graphs": "strong"}},
        {"artifact_id": "synth", "labels": {"graphs": "none"}},
        {"artifact_id": "misc", "labels": {"graphs": "tangential"}},
    ]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n".join(json.dumps(c) for c in cases))
    named_path = tmp_path / "named.json"
    named_path.write_text(
        json.dumps({"graphs": {"distractors": ["synth"], "hard_positives": ["dag"]}})
    )
    return cases_path, named_path


CORPUS = {
    "nx": ["graph library"],
    "dag": ["a graph of tasks to embed", "and some audio"],
    "synth": ["audio audio audio graph"],
    "misc": ["topic notes"],
}


def test_read_graded_cases(labels):
    bench = read_graded_cases(*labels)
    assert bench.qrels == {"graphs": {"nx": 2.0, "dag": 1.0}}
    assert bench.queries == {"graphs": "graph audio"}
    assert bench.distractors == {"graphs": ("synth",)}
    assert bench.hard_positives == {"graphs": ("dag",)}


def test_read_graded_cases_needs_a_probe(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"artifact_id": "a", "labels": {"t": "core"}}))
    with pytest.raises(ValueError, match="probes"):
        read_graded_cases(path)
    assert read_graded_cases(path, probes={"t": "q"}).queries == {"t": "q"}


def test_restrict_drops_unavailable_docs(labels):
    bench = read_graded_cases(*labels).restrict(["nx", "synth"])
    assert bench.qrels == {"graphs": {"nx": 2.0}}
    assert bench.hard_positives == {"graphs": ()}


def test_segment_corpus_segments_strings_and_keeps_lists():
    segs = segment_corpus({"a": "one two", "b": ["x", "y"], "c": "  "})
    assert segs["a"] == ["one two"] and segs["b"] == ["x", "y"] and segs["c"] == ["c"]


def test_max_pool_takes_best_segment():
    r = MaxPoolRetriever.from_segments(CORPUS, _bag_embedder())
    ranking = r("audio", limit=4)
    # dag's pure-audio segment beats synth's mostly-audio one: max, not mean, pooling
    assert ranking[:2] == ["dag", "synth"]


def test_arms_share_document_vectors(labels):
    calls = []
    base = _bag_embedder(calls)
    arms = {"A": Arm(base), "B": Arm(InstructionEmbedder(base, query_instruction="TOPIC: "), base)}
    cache = {}
    run_arms(CORPUS, arms, read_graded_cases(*labels), k_values=(2,), doc_vectors=cache)
    n_segments = sum(len(v) for v in CORPUS.values())
    assert [len(c) for c in calls].count(n_segments) == 1  # corpus embedded once
    [key] = cache
    assert key.startswith(base.model_id + "/")


def test_doc_vector_cache_is_keyed_by_corpus_too():
    cache = {}
    bench = GradedBenchmark({"q": {"x": 1.0}}, {"q": "graph"})
    emb = _bag_embedder()
    first = run_arms({"x": ["graph"], "y": ["audio"]}, {"A": emb}, bench,
                     k_values=(1,), doc_vectors=cache)
    second = run_arms({"x": ["audio"], "y": ["graph"], "z": ["topic"]}, {"A": emb}, bench,
                      k_values=(1,), doc_vectors=cache)
    assert first.rows[0]["recall@1"] == 1.0 and second.rows[0]["recall@1"] == 0.0
    assert len(cache) == 2


def test_retriever_rejects_misaligned_vectors():
    with pytest.raises(ValueError, match="one vector row per segment"):
        MaxPoolRetriever(["a", "b"], np.ones((3, 2)), _bag_embedder())


def test_arm_accepts_an_embedder_spec():
    bench = GradedBenchmark({"q": {"nx": 1.0}}, {"q": "graph"})
    report = run_arms(CORPUS, {"A": "hashing"}, bench, k_values=(1,))
    assert report.rows[0]["embedder"] == "A"


def test_instruction_experiment_end_to_end(labels):
    bench = read_graded_cases(*labels)
    arms = instruction_arms(_bag_embedder(), _bag_embedder(model_id="tuned@4"),
                            instruction="TOPIC: ")
    assert list(arms) == ["A", "B", "C'", "C"]
    report = run_arms(CORPUS, arms, bench, k_values=(1, 2))
    metrics = {row["embedder"]: row for row in report.rows}
    assert metrics["A"]["fp_rate@1"] == 1.0  # the audio-heavy distractor ranks first
    assert metrics["C"]["fp_rate@1"] == 0.0  # the instruction removes it
    verdict = instruction_verdict(metrics, k=1)
    assert verdict["fp_cut_met"] and verdict["recall_held"] and verdict["instruction_beats_bare"]
    # the toy "plain" model also obeys the prefix, so the control is (rightly) flagged
    assert not verdict["control_held"] and not verdict["success"]


def test_bare_embedder_is_an_arm():
    bench = GradedBenchmark({"q": {"nx": 1.0}}, {"q": "graph"})
    report = run_arms(CORPUS, {"A": _bag_embedder()}, bench, k_values=(1,))
    assert report.rows[0]["recall@1"] == 1.0


def test_verdict_handles_zero_baseline_fp():
    m = {k: {"fp_rate@20": 0.0, "recall@20": 0.5, "ndcg@20": 0.5} for k in "ABC"}
    v = instruction_verdict(m)
    assert v["fp_baseline_zero"] and not v["fp_cut_met"] and not v["success"]
    assert v["control_held"] and "instruction_beats_bare" not in v


def test_verdict_names_what_is_missing():
    full = {"fp_rate@20": 0.2, "recall@20": 0.4, "ndcg@20": 0.7}
    with pytest.raises(ValueError, match="'B'"):
        instruction_verdict({"A": full, "B": {}, "C": full})
    with pytest.raises(ValueError, match="'C'"):
        instruction_verdict({"A": full, "B": full})


def test_verdict_two_points_exactly_is_not_held():
    a = {"fp_rate@20": 0.2, "recall@20": 0.40, "ndcg@20": 0.7}
    b = {**a, "recall@20": 0.42}
    c = {"fp_rate@20": 0.05, "recall@20": 0.5}
    assert not instruction_verdict({"A": a, "B": b, "C": c})["control_held"]
