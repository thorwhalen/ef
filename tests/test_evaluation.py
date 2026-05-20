"""Tests for ``ef``'s evaluation layer (Phase 7).

All offline and deterministic — the metric primitives are pure functions, the
``evaluate_retrieval`` integration uses a toy character-count embedder over
``vd``'s in-memory backend, and ``evaluate_rag`` needs no model at all. They
exercise :mod:`ef.evaluation`: the retrieval metrics, :func:`evaluate_retrieval`,
the BEIR I/O helpers, the RAG metrics and :func:`evaluate_rag`.
"""

import numpy as np
import pytest

from ef import (
    RagEvalReport,
    RetrievalEvalReport,
    as_embedder,
    average_precision,
    context_precision,
    context_recall,
    dcg_at_k,
    evaluate_rag,
    evaluate_retrieval,
    exact_match,
    ingest,
    ndcg_at_k,
    precision_at_k,
    read_beir,
    recall_at_k,
    reciprocal_rank,
    token_f1,
    write_beir,
)
from ef.evaluation import as_ragas_dataset

_CHARS = "abcdefghijklmnop"


def _toy(model_id="charcount@16"):
    """A deterministic, dependency-free character-count embedder."""
    return as_embedder(
        lambda texts: np.array(
            [[t.count(c) for c in _CHARS] for t in texts], dtype=float
        ),
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# Retrieval metric primitives
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["d1", "d2"], {"d1": 1.0, "d2": 1.0}, 2) == 1.0


def test_ndcg_no_relevant_doc_is_zero():
    assert ndcg_at_k(["d1"], {}, 10) == 0.0


def test_ndcg_rewards_higher_ranks():
    rel = {"d1": 1.0}
    assert ndcg_at_k(["d1", "x"], rel, 2) > ndcg_at_k(["x", "d1"], rel, 2)


def test_ndcg_uses_graded_relevance():
    # an ideal ranking puts the grade-3 doc first
    ranked = ["d_low", "d_high"]
    rel = {"d_high": 3.0, "d_low": 1.0}
    assert ndcg_at_k(ranked, rel, 2) < 1.0
    assert ndcg_at_k(["d_high", "d_low"], rel, 2) == 1.0


def test_dcg_linear_gain():
    # rank 1: 1/log2(2)=1 ; rank 3: 1/log2(4)=0.5
    assert dcg_at_k(["d1", "x", "d2"], {"d1": 1.0, "d2": 1.0}, 3) == pytest.approx(1.5)


def test_recall_at_k():
    rel = {"d1": 1.0, "d2": 1.0}
    assert recall_at_k(["d1", "x", "d2"], rel, 2) == 0.5
    assert recall_at_k(["d1", "x", "d2"], rel, 3) == 1.0


def test_recall_no_relevant_is_zero():
    assert recall_at_k(["d1"], {"d1": 0.0}, 10) == 0.0


def test_precision_at_k():
    assert precision_at_k(["d1", "x"], {"d1": 1.0}, 2) == 0.5
    assert precision_at_k(["d1"], {"d1": 1.0}, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "d1"], {"d1": 1.0}) == 0.5
    assert reciprocal_rank(["x", "d1"], {"d1": 1.0}, k=1) == 0.0
    assert reciprocal_rank(["x"], {"d1": 1.0}) == 0.0


def test_average_precision():
    assert average_precision(
        ["d1", "x", "d2"], {"d1": 1.0, "d2": 1.0}
    ) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_average_precision_no_relevant_is_zero():
    assert (
        average_precision(
            ["d1"],
            {},
        )
        == 0.0
    )


# ---------------------------------------------------------------------------
# evaluate_retrieval — with a callable retriever
# ---------------------------------------------------------------------------


def _ranking_retriever(rankings):
    """A retriever that returns a fixed ranked list of doc-id strings per query."""

    def retriever(query, *, limit=10):
        return rankings[query][:limit]

    return retriever


def test_evaluate_retrieval_basic():
    qrels = {"q1": {"d1": 1.0, "d2": 1.0}, "q2": {"d3": 1.0}}
    queries = {"q1": "first", "q2": "second"}
    retriever = _ranking_retriever({"first": ["d1", "x", "d2"], "second": ["x", "d3"]})
    report = evaluate_retrieval(retriever, qrels, queries, k_values=(1, 3))
    assert isinstance(report, RetrievalEvalReport)
    assert report.n_queries == 2
    assert report.metrics["recall@3"] == 1.0
    assert report.metrics["ndcg@1"] == 0.5  # 1.0 for q1, 0.0 for q2


def test_evaluate_retrieval_per_query_drilldown():
    qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    queries = {"q1": "a", "q2": "b"}
    retriever = _ranking_retriever({"a": ["d1"], "b": ["x"]})
    report = evaluate_retrieval(retriever, qrels, queries, k_values=(1,))
    assert report.per_query["q1"]["ndcg@1"] == 1.0
    assert report.per_query["q2"]["ndcg@1"] == 0.0


def test_evaluate_retrieval_dedups_to_source_level():
    # the same doc retrieved twice counts once, at its best rank
    qrels = {"q1": {"d1": 1.0}}
    queries = {"q1": "a"}
    retriever = _ranking_retriever({"a": ["d1", "d1", "x"]})
    report = evaluate_retrieval(retriever, qrels, queries, k_values=(1, 2))
    assert report.metrics["precision@2"] == 0.5  # d1 (deduped) + x, one relevant


def test_evaluate_retrieval_skips_unjudgeable_queries():
    # q2 has no positive judgement; q3 is not in `queries` — both are skipped
    qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 0.0}, "q3": {"d3": 1.0}}
    queries = {"q1": "a", "q2": "b"}
    retriever = _ranking_retriever({"a": ["d1"], "b": ["d2"]})
    report = evaluate_retrieval(retriever, qrels, queries, k_values=(1,))
    assert report.n_queries == 1
    assert set(report.per_query) == {"q1"}


def test_evaluate_retrieval_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown retrieval metric"):
        evaluate_retrieval(
            _ranking_retriever({"a": ["d1"]}),
            {"q1": {"d1": 1.0}},
            {"q1": "a"},
            metrics=("ndcg", "bogus"),
        )


def test_evaluate_retrieval_no_scorable_queries_raises():
    with pytest.raises(ValueError, match="No query could be scored"):
        evaluate_retrieval(
            _ranking_retriever({"a": ["d1"]}),
            {"q1": {"d1": 0.0}},  # judged non-relevant only
            {"q1": "a"},
        )


def test_report_primary_helpers():
    report = RetrievalEvalReport(
        metrics={"ndcg@1": 0.4, "ndcg@10": 0.83},
        per_query={},
        n_queries=5,
        k_values=(1, 10),
    )
    assert report.primary == 0.83
    assert report.primary_at(1) == 0.4
    assert report.primary_at(999) is None
    assert "ndcg@10" in str(report)


# ---------------------------------------------------------------------------
# evaluate_retrieval — over a real indexed corpus
# ---------------------------------------------------------------------------


def test_evaluate_retrieval_over_ingested_corpus():
    corpus = {
        "d1": "ocean wave water tide",
        "d2": "mountain peak rock cliff",
        "d3": "desert sand dune heat",
    }
    queries = {"q1": "ocean wave water tide", "q2": "mountain peak rock cliff"}
    qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    index = ingest(corpus, embedder=_toy())
    report = evaluate_retrieval(index, qrels, queries, k_values=(1,))
    # an exact-text query embeds to its document's vector — it ranks #1
    assert report.metrics["ndcg@1"] == 1.0
    assert report.metrics["recall@1"] == 1.0


def test_evaluate_retrieval_rejects_non_retriever():
    with pytest.raises(TypeError, match="retriever"):
        evaluate_retrieval(object(), {"q1": {"d1": 1.0}}, {"q1": "a"})


# ---------------------------------------------------------------------------
# BEIR interchange
# ---------------------------------------------------------------------------


def test_beir_round_trip(tmp_path):
    corpus = {"d1": "alpha document text", "d2": "beta document text"}
    queries = {"q1": "alpha query"}
    qrels = {"q1": {"d1": 1.0, "d2": 0.0}}
    write_beir(str(tmp_path), corpus, queries, qrels)
    read_corpus, read_queries, read_qrels = read_beir(str(tmp_path))
    assert read_corpus == corpus
    assert read_queries == queries
    assert read_qrels == {"q1": {"d1": 1.0, "d2": 0.0}}


def test_read_beir_missing_qrels_raises(tmp_path):
    write_beir(str(tmp_path), {"d1": "x"}, {"q1": "y"}, {"q1": {"d1": 1.0}})
    (tmp_path / "qrels.tsv").unlink()
    with pytest.raises(FileNotFoundError):
        read_beir(str(tmp_path))


def test_read_beir_explicit_qrels_path(tmp_path):
    write_beir(str(tmp_path), {"d1": "x"}, {"q1": "y"}, {"q1": {"d1": 2.0}})
    (tmp_path / "qrels.tsv").rename(tmp_path / "custom.tsv")
    _, _, qrels = read_beir(str(tmp_path), qrels="custom.tsv")
    assert qrels == {"q1": {"d1": 2.0}}


def test_read_beir_joins_title_and_text(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id": "d1", "title": "A Title", "text": "the body"}\n', encoding="utf-8"
    )
    (tmp_path / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "a query"}\n', encoding="utf-8"
    )
    (tmp_path / "qrels.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8"
    )
    corpus, _, _ = read_beir(str(tmp_path))
    assert corpus["d1"] == "A Title\nthe body"


def test_evaluate_retrieval_from_beir_files(tmp_path):
    corpus = {
        "d1": "ocean wave water tide",
        "d2": "mountain peak rock cliff",
    }
    queries = {"q1": "ocean wave water tide"}
    qrels = {"q1": {"d1": 1.0}}
    write_beir(str(tmp_path), corpus, queries, qrels)
    rc, rq, rqrels = read_beir(str(tmp_path))
    report = evaluate_retrieval(ingest(rc, embedder=_toy()), rqrels, rq, k_values=(1,))
    assert report.metrics["ndcg@1"] == 1.0


# ---------------------------------------------------------------------------
# RAG metric primitives
# ---------------------------------------------------------------------------


def test_exact_match_normalizes():
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert exact_match("Paris", "London") == 0.0


def test_token_f1_partial_credit():
    assert token_f1("quick brown fox", "quick brown fox") == 1.0
    assert 0.0 < token_f1("quick brown fox", "quick brown dog") < 1.0
    assert token_f1("cat", "dog") == 0.0


def test_context_recall():
    assert context_recall("paris france", ["Paris is in France."]) == 1.0
    assert context_recall("paris france", ["Berlin is in Germany."]) == 0.0
    assert 0.0 < context_recall("paris france", ["Paris is a city."]) < 1.0


def test_context_precision():
    contexts = ["Paris is in France.", "Totally unrelated content."]
    assert context_precision("paris", contexts) == 0.5
    assert context_precision("paris", []) == 0.0


# ---------------------------------------------------------------------------
# evaluate_rag
# ---------------------------------------------------------------------------


def _rag_samples():
    return [
        {
            "user_input": "capital of France?",
            "response": "Paris",
            "reference": "Paris",
            "retrieved_contexts": ["Paris is the capital of France."],
        },
        {
            "user_input": "2 + 2?",
            "response": "five",
            "reference": "four",
            "retrieved_contexts": [],
        },
    ]


def test_evaluate_rag_basic():
    report = evaluate_rag(_rag_samples())
    assert isinstance(report, RagEvalReport)
    assert report.n_samples == 2
    assert report.metrics["exact_match"] == 0.5
    assert report.metrics["non_empty_rate"] == 0.5


def test_evaluate_rag_skips_samples_without_reference():
    samples = [
        {"response": "Paris", "reference": "Paris", "retrieved_contexts": ["c"]},
        {"response": "Berlin", "retrieved_contexts": ["c"]},  # no reference
    ]
    report = evaluate_rag(samples, metrics=("exact_match", "non_empty_rate"))
    # exact_match applies to one sample, non_empty_rate to both
    assert report.coverage["exact_match"] == 1
    assert report.coverage["non_empty_rate"] == 2
    assert report.metrics["exact_match"] == 1.0


def test_evaluate_rag_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown RAG metric"):
        evaluate_rag(_rag_samples(), metrics=("exact_match", "faithfulness"))


def test_evaluate_rag_empty_samples_raises():
    with pytest.raises(ValueError, match="no samples"):
        evaluate_rag([])


def test_evaluate_rag_report_str():
    report = evaluate_rag(_rag_samples(), metrics=("exact_match",))
    assert "RagEvalReport" in str(report)
    assert "exact_match" in str(report)


# ---------------------------------------------------------------------------
# as_ragas_dataset — the LLM-judged-metrics bridge (optional dependency)
# ---------------------------------------------------------------------------


def test_as_ragas_dataset():
    pytest.importorskip("ragas")
    dataset = as_ragas_dataset(_rag_samples())
    assert len(dataset) == 2
