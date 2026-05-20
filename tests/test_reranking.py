"""Tests for ``ef``'s reranking layer (Phase 7, use case F6).

All offline and deterministic — a toy length-based reranker and (for the
integration cases) a character-count embedder over ``vd``'s in-memory backend.
They exercise :mod:`ef.reranking`: the :class:`~ef.reranking.Reranker` protocol,
the pure :func:`~ef.reranking.rerank` operation and the
:func:`~ef.reranking.with_reranker` decorator.
"""

import numpy as np
import pytest

from ef import as_embedder, evaluate_retrieval, ingest, rerank, with_reranker
from ef.reranking import Reranker

_CHARS = "abcdefghijklmnop"


def _toy(model_id="charcount@16"):
    """A deterministic, dependency-free character-count embedder."""
    return as_embedder(
        lambda texts: np.array(
            [[t.count(c) for c in _CHARS] for t in texts], dtype=float
        ),
        model_id=model_id,
    )


def _by_length(query, segments):
    """A toy reranker — scores each segment by its text length (longest first)."""
    return [len(segment["text"]) for segment in segments]


# ---------------------------------------------------------------------------
# the Reranker protocol
# ---------------------------------------------------------------------------


def test_plain_function_is_a_reranker():
    assert isinstance(_by_length, Reranker)


# ---------------------------------------------------------------------------
# rerank — the pure operation
# ---------------------------------------------------------------------------


def test_rerank_reorders_by_score():
    segments = [
        {"text": "cat", "id": "1"},
        {"text": "dog", "id": "2"},
        {"text": "cathedral", "id": "3"},
    ]
    ranked = rerank("query", segments, _by_length)
    assert [s["text"] for s in ranked] == ["cathedral", "cat", "dog"]


def test_rerank_respects_limit():
    segments = [{"text": "a", "id": "1"}, {"text": "bbb", "id": "2"}]
    ranked = rerank("q", segments, _by_length, limit=1)
    assert len(ranked) == 1
    assert ranked[0]["text"] == "bbb"


def test_rerank_folds_score_into_metadata():
    ranked = rerank("q", [{"text": "abc", "id": "1"}], _by_length)
    assert ranked[0]["metadata"]["rerank_score"] == 3.0


def test_rerank_preserves_existing_metadata():
    segments = [{"text": "abc", "id": "1", "metadata": {"source": "doc-1"}}]
    ranked = rerank("q", segments, _by_length)
    assert ranked[0]["metadata"]["source"] == "doc-1"
    assert ranked[0]["metadata"]["rerank_score"] == 3.0


def test_rerank_does_not_mutate_input():
    segments = [{"text": "abc", "id": "1"}]
    rerank("q", segments, _by_length)
    assert "metadata" not in segments[0]  # the input segment is untouched


def test_rerank_empty_is_empty():
    assert rerank("q", [], _by_length) == []


def test_rerank_ties_keep_incoming_order():
    segments = [{"text": "aa", "id": "1"}, {"text": "bb", "id": "2"}]
    ranked = rerank("q", segments, _by_length)  # equal length → equal score
    assert [s["id"] for s in ranked] == ["1", "2"]


def test_rerank_score_count_mismatch_raises():
    segments = [{"text": "a", "id": "1"}, {"text": "b", "id": "2"}]
    with pytest.raises(ValueError, match="one score per segment"):
        rerank("q", segments, lambda query, segs: [1.0])


# ---------------------------------------------------------------------------
# with_reranker — the decorator
# ---------------------------------------------------------------------------


def test_with_reranker_over_fetches_then_trims():
    pool = [{"text": t, "id": t} for t in ["ab", "abcd", "abc"]]

    def base(query, *, limit=10):
        return pool[:limit]

    retrieve = with_reranker(base, _by_length, fetch_k=10)
    segments = retrieve("a query", limit=2)
    assert [s["text"] for s in segments] == ["abcd", "abc"]


def test_with_reranker_passes_kwargs_through():
    seen = {}

    def base(query, *, limit=10, filter=None):
        seen["limit"] = limit
        seen["filter"] = filter
        return [{"text": "x", "id": "1"}]

    retrieve = with_reranker(base, _by_length, fetch_k=25)
    retrieve("q", limit=3, filter={"tag": "a"})
    assert seen["limit"] == 25  # over-fetched fetch_k, not the caller's limit
    assert seen["filter"] == {"tag": "a"}


def test_with_reranker_over_searchable_corpus():
    index = ingest(["ocean wave", "ocean breeze water", "ocean"], embedder=_toy())
    retrieve = with_reranker(index.retrieve, _by_length, fetch_k=10)
    segments = retrieve("ocean", limit=2)
    assert len(segments) == 2
    # the longest matching segment is reranked to the top
    assert segments[0]["text"] == "ocean breeze water"
    assert all("rerank_score" in s["metadata"] for s in segments)


def test_reranked_retriever_is_evaluable():
    # a with_reranker output is a drop-in retriever for evaluate_retrieval
    corpus = {"d1": "ocean wave water", "d2": "mountain peak rock"}
    queries = {"q1": "ocean wave water"}
    qrels = {"q1": {"d1": 1.0}}
    index = ingest(corpus, embedder=_toy())
    retrieve = with_reranker(index.retrieve, _by_length, fetch_k=10)
    report = evaluate_retrieval(retrieve, qrels, queries, k_values=(2,))
    assert report.n_queries == 1
    assert report.metrics["recall@2"] == 1.0
