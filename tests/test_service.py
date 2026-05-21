"""Tests for the ``ef`` HTTP-service bridge (Phase 2a).

All tests run offline — the dependency-free :class:`~ef.embedders.HashingEmbedder`
(the default) and ``vd``'s in-memory backend, no network and no heavy deps. They
exercise :class:`~ef.service.EfService` and the :class:`~ef.service.CorpusInfo`
summary it returns.
"""

import pytest

from ef import CorpusInfo, EfService, SearchHit

#: A small offline corpus reused across tests.
_ANIMALS = [
    "the cat sat on the mat",
    "dogs are loyal companions",
    "felines and canines coexist",
]


# ---------------------------------------------------------------------------
# create_corpus / corpus_info
# ---------------------------------------------------------------------------


def test_create_corpus_returns_corpus_info():
    service = EfService()
    info = service.create_corpus(_ANIMALS, corpus_id="animals")
    # CorpusInfo is a plain dict (a TypedDict) — JSON-friendly.
    assert isinstance(info, dict)
    assert info["corpus_id"] == "animals"
    assert info["n_sources"] == 3
    assert info["n_segments"] == 3
    assert info["embedder"] == "hashing:v1@512"
    assert info["dim"] == 512
    assert len(info["config_id"]) == 64  # the pipeline content hash


def test_corpus_info_matches_create_corpus():
    service = EfService()
    created = service.create_corpus(_ANIMALS, corpus_id="animals")
    assert service.corpus_info("animals") == created


def test_explicit_embedder_string_is_resolved():
    service = EfService()
    info = service.create_corpus(["hello world"], embedder="hashing")
    assert info["embedder"] == "hashing:v1@512"


# ---------------------------------------------------------------------------
# default_embedder — the per-instance create_corpus default
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_embedder(monkeypatch):
    """Spy on the embedder ``create_corpus`` hands to ``SourceManager``.

    Substitutes ``SourceManager`` with a wrapper that records the resolved
    ``embedder`` argument, then builds a real, offline hashing-backed manager —
    so these tests prove the embedder *wiring* without touching any network
    embedding backend (and stay within the module's offline-only promise).
    """
    import ef.service as service_module

    seen: dict[str, object] = {}
    real_source_manager = service_module.SourceManager

    def spy(sources, *, segmenter=None, embedder=None):
        seen["embedder"] = embedder
        return real_source_manager(sources, segmenter=segmenter, embedder="hashing")

    monkeypatch.setattr(service_module, "SourceManager", spy)
    return seen


def test_default_embedder_defaults_to_the_module_default(captured_embedder):
    """A bare EfService() resolves create_corpus(embedder=None) to DEFAULT_EMBEDDER."""
    from ef.source_manager import DEFAULT_EMBEDDER

    EfService().create_corpus(["hello world"])
    assert captured_embedder["embedder"] == DEFAULT_EMBEDDER


def test_default_embedder_is_used_when_create_corpus_embedder_is_none(captured_embedder):
    """EfService(default_embedder=...) is what create_corpus(embedder=None) resolves."""
    EfService(default_embedder="openai:text-embedding-3-small").create_corpus(
        ["hello world"]
    )
    assert captured_embedder["embedder"] == "openai:text-embedding-3-small"


def test_explicit_embedder_overrides_the_service_default(captured_embedder):
    """A per-call create_corpus(embedder=...) wins over the service-level default."""
    EfService(default_embedder="openai:text-embedding-3-small").create_corpus(
        ["hello world"], embedder="hashing"
    )
    assert captured_embedder["embedder"] == "hashing"


def test_auto_generated_corpus_id_is_unique():
    service = EfService()
    first = service.create_corpus(["alpha"])["corpus_id"]
    second = service.create_corpus(["beta"])["corpus_id"]
    assert first != second
    assert {first, second} <= set(service)


def test_duplicate_corpus_id_is_rejected():
    service = EfService()
    service.create_corpus(["alpha"], corpus_id="dup")
    with pytest.raises(ValueError, match="already registered"):
        service.create_corpus(["beta"], corpus_id="dup")


# ---------------------------------------------------------------------------
# search / retrieve
# ---------------------------------------------------------------------------


def test_search_returns_ranked_hits():
    service = EfService()
    service.create_corpus(_ANIMALS, corpus_id="animals")
    hits = service.search("animals", "cat", limit=2)
    assert len(hits) == 2
    assert all(isinstance(h, SearchHit) for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)  # ranked, best first


def test_retrieve_returns_plain_segments():
    service = EfService()
    service.create_corpus(_ANIMALS, corpus_id="animals")
    segments = service.retrieve("animals", "cat", limit=2)
    assert len(segments) == 2
    for segment in segments:
        assert isinstance(segment["text"], str)
        assert "score" not in segment  # Segment stays pure of result-only keys
        assert segment["metadata"]["source"]  # provenance preserved


def test_limit_caps_result_count():
    service = EfService()
    service.create_corpus(_ANIMALS, corpus_id="animals")
    assert len(service.search("animals", "dog", limit=1)) == 1


def test_corpora_are_isolated():
    """One corpus's vectors must never surface in another corpus's search."""
    service = EfService()
    service.create_corpus(["apple", "banana"], corpus_id="fruit")
    service.create_corpus(["xylophone", "zephyr"], corpus_id="words")
    fruit_texts = {
        h.segment["text"] for h in service.search("fruit", "apple", limit=10)
    }
    assert fruit_texts <= {"apple", "banana"}


# ---------------------------------------------------------------------------
# explore_corpus
# ---------------------------------------------------------------------------


def test_explore_corpus_returns_structured_result():
    service = EfService()
    service.create_corpus(
        [
            "cats and dogs",
            "machine learning",
            "neural networks",
            "kittens playing",
            "deep learning models",
        ],
        corpus_id="mixed",
    )
    result = service.explore_corpus("mixed", projection_method="pca", n_clusters=2)
    assert set(result) == {"ids", "coords", "labels", "cluster_titles"}
    assert len(result["ids"]) == 5
    assert len(result["coords"]) == 5
    assert len(result["labels"]) == 5
    # the result's ids are the corpus's own segment ids
    indexed_ids = {
        h.segment["id"] for h in service.search("mixed", "learning", limit=99)
    }
    assert set(result["ids"]) == indexed_ids


def test_explore_corpus_dims_3():
    service = EfService()
    service.create_corpus(["a b c", "d e f", "g h i", "j k l"], corpus_id="c")
    result = service.explore_corpus("c", dims=3, projection_method="pca")
    assert all(len(row) == 3 for row in result["coords"])


# ---------------------------------------------------------------------------
# list_corpora / delete_corpus
# ---------------------------------------------------------------------------


def test_list_corpora_lists_every_registered_corpus():
    service = EfService()
    service.create_corpus(["alpha"], corpus_id="a")
    service.create_corpus(["beta"], corpus_id="b")
    listed = {info["corpus_id"] for info in service.list_corpora()}
    assert listed == {"a", "b"}


def test_delete_corpus_removes_the_handle():
    service = EfService()
    service.create_corpus(["alpha"], corpus_id="a")
    service.delete_corpus("a")
    assert "a" not in service
    assert service.list_corpora() == []


# ---------------------------------------------------------------------------
# unknown corpus_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.search("nope", "q"),
        lambda s: s.retrieve("nope", "q"),
        lambda s: s.explore_corpus("nope"),
        lambda s: s.corpus_info("nope"),
        lambda s: s.delete_corpus("nope"),
    ],
)
def test_unknown_corpus_id_raises_key_error(call):
    service = EfService()
    with pytest.raises(KeyError, match="Unknown corpus_id"):
        call(service)


# ---------------------------------------------------------------------------
# dunders
# ---------------------------------------------------------------------------


def test_dunders_track_the_registry():
    service = EfService()
    assert len(service) == 0
    assert "a" not in service
    service.create_corpus(["alpha"], corpus_id="a")
    assert len(service) == 1
    assert "a" in service
    assert "1 corpus" in repr(service)


# ---------------------------------------------------------------------------
# typing
# ---------------------------------------------------------------------------


def test_corpus_info_type_is_exported():
    # CorpusInfo is importable from the package root (qh reads it as the schema).
    assert CorpusInfo.__name__ == "CorpusInfo"
