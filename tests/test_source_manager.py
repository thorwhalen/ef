"""Tests for the ``ef`` search facade (Phase 5).

All tests run offline — a toy character-count :class:`~ef.embedders.Embedder`
and ``vd``'s in-memory backend, no network and no heavy deps. They exercise
:func:`~ef.source_manager.ingest`, :class:`~ef.source_manager.SearchableCorpus`,
:class:`~ef.source_manager.SourceManager` and :class:`~ef.source_manager.SearchHit`.
"""

import numpy as np
import pytest

from ef import (
    HashingEmbedder,
    SearchableCorpus,
    SearchHit,
    SourceManager,
    as_embedder,
    ingest,
)

#: The 16 characters the toy embedder counts — enough that realistic words
#: produce distinct, non-degenerate vectors.
_CHARS = "abcdefghijklmnop"


def _toy(model_id: str = "charcount@16"):
    """A deterministic, dependency-free embedder: a per-text character-count vector.

    ``embedder([t]) == embedder([t])`` always, and a query embeds to the *same*
    vector as an identical document — so an exact-text query scores ~1.0.
    """
    return as_embedder(
        lambda texts: np.array(
            [[t.count(c) for c in _CHARS] for t in texts], dtype=float
        ),
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# ingest — the one-shot light path
# ---------------------------------------------------------------------------


def test_ingest_returns_searchable_corpus():
    idx = ingest(["alpha", "beta", "gamma"], embedder=_toy())
    assert isinstance(idx, SearchableCorpus)


def test_ingest_search_finds_exact_match():
    idx = ingest(["machine code", "ocean wave", "forest path"], embedder=_toy())
    hits = idx.search("ocean wave")
    assert hits[0].segment["text"] == "ocean wave"
    assert round(hits[0].score, 6) == 1.0


def test_ingest_from_mapping_corpus():
    idx = ingest({"d1": "alpha bean", "d2": "gamma leaf"}, embedder=_toy())
    hits = idx.search("alpha bean")
    assert hits[0].segment["text"] == "alpha bean"


def test_search_respects_limit():
    idx = ingest(["one fish", "two fish", "red fish", "blue fish"], embedder=_toy())
    assert len(idx.search("fish", limit=2)) == 2


def test_search_returns_search_hits():
    idx = ingest(["alpha bean"], embedder=_toy())
    hit = idx.search("alpha bean")[0]
    assert isinstance(hit, SearchHit)
    assert isinstance(hit.score, float)
    assert hit.segment["text"] == "alpha bean"
    assert "id" in hit.segment


def test_retrieve_is_an_alias_of_search():
    idx = ingest(["alpha bean", "gamma leaf"], embedder=_toy())
    assert idx.retrieve("alpha bean")[0].segment["text"] == "alpha bean"


def test_search_by_precomputed_query_vector():
    toy = _toy()
    idx = ingest(["alpha bean", "gamma leaf"], embedder=toy)
    query_vector = toy(["alpha bean"], input_type="query")[0]
    hits = idx.search(query_vector)
    assert hits[0].segment["text"] == "alpha bean"


def test_source_id_is_recorded():
    idx = ingest({"my-doc": "alpha bean"}, embedder=_toy())
    assert idx.search("alpha bean")[0].source_id == "my-doc"


def test_empty_corpus_search_is_empty():
    idx = ingest([], embedder=_toy())
    assert idx.search("anything") == []


# ---------------------------------------------------------------------------
# ingest — the dependency-free default embedder (thorwhalen/ef#12)
# ---------------------------------------------------------------------------


def test_ingest_with_no_embedder_returns_searchable_corpus():
    # the headline one-liner must work on a bare `pip install ef` — no extras
    idx = ingest(["alpha bean", "gamma leaf"])
    assert isinstance(idx, SearchableCorpus)


def test_ingest_with_no_embedder_uses_the_hashing_default():
    idx = ingest(["alpha bean"])
    assert isinstance(idx.embedder, HashingEmbedder)


def test_ingest_with_no_embedder_finds_exact_match():
    idx = ingest(["machine code", "ocean wave", "forest path"])
    hits = idx.search("ocean wave")
    assert hits[0].segment["text"] == "ocean wave"
    assert round(hits[0].score, 5) == 1.0


def test_ingest_with_no_embedder_ranks_by_lexical_overlap():
    idx = ingest(["green ocean wave", "dry desert sand"])
    hits = idx.search("a big ocean wave")
    assert hits[0].segment["text"] == "green ocean wave"


# ---------------------------------------------------------------------------
# Segment fidelity — metadata & promoted fields round-trip through vd
# ---------------------------------------------------------------------------


def test_segment_carries_tokenizer_metadata():
    idx = ingest(["alpha bean soup"], embedder=_toy())
    hit = idx.search("alpha bean soup")[0]
    assert "tokenizer" in hit.segment["metadata"]


def test_reserved_keys_not_leaked_into_segment_metadata():
    idx = ingest({"d": "alpha bean"}, embedder=_toy())
    metadata = idx.search("alpha bean")[0].segment.get("metadata", {})
    for reserved in ("source_id", "source_hash", "config_hash"):
        assert reserved not in metadata


def test_index_field_roundtrips_for_a_multi_segment_doc():
    big = ". ".join(f"sentence number {i}" for i in range(400))
    sm = SourceManager({"big": big}, embedder=_toy())
    report = sm.materialize()
    assert report["segments"] > 1
    hits = sm.search("sentence number 1", limit=50)
    assert any(isinstance(h.segment.get("index"), int) for h in hits)


def test_parent_id_roundtrips_from_a_source_with_an_id():
    idx = ingest({"k": {"text": "alpha bean", "id": "mydoc"}}, embedder=_toy())
    assert idx.search("alpha bean")[0].segment["parent_id"] == "mydoc"


def test_filtered_search_uses_source_metadata():
    corpus = {
        "fruit": {"text": "apple pie", "metadata": {"kind": "fruit"}},
        "tech": {"text": "apple watch", "metadata": {"kind": "tech"}},
    }
    idx = ingest(corpus, embedder=_toy())
    hits = idx.search("apple", filter={"kind": "fruit"})
    assert all(h.segment["metadata"]["kind"] == "fruit" for h in hits)
    assert hits  # the fruit document is found


# ---------------------------------------------------------------------------
# SourceManager — construction & materialize
# ---------------------------------------------------------------------------


def test_default_config_registered_when_embedder_given():
    sm = SourceManager(["alpha", "beta"], embedder=_toy())
    assert "default" in sm.configs


def test_no_default_config_when_embedder_absent():
    sm = SourceManager(["alpha", "beta"])
    assert sm.configs == {}
    with pytest.raises(ValueError):
        sm.search("alpha")


def test_materialize_report_counts():
    sm = SourceManager(["alpha", "beta", "gamma"], embedder=_toy())
    report = sm.materialize()
    assert report["sources"] == 3
    assert report["segments"] == 3
    assert report["configs"] == 1


def test_materialize_is_idempotent():
    sm = SourceManager(["alpha", "beta"], embedder=_toy())
    sm.materialize()
    artifacts_after_first = len(sm.graph)
    sm.materialize()
    assert len(sm.graph) == artifacts_after_first


def test_search_before_materialize_is_empty():
    sm = SourceManager(["alpha bean"], embedder=_toy())
    assert sm.search("alpha bean") == []  # nothing indexed yet


def test_repr_mentions_sources_and_configs():
    sm = SourceManager(["alpha", "beta"], embedder=_toy())
    assert "2 source(s)" in repr(sm)
    assert "1 config(s)" in repr(sm)


# ---------------------------------------------------------------------------
# SourceManager — multiple configs & config branching
# ---------------------------------------------------------------------------


def test_register_config_returns_a_config_id():
    sm = SourceManager(["alpha"])
    cid = sm.register_config("mini", embedder=_toy("mini@16"))
    assert len(cid) == 64
    assert sm.configs["mini"] == cid


def test_multi_config_search_by_name():
    sm = SourceManager(["alpha bean", "gamma leaf"])
    sm.register_config("a", embedder=_toy("a@16"))
    sm.register_config("b", embedder=_toy("b@16"))
    sm.materialize()
    assert sm.search("alpha bean", config="a")[0].segment["text"] == "alpha bean"
    assert sm.search("alpha bean", config="b")[0].segment["text"] == "alpha bean"


def test_config_branching_shares_segment_artifacts():
    # two configs, same (default) segmenter, different embedders: the segment
    # artifacts must be computed once and shared; only the embed cone diverges.
    sm = SourceManager(["alpha", "beta", "gamma"])
    sm.register_config("a", embedder=_toy("a@16"))
    sm.register_config("b", embedder=_toy("b@16"))
    sm.materialize()
    segment_specs = [
        s for s in sm.graph.producers.values() if s.op.startswith("segment:")
    ]
    embed_specs = [s for s in sm.graph.producers.values() if s.op.startswith("embed:")]
    assert len(segment_specs) == 3  # shared across both configs
    assert len(embed_specs) == 6  # 3 per config — the divergent cone


def test_different_segmenters_do_not_share_segment_artifacts():
    sm = SourceManager(["alpha", "beta", "gamma"])
    shared_embedder = _toy("shared@16")
    sm.register_config("recursive", segmenter="recursive", embedder=shared_embedder)
    sm.register_config("lines", segmenter="lines", embedder=shared_embedder)
    sm.materialize()
    segment_specs = [
        s for s in sm.graph.producers.values() if s.op.startswith("segment:")
    ]
    assert len(segment_specs) == 6  # 3 per config — not shared


def test_searchable_returns_a_bound_view():
    sm = SourceManager(["alpha bean"], embedder=_toy())
    sm.materialize()
    view = sm.searchable()
    assert isinstance(view, SearchableCorpus)
    assert view.search("alpha bean")[0].segment["text"] == "alpha bean"


def test_resolve_unknown_config_raises():
    sm = SourceManager(["alpha"], embedder=_toy())
    with pytest.raises(KeyError):
        sm.search("alpha", config="nonexistent")


def test_ambiguous_config_requires_explicit_choice():
    sm = SourceManager(["alpha bean"])
    sm.register_config("a", embedder=_toy("a@16"))
    sm.register_config("b", embedder=_toy("b@16"))
    sm.materialize()
    with pytest.raises(ValueError):
        sm.search("alpha bean")  # no "default" config, several registered


def test_pipeline_exposes_the_spec():
    sm = SourceManager(["alpha"], embedder=_toy())
    pipeline = sm.pipeline()
    assert pipeline.embed.op == "embed:charcount@16"


# ---------------------------------------------------------------------------
# Store injection — explicit collection, artifact cache
# ---------------------------------------------------------------------------


def test_explicit_collection_rejects_a_second_config():
    import vd

    client = vd.connect("memory", embedding_model=lambda text: [0.0])
    collection = client.create_collection("manual")
    sm = SourceManager(["alpha"], embedder=_toy("a@16"), store=collection)
    with pytest.raises(ValueError):
        sm.register_config("second", embedder=_toy("b@16"))


def test_explicit_collection_is_used():
    import vd

    client = vd.connect("memory", embedding_model=lambda text: [0.0])
    collection = client.create_collection("manual")
    sm = SourceManager(["alpha bean"], embedder=_toy(), store=collection)
    sm.materialize()
    assert len(collection) == 1  # the segment was written into the given collection


def test_injected_cache_is_populated():
    cache: dict = {}
    ingest(["alpha", "beta"], embedder=_toy(), cache=cache)
    assert len(cache) > 0  # leaves, segment and embed artifacts all cached here


def test_bad_store_raises_type_error():
    with pytest.raises(TypeError):
        SourceManager(["alpha"], embedder=_toy(), store=12345)


# ---------------------------------------------------------------------------
# SearchableCorpus — standalone
# ---------------------------------------------------------------------------


def test_searchable_corpus_standalone():
    sm = SourceManager(["alpha bean", "gamma leaf"], embedder=_toy())
    sm.materialize()
    view = SearchableCorpus(sm.searchable().collection, _toy())
    assert view.search("alpha bean")[0].segment["text"] == "alpha bean"
    assert "indexed" in repr(view)
