"""Tests for ``ef``'s refresh layer (Phase 6).

All offline — the dependency-free hashing embedder (and, for the multi-config
cases, a toy character-count embedder) over ``vd``'s in-memory backend. They
exercise :mod:`ef.refresh` (the pure planner + effects) and the
:class:`~ef.source_manager.SourceManager` methods built on it: ``refresh``,
``rebuild``, ``gc_orphans``, ``lineage``, ``scan`` and ``auto_refresh``.
"""

import numpy as np
import pytest

from ef import ArtifactGraph, SourceManager, as_embedder, producer_spec
from ef.corpus import ChangeDetectingCorpus
from ef.diagnostics import IndexedSource, StalenessReport
from ef.refresh import (
    RefreshPlan,
    RefreshReport,
    delete_source_documents,
    plan_refresh,
    prune_dead_leaves,
    refresh_on_change,
)

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
# plan_refresh — the pure policy
# ---------------------------------------------------------------------------


_REPORT = StalenessReport(
    config="c0",
    orphan=("x",),
    missing=("a",),
    stale=("b",),
    misconfigured=("c",),
    fresh=("d",),
)


def test_plan_refresh_full_materializes_missing_and_reindexes():
    plan = plan_refresh(_REPORT, mode="full")
    assert plan.to_materialize == ("a", "b", "c")
    assert plan.to_delete == ("b", "c", "x")


def test_plan_refresh_incremental_keeps_orphans():
    assert plan_refresh(_REPORT, mode="incremental").to_delete == ("b", "c")


def test_plan_refresh_none_deletes_nothing():
    plan = plan_refresh(_REPORT, mode="none")
    assert plan.to_delete == ()
    assert plan.to_materialize == ("a", "b", "c")  # still indexes new content


def test_plan_refresh_scoped_full_restricts_to_the_scope():
    plan = plan_refresh(_REPORT, mode="scoped_full", scope=["a", "b"])
    assert plan.to_materialize == ("a", "b")
    assert plan.to_delete == ("b",)  # 'x' and 'c' are out of scope


def test_plan_refresh_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        plan_refresh(_REPORT, mode="bogus")


def test_refresh_plan_is_falsey_when_empty():
    assert not RefreshPlan()
    assert RefreshPlan(to_materialize=("a",))


# ---------------------------------------------------------------------------
# RefreshReport
# ---------------------------------------------------------------------------


def test_refresh_report_changed_and_bool():
    report = RefreshReport(config="c", mode="full", added=("a",), deleted=("b",))
    assert report.changed == ("a", "b")
    assert report
    assert not RefreshReport(config="c", mode="full", unchanged=("a",))


# ---------------------------------------------------------------------------
# delete_source_documents / prune_dead_leaves — the effects
# ---------------------------------------------------------------------------


def test_delete_source_documents_removes_a_sources_docs():
    collection = {"s1#0": ..., "s1#1": ..., "s2#0": ...}
    indexed = {
        "s1": IndexedSource("s1", ("s1#0", "s1#1"), frozenset(), frozenset()),
        "s2": IndexedSource("s2", ("s2#0",), frozenset(), frozenset()),
    }
    assert delete_source_documents(collection, indexed, ["s1"]) == 2
    assert sorted(collection) == ["s2#0"]


def test_delete_source_documents_tolerates_missing_ids():
    indexed = {"s": IndexedSource("s", ("gone",), frozenset(), frozenset())}
    assert delete_source_documents({}, indexed, ["s", "unknown"]) == 0


def test_prune_dead_leaves_cascades_dead_leaves_only():
    graph = ArtifactGraph()
    graph.register_op("id", lambda x: x)
    graph.put("live", "L")
    graph.put("dead", "D")
    artifact = graph.add(producer_spec("id", "dead", op_version="1"))
    graph.materialize(artifact)
    assert prune_dead_leaves(graph, ["live"]) == 2  # the dead leaf + its artifact
    assert "dead" not in graph and "live" in graph


# ---------------------------------------------------------------------------
# SourceManager.refresh — explicit refresh
# ---------------------------------------------------------------------------


def test_refresh_a_fresh_index_is_a_noop():
    sm = SourceManager(["alpha text", "beta text"], embedder="hashing")
    sm.materialize()
    assert not sm.refresh()  # nothing drifted


def test_refresh_indexes_an_unmaterialized_config():
    sm = SourceManager(["alpha text", "beta text"], embedder="hashing")
    report = sm.refresh()  # never materialized — refresh is the initial index
    assert len(report.added) == 2
    assert not sm.diagnose()


def test_refresh_handles_add_modify_delete_together():
    sm = SourceManager({"a": "alpha text", "b": "beta text"}, embedder="hashing")
    sm.materialize()
    sm.corpus["b"] = "beta rewritten"
    sm.corpus["c"] = "gamma text"
    del sm.corpus["a"]
    report = sm.refresh()
    assert report.added == ("c",)
    assert report.modified == ("b",)
    assert report.deleted == ("a",)
    assert not sm.diagnose()  # the index is in sync again


def test_refresh_is_idempotent():
    sm = SourceManager({"a": "alpha text"}, embedder="hashing")
    sm.materialize()
    sm.corpus["a"] = "alpha changed"
    assert sm.refresh()  # first refresh does work
    assert not sm.refresh()  # second one finds nothing to do


def test_refresh_updates_search_results():
    sm = SourceManager({"doc": "the quick brown fox"}, embedder="hashing")
    sm.materialize()
    sm.corpus["doc"] = "lazy sleeping dog"
    sm.refresh()
    hits = sm.search("lazy sleeping dog")
    assert hits[0].segment["text"] == "lazy sleeping dog"
    # the old content is gone from the index entirely
    assert all(
        "quick brown fox" not in h.segment["text"]
        for h in sm.search("quick brown fox", limit=10)
    )


def test_refresh_none_mode_deletes_nothing():
    sm = SourceManager({"a": "alpha text", "b": "beta text"}, embedder="hashing")
    sm.materialize()
    del sm.corpus["a"]
    report = sm.refresh(mode="none")
    assert report.documents_removed == 0
    assert sm.diagnose().orphan == ("a",)  # the orphan survives


def test_refresh_incremental_replaces_modified_but_keeps_orphans():
    sm = SourceManager({"a": "alpha text", "b": "beta text"}, embedder="hashing")
    sm.materialize()
    sm.corpus["b"] = "beta rewritten"
    del sm.corpus["a"]
    report = sm.refresh(mode="incremental")
    assert report.modified == ("b",)
    assert report.deleted == ()  # 'a' is not deleted by incremental
    assert sm.diagnose().orphan == ("a",)
    assert sm.diagnose().stale == ()  # but 'b' is fresh


def test_refresh_full_deletes_orphans():
    sm = SourceManager({"a": "alpha text", "b": "beta text"}, embedder="hashing")
    sm.materialize()
    del sm.corpus["a"]
    report = sm.refresh(mode="full")
    assert report.deleted == ("a",)
    assert not sm.diagnose()


def test_refresh_full_rejects_a_sources_subset():
    sm = SourceManager(["alpha text"], embedder="hashing")
    sm.materialize()
    with pytest.raises(ValueError):
        sm.refresh(sources=["anything"], mode="full")


def test_refresh_scoped_full_limits_deletion_to_the_subset():
    sm = SourceManager(
        {"a": "alpha text", "b": "beta text", "c": "gamma text"}, embedder="hashing"
    )
    sm.materialize()
    del sm.corpus["a"]
    del sm.corpus["b"]
    report = sm.refresh(sources=["a"], mode="scoped_full")
    assert report.deleted == ("a",)
    assert sm.diagnose().orphan == ("b",)  # 'b' is out of scope, left alone


def test_refresh_targets_a_single_config():
    sm = SourceManager({"a": "alpha text"})
    sm.register_config("x", embedder=_toy("x@16"))
    sm.register_config("y", embedder=_toy("y@16"))
    sm.materialize()
    sm.corpus["b"] = "beta text"
    sm.refresh(config="x")
    assert not sm.diagnose("x")
    assert sm.diagnose("y").missing == ("b",)  # 'y' was not refreshed


def test_refresh_prunes_dead_graph_leaves():
    sm = SourceManager({"a": "alpha text"}, embedder="hashing")
    sm.materialize()
    sm.corpus["a"] = "completely different content"
    report = sm.refresh()
    assert report.artifacts_removed > 0  # the old leaf's cone was freed


# ---------------------------------------------------------------------------
# SourceManager.rebuild
# ---------------------------------------------------------------------------


def test_rebuild_reindexes_from_scratch():
    sm = SourceManager(["alpha text", "beta text"], embedder="hashing")
    sm.materialize()
    report = sm.rebuild()
    assert len(report.added) == 2
    assert report.documents_removed == 2
    assert not sm.diagnose()


def test_rebuild_fixes_a_misconfigured_index():
    sm = SourceManager(["alpha beta gamma"], embedder="hashing")
    sm.materialize()
    sm.register_config("default", segmenter="lines", embedder="hashing")
    assert sm.diagnose().misconfigured  # the old documents are misconfigured
    sm.rebuild()
    assert not sm.diagnose()


# ---------------------------------------------------------------------------
# SourceManager.gc_orphans / lineage
# ---------------------------------------------------------------------------


def test_gc_orphans_removes_only_orphans():
    sm = SourceManager({"a": "alpha text", "b": "beta text"}, embedder="hashing")
    sm.materialize()
    sm.corpus["b"] = "beta rewritten"
    del sm.corpus["a"]
    removed = sm.gc_orphans()
    assert removed == 1
    assert sm.diagnose().orphan == ()
    assert sm.diagnose().stale == ("b",)  # gc_orphans leaves the stale source


def test_lineage_traces_an_artifact_to_its_source():
    from ef.corpus import content_hash

    sm = SourceManager({"doc": "alpha text"}, embedder="hashing")
    sm.materialize()
    embed_ids = [
        aid for aid, spec in sm.graph.producers.items() if spec.op.startswith("embed:")
    ]
    assert embed_ids
    assert content_hash("alpha text") in sm.lineage(embed_ids[0])


# ---------------------------------------------------------------------------
# auto-refresh
# ---------------------------------------------------------------------------


def test_auto_refresh_wraps_the_corpus_in_a_change_detecting_one():
    sm = SourceManager({"a": "alpha text"}, embedder="hashing", auto_refresh=True)
    assert isinstance(sm.corpus, ChangeDetectingCorpus)


def test_auto_refresh_indexes_a_source_added_through_the_wrapper():
    sm = SourceManager({"a": "alpha text"}, embedder="hashing", auto_refresh=True)
    sm.materialize()
    sm.corpus["b"] = "beta text"
    assert not sm.diagnose()
    assert sm.search("beta text")[0].segment["text"] == "beta text"


def test_auto_refresh_replaces_a_modified_source():
    sm = SourceManager({"a": "alpha text"}, embedder="hashing", auto_refresh=True)
    sm.materialize()
    sm.corpus["a"] = "completely different words"
    assert not sm.diagnose()
    assert sm.search("completely different words")[0].source_id == "a"


def test_auto_refresh_removes_a_deleted_source():
    sm = SourceManager(
        {"a": "alpha text", "b": "beta text"}, embedder="hashing", auto_refresh=True
    )
    sm.materialize()
    del sm.corpus["a"]
    assert not sm.diagnose()
    assert all(h.source_id != "a" for h in sm.search("alpha text", limit=10))


def test_auto_refresh_scan_picks_up_out_of_band_edits():
    backing = {"a": "alpha text"}
    sm = SourceManager(backing, embedder="hashing", auto_refresh=True)
    sm.materialize()
    backing["b"] = "beta text"  # edited the backing store directly
    diff = sm.scan()
    assert diff.added == ("b",)
    assert not sm.diagnose()


def test_scan_without_auto_refresh_raises():
    sm = SourceManager(["alpha text"], embedder="hashing")
    with pytest.raises(TypeError):
        sm.scan()


def test_auto_refresh_chains_an_existing_on_change_callback():
    events = []
    corpus = ChangeDetectingCorpus({"a": "alpha text"}, on_change=events.append)
    sm = SourceManager(corpus, embedder="hashing", auto_refresh=True)
    sm.materialize()
    sm.corpus["b"] = "beta text"
    assert any(e.source_id == "b" for e in events)  # the user's callback still fired
    assert not sm.diagnose()  # and the auto-refresh ran too


def test_refresh_on_change_handler_applies_an_event():
    sm = SourceManager({"a": "alpha text"}, embedder="hashing")
    sm.materialize()
    corpus = ChangeDetectingCorpus(sm.corpus, on_change=refresh_on_change(sm))
    sm.corpus = corpus
    corpus["b"] = "beta text"
    assert not sm.diagnose()
