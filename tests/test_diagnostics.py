"""Tests for ``ef``'s staleness diagnostics (Phase 6).

All offline — the dependency-free hashing embedder and ``vd``'s in-memory
backend. They exercise :mod:`ef.diagnostics`: :func:`~ef.diagnostics.indexed_state`,
:func:`~ef.diagnostics.corpus_state`, :func:`~ef.diagnostics.diagnose` and the
:class:`~ef.diagnostics.StalenessReport` it returns.
"""

import pytest
from vd import Document

from ef import SourceManager
from ef.diagnostics import (
    StalenessReport,
    corpus_state,
    diagnose,
    indexed_state,
)


# ---------------------------------------------------------------------------
# indexed_state — reading provenance back out of a vd collection
# ---------------------------------------------------------------------------


def _doc(doc_id, source_id, source_hash, config_hash):
    """A vd Document carrying the provenance metadata ef writes."""
    return Document(
        id=doc_id,
        text="text",
        metadata={
            "source_id": source_id,
            "source_hash": source_hash,
            "config_hash": config_hash,
        },
    )


def test_indexed_state_groups_documents_by_source():
    collection = {
        "s1#0": _doc("s1#0", "s1", "h1", "c1"),
        "s1#1": _doc("s1#1", "s1", "h1", "c1"),
        "s2#0": _doc("s2#0", "s2", "h2", "c1"),
    }
    state = indexed_state(collection)
    assert set(state) == {"s1", "s2"}
    assert set(state["s1"].doc_ids) == {"s1#0", "s1#1"}
    assert state["s1"].source_hashes == frozenset({"h1"})
    assert state["s1"].config_hashes == frozenset({"c1"})


def test_indexed_state_skips_documents_without_a_source_id():
    collection = {"x": Document(id="x", text="t", metadata={})}
    assert indexed_state(collection) == {}


def test_indexed_state_collects_disagreeing_hashes_as_a_set():
    # a half-re-indexed source — its documents disagree on source_hash
    collection = {
        "s#0": _doc("s#0", "s", "old", "c1"),
        "s#1": _doc("s#1", "s", "new", "c1"),
    }
    assert indexed_state(collection)["s"].source_hashes == frozenset({"old", "new"})


def test_indexed_state_of_a_real_collection():
    sm = SourceManager({"a": "alpha text", "b": "beta text"}, embedder="hashing")
    sm.materialize()
    collection = sm.searchable().collection
    state = indexed_state(collection)
    assert set(state) == {"a", "b"}
    assert all(len(entry.source_hashes) == 1 for entry in state.values())


# ---------------------------------------------------------------------------
# corpus_state — the current content hash of every source
# ---------------------------------------------------------------------------


def test_corpus_state_hashes_each_source():
    state = corpus_state({"a": "one", "b": "two"})
    assert set(state) == {"a", "b"}
    assert state["a"] != state["b"]


def test_corpus_state_is_deterministic():
    assert corpus_state({"a": "one"}) == corpus_state({"a": "one"})


# ---------------------------------------------------------------------------
# StalenessReport
# ---------------------------------------------------------------------------


def test_staleness_report_is_falsey_when_clean():
    assert not StalenessReport(config="c", fresh=("a", "b"))


def test_staleness_report_is_truthy_when_drifted():
    assert StalenessReport(config="c", stale=("a",))
    assert StalenessReport(config="c", orphan=("a",))
    assert StalenessReport(config="c", missing=("a",))
    assert StalenessReport(config="c", misconfigured=("a",))


def test_staleness_report_needs_reindexing_unions_stale_and_misconfigured():
    report = StalenessReport(config="c", stale=("b",), misconfigured=("a", "b"))
    assert report.needs_reindexing == ("a", "b")


# ---------------------------------------------------------------------------
# diagnose — the four staleness conditions
# ---------------------------------------------------------------------------


def test_diagnose_a_fresh_index_finds_nothing_wrong():
    sm = SourceManager({"a": "alpha", "b": "beta"}, embedder="hashing")
    sm.materialize()
    report = sm.diagnose()
    assert not report
    assert set(report.fresh) == {"a", "b"}


def test_diagnose_detects_a_missing_source():
    sm = SourceManager({"a": "alpha"}, embedder="hashing")
    sm.materialize()
    sm.corpus["b"] = "beta"  # added, never indexed
    assert sm.diagnose().missing == ("b",)


def test_diagnose_detects_an_orphan_source():
    sm = SourceManager({"a": "alpha", "b": "beta"}, embedder="hashing")
    sm.materialize()
    del sm.corpus["a"]  # the source is gone; its vectors remain
    assert sm.diagnose().orphan == ("a",)


def test_diagnose_detects_a_stale_source():
    sm = SourceManager({"a": "alpha", "b": "beta"}, embedder="hashing")
    sm.materialize()
    sm.corpus["a"] = "alpha rewritten"  # content changed behind the index
    assert sm.diagnose().stale == ("a",)


def test_diagnose_detects_a_misconfigured_source():
    # re-registering "default" with a different segmenter yields a new ConfigId
    # but reuses the "ef:default" collection — its old documents now record a
    # config_hash that no longer matches.
    sm = SourceManager(["alpha beta gamma"], embedder="hashing")
    sm.materialize()
    sm.register_config("default", segmenter="lines", embedder="hashing")
    report = sm.diagnose()
    assert report.misconfigured
    assert report.stale == ()  # the content itself did not change


def test_diagnose_reports_all_four_conditions_at_once():
    sm = SourceManager({"a": "alpha", "b": "beta"}, embedder="hashing")
    sm.materialize()
    sm.corpus["b"] = "beta changed"  # stale
    sm.corpus["c"] = "gamma"  # missing
    del sm.corpus["a"]  # orphan
    report = sm.diagnose()
    assert report.orphan == ("a",)
    assert report.missing == ("c",)
    assert report.stale == ("b",)
    assert bool(report) is True


def test_diagnose_is_read_only():
    sm = SourceManager({"a": "alpha"}, embedder="hashing")
    sm.materialize()
    del sm.corpus["a"]
    before = len(sm.searchable().collection)
    sm.diagnose()
    assert len(sm.searchable().collection) == before  # diagnose changed nothing


def test_diagnose_pure_function_signature():
    sm = SourceManager({"a": "alpha"}, embedder="hashing")
    sm.materialize()
    cid = sm.configs["default"]
    report = diagnose(sm.corpus, sm.searchable().collection, cid)
    assert isinstance(report, StalenessReport)
    assert report.config == cid


def test_diagnose_unknown_config_raises():
    sm = SourceManager(["alpha"], embedder="hashing")
    with pytest.raises(KeyError):
        sm.diagnose("nonexistent")
