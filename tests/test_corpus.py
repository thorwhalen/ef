"""Tests for the ``ef`` corpus facade (Phase 3).

All tests run offline — no network — exercising :func:`~ef.corpus.content_hash`,
the :func:`~ef.corpus.as_corpus` dependency-injection seam, and the
:class:`~ef.corpus.ChangeDetectingCorpus` wrapper. The filesystem tests use
``dol`` (a core dependency) over a ``tmp_path``.
"""

from collections.abc import MutableMapping

import pytest

from ef import (
    ChangeDetectingCorpus,
    ChangeEvent,
    Corpus,
    CorpusDiff,
    Source,
    as_corpus,
    content_hash,
)

# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------


def test_content_hash_deterministic():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_content_hash_normalizes_before_hashing():
    assert content_hash("a\r\nb") == content_hash("a\nb")  # CRLF vs LF
    assert content_hash("﻿hi") == content_hash("hi")  # BOM stripped
    assert content_hash("café") == content_hash("café")  # NFC vs NFD


def test_content_hash_accepts_bytes():
    assert content_hash(b"abc") == content_hash("abc")  # ascii: str/bytes agree
    assert isinstance(content_hash(b"\x00\x01\x02"), str)


def test_content_hash_mapping_ignores_metadata_by_default():
    a = {"text": "body", "metadata": {"tags": ["x"], "title": "A"}}
    b = {"text": "body", "metadata": {"tags": ["y"], "title": "B"}}
    assert content_hash(a) == content_hash(b)


def test_content_hash_content_keys_opt_in():
    a = {"text": "body", "metadata": {"tags": ["x"]}}
    b = {"text": "body", "metadata": {"tags": ["y"]}}
    assert content_hash(a, content_keys=["tags"]) != content_hash(
        b, content_keys=["tags"]
    )
    # a key present nowhere contributes nothing — same as no content_keys
    assert content_hash(a, content_keys=["absent"]) == content_hash(a)


def test_content_hash_content_keys_are_order_independent():
    s = {"text": "body", "lang": "en", "kind": "note"}
    assert content_hash(s, content_keys=["lang", "kind"]) == content_hash(
        s, content_keys=["kind", "lang"]
    )


def test_content_hash_content_keys_found_at_top_level_or_in_metadata():
    top = {"text": "body", "lang": "en"}
    nested = {"text": "body", "metadata": {"lang": "en"}}
    # both places are consulted; same value -> same hash
    assert content_hash(top, content_keys=["lang"]) == content_hash(
        nested, content_keys=["lang"]
    )


def test_content_hash_mapping_wrapper_is_transparent():
    # a source that gains/loses a {'text': ...} wrapper but keeps its text
    # must not look like a change
    assert content_hash({"text": "body"}) == content_hash("body")


def test_content_hash_structured_doc_without_text():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_content_hash_rejects_bad_type():
    with pytest.raises(TypeError):
        content_hash(42)


# ---------------------------------------------------------------------------
# as_corpus — the dependency-injection seam
# ---------------------------------------------------------------------------


def test_as_corpus_none_is_a_fresh_empty_dict():
    c = as_corpus()
    assert c == {} and isinstance(c, dict)
    assert as_corpus() is not c  # a fresh one each call


def test_as_corpus_mapping_passes_straight_through():
    d = {"a": "x"}
    assert as_corpus(d) is d


def test_as_corpus_iterable_is_content_addressed():
    c = as_corpus(["alpha", "beta", "alpha"])
    assert len(c) == 2  # the duplicate collapses
    assert set(c.values()) == {"alpha", "beta"}
    assert c[content_hash("alpha")] == "alpha"


def test_as_corpus_non_directory_string_raises():
    with pytest.raises(TypeError):
        as_corpus("/no/such/directory/ef-test-xyz-123")


def test_as_corpus_bad_type_raises():
    with pytest.raises(TypeError):
        as_corpus(42)


def test_as_corpus_directory(tmp_path):
    pytest.importorskip("dol")
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    c = as_corpus(str(tmp_path))
    assert isinstance(c, MutableMapping)
    assert set(c.values()) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# ChangeEvent / CorpusDiff
# ---------------------------------------------------------------------------


def test_corpus_diff_changed_and_truthiness():
    d = CorpusDiff(added=("b",), modified=("a",))
    assert d.changed == ("b", "a")
    assert bool(d) is True
    assert bool(CorpusDiff(deleted=("a",))) is True
    assert bool(CorpusDiff(unchanged=("a", "b"))) is False
    assert bool(CorpusDiff()) is False


def test_change_event_is_a_frozen_record():
    e = ChangeEvent("id1", "added", None, "h")
    assert (e.source_id, e.kind, e.old_hash, e.new_hash) == ("id1", "added", None, "h")
    with pytest.raises(Exception):  # frozen dataclass
        e.kind = "modified"


# ---------------------------------------------------------------------------
# ChangeDetectingCorpus — through-the-wrapper change detection
# ---------------------------------------------------------------------------


def test_corpus_and_source_are_exported_type_aliases():
    # guards against Corpus/Source being placeholder strings rather than aliases
    assert not isinstance(Corpus, str)
    assert not isinstance(Source, str)


def test_is_a_mutable_mapping():
    corpus = ChangeDetectingCorpus({"a": "x"})
    assert isinstance(corpus, MutableMapping)
    assert len(corpus) == 1
    assert list(corpus) == ["a"]
    assert "a" in corpus and "z" not in corpus
    assert corpus["a"] == "x"
    assert corpus.get("z") is None
    assert dict(corpus.items()) == {"a": "x"}
    corpus["b"] = "y"
    assert set(corpus.keys()) == {"a", "b"}


def test_writes_pass_through_to_the_inner_corpus():
    backing = {}
    corpus = ChangeDetectingCorpus(backing)
    corpus["a"] = "hello"
    assert backing == {"a": "hello"}
    del corpus["a"]
    assert backing == {}


def test_setitem_and_delitem_fire_events():
    events = []
    corpus = ChangeDetectingCorpus(on_change=events.append)
    corpus["d"] = "hello world"  # added
    corpus["d"] = "hello world"  # identical content — idempotent, no event
    corpus["d"] = "changed"  # modified
    del corpus["d"]  # deleted
    assert [(e.source_id, e.kind) for e in events] == [
        ("d", "added"),
        ("d", "modified"),
        ("d", "deleted"),
    ]


def test_events_carry_old_and_new_hashes():
    events = []
    corpus = ChangeDetectingCorpus(on_change=events.append)
    corpus["d"] = "hello world"
    corpus["d"] = "changed"
    del corpus["d"]
    added, modified, deleted = events
    assert added.old_hash is None
    assert added.new_hash == content_hash("hello world")
    assert modified.old_hash == content_hash("hello world")
    assert modified.new_hash == content_hash("changed")
    assert deleted.old_hash == content_hash("changed")
    assert deleted.new_hash is None


def test_delitem_of_absent_key_raises_keyerror():
    corpus = ChangeDetectingCorpus({})
    with pytest.raises(KeyError):
        del corpus["nope"]


def test_construction_adopts_a_silent_baseline():
    events = []
    corpus = ChangeDetectingCorpus({"a": "x", "b": "y"}, on_change=events.append)
    assert events == []  # pre-existing sources fire nothing
    assert set(corpus.hashes) == {"a", "b"}
    corpus["a"] = "x"  # writing identical content to a known key — no-op
    assert events == []
    corpus["b"] = "Y"  # writing different content — a modification
    assert [e.kind for e in events] == ["modified"]


# ---------------------------------------------------------------------------
# ChangeDetectingCorpus — out-of-band detection (diff / scan)
# ---------------------------------------------------------------------------


def test_diff_is_side_effect_free():
    backing = {"a": "one"}
    events = []
    corpus = ChangeDetectingCorpus(backing, on_change=events.append)
    backing["a"] = "two"  # edited behind the wrapper's back
    backing["b"] = "new"
    d1 = corpus.diff()
    assert d1.modified == ("a",) and d1.added == ("b",)
    assert events == []  # diff fires nothing
    d2 = corpus.diff()  # registry untouched — a second diff sees the same
    assert d2.modified == ("a",) and d2.added == ("b",)


def test_scan_detects_out_of_band_changes_and_is_idempotent():
    backing = {"a": "one", "b": "two"}
    events = []
    corpus = ChangeDetectingCorpus(backing, on_change=events.append)
    backing["a"] = "ONE"  # modified out of band
    backing["c"] = "three"  # added out of band
    del backing["b"]  # deleted out of band
    d = corpus.scan()
    assert d.added == ("c",)
    assert d.modified == ("a",)
    assert d.deleted == ("b",)
    assert sorted((e.source_id, e.kind) for e in events) == [
        ("a", "modified"),
        ("b", "deleted"),
        ("c", "added"),
    ]
    d2 = corpus.scan()  # nothing further changed
    assert not d2


def test_injected_hash_registry_is_trusted_as_a_persisted_baseline():
    backing = {"a": "current"}
    # a registry persisted from a prior run, holding a STALE hash for 'a'
    persisted = {"a": content_hash("previous")}
    corpus = ChangeDetectingCorpus(backing, hashes=persisted)
    # construction did not re-baseline (the registry was non-empty), so scan
    # finds the drift between the persisted hash and the current content
    d = corpus.scan()
    assert d.modified == ("a",)


def test_content_keys_are_forwarded_to_change_detection():
    base = {"a": {"text": "body", "metadata": {"tags": ["x"]}}}
    # without content_keys: a tag-only edit is invisible
    events = []
    plain = ChangeDetectingCorpus(dict(base), on_change=events.append)
    plain["a"] = {"text": "body", "metadata": {"tags": ["y"]}}
    assert events == []
    # with content_keys=['tags']: the same tag edit is a modification
    events2 = []
    keyed = ChangeDetectingCorpus(
        dict(base), on_change=events2.append, content_keys=["tags"]
    )
    keyed["a"] = {"text": "body", "metadata": {"tags": ["y"]}}
    assert [e.kind for e in events2] == ["modified"]


# ---------------------------------------------------------------------------
# ChangeDetectingCorpus — the mtime/etag prefilter
# ---------------------------------------------------------------------------


def test_prefilter_skips_rehash_when_fingerprint_is_unchanged():
    backing = {"a": "original"}
    fingerprints = {"a": 1}
    events = []
    corpus = ChangeDetectingCorpus(
        backing,
        on_change=events.append,
        fingerprint=lambda sid: fingerprints.get(sid),
    )
    # mutate content but leave the cheap fingerprint untouched
    backing["a"] = "CHANGED"
    d = corpus.scan()
    assert d.modified == ()  # prefilter said "unchanged" — content not re-hashed
    assert "a" in d.unchanged
    assert events == []
    # advance the fingerprint — now scan re-hashes and the change surfaces
    fingerprints["a"] = 2
    d = corpus.scan()
    assert d.modified == ("a",)
    assert [e.kind for e in events] == ["modified"]


def test_prefilter_change_alone_never_declares_a_change():
    backing = {"a": "stable"}
    fingerprints = {"a": 1}
    events = []
    corpus = ChangeDetectingCorpus(
        backing,
        on_change=events.append,
        fingerprint=lambda sid: fingerprints.get(sid),
    )
    fingerprints["a"] = 999  # fingerprint moved, content did NOT
    d = corpus.scan()
    assert not d  # the content hash is the source of truth — no false positive
    assert events == []


# ---------------------------------------------------------------------------
# ChangeDetectingCorpus — over a real filesystem corpus
# ---------------------------------------------------------------------------


def test_change_detection_over_a_filesystem_corpus(tmp_path):
    pytest.importorskip("dol")
    (tmp_path / "a.txt").write_text("alpha")
    events = []
    corpus = ChangeDetectingCorpus(str(tmp_path), on_change=events.append)
    assert events == []  # baseline adopted silently
    # edits made directly on disk — entirely out of band
    (tmp_path / "a.txt").write_text("ALPHA")
    (tmp_path / "b.txt").write_text("beta")
    d = corpus.scan()
    assert d.modified == ("a.txt",)
    assert d.added == ("b.txt",)
    assert sorted((e.source_id, e.kind) for e in events) == [
        ("a.txt", "modified"),
        ("b.txt", "added"),
    ]
