"""Tests for the ``ef`` segmenter facade (Phase 2).

All tests run offline — no network, no ``imbed`` — exercising the canonical
:class:`~ef.segments.Segment` model and the :class:`~ef.segmenters.Segmenter`
facade with plain strings and bare callables.
"""

import pytest

from ef import (
    APPROX_TOKENIZER,
    BaseSegmenter,
    BatchedSegmenter,
    FunctionSegmenter,
    RecursiveCharacterSegmenter,
    Segment,
    SegmentRecord,
    Segmenter,
    approx_token_count,
    as_segment,
    as_segmenter,
    hierarchical,
    line_segmenter,
    make_segment,
    materialise,
    segment_id,
    segment_record,
    with_overlap,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _assert_offsets_consistent(segments, source):
    """Every segment's text must equal source[start:end] when offsets are set."""
    for seg in segments:
        if "start" in seg and "end" in seg:
            assert source[seg["start"] : seg["end"]] == seg["text"]


# --------------------------------------------------------------------------
# segments.py — the data model
# --------------------------------------------------------------------------


def test_segment_id_is_content_derived_and_stable():
    assert segment_id("hello") == segment_id("hello")
    assert segment_id("hello") != segment_id("world")


def test_segment_id_normalizes_before_hashing():
    # NFC normalization, BOM stripping, CRLF -> LF all collapse to one hash
    assert segment_id("café") == segment_id("café")
    assert segment_id("﻿hi") == segment_id("hi")
    assert segment_id("a\r\nb") == segment_id("a\nb")


def test_segment_id_metadata_participates():
    assert segment_id("x", {"source": "a"}) != segment_id("x", {"source": "b"})
    assert segment_id("x", {"a": 1, "b": 2}) == segment_id("x", {"b": 2, "a": 1})


def test_make_segment_omits_none_fields():
    seg = make_segment("hi", index=0)
    assert set(seg) == {"text", "id", "index"}
    assert seg["id"] == segment_id("hi")


def test_make_segment_explicit_id_wins():
    seg = make_segment("hi", id="custom")
    assert seg["id"] == "custom"


def test_segment_record_derives_id_and_round_trips():
    rec = SegmentRecord("hello", index=3)
    assert rec.id == segment_id("hello")
    seg = make_segment(
        "hello",
        parent_id="doc1",
        start=0,
        end=5,
        index=3,
        tokens=2,
        metadata={"source": "s"},
    )
    assert segment_record(seg).to_segment() == seg


def test_segment_record_is_frozen():
    rec = SegmentRecord("hello")
    with pytest.raises(Exception):
        rec.text = "changed"


def test_as_segment_from_str_mapping_record():
    assert as_segment("plain")["text"] == "plain"
    assert as_segment({"text": "m", "index": 1})["index"] == 1
    assert as_segment(SegmentRecord("r"))["text"] == "r"


def test_as_segment_defaults_are_fallbacks_only():
    # default fills a missing field...
    assert as_segment("t", index=5)["index"] == 5
    # ...but never overrides a field already present
    assert as_segment({"text": "t", "index": 0}, index=5)["index"] == 0


def test_as_segment_rejects_bad_input():
    with pytest.raises(ValueError):
        as_segment({"no_text": 1})
    with pytest.raises(TypeError):
        as_segment(42)


# --------------------------------------------------------------------------
# approx_token_count
# --------------------------------------------------------------------------


def test_approx_token_count():
    assert approx_token_count("") == 0
    assert approx_token_count("a" * 40) == 10
    assert approx_token_count("a") == 1  # never below 1 for non-empty text


# --------------------------------------------------------------------------
# the protocols
# --------------------------------------------------------------------------


def test_protocol_isinstance():
    seg = RecursiveCharacterSegmenter()
    assert isinstance(seg, Segmenter)
    assert isinstance(seg, BatchedSegmenter)  # via BaseSegmenter.batch
    assert isinstance(line_segmenter, Segmenter)


def test_base_segmenter_batch_default():
    seg = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
    streams = list(seg.batch(["one two", "three four five"]))
    assert len(streams) == 2
    assert [s["text"] for s in streams[0]] == ["one two"]


# --------------------------------------------------------------------------
# RecursiveCharacterSegmenter
# --------------------------------------------------------------------------


def test_recursive_basic_split_and_overlap():
    seg = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=1)
    chunks = list(seg("one two three four"))
    assert [c["text"] for c in chunks] == ["one two three", "three four"]
    assert [c["index"] for c in chunks] == [0, 1]


def test_recursive_offsets_round_trip():
    source = (
        "First paragraph has several words here.\n\n"
        "Second paragraph also has a fair number of words to split.\n\n"
        "Third one is shorter."
    )
    seg = RecursiveCharacterSegmenter(chunk_size=8, chunk_overlap=2)
    chunks = list(seg(source))
    assert len(chunks) > 1
    _assert_offsets_consistent(chunks, source)


def test_recursive_records_tokenizer_and_tokens():
    seg = RecursiveCharacterSegmenter(chunk_size=5, chunk_overlap=1)
    chunk = next(iter(seg("alpha beta gamma delta")))
    assert chunk["metadata"]["tokenizer"] == APPROX_TOKENIZER
    assert chunk["tokens"] == approx_token_count(chunk["text"])


def test_recursive_empty_document_yields_nothing():
    assert list(RecursiveCharacterSegmenter()("")) == []


def test_recursive_splits_unbreakable_run_by_character():
    seg = RecursiveCharacterSegmenter(chunk_size=2, chunk_overlap=0)
    source = "x" * 20
    chunks = list(seg(source))
    assert len(chunks) > 1
    _assert_offsets_consistent(chunks, source)
    assert "".join(c["text"] for c in chunks) == source


def test_recursive_mapping_doc_sets_parent_id_and_metadata():
    seg = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
    doc = {"text": "one two three", "id": "doc-1", "source": "file.txt"}
    chunks = list(seg(doc))
    assert all(c["parent_id"] == "doc-1" for c in chunks)
    assert chunks[0]["metadata"]["source"] == "file.txt"


def test_recursive_validation():
    with pytest.raises(ValueError):
        RecursiveCharacterSegmenter(chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveCharacterSegmenter(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        RecursiveCharacterSegmenter(chunk_overlap=-1)


def test_recursive_bad_document_type_raises_on_iteration():
    seg = RecursiveCharacterSegmenter()
    with pytest.raises(TypeError):
        list(seg(123))
    with pytest.raises(ValueError):
        list(seg({"no_text": 1}))


# --------------------------------------------------------------------------
# line_segmenter
# --------------------------------------------------------------------------


def test_line_segmenter_drops_blank_lines_and_keeps_offsets():
    source = "  hello  \n\nworld\n"
    segments = list(line_segmenter(source))
    assert [s["text"] for s in segments] == ["hello", "world"]
    assert [s["index"] for s in segments] == [0, 1]
    _assert_offsets_consistent(segments, source)


def test_line_segmenter_mapping_doc_parent_id():
    segments = list(line_segmenter({"text": "a\nb", "id": "d"}))
    assert all(s["parent_id"] == "d" for s in segments)


# --------------------------------------------------------------------------
# FunctionSegmenter
# --------------------------------------------------------------------------


def test_function_segmenter_string_pieces_get_offsets_and_index():
    fs = FunctionSegmenter(lambda text: text.split(","))
    segments = list(fs("a,bb,ccc"))
    assert [s["text"] for s in segments] == ["a", "bb", "ccc"]
    assert [s["index"] for s in segments] == [0, 1, 2]
    _assert_offsets_consistent(segments, "a,bb,ccc")
    assert segments[1]["metadata"]["tokenizer"] == APPROX_TOKENIZER


def test_function_segmenter_skips_empty_pieces():
    fs = FunctionSegmenter(lambda text: ["a", "", "b"])
    assert [s["text"] for s in fs("a b")] == ["a", "b"]


def test_function_segmenter_accepts_mapping_pieces():
    fs = FunctionSegmenter(lambda text: [{"text": "kept", "tokens": 9}])
    seg = next(iter(fs("ignored")))
    assert seg["text"] == "kept"
    assert seg["tokens"] == 9
    assert seg["index"] == 0


def test_function_segmenter_pass_doc():
    fs = FunctionSegmenter(lambda doc: [doc["text"]], pass_doc=True)
    assert [s["text"] for s in fs({"text": "whole"})] == ["whole"]


def test_function_segmenter_count_tokens_disabled():
    fs = FunctionSegmenter(lambda text: ["a"], count_tokens=None)
    seg = next(iter(fs("a")))
    assert "tokens" not in seg


def test_function_segmenter_rejects_non_callable():
    with pytest.raises(TypeError):
        FunctionSegmenter("not callable")


# --------------------------------------------------------------------------
# composition helpers
# --------------------------------------------------------------------------


def test_with_overlap_extends_segments_forward():
    base = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
    source = "one two three four"
    assert [c["text"] for c in base(source)] == ["one two three", "four"]
    overlapped = list(with_overlap(base, 100)(source))
    assert overlapped[0]["text"] == "one two three four"
    _assert_offsets_consistent(overlapped, source)


def test_with_overlap_re_derives_id_and_recounts_tokens():
    base = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
    source = "one two three four"
    plain = list(base(source))[0]
    grown = list(with_overlap(base, 100)(source))[0]
    assert grown["id"] != plain["id"]
    assert grown["id"] == segment_id(grown["text"], grown.get("metadata"))
    assert grown["tokens"] == approx_token_count(grown["text"])


def test_with_overlap_zero_is_identity():
    base = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
    source = "one two three four"
    assert list(with_overlap(base, 0)(source)) == list(base(source))


def test_with_overlap_passes_through_segments_without_offsets():
    def offsetless(doc, /):
        yield {"text": "no offsets", "id": "x"}

    offsetless._ef_segmenter = True
    out = list(with_overlap(offsetless, 5)("anything"))
    assert out == [{"text": "no offsets", "id": "x"}]


def test_with_overlap_rejects_negative():
    with pytest.raises(ValueError):
        with_overlap(RecursiveCharacterSegmenter(), -1)


def test_hierarchical_links_children_to_parents():
    source = "one two\nthree four"
    words = FunctionSegmenter(lambda text: text.split())
    h = hierarchical([line_segmenter, words])
    segments = list(h(source))
    lines = [s for s in segments if " " in s["text"]]
    word_segs = [s for s in segments if " " not in s["text"]]
    assert [s["text"] for s in lines] == ["one two", "three four"]
    assert [s["text"] for s in word_segs] == ["one", "two", "three", "four"]
    # every word points at one of the two line ids
    line_ids = {s["id"] for s in lines}
    assert all(w["parent_id"] in line_ids for w in word_segs)
    # child offsets were shifted to stay absolute against the source
    _assert_offsets_consistent(word_segs, source)


def test_hierarchical_accepts_strings():
    h = hierarchical(["recursive", "lines"])
    assert isinstance(h, Segmenter)
    assert list(h("a b c"))  # does not raise


def test_hierarchical_rejects_empty():
    with pytest.raises(ValueError):
        hierarchical([])


def test_materialise_segmenter_returns_list():
    eager = materialise(line_segmenter)
    out = eager("a\nb")
    assert isinstance(out, list)
    assert [s["text"] for s in out] == ["a", "b"]


def test_materialise_iterable_returns_list():
    streamed = (s for s in [{"text": "x"}, {"text": "y"}])
    assert materialise(streamed) == [{"text": "x"}, {"text": "y"}]


# --------------------------------------------------------------------------
# as_segmenter — the DI seam
# --------------------------------------------------------------------------


def test_as_segmenter_default_and_aliases():
    assert isinstance(as_segmenter(), RecursiveCharacterSegmenter)
    assert isinstance(as_segmenter("recursive"), RecursiveCharacterSegmenter)
    assert isinstance(as_segmenter("default"), RecursiveCharacterSegmenter)
    assert as_segmenter("lines") is line_segmenter


def test_as_segmenter_forwards_kwargs():
    seg = as_segmenter(chunk_size=256, chunk_overlap=32)
    assert (seg.chunk_size, seg.chunk_overlap) == (256, 32)


def test_as_segmenter_passes_through_segmenters():
    seg = RecursiveCharacterSegmenter()
    assert as_segmenter(seg) is seg
    assert as_segmenter(line_segmenter) is line_segmenter


def test_as_segmenter_wraps_bare_callable():
    seg = as_segmenter(lambda text: text.split("|"))
    assert isinstance(seg, FunctionSegmenter)
    assert [s["text"] for s in seg("a|b|c")] == ["a", "b", "c"]


def test_as_segmenter_rejects_bad_input():
    with pytest.raises(TypeError):
        as_segmenter(42)


# --------------------------------------------------------------------------
# ef.Segment is the canonical model, not the legacy str alias
# --------------------------------------------------------------------------


def test_ef_segment_is_the_typeddict_model():
    # Phase 2: ef.Segment must be the canonical TypedDict, not base.Segment (str)
    assert Segment is not str
    assert "text" in Segment.__annotations__
