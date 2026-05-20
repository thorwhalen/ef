"""The canonical ``Segment`` data model — ``ef``'s interchange type for L2.

A *segment* is a piece of text carved out of a source document by a
:class:`~ef.segmenters.Segmenter`. This module defines the one shape every
segmenter, adapter and store in ``ef`` agrees on, so segments round-trip
losslessly across the whole pipeline.

Two surfaces over the same data:

- :class:`Segment` — a ``TypedDict``: the lightweight, streaming-friendly
  *interchange* type. A segment is just a ``dict``; the type is passive and
  every helper here is a free function over it.
- :class:`SegmentRecord` — a frozen dataclass: the *convenience* surface for
  code that prefers attribute access and immutability.

Only ``text`` is genuinely required; ``id`` is required by convention and is
**content-derived** by default (:func:`segment_id`) so that re-segmenting the
same document yields the same ids — the basis of idempotent ingestion.

The two field names that are *promoted to top level* (rather than buried in
``metadata``) are ``parent_id`` and ``index`` — hierarchy is expressed by a
flat list with ``parent_id`` pointers, never a nested structure.

Example — build a segment, derive its id, view it as a record:

>>> seg = make_segment('Hello world.', index=0, tokens=3)
>>> seg['text']
'Hello world.'
>>> seg['id'] == segment_id('Hello world.')
True
>>> rec = segment_record(seg)
>>> rec.text, rec.index
('Hello world.', 0)
>>> rec.to_segment() == seg
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypedDict

from ef.hashing import canonical_json, normalize_text, sha256_hex

__all__ = [
    "Segment",
    "SegmentRecord",
    "segment_id",
    "make_segment",
    "as_segment",
    "segment_record",
    "PROMOTED_METADATA_KEYS",
]


#: Metadata keys that, by convention, every ``ef`` adapter must round-trip
#: faithfully. They live inside ``Segment["metadata"]`` (they are *not* promoted
#: to top-level fields) but carry meaning the pipeline relies on. Recording
#: ``tokenizer`` in particular is mandatory — omitting it is the single most
#: common silent bug in chunking pipelines (a segment's ``tokens`` count is
#: meaningless without knowing which tokenizer produced it).
PROMOTED_METADATA_KEYS = (
    "source",
    "source_type",
    "tokenizer",
    "token_count",
    "embedding_model",
    "page",
    "license",
    "ingestion_run_id",
)


# ---------------------------------------------------------------------------
# The interchange type
# ---------------------------------------------------------------------------


class Segment(TypedDict, total=False):
    """A piece of text carved from a source document — the interchange type.

    A ``TypedDict``: a :class:`Segment` *is* a plain ``dict``, which keeps it
    cheap to create and stream. ``total=False`` because most keys are optional;
    by convention ``text`` is always present and ``id`` is always set (derived
    from ``text`` if not supplied).

    Keys:
        text: The segment's text. **Required.**
        id: Stable identifier — content-derived by default (:func:`segment_id`).
        parent_id: Id of the source document or parent segment this came from.
        start: Character offset of the segment's start in the source text.
        end: Character offset of the segment's end (exclusive) in the source.
        index: Ordinal position of the segment in its segmenter's output.
        tokens: Token count of ``text`` (meaningful only with ``metadata`` key
            ``tokenizer`` recording which tokenizer counted it).
        metadata: Free-form mapping; framework-specific keys live here. See
            :data:`PROMOTED_METADATA_KEYS` for the conventional keys.
    """

    text: str
    id: str
    parent_id: str
    start: int
    end: int
    index: int
    tokens: int
    metadata: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Content-derived identity
# ---------------------------------------------------------------------------


def segment_id(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Content-derived id for a segment — ``sha256`` of its normalized content.

    The id is ``sha256(normalize(text) + canonical_json(metadata))`` — the
    normalization and hashing primitives live in :mod:`ef.hashing`, the single
    source of truth ``ef`` content-addresses through. Because the id depends
    only on content, re-segmenting an unchanged document produces identical ids
    — which is what makes ingestion idempotent (a re-run is a no-op, not a
    duplicate). Pass ``metadata`` only when two segments could share identical
    ``text`` yet must stay distinct (e.g. same boilerplate from different
    sources).

    >>> segment_id('hello') == segment_id('hello')
    True
    >>> segment_id('hello') == segment_id('world')
    False
    >>> segment_id('x', {'source': 'a'}) == segment_id('x', {'source': 'b'})
    False
    """
    payload = normalize_text(text) + canonical_json(dict(metadata or {}))
    return sha256_hex(payload)


# ---------------------------------------------------------------------------
# The convenience surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """Frozen-dataclass view over a :class:`Segment` — the convenience surface.

    Use this when attribute access and immutability read better than dict
    access; use the :class:`Segment` ``TypedDict`` directly on the hot path.
    The two convert losslessly via :meth:`to_segment` and :func:`segment_record`.

    ``id`` is derived from ``text`` (and ``metadata``) if left empty.

    >>> rec = SegmentRecord('hello', index=2)
    >>> rec.id == segment_id('hello')
    True
    >>> rec.to_segment()['index']
    2
    """

    text: str
    id: str = ""
    parent_id: str | None = None
    start: int | None = None
    end: int | None = None
    index: int | None = None
    tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", segment_id(self.text, self.metadata))

    def to_segment(self) -> Segment:
        """Return an equivalent :class:`Segment` ``dict`` (dropping empty keys)."""
        seg: Segment = {"text": self.text, "id": self.id}
        if self.parent_id is not None:
            seg["parent_id"] = self.parent_id
        if self.start is not None:
            seg["start"] = self.start
        if self.end is not None:
            seg["end"] = self.end
        if self.index is not None:
            seg["index"] = self.index
        if self.tokens is not None:
            seg["tokens"] = self.tokens
        if self.metadata:
            seg["metadata"] = dict(self.metadata)
        return seg


# ---------------------------------------------------------------------------
# Constructors / coercion
# ---------------------------------------------------------------------------


def make_segment(
    text: str,
    *,
    id: str | None = None,
    parent_id: str | None = None,
    start: int | None = None,
    end: int | None = None,
    index: int | None = None,
    tokens: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Segment:
    """Build a :class:`Segment`, deriving a content-based ``id`` if none given.

    Optional fields are omitted from the result when ``None`` — a segment
    ``dict`` carries only the keys it actually has.

    >>> seg = make_segment('chunk one', index=0, start=0, end=9)
    >>> sorted(seg)
    ['end', 'id', 'index', 'start', 'text']
    >>> seg['id'] == segment_id('chunk one')
    True
    """
    seg: Segment = {"text": text}
    seg["id"] = id or segment_id(text, metadata)
    if parent_id is not None:
        seg["parent_id"] = parent_id
    if start is not None:
        seg["start"] = start
    if end is not None:
        seg["end"] = end
    if index is not None:
        seg["index"] = index
    if tokens is not None:
        seg["tokens"] = tokens
    if metadata:
        seg["metadata"] = dict(metadata)
    return seg


def as_segment(x: Any, **defaults: Any) -> Segment:
    """Coerce ``x`` into a :class:`Segment` — accepts a ``str``, ``Mapping`` or record.

    This is the segment-side normalizer that lets a :class:`Segmenter` accept
    loosely-typed output (a bare string, a partial ``dict``) and still emit a
    well-formed :class:`Segment`. ``**defaults`` supplies *fallback* values:
    a default fills a field only when ``x`` does not already carry it.

    Accepts:

    1. a :class:`SegmentRecord` — converted via :meth:`SegmentRecord.to_segment`;
    2. a ``str`` — becomes the ``text`` of a fresh segment;
    3. a ``Mapping`` with a ``text`` key — its keys are carried through.

    The ``id`` is always present on the result (derived if absent).

    Raises:
        TypeError: if ``x`` is none of the accepted forms.
        ValueError: if ``x`` is a ``Mapping`` without a ``text`` key.

    >>> as_segment('plain text', index=3)['index']
    3
    >>> as_segment({'text': 'kept', 'index': 0}, index=9)['index']  # x wins
    0
    """
    if isinstance(x, SegmentRecord):
        seg: Segment = x.to_segment()
    elif isinstance(x, str):
        seg = {"text": x}
    elif isinstance(x, Mapping):
        if "text" not in x:
            raise ValueError("a segment mapping must have a 'text' key")
        seg = {k: v for k, v in x.items() if v is not None}  # type: ignore[assignment]
    else:
        raise TypeError(
            f"Cannot interpret {type(x).__name__} as a Segment. Pass a str, a "
            f"mapping with a 'text' key, or a SegmentRecord."
        )
    for key, value in defaults.items():
        if value is not None and seg.get(key) is None:  # type: ignore[call-overload]
            seg[key] = value  # type: ignore[literal-required]
    if not seg.get("id"):
        seg["id"] = segment_id(seg["text"], seg.get("metadata"))
    return seg


def segment_record(x: Any) -> SegmentRecord:
    """Coerce ``x`` (``str`` / ``Mapping`` / :class:`SegmentRecord`) into a record.

    The dataclass counterpart of :func:`as_segment` — the convenience-surface
    constructor.

    >>> segment_record('hello').text
    'hello'
    >>> segment_record({'text': 'hi', 'tokens': 1}).tokens
    1
    """
    if isinstance(x, SegmentRecord):
        return x
    seg = as_segment(x)
    return SegmentRecord(
        text=seg["text"],
        id=seg.get("id", ""),
        parent_id=seg.get("parent_id"),
        start=seg.get("start"),
        end=seg.get("end"),
        index=seg.get("index"),
        tokens=seg.get("tokens"),
        metadata=seg.get("metadata") or {},
    )
