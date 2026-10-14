# ef.segments

The canonical `Segment` data model — `ef`’s interchange type for L2.

A *segment* is a piece of text carved out of a source document by a
[`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter). This module defines the one shape every
segmenter, adapter and store in `ef` agrees on, so segments round-trip
losslessly across the whole pipeline.

Two surfaces over the same data:

- [`Segment`](#ef.segments.Segment) — a `TypedDict`: the lightweight, streaming-friendly
  *interchange* type. A segment is just a `dict`; the type is passive and
  every helper here is a free function over it.
- [`SegmentRecord`](#ef.segments.SegmentRecord) — a frozen dataclass: the *convenience* surface for
  code that prefers attribute access and immutability.

Only `text` is genuinely required; `id` is required by convention and is
**content-derived** by default ([`segment_id()`](#ef.segments.segment_id)) so that re-segmenting the
same document yields the same ids — the basis of idempotent ingestion.

The two field names that are *promoted to top level* (rather than buried in
`metadata`) are `parent_id` and `index` — hierarchy is expressed by a
flat list with `parent_id` pointers, never a nested structure.

Example — build a segment, derive its id, view it as a record:

```pycon
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
```

### Module Attributes

| [`PROMOTED_METADATA_KEYS`](#ef.segments.PROMOTED_METADATA_KEYS)   | Metadata keys that, by convention, every `ef` adapter must round-trip faithfully.   |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------|

### Functions

| [`segment_id`](#ef.segments.segment_id)(text[, metadata])                 | Content-derived id for a segment — `sha256` of its normalized content.                                             |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`make_segment`](#ef.segments.make_segment)(text, \*[, id, parent_id, ...]) | Build a [`Segment`](#ef.segments.Segment), deriving a content-based `id` if none given.     |
| [`as_segment`](#ef.segments.as_segment)(x, \*\*defaults)                  | Coerce `x` into a [`Segment`](#ef.segments.Segment) — accepts a `str`, `Mapping` or record. |
| [`segment_record`](#ef.segments.segment_record)(x)                            | Coerce `x` (`str` / `Mapping` / [`SegmentRecord`](#ef.segments.SegmentRecord)) into a record.     |

### Classes

| [`Segment`](#ef.segments.Segment)                                          | A piece of text carved from a source document — the interchange type.                                            |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`SegmentRecord`](#ef.segments.SegmentRecord)(text[, id, parent_id, start, ...]) | Frozen-dataclass view over a [`Segment`](#ef.segments.Segment) — the convenience surface. |

### ef.segments.PROMOTED_METADATA_KEYS *= ('source', 'source_type', 'tokenizer', 'token_count', 'embedding_model', 'page', 'license', 'ingestion_run_id')*

Metadata keys that, by convention, every `ef` adapter must round-trip
faithfully. They live inside `Segment["metadata"]` (they are *not* promoted
to top-level fields) but carry meaning the pipeline relies on. Recording
`tokenizer` in particular is mandatory — omitting it is the single most
common silent bug in chunking pipelines (a segment’s `tokens` count is
meaningless without knowing which tokenizer produced it).

### *class* ef.segments.Segment

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

A piece of text carved from a source document — the interchange type.

A `TypedDict`: a [`Segment`](#ef.segments.Segment) *is* a plain `dict`, which keeps it
cheap to create and stream. `total=False` because most keys are optional;
by convention `text` is always present and `id` is always set (derived
from `text` if not supplied).

Keys:
: text: The segment’s text. **Required.**
  id: Stable identifier — content-derived by default ([`segment_id()`](#ef.segments.segment_id)).
  parent_id: Id of the source document or parent segment this came from.
  start: Character offset of the segment’s start in the source text.
  end: Character offset of the segment’s end (exclusive) in the source.
  index: Ordinal position of the segment in its segmenter’s output.
  tokens: Token count of `text` (meaningful only with `metadata` key
  <br/>
  > `tokenizer` recording which tokenizer counted it).
  <br/>
  metadata: Free-form mapping; framework-specific keys live here. See
  : [`PROMOTED_METADATA_KEYS`](#ef.segments.PROMOTED_METADATA_KEYS) for the conventional keys.

### *class* ef.segments.SegmentRecord(text, id='', parent_id=None, start=None, end=None, index=None, tokens=None, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Frozen-dataclass view over a [`Segment`](#ef.segments.Segment) — the convenience surface.

Use this when attribute access and immutability read better than dict
access; use the [`Segment`](#ef.segments.Segment) `TypedDict` directly on the hot path.
The two convert losslessly via [`to_segment()`](#ef.segments.SegmentRecord.to_segment) and [`segment_record()`](#ef.segments.segment_record).

`id` is derived from `text` (and `metadata`) if left empty.

```pycon
>>> rec = SegmentRecord('hello', index=2)
>>> rec.id == segment_id('hello')
True
>>> rec.to_segment()['index']
2
```

#### to_segment()

Return an equivalent [`Segment`](#ef.segments.Segment) `dict` (dropping empty keys).

* **Return type:**
  [`Segment`](#ef.segments.Segment)

### ef.segments.as_segment(x, \*\*defaults)

Coerce `x` into a [`Segment`](#ef.segments.Segment) — accepts a `str`, `Mapping` or record.

This is the segment-side normalizer that lets a `Segmenter` accept
loosely-typed output (a bare string, a partial `dict`) and still emit a
well-formed [`Segment`](#ef.segments.Segment). `**defaults` supplies *fallback* values:
a default fills a field only when `x` does not already carry it.

Accepts:

1. a [`SegmentRecord`](#ef.segments.SegmentRecord) — converted via [`SegmentRecord.to_segment()`](#ef.segments.SegmentRecord.to_segment);
2. a `str` — becomes the `text` of a fresh segment;
3. a `Mapping` with a `text` key — its keys are carried through.

The `id` is always present on the result (derived if absent).

* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is none of the accepted forms.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `x` is a `Mapping` without a `text` key.
* **Return type:**
  [*Segment*](#ef.segments.Segment)

```pycon
>>> as_segment('plain text', index=3)['index']
3
>>> as_segment({'text': 'kept', 'index': 0}, index=9)['index']  # x wins
0
```

* **Return type:**
  [`Segment`](#ef.segments.Segment)

### ef.segments.make_segment(text, , id=None, parent_id=None, start=None, end=None, index=None, tokens=None, metadata=None)

Build a [`Segment`](#ef.segments.Segment), deriving a content-based `id` if none given.

Optional fields are omitted from the result when `None` — a segment
`dict` carries only the keys it actually has.

* **Return type:**
  [`Segment`](#ef.segments.Segment)

```pycon
>>> seg = make_segment('chunk one', index=0, start=0, end=9)
>>> sorted(seg)
['end', 'id', 'index', 'start', 'text']
>>> seg['id'] == segment_id('chunk one')
True
```

### ef.segments.segment_id(text, metadata=None)

Content-derived id for a segment — `sha256` of its normalized content.

The id is `sha256(normalize(text) + canonical_json(metadata))` — the
normalization and hashing primitives live in [`ef.hashing`](ef.hashing.html.md#module-ef.hashing), the single
source of truth `ef` content-addresses through. Because the id depends
only on content, re-segmenting an unchanged document produces identical ids
— which is what makes ingestion idempotent (a re-run is a no-op, not a
duplicate). Pass `metadata` only when two segments could share identical
`text` yet must stay distinct (e.g. same boilerplate from different
sources).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> segment_id('hello') == segment_id('hello')
True
>>> segment_id('hello') == segment_id('world')
False
>>> segment_id('x', {'source': 'a'}) == segment_id('x', {'source': 'b'})
False
```

### ef.segments.segment_record(x)

Coerce `x` (`str` / `Mapping` / [`SegmentRecord`](#ef.segments.SegmentRecord)) into a record.

The dataclass counterpart of [`as_segment()`](#ef.segments.as_segment) — the convenience-surface
constructor.

* **Return type:**
  [`SegmentRecord`](#ef.segments.SegmentRecord)

```pycon
>>> segment_record('hello').text
'hello'
>>> segment_record({'text': 'hi', 'tokens': 1}).tokens
1
```
