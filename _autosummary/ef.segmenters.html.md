# ef.segmenters

The `Segmenter` facade — `ef`’s segment layer (L2).

A *segmenter* turns a document into a stream of [`Segment`](ef.segments.html.md#ef.segments.Segment)
pieces. Like the embedder facade, its core is a tiny structural contract:

```default
Segmenter = Callable[[str | Mapping], Iterable[Segment]]
```

Everything else — overlap, hierarchy, batching, the recursive splitting
algorithm — is *layered scaffolding* over that one shape.

This module defines:

- [`Segmenter`](#ef.segmenters.Segmenter) — the `@runtime_checkable` protocol (a callable). It is
  a protocol, **not** an ABC: any matching callable *is* a segmenter.
- [`BatchedSegmenter`](#ef.segmenters.BatchedSegmenter) — an optional optimization trait (a `batch`
  method) for segmenters that can process many documents at once.
- [`BaseSegmenter`](#ef.segmenters.BaseSegmenter) — an implementation-convenience base (a default
  [`batch()`](#ef.segmenters.BaseSegmenter.batch) and a `repr`). Subclassing is optional.
- [`RecursiveCharacterSegmenter`](#ef.segmenters.RecursiveCharacterSegmenter) — the **default segmenter**: recursive
  character splitting at ~512 tokens with ~12.5% overlap. The honest default —
  within a point or two of fancier methods on realistic corpora.
- [`line_segmenter()`](#ef.segmenters.line_segmenter) — a builtin that splits a document into its non-empty
  lines (a plain function — proof that a bare callable *is* a segmenter).
- [`FunctionSegmenter`](#ef.segmenters.FunctionSegmenter) — wraps a bare `text -> pieces` callable into a
  full segmenter; the bridge for `imbed`’s registered segmenters.
- Composition helpers — [`with_overlap()`](#ef.segmenters.with_overlap), [`hierarchical()`](#ef.segmenters.hierarchical),
  [`materialise()`](#ef.segmenters.materialise).

Streaming is first-class: a segmenter returns an *iterable*, not a list, so the
heavy case (a huge corpus) never has to fit every segment in memory at once.

The dependency-injection seam [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter) and
the `imbed` registry bridge live in [`ef.segmenter_adapters`](ef.segmenter_adapters.html.md#module-ef.segmenter_adapters).

Example — split a document into ~4-token chunks with one token of overlap:

```pycon
>>> seg = RecursiveCharacterSegmenter(chunk_size=4, chunk_overlap=1)
>>> [c['text'] for c in seg('alpha beta gamma delta epsilon')]
['alpha beta gamma delta', 'delta epsilon']
>>> isinstance(seg, Segmenter)
True
```

### Module Attributes

| [`APPROX_TOKENIZER`](#ef.segmenters.APPROX_TOKENIZER)   | Identifier recorded as the `tokenizer` metadata key when token counts come from [`approx_token_count()`](#ef.segmenters.approx_token_count).   |
|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DEFAULT_SEPARATORS`](#ef.segmenters.DEFAULT_SEPARATORS) | Separators tried, in order, by [`RecursiveCharacterSegmenter`](#ef.segmenters.RecursiveCharacterSegmenter).                                             |

### Functions

| [`line_segmenter`](#ef.segmenters.line_segmenter)(doc, /)                         | Segment a document into its non-empty lines.                                |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`with_overlap`](#ef.segmenters.with_overlap)(segmenter, n_chars, /, \*[, ...]) | Wrap `segmenter` so consecutive segments overlap by `n_chars` characters.   |
| [`hierarchical`](#ef.segmenters.hierarchical)(segmenters, /)                    | Compose segmenters into levels — each segments the previous level's output. |
| [`materialise`](#ef.segmenters.materialise)(x)                                 | Force a streaming segmenter (or segment stream) to a concrete list.         |
| [`approx_token_count`](#ef.segmenters.approx_token_count)(text)                       | Rough token count — the common `~4 characters per token` heuristic.         |

### Classes

| [`Segmenter`](#ef.segmenters.Segmenter)(\*args, \*\*kwargs)                      | Structural type for a segmenter: `str | Mapping -> Iterable[Segment]`.                                                    |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`BatchedSegmenter`](#ef.segmenters.BatchedSegmenter)(\*args, \*\*kwargs)               | Optional trait: a segmenter that can process many documents at once.                                                      |
| [`BaseSegmenter`](#ef.segmenters.BaseSegmenter)()                                    | Implementation base for segmenters (optional — [`Segmenter`](#ef.segmenters.Segmenter) is structural). |
| [`FunctionSegmenter`](#ef.segmenters.FunctionSegmenter)(func, \*[, pass_doc, ...])       | Promote a bare `text -> pieces` callable to a full [`Segmenter`](#ef.segmenters.Segmenter).            |
| [`RecursiveCharacterSegmenter`](#ef.segmenters.RecursiveCharacterSegmenter)(\*[, chunk_size, ...]) | The default segmenter — recursive character splitting.                                                                    |

### ef.segmenters.APPROX_TOKENIZER *= 'char-approx'*

Identifier recorded as the `tokenizer` metadata key when token counts come
from [`approx_token_count()`](#ef.segmenters.approx_token_count). Recording *which* tokenizer counted a
segment is mandatory — a `tokens` number is meaningless without it.

### *class* ef.segmenters.BaseSegmenter

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Implementation base for segmenters (optional — [`Segmenter`](#ef.segmenters.Segmenter) is structural).

A subclass implements `__call__`; in return it gets a default
[`batch()`](#ef.segmenters.BaseSegmenter.batch) (one-document-at-a-time — override it for a genuinely batched
backend) and a readable `repr`. Subclassing is purely a convenience;
satisfying [`Segmenter`](#ef.segmenters.Segmenter) structurally is what matters.

#### batch(docs,)

Default: segment each document independently.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]

### *class* ef.segmenters.BatchedSegmenter(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Optional trait: a segmenter that can process many documents at once.

Plain [`Segmenter`](#ef.segmenters.Segmenter) handles one document per call. A segmenter that can
amortize per-call cost across documents (a shared model load, a vectorized
pass) also offers `batch`. [`BaseSegmenter`](#ef.segmenters.BaseSegmenter) supplies a correct — if
un-optimized — default, so every [`BaseSegmenter`](#ef.segmenters.BaseSegmenter) subclass is already
a [`BatchedSegmenter`](#ef.segmenters.BatchedSegmenter).

#### batch(docs,)

Segment each document in `docs`, yielding one segment-stream each.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]

### ef.segmenters.DEFAULT_SEPARATORS *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]* *= ('\\n\\n', '\\n', '. ', ', ', ' ', '')*

Separators tried, in order, by [`RecursiveCharacterSegmenter`](#ef.segmenters.RecursiveCharacterSegmenter). Earlier
(coarser) boundaries are preferred; the empty string is the last resort,
splitting between characters so an oversized run is never left unsplit.

### *class* ef.segmenters.FunctionSegmenter(func, \*, pass_doc=False, count_tokens=<function approx_token_count>, tokenizer='char-approx')

Bases: [`BaseSegmenter`](#ef.segmenters.BaseSegmenter)

Promote a bare `text -> pieces` callable to a full [`Segmenter`](#ef.segmenters.Segmenter).

The bridge for segmenter *functions* that carry no segment schema — most
importantly `imbed`’s registered segmenters, which map text to an iterable
of plain strings. Each piece the callable yields is normalized:

- a `str` piece becomes a [`Segment`](ef.segments.html.md#ef.segments.Segment); its character
  offsets are recovered by locating it in the source text, and a token
  count plus `tokenizer` metadata are attached;
- a `Mapping` / [`SegmentRecord`](ef.segments.html.md#ef.segments.SegmentRecord) piece is coerced with
  [`as_segment()`](ef.segments.html.md#ef.segments.as_segment), gaining only an `index` / `parent_id`
  > where it lacks them.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The callable to wrap. By default it receives the document’s
    *text*; with `pass_doc=True` it receives the whole document.
  * **pass_doc** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Pass the raw document instead of its text.
  * **count_tokens** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`int`](https://docs.python.org/3/builtins/functions.html#int)]]) – Token counter for `str` pieces (`None` disables the
    `tokens` / `tokenizer` annotation).
  * **tokenizer** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name recorded as the `tokenizer` metadata key.

```pycon
>>> fs = FunctionSegmenter(lambda text: text.split(','))
>>> [s['text'] for s in fs('a,b,c')]
['a', 'b', 'c']
>>> s = list(fs('a,b,c'))[1]
>>> (s['start'], s['end'], s['index'])
(2, 3, 1)
```

### *class* ef.segmenters.RecursiveCharacterSegmenter(\*, chunk_size=512, chunk_overlap=64, separators=('\\\\n\\\\n', '\\\\n', '. ', ', ', ' ', ''), count_tokens=<function approx_token_count>, tokenizer='char-approx')

Bases: [`BaseSegmenter`](#ef.segmenters.BaseSegmenter)

The default segmenter — recursive character splitting.

Splits a document on a descending ladder of separators (paragraph → line →
sentence → clause → word → character), merging the resulting pieces into
chunks of about `chunk_size` tokens with `chunk_overlap` tokens shared
between neighbours. This “good enough” default is, on realistic corpora,
within a point or two of far more elaborate methods — upgrade only if you
can measure the win.

Sizes are measured with `count_tokens`. The default counter is the
dependency-free [`approx_token_count()`](#ef.segmenters.approx_token_count); inject a real tokenizer (and its
`tokenizer` name) when an exact count matters. Every emitted segment
records the tokenizer in its metadata and carries exact `start` / `end`
character offsets into the source.

* **Parameters:**
  * **chunk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target chunk size, in `count_tokens` units.
  * **chunk_overlap** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Tokens of overlap between consecutive chunks (must be
    smaller than `chunk_size`).
  * **separators** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The separator ladder, coarsest first; the last should be
    `""` so an unsplittable run is still broken between characters.
  * **count_tokens** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Maps a string to a token count.
  * **tokenizer** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name recorded as the `tokenizer` metadata key.

```pycon
>>> seg = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=1)
>>> [c['text'] for c in seg('one two three four')]
['one two three', 'three four']
```

### *class* ef.segmenters.Segmenter(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Structural type for a segmenter: `str | Mapping -> Iterable[Segment]`.

A segmenter is *just a callable*. It accepts a document — a bare `str` or
a `Mapping` with a `text` key (and optionally an `id`, which becomes
the emitted segments’ `parent_id`) — and yields [`Segment`](ef.segments.html.md#ef.segments.Segment)
pieces. It returns an **iterable**, not a list: streaming is the default so
the heavy case scales.

Because the protocol’s only member is `__call__`, *every* callable
satisfies `isinstance(x, Segmenter)`. Treat it as documentation of intent,
not a discriminating check — [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter) is
the seam that turns an arbitrary callable into a *well-behaved* segmenter.

### ef.segmenters.approx_token_count(text)

Rough token count — the common `~4 characters per token` heuristic.

A dependency-free default so `ef` can talk in “tokens” without pulling in
`tiktoken` or a model tokenizer. Inject a real counter (and its name) into
a segmenter when an exact count matters.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

```pycon
>>> approx_token_count('a' * 40)
10
>>> approx_token_count('')
0
```

### ef.segmenters.hierarchical(segmenters,)

Compose segmenters into levels — each segments the previous level’s output.

Level 0 applies `segmenters[0]` to the document. Level *k* applies
`segmenters[k]` to every level- *(k-1)* segment’s text, setting each
child’s `parent_id` to that segment’s `id` and shifting its offsets to
stay absolute. The result is a flat stream of every segment of every level
(level 0 first), with the tree expressed purely through `parent_id`
pointers.

Each entry of `segmenters` is passed through
[`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter), so strings (e.g. `"recursive"`)
and bare callables are accepted alongside ready-made segmenters.

* **Return type:**
  [`Segmenter`](#ef.segmenters.Segmenter)

### ef.segmenters.line_segmenter(doc,)

Segment a document into its non-empty lines.

Each line is stripped of surrounding whitespace; blank lines are dropped.
Character offsets point at the *stripped* content within the source.

A plain function — and therefore already a [`Segmenter`](#ef.segmenters.Segmenter). It carries
the marker [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter) uses to recognize a
ready-made segmenter.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]

```pycon
>>> [s['text'] for s in line_segmenter('  hello  \n\nworld\n')]
['hello', 'world']
```

### ef.segmenters.materialise(x)

Force a streaming segmenter (or segment stream) to a concrete list.

Streaming is the default, but the light case often wants everything in
hand. [`materialise()`](#ef.segmenters.materialise) is polymorphic:

- given a **segmenter** (a callable), it returns a new segmenter whose
  output is a `list` rather than a lazy iterator;
- given an **iterable of segments**, it returns a `list` of them.

```pycon
>>> eager = materialise(line_segmenter)
>>> out = eager('a\nb')
>>> isinstance(out, list)
True
>>> [s['text'] for s in out]
['a', 'b']
>>> materialise(s for s in [{'text': 'x'}])
[{'text': 'x'}]
```

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### ef.segmenters.with_overlap(segmenter, n_chars, /, \*, count_tokens=<function approx_token_count>, tokenizer='char-approx')

Wrap `segmenter` so consecutive segments overlap by `n_chars` characters.

Each segment is extended *forward* by up to `n_chars` characters (clamped
to the source length), so segment *K* shares its tail with segment *K+1*’s
head. A segment whose text changes has its `id` re-derived and its
`tokens` recounted; segments lacking `start` / `end` offsets pass
through untouched.

Overlap is expressed in *characters* here (this is a generic post-hoc
helper) — for token-measured overlap use a segmenter’s own `chunk_overlap`.

* **Return type:**
  [`Segmenter`](#ef.segmenters.Segmenter)

```pycon
>>> base = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
>>> [c['text'] for c in base('one two three four')]
['one two three', 'four']
>>> [c['text'] for c in with_overlap(base, 100)('one two three four')]
['one two three four', 'four']
```
