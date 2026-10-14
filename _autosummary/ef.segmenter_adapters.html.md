# ef.segmenter_adapters

The segmenter dependency-injection seam and the `imbed` registry bridge.

[`as_segmenter()`](#ef.segmenter_adapters.as_segmenter) is the single place `ef` coerces a user-supplied
segmenter argument into a well-behaved [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) — the
exact mirror of [`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder) for the embed layer.

[`imbed_segmenter()`](#ef.segmenter_adapters.imbed_segmenter) builds a segmenter from `imbed`’s segmenter registry
(`imbed.components.segmentation`); `imbed` is imported lazily, so importing
this module costs nothing extra and `ef` works without `imbed` installed.

Example — the DI seam in action (no `imbed` needed):

```pycon
>>> from ef.segmenters import Segmenter, RecursiveCharacterSegmenter
>>> seg = as_segmenter('recursive', chunk_size=256)
>>> isinstance(seg, RecursiveCharacterSegmenter)
True
>>> as_segmenter(seg) is seg          # a ready segmenter passes straight through
True
>>> custom = as_segmenter(lambda text: text.split('|'))
>>> [s['text'] for s in custom('a|b|c')]
['a', 'b', 'c']
```

### Functions

| [`as_segmenter`](#ef.segmenter_adapters.as_segmenter)([x])       | Normalize `x` into a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) — the DI seam.        |
|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| [`imbed_segmenter`](#ef.segmenter_adapters.imbed_segmenter)([name]) | Build a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) from `imbed`'s segmenter registry. |

### ef.segmenter_adapters.as_segmenter(x=None, , \*\*kwargs)

Normalize `x` into a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) — the DI seam.

The single place every `ef` entry point coerces a user-supplied segmenter
argument. Accepts, in order:

1. `None` or `"default"` / `"recursive"` — a
   [`RecursiveCharacterSegmenter`](ef.segmenters.html.md#ef.segmenters.RecursiveCharacterSegmenter) (the default);
2. `"lines"` — [`line_segmenter()`](ef.segmenters.html.md#ef.segmenters.line_segmenter);
3. any other string — looked up in `imbed`’s registry via
   [`imbed_segmenter()`](#ef.segmenter_adapters.imbed_segmenter);
4. a ready-made segmenter (a [`BaseSegmenter`](ef.segmenters.html.md#ef.segmenters.BaseSegmenter) or a
   callable marked by `ef`) — returned unchanged;
5. a bare callable — wrapped in [`FunctionSegmenter`](ef.segmenters.html.md#ef.segmenters.FunctionSegmenter).

Extra `**kwargs` are forwarded to the chosen factory (ignored when `x`
is already a segmenter).

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is none of the above.
* **Return type:**
  [*Segmenter*](ef.segmenters.html.md#ef.segmenters.Segmenter)

```pycon
>>> seg = as_segmenter(chunk_size=256, chunk_overlap=32)
>>> (seg.chunk_size, seg.chunk_overlap)
(256, 32)
```

* **Return type:**
  [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter)

### ef.segmenter_adapters.imbed_segmenter(name='default', , \*\*kwargs)

Build a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) from `imbed`’s segmenter registry.

`imbed`’s registered segmenters are bare `text -> Iterable[str]`
callables; this wraps the named one in a [`FunctionSegmenter`](ef.segmenters.html.md#ef.segmenters.FunctionSegmenter)
so it emits well-formed [`Segment`](ef.segments.html.md#ef.segments.Segment) pieces.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A key in `imbed.components.segmentation.segmenters` (`"default"`
    resolves to `imbed`’s configured default segmenter).
  * **\*\*kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Forwarded to [`FunctionSegmenter`](ef.segmenters.html.md#ef.segmenters.FunctionSegmenter).
* **Raises:**
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – if `imbed` is not installed.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `name` is not a registered segmenter.
* **Return type:**
  [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter)

Requires the `imbed` package (`pip install 'ef[imbed]'`).
