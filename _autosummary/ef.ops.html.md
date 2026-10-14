# ef.ops

The pipeline ops — the callables an [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) runs.

The [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) resolves a [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec)’s string `op` key to a callable through its
injectable `ops` registry, then calls `op(*input_values, **params)`. This
module supplies the two ops Phase 5’s pipeline needs and the machinery that
turns a *live component* (a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) /
[`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) into a serializable [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec)
*plus* the bound callable to register.

The split is deliberate:

- a [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec) is **data** — an op *key*, a version,
  params — so it can be persisted (the recipe is content-addressed);
- the callable is **not** data — it closes over a live component — so it is
  re-registered into the graph’s `ops` registry each process.

The op key carries the **component identity** ([`segmenter_identity()`](#ef.ops.segmenter_identity),
`embedder.model_id`): `"segment:<identity>"` / `"embed:<model_id>"`.
Because `op` participates in [`artifact_id()`](ef.artifact_graph.html.md#ef.artifact_graph.artifact_id), two
configs with different components get disjoint artifact cones automatically,
and two configs with the *same* component share every upstream artifact — which
is what makes config branching free (design notes §3.1).

```pycon
>>> import numpy as np
>>> from ef import as_embedder, as_segmenter
>>> seg = as_segmenter('recursive', chunk_size=128)
>>> seg_spec, seg_fn = segment_step(seg)
>>> seg_spec.op.startswith('segment:recursive-char')
True
>>> [s['text'] for s in seg_fn('hello world')]
['hello world']
>>> emb = as_embedder(lambda ts: np.ones((len(ts), 4)), model_id='ones@4')
>>> emb_spec, emb_fn = embed_step(emb)
>>> emb_spec.op, emb_spec.params
('embed:ones@4', {'input_type': 'document'})
>>> emb_fn([{'text': 'a'}, {'text': 'b'}]).shape
(2, 4)
```

### Module Attributes

| [`SEGMENT_OP_VERSION`](#ef.ops.SEGMENT_OP_VERSION)   | Version of the [`make_segment_op()`](#ef.ops.make_segment_op) wrapper.   |
|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`EMBED_OP_VERSION`](#ef.ops.EMBED_OP_VERSION)     | Version of the [`make_embed_op()`](#ef.ops.make_embed_op) wrapper.     |

### Functions

| [`segmenter_identity`](#ef.ops.segmenter_identity)(segmenter)   | A stable identity string for a segmenter — the content-addressing key.                                                          |
|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [`embedder_identity`](#ef.ops.embedder_identity)(embedder)     | The content-addressing identity of an embedder — its `model_id`.                                                                |
| [`make_segment_op`](#ef.ops.make_segment_op)(segmenter)      | Build the **segment** op — a leaf source → its list of `Segment`s.                                                              |
| [`make_embed_op`](#ef.ops.make_embed_op)(embedder)         | Build the **embed** op — a list of `Segment`s → an `(n, dim)` array.                                                            |
| [`segment_step`](#ef.ops.segment_step)(segmenter)         | A segmenter → its [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec) and bound op callable. |
| [`embed_step`](#ef.ops.embed_step)(embedder)            | An embedder → its [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec) and bound op callable. |

### ef.ops.EMBED_OP_VERSION *= '1'*

Version of the [`make_embed_op()`](#ef.ops.make_embed_op) wrapper. Bump on any behavior change —
see [`SEGMENT_OP_VERSION`](#ef.ops.SEGMENT_OP_VERSION). (This is the *wrapper* version; the embedding
*model*’s identity already lives in the op key via `embedder.model_id`.)

### ef.ops.SEGMENT_OP_VERSION *= '1'*

Version of the [`make_segment_op()`](#ef.ops.make_segment_op) wrapper. It participates in every
downstream [`artifact_id()`](ef.artifact_graph.html.md#ef.artifact_graph.artifact_id), so \*\*bump it on any
behavior change\*\* to how a segment artifact is produced — the graph trusts
the version and cannot inspect the code (a silent change serves stale cache).

### ef.ops.embed_step(embedder)

An embedder → its [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec) and bound op callable.

The mirror of [`segment_step()`](#ef.ops.segment_step) for the embed layer. The op key is
`"embed:<model_id>"` — the model identity already content-addresses the
vector — and `params` carries the defaults-filled `input_type`.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TransformSpec`](ef.config.html.md#ef.config.TransformSpec), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

```pycon
>>> import numpy as np
>>> from ef import as_embedder
>>> spec, fn = embed_step(as_embedder(lambda ts: np.ones((len(ts), 2)), model_id='m@2'))
>>> spec.op, spec.params
('embed:m@2', {'input_type': 'document'})
```

### ef.ops.embedder_identity(embedder)

The content-addressing identity of an embedder — its `model_id`.

[`Embedder.model_id`](ef.embedders.html.md#ef.embedders.Embedder.model_id) already bakes in
everything that changes the vector (provider, model, dimensionality), so it
*is* the identity — this function just names that fact for symmetry with
[`segmenter_identity()`](#ef.ops.segmenter_identity).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> from ef import as_embedder
>>> import numpy as np
>>> embedder_identity(as_embedder(lambda ts: np.ones((len(ts), 2)), model_id='m@2'))
'm@2'
```

### ef.ops.make_embed_op(embedder)

Build the **embed** op — a list of `Segment`s → an `(n, dim)` array.

The returned callable closes over `embedder` and embeds the segments’
`text` in caller order; row `i` of the result is the vector of segment
`i`. `input_type` is a hashable step param (it changes the vector for
task-aware embedders) — it defaults to `"document"`, the indexing role.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> import numpy as np
>>> from ef import as_embedder
>>> op = make_embed_op(as_embedder(lambda ts: np.ones((len(ts), 3)), model_id='m@3'))
>>> op([{'text': 'a'}, {'text': 'b'}]).shape
(2, 3)
>>> op([]).shape[0]                     # an empty doc embeds to zero rows
0
```

### ef.ops.make_segment_op(segmenter)

Build the **segment** op — a leaf source → its list of `Segment`s.

The returned callable closes over `segmenter`; it is what gets registered
into the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph)’s `ops` registry under
the key [`segment_step()`](#ef.ops.segment_step) computes. It materializes the (streaming)
segmenter output into a `list` — a graph artifact must be a concrete,
re-readable value.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]

```pycon
>>> from ef import as_segmenter
>>> op = make_segment_op(as_segmenter('lines'))
>>> [s['text'] for s in op('one\ntwo')]
['one', 'two']
```

### ef.ops.segment_step(segmenter)

A segmenter → its [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec) and bound op callable.

The [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec) is serializable data (op key + version

+ params); the callable closes over the live `segmenter` and is registered
  into the graph’s `ops` registry under `spec.op`. Pair them: persist the
  spec, re-register the callable each process.

```pycon
>>> from ef import as_segmenter
>>> spec, fn = segment_step(as_segmenter('recursive', chunk_size=128))
>>> spec.op_version, spec.params
('1', {})
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TransformSpec`](ef.config.html.md#ef.config.TransformSpec), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### ef.ops.segmenter_identity(segmenter)

A stable identity string for a segmenter — the content-addressing key.

The segmenter’s counterpart of [`Embedder.model_id`](ef.embedders.html.md#ef.embedders.Embedder.model_id). Two segmenters that segment identically
must return the *same* identity; any difference that changes the segments
(chunk size, overlap, separators, tokenizer) must change it. Resolution
order:

1. an explicit `segmenter_id` attribute, if the segmenter advertises one;
2. a [`RecursiveCharacterSegmenter`](ef.segmenters.html.md#ef.segmenters.RecursiveCharacterSegmenter) — identified by its
   config-bearing attributes (`chunk_size` / `chunk_overlap` /
   `separators` / `tokenizer`);
3. a [`FunctionSegmenter`](ef.segmenters.html.md#ef.segmenters.FunctionSegmenter) — by its wrapped callable;
4. otherwise the segmenter’s own `__qualname__` / `__name__` / type name.

```pycon
>>> from ef import as_segmenter
>>> a = segmenter_identity(as_segmenter('recursive', chunk_size=256))
>>> b = segmenter_identity(as_segmenter('recursive', chunk_size=512))
>>> a == b                              # different chunk_size → different id
False
>>> a == segmenter_identity(as_segmenter('recursive', chunk_size=256))
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
