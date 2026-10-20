# ef.embedders

The `Embedder` facade — `ef`’s embed layer (L3).

An *embedder* is, fundamentally, a batch callable:

```default
Iterable[str] -> ndarray of shape (n, dim)
```

Everything else — batching, async/batch-APIs, normalization, task hints,
caching, retries — is *layered scaffolding* over that one contract.

This module defines the small, structural core of the facade:

- [`Embedder`](#ef.embedders.Embedder) — the `@runtime_checkable` protocol (a callable plus four
  introspectable metadata attributes). It is a protocol, **not** an ABC: any
  object with the right shape *is* an embedder.
- [`BatchHandle`](#ef.embedders.BatchHandle) / [`ready_handle()`](#ef.embedders.ready_handle) — a uniform poll/result shape that
  collapses synchronous embedding and provider Batch APIs into one thing.
- [`BaseEmbedder`](#ef.embedders.BaseEmbedder) — an implementation convenience base (supplies a default
  synchronous [`embed_batch()`](#ef.embedders.BaseEmbedder.embed_batch) and a `repr`). Subclassing it
  is optional; satisfying [`Embedder`](#ef.embedders.Embedder) structurally is what matters.
- [`FunctionEmbedder`](#ef.embedders.FunctionEmbedder) — wraps a bare `texts -> vectors` callable into a
  full [`Embedder`](#ef.embedders.Embedder) (the bridge for e.g. `imbed`’s embedder functions).
- [`HashingEmbedder`](#ef.embedders.HashingEmbedder) — a dependency-free embedder (the feature-hashing
  trick, numpy only); `ef`’s zero-install default when no embedder is given.
- [`cache_key()`](#ef.embedders.cache_key) — a deterministic key pinning everything that changes a
  vector, for [`CachedEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.CachedEmbedder).
- [`embed_length_sorted()`](#ef.embedders.embed_length_sorted) — length-sorted batching with order
  reconstruction; a 5–10× throughput win that `ef` owns so users never have
  to think about padding or sort order.

Composition wrappers live in [`ef.embedder_wrappers`](ef.embedder_wrappers.html.md#module-ef.embedder_wrappers); concrete provider /
local adapters and the [`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder) dependency
-injection seam live in [`ef.embedder_adapters`](ef.embedder_adapters.html.md#module-ef.embedder_adapters).

Example — wrap a plain function and embed two strings:

```pycon
>>> import numpy as np
>>> def toy(texts):  # a bare callable: list[str] -> array(n, 3)
...     return np.array([[len(t), t.count(' '), 1.0] for t in texts])
>>> embedder = FunctionEmbedder(toy, model_id='toy@3')
>>> vectors = embedder(['hello', 'a b c'])
>>> vectors.shape
(2, 3)
>>> embedder.dim  # inferred on first call
3
>>> isinstance(embedder, Embedder)
True
```

### Module Attributes

| [`InputType`](#ef.embedders.InputType)   | Canonical embedding-task vocabulary.                                                    |
|--------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`BatchStatus`](#ef.embedders.BatchStatus) | The lifecycle states of a [`BatchHandle`](#ef.embedders.BatchHandle). |

### Functions

| [`ready_handle`](#ef.embedders.ready_handle)(array)                           | Wrap an already-computed array in a finished [`BatchHandle`](#ef.embedders.BatchHandle).   |
|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [`cache_key`](#ef.embedders.cache_key)(embedder, text[, input_type])       | Deterministic cache key for one `(embedder, text, input_type)` triple.                                       |
| [`embed_length_sorted`](#ef.embedders.embed_length_sorted)(texts, encode, \*[, ...]) | Encode `texts` in length-sorted batches; return vectors in caller order.                                     |

### Classes

| [`BatchHandle`](#ef.embedders.BatchHandle)(poll, result[, cancel])              | A poll/result handle over an in-flight (or already-finished) embed job.                                                 |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [`Embedder`](#ef.embedders.Embedder)(\*args, \*\*kwargs)                     | Structural type for an embedder: a batch callable plus metadata.                                                        |
| [`BaseEmbedder`](#ef.embedders.BaseEmbedder)()                                   | Implementation base for embedders (optional — [`Embedder`](#ef.embedders.Embedder) is structural). |
| [`FunctionEmbedder`](#ef.embedders.FunctionEmbedder)(func, \*[, model_id, dim, ...]) | Promote a bare `texts -> vectors` callable to a full [`Embedder`](#ef.embedders.Embedder).         |
| [`HashingEmbedder`](#ef.embedders.HashingEmbedder)(\*[, dim, ngram_range, ...])     | A dependency-free embedder — the feature-hashing trick, numpy only.                                                     |

### Exceptions

| [`EmbedderError`](#ef.embedders.EmbedderError)   | Raised for embedder-specific failures (shape mismatch, batch job failure).   |
|------------------------------------------------------------------|------------------------------------------------------------------------------|

### *class* ef.embedders.BaseEmbedder

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Implementation base for embedders (optional — [`Embedder`](#ef.embedders.Embedder) is structural).

Subclasses set `model_id` / `dim` / `normalized` /
`honored_input_types` and implement `__call__`; they then satisfy
[`Embedder`](#ef.embedders.Embedder) structurally. In return they get:

- a default *synchronous* [`embed_batch()`](#ef.embedders.BaseEmbedder.embed_batch) (`ready_handle(self(...))`) —
  override it only for a genuinely async provider Batch API;
- a readable `repr`.

#### embed_batch(texts, , input_type=None, \*\*backend)

Default: embed synchronously and return a finished handle.

* **Return type:**
  [`BatchHandle`](#ef.embedders.BatchHandle)

### *class* ef.embedders.BatchHandle(poll, result, cancel=<function BatchHandle.<lambda>>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A poll/result handle over an in-flight (or already-finished) embed job.

Batch-and-poll is just a special case of async. A synchronous backend
returns a *ready* handle (via [`ready_handle()`](#ef.embedders.ready_handle)); a provider Batch API
returns a genuinely pending one. Either way the caller sees one shape.

#### poll

`() -> BatchStatus` — non-blocking status check.

* **Type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'pending'`, `'done'`, `'failed'`]]

#### result

`() -> np.ndarray` — blocks until done, returns `(n, dim)`.

* **Type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], `ndarray`]

#### cancel

`() -> None` — best-effort cancellation.

* **Type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`None`](https://docs.python.org/3/builtins/constants.html#None)]

### ef.embedders.BatchStatus

The lifecycle states of a [`BatchHandle`](#ef.embedders.BatchHandle).

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘pending’, ‘done’, ‘failed’]

### *class* ef.embedders.Embedder(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Structural type for an embedder: a batch callable plus metadata.

An embedder maps `Iterable[str]` to an `ndarray` of shape `(n, dim)`.
The four metadata attributes are the single source of truth \*on the object
itself\* — no global registry, no config singleton.

#### model_id

Identity that bakes in everything affecting the vector —
`"openai:text-embedding-3-large@1024"` (`provider:model@dim`).
Works directly as a cache namespace.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### dim

Output dimensionality. May be `None` until first call for
lazily-probed embedders (e.g. a bare callable of unknown width).

* **Type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### normalized

`True` iff `||v|| == 1` by construction.

* **Type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

#### honored_input_types

The [`InputType`](#ef.embedders.InputType) values this embedder acts on
(empty if it ignores the task hint).

* **Type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'query'`, `'document'`, `'classification'`, `'clustering'`], [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

#### embed_batch(texts, , input_type=None, \*\*backend)

Submit an embed job, returning a [`BatchHandle`](#ef.embedders.BatchHandle).

* **Return type:**
  [`BatchHandle`](#ef.embedders.BatchHandle)

### *exception* ef.embedders.EmbedderError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Raised for embedder-specific failures (shape mismatch, batch job failure).

### *class* ef.embedders.FunctionEmbedder(func, , model_id=None, dim=None, normalized=False, honored_input_types=(), pass_backend=True)

Bases: [`BaseEmbedder`](#ef.embedders.BaseEmbedder)

Promote a bare `texts -> vectors` callable to a full [`Embedder`](#ef.embedders.Embedder).

This is the bridge for embedder *functions* that carry no metadata — most
importantly `imbed`’s registered embedders. The wrapped callable is asked
only for `func(texts)` (it knows nothing of `input_type`); the wrapper
advertises `honored_input_types=()` accordingly.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – A callable mapping a list of strings to an array-like of shape
    `(n, dim)` (a numpy array or a nested sequence — both accepted).
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Vector-identity string. Defaults to `"function:<name>"`.
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Output width. If `None` it is inferred (and frozen) on the first
    call.
  * **normalized** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether `func` already returns unit vectors.
  * **honored_input_types** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'query'`, `'document'`, `'classification'`, `'clustering'`]]) – Task hints `func` honors (usually none).
  * **pass_backend** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `True` (default), forward any `**backend` kwargs to
    `func`; set `False` for callables with a strict `(texts,)`
    signature.

```pycon
>>> import numpy as np
>>> e = FunctionEmbedder(lambda ts: np.ones((len(ts), 4)), model_id='ones@4')
>>> e(['a', 'b', 'c']).shape
(3, 4)
```

### *class* ef.embedders.HashingEmbedder(, dim=512, ngram_range=(1, 2), sublinear_tf=True)

Bases: [`BaseEmbedder`](#ef.embedders.BaseEmbedder)

A dependency-free embedder — the feature-hashing trick, numpy only.

[`HashingEmbedder`](#ef.embedders.HashingEmbedder) is `ef`’s **zero-install default**: the embedder
[`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest) resolves to when none is given, so the
headline `ingest([...]).search(query)` works on a bare `pip install ef`
— no torch, no model download, no network.

It is a *lexical* embedder, not a neural one. Each text is tokenized into
lowercased word n-grams; every token is hashed into one of `dim` buckets
with a sign (the hashing trick — no vocabulary, no fitting, fixed memory);
the signed, sublinear-tf-weighted bucket counts are L2-normalized. The
cosine similarity of two vectors then ranks by shared vocabulary — enough
for demos, doctests and offline tests, and a sane fallback in production.
For genuine semantic quality install a real embedder (`ef[sentence-transformers]`,
`ef[openai]`) and pass it explicitly.

Every parameter that changes a vector is baked into `model_id`, so a
[`HashingEmbedder`](#ef.embedders.HashingEmbedder)’s artifacts stay correctly content-addressed in the
[`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph).

* **Parameters:**
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Output width. Wider → fewer hash collisions, larger vectors.
  * **ngram_range** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Inclusive `(min, max)` word-n-gram sizes. The default
    `(1, 2)` mixes single words with adjacent pairs — a little phrase
    sensitivity at no real cost.
  * **sublinear_tf** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Weight a token by `1 + log(count)` rather than its raw
    `count`, damping a word repeated many times in one text.

```pycon
>>> e = HashingEmbedder()
>>> e.model_id
'hashing:v1@512'
>>> import numpy as np
>>> v = e(['the quick brown fox'])
>>> v.shape
(1, 512)
>>> bool(np.isclose(np.linalg.norm(v[0]), 1.0))  # rows are L2-normalized
True
>>> a, b = e(['hello world']), e(['hello world'])
>>> round(float(a[0] @ b[0]), 5)  # identical text → cosine similarity 1.0
1.0
>>> isinstance(e, Embedder)
True
```

### ef.embedders.InputType

Canonical embedding-task vocabulary. Every provider names the same
distinction differently (`input_type` / `task_type` / `task` /
`prompt_name`); `ef` defines *one* canonical set and each adapter
translates it at its boundary. This is the facade’s single most valuable
normalization. Backends that ignore the hint advertise an empty
`honored_input_types`.

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘query’, ‘document’, ‘classification’, ‘clustering’]

### ef.embedders.cache_key(embedder, text, input_type=None)

Deterministic cache key for one `(embedder, text, input_type)` triple.

The key pins everything that changes the resulting vector. `model_id` is
the embedder-identity SSOT — it already bakes in provider, model and `dim`
— so `dim` is *not* keyed separately (keying it would also break
[`FunctionEmbedder`](#ef.embedders.FunctionEmbedder)’s lazy `dim` inference). `normalized` is keyed
because a [`NormalizingEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.NormalizingEmbedder) shares its
inner’s `model_id` yet yields different vectors.

Returns `"<model_id>/<sha256>"` — a namespaced key safe for any
`MutableMapping` store.

```pycon
>>> class _E:
...     model_id, normalized = 'm@8', False
>>> k1 = cache_key(_E(), 'hello', 'document')
>>> k2 = cache_key(_E(), 'hello', 'query')
>>> k1 == k2  # input_type participates
False
>>> k1.startswith('m@8/')
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ef.embedders.embed_length_sorted(texts, encode, , batch_size=128)

Encode `texts` in length-sorted batches; return vectors in caller order.

Transformer self-attention is O(n²) in sequence length and a batch is padded
to its longest member, so grouping similar-length texts together avoids
wasted compute — a 5–10× speedup on skewed corpora. `ef` owns this so
callers never sort, pad, or un-sort themselves.

* **Parameters:**
  * **texts** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The strings to embed.
  * **encode** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Maps a list of strings to an array-like of shape `(k, dim)`.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per call to `encode`.
* **Return type:**
  `ndarray`
* **Returns:**
  A `float32` array of shape `(len(texts), dim)` in the *original*
  order of `texts`.

```pycon
>>> import numpy as np
>>> def enc(batch):  # toy encoder: vector = [length]
...     return np.array([[len(t)] for t in batch], dtype=np.float32)
>>> out = embed_length_sorted(['aaa', 'a', 'aaaaa', 'aa'], enc, batch_size=2)
>>> out.ravel().tolist()  # order preserved despite internal sorting
[3.0, 1.0, 5.0, 2.0]
```

### ef.embedders.ready_handle(array)

Wrap an already-computed array in a finished [`BatchHandle`](#ef.embedders.BatchHandle).

```pycon
>>> import numpy as np
>>> h = ready_handle(np.zeros((2, 3)))
>>> h.poll()
'done'
>>> h.result().shape
(2, 3)
```

* **Return type:**
  [`BatchHandle`](#ef.embedders.BatchHandle)
