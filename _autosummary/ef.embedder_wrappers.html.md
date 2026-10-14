# ef.embedder_wrappers

Composition wrappers for [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Each wrapper *takes an inner embedder and is itself an embedder* — they compose
freely (`CachedEmbedder(RetryingEmbedder(NormalizingEmbedder(inner)), store)`).
This keeps the core protocol tiny: caching, retries, routing and normalization
are opt-in layers, not protocol obligations.

- [`CachedEmbedder`](#ef.embedder_wrappers.CachedEmbedder) — memoize vectors in any `MutableMapping` store.
- [`RetryingEmbedder`](#ef.embedder_wrappers.RetryingEmbedder) — exponential backoff + jitter, retryable-aware.
- [`MultiEmbedder`](#ef.embedder_wrappers.MultiEmbedder) — route texts to different embedders by a predicate.
- [`NormalizingEmbedder`](#ef.embedder_wrappers.NormalizingEmbedder) — L2-normalize an inner embedder’s output.
- [`InstructionEmbedder`](#ef.embedder_wrappers.InstructionEmbedder) — prepend a per-`input_type` instruction
  (task-conditioned, asymmetric query/document embedding).

They delegate their metadata (`model_id` / `dim` / `normalized` /
`honored_input_types`) to the inner embedder, except where the wrapper changes
the vector ([`NormalizingEmbedder`](#ef.embedder_wrappers.NormalizingEmbedder) reports `normalized=True`;
[`InstructionEmbedder`](#ef.embedder_wrappers.InstructionEmbedder) bakes its instructions into `model_id` and adds the
conditioned roles to `honored_input_types`).

The inner argument must already be an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder); use
[`ef.embedder_adapters.as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder) to coerce a string / callable first.

Example — cache an embedder so repeats are free:

```pycon
>>> import numpy as np
>>> from ef.embedders import FunctionEmbedder
>>> calls = []
>>> def toy(texts):
...     calls.extend(texts)
...     return np.ones((len(texts), 3))
>>> cached = CachedEmbedder(FunctionEmbedder(toy, model_id='toy@3'), store={})
>>> _ = cached(['a', 'b'])
>>> _ = cached(['b', 'c'])   # 'b' served from cache
>>> calls                    # 'b' embedded once
['a', 'b', 'c']
```

### Classes

| [`CachedEmbedder`](#ef.embedder_wrappers.CachedEmbedder)(inner, store, \*[, cache_queries])   | Memoize embeddings in a `MutableMapping` store.                                                         |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [`RetryPolicy`](#ef.embedder_wrappers.RetryPolicy)([max_attempts, base_delay, ...])        | Exponential-backoff retry policy for [`RetryingEmbedder`](#ef.embedder_wrappers.RetryingEmbedder). |
| [`RetryingEmbedder`](#ef.embedder_wrappers.RetryingEmbedder)(inner[, policy, sleep])            | Retry an inner embedder's calls on transient failures.                                                  |
| [`MultiEmbedder`](#ef.embedder_wrappers.MultiEmbedder)(routes, predicate, \*[, default])     | Route each text to one of several embedders by a predicate.                                             |
| [`NormalizingEmbedder`](#ef.embedder_wrappers.NormalizingEmbedder)(inner)                          | L2-normalize an inner embedder's output.                                                                |
| [`InstructionEmbedder`](#ef.embedder_wrappers.InstructionEmbedder)(inner, \*[, ...])               | Prepend a per-`input_type` instruction before delegating to `inner`.                                    |

### *class* ef.embedder_wrappers.CachedEmbedder(inner, store, , cache_queries=False)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

Memoize embeddings in a `MutableMapping` store.

On each call, texts already in the store are served from it; only the misses
are forwarded to the inner embedder (in a single batch) and then written
back. The store can be any `dol`-style mapping — an in-RAM `dict`, a
directory of `.npy` files, a key-value DB.

Document embeddings are cached unconditionally; *query* embeddings are
skipped by default (they are rarely repeated) — pass `cache_queries=True`
to cache them too.

* **Parameters:**
  * **inner** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) – The embedder to memoize.
  * **store** ([`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `ndarray`]) – A `MutableMapping[str, ndarray]` keyed by `cache_key()`.
  * **cache_queries** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to also cache `input_type="query"` calls.

#### *property* honored_input_types *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['query', 'document', 'classification', 'clustering'], ...]*

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable’s items.

If the argument is a tuple, the return value is the same object.

#### *property* normalized *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### *class* ef.embedder_wrappers.InstructionEmbedder(inner, , query_instruction=None, document_instruction=None, classification_instruction=None, clustering_instruction=None)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

Prepend a per-`input_type` instruction before delegating to `inner`.

Instruction-tuned embedding models (E5-instruct, BGE, GTE, Qwen3-Embedding…)
are trained to read a task description in front of the text, typically on the
*query* side only, with documents embedded bare. That asymmetry is where the
gain comes from: the query says *what kind of relevance* is wanted (topic vs.
keyword), which a plain model cannot be told. This wrapper adds that to any
embedder: a text embedded with `input_type="query"` becomes
`query_instruction + text`, and so on per role. Roles without an instruction
(and calls with `input_type=None`) pass through unchanged.

**Only trust the prefix on a genuinely instruction-tuned model.** On a plain
model (e.g. `all-MiniLM-L6-v2`) the prefix is expected to be a near no-op or
noise: use that as the control arm of an experiment, not as an improvement.

The instruction strings are used verbatim, so include the model’s expected
separator (e.g. `"Instruct: ...\nQuery: "` for Qwen3/E5-instruct,
`"Represent this sentence for searching relevant passages: "` for BGE).

Metadata:

- `model_id` bakes in a digest of the instructions (they change the vector),
  so [`CachedEmbedder`](#ef.embedder_wrappers.CachedEmbedder) and content-addressed artifacts never mix vectors
  made with different instructions.
- `honored_input_types` is the inner embedder’s plus every instructed role.
- `input_type` is still forwarded to `inner`; a backend that also acts on
  it natively (Cohere, Voyage, Gemini, or a sentence-transformers adapter
  built with `query_prompt=`) would then be conditioned twice, so use this
  wrapper for backends that ignore the hint. For sentence-transformers models
  prefer the adapter’s own `query_prompt=`.

* **Parameters:**
  * **inner** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) – The embedder to condition.
  * **query_instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prefix for `input_type="query"`.
  * **document_instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prefix for `input_type="document"` (usually none).
  * **classification_instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prefix for `input_type="classification"`.
  * **clustering_instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prefix for `input_type="clustering"`.

```pycon
>>> import numpy as np
>>> from ef.embedders import FunctionEmbedder
>>> seen = []
>>> def toy(texts):
...     seen.extend(texts)
...     return np.ones((len(texts), 2))
>>> base = FunctionEmbedder(toy, model_id='toy@2')
>>> emb = InstructionEmbedder(base, query_instruction='Find packages about: ')
>>> _ = emb(['graphs'], input_type='query')
>>> _ = emb(['a DAG library'], input_type='document')
>>> seen
['Find packages about: graphs', 'a DAG library']
>>> emb.honored_input_types
('query',)
>>> emb.model_id.startswith('toy@2+instruct:')
True
```

#### embed_batch(texts, , input_type=None, \*\*backend)

Prefix, then use the inner embedder’s own (possibly async) batch API.

#### *property* honored_input_types *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['query', 'document', 'classification', 'clustering'], ...]*

The inner embedder’s honored roles plus every instructed role.

#### *property* model_id *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The inner `model_id` plus a digest of the instructions.

#### *property* normalized *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### *class* ef.embedder_wrappers.MultiEmbedder(routes, predicate, , default=None)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

Route each text to one of several embedders by a predicate.

Useful for domain-specific models (e.g. a code embedder for code, a prose
embedder for prose). All routes must share the same `dim` — the output is
a single `(n, dim)` array, with each row produced by its text’s route.

* **Parameters:**
  * **routes** ([`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`Hashable`](https://docs.python.org/3/library/typing.html#typing.Hashable), [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)]) – `{route_key: Embedder}`.
  * **predicate** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Hashable`](https://docs.python.org/3/library/typing.html#typing.Hashable)]) – `text -> route_key` — picks a route per text.
  * **default** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Embedder for texts whose `route_key` is not in `routes`
    (otherwise an unknown key raises `EmbedderError`).

```pycon
>>> import numpy as np
>>> from ef.embedders import FunctionEmbedder
>>> short = FunctionEmbedder(lambda ts: np.zeros((len(ts), 2)), model_id='s@2')
>>> long = FunctionEmbedder(lambda ts: np.ones((len(ts), 2)), model_id='l@2')
>>> m = MultiEmbedder({'s': short, 'l': long},
...                   predicate=lambda t: 's' if len(t) < 4 else 'l')
>>> m(['hi', 'hello']).tolist()
[[0.0, 0.0], [1.0, 1.0]]
```

### *class* ef.embedder_wrappers.NormalizingEmbedder(inner)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

L2-normalize an inner embedder’s output.

Use this for MRL-truncated vectors (which lose unit norm when sliced to a
smaller dimension) or any backend that does not normalize by construction.
Reports `normalized=True`; `cache_key()` keys on `normalized`, so a
normalized and an un-normalized view of the same model never collide in a
shared cache.

* **Parameters:**
  **inner** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) – The embedder whose output to renormalize.

#### *property* honored_input_types *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['query', 'document', 'classification', 'clustering'], ...]*

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable’s items.

If the argument is a tuple, the return value is the same object.

### *class* ef.embedder_wrappers.RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0, jitter=0.3, retry_on=<function \_default_is_retryable>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Exponential-backoff retry policy for [`RetryingEmbedder`](#ef.embedder_wrappers.RetryingEmbedder).

#### max_attempts

Total tries (the first call counts as attempt 1).

#### base_delay

Seconds before the first retry; doubles each attempt.

#### max_delay

Upper bound on any single backoff sleep.

#### jitter

Random fraction (`0..1`) of the delay added on top, to
de-correlate concurrent clients.

#### retry_on

Predicate deciding whether an exception is worth retrying.

* **Return type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

#### retry_on()

Heuristic: retry transient failures (429, 5xx, connection/timeout).

Per-request 400-class errors are *not* retried — the same request would fail
identically. Recognizes the `status_code`/`status` attribute carried by
most HTTP-client SDK exceptions (incl. the OpenAI SDK).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* ef.embedder_wrappers.RetryingEmbedder(inner, policy=None, \*, sleep=<built-in function sleep>)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

Retry an inner embedder’s calls on transient failures.

* **Parameters:**
  * **inner** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) – The embedder to guard.
  * **policy** ([`RetryPolicy`](#ef.embedder_wrappers.RetryPolicy) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A [`RetryPolicy`](#ef.embedder_wrappers.RetryPolicy) (sensible defaults if omitted).
  * **sleep** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – The sleep function — injectable so tests run instantly.

#### *property* honored_input_types *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['query', 'document', 'classification', 'clustering'], ...]*

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable’s items.

If the argument is a tuple, the return value is the same object.

#### *property* normalized *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.
