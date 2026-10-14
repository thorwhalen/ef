# ef.reranking

Reranking — a precision pass over retrieved segments (use case F6).

First-stage retrieval (a dense ANN query over `vd`) is tuned for *recall*: it
casts a wide, cheap net. A **reranker** is the precision counterpart — a slower,
more accurate model that re-scores a handful of candidates against the query.
The standard two-stage pattern is “retrieve `fetch_k` cheaply, rerank, keep
the top `limit`”.

This module keeps reranking a *decorator over a retriever*, never a new pipeline
stage — it composes with the existing surface rather than complicating it:

- [`Reranker`](#ef.reranking.Reranker) — the structural protocol: a callable
  `(query, segments) -> one score per segment`. Any plain function of that
  shape *is* a reranker; no base class to subclass.
- [`rerank()`](#ef.reranking.rerank) — the pure operation: re-score and re-order a list of
  [`Segment`](ef.segments.html.md#ef.segments.Segment)s. The reranker’s score is folded into each
  > segment’s `metadata["rerank_score"]`.
- [`with_reranker()`](#ef.reranking.with_reranker) — the decorator: wrap a retrieve-style callable (such as
  [`SourceManager.retrieve`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve)) so
  > every call over-fetches, reranks, and trims — transparently.
- [`cross_encoder_reranker()`](#ef.reranking.cross_encoder_reranker) — a ready reranker backed by a
  sentence-transformers `CrossEncoder` (an optional heavy dependency, imported
  lazily — the protocol and [`rerank()`](#ef.reranking.rerank) cost nothing).

Example — rerank three segments by a toy length score, longest first:

```pycon
>>> segments = [
...     {'text': 'cat', 'id': '1'},
...     {'text': 'dog', 'id': '2'},
...     {'text': 'cathedral', 'id': '3'},
... ]
>>> def by_length(query, segments):
...     return [len(segment['text']) for segment in segments]
>>> [segment['text'] for segment in rerank('a query', segments, by_length, limit=2)]
['cathedral', 'cat']
```

### Functions

| [`rerank`](#ef.reranking.rerank)(query, segments, reranker, \*[, limit])    | Re-score `segments` against `query` with `reranker` and re-order them.                                              |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`with_reranker`](#ef.reranking.with_reranker)(retriever, reranker, \*[, fetch_k]) | Decorate a retrieve-style callable with a two-stage reranking pass.                                                 |
| [`cross_encoder_reranker`](#ef.reranking.cross_encoder_reranker)([model_name, device, ...]) | Build a [`Reranker`](#ef.reranking.Reranker) backed by a sentence-transformers `CrossEncoder`. |

### Classes

| [`Reranker`](#ef.reranking.Reranker)(\*args, \*\*kwargs)   | A callable that re-scores `(query, segments)` — higher means more relevant.   |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### *class* ef.reranking.Reranker(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A callable that re-scores `(query, segments)` — higher means more relevant.

The structural contract a reranker satisfies: given the query and a list of
candidate [`Segment`](ef.segments.html.md#ef.segments.Segment)s, return \*\*one score per
segment\*\*, in the same order. [`rerank()`](#ef.reranking.rerank) sorts by these scores. It is a
`@runtime_checkable Protocol`, not a base class — a plain function of the
right shape already *is* a [`Reranker`](#ef.reranking.Reranker).

```pycon
>>> def by_length(query, segments):
...     return [len(s['text']) for s in segments]
>>> isinstance(by_length, Reranker)
True
```

### ef.reranking.cross_encoder_reranker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', , device=None, batch_size=32, \*\*predict_kwargs)

Build a [`Reranker`](#ef.reranking.Reranker) backed by a sentence-transformers `CrossEncoder`.

A cross-encoder scores a `(query, passage)` pair *jointly* — it is the
accurate, slower second stage two-stage retrieval is built for. The model is
a heavy, optional dependency: it is imported only when this factory is
called, so importing [`ef.reranking`](#module-ef.reranking) itself stays cheap.

Example:

```default
from ef import ingest, with_reranker, cross_encoder_reranker

index = ingest(corpus, embedder="st:all-MiniLM-L6-v2")
retrieve = with_reranker(index.retrieve, cross_encoder_reranker())
segments = retrieve("a query", limit=5)
```

* **Parameters:**
  * **model_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – a sentence-transformers `CrossEncoder` model name.
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – torch device (`"cuda"` / `"cpu"` / `"mps"`); `None`
    auto-selects.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – how many pairs to score per forward pass.
  * **predict_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – extra keyword arguments forwarded to `CrossEncoder.predict`.
* **Return type:**
  [`Reranker`](#ef.reranking.Reranker)
* **Returns:**
  a [`Reranker`](#ef.reranking.Reranker) — `(query, segments) -> scores`.
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – if the optional `sentence-transformers` package is absent.

### ef.reranking.rerank(query, segments, reranker, , limit=None)

Re-score `segments` against `query` with `reranker` and re-order them.

The pure heart of the module — no I/O, no retrieval. The `reranker` is
called once with the whole candidate list; the segments are returned sorted
by descending score (ties keep their incoming order), optionally trimmed to
`limit`. Each returned segment is a copy carrying the reranker’s score in
`metadata["rerank_score"]` — so the new ranking is auditable.

* **Parameters:**
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the search query the segments are scored against.
  * **segments** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]) – the candidate segments (from a first-stage retriever).
  * **reranker** ([`Reranker`](#ef.reranking.Reranker)) – a [`Reranker`](#ef.reranking.Reranker) — a callable `(query, segments) -> scores`.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – keep only the top `limit` after reranking; `None` keeps all.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]
* **Returns:**
  the reranked segments — copies, best first.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the reranker returns a number of scores that does not
      match the number of segments.

```pycon
>>> segs = [{'text': 'a', 'id': '1'}, {'text': 'bbb', 'id': '2'}]
>>> ranked = rerank('q', segs, lambda q, s: [len(x['text']) for x in s])
>>> [r['text'] for r in ranked]
['bbb', 'a']
>>> ranked[0]['metadata']['rerank_score']
3.0
```

### ef.reranking.with_reranker(retriever, reranker, , fetch_k=50)

Decorate a retrieve-style callable with a two-stage reranking pass.

Wraps `retriever` (e.g. [`SourceManager.retrieve`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve), or any callable
`query -> list[Segment]`) so every call over-fetches `fetch_k`
candidates cheaply, [`rerank()`](#ef.reranking.rerank)s them, and returns the top `limit`.
The decorated callable keeps the retriever’s interface — `query` plus a
keyword-only `limit` — so it is a drop-in replacement, including as the
`retriever` argument of [`evaluate_retrieval()`](ef.evaluation.html.md#ef.evaluation.evaluate_retrieval).

* **Parameters:**
  * **retriever** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]) – the first-stage retriever — a callable
    `retriever(query, *, limit=...) -> segments`.
  * **reranker** ([`Reranker`](#ef.reranking.Reranker)) – the [`Reranker`](#ef.reranking.Reranker) applied to the over-fetched candidates.
  * **fetch_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – how many candidates to retrieve before reranking. Larger trades
    latency for recall headroom; it should comfortably exceed the
    `limit` callers ask for.
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]
* **Returns:**
  a new callable `(query, *, limit=10, **kwargs) -> list[Segment]` —
  extra keyword arguments (e.g. `filter`) pass straight through to
  `retriever`.

```pycon
>>> pool = [{'text': t, 'id': t} for t in ['ab', 'abcd', 'abc']]
>>> def base(query, *, limit=10):
...     return pool[:limit]
>>> retrieve = with_reranker(
...     base, lambda q, segs: [len(s['text']) for s in segs], fetch_k=10)
>>> [s['text'] for s in retrieve('a query', limit=2)]
['abcd', 'abc']
```
