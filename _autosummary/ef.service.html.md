# ef.service

The HTTP-service bridge — [`EfService`](#ef.service.EfService), a stateless-friendly facade.

`ef`’s query surface is **stateful**: [`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest)
returns a live [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus), and a
[`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) holds a live corpus, an
[`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) and `vd` collections. HTTP is
**stateless**. To serve `ef` over HTTP — so a frontend (`app_ef`) can reach
it through `qh.mk_app()` — something must map a JSON-friendly `corpus_id`
back to the live indexed object across requests.

[`EfService`](#ef.service.EfService) is that bridge: a facade holding a **handle registry**
`{corpus_id: SourceManager}` **on the instance**. A single `EfService()` is
constructed once at server start-up, and its seven bound methods —
[`create_corpus()`](#ef.service.EfService.create_corpus), [`search()`](#ef.service.EfService.search),
[`retrieve()`](#ef.service.EfService.retrieve), [`explore_corpus()`](#ef.service.EfService.explore_corpus),
[`corpus_info()`](#ef.service.EfService.corpus_info), [`list_corpora()`](#ef.service.EfService.list_corpora),
[`delete_corpus()`](#ef.service.EfService.delete_corpus) — are handed to `qh.mk_app()` as the HTTP
surface. The registry lives on the instance,
never as a module global: a process-wide mutable registry would be the
`ServiceContext` singleton anti-pattern `ef` rejects (`.claude/CLAUDE.md`
§6) — inject an [`EfService`](#ef.service.EfService) explicitly instead.

Every method is JSON-friendly and fully type-annotated, because `qh` derives
the HTTP schema from the type hints. `ef`’s stateful objects never cross the
boundary: a corpus is addressed by its string `corpus_id`, an embedder /
segmenter by a string the DI seam resolves (`"hashing"`, `"openai:..."`,
`"cohere:..."`, …), and results are plain [`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)
/ [`Segment`](ef.segments.html.md#ef.segments.Segment) data.

This is *only* a transport bridge — it adds no orchestration. `ef` stays a
facade (`.claude/CLAUDE.md` §6); answer synthesis, agents and UI live outside.

Example — create a corpus, search it, then drop it, all offline:

```pycon
>>> from ef.service import EfService
>>> service = EfService()
>>> info = service.create_corpus(
...     ['the cat sat on the mat', 'dogs are loyal', 'felines and canines'],
...     corpus_id='animals',
... )
>>> info['corpus_id'], info['n_sources'], info['n_segments']
('animals', 3, 3)
>>> info['embedder'], info['dim']
('hashing:v1@512', 512)
>>> hits = service.search('animals', 'cat', limit=2)
>>> len(hits)
2
>>> isinstance(hits[0].score, float)
True
>>> segments = service.retrieve('animals', 'cat', limit=1)
>>> isinstance(segments[0]['text'], str)
True
>>> [c['corpus_id'] for c in service.list_corpora()]
['animals']
>>> service.delete_corpus('animals')
>>> service.list_corpora()
[]
```

### Classes

| [`CorpusInfo`](#ef.service.CorpusInfo)                        | A JSON-friendly summary of one registered corpus.                         |
|------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`EfService`](#ef.service.EfService)(\*[, default_embedder]) | A registry-backed facade bridging stateful `ef` to a stateless transport. |

### *class* ef.service.CorpusInfo

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

A JSON-friendly summary of one registered corpus.

The return shape shared by [`EfService.create_corpus()`](#ef.service.EfService.create_corpus),
[`EfService.corpus_info()`](#ef.service.EfService.corpus_info) and (as a list) [`EfService.list_corpora()`](#ef.service.EfService.list_corpora)
— a plain `dict` an HTTP client and `qh`’s schema both understand.

Keys:
: corpus_id: the registry handle — how every other method addresses it.
  n_sources: the number of source documents in the corpus.
  n_segments: the number of indexed segments (sources cut by the segmenter).
  embedder: the embedder’s `model_id` (e.g. `"hashing:v1@512"`).
  dim: the embedding dimensionality.
  config_id: the content hash of the segmenter+embedder pipeline.

### *class* ef.service.EfService(, default_embedder=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A registry-backed facade bridging stateful `ef` to a stateless transport.

Construct one `EfService` per process, register corpora with
[`create_corpus()`](#ef.service.EfService.create_corpus), then query them by `corpus_id`. Hand the seven
bound methods to `qh.mk_app()` to expose them over HTTP — see the module
docstring.

The handle registry is private instance state — never a module global.
Corpora are isolated: each [`create_corpus()`](#ef.service.EfService.create_corpus) builds its own
[`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) over its own in-memory `vd`
backend, so one corpus’s vectors never leak into another’s search.

`default_embedder` sets the embedder [`create_corpus()`](#ef.service.EfService.create_corpus) resolves when
its own `embedder` argument is `None` — the per-instance hook a host
(e.g. `app_ef`) uses to pick a policy default without passing
`embedder=` on every call. It defaults to
[`DEFAULT_EMBEDDER`](ef.source_manager.html.md#ef.source_manager.DEFAULT_EMBEDDER) (`"hashing"`), so a bare
`EfService()` stays dependency-free and offline.

```pycon
>>> service = EfService()
>>> len(service)
0
>>> _ = service.create_corpus(['hello world'], corpus_id='greetings')
>>> 'greetings' in service
True
>>> len(service)
1
```

A host can swap the default embedder at construction (the string is resolved
lazily, per corpus, by the DI seam — constructing the service touches no
network):

```pycon
>>> EfService(default_embedder='openai:text-embedding-3-small')
<EfService: 0 corpus(es) registered>
```

#### corpus_info(corpus_id)

Return the [`CorpusInfo`](#ef.service.CorpusInfo) of a registered corpus.

* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `corpus_id` is not registered.
* **Return type:**
  [`CorpusInfo`](#ef.service.CorpusInfo)

#### create_corpus(sources, , embedder=None, segmenter=None, corpus_id=None, embedder_api_key=None)

Index `sources` into a new corpus, register it, return its [`CorpusInfo`](#ef.service.CorpusInfo).

* **Parameters:**
  * **sources** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the corpus — a list of text documents.
  * **embedder** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the embedder, as a string the DI seam resolves
    ([`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder)) — `"hashing"`,
    `"openai:text-embedding-3-small"`, `"cohere:..."`, an
    `http(s)://` URL, …. `None` → the service’s
    `default_embedder` (chosen at construction; itself
    [`DEFAULT_EMBEDDER`](ef.source_manager.html.md#ef.source_manager.DEFAULT_EMBEDDER), the dependency-free
    [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder), unless a host overrode it).
  * **segmenter** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the segmenter, as a string
    ([`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter)) — `None` → the
    recursive-character default.
  * **corpus_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the handle to register the corpus under; `None` → a
    fresh random id. Reusing a live id is an error.
  * **embedder_api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional API key for the embedder — the
    bring-your-own-key seam. When given and `embedder` is a
    hosted-API spec (`"openai:…"`, `"cohere:…"`, …), the key is
    forwarded to that adapter, so the service itself never needs to
    hold one. `None` → the adapter’s usual environment-variable
    resolution. Ignored for key-less embedders (e.g. `"hashing"`).
    The corpus’s embedder keeps the key for the life of the corpus,
    so [`search()`](#ef.service.EfService.search) / [`retrieve()`](#ef.service.EfService.retrieve) / [`explore_corpus()`](#ef.service.EfService.explore_corpus)
    need no key of their own.
* **Return type:**
  [`CorpusInfo`](#ef.service.CorpusInfo)
* **Returns:**
  the [`CorpusInfo`](#ef.service.CorpusInfo) of the freshly indexed corpus.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `corpus_id` is already registered.

#### delete_corpus(corpus_id)

Drop a corpus from the registry, releasing its index.

* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `corpus_id` is not registered.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### explore_corpus(corpus_id, , dims=2, projection_method='auto', cluster_method='kmeans', n_clusters=8, label=False)

Project & cluster a registered corpus — the corpus-map surface.

Runs [`ef.exploration.explore()`](ef.exploration.html.md#ef.exploration.explore) over the corpus: every indexed
segment is projected to `dims` coordinates and assigned a cluster,
returned as a row-aligned [`ExploreResult`](ef.exploration.html.md#ef.exploration.ExploreResult) (`ids` /
`coords` / `labels` / `cluster_titles`) — the JSON-friendly shape
an `app_ef` corpus map consumes.

* **Parameters:**
  * **corpus_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the corpus to explore.
  * **dims** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – projection target dimensionality — `2` or `3`.
  * **projection_method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"auto"` / `"umap"` / `"pca"`.
  * **cluster_method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `"kmeans"` / `"hdbscan"`.
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of k-means clusters.
  * **label** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – when `True`, also name each cluster with an LLM (needs the
    `ef[imbed]` extra and a key); default `False`.
* **Raises:**
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `corpus_id` is not registered.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the corpus has fewer than 2 indexed segments.
* **Return type:**
  [`ExploreResult`](ef.exploration.html.md#ef.exploration.ExploreResult)

#### list_corpora()

Return the [`CorpusInfo`](#ef.service.CorpusInfo) of every registered corpus.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`CorpusInfo`](#ef.service.CorpusInfo)]

#### retrieve(corpus_id, query, , limit=10)

Retrieve ranked [`Segment`](ef.segments.html.md#ef.segments.Segment)s — the RAG-plug-in shape.

Like [`search()`](#ef.service.EfService.search), but returns plain segments in rank order (the
`score` dropped, the `source_id` folded into `metadata["source"]`)
— clean context to hand to an external RAG/agent framework. `ef`
returns context; it does not synthesize answers.

* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `corpus_id` is not registered.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]

#### search(corpus_id, query, , limit=10)

Search a registered corpus — up to `limit` ranked [`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)s.

Each hit carries the matched [`Segment`](ef.segments.html.md#ef.segments.Segment), its
similarity `score` (higher = closer) and the `source_id` it was cut
from.

* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `corpus_id` is not registered.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)]
