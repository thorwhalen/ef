# ef.source_manager

The search facade — one-shot [`ingest()`](#ef.source_manager.ingest), [`SearchableCorpus`](#ef.source_manager.SearchableCorpus), [`SourceManager`](#ef.source_manager.SourceManager).

This is where `ef`’s layers come together. Phases 1–4 built the parts — a
[`Corpus`](ef.corpus.html.md#ef.corpus.Corpus), a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter), an
[`Embedder`](ef.embedders.html.md#ef.embedders.Embedder), the content-addressed
[`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) — and this module wires
**corpus → segment → embed → vd** into a usable search facade with the
progressive disclosure `ef` promises:

- **The light path** — [`ingest()`](#ef.source_manager.ingest) — one line. A list of strings (or any
  corpus) in, a [`SearchableCorpus`](#ef.source_manager.SearchableCorpus) out, `search(query)` ready.
- **The heavy path** — [`SourceManager`](#ef.source_manager.SourceManager) — the same machinery with the
  components named explicitly and **multiple configs** (a config is one
  segmenter+embedder pipeline). Two configs that share a step share that step’s
  artifacts in the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) for free — config
  branching costs only the divergent cone.
- **Keeping it fresh** — [`SourceManager.diagnose()`](#ef.source_manager.SourceManager.diagnose) reports how far the
  index has drifted from the corpus (the four staleness conditions);
  [`SourceManager.refresh()`](#ef.source_manager.SourceManager.refresh) re-syncs it. Pass `auto_refresh=True` to keep
  > the index live as the corpus is edited. See [`ef.diagnostics`](ef.diagnostics.html.md#module-ef.diagnostics) and

  [`ef.refresh`](ef.refresh.html.md#module-ef.refresh).

How the wiring works (one source, one config):

1. the source is a graph **leaf**, keyed by [`content_hash()`](ef.corpus.html.md#ef.corpus.content_hash);
2. a `segment` [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec) over that leaf →
   a `list` of [`Segment`](ef.segments.html.md#ef.segments.Segment)s;
3. an `embed` [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec) over the segment
   artifact → an `(n, dim)` array;
4. each `(segment, vector)` pair is written into a `vd` collection as a
   `vd.Document`, with `source_id` / `source_hash` / `config_hash`
   > recorded in its metadata.

`ef` owns the embedding: it always supplies vectors to `vd` (the index is
`vd`’s job, the vectors are `ef`’s — design notes §7), and at query time it
embeds the query with `input_type="query"` and hands `vd` the vector.

Example — the light path, end to end, offline:

```pycon
>>> import numpy as np
>>> from ef import ingest, as_embedder
>>> toy = as_embedder(
...     lambda texts: np.array(
...         [[t.count('a'), t.count('e'), t.count('o')] for t in texts], dtype=float
...     ),
...     model_id='toy-vowels@3',
... )
>>> idx = ingest(['apple orchard', 'ocean breeze', 'green meadow'], embedder=toy)
>>> hits = idx.search('ocean breeze', limit=2)
>>> hits[0].segment['text']
'ocean breeze'
>>> round(hits[0].score, 6)
1.0
>>> hits[0].source_id is not None
True
```

And the true light path — no `embedder` at all, nothing to install beyond
`ef` itself (the default is the dependency-free
[`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder)):

```pycon
>>> idx = ingest(['apple orchard', 'ocean breeze', 'green meadow'])
>>> idx.search('ocean breeze', limit=1)[0].segment['text']
'ocean breeze'
```

### Module Attributes

| [`DEFAULT_EMBEDDER`](#ef.source_manager.DEFAULT_EMBEDDER)   | The model name [`ingest()`](#ef.source_manager.ingest) / [`SourceManager`](#ef.source_manager.SourceManager) resolve an absent `embedder` to — [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder), the dependency-free default (numpy only, no extra to install).   |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`hits_to_segments`](#ef.source_manager.hits_to_segments)(hits)                          | Project [`SearchHit`](#ef.source_manager.SearchHit)s to plain [`Segment`](ef.segments.html.md#ef.segments.Segment)s.   |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ingest`](#ef.source_manager.ingest)(sources, \*[, segmenter, embedder, ...]) | Index `sources` and return a ready-to-search [`SearchableCorpus`](#ef.source_manager.SearchableCorpus) — one call.                                    |

### Classes

| [`SearchHit`](#ef.source_manager.SearchHit)(segment, score[, source_id])            | One ranked search result — a [`Segment`](ef.segments.html.md#ef.segments.Segment) and its score.   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| [`SearchableCorpus`](#ef.source_manager.SearchableCorpus)(collection, embedder, \*[, ...]) | A ready-to-search view over one indexed config — the thin facade.                                                          |
| [`SourceManager`](#ef.source_manager.SourceManager)([corpus, segmenter, embedder, ...]) | The heavy search facade — multi-config corpus indexing over one corpus.                                                    |

### ef.source_manager.DEFAULT_EMBEDDER *= 'hashing'*

The model name [`ingest()`](#ef.source_manager.ingest) / [`SourceManager`](#ef.source_manager.SourceManager) resolve an absent
`embedder` to — [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder), the dependency-free
default (numpy only, no extra to install). It is a lexical embedder; pass an
explicit `embedder` (`"st:..."`, `"openai:..."`, …) for neural
semantic quality.

### *class* ef.source_manager.SearchHit(segment, score, source_id=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One ranked search result — a [`Segment`](ef.segments.html.md#ef.segments.Segment) and its score.

[`SearchableCorpus.search()`](#ef.source_manager.SearchableCorpus.search) / [`SourceManager.search()`](#ef.source_manager.SourceManager.search) return a
ranked `list` of these. Keeping [`Segment`](ef.segments.html.md#ef.segments.Segment) (the
interchange `TypedDict`) pure of a result-only `score` key, a
[`SearchHit`](#ef.source_manager.SearchHit) *wraps* it alongside the similarity `score` and the
`source_id` of the document the segment was cut from.

#### segment

the matched segment — the canonical [`Segment`](ef.segments.html.md#ef.segments.Segment).

* **Type:**
  [`Segment`](ef.segments.html.md#ef.segments.Segment)

#### score

the similarity score from the vector store (higher = closer).

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### source_id

the corpus key of the source document, if recorded.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> hit = SearchHit(segment={'text': 'hi', 'id': 'x'}, score=0.9, source_id='doc-1')
>>> hit.segment['text'], hit.score, hit.source_id
('hi', 0.9, 'doc-1')
```

### *class* ef.source_manager.SearchableCorpus(collection, embedder, , config=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A ready-to-search view over one indexed config — the thin facade.

What [`ingest()`](#ef.source_manager.ingest) returns and what [`SourceManager.searchable()`](#ef.source_manager.SourceManager.searchable) hands
out: a small, read-mostly object that knows just enough to answer a query —
the `vd` collection holding the vectors and the [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
that embeds the query. It does **not** re-index; mutation, refresh and
multi-config live on [`SourceManager`](#ef.source_manager.SourceManager).

* **Parameters:**
  * **collection** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the `vd` collection holding this config’s vectors.
  * **embedder** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) – the embedder — used to embed queries with `input_type="query"`.
  * **config** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the [`ConfigId`](ef.config.html.md#ef.config.ConfigId) of the indexed pipeline, if known.

```pycon
>>> # built for you by ingest(); see the module docstring for a full example.
```

#### retrieve(query, , limit=10, filter=None)

Retrieve ranked [`Segment`](ef.segments.html.md#ef.segments.Segment)s — the RAG-plug-in surface.

The name an external RAG/agent framework reaches for. Where
[`search()`](#ef.source_manager.SearchableCorpus.search) returns scored [`SearchHit`](#ef.source_manager.SearchHit)s (for inspecting the
ranking), `retrieve` returns **plain segments** in rank order — clean
context to hand to an LLM, with no `ef`-specific type to learn (design
notes §F5). `ef` returns the context; it does *not* synthesize answers.
See [`hits_to_segments()`](#ef.source_manager.hits_to_segments) for how the score is dropped and the
`source_id` provenance is preserved in `metadata["source"]`.

```pycon
>>> # idx.retrieve('a query') -> [{'text': ..., 'id': ..., 'metadata': ...}, ...]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]

#### search(query, , limit=10, filter=None)

Search the indexed corpus — return up to `limit` ranked [`SearchHit`](#ef.source_manager.SearchHit)s.

`query` is text (embedded with `input_type="query"`) or an already
computed query vector. `filter` is a `vd` metadata filter
(MongoDB-style) — see `vd.filters`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](#ef.source_manager.SearchHit)]

### *class* ef.source_manager.SourceManager(corpus=None, , segmenter=None, embedder=None, store=None, cache=None, auto_refresh=False, embedder_api_key=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The heavy search facade — multi-config corpus indexing over one corpus.

A [`SourceManager`](#ef.source_manager.SourceManager) holds a corpus, an [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph)
and one or more **configs**. A config is a named segmenter+embedder pipeline
([`PipelineSpec`](ef.config.html.md#ef.config.PipelineSpec)); [`register_config()`](#ef.source_manager.SourceManager.register_config) adds one,
[`materialize()`](#ef.source_manager.SourceManager.materialize) runs the corpus through it into a `vd` collection, and
[`search()`](#ef.source_manager.SourceManager.search) queries it. Because every artifact is content-addressed in
the shared graph, two configs that share a step (e.g. the same segmenter)
compute that step’s artifacts **once** — config branching is free.

The constructor optionally registers a `"default"` config: pass an
`embedder` (and optionally a `segmenter`) and it is registered eagerly;
omit `embedder` to build every config explicitly via [`register_config()`](#ef.source_manager.SourceManager.register_config).

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the source corpus — coerced by [`as_corpus()`](ef.corpus.html.md#ef.corpus.as_corpus)
    (a mapping, a directory path, an iterable of sources, or `None`).
  * **segmenter** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the default config’s segmenter — coerced by
    [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter) (`None` → the recursive
    default). Used only when `embedder` is also given.
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the default config’s embedder — coerced by
    [`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder). If given, a `"default"`
    config is registered; if `None`, no default config is registered.
  * **embedder_api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional API key for the default config’s embedder
    — forwarded to a hosted-API adapter (`"openai:…"` etc.) when the
    `embedder` is such a spec. The bring-your-own-key seam; ignored
    for key-less embedders. See [`register_config()`](#ef.source_manager.SourceManager.register_config).
  * **store** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – where the vectors go. `None` → an in-memory `vd` backend; a
    `vd` client → collections are created on it; a `vd` collection →
    used directly (then only one config is supported); a backend-name
    string → `vd.connect` of that backend.
  * **cache** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph)’s value cache — any
    `MutableMapping`; `None` → an in-RAM `dict`.
  * **auto_refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – when `True`, the corpus is wrapped in a
    [`ChangeDetectingCorpus`](ef.corpus.html.md#ef.corpus.ChangeDetectingCorpus) and every edit made
    *through* it is incrementally re-indexed into each materialized
    config — the index stays live without an explicit
    [`refresh()`](#ef.source_manager.SourceManager.refresh). [`scan()`](#ef.source_manager.SourceManager.scan) then also picks up out-of-band edits.

```pycon
>>> import numpy as np
>>> from ef import as_embedder
>>> e1 = as_embedder(lambda ts: np.array([[len(t)] for t in ts], float), model_id='len@1')
>>> sm = SourceManager(['hello', 'hi there'], embedder=e1)
>>> report = sm.materialize()
>>> report['segments']
2
>>> len(sm.search('hello')) >= 1
True
```

#### *property* configs *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

The registered configs as a `{name: config_id}` mapping.

#### diagnose(config=None)

Report how far one config’s index has drifted from the corpus.

Computes the four staleness conditions — `orphan` / `missing` /
`stale` / `misconfigured` — by comparing the config’s `vd`
collection against the current corpus (see [`ef.diagnostics`](ef.diagnostics.html.md#module-ef.diagnostics)).
Read-only: it changes nothing. [`refresh()`](#ef.source_manager.SourceManager.refresh) acts on the result.

```pycon
>>> sm = SourceManager({'a': 'alpha', 'b': 'beta'}, embedder='hashing')
>>> _ = sm.materialize()
>>> bool(sm.diagnose())                  # freshly indexed — in sync
False
>>> sm.corpus['a'] = 'alpha rewritten'
>>> sm.diagnose().stale
('a',)
```

* **Return type:**
  [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport)

#### gc_orphans(config=None)

Delete the documents of sources that left the corpus — return the count.

The garbage-collection slice of [`refresh()`](#ef.source_manager.SourceManager.refresh): it removes only
**orphan** documents (a source gone from the corpus) and prunes the
dead leaves out of the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph),
touching nothing else. `gc_orphans()` is the orphan-deletion half of
`refresh(mode='full')`.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### lineage(key)

The artifacts `key` was produced from — its provenance.

A thin pass-through to [`ArtifactGraph.ancestors`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph.ancestors): “what produced this
vector / segment?”. The leaves of the returned set are the source
content hashes the artifact ultimately derives from.

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### materialize(config=None, , sources=None)

Index the corpus through one config (or all) — corpus → segment → embed → `vd`.

For each source and each selected config: the source is added to the
[`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) as a content-addressed leaf,
its `segment` and `embed` artifacts are [`materialize()`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph.materialize)d
(a cache hit if already computed — so a shared segment step runs once
across configs), and every `(segment, vector)` pair is written into
the config’s `vd` collection. Idempotent: re-materializing rewrites
identical documents.

* **Parameters:**
  * **config** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the config to index — a name, a [`ConfigId`](ef.config.html.md#ef.config.ConfigId),
    or `None` for *every* registered config.
  * **sources** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional subset of source ids to index — `None` (the
    default) indexes the whole corpus. [`refresh()`](#ef.source_manager.SourceManager.refresh) uses this to
    re-index only the sources that changed; ids not in the corpus
    are skipped.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  a report `dict` with `configs` / `sources` / `segments` counts.

#### pipeline(config=None)

The [`PipelineSpec`](ef.config.html.md#ef.config.PipelineSpec) of a registered config.

* **Return type:**
  [`PipelineSpec`](ef.config.html.md#ef.config.PipelineSpec)

#### rebuild(config=None)

Drop one config’s index entirely and re-index the whole corpus.

The heavy hammer: every document in the config’s `vd` collection is
deleted, then the whole corpus is re-[`materialize()`](#ef.source_manager.SourceManager.materialize)d from
scratch. Use it to recover from a config whose `vd` documents were
produced by an earlier pipeline (every source `misconfigured`), or
whenever a guaranteed-clean index is wanted over the surgical
[`refresh()`](#ef.source_manager.SourceManager.refresh). Every source is reported as `added` — the collection
was emptied and rebuilt.

* **Return type:**
  [`RefreshReport`](ef.refresh.html.md#ef.refresh.RefreshReport)
* **Returns:**
  a [`RefreshReport`](ef.refresh.html.md#ef.refresh.RefreshReport).

#### refresh(config=None, , sources=None, mode='full')

Re-sync one config’s `vd` index with the current corpus.

The explicit-refresh entry point. It [`diagnose()`](#ef.source_manager.SourceManager.diagnose)s the config,
turns the staleness report into a [`RefreshPlan`](ef.refresh.html.md#ef.refresh.RefreshPlan) for
the chosen `mode` ([`plan_refresh()`](ef.refresh.html.md#ef.refresh.plan_refresh)), deletes the
stale/orphan documents, re-[`materialize()`](#ef.source_manager.SourceManager.materialize)s what changed, and
prunes the dead leaves out of the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph).
Because it re-reads and re-hashes the corpus, it catches out-of-band
edits without any change-detection wrapper.

* **Parameters:**
  * **config** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the config to refresh (name / id / `None` for the default
    or sole config).
  * **sources** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional subset of source ids to refresh — sources
    outside it are left untouched. Pair it with `mode='scoped_full'`
    (or `'incremental'`); `'full'` rejects it.
  * **mode** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'none'`, `'incremental'`, `'full'`, `'scoped_full'`]) – one of [`REFRESH_MODES`](ef.refresh.html.md#ef.refresh.REFRESH_MODES) — `'none'` (index
    new/changed content, delete nothing), `'incremental'` (also
    replace changed sources), `'full'` (also delete orphans — the
    corpus is authoritative), `'scoped_full'` (`full` over the
    `sources` subset). Default `'full'`.
* **Return type:**
  [`RefreshReport`](ef.refresh.html.md#ef.refresh.RefreshReport)
* **Returns:**
  a [`RefreshReport`](ef.refresh.html.md#ef.refresh.RefreshReport).

```pycon
>>> sm = SourceManager({'a': 'alpha text', 'b': 'beta text'}, embedder='hashing')
>>> _ = sm.materialize()
>>> sm.corpus['b'] = 'beta rewritten'        # edit
>>> sm.corpus['c'] = 'gamma text'            # add
>>> del sm.corpus['a']                       # delete
>>> report = sm.refresh()
>>> report.added, report.modified, report.deleted
(('c',), ('b',), ('a',))
>>> bool(sm.diagnose())                      # in sync again
False
```

#### register_config(name='default', , segmenter=None, embedder=None, embedder_api_key=None)

Register a named segmenter+embedder pipeline; return its [`ConfigId`](ef.config.html.md#ef.config.ConfigId).

The segmenter and embedder are coerced through `ef`’s DI seams
([`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter) /
[`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder); an absent `embedder`
defaults to [`DEFAULT_EMBEDDER`](#ef.source_manager.DEFAULT_EMBEDDER)). Their ops are registered into the
shared [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) and a `vd` collection
is created for the config. Registering a config does *not* index it —
call [`materialize()`](#ef.source_manager.SourceManager.materialize).

`embedder_api_key` is the per-call bring-your-own-key seam: when given
and `embedder` is a hosted-API provider spec (`"openai:…"` etc.),
the key is forwarded to that adapter instead of relying on a server
environment variable. Ignored for key-less embedders.

Re-registering the same `name` replaces it. Two configs that resolve
to the same [`PipelineSpec`](ef.config.html.md#ef.config.PipelineSpec) share one
[`ConfigId`](ef.config.html.md#ef.config.ConfigId) (and one collection).

```pycon
>>> import numpy as np
>>> from ef import as_embedder
>>> sm = SourceManager(['a', 'b'])
>>> cid = sm.register_config(
...     'mini', embedder=as_embedder(
...         lambda ts: np.ones((len(ts), 2)), model_id='ones@2'))
>>> len(cid)
64
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### retrieve(query, , config=None, limit=10, filter=None)

Retrieve ranked [`Segment`](ef.segments.html.md#ef.segments.Segment)s — the RAG-plug-in surface.

The [`SourceManager`](#ef.source_manager.SourceManager) counterpart of
[`SearchableCorpus.retrieve()`](#ef.source_manager.SearchableCorpus.retrieve): plain segments in rank order, the
clean context shape an external RAG/agent framework consumes (`ef`
does not synthesize answers). [`search()`](#ef.source_manager.SourceManager.search) is the scored counterpart.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]

#### scan()

Re-scan the corpus for out-of-band edits — needs `auto_refresh=True`.

Picks up changes made to the backing store *directly* (a file edited on
disk, an S3 object replaced) rather than through the corpus wrapper.
When the manager was built with `auto_refresh=True` the corpus is a
[`ChangeDetectingCorpus`](ef.corpus.html.md#ef.corpus.ChangeDetectingCorpus); this calls its
[`scan()`](ef.corpus.html.md#ef.corpus.ChangeDetectingCorpus.scan), which fires a
[`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent) per drift — each one auto-applied to
every materialized config. Returns the [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff).

For a large bulk of out-of-band edits, prefer [`refresh()`](#ef.source_manager.SourceManager.refresh) (one
corpus pass) over `scan` (one collection pass per changed source).

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if the corpus is not change-detecting (the manager was
      not built with `auto_refresh=True`).
* **Return type:**
  [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff)

#### search(query, , config=None, limit=10, filter=None)

Search one config’s index — return up to `limit` ranked [`SearchHit`](#ef.source_manager.SearchHit)s.

`config` selects which config to query (a name / [`ConfigId`](ef.config.html.md#ef.config.ConfigId),
or `None` for the default or sole config). `query` is text or a
pre-computed query vector; `filter` is a `vd` metadata filter.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](#ef.source_manager.SearchHit)]

#### searchable(config=None)

Return a thin [`SearchableCorpus`](#ef.source_manager.SearchableCorpus) bound to one config’s index.

* **Return type:**
  [`SearchableCorpus`](#ef.source_manager.SearchableCorpus)

### ef.source_manager.hits_to_segments(hits)

Project [`SearchHit`](#ef.source_manager.SearchHit)s to plain [`Segment`](ef.segments.html.md#ef.segments.Segment)s.

The shape [`SearchableCorpus.retrieve()`](#ef.source_manager.SearchableCorpus.retrieve) / [`SourceManager.retrieve()`](#ef.source_manager.SourceManager.retrieve)
hand to an external RAG/agent framework — the RAG-plug-in surface (design
notes §F5). A [`SearchHit`](#ef.source_manager.SearchHit) carries an `ef`-specific `score` and
`source_id`; a [`Segment`](ef.segments.html.md#ef.segments.Segment) is a plain `TypedDict` an
external framework already understands (and trivially adapts to a LangChain
`Document` / LlamaIndex `TextNode` / a Ragas `retrieved_contexts`
`list[str]` via `[s["text"] for s in segments]`).

The similarity `score` is dropped — [`Segment`](ef.segments.html.md#ef.segments.Segment) is the
interchange type and stays pure of a result-only key (call `search()`
when the score matters). Provenance is *not* lost: each hit’s `source_id`
is folded into the segment’s `metadata` under the conventional `"source"`
key (one of [`PROMOTED_METADATA_KEYS`](ef.segments.html.md#ef.segments.PROMOTED_METADATA_KEYS)), so a plain segment
still records which source document it was retrieved from. A segment that
already carries `metadata["source"]` keeps its own value.

```pycon
>>> hit = SearchHit(segment={'text': 'hi', 'id': 'x'}, score=0.9, source_id='doc-1')
>>> segs = hits_to_segments([hit])
>>> segs[0]['text'], segs[0]['metadata']['source']
('hi', 'doc-1')
>>> 'score' in segs[0]                          # the score is not a Segment key
False
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]

### ef.source_manager.ingest(sources, , segmenter=None, embedder=None, store=None, cache=None)

Index `sources` and return a ready-to-search [`SearchableCorpus`](#ef.source_manager.SearchableCorpus) — one call.

The light path: corpus in, `search`-ready object out, every component
defaulted. `ingest(['a', 'b', 'c']).search('query')` is the whole story.
It builds a single-config [`SourceManager`](#ef.source_manager.SourceManager), [`materialize()`](#ef.source_manager.SourceManager.materialize)s
it, and hands back the thin [`SearchableCorpus`](#ef.source_manager.SearchableCorpus).

* **Parameters:**
  * **sources** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the corpus — coerced by [`as_corpus()`](ef.corpus.html.md#ef.corpus.as_corpus) (a mapping,
    a directory path, an iterable of strings/mappings, or `None`).
  * **segmenter** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the segmenter — [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter)
    coerces it (`None` → the recursive default).
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the embedder — [`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder)
    coerces it; `None` → [`DEFAULT_EMBEDDER`](#ef.source_manager.DEFAULT_EMBEDDER), the dependency-free
    [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder) (so this call needs nothing
    beyond `pip install ef`). Pass an explicit embedder — e.g.
    `"st:all-MiniLM-L6-v2"` — for neural semantic quality.
  * **store** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the vector store — `None` → an in-memory `vd` backend. See
    [`SourceManager`](#ef.source_manager.SourceManager) for the other accepted forms.
  * **cache** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the artifact-graph value cache — any `MutableMapping`.
* **Return type:**
  [`SearchableCorpus`](#ef.source_manager.SearchableCorpus)
* **Returns:**
  a [`SearchableCorpus`](#ef.source_manager.SearchableCorpus) over the freshly indexed corpus.

See the module docstring for a complete offline example.
