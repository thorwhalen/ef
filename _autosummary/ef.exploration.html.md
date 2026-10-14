# ef.exploration

Layer L5 — *explore* a corpus: project to 2-D/3-D, cluster, label clusters.

This is `ef`’s visualization heritage, inherited from `imbed` and
re-presented as a **secondary** surface. The *primary* surface of `ef` is
search / RAG / corpus-indexing (see [`ef.ingest()`](ef.html.md#ef.ingest) and
[`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager)); `explore` is the “see the shape of
the corpus” tool — the backend an `app_ef` corpus map consumes.

Four free functions, each accepting a corpus *or* an already-computed vector
matrix (whatever `_resolve_vectors()` understands):

- [`project()`](#ef.exploration.project) — reduce embeddings to 2-D/3-D coordinates (PCA → UMAP).
- [`cluster()`](#ef.exploration.cluster) — group embeddings into clusters (k-means / HDBSCAN).
- [`label_clusters()`](#ef.exploration.label_clusters) — name clusters with an LLM (via `imbed`).
- [`explore()`](#ef.exploration.explore) — the orchestrator: project + cluster (+ optionally label) in
  one call, returning a structured, JSON-friendly [`ExploreResult`](#ef.exploration.ExploreResult)
  (`ids` / `coords` / `labels` / `cluster_titles`) — the surface an
  `app_ef` corpus map (or [`ef.service.EfService.explore_corpus()`](ef.service.html.md#ef.service.EfService.explore_corpus))
  consumes directly.

The module imports with **numpy only**. UMAP, HDBSCAN and `imbed` are
optional: each is imported lazily, inside the function that needs it, and only
that heavy path requires the corresponding extra (`ef[explore]` /
`ef[imbed]`). The numpy-only paths — PCA projection and k-means clustering —
work out of the box with no install.

Example — project a small set of vectors to the plane (numpy only):

```pycon
>>> import numpy as np
>>> vectors = np.random.RandomState(0).rand(12, 16)
>>> coords = project(vectors, dims=2, method='pca')
>>> coords.shape
(12, 2)
>>> labels = cluster(vectors, method='kmeans', n_clusters=3, random_state=0)
>>> labels.shape
(12,)
```

Each returned array’s rows are in the iteration order of the input — for a
[`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus) that is `list(searchable.collection)`,
which the caller zips back against to recover `{id: coords}`. [`explore()`](#ef.exploration.explore)
does that zip-back for you — its [`ExploreResult`](#ef.exploration.ExploreResult) keeps `ids` aligned
with `coords` and `labels`.

### Functions

| [`project`](#ef.exploration.project)(data, \*[, dims, method, embedder, ...])   | Project a corpus's embeddings to `dims` coordinates for visualization.   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`cluster`](#ef.exploration.cluster)(data, \*[, method, n_clusters, ...])       | Cluster a corpus's embeddings (use cases §G2).                           |
| [`label_clusters`](#ef.exploration.label_clusters)(segments, labels, \*[, ...])        | Name each cluster with a short LLM-generated title (use cases §G3).      |
| [`explore`](#ef.exploration.explore)(data, \*[, dims, projection_method, ...])  | Project, cluster and (optionally) label a corpus in one call.            |

### Classes

| [`ExploreResult`](#ef.exploration.ExploreResult)   | A structured, JSON-friendly corpus-exploration result.   |
|------------------------------------------------------------------|----------------------------------------------------------|

### *class* ef.exploration.ExploreResult

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

A structured, JSON-friendly corpus-exploration result.

What [`explore()`](#ef.exploration.explore) returns and [`ef.service.EfService.explore_corpus()`](ef.service.html.md#ef.service.EfService.explore_corpus)
serves. Every list is **row-aligned**: `ids[i]`, `coords[i]` and
`labels[i]` all describe the same item.

Keys:
: ids: the per-item identifiers — a corpus’s keys, or positional `"0"`,
  : `"1"`, … for an id-less vector matrix.
  <br/>
  coords: the projected coordinates — one `[x, y]` (or `[x, y, z]`)
  : row per item.
  <br/>
  labels: the cluster id of each item (HDBSCAN marks noise `-1`).
  cluster_titles: `{cluster_id: title}` — empty unless [`explore()`](#ef.exploration.explore)
  <br/>
  > was called with `label=True`.

### ef.exploration.cluster(data, , method='kmeans', n_clusters=8, min_cluster_size=5, embedder=None, normalize=True, random_state=42)

Cluster a corpus’s embeddings (use cases §G2).

`method='kmeans'` is the default and is **dependency-free** — a numpy
Lloyd’s algorithm with k-means++ seeding. `method='hdbscan'` finds a
density-based clustering (and a noise label `-1`) and needs the
`ef[explore]` extra.

* **Parameters:**
  * **data** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – what to cluster — see [`project()`](#ef.exploration.project) / `_resolve_vectors()`.
  * **method** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'kmeans'`, `'hdbscan'`]) – `'kmeans'` (numpy-only) or `'hdbscan'` (optional extra).
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of clusters for k-means (clamped to `n_samples`).
  * **min_cluster_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – smallest cluster HDBSCAN will form.
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – embedder used when `data` is raw text.
  * **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – L2-normalize the vectors first, so Euclidean distance
    behaves like cosine similarity — the right default for semantic
    embeddings. Set `False` to cluster raw vectors.
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – seed for k-means initialization.
* **Return type:**
  `ndarray`
* **Returns:**
  An `ndarray` of integer cluster labels, shape `(n_samples,)`, in
  input order. HDBSCAN labels unclustered points `-1`.

```pycon
>>> import numpy as np
>>> vectors = np.random.RandomState(2).rand(15, 8)
>>> labels = cluster(vectors, method='kmeans', n_clusters=3, random_state=0)
>>> labels.shape
(15,)
>>> set(labels.tolist()) <= {0, 1, 2}
True
```

### ef.exploration.explore(data, , dims=2, projection_method='auto', cluster_method='kmeans', n_clusters=8, min_cluster_size=5, label=False, context=' ', n_words=4, embedder=None, random_state=42)

Project, cluster and (optionally) label a corpus in one call.

The layer-L5 orchestrator — [`project()`](#ef.exploration.project) + [`cluster()`](#ef.exploration.cluster) (+ optionally
[`label_clusters()`](#ef.exploration.label_clusters)) wired into one structured result. Where
[`project()`](#ef.exploration.project) / [`cluster()`](#ef.exploration.cluster) return *bare arrays in input order*,
`explore` keeps every row tied to its id: it returns an
[`ExploreResult`](#ef.exploration.ExploreResult) with `ids` / `coords` / `labels` row-aligned
(plus `cluster_titles`) — the JSON-friendly shape an `app_ef` corpus map
(or [`ef.service.EfService.explore_corpus()`](ef.service.html.md#ef.service.EfService.explore_corpus)) consumes directly.

* **Parameters:**
  * **data** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the corpus to explore — anything `_resolve_explorable()`
    understands (a [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) /
    [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus), a
    [`Corpus`](ef.corpus.html.md#ef.corpus.Corpus) mapping, an iterable of texts or
    [`Segment`](ef.segments.html.md#ef.segments.Segment)s, or a vector matrix).
  * **dims** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – projection target dimensionality — `2` or `3`.
  * **projection_method** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'auto'`, `'umap'`, `'pca'`]) – forwarded to [`project()`](#ef.exploration.project) — `'auto'` /
    `'umap'` / `'pca'`.
  * **cluster_method** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'kmeans'`, `'hdbscan'`]) – forwarded to [`cluster()`](#ef.exploration.cluster) — `'kmeans'` /
    `'hdbscan'`.
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of k-means clusters.
  * **min_cluster_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – smallest cluster HDBSCAN will form.
  * **label** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – when `True`, name each cluster with [`label_clusters()`](#ef.exploration.label_clusters)
    (needs the `ef[imbed]` extra, an LLM key, and text-bearing
    `data`); when `False` (the default) `cluster_titles` is empty.
  * **context** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the corpus topic passed to [`label_clusters()`](#ef.exploration.label_clusters).
  * **n_words** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – maximum cluster-title length, in words.
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – embedder used when `data` is raw text.
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – seed — projection and clustering are reproducible.
* **Return type:**
  [`ExploreResult`](#ef.exploration.ExploreResult)
* **Returns:**
  an [`ExploreResult`](#ef.exploration.ExploreResult) — `ids`, `coords` and `labels`
  row-aligned, and `cluster_titles` (empty unless `label=True`).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the corpus is empty, has fewer than 2 samples, or
      `label=True` is asked of a text-less vector matrix.

```pycon
>>> import numpy as np
>>> vectors = np.random.RandomState(0).rand(12, 16)
>>> result = explore(vectors, projection_method='pca', n_clusters=3, random_state=0)
>>> len(result['ids']), len(result['coords']), len(result['labels'])
(12, 12, 12)
>>> result['ids'][:3]
['0', '1', '2']
>>> len(result['coords'][0])                 # each row has `dims` coordinates
2
>>> set(result['labels']) <= {0, 1, 2}
True
>>> result['cluster_titles']                 # empty unless label=True
{}
```

### ef.exploration.label_clusters(segments, labels, , context=' ', n_words=4, n_samples=None, \*\*labeler_kwargs)

Name each cluster with a short LLM-generated title (use cases §G3).

A thin wrapper over `imbed`’s `imbed.tools.ClusterLabeler`: it
samples each cluster’s segments, asks an LLM for a few-word title, and
returns `{cluster_id: title}`. `ef` does not reimplement this — it
reuses `imbed` (design notes §7).

* **Parameters:**
  * **segments** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the segment texts — plain strings or [`Segment`](ef.segments.html.md#ef.segments.Segment)
    mappings (the `"text"` field is used). Aligned with `labels`.
  * **labels** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the cluster id of each segment — typically the output of
    [`cluster()`](#ef.exploration.cluster).
  * **context** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the corpus’s overall topic, given to the LLM so titles
    describe how a cluster *differs* from the rest, not the shared topic.
  * **n_words** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – maximum title length, in words.
  * **n_samples** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – how many segments to sample per cluster when prompting;
    `None` keeps `ClusterLabeler`’s default.
  * **\*\*labeler_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – forwarded to `imbed.tools.ClusterLabeler`
    (e.g. `max_unique_clusters`, `prompt`).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  `{cluster_id: title}` for every distinct label.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `segments` and `labels` differ in length.
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – if `imbed` (and its `pandas`/`oa` deps) is missing.

```pycon
>>> # needs ef[imbed] + an LLM key — illustrative only:
>>> # label_clusters(['neural net training ...', ...], [0, 1, 0, ...])
>>> # -> {0: 'Neural network training', 1: 'Dataset preprocessing'}
```

### ef.exploration.project(data, , dims=2, method='auto', embedder=None, n_neighbors=15, min_dist=0.1, metric='cosine', pca_components=50, random_state=42)

Project a corpus’s embeddings to `dims` coordinates for visualization.

The canonical “show me the shape of the corpus” operation (use cases §G1).
The default `method='auto'` runs the design-notes §8.4 recipe — \*\*PCA →
UMAP\*\*, `cosine` metric, seeded: PCA first compresses high-dimensional
embeddings to `pca_components` (cheap denoising), then UMAP lays them out
in `dims` dimensions.

* **Parameters:**
  * **data** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – what to project — anything `_resolve_vectors()` understands:
    an `ndarray` (or nested sequence) of vectors, an iterable of
    texts (embedded with `embedder`), a [`Corpus`](ef.corpus.html.md#ef.corpus.Corpus)
    mapping, or a [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus) /
    [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) (vectors pulled from the
    index).
  * **dims** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – target dimensionality — `2` or `3` for plotting.
  * **method** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'auto'`, `'umap'`, `'pca'`]) – `'auto'` (UMAP when available and the corpus is large enough,
    else PCA), `'umap'` (force UMAP — needs the `ef[explore]`
    extra), or `'pca'` (numpy-only, always available).
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – embedder used when `data` is raw text; defaults to the
    dependency-free [`DEFAULT_EMBEDDER`](ef.source_manager.html.md#ef.source_manager.DEFAULT_EMBEDDER).
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – UMAP neighborhood size (clamped to `n_samples - 1`).
  * **min_dist** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – UMAP minimum point separation in the embedding.
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – UMAP distance metric — `'cosine'` for semantic embeddings.
  * **pca_components** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – PCA width for the pre-UMAP compression step; PCA is
    skipped when the source dimensionality is already smaller.
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – seed — projection is reproducible.
* **Return type:**
  `ndarray`
* **Returns:**
  An `ndarray` of shape `(n_samples, dims)`, rows in input order.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if there are fewer than 2 samples.
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – if `method='umap'` and `umap-learn` is not installed.

```pycon
>>> import numpy as np
>>> vectors = np.random.RandomState(1).rand(20, 32)
>>> project(vectors, dims=3, method='pca').shape
(20, 3)
```
