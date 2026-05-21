# ef — Embedding Flow

**A facade for boilerplate-less semantic search, corpus indexing, and
RAG-plug-in readiness.**

`ef` makes the modern embedding pipeline — corpus → segment → embed → vector
store → retrieve — usable with *progressive disclosure*: the light case (a list
of strings, search in one or two lines) and the heavy case (huge corpora, many
segmentations and embedders, varied sources and vector DBs) share **one
facade**. `ef` is *not* a RAG framework — it returns ranked context; you bring
your own LLM.

```python
import ef

index = ef.ingest([
    "The cat sat on the mat",
    "Dogs are loyal companions",
    "Neural networks learn from data",
])

for hit in index.search("loyal dogs", limit=2):
    print(hit.score, hit.segment["text"])
```

That is the whole light path — no configuration, no install beyond `ef` itself.
`ingest` returns a `SearchableCorpus` ready to `search`.

## Installation

```bash
pip install ef                       # core: search, indexing, refresh, eval
pip install "ef[openai]"             # OpenAI embeddings
pip install "ef[sentence-transformers]"  # local sentence-transformers embeddings
pip install "ef[explore]"            # the L5 explore layer (UMAP, HDBSCAN)
pip install "ef[imbed]"              # imbed-backed components & cluster labelling
```

The core install needs only **numpy** (plus `dol`, `i2`, `vd`). The default
embedder is dependency-free — feature hashing, lexical not semantic; for real
semantic search pass a `sentence-transformers` or provider embedder (see below).

## What `ef` is

`ef` is a **facade, not a framework**. It owns the schemas (`Segment`,
`Embedder`, `Segmenter`, `Corpus`), the indexing core, refresh, and the
RAG-plug-in surface — and it stops there: no agent loops, no prompt templating,
no answer synthesis. "Bring your own LLM, your own agent framework, your own UI."

It is built on five layers, the same facade covering all of them:

```
L0 Sources    Corpus = MutableMapping[source_id, Source]   (dol store: fs/S3/API/RAM)
L1 Parse      pluggable text extraction
L2 Segment    Segmenter facade (chunkers)
L3 Embed      Embedder facade (provider / local adapters)
L4 Index      vd.Collection  (ef writes; vd owns the index)
L5 Derive     project / cluster / label   ("explore the corpus")
──────────────────────────────────────────────────────────────
   Search     search(query) -> ranked SearchHits
   RAG plug   retrieve(query) -> list[Segment]  handed to your LLM/agent
```

## Choosing an embedder

`ingest` and `SourceManager` take an `embedder=` — a string, a callable, a URL,
or an `Embedder`. The `as_embedder` seam normalizes all of them:

```python
from ef import as_embedder, openai_embedder, sentence_transformers_embedder

index = ef.ingest(corpus, embedder=sentence_transformers_embedder("all-MiniLM-L6-v2"))
index = ef.ingest(corpus, embedder=openai_embedder("text-embedding-3-small"))
index = ef.ingest(corpus, embedder=as_embedder("cohere:embed-v4.0"))   # also voyage:/gemini:
index = ef.ingest(corpus, embedder=as_embedder(my_callable, model_id="custom@768"))
```

Hosted-API adapters: `openai_embedder` (needs `ef[openai]`) plus
`cohere_embedder`, `voyage_embedder` and `gemini_embedder` — the latter three
speak their providers' REST endpoints directly, so they need only an API key,
no SDK. Each translates `ef`'s canonical `input_type` (`query` / `document` /
`classification` / `clustering`) to the vendor's own task name. Local options:
`sentence_transformers_embedder`, `http_embedder` (any TEI-style service), and
the dependency-free `HashingEmbedder` default.

An `Embedder` is just a batch callable `Iterable[str] -> ndarray(n, dim)` with a
little metadata. Composition wrappers — `CachedEmbedder`, `RetryingEmbedder`,
`MultiEmbedder`, `NormalizingEmbedder` — each wrap an inner embedder.

## The heavy case — `SourceManager`

For large or changing corpora, multiple segmentations/embedders, and explicit
control, use `SourceManager`. Configs that share a pipeline step **share its
artifacts for free** — the indexing core is a content-addressed producer graph,
so a second embedder or segmenter re-uses everything upstream of it.

```python
from ef import SourceManager

manager = SourceManager(corpus, store="my_vectors")
manager.ingest(segmenter="recursive", embedder="openai:text-embedding-3-small")
index = manager.searchable()
```

## Keeping an index fresh

As sources change, an index drifts. `SourceManager` diagnoses and repairs it:

```python
report = manager.diagnose()        # the four staleness conditions
manager.refresh(mode="incremental")  # none | incremental | full | scoped_full
manager = SourceManager(corpus, store="my_vectors", auto_refresh=True)  # live
```

## RAG plug-in & evaluation

`ef` hands a corpus to *your* RAG/agent framework and measures retrieval
quality — it does not synthesize answers.

```python
segments = index.retrieve("how do neural networks learn?", limit=5)
context = "\n\n".join(s["text"] for s in segments)   # feed context to your LLM

from ef import evaluate_retrieval, evaluate_rag
retrieval = evaluate_retrieval(index.retrieve, qrels, queries)  # BEIR-shaped, NDCG@10
rag = evaluate_rag(samples)                          # deterministic lexical metrics
```

`retrieve()` returns plain `Segment`s (provenance preserved in
`metadata["source"]`); `search()` returns scored `SearchHit`s. `with_reranker`
adds a two-stage reranking pass. `as_ragas_dataset` bridges to Ragas for
LLM-judged metrics.

## Exploring a corpus (layer L5)

The secondary "see the shape of the corpus" surface — `ef`'s visualization
heritage, the backend an `app_ef` corpus map consumes. Three functions, each
taking a corpus *or* a vector matrix:

```python
coords = ef.project(index, dims=2)          # PCA -> UMAP, 2-D coordinates
labels = ef.cluster(index, n_clusters=8)    # k-means (or method="hdbscan")
titles = ef.label_clusters(segments, labels)  # LLM-titled clusters (via imbed)
```

`project` and `cluster` import numpy-only; their default paths (PCA, k-means)
need no extra. `method="umap"`, `method="hdbscan"` and `label_clusters` use the
`ef[explore]` / `ef[imbed]` extras, imported lazily.

## What `ef` is *not*

No agent loops, no tool-calling, no conversation memory, no prompt templating,
no **LLM answer synthesis**, no bundled UI, no global config singleton. The
RAG-plug-in surface is the boundary: `ef` returns `retrieve(query) ->
list[Segment]`; the application (or `srag` / `raglab` / LangGraph) takes it from
there.

## Links

- **GitHub**: https://github.com/thorwhalen/ef
- **vd** (vector-store interface): https://github.com/i2mint/vd
- **imbed** (heavy embedding/clustering implementations): https://github.com/thorwhalen/imbed

## License

MIT
