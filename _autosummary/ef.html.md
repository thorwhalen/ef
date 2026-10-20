# ef

ef (Embedding Flow) — a facade for semantic-embeddings user journeys.

`ef` makes the modern semantic-search pipeline — corpus, segmenter, embedder,
vector store, retriever — usable with progressive disclosure: the light case
(a list of strings, search in one or two lines) and the heavy case (huge
corpora, many segmentations/embedders, varied sources and vector DBs) share one
facade. `ef` is *not* RAG — it returns ranked segments; bring your own LLM.

The **embedder facade** is the core surface and is always importable (it costs
only numpy):

- [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) — the structural protocol for a batch
  `Iterable[str] -> ndarray(n, dim)` callable.
- [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder) — the dependency-free default embedder
  (the feature-hashing trick, numpy only); what [`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest)
  resolves to when given no embedder, so the light path needs nothing to install.
- [`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder) — the dependency-injection seam
  (string / callable / URL / existing embedder → `Embedder`).
- Adapters: [`openai_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.openai_embedder),
  [`sentence_transformers_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.sentence_transformers_embedder),
  [`http_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.http_embedder),
  [`cohere_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.cohere_embedder),
  [`voyage_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.voyage_embedder),
  [`gemini_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.gemini_embedder).
- Composition wrappers: [`CachedEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.CachedEmbedder),
  [`RetryingEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.RetryingEmbedder),
  [`MultiEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.MultiEmbedder),
  [`NormalizingEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.NormalizingEmbedder),
  [`InstructionEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.InstructionEmbedder).

The **segmenter facade** is the other always-importable core surface (it needs
no numpy at all):

- [`Segment`](ef.segments.html.md#ef.segments.Segment) — the canonical segment data model (the
  `TypedDict` interchange type; [`SegmentRecord`](ef.segments.html.md#ef.segments.SegmentRecord) is its
  dataclass convenience surface).
- [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) — the structural protocol for a
  `str | Mapping -> Iterable[Segment]` callable.
- [`RecursiveCharacterSegmenter`](ef.segmenters.html.md#ef.segmenters.RecursiveCharacterSegmenter) — the default splitter;
  [`line_segmenter()`](ef.segmenters.html.md#ef.segmenters.line_segmenter) a builtin line splitter.
- [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter) — the dependency-injection seam.
- Composition helpers: [`with_overlap()`](ef.segmenters.html.md#ef.segmenters.with_overlap),
  [`hierarchical()`](ef.segmenters.html.md#ef.segmenters.hierarchical), [`materialise()`](ef.segmenters.html.md#ef.segmenters.materialise).

The **corpus facade** is `ef`’s source layer (L0) — also always importable:

- [`Corpus`](ef.corpus.html.md#ef.corpus.Corpus) / [`Source`](ef.corpus.html.md#ef.corpus.Source) — the type aliases: a
  corpus is just a `MutableMapping[source_id, Source]` (any `dol` store).
- [`as_corpus()`](ef.corpus.html.md#ef.corpus.as_corpus) — the dependency-injection seam (`None` /
  mapping / directory path / iterable of sources → a corpus).
- [`content_hash()`](ef.corpus.html.md#ef.corpus.content_hash) — the content hash of a source.
- [`ChangeDetectingCorpus`](ef.corpus.html.md#ef.corpus.ChangeDetectingCorpus) — a corpus wrapper that detects and
  reports changes ([`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent) / [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff)).

The **artifact graph** is `ef`’s corpus-indexing core (also always importable)
— a content-addressed producer graph in which cascade invalidation and config
branching are the same operation:

- [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec) — the declarative recipe for one
  produced artifact; [`artifact_id()`](ef.artifact_graph.html.md#ef.artifact_graph.artifact_id) is its content hash;
  [`producer_spec()`](ef.artifact_graph.html.md#ef.artifact_graph.producer_spec) builds one ergonomically.
- [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) — the graph: `materialize` (lazy
  backward), `mark_stale` / `delete_cascade` (forward) and `freshness`.

The **search facade** is where the layers come together — corpus → segment →
embed → `vd` → ranked search, with progressive disclosure:

- [`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest) — the one-shot light path: a corpus in, a
  [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus) out, `search(query)` ready.
- [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus) — the thin ready-search object;
  [`search()`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus.search) returns scored
  [`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)s, [`retrieve()`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus.retrieve)
  > returns plain ranked [`Segment`](ef.segments.html.md#ef.segments.Segment)s — the RAG-plug-in shape.
- [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) — the heavy facade: multi-config
  corpus indexing, where configs sharing a step share its artifacts for free.
- [`PipelineSpec`](ef.config.html.md#ef.config.PipelineSpec) / [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec) /
  [`config_id()`](ef.config.html.md#ef.config.config_id) — a pipeline as serializable, content-hashed data.

The **refresh layer** keeps an indexed corpus in sync as its sources change:

- [`SourceManager.diagnose`](ef.source_manager.html.md#ef.source_manager.SourceManager.diagnose) —
  the four staleness conditions, as a [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport).
- [`SourceManager.refresh`](ef.source_manager.html.md#ef.source_manager.SourceManager.refresh) —
  re-sync the index, returning a [`RefreshReport`](ef.refresh.html.md#ef.refresh.RefreshReport); one of four
  [`RefreshMode`](ef.refresh.html.md#ef.refresh.RefreshMode)s. `SourceManager(auto_refresh=True)` keeps
  > the index live as the corpus is edited.

The **RAG-plug-in & evaluation layer** hands a corpus to an external RAG/agent
framework and measures its quality — `ef` returns ranked context, it does not
synthesize answers:

- [`SourceManager.retrieve`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve)
  returns plain ranked [`Segment`](ef.segments.html.md#ef.segments.Segment)s — the RAG-plug-in shape;
  [`hits_to_segments()`](ef.source_manager.html.md#ef.source_manager.hits_to_segments) is the projection from
  [`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)s.
- [`evaluate_retrieval()`](ef.evaluation.html.md#ef.evaluation.evaluate_retrieval) — BEIR/MTEB-shaped retrieval scoring
  (primary metric [NDCG@10](mailto:NDCG@10)); [`read_beir()`](ef.evaluation.html.md#ef.evaluation.read_beir) /
  [`write_beir()`](ef.evaluation.html.md#ef.evaluation.write_beir) move the corpus/queries/qrels triple to disk.
- [`evaluate_rag()`](ef.evaluation.html.md#ef.evaluation.evaluate_rag) — deterministic lexical RAG metrics over
  [`RagSample`](ef.evaluation.html.md#ef.evaluation.RagSample)s; [`as_ragas_dataset()`](ef.evaluation.html.md#ef.evaluation.as_ragas_dataset)
  > bridges to Ragas for the LLM-judged metrics.
- [`with_reranker()`](ef.reranking.html.md#ef.reranking.with_reranker) — a two-stage reranking decorator over any
  retriever; [`cross_encoder_reranker()`](ef.reranking.html.md#ef.reranking.cross_encoder_reranker) is a ready reranker.

The **explore layer** (L5) is `ef`’s visualization heritage, kept as a
*secondary* “see the shape of the corpus” surface — search/RAG/indexing is the
primary one. It imports numpy-only; UMAP, HDBSCAN and `imbed` are lazy extras:

- [`project()`](ef.exploration.html.md#ef.exploration.project) — reduce embeddings to 2-D/3-D coordinates
  (PCA → UMAP); [`cluster()`](ef.exploration.html.md#ef.exploration.cluster) — group them (k-means / HDBSCAN);
  [`label_clusters()`](ef.exploration.html.md#ef.exploration.label_clusters) — name clusters with an LLM via `imbed`.
- [`explore()`](ef.exploration.html.md#ef.exploration.explore) — the orchestrator: project + cluster (+ label)
  in one call, returning a structured, JSON-friendly
  [`ExploreResult`](ef.exploration.html.md#ef.exploration.ExploreResult) (`ids` / `coords` / `labels`).

The **service layer** bridges `ef`’s stateful objects to a stateless
transport — hand its methods to `qh.mk_app()` to serve `ef` over HTTP:

- [`EfService`](ef.service.html.md#ef.service.EfService) — a facade holding a `{corpus_id: SourceManager}`
  handle registry: [`create_corpus()`](ef.service.html.md#ef.service.EfService.create_corpus) indexes a corpus,
  [`search()`](ef.service.html.md#ef.service.EfService.search) / [`retrieve()`](ef.service.html.md#ef.service.EfService.retrieve)
  > query it by id. [`CorpusInfo`](ef.service.html.md#ef.service.CorpusInfo) is its JSON-friendly summary.

Example — wrap a plain function and embed two strings:

```pycon
>>> import numpy as np
>>> from ef import as_embedder
>>> embedder = as_embedder(
...     lambda texts: np.array([[len(t)] for t in texts]), model_id='len@1'
... )
>>> embedder(['hello', 'hi']).ravel().tolist()
[5.0, 2.0]
```

### Functions

| [`ready_handle`](#ef.ready_handle)(array)                                 | Wrap an already-computed array in a finished [`BatchHandle`](#ef.BatchHandle).                                                                  |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cache_key`](#ef.cache_key)(embedder, text[, input_type])             | Deterministic cache key for one `(embedder, text, input_type)` triple.                                                                                                      |
| [`embed_length_sorted`](#ef.embed_length_sorted)(texts, encode, \*[, ...])       | Encode `texts` in length-sorted batches; return vectors in caller order.                                                                                                    |
| [`as_embedder`](#ef.as_embedder)(x, \*\*kwargs)                          | Normalize `x` into an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) — the DI seam.                                                        |
| [`openai_embedder`](#ef.openai_embedder)([model, dim, api_key, ...])         | Build an OpenAI-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).                                                                 |
| [`sentence_transformers_embedder`](#ef.sentence_transformers_embedder)([model_name, ...])   | Build a local sentence-transformers [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).                                                        |
| [`http_embedder`](#ef.http_embedder)(url, \*[, model_id, dim, ...])        | Build an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over a remote HTTP service.                                                        |
| [`cohere_embedder`](#ef.cohere_embedder)([model, dim, api_key, ...])         | Build a Cohere-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).                                                                  |
| [`voyage_embedder`](#ef.voyage_embedder)([model, dim, api_key, ...])         | Build a Voyage AI [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).                                                                          |
| [`gemini_embedder`](#ef.gemini_embedder)([model, dim, api_key, ...])         | Build a Google Gemini [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).                                                                      |
| [`segment_id`](#ef.segment_id)(text[, metadata])                        | Content-derived id for a segment — `sha256` of its normalized content.                                                                                                      |
| [`make_segment`](#ef.make_segment)(text, \*[, id, parent_id, ...])        | Build a [`Segment`](#ef.Segment), deriving a content-based `id` if none given.                                                              |
| [`as_segment`](#ef.as_segment)(x, \*\*defaults)                         | Coerce `x` into a [`Segment`](#ef.Segment) — accepts a `str`, `Mapping` or record.                                                          |
| [`segment_record`](#ef.segment_record)(x)                                   | Coerce `x` (`str` / `Mapping` / [`SegmentRecord`](#ef.SegmentRecord)) into a record.                                                              |
| [`approx_token_count`](#ef.approx_token_count)(text)                            | Rough token count — the common `~4 characters per token` heuristic.                                                                                                         |
| [`line_segmenter`](#ef.line_segmenter)(doc, /)                              | Segment a document into its non-empty lines.                                                                                                                                |
| [`with_overlap`](#ef.with_overlap)(segmenter, n_chars, /, \*[, ...])      | Wrap `segmenter` so consecutive segments overlap by `n_chars` characters.                                                                                                   |
| [`hierarchical`](#ef.hierarchical)(segmenters, /)                         | Compose segmenters into levels — each segments the previous level's output.                                                                                                 |
| [`materialise`](#ef.materialise)(x)                                      | Force a streaming segmenter (or segment stream) to a concrete list.                                                                                                         |
| [`as_segmenter`](#ef.as_segmenter)([x])                                   | Normalize `x` into a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) — the DI seam.                                                      |
| [`imbed_segmenter`](#ef.imbed_segmenter)([name])                             | Build a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) from `imbed`'s segmenter registry.                                               |
| [`content_hash`](#ef.content_hash)(source, \*[, content_keys])            | Content hash of a source document — `sha256` of its normalized content.                                                                                                     |
| [`as_corpus`](#ef.as_corpus)([x])                                      | Normalize `x` into a corpus (`MutableMapping[source_id, Source]`).                                                                                                          |
| [`artifact_id`](#ef.artifact_id)(spec)                                   | Content-addressed id of `spec` — `sha256` over its four fields.                                                                                                             |
| [`producer_spec`](#ef.producer_spec)(op, \*inputs, op_version, \*\*params) | Build a [`ProducerSpec`](#ef.ProducerSpec) ergonomically — inputs positional, params keyword.                                                    |
| [`config_id`](#ef.config_id)(spec)                                     | Content-addressed id of a [`PipelineSpec`](#ef.PipelineSpec) — `sha256` over its steps.                                                          |
| [`full_kwargs`](#ef.full_kwargs)(op[, args, kwargs])                     | Normalize a call into a canonical, fully-named, defaults-filled kwargs dict.                                                                                                |
| [`step_params`](#ef.step_params)(op[, raw_params, n_inputs])             | The canonical *keyword* params of an op step — its inputs dropped.                                                                                                          |
| [`ingest`](#ef.ingest)(sources, \*[, segmenter, embedder, ...])     | Index `sources` and return a ready-to-search [`SearchableCorpus`](#ef.SearchableCorpus) — one call.                                                  |
| [`hits_to_segments`](#ef.hits_to_segments)(hits)                              | Project [`SearchHit`](#ef.SearchHit)s to plain [`Segment`](ef.segments.html.md#ef.segments.Segment)s.                 |
| [`diagnose`](#ef.diagnose)(corpus, collection, config)                | Compute the four staleness conditions of `collection` against `corpus`.                                                                                                     |
| [`plan_refresh`](#ef.plan_refresh)(report, \*[, mode, scope])             | Turn a [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport) into a [`RefreshPlan`](#ef.RefreshPlan). |
| [`refresh_on_change`](#ef.refresh_on_change)(manager)                          | Build a [`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent) handler that auto-refreshes `manager`.                                           |
| [`evaluate_retrieval`](#ef.evaluate_retrieval)(retriever, qrels, queries, \*)   | Score a retriever against a BEIR-shaped evaluation set.                                                                                                                     |
| [`evaluate_rag`](#ef.evaluate_rag)(samples, \*[, metrics])                | Score RAG samples with deterministic, reference-based lexical metrics.                                                                                                      |
| [`read_beir`](#ef.read_beir)(directory, \*[, qrels])                   | Load a BEIR-format dataset from disk into the in-memory evaluation triple.                                                                                                  |
| [`write_beir`](#ef.write_beir)(directory, corpus, queries, qrels)       | Write an evaluation triple to disk in BEIR format — the inverse of [`read_beir()`](#ef.read_beir).                                            |
| [`as_ragas_dataset`](#ef.as_ragas_dataset)(samples)                           | Convert [`RagSample`](#ef.RagSample)s into a Ragas `EvaluationDataset`.                                                                       |
| [`ndcg_at_k`](#ef.ndcg_at_k)(ranked, relevant, k)                      | Normalized DCG at cutoff `k` — `ef`'s **primary** retrieval metric.                                                                                                         |
| [`dcg_at_k`](#ef.dcg_at_k)(ranked, relevant, k)                       | Discounted cumulative gain of `ranked` at cutoff `k`.                                                                                                                       |
| [`recall_at_k`](#ef.recall_at_k)(ranked, relevant, k)                    | Fraction of the relevant docs that appear in the top `k` results.                                                                                                           |
| [`fp_rate_at_k`](#ef.fp_rate_at_k)(ranked, distractors, k)                | Fraction of the *named distractors* that intrude into the top `k`.                                                                                                          |
| [`hard_positive_recall_at_k`](#ef.hard_positive_recall_at_k)(ranked, ...)              | Fraction of the *hard positives* (e.g. thin-description docs) found in the top `k`.                                                                                         |
| [`benchmark_embedders`](#ef.benchmark_embedders)(corpus_sources, ...[, ...])     | Re-index the same corpus under each embedder and tabulate the metrics.                                                                                                      |
| [`precision_at_k`](#ef.precision_at_k)(ranked, relevant, k)                 | Fraction of the top `k` results that are relevant.                                                                                                                          |
| [`reciprocal_rank`](#ef.reciprocal_rank)(ranked, relevant[, k])              | Reciprocal of the rank of the first relevant result (`0` if none).                                                                                                          |
| [`average_precision`](#ef.average_precision)(ranked, relevant[, k])            | Average precision of `ranked` — the per-query term of MAP.                                                                                                                  |
| [`exact_match`](#ef.exact_match)(response, reference)                    | `1.0` if `response` equals `reference` after normalization, else `0.0`.                                                                                                     |
| [`token_f1`](#ef.token_f1)(response, reference)                       | Token-overlap F1 of `response` against `reference` (the SQuAD F1).                                                                                                          |
| [`context_recall`](#ef.context_recall)(reference, retrieved_contexts)       | Lexical recall of the `reference` tokens by the retrieved contexts.                                                                                                         |
| [`context_precision`](#ef.context_precision)(reference, retrieved_contexts)    | Fraction of the retrieved contexts that lexically overlap the `reference`.                                                                                                  |
| [`rerank`](#ef.rerank)(query, segments, reranker, \*[, limit])      | Re-score `segments` against `query` with `reranker` and re-order them.                                                                                                      |
| [`with_reranker`](#ef.with_reranker)(retriever, reranker, \*[, fetch_k])   | Decorate a retrieve-style callable with a two-stage reranking pass.                                                                                                         |
| [`cross_encoder_reranker`](#ef.cross_encoder_reranker)([model_name, device, ...])   | Build a [`Reranker`](#ef.Reranker) backed by a sentence-transformers `CrossEncoder`.                                                         |
| [`project`](#ef.project)(data, \*[, dims, method, embedder, ...])    | Project a corpus's embeddings to `dims` coordinates for visualization.                                                                                                      |
| [`cluster`](#ef.cluster)(data, \*[, method, n_clusters, ...])        | Cluster a corpus's embeddings (use cases §G2).                                                                                                                              |
| [`label_clusters`](#ef.label_clusters)(segments, labels, \*[, ...])         | Name each cluster with a short LLM-generated title (use cases §G3).                                                                                                         |
| [`explore`](#ef.explore)(data, \*[, dims, projection_method, ...])   | Project, cluster and (optionally) label a corpus in one call.                                                                                                               |

### Classes

| [`Embedder`](#ef.Embedder)(\*args, \*\*kwargs)                       | Structural type for an embedder: a batch callable plus metadata.                                                          |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`BaseEmbedder`](#ef.BaseEmbedder)()                                     | Implementation base for embedders (optional — [`Embedder`](#ef.Embedder) is structural).   |
| [`BatchHandle`](#ef.BatchHandle)(poll, result[, cancel])                | A poll/result handle over an in-flight (or already-finished) embed job.                                                   |
| [`FunctionEmbedder`](#ef.FunctionEmbedder)(func, \*[, model_id, dim, ...])   | Promote a bare `texts -> vectors` callable to a full [`Embedder`](#ef.Embedder).           |
| [`HashingEmbedder`](#ef.HashingEmbedder)(\*[, dim, ngram_range, ...])       | A dependency-free embedder — the feature-hashing trick, numpy only.                                                       |
| [`CachedEmbedder`](#ef.CachedEmbedder)(inner, store, \*[, cache_queries])  | Memoize embeddings in a `MutableMapping` store.                                                                           |
| [`RetryingEmbedder`](#ef.RetryingEmbedder)(inner[, policy, sleep])           | Retry an inner embedder's calls on transient failures.                                                                    |
| [`RetryPolicy`](#ef.RetryPolicy)([max_attempts, base_delay, ...])       | Exponential-backoff retry policy for [`RetryingEmbedder`](#ef.RetryingEmbedder).                   |
| [`MultiEmbedder`](#ef.MultiEmbedder)(routes, predicate, \*[, default])    | Route each text to one of several embedders by a predicate.                                                               |
| [`NormalizingEmbedder`](#ef.NormalizingEmbedder)(inner)                         | L2-normalize an inner embedder's output.                                                                                  |
| [`InstructionEmbedder`](#ef.InstructionEmbedder)(inner, \*[, ...])              | Prepend a per-`input_type` instruction before delegating to `inner`.                                                      |
| [`Segment`](#ef.Segment)                                            | A piece of text carved from a source document — the interchange type.                                                     |
| [`SegmentRecord`](#ef.SegmentRecord)(text[, id, parent_id, start, ...])   | Frozen-dataclass view over a [`Segment`](#ef.Segment) — the convenience surface.          |
| [`Segmenter`](#ef.Segmenter)(\*args, \*\*kwargs)                      | Structural type for a segmenter: `str | Mapping -> Iterable[Segment]`.                                                    |
| [`BatchedSegmenter`](#ef.BatchedSegmenter)(\*args, \*\*kwargs)               | Optional trait: a segmenter that can process many documents at once.                                                      |
| [`BaseSegmenter`](#ef.BaseSegmenter)()                                    | Implementation base for segmenters (optional — [`Segmenter`](#ef.Segmenter) is structural). |
| [`RecursiveCharacterSegmenter`](#ef.RecursiveCharacterSegmenter)(\*[, chunk_size, ...]) | The default segmenter — recursive character splitting.                                                                    |
| [`FunctionSegmenter`](#ef.FunctionSegmenter)(func, \*[, pass_doc, ...])       | Promote a bare `text -> pieces` callable to a full [`Segmenter`](#ef.Segmenter).            |
| [`ChangeDetectingCorpus`](#ef.ChangeDetectingCorpus)([corpus, on_change, ...])    | A corpus wrapper that content-hashes values and reports what changed.                                                     |
| [`ChangeEvent`](#ef.ChangeEvent)(source_id, kind[, old_hash, ...])      | A single detected change to a source in a corpus.                                                                         |
| [`CorpusDiff`](#ef.CorpusDiff)([added, modified, deleted, unchanged])  | The result of comparing a corpus's content against known content hashes.                                                  |
| [`ArtifactGraph`](#ef.ArtifactGraph)(\*[, store, producers, edges, ops])  | A content-addressed producer graph — the corpus-indexing & refresh core.                                                  |
| [`ArtifactId`](#ef.ArtifactId)                                         |                                                                                                                           |
| [`OpKey`](#ef.OpKey)                                              |                                                                                                                           |
| [`ProducerSpec`](#ef.ProducerSpec)(op, op_version[, inputs, params])     | A declarative recipe for one produced artifact — the graph's node type.                                                   |
| [`TransformSpec`](#ef.TransformSpec)(op, op_version[, params])            | One step of a pipeline — a serializable recipe for an op application.                                                     |
| [`PipelineSpec`](#ef.PipelineSpec)(segment, embed)                       | A whole indexing pipeline as data — the segment→embed chain.                                                              |
| [`ConfigId`](#ef.ConfigId)                                           |                                                                                                                           |
| [`SearchableCorpus`](#ef.SearchableCorpus)(collection, embedder, \*[, ...])  | A ready-to-search view over one indexed config — the thin facade.                                                         |
| [`SourceManager`](#ef.SourceManager)([corpus, segmenter, embedder, ...])  | The heavy search facade — multi-config corpus indexing over one corpus.                                                   |
| [`SearchHit`](#ef.SearchHit)(segment, score[, source_id])             | One ranked search result — a [`Segment`](ef.segments.html.md#ef.segments.Segment) and its score.  |
| [`StalenessReport`](#ef.StalenessReport)(config[, orphan, missing, ...])    | The four staleness conditions of one config — what [`diagnose()`](#ef.diagnose) returns.   |
| [`IndexedSource`](#ef.IndexedSource)(source_id, doc_ids, ...)             | The provenance a `vd` collection records for one source document.                                                         |
| [`RefreshReport`](#ef.RefreshReport)(config, mode[, added, ...])          | The outcome of a refresh — what [`SourceManager.refresh()`](#ef.SourceManager.refresh) returns.         |
| [`RefreshPlan`](#ef.RefreshPlan)([to_materialize, to_delete])           | What a refresh will do — [`plan_refresh()`](#ef.plan_refresh)'s output.                        |
| [`RetrievalEvalReport`](#ef.RetrievalEvalReport)(metrics, per_query, ...)       | The outcome of [`evaluate_retrieval()`](#ef.evaluate_retrieval) — metrics over a query set.          |
| [`RagEvalReport`](#ef.RagEvalReport)(metrics, coverage, n_samples)        | The outcome of [`evaluate_rag()`](#ef.evaluate_rag) — lexical RAG metrics over a sample set.   |
| [`RagSample`](#ef.RagSample)                                          | One end-to-end RAG interaction — the unit [`evaluate_rag()`](#ef.evaluate_rag) scores.         |
| [`BenchmarkReport`](#ef.BenchmarkReport)(rows, reports)                     | One row per embedder: its `{metric: value}` (plus an `embedder` key).                                                     |
| [`Reranker`](#ef.Reranker)(\*args, \*\*kwargs)                       | A callable that re-scores `(query, segments)` — higher means more relevant.                                               |
| [`ExploreResult`](#ef.ExploreResult)                                      | A structured, JSON-friendly corpus-exploration result.                                                                    |
| [`EfService`](#ef.EfService)(\*[, default_embedder])                  | A registry-backed facade bridging stateful `ef` to a stateless transport.                                                 |
| [`CorpusInfo`](#ef.CorpusInfo)                                         | A JSON-friendly summary of one registered corpus.                                                                         |

### Exceptions

| [`EmbedderError`](#ef.EmbedderError)   | Raised for embedder-specific failures (shape mismatch, batch job failure).   |
|------------------------------------------------------------------|------------------------------------------------------------------------------|

### *class* ef.ArtifactGraph(, store=None, producers=None, edges=None, ops=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A content-addressed producer graph — the corpus-indexing & refresh core.

The graph holds two kinds of node:

- **Leaves** — externally content-addressed inputs (an L0 source, hashed by
  [`ef.corpus.content_hash()`](ef.corpus.html.md#ef.corpus.content_hash)). A leaf has a *value* but no recipe; it is
  > supplied via [`put()`](#ef.ArtifactGraph.put).
- **Produced artifacts** — a node with a [`ProducerSpec`](#ef.ProducerSpec) recipe,
  registered via [`add()`](#ef.ArtifactGraph.add). Its id *is* [`artifact_id()`](#ef.artifact_id) of its recipe.

and four small `MutableMapping` stores, every one injectable so the graph
can be backed by RAM (the default) or by a persistent `dol` store:

- `store` — the content-addressed cache: `artifact_id -> value`.
- `producers` — the recipe registry: `artifact_id -> ProducerSpec`.
- `edges` — the reverse index: `artifact_id -> (dependent ids, ...)`,
  so the *downstream* cone of any node is one lookup away.
- `ops` — the op registry: `op_key -> callable`. The one store that is
  *not* persisted (callables are not serializable) — repopulate it with
  [`register_op()`](#ef.ArtifactGraph.register_op) each process.

The four contract operations:

- [`materialize()`](#ef.ArtifactGraph.materialize) — lazy *backward*: compute a node by recursively
  materializing its inputs, then memoize.
- [`mark_stale()`](#ef.ArtifactGraph.mark_stale) — *forward*: drop the cached values of a node’s
  produced downstream cone, keeping the recipes (they will rebuild).
- [`delete_cascade()`](#ef.ArtifactGraph.delete_cascade) — *forward*: remove a node and its whole downstream
  cone entirely (values, recipes and edges).
- [`freshness()`](#ef.ArtifactGraph.freshness) — query a node’s state (`Freshness`).

plus the reachability queries [`descendants()`](#ef.ArtifactGraph.descendants) (the downstream cone) and
[`ancestors()`](#ef.ArtifactGraph.ancestors) (lineage — what an artifact was produced from).

* **Parameters:**
  * **store** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the content-addressed value cache. Defaults to a fresh `dict`.
  * **producers** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the recipe registry. Defaults to a fresh `dict`. If passed
    already-populated (a resumed/persisted graph) and `edges` is left
    empty, the reverse index is rebuilt from it on construction.
  * **edges** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the reverse-dependency index. Defaults to a fresh `dict`; if
    passed already-populated it is trusted as-is (not rebuilt).
  * **ops** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the op registry. Defaults to a fresh `dict`; [`register_op()`](#ef.ArtifactGraph.register_op)
    adds to it.

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('join', lambda a, b, *, sep: f'{a}{sep}{b}')
>>> x = graph.put('x', 'hello')
>>> y = graph.put('y', 'world')
>>> both = graph.add(producer_spec('join', x, y, op_version='1', sep=', '))
>>> graph.materialize(both)
'hello, world'
>>> sorted(graph.descendants('x'))
[...]
>>> graph.delete_cascade('y')                     # the 'y' source is removed
frozenset(...)
>>> graph.freshness(both)                         # its consumer is gone too
'unknown'
```

#### add(spec)

Register a producer recipe; return its content-addressed id.

Idempotent: the id *is* the recipe, so adding an equal [`ProducerSpec`](#ef.ProducerSpec)
twice (or adding the shared upstream step of two branching configs)
registers one node and returns the same id — the basis of free config
branching. Inputs need not exist yet; the graph is declarative and
[`materialize()`](#ef.ArtifactGraph.materialize) resolves them lazily.

```pycon
>>> graph = ArtifactGraph()
>>> a = graph.add(producer_spec('op', 'leaf', op_version='1'))
>>> a == graph.add(producer_spec('op', 'leaf', op_version='1'))  # idempotent
True
>>> len(graph.producers)
1
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### ancestors(key)

The lineage of `key` — every artifact it was transitively produced from.

Excludes `key` itself. Answers “what produced this?” — provenance /
lineage queries. The leaves of the returned set are the original
sources the artifact ultimately derives from.

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> a = graph.add(producer_spec('id', 'src', op_version='1'))
>>> b = graph.add(producer_spec('id', a, op_version='1'))
>>> graph.ancestors(b) == {'src', a}
True
```

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### delete_cascade(key)

Remove `key` and its whole downstream cone — *forward*; return what was removed.

Deletes every artifact in `{key} ∪ descendants(key)` *entirely* —
cached value, recipe and edges — and detaches the cone from any upstream
inputs that survive it. Unlike [`mark_stale()`](#ef.ArtifactGraph.mark_stale) this removes the
recipes too: use it when an artifact can never be rebuilt, the canonical
case being a **deleted source** (its segments, vectors and index entries
cascade away with it).

Returns the ids actually removed (those that had a value or a recipe).

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> graph.put('s', 'v')
's'
>>> a = graph.add(producer_spec('id', 's', op_version='1'))
>>> graph.delete_cascade('s') == {'s', a}
True
>>> 's' in graph, a in graph
(False, False)
```

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### descendants(key)

The downstream cone of `key` — every artifact transitively produced from it.

Excludes `key` itself. This is the set of artifacts a change to
`key` would invalidate; [`mark_stale()`](#ef.ArtifactGraph.mark_stale) and [`delete_cascade()`](#ef.ArtifactGraph.delete_cascade)
are built on it.

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> a = graph.add(producer_spec('id', 'src', op_version='1'))
>>> b = graph.add(producer_spec('id', a, op_version='1'))
>>> graph.descendants('src') == {a, b}
True
```

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### freshness(key)

Report the state of `key` — `Freshness`.

`"materialized"` if a value is cached; `"stale"` if a recipe exists
but no value (a [`materialize()`](#ef.ArtifactGraph.materialize) will build it); `"unknown"` if the
graph has neither.

Because the graph is content-addressed, a *cached* value is always
correct *for its id* — staleness here means “not yet computed”, never
“computed wrong”. The richer staleness conditions (orphan / missing /
misconfigured, computed against `vd` index metadata) belong to a later
diagnostics phase, not to the graph core.

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> graph.put('s', 'v')
's'
>>> a = graph.add(producer_spec('id', 's', op_version='1'))
>>> graph.freshness('s'), graph.freshness(a), graph.freshness('?')
('materialized', 'stale', 'unknown')
```

* **Return type:**
  [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'materialized'`, `'stale'`, `'unknown'`]

#### mark_stale(key)

Invalidate `key`’s produced downstream cone — *forward*; return that cone.

Drops the **cached values** of every produced artifact in
`{key} ∪ descendants(key)`, *keeping the recipes* — so the next
[`materialize()`](#ef.ArtifactGraph.materialize) recomputes them. Leaf nodes (inputs, no recipe) are
never touched: they cannot be recomputed, so dropping their value would
be data loss — use [`delete_cascade()`](#ef.ArtifactGraph.delete_cascade) to remove an input.

This is the invalidation half of a refresh: when an op’s implementation
changes, or to force a recompute. (A *source* change is usually a
[`delete_cascade()`](#ef.ArtifactGraph.delete_cascade) of the old content hash — the new content yields
a new leaf id.) Returns the produced artifacts that are now stale.

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> graph.put('s', 'v')
's'
>>> a = graph.add(producer_spec('id', 's', op_version='1'))
>>> _ = graph.materialize(a)
>>> graph.mark_stale('s') == {a}        # the leaf 's' itself is not "stale"
True
>>> graph.freshness(a)
'stale'
```

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### materialize(key)

Compute (or fetch the cached value of) the artifact `key` — lazy *backward*.

If `key` is already cached in `store` its value is returned
directly. Otherwise its recipe’s inputs are materialized recursively
(depth-first), the op is called `op(*input_values, **params)`, and the
result is memoized in `store` before being returned — so a second
[`materialize()`](#ef.ArtifactGraph.materialize) is a cache hit, and a shared upstream artifact is
computed exactly once however many configs depend on it.

* **Raises:**
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `key` is neither cached nor has a producer recipe
        (an unresolved leaf — supply it with [`put()`](#ef.ArtifactGraph.put)).
  * [**LookupError**](https://docs.python.org/3/builtins/exceptions.html#LookupError) – if a recipe’s `op` is not in the `ops` registry.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the producer graph contains a cycle (a content-
        addressed graph cannot — this guards a hand-corrupted registry).
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('double', lambda x: x * 2)
>>> graph.put('n', 21)
'n'
>>> out = graph.add(producer_spec('double', 'n', op_version='1'))
>>> graph.materialize(out)
42
```

#### put(key, value)

Store a **leaf** `value` (an external input) under `key`; return `key`.

A leaf has a value but no recipe — typically an L0 source keyed by its
[`ef.corpus.content_hash()`](ef.corpus.html.md#ef.corpus.content_hash). Returns `key` so the call composes:

```default
src = graph.put(content_hash(text), text)
```

```pycon
>>> graph = ArtifactGraph()
>>> graph.put('doc-1', 'some text')
'doc-1'
>>> graph.freshness('doc-1')
'materialized'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### register_op(name, fn)

Register the callable `fn` under the op key `name`; return `fn`.

[`materialize()`](#ef.ArtifactGraph.materialize) resolves a [`ProducerSpec`](#ef.ProducerSpec)’s string `op`
through this registry. The op is called `fn(*input_values, **params)`:
the materialized inputs positionally, the spec’s `params` as keywords.

```pycon
>>> graph = ArtifactGraph()
>>> graph.register_op('upper', str.upper)
<method 'upper' of 'str' objects>
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### ef.ArtifactId

alias of [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* ef.BaseEmbedder

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Implementation base for embedders (optional — [`Embedder`](#ef.Embedder) is structural).

Subclasses set `model_id` / `dim` / `normalized` /
`honored_input_types` and implement `__call__`; they then satisfy
[`Embedder`](#ef.Embedder) structurally. In return they get:

- a default *synchronous* [`embed_batch()`](#ef.BaseEmbedder.embed_batch) (`ready_handle(self(...))`) —
  override it only for a genuinely async provider Batch API;
- a readable `repr`.

#### embed_batch(texts, , input_type=None, \*\*backend)

Default: embed synchronously and return a finished handle.

* **Return type:**
  [`BatchHandle`](ef.embedders.html.md#ef.embedders.BatchHandle)

### *class* ef.BaseSegmenter

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Implementation base for segmenters (optional — [`Segmenter`](#ef.Segmenter) is structural).

A subclass implements `__call__`; in return it gets a default
[`batch()`](#ef.BaseSegmenter.batch) (one-document-at-a-time — override it for a genuinely batched
backend) and a readable `repr`. Subclassing is purely a convenience;
satisfying [`Segmenter`](#ef.Segmenter) structurally is what matters.

#### batch(docs,)

Default: segment each document independently.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]

### *class* ef.BatchHandle(poll, result, cancel=<function BatchHandle.<lambda>>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A poll/result handle over an in-flight (or already-finished) embed job.

Batch-and-poll is just a special case of async. A synchronous backend
returns a *ready* handle (via [`ready_handle()`](#ef.ready_handle)); a provider Batch API
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

### *class* ef.BatchedSegmenter(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Optional trait: a segmenter that can process many documents at once.

Plain [`Segmenter`](#ef.Segmenter) handles one document per call. A segmenter that can
amortize per-call cost across documents (a shared model load, a vectorized
pass) also offers `batch`. [`BaseSegmenter`](#ef.BaseSegmenter) supplies a correct — if
un-optimized — default, so every [`BaseSegmenter`](#ef.BaseSegmenter) subclass is already
a [`BatchedSegmenter`](#ef.BatchedSegmenter).

#### batch(docs,)

Segment each document in `docs`, yielding one segment-stream each.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]

### *class* ef.BenchmarkReport(rows, reports)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One row per embedder: its `{metric: value}` (plus an `embedder` key).

`reports` keeps each full [`RetrievalEvalReport`](#ef.RetrievalEvalReport) for drill-down
(e.g. `distractor_hits`). `to_frame()` gives a pandas DataFrame
(pandas is imported lazily).

```pycon
>>> r = BenchmarkReport(rows=[{'embedder': 'a', 'ndcg@10': 0.5}], reports={})
>>> r.to_records()
[{'embedder': 'a', 'ndcg@10': 0.5}]
```

### *class* ef.CachedEmbedder(inner, store, , cache_queries=False)

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
  * **store** ([`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `ndarray`]) – A `MutableMapping[str, ndarray]` keyed by [`cache_key()`](#ef.cache_key).
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

### *class* ef.ChangeDetectingCorpus(corpus=None, , on_change=None, hashes=None, content_keys=None, fingerprint=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

A corpus wrapper that content-hashes values and reports what changed.

Wraps any inner corpus (a `MutableMapping[source_id, Source]`) and adds
change detection *without* altering its mapping behavior — reads, writes and
deletes pass straight through to the inner store. It is to a corpus what
[`CachedEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.CachedEmbedder) is to an embedder: a
transparent, composable layer.

Changes are noticed two ways:

- **Through the wrapper.** A `corpus[id] = source` or `del corpus[id]`
  that actually changes content fires the injected `on_change` callback
  with a [`ChangeEvent`](#ef.ChangeEvent). Writing *identical* content is a no-op (no
  event) — the basis of idempotent ingestion.
- **Out of band.** When the backing store is edited directly (a file
  changed on disk, an S3 object replaced), [`scan()`](#ef.ChangeDetectingCorpus.scan) re-reads every
  source, compares hashes, fires events for what differs and returns a
  [`CorpusDiff`](#ef.CorpusDiff). [`diff()`](#ef.ChangeDetectingCorpus.diff) runs the same comparison with no side
  > effects.

The `on_change` callback is the seam the downstream `ArtifactGraph`
plugs into: a `"modified"` / `"deleted"` event is exactly the trigger
for a cascade invalidation.

On construction a *baseline* is established: the current content hashes of
the inner corpus are recorded silently (no events). Pass an already-populated
`hashes` registry to instead resume from a persisted baseline — then call
[`scan()`](#ef.ChangeDetectingCorpus.scan) to reconcile any drift.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the inner corpus to wrap. Coerced via [`as_corpus()`](#ef.as_corpus), so a
    directory path / `dict` / iterable of sources / `None` all work.
  * **on_change** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – called with each [`ChangeEvent`](#ef.ChangeEvent). Defaults to a no-op.
    It should be cheap and not raise; an exception propagates to the
    caller *after* the corpus and registry are already consistent.
  * **hashes** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the hash registry — a `MutableMapping[source_id, str]` of the
    last-known content hash per source. Injectable so it can be
    persisted (e.g. a `dol` store); defaults to an in-RAM `dict`.
    If passed already-populated it is trusted as the baseline.
  * **content_keys** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – forwarded to [`content_hash()`](#ef.content_hash) — metadata keys that
    participate in the hash.
  * **fingerprint** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Hashable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional cheap `source_id -> Hashable` callback (an
    mtime / etag prefilter). When given, [`scan()`](#ef.ChangeDetectingCorpus.scan) / [`diff()`](#ef.ChangeDetectingCorpus.diff)
    skip re-hashing a source whose fingerprint is unchanged. The
    content hash always remains the source of truth — the fingerprint
    is only a speed optimization, never trusted to *declare* a change.

```pycon
>>> backing = {'a': 'one', 'b': 'two'}
>>> corpus = ChangeDetectingCorpus(backing)   # baseline adopts 'a' and 'b'
>>> backing['a'] = 'ONE'                      # edited behind the wrapper's back
>>> backing['c'] = 'three'
>>> del backing['b']
>>> d = corpus.scan()
>>> (d.added, d.modified, d.deleted)
(('c',), ('a',), ('b',))
```

#### diff()

Compare current corpus content against the registry — **no side effects**.

Detects out-of-band edits (changes made to the backing store directly)
without firing events or updating the registry. [`scan()`](#ef.ChangeDetectingCorpus.scan) is the
effectful counterpart.

* **Return type:**
  [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff)

#### scan()

Reconcile the registry with the corpus, fire events, return the diff.

Use this to pick up **out-of-band** edits — changes made to the backing
store directly rather than through this wrapper. Every detected change
fires a [`ChangeEvent`](#ef.ChangeEvent) and the registry (and fingerprints) are
advanced to the new state, so a second [`scan()`](#ef.ChangeDetectingCorpus.scan) with nothing
further changed is a no-op. [`diff()`](#ef.ChangeDetectingCorpus.diff) is the side-effect-free version.

* **Return type:**
  [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff)

### *class* ef.ChangeEvent(source_id, kind, old_hash=None, new_hash=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A single detected change to a source in a corpus.

Emitted by [`ChangeDetectingCorpus`](#ef.ChangeDetectingCorpus) to its `on_change` callback.
`old_hash` is `None` for an `"added"` event, `new_hash` is `None`
for a `"deleted"` one.

#### source_id

the key of the source that changed.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### kind

`"added"`, `"modified"` or `"deleted"`.

* **Type:**
  [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'added'`, `'modified'`, `'deleted'`]

#### old_hash

the source’s previous content hash, if any.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### new_hash

the source’s new content hash, if any.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ef.ConfigId

alias of [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* ef.CorpusDiff(added=(), modified=(), deleted=(), unchanged=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of comparing a corpus’s content against known content hashes.

The four fields partition the corpus’s source ids; each is a sorted tuple,
so a [`CorpusDiff`](#ef.CorpusDiff) is deterministic and JSON-friendly. Truthiness
reports whether *anything* changed.

```pycon
>>> d = CorpusDiff(added=('b',), modified=('a',))
>>> bool(d), d.changed
(True, ('b', 'a'))
>>> bool(CorpusDiff(unchanged=('a', 'b')))
False
```

#### *property* changed *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

The source ids that are new or modified — the set to (re-)materialize.

### *class* ef.CorpusInfo

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

A JSON-friendly summary of one registered corpus.

The return shape shared by [`EfService.create_corpus()`](#ef.EfService.create_corpus),
[`EfService.corpus_info()`](#ef.EfService.corpus_info) and (as a list) [`EfService.list_corpora()`](#ef.EfService.list_corpora)
— a plain `dict` an HTTP client and `qh`’s schema both understand.

Keys:
: corpus_id: the registry handle — how every other method addresses it.
  n_sources: the number of source documents in the corpus.
  n_segments: the number of indexed segments (sources cut by the segmenter).
  embedder: the embedder’s `model_id` (e.g. `"hashing:v1@512"`).
  dim: the embedding dimensionality.
  config_id: the content hash of the segmenter+embedder pipeline.

### *class* ef.EfService(, default_embedder=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A registry-backed facade bridging stateful `ef` to a stateless transport.

Construct one `EfService` per process, register corpora with
[`create_corpus()`](#ef.EfService.create_corpus), then query them by `corpus_id`. Hand the seven
bound methods to `qh.mk_app()` to expose them over HTTP — see the module
docstring.

The handle registry is private instance state — never a module global.
Corpora are isolated: each [`create_corpus()`](#ef.EfService.create_corpus) builds its own
[`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) over its own in-memory `vd`
backend, so one corpus’s vectors never leak into another’s search.

`default_embedder` sets the embedder [`create_corpus()`](#ef.EfService.create_corpus) resolves when
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

Return the [`CorpusInfo`](#ef.CorpusInfo) of a registered corpus.

* **Raises:**
  [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `corpus_id` is not registered.
* **Return type:**
  [`CorpusInfo`](ef.service.html.md#ef.service.CorpusInfo)

#### create_corpus(sources, , embedder=None, segmenter=None, corpus_id=None, embedder_api_key=None)

Index `sources` into a new corpus, register it, return its [`CorpusInfo`](#ef.CorpusInfo).

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
    so [`search()`](#ef.EfService.search) / [`retrieve()`](#ef.EfService.retrieve) / [`explore_corpus()`](#ef.EfService.explore_corpus)
    need no key of their own.
* **Return type:**
  [`CorpusInfo`](ef.service.html.md#ef.service.CorpusInfo)
* **Returns:**
  the [`CorpusInfo`](#ef.CorpusInfo) of the freshly indexed corpus.
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

Return the [`CorpusInfo`](#ef.CorpusInfo) of every registered corpus.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`CorpusInfo`](ef.service.html.md#ef.service.CorpusInfo)]

#### retrieve(corpus_id, query, , limit=10)

Retrieve ranked [`Segment`](ef.segments.html.md#ef.segments.Segment)s — the RAG-plug-in shape.

Like [`search()`](#ef.EfService.search), but returns plain segments in rank order (the
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

### *class* ef.Embedder(\*args, \*\*kwargs)

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

The `InputType` values this embedder acts on
(empty if it ignores the task hint).

* **Type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'query'`, `'document'`, `'classification'`, `'clustering'`], [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

#### embed_batch(texts, , input_type=None, \*\*backend)

Submit an embed job, returning a [`BatchHandle`](#ef.BatchHandle).

* **Return type:**
  [`BatchHandle`](ef.embedders.html.md#ef.embedders.BatchHandle)

### *exception* ef.EmbedderError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Raised for embedder-specific failures (shape mismatch, batch job failure).

### *class* ef.ExploreResult

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

A structured, JSON-friendly corpus-exploration result.

What [`explore()`](#ef.explore) returns and [`ef.service.EfService.explore_corpus()`](ef.service.html.md#ef.service.EfService.explore_corpus)
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
  cluster_titles: `{cluster_id: title}` — empty unless [`explore()`](#ef.explore)
  <br/>
  > was called with `label=True`.

### *class* ef.FunctionEmbedder(func, , model_id=None, dim=None, normalized=False, honored_input_types=(), pass_backend=True)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

Promote a bare `texts -> vectors` callable to a full [`Embedder`](#ef.Embedder).

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

### *class* ef.FunctionSegmenter(func, \*, pass_doc=False, count_tokens=<function approx_token_count>, tokenizer='char-approx')

Bases: [`BaseSegmenter`](ef.segmenters.html.md#ef.segmenters.BaseSegmenter)

Promote a bare `text -> pieces` callable to a full [`Segmenter`](#ef.Segmenter).

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

### *class* ef.HashingEmbedder(, dim=512, ngram_range=(1, 2), sublinear_tf=True)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

A dependency-free embedder — the feature-hashing trick, numpy only.

[`HashingEmbedder`](#ef.HashingEmbedder) is `ef`’s **zero-install default**: the embedder
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
[`HashingEmbedder`](#ef.HashingEmbedder)’s artifacts stay correctly content-addressed in the
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

### *class* ef.IndexedSource(source_id, doc_ids, source_hashes, config_hashes)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The provenance a `vd` collection records for one source document.

A source is segmented into several documents; they all carry the same
`source_id` and *should* carry one `source_hash` / `config_hash` each.
`indexed_state()` groups a collection’s documents into one of these per
source — collecting the hashes as **sets** so a partially re-indexed source
(documents disagreeing on a hash) is detectable rather than papered over.

#### source_id

the corpus key of the source these documents were cut from.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### doc_ids

the ids of every `vd` document belonging to the source.

* **Type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

#### source_hashes

the distinct `source_hash` values its documents record
— a single-element set for a cleanly indexed source.

* **Type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### config_hashes

likewise the distinct `config_hash` values.

* **Type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* ef.InstructionEmbedder(inner, , query_instruction=None, document_instruction=None, classification_instruction=None, clustering_instruction=None)

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
  so [`CachedEmbedder`](#ef.CachedEmbedder) and content-addressed artifacts never mix vectors
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

### *class* ef.MultiEmbedder(routes, predicate, , default=None)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

Route each text to one of several embedders by a predicate.

Useful for domain-specific models (e.g. a code embedder for code, a prose
embedder for prose). All routes must share the same `dim` — the output is
a single `(n, dim)` array, with each row produced by its text’s route.

* **Parameters:**
  * **routes** ([`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`Hashable`](https://docs.python.org/3/library/typing.html#typing.Hashable), [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)]) – `{route_key: Embedder}`.
  * **predicate** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Hashable`](https://docs.python.org/3/library/typing.html#typing.Hashable)]) – `text -> route_key` — picks a route per text.
  * **default** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Embedder for texts whose `route_key` is not in `routes`
    (otherwise an unknown key raises [`EmbedderError`](#ef.EmbedderError)).

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

### *class* ef.NormalizingEmbedder(inner)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

L2-normalize an inner embedder’s output.

Use this for MRL-truncated vectors (which lose unit norm when sliced to a
smaller dimension) or any backend that does not normalize by construction.
Reports `normalized=True`; [`cache_key()`](#ef.cache_key) keys on `normalized`, so a
normalized and an un-normalized view of the same model never collide in a
shared cache.

* **Parameters:**
  **inner** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) – The embedder whose output to renormalize.

#### *property* honored_input_types *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['query', 'document', 'classification', 'clustering'], ...]*

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable’s items.

If the argument is a tuple, the return value is the same object.

### ef.OpKey

alias of [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* ef.PipelineSpec(segment, embed)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A whole indexing pipeline as data — the segment→embed chain.

A [`PipelineSpec`](#ef.PipelineSpec) is the declarative description of one *config*: a
[`TransformSpec`](#ef.TransformSpec) for the segmenter and one for the embedder. It is
fully serializable ([`as_dict()`](#ef.PipelineSpec.as_dict) / [`from_dict()`](#ef.PipelineSpec.from_dict)) and content-hashed
by [`config_id()`](#ef.config_id). Registering a second [`PipelineSpec`](#ef.PipelineSpec) that
differs only in its embed step shares the segment artifacts in the
[`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) — that is config branching, and
the graph gives it for free.

The `segment``→``embed` ordering *is* the input wiring: the embed step
consumes the segment step’s output. (Phase 5 fixes this two-step shape; a
later phase may generalize it to an arbitrary step list.)

```pycon
>>> seg = TransformSpec(op='segment:r', op_version='1', params={})
>>> emb = TransformSpec(op='embed:m@8', op_version='1', params={'input_type': 'document'})
>>> spec = PipelineSpec(segment=seg, embed=emb)
>>> spec.embed.params['input_type']
'document'
>>> PipelineSpec.from_dict(spec.as_dict()) == spec
True
```

#### as_dict()

Return a plain-`dict` (JSON-serializable) form of the pipeline.

```pycon
>>> seg = TransformSpec(op='segment:r', op_version='1', params={})
>>> emb = TransformSpec(op='embed:m', op_version='1', params={})
>>> PipelineSpec(segment=seg, embed=emb).as_dict()['segment']['op']
'segment:r'
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *classmethod* from_dict(d)

Rebuild a [`PipelineSpec`](#ef.PipelineSpec) from its [`as_dict()`](#ef.PipelineSpec.as_dict) form.

* **Return type:**
  [`PipelineSpec`](ef.config.html.md#ef.config.PipelineSpec)

### *class* ef.ProducerSpec(op, op_version, inputs=(), params=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A declarative recipe for one produced artifact — the graph’s node type.

A [`ProducerSpec`](#ef.ProducerSpec) says: *this artifact is produced by op* `op` \*at
version\* `op_version`, *applied to the artifacts* `inputs`  *(passed
positionally), with the keyword parameters* `params`. It does **not**
hold the producer callable — only the string `op` key — so a spec is
fully serializable ([`as_dict()`](#ef.ProducerSpec.as_dict) / [`from_dict()`](#ef.ProducerSpec.from_dict)) and the graph’s
`producers` registry can be persisted.

The recipe is the artifact’s *identity*: [`artifact_id()`](#ef.artifact_id) hashes exactly
these four fields. Treat an instance as immutable — it is frozen, and
mutating `params` in place would silently desynchronize it from its id.

#### op

the op key, resolved to a callable via the graph’s `ops` registry.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### op_version

the op’s version string. It participates in the id, so a
behavior change to an op **must** be paired with a version bump —
reusing a version for changed behavior silently serves stale cached
values (the graph trusts the version; it cannot inspect the code).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### inputs

the input artifact ids, in the order the op receives them
positionally. Order is significant — it is part of the id.

* **Type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

#### params

the op’s keyword arguments. Key order is *not* significant
(the id is computed over canonical, sorted-key JSON); pass
defaults-filled, fully-named params so that two semantically equal
calls hash equal.

* **Type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> spec = producer_spec('embed', 'seg-1', op_version='3', model='small')
>>> spec.op, spec.inputs, spec.op_version
('embed', ('seg-1',), '3')
>>> spec.params == {'model': 'small'}
True
```

#### as_dict()

Return a plain-`dict` form of the spec — JSON-serializable.

The inverse of [`from_dict()`](#ef.ProducerSpec.from_dict). Use it to persist a `producers`
registry to a `dol` JSON store.

```pycon
>>> producer_spec('embed', 'a', 'b', op_version='1', dim=256).as_dict()
{'op': 'embed', 'op_version': '1', 'inputs': ['a', 'b'], 'params': {'dim': 256}}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *classmethod* from_dict(d)

Rebuild a [`ProducerSpec`](#ef.ProducerSpec) from its [`as_dict()`](#ef.ProducerSpec.as_dict) form.

```pycon
>>> spec = producer_spec('embed', 'a', 'b', op_version='1', dim=256)
>>> ProducerSpec.from_dict(spec.as_dict()) == spec
True
```

* **Return type:**
  [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec)

### *class* ef.RagEvalReport(metrics, coverage, n_samples)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`evaluate_rag()`](#ef.evaluate_rag) — lexical RAG metrics over a sample set.

`metrics` maps each metric to its mean over the samples it applied to;
`coverage` records how many samples that was (a metric needing a
`reference` skips samples without one). `n_samples` is the total scored.

```pycon
>>> report = RagEvalReport(
...     metrics={'exact_match': 0.5}, coverage={'exact_match': 4}, n_samples=4)
>>> report.metrics['exact_match']
0.5
```

### *class* ef.RagSample

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

One end-to-end RAG interaction — the unit [`evaluate_rag()`](#ef.evaluate_rag) scores.

The field names match Ragas’ `SingleTurnSample` exactly, so a
[`RagSample`](#ef.RagSample) is the interchange shape `app_ef` / `srag` / `raglab`
produce and [`as_ragas_dataset()`](#ef.as_ragas_dataset) forwards unchanged. `ef` itself fills
only `retrieved_contexts` (from [`retrieve()`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve));
`response` is the *caller’s* LLM output — `ef` does not synthesize it.

Keys:
: user_input: the user’s question.
  response: the generated answer (the caller’s LLM produced it).
  retrieved_contexts: the context passages handed to the LLM — what
  <br/>
  > [`retrieve()`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve) returned, as text.
  <br/>
  reference: the ground-truth answer, if known.
  reference_contexts: the ground-truth context passages, if known.

### *class* ef.RecursiveCharacterSegmenter(\*, chunk_size=512, chunk_overlap=64, separators=('\\n\\n', '\\n', '. ', ', ', ' ', ''), count_tokens=<function approx_token_count>, tokenizer='char-approx')

Bases: [`BaseSegmenter`](ef.segmenters.html.md#ef.segmenters.BaseSegmenter)

The default segmenter — recursive character splitting.

Splits a document on a descending ladder of separators (paragraph → line →
sentence → clause → word → character), merging the resulting pieces into
chunks of about `chunk_size` tokens with `chunk_overlap` tokens shared
between neighbours. This “good enough” default is, on realistic corpora,
within a point or two of far more elaborate methods — upgrade only if you
can measure the win.

Sizes are measured with `count_tokens`. The default counter is the
dependency-free [`approx_token_count()`](#ef.approx_token_count); inject a real tokenizer (and its
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

### *class* ef.RefreshPlan(to_materialize=(), to_delete=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What a refresh will do — [`plan_refresh()`](#ef.plan_refresh)’s output.

The two fields are disjoint in *intent* but may overlap in membership: a
stale source is in **both** — its old documents are deleted *and* it is
re-materialized. An orphan is only in `to_delete`; a brand-new source only
in `to_materialize`.

#### to_materialize

source ids to (re-)run through the pipeline.

* **Type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

#### to_delete

source ids whose currently-indexed documents are removed
(before re-materializing, for the ones that also reappear above).

* **Type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

### *class* ef.RefreshReport(config, mode, added=(), modified=(), deleted=(), unchanged=(), documents_written=0, documents_removed=0, artifacts_removed=0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of a refresh — what [`SourceManager.refresh()`](#ef.SourceManager.refresh) returns.

`added` / `modified` / `deleted` / `unchanged` partition the
source ids the refresh considered (mirroring [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff)
field names): newly indexed, re-indexed (was stale/misconfigured), removed
(orphan), and already-fresh-left-alone. The three counts report the work
done. Truthiness reports whether anything changed.

```pycon
>>> r = RefreshReport(config='c0', mode='full', added=('a',), modified=('b',))
>>> bool(r), r.changed
(True, ('a', 'b'))
>>> bool(RefreshReport(config='c0', mode='full', unchanged=('a',)))
False
```

#### *property* changed *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

The source ids that were added, modified or deleted.

### *class* ef.Reranker(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A callable that re-scores `(query, segments)` — higher means more relevant.

The structural contract a reranker satisfies: given the query and a list of
candidate [`Segment`](ef.segments.html.md#ef.segments.Segment)s, return \*\*one score per
segment\*\*, in the same order. [`rerank()`](#ef.rerank) sorts by these scores. It is a
`@runtime_checkable Protocol`, not a base class — a plain function of the
right shape already *is* a [`Reranker`](#ef.Reranker).

```pycon
>>> def by_length(query, segments):
...     return [len(s['text']) for s in segments]
>>> isinstance(by_length, Reranker)
True
```

### *class* ef.RetrievalEvalReport(metrics, per_query, n_queries, k_values, distractor_hits=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`evaluate_retrieval()`](#ef.evaluate_retrieval) — metrics over a query set.

`metrics` maps a `"<name>@<k>"` key (e.g. `"ndcg@10"`) to the mean of
that metric over every scored query; `per_query` keeps the same keys per
query for drill-down. `n_queries` is how many queries were actually
scored (a query with no positive judgement is skipped).

```pycon
>>> report = RetrievalEvalReport(
...     metrics={'ndcg@10': 0.83}, per_query={}, n_queries=12, k_values=(10,))
>>> report.primary
0.83
```

#### distractor_hits *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [int](https://docs.python.org/3/builtins/functions.html#int)]]*

`{query_id: {distractor_id: rank}}` — the named distractors that landed in
the top `max(k_values)` (1-based rank); empty unless `distractors` was given.

#### *property* primary *: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The headline number — mean [NDCG@10](mailto:NDCG@10), or `None` if it was not computed.

#### primary_at(k)

Mean NDCG at cutoff `k` — `None` if NDCG@\`\`k\`\` was not computed.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* ef.RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0, jitter=0.3, retry_on=<function \_default_is_retryable>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Exponential-backoff retry policy for [`RetryingEmbedder`](#ef.RetryingEmbedder).

#### max_attempts

Total tries (the first call counts as attempt 1).

* **Type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### base_delay

Seconds before the first retry; doubles each attempt.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### max_delay

Upper bound on any single backoff sleep.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### jitter

Random fraction (`0..1`) of the delay added on top, to
de-correlate concurrent clients.

* **Type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### retry_on

Predicate deciding whether an exception is worth retrying.

* **Type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`BaseException`](https://docs.python.org/3/builtins/exceptions.html#BaseException)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]
* **Return type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

#### retry_on()

Heuristic: retry transient failures (429, 5xx, connection/timeout).

Per-request 400-class errors are *not* retried — the same request would fail
identically. Recognizes the `status_code`/`status` attribute carried by
most HTTP-client SDK exceptions (incl. the OpenAI SDK).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### *class* ef.RetryingEmbedder(inner, policy=None, \*, sleep=<built-in function sleep>)

Bases: [`BaseEmbedder`](ef.embedders.html.md#ef.embedders.BaseEmbedder)

Retry an inner embedder’s calls on transient failures.

* **Parameters:**
  * **inner** ([`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)) – The embedder to guard.
  * **policy** ([`RetryPolicy`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.RetryPolicy) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A [`RetryPolicy`](#ef.RetryPolicy) (sensible defaults if omitted).
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

### *class* ef.SearchHit(segment, score, source_id=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One ranked search result — a [`Segment`](ef.segments.html.md#ef.segments.Segment) and its score.

[`SearchableCorpus.search()`](#ef.SearchableCorpus.search) / [`SourceManager.search()`](#ef.SourceManager.search) return a
ranked `list` of these. Keeping [`Segment`](ef.segments.html.md#ef.segments.Segment) (the
interchange `TypedDict`) pure of a result-only `score` key, a
[`SearchHit`](#ef.SearchHit) *wraps* it alongside the similarity `score` and the
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

### *class* ef.SearchableCorpus(collection, embedder, , config=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A ready-to-search view over one indexed config — the thin facade.

What [`ingest()`](#ef.ingest) returns and what [`SourceManager.searchable()`](#ef.SourceManager.searchable) hands
out: a small, read-mostly object that knows just enough to answer a query —
the `vd` collection holding the vectors and the [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
that embeds the query. It does **not** re-index; mutation, refresh and
multi-config live on [`SourceManager`](#ef.SourceManager).

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
[`search()`](#ef.SearchableCorpus.search) returns scored [`SearchHit`](#ef.SearchHit)s (for inspecting the
ranking), `retrieve` returns **plain segments** in rank order — clean
context to hand to an LLM, with no `ef`-specific type to learn (design
notes §F5). `ef` returns the context; it does *not* synthesize answers.
See [`hits_to_segments()`](#ef.hits_to_segments) for how the score is dropped and the
`source_id` provenance is preserved in `metadata["source"]`.

```pycon
>>> # idx.retrieve('a query') -> [{'text': ..., 'id': ..., 'metadata': ...}, ...]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]

#### search(query, , limit=10, filter=None)

Search the indexed corpus — return up to `limit` ranked [`SearchHit`](#ef.SearchHit)s.

`query` is text (embedded with `input_type="query"`) or an already
computed query vector. `filter` is a `vd` metadata filter
(MongoDB-style) — see `vd.filters`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)]

### *class* ef.Segment

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

A piece of text carved from a source document — the interchange type.

A `TypedDict`: a [`Segment`](#ef.Segment) *is* a plain `dict`, which keeps it
cheap to create and stream. `total=False` because most keys are optional;
by convention `text` is always present and `id` is always set (derived
from `text` if not supplied).

Keys:
: text: The segment’s text. **Required.**
  id: Stable identifier — content-derived by default ([`segment_id()`](#ef.segment_id)).
  parent_id: Id of the source document or parent segment this came from.
  start: Character offset of the segment’s start in the source text.
  end: Character offset of the segment’s end (exclusive) in the source.
  index: Ordinal position of the segment in its segmenter’s output.
  tokens: Token count of `text` (meaningful only with `metadata` key
  <br/>
  > `tokenizer` recording which tokenizer counted it).
  <br/>
  metadata: Free-form mapping; framework-specific keys live here. See
  : `PROMOTED_METADATA_KEYS` for the conventional keys.

### *class* ef.SegmentRecord(text, id='', parent_id=None, start=None, end=None, index=None, tokens=None, metadata=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Frozen-dataclass view over a [`Segment`](#ef.Segment) — the convenience surface.

Use this when attribute access and immutability read better than dict
access; use the [`Segment`](#ef.Segment) `TypedDict` directly on the hot path.
The two convert losslessly via [`to_segment()`](#ef.SegmentRecord.to_segment) and [`segment_record()`](#ef.segment_record).

`id` is derived from `text` (and `metadata`) if left empty.

```pycon
>>> rec = SegmentRecord('hello', index=2)
>>> rec.id == segment_id('hello')
True
>>> rec.to_segment()['index']
2
```

#### to_segment()

Return an equivalent [`Segment`](#ef.Segment) `dict` (dropping empty keys).

* **Return type:**
  [`Segment`](ef.segments.html.md#ef.segments.Segment)

### *class* ef.Segmenter(\*args, \*\*kwargs)

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

### *class* ef.SourceManager(corpus=None, , segmenter=None, embedder=None, store=None, cache=None, auto_refresh=False, embedder_api_key=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The heavy search facade — multi-config corpus indexing over one corpus.

A [`SourceManager`](#ef.SourceManager) holds a corpus, an [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph)
and one or more **configs**. A config is a named segmenter+embedder pipeline
([`PipelineSpec`](ef.config.html.md#ef.config.PipelineSpec)); [`register_config()`](#ef.SourceManager.register_config) adds one,
[`materialize()`](#ef.SourceManager.materialize) runs the corpus through it into a `vd` collection, and
[`search()`](#ef.SourceManager.search) queries it. Because every artifact is content-addressed in
the shared graph, two configs that share a step (e.g. the same segmenter)
compute that step’s artifacts **once** — config branching is free.

The constructor optionally registers a `"default"` config: pass an
`embedder` (and optionally a `segmenter`) and it is registered eagerly;
omit `embedder` to build every config explicitly via [`register_config()`](#ef.SourceManager.register_config).

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
    for key-less embedders. See [`register_config()`](#ef.SourceManager.register_config).
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
    [`refresh()`](ef.refresh.html.md#module-ef.refresh). [`scan()`](#ef.SourceManager.scan) then also picks up out-of-band edits.

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
Read-only: it changes nothing. [`refresh()`](ef.refresh.html.md#module-ef.refresh) acts on the result.

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

The garbage-collection slice of [`refresh()`](ef.refresh.html.md#module-ef.refresh): it removes only
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
    default) indexes the whole corpus. [`refresh()`](ef.refresh.html.md#module-ef.refresh) uses this to
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
deleted, then the whole corpus is re-[`materialize()`](#ef.SourceManager.materialize)d from
scratch. Use it to recover from a config whose `vd` documents were
produced by an earlier pipeline (every source `misconfigured`), or
whenever a guaranteed-clean index is wanted over the surgical
[`refresh()`](ef.refresh.html.md#module-ef.refresh). Every source is reported as `added` — the collection
was emptied and rebuilt.

* **Return type:**
  [`RefreshReport`](ef.refresh.html.md#ef.refresh.RefreshReport)
* **Returns:**
  a [`RefreshReport`](ef.refresh.html.md#ef.refresh.RefreshReport).

#### refresh(config=None, , sources=None, mode='full')

Re-sync one config’s `vd` index with the current corpus.

The explicit-refresh entry point. It [`diagnose()`](#ef.diagnose)s the config,
turns the staleness report into a [`RefreshPlan`](ef.refresh.html.md#ef.refresh.RefreshPlan) for
the chosen `mode` ([`plan_refresh()`](ef.refresh.html.md#ef.refresh.plan_refresh)), deletes the
stale/orphan documents, re-[`materialize()`](#ef.SourceManager.materialize)s what changed, and
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
defaults to `DEFAULT_EMBEDDER`). Their ops are registered into the
shared [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) and a `vd` collection
is created for the config. Registering a config does *not* index it —
call [`materialize()`](#ef.SourceManager.materialize).

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

The [`SourceManager`](#ef.SourceManager) counterpart of
[`SearchableCorpus.retrieve()`](#ef.SearchableCorpus.retrieve): plain segments in rank order, the
clean context shape an external RAG/agent framework consumes (`ef`
does not synthesize answers). [`search()`](#ef.SourceManager.search) is the scored counterpart.

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

For a large bulk of out-of-band edits, prefer [`refresh()`](ef.refresh.html.md#module-ef.refresh) (one
corpus pass) over `scan` (one collection pass per changed source).

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if the corpus is not change-detecting (the manager was
      not built with `auto_refresh=True`).
* **Return type:**
  [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff)

#### search(query, , config=None, limit=10, filter=None)

Search one config’s index — return up to `limit` ranked [`SearchHit`](#ef.SearchHit)s.

`config` selects which config to query (a name / [`ConfigId`](ef.config.html.md#ef.config.ConfigId),
or `None` for the default or sole config). `query` is text or a
pre-computed query vector; `filter` is a `vd` metadata filter.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)]

#### searchable(config=None)

Return a thin [`SearchableCorpus`](#ef.SearchableCorpus) bound to one config’s index.

* **Return type:**
  [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus)

### *class* ef.StalenessReport(config, orphan=(), missing=(), stale=(), misconfigured=(), fresh=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The four staleness conditions of one config — what [`diagnose()`](#ef.diagnose) returns.

Each field is a sorted tuple of `source_id`s, so a report is
deterministic and JSON-friendly. `orphan` and `missing` are disjoint
from the rest (a source absent from one side cannot also be compared); a
source can be **both** `stale` and `misconfigured` (its content changed
*and* the config did) — the conditions are reported independently.
`fresh` is the complement: indexed, current and built by this config.

Truthiness reports whether *anything is wrong* — `bool(report)` is
`False` only for a perfectly in-sync index.

```pycon
>>> r = StalenessReport(config='c0', stale=('a',), fresh=('b',))
>>> bool(r), r.needs_reindexing
(True, ('a',))
>>> bool(StalenessReport(config='c0', fresh=('a', 'b')))
False
```

#### *property* needs_reindexing *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

Sources whose vectors must be recomputed — `stale` ∪ `misconfigured`.

### *class* ef.TransformSpec(op, op_version, params=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One step of a pipeline — a serializable recipe for an op application.

A [`TransformSpec`](#ef.TransformSpec) names *which op* (`op`, a key resolved through
the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph)’s `ops` registry), \*at which
version\* (`op_version` — it participates in every downstream hash), and
*with which keyword params* (`params`). It does **not** record the step’s
inputs: those are wired structurally by the [`PipelineSpec`](#ef.PipelineSpec) (the embed
step consumes the segment step’s output). It is the per-step counterpart of
a [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec).

The op key carries the *component identity* — e.g. `"embed:openai:text-
embedding-3-large@1024"` or `"segment:<segmenter-identity>"` (see
[`ef.ops`](ef.ops.html.md#module-ef.ops)) — so two configs with different components get disjoint
artifact cones automatically.

```pycon
>>> ts = TransformSpec(op='embed:m@8', op_version='2', params={'input_type': 'document'})
>>> ts.op, ts.op_version
('embed:m@8', '2')
>>> TransformSpec.from_dict(ts.as_dict()) == ts
True
```

#### as_dict()

Return a plain-`dict` (JSON-serializable) form of the spec.

```pycon
>>> TransformSpec(op='segment:r', op_version='1', params={}).as_dict()
{'op': 'segment:r', 'op_version': '1', 'params': {}}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *classmethod* from_dict(d)

Rebuild a [`TransformSpec`](#ef.TransformSpec) from its [`as_dict()`](#ef.TransformSpec.as_dict) form.

* **Return type:**
  [`TransformSpec`](ef.config.html.md#ef.config.TransformSpec)

### ef.approx_token_count(text)

Rough token count — the common `~4 characters per token` heuristic.

A dependency-free default so `ef` can talk in “tokens” without pulling in
`tiktoken` or a model tokenizer. Inject a real counter (and its name) into
a segmenter when an exact count matters.

```pycon
>>> approx_token_count('a' * 40)
10
>>> approx_token_count('')
0
```

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### ef.artifact_id(spec)

Content-addressed id of `spec` — `sha256` over its four fields.

The id is `sha256` of canonical (sorted-key) JSON of
`{op, op_version, inputs, params}`. Two specs that describe the same
computation hash *identically*; any change to the op, its version, its
inputs (including their order) or its params yields a *different* id. That
is what makes the graph content-addressed: the id *is* the recipe.

```pycon
>>> a = producer_spec('embed', 'seg-1', op_version='2', model='small')
>>> artifact_id(a) == artifact_id(producer_spec(
...     'embed', 'seg-1', op_version='2', model='small'))
True
>>> len(artifact_id(a))
64
>>> artifact_id(a) == artifact_id(producer_spec(
...     'embed', 'seg-1', op_version='3', model='small'))   # version bump
False
>>> artifact_id(producer_spec('op', 'x', 'y', op_version='1')) == artifact_id(
...     producer_spec('op', 'y', 'x', op_version='1'))      # input order matters
False
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ef.as_corpus(x=None, , \*\*kwargs)

Normalize `x` into a corpus (`MutableMapping[source_id, Source]`).

The single dependency-injection seam through which every `ef` entry point
accepts a user-supplied corpus — the mirror of
[`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder) and
[`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter). Accepts, in order:

1. `None` — a fresh empty in-RAM `dict`;
2. a `Mapping` — returned **unchanged** (a `dict`, a `dol` store,
   another corpus: it already *is* a corpus);
3. a directory path (`str` / `os.PathLike` naming an existing
   directory) — a `dol` filesystem store of the text files under it
   (`dol` is imported lazily; only this branch needs it);
4. an iterable of sources (strings or mappings) — an in-RAM `dict` keyed
   by each source’s [`content_hash()`](#ef.content_hash) (content-addressed, so duplicate
   sources collapse to one entry).

Extra `**kwargs` are forwarded to the `dol` filesystem store (the
directory branch only) and ignored otherwise.

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is a non-directory string, or none of the above.

```pycon
>>> as_corpus()
{}
>>> as_corpus({'a': 'x'})                 # a mapping passes straight through
{'a': 'x'}
>>> sorted(as_corpus(['alpha', 'beta']).values())
['alpha', 'beta']
>>> len(as_corpus(['dup', 'dup']))        # content-addressed: duplicates collapse
1
```

### ef.as_embedder(x, \*\*kwargs)

Normalize `x` into an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) — the DI seam.

The single place every `ef` entry point coerces a user-supplied embedder
argument. Accepts, in order:

1. an existing [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) — returned unchanged;
2. a URL string (`http://` / `https://`) — [`http_embedder()`](#ef.http_embedder);
3. a provider-prefixed string — `"openai:<model>"`, `"cohere:<model>"`,
   `"voyage:<model>"` or `"gemini:<model>"` — the matching hosted-API
   adapter;
4. the bare string `"hashing"` — the dependency-free
   [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder) (`ef`’s zero-install default);
5. an `"st:<model>"` string, or any other bare model name —
   [`sentence_transformers_embedder()`](#ef.sentence_transformers_embedder) (the local neural default);
6. a bare callable — wrapped in
   [`FunctionEmbedder`](ef.embedders.html.md#ef.embedders.FunctionEmbedder) (the `imbed`-embedder bridge).

Extra `**kwargs` are forwarded to the chosen factory.

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is none of the above.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)

```pycon
>>> import numpy as np
>>> e = as_embedder(lambda ts: np.zeros((len(ts), 2)), model_id='zero@2')
>>> type(e).__name__
'FunctionEmbedder'
```

### ef.as_ragas_dataset(samples)

Convert [`RagSample`](#ef.RagSample)s into a Ragas `EvaluationDataset`.

The hookpoint for the *LLM-judged* RAG metrics `ef` deliberately does not
compute itself (faithfulness, answer relevancy, …). [`RagSample`](#ef.RagSample)
already mirrors Ragas’ `SingleTurnSample` field-for-field, so this is a
thin, lossless adapter. Run the evaluation with your own LLM:

```default
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

dataset = as_ragas_dataset(samples)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

* **Parameters:**
  **samples** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`RagSample`](ef.evaluation.html.md#ef.evaluation.RagSample) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – an iterable of [`RagSample`](#ef.RagSample)s (or plain mappings).
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  a `ragas.EvaluationDataset`.
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – if the optional `ragas` package is not installed.

### ef.as_segment(x, \*\*defaults)

Coerce `x` into a [`Segment`](#ef.Segment) — accepts a `str`, `Mapping` or record.

This is the segment-side normalizer that lets a [`Segmenter`](#ef.Segmenter) accept
loosely-typed output (a bare string, a partial `dict`) and still emit a
well-formed [`Segment`](#ef.Segment). `**defaults` supplies *fallback* values:
a default fills a field only when `x` does not already carry it.

Accepts:

1. a [`SegmentRecord`](#ef.SegmentRecord) — converted via [`SegmentRecord.to_segment()`](#ef.SegmentRecord.to_segment);
2. a `str` — becomes the `text` of a fresh segment;
3. a `Mapping` with a `text` key — its keys are carried through.

The `id` is always present on the result (derived if absent).

* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is none of the accepted forms.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `x` is a `Mapping` without a `text` key.
* **Return type:**
  [`Segment`](ef.segments.html.md#ef.segments.Segment)

```pycon
>>> as_segment('plain text', index=3)['index']
3
>>> as_segment({'text': 'kept', 'index': 0}, index=9)['index']  # x wins
0
```

### ef.as_segmenter(x=None, , \*\*kwargs)

Normalize `x` into a [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) — the DI seam.

The single place every `ef` entry point coerces a user-supplied segmenter
argument. Accepts, in order:

1. `None` or `"default"` / `"recursive"` — a
   [`RecursiveCharacterSegmenter`](ef.segmenters.html.md#ef.segmenters.RecursiveCharacterSegmenter) (the default);
2. `"lines"` — [`line_segmenter()`](ef.segmenters.html.md#ef.segmenters.line_segmenter);
3. any other string — looked up in `imbed`’s registry via
   [`imbed_segmenter()`](#ef.imbed_segmenter);
4. a ready-made segmenter (a [`BaseSegmenter`](ef.segmenters.html.md#ef.segmenters.BaseSegmenter) or a
   callable marked by `ef`) — returned unchanged;
5. a bare callable — wrapped in [`FunctionSegmenter`](ef.segmenters.html.md#ef.segmenters.FunctionSegmenter).

Extra `**kwargs` are forwarded to the chosen factory (ignored when `x`
is already a segmenter).

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is none of the above.
* **Return type:**
  [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter)

```pycon
>>> seg = as_segmenter(chunk_size=256, chunk_overlap=32)
>>> (seg.chunk_size, seg.chunk_overlap)
(256, 32)
```

### ef.average_precision(ranked, relevant, k=None)

Average precision of `ranked` — the per-query term of MAP.

The mean of the precision measured at each relevant result’s rank. `k`
optionally caps the ranking.

```pycon
>>> average_precision(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0})
0.8333333333333333
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.benchmark_embedders(corpus_sources, embedders, qrels, queries, , distractors=None, hard_positives=None, k_values=(10, 20), segmenter=None, limit=None)

Re-index the same corpus under each embedder and tabulate the metrics.

For every `name -> spec` in `embedders` (a spec is anything
[`ef.as_embedder()`](#ef.as_embedder) accepts: an `Embedder`, `'st:bge-large-en-v1.5'`,
`'openai:text-embedding-3-large'` …) it runs [`ef.source_manager.ingest()`](ef.source_manager.html.md#ef.source_manager.ingest),
then [`evaluate_retrieval()`](#ef.evaluate_retrieval), and stacks one row per embedder. Heavy
adapters are only imported when their spec is resolved. Deterministic
metrics only — no LLM.

```pycon
>>> from ef.embedders import HashingEmbedder
>>> corpus = {'d1': 'red apple fruit', 'd2': 'blue car engine'}
>>> report = benchmark_embedders(
...     corpus, {'hash': HashingEmbedder()},
...     {'q': {'d1': 1.0}}, {'q': 'apple'}, k_values=(1,))
>>> report.rows[0]['embedder'], report.rows[0]['recall@1']
('hash', 1.0)
```

* **Return type:**
  [`BenchmarkReport`](ef.evaluation.html.md#ef.evaluation.BenchmarkReport)

### ef.cache_key(embedder, text, input_type=None)

Deterministic cache key for one `(embedder, text, input_type)` triple.

The key pins everything that changes the resulting vector. `model_id` is
the embedder-identity SSOT — it already bakes in provider, model and `dim`
— so `dim` is *not* keyed separately (keying it would also break
[`FunctionEmbedder`](#ef.FunctionEmbedder)’s lazy `dim` inference). `normalized` is keyed
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

### ef.cluster(data, , method='kmeans', n_clusters=8, min_cluster_size=5, embedder=None, normalize=True, random_state=42)

Cluster a corpus’s embeddings (use cases §G2).

`method='kmeans'` is the default and is **dependency-free** — a numpy
Lloyd’s algorithm with k-means++ seeding. `method='hdbscan'` finds a
density-based clustering (and a noise label `-1`) and needs the
`ef[explore]` extra.

* **Parameters:**
  * **data** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – what to cluster — see [`project()`](#ef.project) / `_resolve_vectors()`.
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

### ef.cohere_embedder(model='embed-v4.0', , dim=None, api_key=None, batch_size=96, timeout=60.0)

Build a Cohere-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Talks to Cohere’s `/v2/embed` REST endpoint directly — no `cohere` SDK,
no extra dependency. Cohere’s v3+ models *require* a task hint; this adapter
supplies one for every canonical [`InputType`](ef.embedders.html.md#ef.embedders.InputType) and
defaults to the document role when none is given.

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A Cohere embedding model name.
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated width (`embed-v4.0` only). Omit for the
    model’s native width. The dim is part of the embedder’s identity —
    set it once here, not per call.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Cohere API key. Falls back to `CO_API_KEY` / `COHERE_API_KEY`.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at Cohere’s 96 limit).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over the Cohere API.

### ef.config_id(spec)

Content-addressed id of a [`PipelineSpec`](#ef.PipelineSpec) — `sha256` over its steps.

The id is `sha256` of the canonical (sorted-key) JSON of
[`PipelineSpec.as_dict()`](#ef.PipelineSpec.as_dict). Two pipelines that describe the same
computation hash identically; any change to a step’s op, version or params
yields a different id. `ef` writes this digest into `vd` index metadata
as `config_hash`.

```pycon
>>> seg = TransformSpec(op='segment:r', op_version='1', params={})
>>> emb = TransformSpec(op='embed:m@8', op_version='1', params={})
>>> cid = config_id(PipelineSpec(segment=seg, embed=emb))
>>> len(cid)
64
>>> cid == config_id(PipelineSpec(segment=seg, embed=emb))
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ef.content_hash(source, , content_keys=None)

Content hash of a source document — `sha256` of its normalized content.

This is the basis of change detection: re-reading an unchanged source
yields an identical hash, so a refresh that finds the same hash does
nothing. Hashing *normalized* content (NFC, no BOM, `\n` line endings —
see [`ef.hashing`](ef.hashing.html.md#module-ef.hashing)) means cosmetic encoding differences never look like
a change.

The treatment of a `Mapping` source is deliberate:

- a plain `str` source — the hash covers the text;
- a `Mapping` with a `text` key — the hash covers `source['text']`
  **only**. A change to title, tags, ACL or any other metadata does *not*
  change the hash, so it never triggers a spurious re-segment/re-embed
  (embedding is the pipeline’s most expensive step). Name the metadata keys
  that *do* feed downstream in `content_keys` and changes to those keys
  will be detected;
- a `Mapping` *without* a `text` key — treated as a structured document
  and hashed via canonical (sorted-key) JSON of the whole mapping;
- `bytes` — hashed raw (no text normalization is possible pre-parse).

* **Parameters:**
  * **source** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a `str`, `bytes`, or a `Mapping` (ideally with a `text`
    key).
  * **content_keys** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – metadata keys whose values join the text in the hash.
    Each key is looked up at the top level of `source` or inside a
    `source['metadata']` sub-mapping. Use this *only* for metadata
    that genuinely affects the L1–L4 pipeline (e.g. a title prepended
    to the text before embedding); key order does not matter.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `source` is not a `str`, `bytes` or `Mapping`.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> content_hash('hello') == content_hash('hello')
True
>>> content_hash('hello') == content_hash('hello world')
False
>>> content_hash('a\r\nb') == content_hash('a\nb')   # line endings normalized
True
>>> a = {'text': 'body', 'metadata': {'tags': ['x']}}
>>> b = {'text': 'body', 'metadata': {'tags': ['y']}}
>>> content_hash(a) == content_hash(b)                # metadata ignored by default
True
>>> content_hash(a, content_keys=['tags']) == content_hash(b, content_keys=['tags'])
False
>>> content_hash({'text': 'body'}) == content_hash('body')  # wrapper is transparent
True
```

### ef.context_precision(reference, retrieved_contexts)

Fraction of the retrieved contexts that lexically overlap the `reference`.

A deterministic *proxy* for retrieval precision — a context counts as useful
if it shares at least one (normalized) token with the reference answer.

```pycon
>>> context_precision('paris', ['Paris is in France.', 'Unrelated text.'])
0.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.context_recall(reference, retrieved_contexts)

Lexical recall of the `reference` tokens by the retrieved contexts.

The fraction of the (normalized) reference’s distinct tokens that appear
anywhere in `retrieved_contexts` — a deterministic *proxy* for “did
retrieval surface the information the answer needs”. For the LLM-judged
context recall, use [`as_ragas_dataset()`](#ef.as_ragas_dataset) and Ragas.

```pycon
>>> context_recall('paris france', ['Paris is the capital of France.'])
1.0
>>> context_recall('paris france', ['London is in England.'])
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.cross_encoder_reranker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', , device=None, batch_size=32, \*\*predict_kwargs)

Build a [`Reranker`](#ef.Reranker) backed by a sentence-transformers `CrossEncoder`.

A cross-encoder scores a `(query, passage)` pair *jointly* — it is the
accurate, slower second stage two-stage retrieval is built for. The model is
a heavy, optional dependency: it is imported only when this factory is
called, so importing [`ef.reranking`](ef.reranking.html.md#module-ef.reranking) itself stays cheap.

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
  [`Reranker`](ef.reranking.html.md#ef.reranking.Reranker)
* **Returns:**
  a [`Reranker`](#ef.Reranker) — `(query, segments) -> scores`.
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – if the optional `sentence-transformers` package is absent.

### ef.dcg_at_k(ranked, relevant, k)

Discounted cumulative gain of `ranked` at cutoff `k`.

`sum(grade_i / log2(i + 2))` over the top `k` results (`i` 0-indexed),
with `grade_i` the relevance of the `i`-th doc (`0` if unjudged) — the
linear-gain DCG that BEIR / `pytrec_eval` use.

```pycon
>>> round(dcg_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 3), 4)
1.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.diagnose(corpus, collection, config)

Compute the four staleness conditions of `collection` against `corpus`.

The pure core of `ef`’s diagnostics — no side effects. It reads what the
`vd` `collection` records (`indexed_state()`), what the `corpus`
currently holds (`corpus_state()`), and partitions every source id into
exactly the buckets it belongs in:

- **orphan** — in the collection, not in the corpus;
- **missing** — in the corpus, not in the collection;
- **stale** — in both, but the recorded `source_hash` ≠ the current one;
- **misconfigured** — in both, but a recorded `config_hash` ≠ `config`;
- **fresh** — in both, content current and config matching.

* **Parameters:**
  * **corpus** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the source corpus.
  * **collection** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the `vd` collection backing the config being diagnosed.
  * **config** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the [`ConfigId`](ef.config.html.md#ef.config.ConfigId) the collection *should* hold
    — documents recording any other `config_hash` are misconfigured.
* **Return type:**
  [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport)
* **Returns:**
  a [`StalenessReport`](#ef.StalenessReport).

### ef.embed_length_sorted(texts, encode, , batch_size=128)

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

### ef.evaluate_rag(samples, , metrics=('exact_match', 'token_f1', 'context_recall', 'context_precision', 'non_empty_rate'))

Score RAG samples with deterministic, reference-based lexical metrics.

Each sample is a [`RagSample`](#ef.RagSample) — `(user_input, response,
retrieved_contexts, reference)`. `evaluate_rag` computes only metrics
that need **no LLM**: [`exact_match()`](#ef.exact_match), [`token_f1()`](#ef.token_f1),
[`context_recall()`](#ef.context_recall), [`context_precision()`](#ef.context_precision) and `non_empty_rate`
(the share of samples that retrieved any context). This keeps `ef` a
facade — it never synthesizes an answer and never calls a model (§6).

For the LLM-judged metrics — faithfulness, answer relevancy, … — pass the
same samples to [`as_ragas_dataset()`](#ef.as_ragas_dataset) and run Ragas with your own LLM.

* **Parameters:**
  * **samples** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`RagSample`](ef.evaluation.html.md#ef.evaluation.RagSample) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – an iterable of [`RagSample`](#ef.RagSample)s (or plain mappings of the
    same shape).
  * **metrics** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – which metrics to compute — any of `exact_match` /
    `token_f1` / `context_recall` / `context_precision` /
    `non_empty_rate`. The reference-based ones skip samples that carry
    no `reference`.
* **Return type:**
  [`RagEvalReport`](ef.evaluation.html.md#ef.evaluation.RagEvalReport)
* **Returns:**
  a [`RagEvalReport`](#ef.RagEvalReport).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – for an unknown metric name, or if `samples` is empty.

```pycon
>>> samples = [
...     {'user_input': 'capital of France?', 'response': 'Paris',
...      'reference': 'Paris',
...      'retrieved_contexts': ['Paris is the capital of France.']},
...     {'user_input': '2 + 2?', 'response': 'five', 'reference': 'four',
...      'retrieved_contexts': []},
... ]
>>> report = evaluate_rag(samples)
>>> report.metrics['exact_match']
0.5
>>> report.metrics['non_empty_rate']
0.5
```

### ef.evaluate_retrieval(retriever, qrels, queries, , k_values=(1, 5, 10, 100), metrics=('ndcg', 'recall', 'precision', 'mrr', 'map'), limit=None, distractors=None, hard_positives=None)

Score a retriever against a BEIR-shaped evaluation set.

For every query judged in `qrels`, the `retriever` is run, its results
are mapped back to *source document ids* (`_result_doc_id()`,
de-duplicated to the best rank per source), and each requested metric is
computed at each cutoff in `k_values`. The report holds the mean of each
`"<metric>@<k>"` over all scored queries — primary metric **NDCG@10**.

The natural pairing with [`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest):

```default
corpus, queries, qrels = read_beir("path/to/beir/dataset")
index = ingest(corpus, embedder="st:all-MiniLM-L6-v2")
report = evaluate_retrieval(index, qrels, queries)
report.primary            # mean NDCG@10
```

* **Parameters:**
  * **retriever** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – what answers a query — a [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus),
    a [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager), or any callable
    `query -> ranked results`. Results may be
    [`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)s, [`Segment`](ef.segments.html.md#ef.segments.Segment)s
    or bare doc-id strings.
  * **qrels** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – the relevance judgements — `{query_id: {doc_id: relevance}}`.
    Queries with no positive judgement are skipped.
  * **queries** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the query texts — `{query_id: query_text}`. A judged query
    absent here cannot be run and is skipped.
  * **k_values** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – the rank cutoffs to evaluate at.
  * **metrics** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – which metrics to compute — any of `ndcg` / `recall` /
    `precision` / `mrr` / `map`.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – how many results to retrieve per query. `None` uses
    `max(k_values)`; raise it when sources segment into many pieces
    (de-duplication to source level can otherwise shorten the ranking).
  * **distractors** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional `{query_id: named distractor doc ids}`; adds
    `fp_rate@k` ([`fp_rate_at_k()`](#ef.fp_rate_at_k)) to the metrics, averaged over the
    queries that name distractors, and fills `report.distractor_hits`.
  * **hard_positives** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional `{query_id: hard-positive doc ids}`; adds
    `hard_recall@k` ([`hard_positive_recall_at_k()`](#ef.hard_positive_recall_at_k)), likewise.
* **Return type:**
  [`RetrievalEvalReport`](ef.evaluation.html.md#ef.evaluation.RetrievalEvalReport)
* **Returns:**
  a [`RetrievalEvalReport`](#ef.RetrievalEvalReport).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – for an unknown metric name, or if no query can be scored.

```pycon
>>> qrels = {'q1': {'d1': 1.0}, 'q2': {'d2': 1.0}}
>>> queries = {'q1': 'alpha', 'q2': 'beta'}
>>> def retriever(query, *, limit=10):
...     return {'alpha': ['d1', 'd2'], 'beta': ['d1', 'd2']}[query][:limit]
>>> report = evaluate_retrieval(retriever, qrels, queries, k_values=(1, 2))
>>> report.metrics['recall@2'], report.n_queries
(1.0, 2)
```

### ef.exact_match(response, reference)

`1.0` if `response` equals `reference` after normalization, else `0.0`.

```pycon
>>> exact_match('The Eiffel Tower', 'eiffel tower')
1.0
>>> exact_match('Paris', 'London')
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.explore(data, , dims=2, projection_method='auto', cluster_method='kmeans', n_clusters=8, min_cluster_size=5, label=False, context=' ', n_words=4, embedder=None, random_state=42)

Project, cluster and (optionally) label a corpus in one call.

The layer-L5 orchestrator — [`project()`](#ef.project) + [`cluster()`](#ef.cluster) (+ optionally
[`label_clusters()`](#ef.label_clusters)) wired into one structured result. Where
[`project()`](#ef.project) / [`cluster()`](#ef.cluster) return *bare arrays in input order*,
`explore` keeps every row tied to its id: it returns an
[`ExploreResult`](#ef.ExploreResult) with `ids` / `coords` / `labels` row-aligned
(plus `cluster_titles`) — the JSON-friendly shape an `app_ef` corpus map
(or [`ef.service.EfService.explore_corpus()`](ef.service.html.md#ef.service.EfService.explore_corpus)) consumes directly.

* **Parameters:**
  * **data** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the corpus to explore — anything `_resolve_explorable()`
    understands (a [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) /
    [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus), a
    [`Corpus`](ef.corpus.html.md#ef.corpus.Corpus) mapping, an iterable of texts or
    [`Segment`](ef.segments.html.md#ef.segments.Segment)s, or a vector matrix).
  * **dims** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – projection target dimensionality — `2` or `3`.
  * **projection_method** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'auto'`, `'umap'`, `'pca'`]) – forwarded to [`project()`](#ef.project) — `'auto'` /
    `'umap'` / `'pca'`.
  * **cluster_method** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'kmeans'`, `'hdbscan'`]) – forwarded to [`cluster()`](#ef.cluster) — `'kmeans'` /
    `'hdbscan'`.
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – number of k-means clusters.
  * **min_cluster_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – smallest cluster HDBSCAN will form.
  * **label** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – when `True`, name each cluster with [`label_clusters()`](#ef.label_clusters)
    (needs the `ef[imbed]` extra, an LLM key, and text-bearing
    `data`); when `False` (the default) `cluster_titles` is empty.
  * **context** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the corpus topic passed to [`label_clusters()`](#ef.label_clusters).
  * **n_words** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – maximum cluster-title length, in words.
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – embedder used when `data` is raw text.
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – seed — projection and clustering are reproducible.
* **Return type:**
  [`ExploreResult`](ef.exploration.html.md#ef.exploration.ExploreResult)
* **Returns:**
  an [`ExploreResult`](#ef.ExploreResult) — `ids`, `coords` and `labels`
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

### ef.fp_rate_at_k(ranked, distractors, k)

Fraction of the *named distractors* that intrude into the top `k`.

A distractor is a document known to be topically near the query but not
relevant; aggregate NDCG can improve while these keep over-ranking. 0.0 is
best.

```pycon
>>> fp_rate_at_k(['au', 'a', 'su', 'b'], ['au', 'su', 'csm'], 3)
0.6666666666666666
>>> fp_rate_at_k(['a', 'b'], ['au'], 2)
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.full_kwargs(op, args=(), kwargs=None)

Normalize a call into a canonical, fully-named, defaults-filled kwargs dict.

The `i2.Sig` recipe of design notes §3.8. A call expressed as positional
`args` + keyword `kwargs` is mapped onto `op`’s signature: every
parameter is named, and any parameter the call left implicit is filled with
its default. The result is therefore *stable* — `op(x)` and
`op(x, size=512)` (`512` the default) yield the **same** dict, which is
what makes a content hash built from it correct.

* **Parameters:**
  * **op** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the callable whose signature defines the parameter names/defaults.
  * **args** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]) – the call’s positional arguments — for an [`ef.ops`](ef.ops.html.md#module-ef.ops) op these
    are the upstream artifact ids; [`step_params()`](#ef.step_params) drops them again.
  * **kwargs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the call’s keyword arguments.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> def embed(segments, *, input_type='document', dim=None): ...
>>> full_kwargs(embed, ('seg-1',), {'input_type': 'query'})
{'segments': 'seg-1', 'input_type': 'query', 'dim': None}
>>> full_kwargs(embed, ('seg-1',))                 # defaults filled in
{'segments': 'seg-1', 'input_type': 'document', 'dim': None}
```

### ef.gemini_embedder(model='gemini-embedding-001', , dim=None, api_key=None, batch_size=100, timeout=60.0)

Build a Google Gemini [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Talks to the Generative Language `batchEmbedContents` REST endpoint
directly — no `google-genai` SDK, no extra dependency.

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A Gemini embedding model name (a leading `models/` is
    tolerated and stripped).
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated width (`gemini-embedding-001` only). Omit
    for the model’s native width. Note that Gemini, unlike OpenAI, does
    *not* re-normalize MRL-truncated output — a truncated embedder
    therefore reports `normalized=False`.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Gemini API key. Falls back to `GEMINI_API_KEY` /
    `GOOGLE_API_KEY`.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at Gemini’s 100 limit).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over the Gemini API.

### ef.hard_positive_recall_at_k(ranked, hard_positives, k)

Fraction of the *hard positives* (e.g. thin-description docs) found in the top `k`.

```pycon
>>> hard_positive_recall_at_k(['a', 'imbed', 'b'], ['imbed', 'meshed'], 2)
0.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.hierarchical(segmenters,)

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
  [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter)

### ef.hits_to_segments(hits)

Project [`SearchHit`](#ef.SearchHit)s to plain [`Segment`](ef.segments.html.md#ef.segments.Segment)s.

The shape [`SearchableCorpus.retrieve()`](#ef.SearchableCorpus.retrieve) / [`SourceManager.retrieve()`](#ef.SourceManager.retrieve)
hand to an external RAG/agent framework — the RAG-plug-in surface (design
notes §F5). A [`SearchHit`](#ef.SearchHit) carries an `ef`-specific `score` and
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

### ef.http_embedder(url, \*, model_id=None, dim=None, normalized=False, batch_size=64, headers=None, timeout=30.0, payload_builder=<function \_tei_payload>, response_parser=<function \_tei_parse>)

Build an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over a remote HTTP service.

Defaults target the Text-Embeddings-Inference / infinity shape (a POST of
`{"inputs": [...]}` returning a JSON list of vectors). For a different
service, override `payload_builder` and/or `response_parser`.

* **Parameters:**
  * **url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The embedding endpoint.
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Vector-identity string (defaults to `"http:<url>"`).
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Output width; inferred from the first response if omitted.
  * **normalized** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether the service returns unit vectors.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (used with length-sorted batching).
  * **headers** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Extra HTTP headers (e.g. an auth token).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
  * **payload_builder** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]], [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – `texts -> request body bytes`.
  * **response_parser** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `decoded JSON -> array-like (n, dim)`.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)

Uses only the stdlib — no extra dependency.

### ef.imbed_segmenter(name='default', , \*\*kwargs)

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

### ef.ingest(sources, , segmenter=None, embedder=None, store=None, cache=None)

Index `sources` and return a ready-to-search [`SearchableCorpus`](#ef.SearchableCorpus) — one call.

The light path: corpus in, `search`-ready object out, every component
defaulted. `ingest(['a', 'b', 'c']).search('query')` is the whole story.
It builds a single-config [`SourceManager`](#ef.SourceManager), [`materialize()`](#ef.SourceManager.materialize)s
it, and hands back the thin [`SearchableCorpus`](#ef.SearchableCorpus).

* **Parameters:**
  * **sources** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the corpus — coerced by [`as_corpus()`](ef.corpus.html.md#ef.corpus.as_corpus) (a mapping,
    a directory path, an iterable of strings/mappings, or `None`).
  * **segmenter** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the segmenter — [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter)
    coerces it (`None` → the recursive default).
  * **embedder** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the embedder — [`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder)
    coerces it; `None` → `DEFAULT_EMBEDDER`, the dependency-free
    [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder) (so this call needs nothing
    beyond `pip install ef`). Pass an explicit embedder — e.g.
    `"st:all-MiniLM-L6-v2"` — for neural semantic quality.
  * **store** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the vector store — `None` → an in-memory `vd` backend. See
    [`SourceManager`](#ef.SourceManager) for the other accepted forms.
  * **cache** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the artifact-graph value cache — any `MutableMapping`.
* **Return type:**
  [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus)
* **Returns:**
  a [`SearchableCorpus`](#ef.SearchableCorpus) over the freshly indexed corpus.

See the module docstring for a complete offline example.

### ef.label_clusters(segments, labels, , context=' ', n_words=4, n_samples=None, \*\*labeler_kwargs)

Name each cluster with a short LLM-generated title (use cases §G3).

A thin wrapper over `imbed`’s `imbed.tools.ClusterLabeler`: it
samples each cluster’s segments, asks an LLM for a few-word title, and
returns `{cluster_id: title}`. `ef` does not reimplement this — it
reuses `imbed` (design notes §7).

* **Parameters:**
  * **segments** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the segment texts — plain strings or [`Segment`](ef.segments.html.md#ef.segments.Segment)
    mappings (the `"text"` field is used). Aligned with `labels`.
  * **labels** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the cluster id of each segment — typically the output of
    [`cluster()`](#ef.cluster).
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

### ef.line_segmenter(doc,)

Segment a document into its non-empty lines.

Each line is stripped of surrounding whitespace; blank lines are dropped.
Character offsets point at the *stripped* content within the source.

A plain function — and therefore already a [`Segmenter`](#ef.Segmenter). It carries
the marker [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter) uses to recognize a
ready-made segmenter.

```pycon
>>> [s['text'] for s in line_segmenter('  hello  \n\nworld\n')]
['hello', 'world']
```

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]

### ef.make_segment(text, , id=None, parent_id=None, start=None, end=None, index=None, tokens=None, metadata=None)

Build a [`Segment`](#ef.Segment), deriving a content-based `id` if none given.

Optional fields are omitted from the result when `None` — a segment
`dict` carries only the keys it actually has.

```pycon
>>> seg = make_segment('chunk one', index=0, start=0, end=9)
>>> sorted(seg)
['end', 'id', 'index', 'start', 'text']
>>> seg['id'] == segment_id('chunk one')
True
```

* **Return type:**
  [`Segment`](ef.segments.html.md#ef.segments.Segment)

### ef.materialise(x)

Force a streaming segmenter (or segment stream) to a concrete list.

Streaming is the default, but the light case often wants everything in
hand. [`materialise()`](#ef.materialise) is polymorphic:

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

### ef.ndcg_at_k(ranked, relevant, k)

Normalized DCG at cutoff `k` — `ef`’s **primary** retrieval metric.

[`dcg_at_k()`](#ef.dcg_at_k) divided by the ideal DCG (the DCG of the best possible
ranking of the judged docs). `1.0` is a perfect ranking; `0.0` when no
relevant doc was judged.

```pycon
>>> ndcg_at_k(['d1', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)   # perfect order
1.0
>>> round(ndcg_at_k(['x', 'd1'], {'d1': 1.0}, 2), 4)     # relevant doc at rank 2
0.6309
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.openai_embedder(model='text-embedding-3-small', , dim=None, api_key=None, client=None, batch_size=2048, poll_interval=30.0)

Build an OpenAI-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – An OpenAI embedding model name.
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated output width (`text-embedding-3-*` only).
    Omit for the model’s native dimension. The dim is part of the
    embedder’s identity — set it once here, not per call.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – OpenAI API key (else the SDK’s usual env-var resolution).
  * **client** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A pre-built `openai.OpenAI` client (overrides `api_key`).
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at OpenAI’s 2048 limit).
  * **poll_interval** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Seconds between polls when blocking on a Batch-API job.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An embedder whose `embed_batch` uses the async OpenAI Batch API.

Requires the `openai` package (`pip install 'ef[openai]'`).

### ef.plan_refresh(report, , mode='full', scope=None)

Turn a [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport) into a [`RefreshPlan`](#ef.RefreshPlan).

The pure, side-effect-free core of the refresh policy. Every source that is
`missing`, `stale` or `misconfigured` is (re-)materialized; what gets
*deleted* is what the `mode` dictates:

- `none` — delete nothing (stale content is added alongside the old);
- `incremental` — delete the stale/misconfigured sources’ old documents;
- `full` / `scoped_full` — also delete orphans.

* **Parameters:**
  * **report** ([`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport)) – the staleness report to act on.
  * **mode** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'none'`, `'incremental'`, `'full'`, `'scoped_full'`]) – one of `REFRESH_MODES`.
  * **scope** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional subset of source ids the plan is restricted to —
    sources outside it are left entirely alone. `None` means the whole
    report. (With `scope=None`, `full` and `scoped_full` coincide.)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `mode` is not a recognized `RefreshMode`.
* **Return type:**
  [`RefreshPlan`](ef.refresh.html.md#ef.refresh.RefreshPlan)

```pycon
>>> from ef.diagnostics import StalenessReport
>>> report = StalenessReport(config='c0', orphan=('x',), missing=('a',),
...                          stale=('b',), misconfigured=('c',), fresh=('d',))
>>> plan_refresh(report, mode='full').to_materialize
('a', 'b', 'c')
>>> plan_refresh(report, mode='full').to_delete
('b', 'c', 'x')
>>> plan_refresh(report, mode='incremental').to_delete
('b', 'c')
>>> plan_refresh(report, mode='none').to_delete
()
>>> plan_refresh(report, mode='scoped_full', scope=['a', 'b']).to_delete
('b',)
```

### ef.precision_at_k(ranked, relevant, k)

Fraction of the top `k` results that are relevant.

```pycon
>>> precision_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)
0.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.producer_spec(op, \*inputs, op_version, \*\*params)

Build a [`ProducerSpec`](#ef.ProducerSpec) ergonomically — inputs positional, params keyword.

The natural way to write a recipe: the input artifact ids are positional
(their order is significant — it is the order the op receives them), and
every keyword argument is an op parameter. `op_version` is keyword-only
and required — an op without a stated version cannot be content-addressed
safely.

```pycon
>>> spec = producer_spec('segment', 'doc-1', op_version='1', size=512, overlap=64)
>>> spec.op, spec.inputs
('segment', ('doc-1',))
>>> spec.params == {'size': 512, 'overlap': 64}
True
```

* **Return type:**
  [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec)

### ef.project(data, , dims=2, method='auto', embedder=None, n_neighbors=15, min_dist=0.1, metric='cosine', pca_components=50, random_state=42)

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

### ef.read_beir(directory, , qrels=None)

Load a BEIR-format dataset from disk into the in-memory evaluation triple.

Reads `corpus.jsonl` (records `{"_id", "title", "text"}`),
`queries.jsonl` (records `{"_id", "text"}`) and a `qrels` TSV
(`query-id`, `corpus-id`, `score` columns). The corpus text is the
record’s `title` and `text` joined — the BEIR convention.

* **Parameters:**
  * **directory** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the dataset directory.
  * **qrels** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the qrels file, relative to `directory`. `None` tries
    `_DEFAULT_QRELS_PATHS` in order.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]
* **Returns:**
  a `(corpus, queries, qrels)` triple — `corpus` and `queries` are
  `{id: text}` dicts, `qrels` is `{query_id: {doc_id: relevance}}`.
  Feed it straight to [`evaluate_retrieval()`](#ef.evaluate_retrieval) (after indexing `corpus`
  with [`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest)).
* **Raises:**
  [**FileNotFoundError**](https://docs.python.org/3/builtins/exceptions.html#FileNotFoundError) – if `corpus.jsonl`, `queries.jsonl` or the qrels
      file is missing.

### ef.ready_handle(array)

Wrap an already-computed array in a finished [`BatchHandle`](#ef.BatchHandle).

```pycon
>>> import numpy as np
>>> h = ready_handle(np.zeros((2, 3)))
>>> h.poll()
'done'
>>> h.result().shape
(2, 3)
```

* **Return type:**
  [`BatchHandle`](ef.embedders.html.md#ef.embedders.BatchHandle)

### ef.recall_at_k(ranked, relevant, k)

Fraction of the relevant docs that appear in the top `k` results.

```pycon
>>> recall_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)
0.5
>>> recall_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 3)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.reciprocal_rank(ranked, relevant, k=None)

Reciprocal of the rank of the first relevant result (`0` if none).

Averaged over queries this is MRR. `k` optionally caps how far down the
ranking to look.

```pycon
>>> reciprocal_rank(['x', 'd1', 'd2'], {'d1': 1.0})
0.5
>>> reciprocal_rank(['x', 'd1'], {'d1': 1.0}, k=1)       # relevant doc past the cutoff
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.refresh_on_change(manager)

Build a [`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent) handler that auto-refreshes `manager`.

The *auto* refresh seam. The returned callable is meant to be a
[`ChangeDetectingCorpus`](ef.corpus.html.md#ef.corpus.ChangeDetectingCorpus)’s `on_change` — each detected
edit (through the wrapper, or surfaced out-of-band by `scan()`) is
incrementally applied to every materialized config of `manager`:
an `added` source is indexed, a `modified` one is re-indexed
(delete-then-add), a `deleted` one is removed.

[`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) wires this automatically when
constructed with `auto_refresh=True`; call this directly only to attach a
manager to a corpus you build yourself.

* **Parameters:**
  **manager** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) (duck-typed — any
  object with an `_apply_change(event)` method).
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]
* **Returns:**
  a `Callable[[ChangeEvent], None]` for `on_change`.

### ef.rerank(query, segments, reranker, , limit=None)

Re-score `segments` against `query` with `reranker` and re-order them.

The pure heart of the module — no I/O, no retrieval. The `reranker` is
called once with the whole candidate list; the segments are returned sorted
by descending score (ties keep their incoming order), optionally trimmed to
`limit`. Each returned segment is a copy carrying the reranker’s score in
`metadata["rerank_score"]` — so the new ranking is auditable.

* **Parameters:**
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the search query the segments are scored against.
  * **segments** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]) – the candidate segments (from a first-stage retriever).
  * **reranker** ([`Reranker`](ef.reranking.html.md#ef.reranking.Reranker)) – a [`Reranker`](#ef.Reranker) — a callable `(query, segments) -> scores`.
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

### ef.segment_id(text, metadata=None)

Content-derived id for a segment — `sha256` of its normalized content.

The id is `sha256(normalize(text) + canonical_json(metadata))` — the
normalization and hashing primitives live in [`ef.hashing`](ef.hashing.html.md#module-ef.hashing), the single
source of truth `ef` content-addresses through. Because the id depends
only on content, re-segmenting an unchanged document produces identical ids
— which is what makes ingestion idempotent (a re-run is a no-op, not a
duplicate). Pass `metadata` only when two segments could share identical
`text` yet must stay distinct (e.g. same boilerplate from different
sources).

```pycon
>>> segment_id('hello') == segment_id('hello')
True
>>> segment_id('hello') == segment_id('world')
False
>>> segment_id('x', {'source': 'a'}) == segment_id('x', {'source': 'b'})
False
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ef.segment_record(x)

Coerce `x` (`str` / `Mapping` / [`SegmentRecord`](#ef.SegmentRecord)) into a record.

The dataclass counterpart of [`as_segment()`](#ef.as_segment) — the convenience-surface
constructor.

```pycon
>>> segment_record('hello').text
'hello'
>>> segment_record({'text': 'hi', 'tokens': 1}).tokens
1
```

* **Return type:**
  [`SegmentRecord`](ef.segments.html.md#ef.segments.SegmentRecord)

### ef.sentence_transformers_embedder(model_name='all-MiniLM-L6-v2', , device=None, normalize=False, batch_size=128, model_path=None, query_prompt=None, document_prompt=None, prompt_names=None, \*\*st_kwargs)

Build a local sentence-transformers [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Instruction-tuned models (bge, gte, e5-instruct, Qwen3-Embedding…) are
conditioned per `input_type`: pass a literal `query_prompt` (and, rarely,
`document_prompt`), or map input types to prompts the model defines itself
with `prompt_names` (e.g. `{"query": "query"}` for Qwen3-Embedding). Those
roles are then advertised in `honored_input_types`, and the prompt text is
baked into `model_id`. Only trust prompts on genuinely instruction-tuned
models; on `all-MiniLM` they are expected to be a near no-op. Prefer these
over wrapping the adapter in [`InstructionEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.InstructionEmbedder):
sentence-transformers knows the prompt’s length, so models that pool without
the prompt tokens (INSTRUCTOR, GRIT) stay correct. Needs
sentence-transformers >= 2.3.

* **Parameters:**
  * **model_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A sentence-transformers model name (also used in
    `model_id`).
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Torch device (`"cuda"` / `"cpu"` / `"mps"`); `None`
    auto-selects.
  * **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to L2-normalize outputs. sentence-transformers does
    **not** normalize by default — set this truthfully.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Encoding batch size (smaller on CPU).
  * **model_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A local filesystem path to load from instead of the hub —
    for air-gapped use. Loading still records `model_name` in
    `model_id`.
  * **query_prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prompt prepended to `input_type="query"` texts (passed
    as `encode(prompt=...)`).
  * **document_prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prompt for `input_type="document"` texts (usually
    none: the asymmetric pattern leaves documents bare). `""` means an
    explicit empty prompt, which stops a model’s `default_prompt_name`
    from being applied to documents.
  * **prompt_names** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `{input_type: name}` of prompts defined in the model’s
    own `prompts` config (passed as `encode(prompt_name=...)`).
    An explicit `prompt=` / `prompt_name=` in a call overrides both.
  * **\*\*st_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Forwarded to the `SentenceTransformer` constructor.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)

Requires `sentence-transformers` (`pip install 'ef[sentence-transformers]'`).

### ef.step_params(op, raw_params=None, , n_inputs=1)

The canonical *keyword* params of an op step — its inputs dropped.

An [`ef.ops`](ef.ops.html.md#module-ef.ops) op is called `op(*input_values, **params)`: the first
`n_inputs` positional parameters are upstream artifacts (they belong in
[`ProducerSpec.inputs`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec.inputs), not in
the params), and every remaining parameter is a hashable step param. This
runs [`full_kwargs()`](#ef.full_kwargs) with placeholder inputs, then drops those input
slots — leaving exactly the params, defaults filled, ready for a
[`TransformSpec`](#ef.TransformSpec).

* **Parameters:**
  * **op** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the op callable whose signature is the param contract.
  * **raw_params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the keyword params the caller actually supplied; anything
    omitted is defaulted.
  * **n_inputs** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – how many leading positional parameters are upstream inputs.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> def embed(segments, *, input_type='document'): ...
>>> step_params(embed, {'input_type': 'query'})
{'input_type': 'query'}
>>> step_params(embed)                              # default filled in
{'input_type': 'document'}
>>> def segment(source): ...
>>> step_params(segment)                            # no params at all
{}
```

### ef.token_f1(response, reference)

Token-overlap F1 of `response` against `reference` (the SQuAD F1).

The harmonic mean of token precision and recall over the normalized token
bags — a graded credit where [`exact_match()`](#ef.exact_match) is all-or-nothing.

```pycon
>>> token_f1('the quick brown fox', 'a quick brown fox')
1.0
>>> round(token_f1('quick brown fox', 'quick brown dog'), 4)
0.6667
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.voyage_embedder(model='voyage-3.5', , dim=None, api_key=None, batch_size=1000, timeout=60.0)

Build a Voyage AI [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Talks to Voyage’s `/v1/embeddings` REST endpoint directly — no
`voyageai` SDK, no extra dependency. Voyage embeddings are L2-normalized.
Voyage distinguishes only `query` from `document`; a `classification`
or `clustering` hint is simply not sent.

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A Voyage embedding model name.
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated width (the `voyage-3.5` family and
    `voyage-code-3` only). Omit for the model’s native width.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voyage API key. Falls back to `VOYAGE_API_KEY`.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at Voyage’s 1000 limit).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over the Voyage API.

### ef.with_overlap(segmenter, n_chars, /, \*, count_tokens=<function approx_token_count>, tokenizer='char-approx')

Wrap `segmenter` so consecutive segments overlap by `n_chars` characters.

Each segment is extended *forward* by up to `n_chars` characters (clamped
to the source length), so segment *K* shares its tail with segment *K+1*’s
head. A segment whose text changes has its `id` re-derived and its
`tokens` recounted; segments lacking `start` / `end` offsets pass
through untouched.

Overlap is expressed in *characters* here (this is a generic post-hoc
helper) — for token-measured overlap use a segmenter’s own `chunk_overlap`.

```pycon
>>> base = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
>>> [c['text'] for c in base('one two three four')]
['one two three', 'four']
>>> [c['text'] for c in with_overlap(base, 100)('one two three four')]
['one two three four', 'four']
```

* **Return type:**
  [`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter)

### ef.with_reranker(retriever, reranker, , fetch_k=50)

Decorate a retrieve-style callable with a two-stage reranking pass.

Wraps `retriever` (e.g. [`SourceManager.retrieve`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve), or any callable
`query -> list[Segment]`) so every call over-fetches `fetch_k`
candidates cheaply, [`rerank()`](#ef.rerank)s them, and returns the top `limit`.
The decorated callable keeps the retriever’s interface — `query` plus a
keyword-only `limit` — so it is a drop-in replacement, including as the
`retriever` argument of [`evaluate_retrieval()`](ef.evaluation.html.md#ef.evaluation.evaluate_retrieval).

* **Parameters:**
  * **retriever** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Segment`](ef.segments.html.md#ef.segments.Segment)]]) – the first-stage retriever — a callable
    `retriever(query, *, limit=...) -> segments`.
  * **reranker** ([`Reranker`](ef.reranking.html.md#ef.reranking.Reranker)) – the [`Reranker`](#ef.Reranker) applied to the over-fetched candidates.
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

### ef.write_beir(directory, corpus, queries, qrels)

Write an evaluation triple to disk in BEIR format — the inverse of [`read_beir()`](#ef.read_beir).

Creates `directory` (and parents) and writes `corpus.jsonl`,
`queries.jsonl` and `qrels.tsv`. Use it to snapshot an `ef` corpus as
a reusable benchmark (`ef_use_cases.md` §H4).

* **Parameters:**
  * **directory** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the output directory — created if absent.
  * **corpus** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the documents — `{doc_id: text}`.
  * **queries** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the queries — `{query_id: query_text}`.
  * **qrels** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – the relevance judgements — `{query_id: {doc_id: relevance}}`.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Modules

| [`artifact_graph`](ef.artifact_graph.html.md#module-ef.artifact_graph)         | The `ArtifactGraph` — `ef`'s content-addressed corpus-indexing core.                                                                                                                                                       |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`config`](ef.config.html.md#module-ef.config)                         | The config layer — declarative, content-addressable pipeline specs.                                                                                                                                                        |
| [`corpus`](ef.corpus.html.md#module-ef.corpus)                         | The `Corpus` abstraction — `ef`'s source layer (L0) — and change detection.                                                                                                                                                |
| [`diagnostics`](ef.diagnostics.html.md#module-ef.diagnostics)               | Staleness diagnostics — has an indexed corpus drifted from its source?                                                                                                                                                     |
| [`embedder_adapters`](ef.embedder_adapters.html.md#module-ef.embedder_adapters)   | Concrete [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) adapters and the DI seam.                                                                                                         |
| [`embedder_wrappers`](ef.embedder_wrappers.html.md#module-ef.embedder_wrappers)   | Composition wrappers for [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).                                                                                                                  |
| [`embedders`](ef.embedders.html.md#module-ef.embedders)                   | The `Embedder` facade — `ef`'s embed layer (L3).                                                                                                                                                                           |
| [`evaluation`](ef.evaluation.html.md#module-ef.evaluation)                 | Evaluation hookpoints — measuring retrieval and RAG quality.                                                                                                                                                               |
| [`exploration`](ef.exploration.html.md#module-ef.exploration)               | Layer L5 — *explore* a corpus: project to 2-D/3-D, cluster, label clusters.                                                                                                                                                |
| [`hashing`](ef.hashing.html.md#module-ef.hashing)                       | Content-hashing primitives shared across `ef`'s layers.                                                                                                                                                                    |
| [`ops`](ef.ops.html.md#module-ef.ops)                               | The pipeline ops — the callables an [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) runs.                                                                                   |
| [`refresh`](ef.refresh.html.md#module-ef.refresh)                       | Refresh — keeping an indexed corpus in sync as its sources change.                                                                                                                                                         |
| [`reranking`](ef.reranking.html.md#module-ef.reranking)                   | Reranking — a precision pass over retrieved segments (use case F6).                                                                                                                                                        |
| [`segmenter_adapters`](ef.segmenter_adapters.html.md#module-ef.segmenter_adapters) | The segmenter dependency-injection seam and the `imbed` registry bridge.                                                                                                                                                   |
| [`segmenters`](ef.segmenters.html.md#module-ef.segmenters)                 | The `Segmenter` facade — `ef`'s segment layer (L2).                                                                                                                                                                        |
| [`segments`](ef.segments.html.md#module-ef.segments)                     | The canonical `Segment` data model — `ef`'s interchange type for L2.                                                                                                                                                       |
| [`service`](ef.service.html.md#module-ef.service)                       | The HTTP-service bridge — [`EfService`](#ef.EfService), a stateless-friendly facade.                                                                                                         |
| [`source_manager`](ef.source_manager.html.md#module-ef.source_manager)         | The search facade — one-shot [`ingest()`](#ef.ingest), [`SearchableCorpus`](#ef.SearchableCorpus), [`SourceManager`](#ef.SourceManager). |
