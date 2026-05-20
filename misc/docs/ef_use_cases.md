# `ef` — User Journeys, Use Cases & Common Operations

> The **living catalogue** of what people do with semantic embeddings — and
> therefore what `ef` must make boilerplate-less. Add to this doc whenever a
> new journey is encountered. Items marked **★** are headline journeys worth a
> dedicated one-line facade affordance.
>
> Companion docs: [`ef_design_notes.md`](ef_design_notes.md) (the contracts),
> [`../../.claude/CLAUDE.md`](../../.claude/CLAUDE.md) (design constitution).

---

## A. The light path (must be ~1 line)

| # | Journey | Sketch |
|---|---|---|
| A1 ★ | **Search a small list of strings** | in-RAM, embed on the fly, no persistence, no config |
| A2 | Embed a small list | `embedder(["a","b"]) -> ndarray` |
| A3 | Quick "more like this" | given strings + a query, rank them |
| A4 | One-off RAG context fetch | `retrieve(query)` over an ad-hoc list → hand to an LLM |

The light path uses the default embedder + an in-memory `vd` backend + the
default recursive segmenter, all implicit.

## B. Corpus definition & indexing (the headline journey)

| # | Journey | Notes |
|---|---|---|
| B1 ★ | **Define a corpus** | from a `Mapping` to text sources (filesystem, S3, API, dict) |
| B2 ★ | **One-time index** | corpus assumed static — `corpus → segment → embed → index`; simplest path; `materialize` once |
| B3 ★ | **Ready search over a corpus** | the indexed corpus yields a `search(query) -> ranked Segments` object with zero boilerplate |
| B4 | Custom segmentation | swap a recursive / markdown / semantic / hierarchical / code-aware segmenter |
| B5 | Custom embedder | inject any provider or local model (see §D) |
| B6 | Bulk insert with metadata | batched embed-and-add, metadata attached |
| B7 | Code-corpus indexing | AST-aware splitting keeps functions/classes intact |
| B8 | Heavy corpus | huge data, multiple segmentations, multiple embedders, varied sources/targets — same facade, scaled config |

## C. Incremental update & refresh

| # | Journey | Notes |
|---|---|---|
| C1 ★ | **Explicit refresh** | `refresh(config)` — diff source hashes, `delete_cascade` changed/deleted, `materialize` new/changed |
| C2 ★ | **Auto-refresh** | two-stage: a cheap *detector* ("something changed" — file-watch, CDC token, mtime prefilter) triggers a *differ* ("what changed" — content-hash comparison) |
| C3 | Add sources (incremental upsert) | new `source_id`s; net-new artifacts, nothing invalidated |
| C4 | Edit a source | same id, new content hash → old segments+vectors replaced (transactional per source) |
| C5 | Delete a source | cascade-delete its segments/vectors/index entries; flag affected derived artifacts |
| C6 | Metadata-only update | change tags/title/ACL without re-embedding (metadata that doesn't enter chunking/embedding ⇒ zero recomputation) |
| C7 ★ | **Full rebuild from authoritative corpus** | "this batch is the entire current state" — anything absent is deleted |
| C8 | Scoped rebuild | reindex a subset, clean up disappeared segments within those sources only |
| C9 | Staleness audit | periodic job computing orphan / missing / stale / misconfigured |
| C10 | Orphan GC | `gc_orphans()` — delete unreachable artifacts |
| C11 | Rebuild-vs-incremental decision | switch to full rebuild when tombstone/stale fraction crosses a threshold |

Refresh modes (mirror LangChain): `none` (dedup only) · `incremental` (replace
changed, leave absent alone) · `full` (batch == whole corpus) · `scoped_full`.

## D. Embedding journeys

| # | Journey | Notes |
|---|---|---|
| D1 | Embed a query for search | `input_type="query"` — adapter applies the right prefix/task hint |
| D2 | Embed a document corpus | `input_type="document"`; auto-batched into provider-legal sizes |
| D3 ★ | **Bulk / offline embed** | `embed_batch` → `BatchHandle`; API backends hit the 50%-discount Batch API |
| D4 | Switch providers / models | same `Embedder` protocol; swap the adapter via `from_dict` |
| D5 | Choose dimensionality (MRL) | smaller dim = cheaper storage/faster ANN; `ef` warns on aggressive truncation recall loss |
| D6 | Two-stage retrieval | coarse ANN on truncated MRL vectors → rerank top-k on full dim |
| D7 | Quantized embeddings | request `int8`/`binary`; binary needs float rescoring |
| D8 ★ | **Local / offline / air-gapped embedding** | sentence-transformers / ONNX / llama.cpp / Ollama; no API key, no network; load model from local path |
| D9 | GPU batch embedding | length-sorted parallel encoding, FP16; multi-GPU |
| D10 | Quantized CPU embedding | ONNX/OpenVINO quantized model, thread-capped to physical cores |
| D11 | Edge / consumer-hardware embedding | llama.cpp/Ollama GGUF on a laptop / Apple Silicon |
| D12 | Decoupled embedding microservice | embedder served separately (TEI/infinity/BentoML/Triton); `ef` consumes over HTTP |
| D13 | Domain-specific embedding | route to a code/legal/finance model |
| D14 | Multilingual / long-document embedding | select a multilingual or long-context model; optional late chunking |
| D15 | Cache-backed embedding | `CachedEmbedder(inner, store)` — unchanged docs free |
| D16 | Resilient embedding | `RetryingEmbedder(inner, policy)` — backoff under rate limits |
| D17 ★ | **Re-embed on model change** | embedder swap invalidates 100% of vectors; namespace-bump forces a clean full re-embed |
| D18 | Cost / capacity estimation | inspect token usage before a big batch |

## E. Segmentation journeys

| # | Journey | Notes |
|---|---|---|
| E1 | Default recursive chunking | ~512 tokens, 10–20% overlap — the zero-config default |
| E2 | Structure-aware chunking | markdown/HTML header splitting; page-level for paginated docs |
| E3 | Semantic chunking | breakpoints where adjacent-sentence similarity drops (needs an injected embedder) |
| E4 | Late chunking | embed full doc with a long-context model, pool per-segment — preserves cross-segment coreference |
| E5 | Hierarchical / parent-child chunking | small segments for retrieval precision + large for LLM context |
| E6 | Swap segmenter without touching call sites | `registry["name"](doc)` — engine-agnostic |
| E7 | Evaluate chunking quality | IoU-style recall metrics (requires preserved offsets) |

## F. Search & RAG-plug-in journeys

| # | Journey | Notes |
|---|---|---|
| F1 ★ | **Semantic search over a corpus** | `search(query, limit, filter)` → ranked segments |
| F2 | Filtered search | metadata filters combined with vector similarity |
| F3 | Multi-query search + RRF | several query phrasings, fused (delegates to `vd`) |
| F4 | Hybrid search | dense + lexical, RRF-fused (delegates to `vd`) |
| F5 ★ | **RAG-plug-in readiness** | `ef` returns `retrieve(query) -> list[Segment]` for an external RAG/agent framework — `ef` does NOT synthesize answers |
| F6 | Reranking | apply a reranker as a decorator over a base retriever |
| F7 ★ | **Retrieval evaluation** | `evaluate_retrieval(...)` — BEIR/MTEB-shaped, NDCG@10 / Recall@100 |
| F8 | End-to-end RAG evaluation | `evaluate_rag(samples, metrics=...)` — Ragas `SingleTurnSample`-shaped rows |

## G. Explore / visualize a corpus (layer L5 — secondary surface)

| # | Journey | Notes |
|---|---|---|
| G1 ★ | **Project a corpus to 2D/3D** | PCA → UMAP (cosine, seeded) — `ef.project(corpus, dims=2|3)` |
| G2 | Cluster a corpus | HDBSCAN/k-means on reduced or full vectors |
| G3 | Auto-label clusters | LLM-titled clusters (`imbed`'s `ClusterLabeler`) |

> This is `ef`'s visualization heritage from `imbed`. Keep it as a *secondary*
> "explore the corpus" capability — `app_ef` consumes it — not as "the pipeline".

## H. Interoperability & persistence

| # | Journey | Notes |
|---|---|---|
| H1 | Round-trip a segment to/from another framework | `to_langchain`/`from_langchain`, LlamaIndex, Haystack |
| H2 | Drop an `ef` corpus into a LangChain pipeline | adapter shim |
| H3 | Export a corpus to interchange format | Parquet (Arrow `FixedSizeList<Float32>`) / HF `datasets` |
| H4 | Export to BEIR shape | `corpus.jsonl` / `queries.jsonl` / `qrels.tsv` for benchmarking |
| H5 | Deploy a corpus as a service | one-liner → FastAPI (`qh`) / MCP server / Streamlit UI |
| H6 | Trace any operation | `embed`/`search`/`segment`/`retrieve` emit OpenTelemetry GenAI spans |

## I. Config-branching & experimentation

| # | Journey | Notes |
|---|---|---|
| I1 ★ | **Config branching** | register a second config differing only in segmenter/embedder/index params — upstream artifacts shared, only the divergent cone computed |
| I2 | A/B test embedders / configs | two config leaves served with a traffic split |
| I3 | ANN re-tune | change HNSW `M`/`efConstruction` — rebuild index only, reuse all vectors |
| I4 | Multi-vector retrieval | same segments, multiple embedders (dense + sparse + late-interaction) |
| I5 | Provenance query | `lineage(key)` — "what produced this vector?" / "what depends on this source?" |

---

## The four staleness conditions (`ef`'s diagnostics contract)

| Condition | Definition | Detection |
|---|---|---|
| **Orphan** | vector exists; source gone | `vectors.source_id − sources.source_id` |
| **Missing** | source exists; no vector | `sources.source_id − vectors.source_id` |
| **Stale** | source changed since vector computed | `vector.source_hash != source.current_hash` |
| **Misconfigured** | vector produced with wrong embedder/segmenter for the current config | `vector.config_hash != current_config_hash` |
