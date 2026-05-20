# CLAUDE.md — `ef` (Embedding Flow)

> Design constitution for refactoring/implementing `ef` to a highly functional,
> robust state. Distilled 2026-05-20 from deep research — full notes in
> [`misc/docs/ef_design_notes.md`](../misc/docs/ef_design_notes.md), use cases
> in [`misc/docs/ef_use_cases.md`](../misc/docs/ef_use_cases.md). When developed
> as part of the embeddings package group, see also that group's
> `semantic_search_design_notes.md` for cross-package architecture.

---

## 1. What `ef` is meant to become

`ef` ("Embedding Flow") is a **facade for boilerplate-less,
semantic-embeddings-based user journeys** — semantic search, corpus indexing,
and RAG-plug-in readiness. The defining requirements:

- **Cover the light case lightly** — a small list of strings, all in RAM,
  vectorize on the fly, semantic search in one or two lines.
- **Cover the heavy case too** — huge corpora, multiple segmentations, multiple
  embedding functions, varied data sources and targets (vectorDBs) — with the
  **same facade**, via progressive disclosure.
- **Use `vd`** as its interface to vector databases.
- **Plug in RAG** — make a corpus easy to hand to an external RAG/agent
  framework. `ef` is *not* RAG: no answer synthesis, no agents (see §6).
- **The corpus journey** — define a corpus from a `Mapping` to text sources
  (filesystem, etc.), define segmentation→vectorization pipelines over it, get
  ready search. Support one-time indexing (corpus static — simplest path) AND
  updates: explicit refresh, and automatic refresh via change-detection.

Progressive disclosure + dependency injection are the two non-negotiable design
principles.

## 2. Current state vs. target — this is a real refactor

> **The refactor is complete (Phase 8, ef#18).** This section is the historical
> framing — `ef` *is* the target facade now; kept for context.

**Current `ef`** (~1000 LOC) is a *prototype* of an embedding **visualization
pipeline** inherited from `imbed`: `Project`/`Projects`, `ComponentRegistry`,
the mall pattern, a simple DAG, and the `segment → embed → planarize → cluster`
stages. It has **no search, no corpus abstraction, no vectorDB integration, no
incremental refresh**. It does not use `vd`.

**Target `ef`** is the semantic-search/RAG facade above. The refactor:

- **Primary surface becomes search / RAG / corpus-indexing.** `segment → embed
  → planarize → cluster` becomes *one secondary use case* — "explore a corpus"
  (layer L5) — not "the pipeline". Don't delete the viz code; demote it.
- Keep the good bones: `Project` facade, `MutableMapping`-everywhere, the mall
  pattern, plugin/registry idea, the `imbed` plugin bridge.
- Add the new spine: `Embedder` protocol, `Segmenter` facade, `Corpus`
  abstraction, the `ArtifactGraph` indexing core, the "ready search" object.

`ef` is **free to change** (no users) — refactor, rename, break freely.

## 3. The layered model (the spine)

```
L0 Sources    Corpus = MutableMapping[source_id, Source]   (dol store: fs/S3/API/RAM)
L1 Parse      pluggable parser (text extraction)
L2 Segment    Segmenter facade (chunkers) — built on imbed.components.segmentation
L3 Embed      Embedder facade — wraps imbed + provider/local adapters
L4 Index      vd.Collection  (ef writes; vd owns)
L5 Derive     planarize / cluster / label  ("explore the corpus")
──────────────────────────────────────────────────────────────────────
   Search     "ready search" object: search(query) -> ranked Segments
   RAG plug   retrieve() handed to an external LLM/agent framework
```

## 4. Core contracts

### 4.1 `Embedder` — a batch callable, structurally typed
An embedder is fundamentally `Iterable[str] -> ndarray(n, dim)`. Make the simple
case trivial: `embedder(["hello","world"]) -> ndarray(2, dim)`.

- `@runtime_checkable Protocol`, **not** an ABC. Callable + ~4 metadata attrs:
  `model_id` (`"openai:text-embedding-3-large@1024"` — provider:model@dim),
  `dim`, `normalized`, and which `input_type`s it honors.
- **Batch-first.** Single-string embedding is a trivial wrapper.
- Canonical task vocabulary: `InputType = Literal["query","document",
  "classification","clustering"]` — translate to each vendor's name
  (`input_type`/`task_type`/`task`/`prompt_name`) at the adapter boundary. This
  is `ef`'s single most valuable normalization.
- Composition wrappers (each takes an inner `Embedder`): `CachedEmbedder(inner,
  store)`, `RetryingEmbedder(inner, policy)`, `MultiEmbedder(routes, predicate)`,
  `NormalizingEmbedder(inner)`.
- Adapters as lazily-imported factory functions: API providers (OpenAI, Cohere,
  Voyage, Gemini) and local runtimes (sentence-transformers, ONNX/Optimum,
  llama.cpp/Ollama, `http_embedder` for TEI/infinity). Heavy deps → extras.
- Local embedder = a batch callable too; the three runtime families are adapter
  factories. **Length-sorted batching** (sort by length, batch, un-sort back to
  caller order) is `ef`'s job to own — never push padding/sorting onto users.

### 4.2 `Segmenter` — a callable, streaming-first
`Segmenter` = `Callable[[str | Mapping], Iterable[Segment]]`. `@runtime_checkable
Protocol`. Return an **iterable**, not a list (streaming serves the heavy case).
Stateful segmenters (semantic, late chunking) **inject** their embedder — never
a global. Build on `imbed.components.segmentation` (registry + `SegmentStore`
keyed by `(doc_key, start, end)`) rather than duplicating it. Default segmenter
= recursive character splitting @ ~512 tokens, 10–20% overlap. Standardize on
the word **"segment"** (not "chunk") — `imbed` already uses it.

### 4.3 `Corpus` — `MutableMapping[source_id, Source]`
A corpus is a `dol`-style mapping; the backing store (fs/S3/API/RAM) is
swappable. Change detection is a **wrapper** that hooks `__setitem__`/
`__delitem__`, content-hashes the value, and fires invalidation on hash change.

### 4.4 `ArtifactGraph` — the indexing & refresh core
The corpus-indexing engine is a **content-addressed artifact graph** (Reports
06+07). `artifact_id = H(op, op_version, inputs, params)`. Two stores (a
content-addressed cache + a producer graph) and four operations: `materialize`
(lazy backward), `mark_stale` / `delete_cascade` (forward), `freshness`.

**Cascade invalidation and config branching are the same operation** — do not
build them separately. One-time index = `materialize` once. Refresh = diff
source hashes → `delete_cascade` changed leaves → `materialize`. Config
branching (a second embedder/segmenter) = free, shares upstream artifacts.

The public API exposes **sources and configs only** — no public `add_segment`/
`add_vector`. See `ef_design_notes.md` §3 for the full `SourceManager` contract.

### 4.5 The facade — two-class progressive disclosure
Thin top class (`Project`/`Corpus`-style) for the 90% case; `Embedder`,
`Segmenter`, `Corpus`, `ArtifactGraph` all usable standalone. A one-shot
`ingest(sources, *, segmenter, embedder, store, cache)` callable as a single
function — not a four-component DAG the user must wire.

## 5. Canonical data model

Field names: `text`, `metadata`, `id`, `embedding` (not `page_content`/
`content`). Required: only `id` + `text`. `parent_id` / `chunk_idx` are
**promoted top-level fields**, not metadata. `TypedDict` is the interchange
type; a frozen dataclass / Pydantic v2 model is the convenience surface.
Promoted metadata keys that must round-trip: `source`, `source_type`,
`token_count`, `tokenizer`, `embedding_model`, `page`, `license`. **Record the
tokenizer with every segment** — omitting it is "the single most common silent
bug in chunking pipelines". Default IDs = content-derived
(`sha256(nfc(text) + canonical_json(promoted_metadata))`) for idempotent
ingestion.

## 6. Boundaries — what `ef` must NOT do

`ef` is a **facade, not a framework**. Adding orchestration compromises it.

- **Inside `ef`:** corpus/segment/embedder/vector schemas; `Embedder`/
  `Segmenter` protocols; one-shot `ingest`; the `ArtifactGraph` + per-step
  caching; "ready search" returning ranked segments; `retrieve()`; evaluation
  hookpoints; OTel spans; thin `to_X`/`from_X` adapters.
- **Outside `ef`:** agent loops, tool-calling, memory/conversation managers,
  prompt templating, structured-output coercion, **LLM answer synthesis**.
  "Bring your own LLM, your own agent framework, your own UI." The RAG-plug-in
  surface = `ef` hands back a clean `retrieve(query) -> list[Segment]`; the
  application (or `srag`/`raglab`/LangGraph) synthesizes answers.
- **No global config singleton.** A process-wide mutable `ComponentRegistry`
  used as global state would replay LlamaIndex's deprecated `ServiceContext`.
  Inject explicitly.

## 7. Ecosystem wiring

- **`vd`** — `ef`'s vectorDB interface. `ef` writes layer L4 into `vd.Collection`s;
  `vd` owns the index. `ef` stores `source_hash`/`config_hash` in `vd` metadata
  so staleness conditions are filtered queries.
- **`imbed`** — heavy implementations. `ef` defines the protocols and wraps
  `imbed`'s embedders/segmenters/planarizers/clusterers via structural typing.
  Reuse, don't reimplement. `imbed` interface-preserving additions are always
  welcome; breaking changes need review first (group policy).
- **`qh`** — wrap `ef` functions into FastAPI HTTP services for `app_ef`.
- **`ju`** — OpenAPI-spec parsing for the `ef`↔`app_ef` bridge.
- **`raglab` / `srag`** — consumers of `ef`'s `retrieve()` surface.
- **`dol`** — every store (corpus, cache, producer graph, segment store) is a
  `MutableMapping`; backends swappable.

## 8. Refactor roadmap (suggested phases)

**Status:** all 8 phases are implemented (ef#3 · #5 · #7 · #9 · #11 · #14 · #16
· #18, plus the zero-install default embedder ef#13). **The refactor is
complete** — `ef` *is* the search / RAG / corpus-indexing facade described above.

1. **`Embedder` protocol + composition wrappers + 2 adapters** (OpenAI,
   sentence-transformers). `as_embedder(x)` normalizer (str/callable/url).
2. **`Segmenter` facade** on top of `imbed.components.segmentation`; default
   recursive splitter; canonical `Segment` schema.
3. **`Corpus` abstraction** + `ChangeDetecting` wrapper.
4. **`ArtifactGraph` core** (`ef/artifact_graph.py`) — SQLite dependency graph;
   `materialize`/`delete_cascade`/`mark_stale`/`freshness`.
5. **"Ready search" + one-shot `ingest`** wiring corpus→segment→embed→`vd`.
6. **Refresh** — explicit + auto (the "something changed" detector + "what
   changed" differ); the four staleness conditions; refresh modes
   (`none`/`incremental`/`full`/`scoped_full`, mirroring LangChain). *Done:*
   `ef/diagnostics.py` (`diagnose`, `StalenessReport`), `ef/refresh.py`
   (`plan_refresh`, `RefreshReport`, `refresh_on_change`), and the
   `SourceManager` surface `diagnose`/`refresh`/`rebuild`/`gc_orphans`/
   `lineage`/`scan` + `auto_refresh=True`.
7. **RAG-plug-in surface** (`retrieve()`) + evaluation hookpoints
   (`evaluate_retrieval` BEIR-shaped, `evaluate_rag` Ragas-shaped). *Done:*
   `retrieve()` reconciled to return plain `Segment`s (provenance folded into
   `metadata["source"]` by `hits_to_segments`); `ef/evaluation.py`
   (`evaluate_retrieval` + NDCG@10/recall/precision/MRR/MAP primitives,
   `evaluate_rag` + deterministic lexical metrics, `read_beir`/`write_beir`,
   `as_ragas_dataset` bridge); `ef/reranking.py` (`Reranker` protocol, `rerank`,
   `with_reranker` decorator, lazy `cross_encoder_reranker`).
8. **Demote viz to "explore"** — keep planarize/cluster/label as L5. *Done:*
   `ef/explore.py` — `project` (PCA→UMAP, cosine, seeded), `cluster` (numpy
   k-means / HDBSCAN), `label_clusters` (`imbed`'s `ClusterLabeler`); numpy-only
   import, heavy deps lazy. The viz-era prototype (`Project`/`Projects`/
   `ComponentRegistry`/`mall`/`dag.py`/`plugins/`) was deleted.

## 9. Conventions

Follow the user's global CLAUDE.md (functional > OOP, keyword-only after 3rd
arg, module docstrings, doctests, progressive disclosure, smart defaults).
Naming: `text`/`metadata`/`id`/`embedding`; `Segment`/`Segmenter`/`Embedder`/
`Corpus`; `ingest`/`retrieve`/`search`/`materialize`; `InputType` for embed
roles; `artifact_id`/`ProducerSpec`/`PipelineSpec`/`ConfigId`. When working on
`ef`, load the **`ef-architecture`** dev skill in
[`.claude/skills/ef-architecture/`](skills/ef-architecture/SKILL.md).
