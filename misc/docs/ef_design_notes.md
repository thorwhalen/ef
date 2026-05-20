# `ef` — Design Notes (research distillation)

> Deep, actionable notes for the `ef` refactor. Distilled 2026-05-20 from the
> research reports in `embeddings/docs/research/semantic_search/` (docs 01, 02,
> 04, 05, 06, 07, 09, 10). Companion: [`ef_use_cases.md`](ef_use_cases.md),
> [`../../.claude/CLAUDE.md`](../../.claude/CLAUDE.md).

---

## 1. The Embedder facade

### 1.1 The contract
Every embedder — API or local — is fundamentally `Iterable[str] -> ndarray(n,
dim)`. Everything else (batching, async, normalization, task hints, caching) is
*layered scaffolding*. `Embedder` is a tiny `@runtime_checkable Protocol`, not
an ABC — a callable + ~4 introspectable metadata attributes.

```python
from typing import Protocol, Iterable, Literal, runtime_checkable
import numpy as np

InputType = Literal["query", "document", "classification", "clustering"]

@runtime_checkable
class Embedder(Protocol):
    """Iterable[str] -> ndarray of shape (n, dim). SSOT metadata on `self`."""
    model_id: str        # "openai:text-embedding-3-large@1024" — provider:model@dim
    dim: int
    normalized: bool     # True if ||v|| == 1 by construction
    honored_input_types: tuple[InputType, ...]

    def __call__(self, texts: Iterable[str], *,
                 input_type: InputType | None = None,
                 **backend) -> np.ndarray: ...

    def embed_batch(self, texts: Iterable[str], *,
                    input_type: InputType | None = None,
                    **backend) -> "BatchHandle": ...
```

### 1.2 Key decisions
- **`model_id` bakes in everything that changes the vector** — provider, model,
  dim — so it works directly as a cache namespace. MRL dim is part of identity:
  construct `OpenAIEmbedder("text-embedding-3-large", dim=512)` at adapter
  construction, not per-call.
- **Standardize task semantics** — the field's single most valuable
  normalization. Every provider names the same `(query, document,
  classification, clustering)` distinction differently (`input_type` /
  `task_type` / `task` / `prompt_name`). Define ONE canonical `InputType`,
  translate at the adapter boundary. Backends that ignore the hint (OpenAI)
  advertise an empty honored-set.
- **Batch-and-poll = a special case of async** via a `BatchHandle`
  (`poll() -> "pending"|"done"|"failed"`, `result() -> ndarray`, `cancel()`).
  Synchronous backends return a *ready* handle. Collapses OpenAI/Cohere/Vertex
  Batch APIs and the trivial in-RAM case into one shape.
- **Backend-specific options behind `**backend` kwargs on the concrete adapter**
  — never in the core protocol signature.
- **Normalization must be explicit and known.** Defaults differ wildly (OpenAI
  L2 always; sentence-transformers OFF by default; Gemini NOT normalized below
  native dim — a real footgun). Expose `normalized` truthfully; offer a
  `NormalizingEmbedder` wrapper for MRL-truncated vectors.
- **No streaming** — an embedding is a fixed-size vector, not a token stream.

### 1.3 Composition wrappers (each takes an inner `Embedder`)
- `CachedEmbedder(inner, store)` — `store: MutableMapping[str, ndarray]`. Cache
  key pins everything that affects the vector:
  `sha256(f"{model_id}|{dim}|{normalize}|{input_type}|{text}")`. Use SHA-256.
  Cache document embeddings always; query embeddings on opt-in.
- `RetryingEmbedder(inner, policy)` — exponential backoff + jitter; distinguish
  per-request 400s (skip) from 429/5xx (retry).
- `MultiEmbedder(routes, predicate)` — route by a predicate over `(text,
  metadata)` (domain-specific models).
- `NormalizingEmbedder(inner)` — renormalize MRL-truncated outputs.

### 1.4 Model-version invalidation
There is NO community standard for invalidating embeddings on model upgrade.
Use `namespace = f"{model_id}@{version}@{dim}"`. Silent re-embedding on version
mismatch is a footgun — retrieval recall against an index built at version *n*
is undefined when queries are embedded at *n+1*. Force explicit invalidation.

### 1.5 Local / on-premises embedders
A local embedder is *also* just a batch callable: `Callable[[Sequence[str]],
np.ndarray]`. The three runtime families are **adapter factories** that produce
one — all lazily imported, heavy deps in extras:

| Adapter | Backs onto | When |
|---|---|---|
| `sentence_transformers_embedder` | PyTorch | default; GPU available; flexible |
| `onnx_embedder` / `openvino_embedder` | HF Optimum (ONNX RT / OpenVINO / TensorRT) | production CPU microservice; quantized |
| `llama_cpp_embedder` / `ollama_embedder` | GGUF / ggml C++ | edge, consumer hardware, Apple Silicon, no-Python |
| `http_embedder(url)` | any remote service (TEI, infinity, BentoML, Triton) | decoupled microservice |

**Injection ladder** (`as_embedder(x)` is the single DI seam):
1. string shorthand — `Ef(embedder="all-MiniLM-L6-v2")`
2. adapter factory — `Ef(embedder=onnx_embedder("all-mpnet-base-v2"))`
3. bare callable — `Ef(embedder=my_func)`
4. remote — `Ef(embedder=http_embedder("http://localhost:8080/embed"))`

**Throughput rules `ef` must own (never push onto users):**
- **Length-sorted batching is mandatory** — self-attention is O(n²) in sequence
  length; padding to the longest batch member wastes cycles. Sort by length,
  batch, **un-sort embeddings back to caller order**. 5–10× speedup.
- Batch size is a keyword-only parameter (default ~128 GPU, smaller CPU).
- GPU: FP16/BF16, `attn_implementation="sdpa"`, keep tensors on device.
- CPU: cap threads to *physical* cores — over-subscription degrades throughput.
- Multi-GPU: GPU-0 holds weights (~70–80% VRAM) before inference → symmetric
  batches OOM GPU-0; use asymmetric allocation or exclude GPU-0.
- **Air-gapped first** — every adapter accepts a local filesystem model path;
  never require hub access at embed time.
- **Pin embedder identity to the collection** — store `model_id`+`dim` as `vd`
  collection metadata; mismatched-model query is a silent correctness bug.

## 2. The Segmentation facade

### 2.1 The contract
A segmenter is a **callable**: `Callable[[str | Mapping], Iterable[Segment]]`.
`@runtime_checkable Protocol`, not an ABC. Streaming-first (`Iterable`, not
`list`) — no competitor offers this; it serves the heavy case directly.
Implementations may be stateful (hold an embedder/LLM) but must be re-entrant.

```python
@runtime_checkable
class Segmenter(Protocol):
    def __call__(self, doc: str | Mapping[str, Any], /) -> Iterable[Segment]: ...

@runtime_checkable
class BatchedSegmenter(Protocol):   # optional optimization trait
    def batch(self, docs: Iterable[...], /) -> Iterable[Iterable[Segment]]: ...
```

### 2.2 Key decisions
- **Build on `imbed.components.segmentation`** (registry + `SegmentStore` keyed
  by `(doc_key, start, end)`) — don't duplicate it. `SegmentStore`'s tuple
  keying makes incremental refresh natural (re-key only changed docs).
- **Stateful segmenters inject their embedder/LLM** — never a global. Late
  chunking = "a normal segmenter emitting spans + a special embedder".
- **Default segmenter** = recursive character splitting @ ~512 tokens, 10–20%
  overlap. Document it as "good enough; upgrade only if you can measure the win."
- **Preserve byte/character offsets** (`start`/`end`) through every adapter so
  chunking-quality evaluation (IoU metrics) stays possible.
- Composition helpers: `with_overlap(seg, n)`, `hierarchical([seg1, seg2])`,
  `materialise(seg)` (streaming → list).
- Six lazily-imported adapters cover most production usage: LangChain,
  LlamaIndex, Haystack, Unstructured, Chonkie, semantic-text-splitter.
- **Standardize on the word "segment"** (not "chunk") — `imbed` already uses it.

### 2.3 Chunking landscape facts
- `RecursiveCharacterTextSplitter` @ ~400–512 tokens, 10–20% overlap is within
  1–2 points of every elaborate method on realistic corpora — the honest default.
- **Embedding-model quality dominates chunking-strategy effects** — don't
  over-invest in fancy chunkers before settling the embedder.
- **Document structure beats algorithm cleverness** — markdown-aware splitting
  lifts accuracy 5–10% on structured docs; page-level won NVIDIA's benchmark.
- Semantic chunking is *conditional*, not a default. Late chunking wins when
  segments need long-distance entity context (anaphora) + a long-context model.
- The algorithm space is **not converging on a winner** → segmenter choice must
  be dependency-injected and swappable.

## 3. The corpus-indexing core — the `ArtifactGraph`

### 3.1 The thesis
> **Cascade invalidation and configuration branching are the same operation.**
> A source edit changes a leaf hash; a param change changes a leaf hash. Both
> produce a new `artifact_id` whose downstream cone needs (re-)materialization.
> The graph does not distinguish them. "Incremental refresh", "experiment
> tracking", "hot reload", "cache invalidation", "reactivity" are five names for
> one operation.

A "pipeline" is the wrong abstraction — it is a **declared, introspectable
producer graph** (DAG). Multiple configs = multiple paths through *one* graph,
sharing every upstream node where `(op, params)` agree (each segment computed
once, each vector once).

### 3.2 Artifact levels & content addressing
`artifact_id = H(op, op_version, inputs, params)`. Levels: L0 Source → L1 Parsed
→ L2 Segment → L3 Vector → L4 Index → L5 Derived. Contracts:
- **Hash bytes, never timestamps.** mtime/etag only as a cache *prefilter*.
- **Normalize before hashing** — Unicode NFC, strip line-ending/BOM variance,
  canonical (sorted-key) JSON for structured docs. SHA-256.
- **Params are inputs** — `chunk_size`, an env var, an embedder name all
  participate in `H()` like input bytes. (DVC `{cmd, deps, params, outs}`.)
- **Every `op` carries a version that participates in the hash.** A library
  upgrade that silently changes chunking without an `op_version` bump is a
  *correctness violation*.
- **Don't hash metadata that doesn't affect L1–L4** — a title/tag change must
  not trigger a spurious full re-embed.

### 3.3 The four operations & the `SourceManager` public API
Four operations: `materialize(key)` (lazy backward), `mark_stale(key)` /
`delete_cascade(key)` (forward), `freshness(key)`. The public API exposes
**sources and configs only** — no public `add_segment`/`add_vector`:

```python
class SourceManager(Protocol):
    sources: MutableMapping[SourceId, Source]       # the public surface

    def register_config(self, name, pipeline: PipelineSpec) -> ConfigId: ...
    def materialize(self, config: ConfigId, *, sources=None) -> MaterializeReport: ...
    def orphans(self, config) -> Iterable[ArtifactKey]: ...
    def missing(self, config) -> Iterable[SourceId]: ...
    def stale(self, config) -> Iterable[ArtifactKey]: ...
    def misconfigured(self, config) -> Iterable[ArtifactKey]: ...
    def gc_orphans(self, config=None) -> int: ...
    def rebuild(self, config, *, level="index"|"vectors"|"all") -> RebuildReport: ...
    def lineage(self, key) -> Iterable[ArtifactKey]: ...
```

### 3.4 Parameter-change invalidation (asymmetric — `ef` must distinguish)
| Parameter | Invalidates | Cost |
|---|---|---|
| Parser swap | L1 onward | high |
| Segmenter change | L2 onward | high |
| Embedder identity/version | L3 onward | **most expensive — $$ + time** |
| Distance metric | L4 only | low |
| HNSW `M`/`efConstruction` | L4 only | medium |
| HNSW `efSearch` | nothing (query-time knob) | zero |

### 3.5 Suggested module structure
`ef/artifact_graph.py` (graph + 4 ops) · `ef/corpus.py` (`Corpus` +
`ChangeDetectingCorpus`) · `ef/config.py` (`PipelineSpec`, `TransformSpec`) ·
`ef/change_detection.py` (hashing, mtime prefilter, watchers, CDC watermarks) ·
`ef/refresh.py` (explicit/auto refresh, rebuild-vs-incremental heuristics) ·
`ef/source_manager.py` (the facade) · `ef/diagnostics.py` (4 staleness
conditions, audit). Start the dependency graph in **SQLite** (~10⁷ nodes,
single-process); migration to Postgres is mechanical.

### 3.6 Change detection mechanisms
Content hashing (dominant); mtime/etag (prefilter only); SaaS change-feeds with
**CDC-watermark** (Notion `last_edited_time`, Drive `pageToken`, GitHub
`compare`) — advance the watermark only after successful downstream processing;
`watchdog` for local files (debounce 1–5 s; atomic-save shows as delete+create).

### 3.7 Re-embedding patterns (embedder change ⇒ always a full re-embed)
Dual-write / blue-green collection swap (2× cost during migration, instant
rollback); collection aliases (Weaviate/Elasticsearch — atomic repoint);
namespace-bump encoding embedder identity (lowest friction). Partial
re-embedding is almost never mathematically valid.

### 3.8 Building a `ProducerSpec` from an op call (the `i2.Sig` recipe)
`artifact_graph.py` (Phase 4) lands `ProducerSpec` as a *fully serializable*
node: `op` is a **string key**, not a `Callable` (a raw callable can't be
persisted, and §3.5 wants the graph in SQLite). The `ArtifactGraph` resolves
the key to a function through an injected `ops` registry at `materialize` time.
`artifact_id` then content-addresses the spec: `sha256(canonical_json({op,
op_version, inputs, params}))` (`canonical_json`/`sha256_hex` from
`ef/hashing.py` — the SSOT).

For content addressing to be *correct*, two semantically equal op calls must
produce **identical** `params` — so `op(text)` and `op(text, size=512)` (512
being the default) must hash the same. Normalize a call into a stable,
fully-named, defaults-filled kwargs dict with the canonical `i2` idiom:

```python
from i2 import Sig

def full_kwargs(op, args=(), kwargs=None):
    """Call (args, kwargs) -> a canonical, defaults-filled, fully-named dict."""
    return Sig(op).map_arguments(args, kwargs or {}, apply_defaults=True)
```

`Sig.map_arguments` is the current API (`kwargs_from_args_and_kwargs` is a
deprecated alias). The *input* positional args (upstream artifact ids) are
separated out into `ProducerSpec.inputs`; the remaining keyword args become
`params`. Phase 5 (the config layer / `source_manager.py`) is where calls are
turned into specs and this recipe is applied — Phase 4's `ArtifactGraph` takes
specs as given. Full detail in the `i2-signatures` skill.

## 4. Canonical data model

Field names: `text`, `metadata`, `id`, `embedding` (NOT `page_content` /
`content`). Required: only `id` + `text`. `parent_id` / `chunk_idx` are
**promoted top-level fields**, not metadata. Hierarchy is opt-in via
`parent_id`/`children_ids` pointers — a flat list with ID parent pointers, never
a first-class structural field.

```python
class Segment(TypedDict, total=False):
    text: str                       # required
    id: str                         # required by convention; sha256 of normalized text
    parent_id: str                  # source document or parent segment id
    children_ids: list[str]
    start: int                      # char offset into the source
    end: int
    index: int                      # ordinal in the segmenter's output
    tokens: int
    metadata: Mapping[str, Any]     # framework-specific keys live here, namespaced
```

Promoted metadata keys that must round-trip every adapter: `source`,
`source_type`, `token_count`, `tokenizer`, `embedding_model`, `page`, `license`,
`ingestion_run_id`. **Record the tokenizer with every segment** — omitting it is
"the single most common silent bug in chunking pipelines". Default IDs =
content-derived: `sha256(nfc(text) + canonical_json(promoted_metadata))` for
idempotent ingestion. Public type = Pydantic v2 (validation, JSON-schema
export); cheap path = frozen dataclass / `TypedDict`. The type is passive —
adapters are free functions.

## 5. Frameworks — what to imitate / avoid

**Imitate:** RAGatouille's two-class progressive-disclosure facade; LlamaIndex's
`IngestionPipeline` per-step caching + dedup and typed `relationships`; Haystack
v2's typed components + build-time validation (advanced path only); Cohere's
`input_type` flag.

**Avoid:** a global config/`Settings` singleton (LlamaIndex deprecated theirs in
14 months); deep chain class hierarchies (LangChain rewrote 3× in 2 years);
`**kwargs` tails on public methods; silent doc/query embedding symmetry (the
Bedrock-Cohere bug); bundling orchestration (agents/memory/tools — "now we are
LangChain"); a bundled UI (that is `app_ef`'s job); document-loader sprawl (let
users pull `pypdf`/`unstructured`/`firecrawl`).

**Positioning (adapt for `ef`):**
> We are not LangChain — no orchestration layer. Not LlamaIndex — no global
> config, no QueryEngine. Not Haystack — no pipeline DSL. Not txtai — we don't
> bundle the database. We are a facade over the modern semantic-search pipeline:
> corpus, segmenter, embedder, vector store, retriever — with progressive
> disclosure and OTel observability. Bring your own LLM, agent framework, UI.

## 6. Conventions to adopt below the control surface
- **Storage:** Parquet with Arrow `FixedSizeList<Float32, dim>` + scalar
  metadata columns is the portable interchange format; `.npy` for offline dumps.
- **Evaluation:** BEIR triple (`corpus.jsonl`/`queries.jsonl`/`qrels.tsv`),
  primary metric NDCG@10; Ragas `SingleTurnSample` `(user_input, response,
  retrieved_contexts, reference)` for RAG eval.
- **Observability:** OpenTelemetry GenAI semantic conventions; use OpenInference
  span kinds `RETRIEVER`/`EMBEDDING`/`RERANKER` (OTel GenAI lacks them). Don't
  depend on the OTel SDK — `get_tracer(__name__)` is a no-op if unconfigured.

## 7. Embedding-model landscape facts (for model-selection guidance)
- Decoder-only LLM-derived embedders (NV-Embed, Qwen3-Embedding, E5-Mistral) now
  top MTEB. `gemini-embedding-001` leads English MTEB; `Qwen3-Embedding-8B`
  (Apache-2.0) leads open-weight.
- **MRL (Matryoshka)** is near-universal — one vector truncatable to nested
  dims. Retrieval can lose 5–20% Recall@10 under aggressive truncation;
  classification/clustering are robust.
- Batch limits: Cohere 96/req, Vertex 250, Voyage 1000, OpenAI 2048. Batch APIs:
  50% discount, 24h SLA.
- Quantization enum converging on `{float32, int8, uint8, binary, ubinary}`.
  INT8 = 4× storage, binary = 32× (needs float rescoring).
- Context lengths: 512 (legacy SBERT) → 8192 (BGE-M3, OpenAI) → 32K (Qwen3,
  Voyage) → 128K (Cohere embed-v4). Effective ≪ nominal ("Lost in the Middle").
- **Don't trust MTEB leaderboard averages** — recommend task-specific NDCG@10 on
  the user's own corpus.
- Multi-vector/sparse (ColBERT, SPLADE, BGE-M3) are out of the dense
  `ndarray(n,dim)` contract — a separate protocol for later.

## 8. Verbatim-worthy snippets

### 8.1 `BatchHandle` + `cache_key`
```python
@dataclass(frozen=True, slots=True)
class BatchHandle:
    poll: Callable        # () -> "pending" | "done" | "failed"
    result: Callable      # () -> np.ndarray  (blocks until done)
    cancel: Callable = lambda: None

def cache_key(e: Embedder, text: str, input_type: InputType | None) -> str:
    h = hashlib.sha256(f"{e.model_id}|{e.dim}|{input_type}|{text}".encode()).hexdigest()
    return f"{e.model_id}/{h}"
```

### 8.2 Length-sorted batching with order reconstruction
```python
indexed = list(enumerate(corpus))
ordered = sorted(indexed, key=lambda x: len(x[1]))
orig_idx = [i for i, _ in ordered]
sorted_emb = model.encode([t for _, t in ordered], batch_size=batch_size)
embeddings = np.empty((len(corpus), sorted_emb.shape[1]), dtype=np.float32)
for s, o in enumerate(orig_idx):
    embeddings[o] = sorted_emb[s]
```

### 8.3 The artifact-graph core
```python
class ProducerSpec:
    op: Callable
    op_version: str                       # participates in the id
    inputs: tuple[ArtifactId, ...]
    params: Mapping[str, Hashable]

def artifact_id(spec: ProducerSpec) -> str:
    payload = {"op_version": spec.op_version, "inputs": list(spec.inputs),
               "params": canonical_json(spec.params)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def materialize(self, key):              # lazy backward
    if key in self.store: return self.store[key]
    spec = self.producers[key]
    inputs = [self.materialize(i) for i in spec.inputs]
    self.store[key] = serialize(spec.op(*inputs, **spec.params))
    return self.store[key]

def delete_cascade(self, key):           # forward purge
    for k in self.mark_stale(key) | {key}:
        self.store.pop(k, None)
```

### 8.4 PCA → UMAP visualization (`ef.project`)
```python
reduced = (PCA(n_components=min(50, n_samples - 1), random_state=42)
           .fit_transform(vectors)) if source_dim > 50 else vectors
coords = umap.UMAP(n_components=target_dim, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=42, n_jobs=-1).fit_transform(reduced)
```

### 8.5 Refresh-mode semantics
| Mode | Behavior |
|---|---|
| `none` | pure dedup; nothing deleted |
| `incremental` | on hash change, old version deleted before new added; absent sources untouched |
| `full` | "this batch is the complete corpus"; anything absent is deleted |
| `scoped_full` | like `full`, scoped to the `source_id`s present in the batch |
