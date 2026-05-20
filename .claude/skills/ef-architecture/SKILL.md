---
name: ef-architecture
description: >-
  Developer skill: the target architecture of the ef (Embedding Flow) package.
  Load at the start of any non-trivial task in the ef repository — refactoring,
  adding the Embedder/Segmenter facades, the Corpus abstraction, the
  ArtifactGraph indexing core, or the "ready search" surface. Covers the
  five-layer model, the core contracts, the light/heavy progressive-disclosure
  split, and the hard boundaries. Trigger on "work on ef", "refactor ef",
  "implement the embedder/segmenter/corpus/artifact graph", "ef search".
audience: developers
---

# `ef` target architecture primer

Load this before any non-trivial `ef` work. Full detail:
[`misc/docs/ef_design_notes.md`](../../../misc/docs/ef_design_notes.md),
[`misc/docs/ef_use_cases.md`](../../../misc/docs/ef_use_cases.md),
[`.claude/CLAUDE.md`](../../CLAUDE.md).

## What `ef` is becoming

A **facade for boilerplate-less semantic-search / RAG / corpus-indexing**.
Light case in one line (a list of strings → search); heavy case (huge corpora,
many segmentations/embedders, varied sources/targets) with the *same* facade.
Uses `vd` for vector storage. `ef` is **not RAG** — no answer synthesis.

**Current `ef` is a prototype of an embedding *visualization* pipeline**
(`segment → embed → planarize → cluster`). The refactor makes search/RAG/
corpus-indexing primary and demotes visualization to one secondary use case
("explore the corpus", layer L5). `ef` is free to change — no users.

## The five-layer spine

```
L0 Sources   Corpus = MutableMapping[source_id, Source]
L1 Parse     pluggable parser
L2 Segment   Segmenter facade (on imbed.components.segmentation)
L3 Embed     Embedder facade (wraps imbed + provider/local adapters)
L4 Index     vd.Collection  (ef writes; vd owns)
L5 Derive    planarize / cluster / label  ("explore")
→ Search → "ready search": search(query) -> ranked Segments
→ RAG plug → retrieve() handed to an external framework
```

## The core contracts (all `@runtime_checkable Protocol`, never ABC)

1. **`Embedder`** = batch callable `Iterable[str] -> ndarray(n,dim)` + metadata
   attrs `model_id` / `dim` / `normalized` / `honored_input_types`. Canonical
   `InputType = Literal["query","document","classification","clustering"]`.
   Composition wrappers: `Cached*`, `Retrying*`, `Multi*`, `Normalizing*`.
2. **`Segmenter`** = callable `str|Mapping -> Iterable[Segment]`. Streaming-first.
   Stateful ones inject their embedder.
3. **`Corpus`** = `MutableMapping[source_id, Source]`; `ChangeDetecting` wrapper
   content-hashes values and fires invalidation.
4. **`ArtifactGraph`** = content-addressed producer graph. `artifact_id =
   H(op, op_version, inputs, params)`. Four ops: `materialize` (lazy backward),
   `mark_stale` / `delete_cascade` (forward), `freshness`.
5. **Facade** = thin two-class top (`Project`/`Corpus`) + standalone components;
   a one-shot `ingest(sources, *, segmenter, embedder, store, cache)`.

## The data model

Fields: `text`, `id` (required); `metadata`, `embedding`, `parent_id`,
`chunk_idx`, `start`/`end`, `tokens` (optional). `parent_id`/`chunk_idx` are
**top-level**, not metadata. Default IDs are content-derived. Always record the
`tokenizer` with a segment. Standardize on the word **"segment"**, not "chunk".

## The one idea that drives indexing & refresh

**Cascade invalidation and config branching are the same operation.** A source
edit and a parameter change both produce a new leaf hash whose downstream cone
must be (re-)materialized. One-time index = `materialize` once. Refresh = diff
hashes → `delete_cascade` → `materialize`. Config branching = free (shares
upstream artifacts). Don't build these as separate code paths.

## Hard boundaries — do NOT add

- No orchestration: no agents, tool registries, memory, conversation managers.
- No LLM answer synthesis — `ef` returns `retrieve()`; the app synthesizes.
- No global config singleton — inject embedders/segmenters/stores explicitly.
  (A process-wide mutable `ComponentRegistry` as global state = LlamaIndex's
  deprecated `ServiceContext` mistake.)
- No bundled UI — that is `app_ef`.
- No document-loader sprawl — let users pull `pypdf`/`unstructured`/`firecrawl`.
- Heavy deps (torch, openai, langchain…) → optional extras, lazily imported.

## Ecosystem wiring

`vd` = vectorDB interface (L4). `imbed` = heavy implementations — wrap via the
protocols, don't reimplement (`imbed` interface-preserving additions welcome;
breaking changes need review). `qh` = expose `ef` over HTTP for `app_ef`. `ju` =
OpenAPI bridge. `raglab`/`srag` = consumers of `retrieve()`. `dol` = every store
is a `MutableMapping`.

## Refactor order

Phases 1–6 are **done**; 7–8 remain.

1. ✅ `Embedder` protocol + wrappers + 2 adapters (OpenAI, sentence-transformers).
2. ✅ `Segmenter` facade on `imbed.components.segmentation`; canonical `Segment`.
3. ✅ `Corpus` + `ChangeDetecting` wrapper.
4. ✅ `ArtifactGraph` core (content-addressed producer graph).
5. ✅ "Ready search" + one-shot `ingest` wiring corpus→segment→embed→`vd`.
6. ✅ Refresh — `ef/diagnostics.py` (the four staleness conditions) +
   `ef/refresh.py` (four refresh modes, `plan_refresh`, auto-refresh); the
   `SourceManager` surface `diagnose`/`refresh`/`rebuild`/`gc_orphans`/`lineage`.
7. RAG-plug-in `retrieve()` + evaluation hookpoints.
8. Demote viz to L5 "explore".

When in doubt, re-read `ef_design_notes.md` — it has the verbatim contracts and
code skeletons.
