"""ef (Embedding Flow) — a facade for semantic-embeddings user journeys.

``ef`` makes the modern semantic-search pipeline — corpus, segmenter, embedder,
vector store, retriever — usable with progressive disclosure: the light case
(a list of strings, search in one or two lines) and the heavy case (huge
corpora, many segmentations/embedders, varied sources and vector DBs) share one
facade. ``ef`` is *not* RAG — it returns ranked segments; bring your own LLM.

The **embedder facade** is the core surface and is always importable (it costs
only numpy):

- :class:`~ef.embedders.Embedder` — the structural protocol for a batch
  ``Iterable[str] -> ndarray(n, dim)`` callable.
- :class:`~ef.embedders.HashingEmbedder` — the dependency-free default embedder
  (the feature-hashing trick, numpy only); what :func:`~ef.source_manager.ingest`
  resolves to when given no embedder, so the light path needs nothing to install.
- :func:`~ef.embedder_adapters.as_embedder` — the dependency-injection seam
  (string / callable / URL / existing embedder → ``Embedder``).
- Adapters: :func:`~ef.embedder_adapters.openai_embedder`,
  :func:`~ef.embedder_adapters.sentence_transformers_embedder`,
  :func:`~ef.embedder_adapters.http_embedder`.
- Composition wrappers: :class:`~ef.embedder_wrappers.CachedEmbedder`,
  :class:`~ef.embedder_wrappers.RetryingEmbedder`,
  :class:`~ef.embedder_wrappers.MultiEmbedder`,
  :class:`~ef.embedder_wrappers.NormalizingEmbedder`.

The **segmenter facade** is the other always-importable core surface (it needs
no numpy at all):

- :class:`~ef.segments.Segment` — the canonical segment data model (the
  ``TypedDict`` interchange type; :class:`~ef.segments.SegmentRecord` is its
  dataclass convenience surface).
- :class:`~ef.segmenters.Segmenter` — the structural protocol for a
  ``str | Mapping -> Iterable[Segment]`` callable.
- :class:`~ef.segmenters.RecursiveCharacterSegmenter` — the default splitter;
  :func:`~ef.segmenters.line_segmenter` a builtin line splitter.
- :func:`~ef.segmenter_adapters.as_segmenter` — the dependency-injection seam.
- Composition helpers: :func:`~ef.segmenters.with_overlap`,
  :func:`~ef.segmenters.hierarchical`, :func:`~ef.segmenters.materialise`.

The **corpus facade** is ``ef``'s source layer (L0) — also always importable:

- :data:`~ef.corpus.Corpus` / :data:`~ef.corpus.Source` — the type aliases: a
  corpus is just a ``MutableMapping[source_id, Source]`` (any ``dol`` store).
- :func:`~ef.corpus.as_corpus` — the dependency-injection seam (``None`` /
  mapping / directory path / iterable of sources → a corpus).
- :func:`~ef.corpus.content_hash` — the content hash of a source.
- :class:`~ef.corpus.ChangeDetectingCorpus` — a corpus wrapper that detects and
  reports changes (:class:`~ef.corpus.ChangeEvent` / :class:`~ef.corpus.CorpusDiff`).

The **artifact graph** is ``ef``'s corpus-indexing core (also always importable)
— a content-addressed producer graph in which cascade invalidation and config
branching are the same operation:

- :class:`~ef.artifact_graph.ProducerSpec` — the declarative recipe for one
  produced artifact; :func:`~ef.artifact_graph.artifact_id` is its content hash;
  :func:`~ef.artifact_graph.producer_spec` builds one ergonomically.
- :class:`~ef.artifact_graph.ArtifactGraph` — the graph: ``materialize`` (lazy
  backward), ``mark_stale`` / ``delete_cascade`` (forward) and ``freshness``.

The **search facade** is where the layers come together — corpus → segment →
embed → ``vd`` → ranked search, with progressive disclosure:

- :func:`~ef.source_manager.ingest` — the one-shot light path: a corpus in, a
  :class:`~ef.source_manager.SearchableCorpus` out, ``search(query)`` ready.
- :class:`~ef.source_manager.SearchableCorpus` — the thin ready-search object;
  :meth:`~ef.source_manager.SearchableCorpus.search` returns scored
  :class:`~ef.source_manager.SearchHit`\\ s, :meth:`~ef.source_manager.SearchableCorpus.retrieve`
  returns plain ranked :class:`~ef.segments.Segment`\\ s — the RAG-plug-in shape.
- :class:`~ef.source_manager.SourceManager` — the heavy facade: multi-config
  corpus indexing, where configs sharing a step share its artifacts for free.
- :class:`~ef.config.PipelineSpec` / :class:`~ef.config.TransformSpec` /
  :func:`~ef.config.config_id` — a pipeline as serializable, content-hashed data.

The **refresh layer** keeps an indexed corpus in sync as its sources change:

- :meth:`SourceManager.diagnose <ef.source_manager.SourceManager.diagnose>` —
  the four staleness conditions, as a :class:`~ef.diagnostics.StalenessReport`.
- :meth:`SourceManager.refresh <ef.source_manager.SourceManager.refresh>` —
  re-sync the index, returning a :class:`~ef.refresh.RefreshReport`; one of four
  :data:`~ef.refresh.RefreshMode`\\ s. ``SourceManager(auto_refresh=True)`` keeps
  the index live as the corpus is edited.

The **RAG-plug-in & evaluation layer** hands a corpus to an external RAG/agent
framework and measures its quality — ``ef`` returns ranked context, it does not
synthesize answers:

- :meth:`SourceManager.retrieve <ef.source_manager.SourceManager.retrieve>`
  returns plain ranked :class:`~ef.segments.Segment`\\ s — the RAG-plug-in shape;
  :func:`~ef.source_manager.hits_to_segments` is the projection from
  :class:`~ef.source_manager.SearchHit`\\ s.
- :func:`~ef.evaluation.evaluate_retrieval` — BEIR/MTEB-shaped retrieval scoring
  (primary metric NDCG@10); :func:`~ef.evaluation.read_beir` /
  :func:`~ef.evaluation.write_beir` move the corpus/queries/qrels triple to disk.
- :func:`~ef.evaluation.evaluate_rag` — deterministic lexical RAG metrics over
  :class:`~ef.evaluation.RagSample`\\ s; :func:`~ef.evaluation.as_ragas_dataset`
  bridges to Ragas for the LLM-judged metrics.
- :func:`~ef.reranking.with_reranker` — a two-stage reranking decorator over any
  retriever; :func:`~ef.reranking.cross_encoder_reranker` is a ready reranker.

The **explore layer** (L5) is ``ef``'s visualization heritage, kept as a
*secondary* "see the shape of the corpus" surface — search/RAG/indexing is the
primary one. It imports numpy-only; UMAP, HDBSCAN and ``imbed`` are lazy extras:

- :func:`~ef.explore.project` — reduce embeddings to 2-D/3-D coordinates
  (PCA → UMAP); :func:`~ef.explore.cluster` — group them (k-means / HDBSCAN);
  :func:`~ef.explore.label_clusters` — name clusters with an LLM via ``imbed``.

Example — wrap a plain function and embed two strings:

>>> import numpy as np
>>> from ef import as_embedder
>>> embedder = as_embedder(
...     lambda texts: np.array([[len(t)] for t in texts]), model_id='len@1'
... )
>>> embedder(['hello', 'hi']).ravel().tolist()
[5.0, 2.0]
"""

from ef.embedders import (
    BaseEmbedder,
    BatchHandle,
    BatchStatus,
    Embedder,
    EmbedderError,
    FunctionEmbedder,
    HashingEmbedder,
    InputType,
    cache_key,
    embed_length_sorted,
    ready_handle,
)
from ef.embedder_wrappers import (
    CachedEmbedder,
    MultiEmbedder,
    NormalizingEmbedder,
    RetryPolicy,
    RetryingEmbedder,
)
from ef.embedder_adapters import (
    as_embedder,
    http_embedder,
    openai_embedder,
    sentence_transformers_embedder,
)
from ef.segments import (
    Segment,
    SegmentRecord,
    as_segment,
    make_segment,
    segment_id,
    segment_record,
)
from ef.segmenters import (
    APPROX_TOKENIZER,
    DEFAULT_SEPARATORS,
    BaseSegmenter,
    BatchedSegmenter,
    FunctionSegmenter,
    RecursiveCharacterSegmenter,
    Segmenter,
    approx_token_count,
    hierarchical,
    line_segmenter,
    materialise,
    with_overlap,
)
from ef.segmenter_adapters import as_segmenter, imbed_segmenter
from ef.corpus import (
    ChangeDetectingCorpus,
    ChangeEvent,
    ChangeKind,
    Corpus,
    CorpusDiff,
    Source,
    as_corpus,
    content_hash,
)
from ef.artifact_graph import (
    ArtifactGraph,
    ArtifactId,
    Freshness,
    OpKey,
    ProducerSpec,
    artifact_id,
    producer_spec,
)
from ef.config import (
    ConfigId,
    PipelineSpec,
    TransformSpec,
    config_id,
    full_kwargs,
    step_params,
)
from ef.diagnostics import (
    IndexedSource,
    StalenessReport,
    diagnose,
)
from ef.refresh import (
    RefreshMode,
    RefreshPlan,
    RefreshReport,
    plan_refresh,
    refresh_on_change,
)
from ef.source_manager import (
    DEFAULT_EMBEDDER,
    SearchHit,
    SearchableCorpus,
    SourceManager,
    hits_to_segments,
    ingest,
)
from ef.evaluation import (
    Qrels,
    RagEvalReport,
    RagSample,
    RetrievalEvalReport,
    as_ragas_dataset,
    average_precision,
    context_precision,
    context_recall,
    dcg_at_k,
    evaluate_rag,
    evaluate_retrieval,
    exact_match,
    ndcg_at_k,
    precision_at_k,
    read_beir,
    recall_at_k,
    reciprocal_rank,
    token_f1,
    write_beir,
)
from ef.reranking import (
    Reranker,
    cross_encoder_reranker,
    rerank,
    with_reranker,
)
from ef.explore import (
    cluster,
    label_clusters,
    project,
)

# ============================================================================
# Public API — the embedder, segmenter & corpus facades (always available)
# ============================================================================

__all__ = [
    # protocol & core types
    "Embedder",
    "BaseEmbedder",
    "InputType",
    "BatchHandle",
    "BatchStatus",
    "ready_handle",
    "EmbedderError",
    "FunctionEmbedder",
    "HashingEmbedder",
    "cache_key",
    "embed_length_sorted",
    # composition wrappers
    "CachedEmbedder",
    "RetryingEmbedder",
    "RetryPolicy",
    "MultiEmbedder",
    "NormalizingEmbedder",
    # adapters & the DI seam
    "as_embedder",
    "openai_embedder",
    "sentence_transformers_embedder",
    "http_embedder",
    # --- segmenter facade ---
    # data model
    "Segment",
    "SegmentRecord",
    "segment_id",
    "make_segment",
    "as_segment",
    "segment_record",
    # protocols & core types
    "Segmenter",
    "BatchedSegmenter",
    "BaseSegmenter",
    "approx_token_count",
    "APPROX_TOKENIZER",
    "DEFAULT_SEPARATORS",
    # segmenters
    "RecursiveCharacterSegmenter",
    "FunctionSegmenter",
    "line_segmenter",
    # composition helpers
    "with_overlap",
    "hierarchical",
    "materialise",
    # adapters & the DI seam
    "as_segmenter",
    "imbed_segmenter",
    # --- corpus facade ---
    # the contract
    "Corpus",
    "Source",
    "content_hash",
    # the DI seam
    "as_corpus",
    # change detection
    "ChangeDetectingCorpus",
    "ChangeEvent",
    "ChangeKind",
    "CorpusDiff",
    # --- artifact graph (corpus-indexing core) ---
    "ArtifactGraph",
    "ArtifactId",
    "OpKey",
    "Freshness",
    "ProducerSpec",
    "artifact_id",
    "producer_spec",
    # --- config layer (declarative pipeline specs) ---
    "TransformSpec",
    "PipelineSpec",
    "ConfigId",
    "config_id",
    "full_kwargs",
    "step_params",
    # --- search facade (ready search + one-shot ingest) ---
    "ingest",
    "SearchableCorpus",
    "SourceManager",
    "SearchHit",
    "hits_to_segments",
    "DEFAULT_EMBEDDER",
    # --- diagnostics & refresh (keeping an index in sync) ---
    "StalenessReport",
    "IndexedSource",
    "diagnose",
    "RefreshReport",
    "RefreshPlan",
    "RefreshMode",
    "plan_refresh",
    "refresh_on_change",
    # --- evaluation (retrieval + RAG quality measurement) ---
    "evaluate_retrieval",
    "RetrievalEvalReport",
    "evaluate_rag",
    "RagEvalReport",
    "RagSample",
    "Qrels",
    "read_beir",
    "write_beir",
    "as_ragas_dataset",
    "ndcg_at_k",
    "dcg_at_k",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "average_precision",
    "exact_match",
    "token_f1",
    "context_recall",
    "context_precision",
    # --- reranking (a precision pass over retrieved segments) ---
    "Reranker",
    "rerank",
    "with_reranker",
    "cross_encoder_reranker",
    # --- explore (layer L5 — project / cluster / label a corpus) ---
    "project",
    "cluster",
    "label_clusters",
]
