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
  :meth:`~ef.source_manager.SearchableCorpus.search` /
  :meth:`~ef.source_manager.SearchableCorpus.retrieve` return ranked
  :class:`~ef.source_manager.SearchHit`\\ s.
- :class:`~ef.source_manager.SourceManager` — the heavy facade: multi-config
  corpus indexing, where configs sharing a step share its artifacts for free.
- :class:`~ef.config.PipelineSpec` / :class:`~ef.config.TransformSpec` /
  :func:`~ef.config.config_id` — a pipeline as serializable, content-hashed data.

Example — wrap a plain function and embed two strings:

>>> import numpy as np
>>> from ef import as_embedder
>>> embedder = as_embedder(
...     lambda texts: np.array([[len(t)] for t in texts]), model_id='len@1'
... )
>>> embedder(['hello', 'hi']).ravel().tolist()
[5.0, 2.0]

The legacy embedding-*visualization* pipeline (``Project`` / ``Projects`` —
``segment → embed → planarize → cluster``) is being demoted to one secondary
use case of the refactored ``ef``. It needs the ``full`` / ``imbed`` extras and
is imported best-effort: if those dependencies are absent, the embedder facade
above still works.
"""

from ef.embedders import (
    BaseEmbedder,
    BatchHandle,
    BatchStatus,
    Embedder,
    EmbedderError,
    FunctionEmbedder,
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
from ef.source_manager import (
    DEFAULT_EMBEDDER,
    SearchHit,
    SearchableCorpus,
    SourceManager,
    ingest,
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
    "DEFAULT_EMBEDDER",
]

# ============================================================================
# Legacy visualization pipeline — optional (needs `ef[full]` / `ef[imbed]`)
# ============================================================================

# Note: the viz-era ``ef.base`` defines a ``Segment = str`` alias; it is
# deliberately *not* re-exported here — ``ef.Segment`` is the canonical segment
# data model (the ``TypedDict`` from :mod:`ef.segments`). The legacy alias is
# removed entirely when the visualization code is demoted to layer L5.
try:
    from ef.projects import Project, Projects
    from ef.base import (
        ClusterIndex,
        ComponentRegistry,
        PlanarVector,
        SegmentKey,
        Vector,
    )

    __all__ += [
        "Project",
        "Projects",
        "ComponentRegistry",
        "SegmentKey",
        "Vector",
        "PlanarVector",
        "ClusterIndex",
    ]
except ImportError:  # pragma: no cover - depends on optional extras
    pass
