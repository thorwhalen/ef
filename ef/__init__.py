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

# ============================================================================
# Public API — the embedder facade (always available)
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
]

# ============================================================================
# Legacy visualization pipeline — optional (needs `ef[full]` / `ef[imbed]`)
# ============================================================================

try:
    from ef.projects import Project, Projects
    from ef.base import (
        ClusterIndex,
        ComponentRegistry,
        PlanarVector,
        Segment,
        SegmentKey,
        Vector,
    )

    __all__ += [
        "Project",
        "Projects",
        "ComponentRegistry",
        "SegmentKey",
        "Segment",
        "Vector",
        "PlanarVector",
        "ClusterIndex",
    ]
except ImportError:  # pragma: no cover - depends on optional extras
    pass
