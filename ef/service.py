"""The HTTP-service bridge — :class:`EfService`, a stateless-friendly facade.

``ef``'s query surface is **stateful**: :func:`~ef.source_manager.ingest`
returns a live :class:`~ef.source_manager.SearchableCorpus`, and a
:class:`~ef.source_manager.SourceManager` holds a live corpus, an
:class:`~ef.artifact_graph.ArtifactGraph` and ``vd`` collections. HTTP is
**stateless**. To serve ``ef`` over HTTP — so a frontend (``app_ef``) can reach
it through ``qh.mk_app()`` — something must map a JSON-friendly ``corpus_id``
back to the live indexed object across requests.

:class:`EfService` is that bridge: a facade holding a **handle registry**
``{corpus_id: SourceManager}`` **on the instance**. A single ``EfService()`` is
constructed once at server start-up, and its seven bound methods —
:meth:`~EfService.create_corpus`, :meth:`~EfService.search`,
:meth:`~EfService.retrieve`, :meth:`~EfService.explore_corpus`,
:meth:`~EfService.corpus_info`, :meth:`~EfService.list_corpora`,
:meth:`~EfService.delete_corpus` — are handed to ``qh.mk_app()`` as the HTTP
surface. The registry lives on the instance,
never as a module global: a process-wide mutable registry would be the
``ServiceContext`` singleton anti-pattern ``ef`` rejects (``.claude/CLAUDE.md``
§6) — inject an :class:`EfService` explicitly instead.

Every method is JSON-friendly and fully type-annotated, because ``qh`` derives
the HTTP schema from the type hints. ``ef``'s stateful objects never cross the
boundary: a corpus is addressed by its string ``corpus_id``, an embedder /
segmenter by a string the DI seam resolves (``"hashing"``, ``"openai:..."``,
``"cohere:..."``, …), and results are plain :class:`~ef.source_manager.SearchHit`
/ :class:`~ef.segments.Segment` data.

This is *only* a transport bridge — it adds no orchestration. ``ef`` stays a
facade (``.claude/CLAUDE.md`` §6); answer synthesis, agents and UI live outside.

Example — create a corpus, search it, then drop it, all offline:

>>> from ef.service import EfService
>>> service = EfService()
>>> info = service.create_corpus(
...     ['the cat sat on the mat', 'dogs are loyal', 'felines and canines'],
...     corpus_id='animals',
... )
>>> info['corpus_id'], info['n_sources'], info['n_segments']
('animals', 3, 3)
>>> info['embedder'], info['dim']
('hashing:v1@512', 512)
>>> hits = service.search('animals', 'cat', limit=2)
>>> len(hits)
2
>>> isinstance(hits[0].score, float)
True
>>> segments = service.retrieve('animals', 'cat', limit=1)
>>> isinstance(segments[0]['text'], str)
True
>>> [c['corpus_id'] for c in service.list_corpora()]
['animals']
>>> service.delete_corpus('animals')
>>> service.list_corpora()
[]
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import TypedDict

from ef.exploration import ExploreResult, explore
from ef.segments import Segment
from ef.source_manager import DEFAULT_EMBEDDER, SearchHit, SourceManager

__all__ = ["CorpusInfo", "EfService"]


# ---------------------------------------------------------------------------
# The JSON-friendly corpus summary
# ---------------------------------------------------------------------------


class CorpusInfo(TypedDict):
    """A JSON-friendly summary of one registered corpus.

    The return shape shared by :meth:`EfService.create_corpus`,
    :meth:`EfService.corpus_info` and (as a list) :meth:`EfService.list_corpora`
    — a plain ``dict`` an HTTP client and ``qh``'s schema both understand.

    Keys:
        corpus_id: the registry handle — how every other method addresses it.
        n_sources: the number of source documents in the corpus.
        n_segments: the number of indexed segments (sources cut by the segmenter).
        embedder: the embedder's ``model_id`` (e.g. ``"hashing:v1@512"``).
        dim: the embedding dimensionality.
        config_id: the content hash of the segmenter+embedder pipeline.
    """

    corpus_id: str
    n_sources: int
    n_segments: int
    embedder: str
    dim: int
    config_id: str


def _new_corpus_id() -> str:
    """A fresh, URL-safe corpus id — a short random hex token."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# The service facade
# ---------------------------------------------------------------------------


class EfService:
    """A registry-backed facade bridging stateful ``ef`` to a stateless transport.

    Construct one ``EfService`` per process, register corpora with
    :meth:`create_corpus`, then query them by ``corpus_id``. Hand the seven
    bound methods to ``qh.mk_app()`` to expose them over HTTP — see the module
    docstring.

    The handle registry is private instance state — never a module global.
    Corpora are isolated: each :meth:`create_corpus` builds its own
    :class:`~ef.source_manager.SourceManager` over its own in-memory ``vd``
    backend, so one corpus's vectors never leak into another's search.

    ``default_embedder`` sets the embedder :meth:`create_corpus` resolves when
    its own ``embedder`` argument is ``None`` — the per-instance hook a host
    (e.g. ``app_ef``) uses to pick a policy default without passing
    ``embedder=`` on every call. It defaults to
    :data:`~ef.source_manager.DEFAULT_EMBEDDER` (``"hashing"``), so a bare
    ``EfService()`` stays dependency-free and offline.

    >>> service = EfService()
    >>> len(service)
    0
    >>> _ = service.create_corpus(['hello world'], corpus_id='greetings')
    >>> 'greetings' in service
    True
    >>> len(service)
    1

    A host can swap the default embedder at construction (the string is resolved
    lazily, per corpus, by the DI seam — constructing the service touches no
    network):

    >>> EfService(default_embedder='openai:text-embedding-3-small')
    <EfService: 0 corpus(es) registered>
    """

    def __init__(self, *, default_embedder: str | None = None) -> None:
        self._corpora: dict[str, SourceManager] = {}
        self._default_embedder: str = (
            default_embedder if default_embedder is not None else DEFAULT_EMBEDDER
        )

    # -- corpus lifecycle ---------------------------------------------------

    def create_corpus(
        self,
        sources: list[str],
        *,
        embedder: str | None = None,
        segmenter: str | None = None,
        corpus_id: str | None = None,
    ) -> CorpusInfo:
        """Index ``sources`` into a new corpus, register it, return its :class:`CorpusInfo`.

        Args:
            sources: the corpus — a list of text documents.
            embedder: the embedder, as a string the DI seam resolves
                (:func:`~ef.embedder_adapters.as_embedder`) — ``"hashing"``,
                ``"openai:text-embedding-3-small"``, ``"cohere:..."``, an
                ``http(s)://`` URL, …. ``None`` → the service's
                ``default_embedder`` (chosen at construction; itself
                :data:`~ef.source_manager.DEFAULT_EMBEDDER`, the dependency-free
                :class:`~ef.embedders.HashingEmbedder`, unless a host overrode it).
            segmenter: the segmenter, as a string
                (:func:`~ef.segmenter_adapters.as_segmenter`) — ``None`` → the
                recursive-character default.
            corpus_id: the handle to register the corpus under; ``None`` → a
                fresh random id. Reusing a live id is an error.

        Returns:
            the :class:`CorpusInfo` of the freshly indexed corpus.

        Raises:
            ValueError: if ``corpus_id`` is already registered.
        """
        cid = corpus_id if corpus_id is not None else _new_corpus_id()
        if cid in self._corpora:
            raise ValueError(
                f"corpus_id {cid!r} is already registered; delete it first "
                f"or choose another id."
            )
        manager = SourceManager(
            sources,
            segmenter=segmenter,
            embedder=embedder if embedder is not None else self._default_embedder,
        )
        manager.materialize()
        self._corpora[cid] = manager
        return self._info(cid)

    def corpus_info(self, corpus_id: str) -> CorpusInfo:
        """Return the :class:`CorpusInfo` of a registered corpus.

        Raises:
            KeyError: if ``corpus_id`` is not registered.
        """
        return self._info(corpus_id)

    def list_corpora(self) -> list[CorpusInfo]:
        """Return the :class:`CorpusInfo` of every registered corpus."""
        return [self._info(cid) for cid in self._corpora]

    def delete_corpus(self, corpus_id: str) -> None:
        """Drop a corpus from the registry, releasing its index.

        Raises:
            KeyError: if ``corpus_id`` is not registered.
        """
        self._manager(corpus_id)  # validate first, with a clear error
        del self._corpora[corpus_id]

    # -- querying -----------------------------------------------------------

    def search(self, corpus_id: str, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Search a registered corpus — up to ``limit`` ranked :class:`~ef.source_manager.SearchHit`\\ s.

        Each hit carries the matched :class:`~ef.segments.Segment`, its
        similarity ``score`` (higher = closer) and the ``source_id`` it was cut
        from.

        Raises:
            KeyError: if ``corpus_id`` is not registered.
        """
        return self._manager(corpus_id).search(query, limit=limit)

    def retrieve(self, corpus_id: str, query: str, *, limit: int = 10) -> list[Segment]:
        """Retrieve ranked :class:`~ef.segments.Segment`\\ s — the RAG-plug-in shape.

        Like :meth:`search`, but returns plain segments in rank order (the
        ``score`` dropped, the ``source_id`` folded into ``metadata["source"]``)
        — clean context to hand to an external RAG/agent framework. ``ef``
        returns context; it does not synthesize answers.

        Raises:
            KeyError: if ``corpus_id`` is not registered.
        """
        return self._manager(corpus_id).retrieve(query, limit=limit)

    def explore_corpus(
        self,
        corpus_id: str,
        *,
        dims: int = 2,
        projection_method: str = "auto",
        cluster_method: str = "kmeans",
        n_clusters: int = 8,
        label: bool = False,
    ) -> ExploreResult:
        """Project & cluster a registered corpus — the corpus-map surface.

        Runs :func:`ef.exploration.explore` over the corpus: every indexed
        segment is projected to ``dims`` coordinates and assigned a cluster,
        returned as a row-aligned :class:`~ef.exploration.ExploreResult` (``ids`` /
        ``coords`` / ``labels`` / ``cluster_titles``) — the JSON-friendly shape
        an ``app_ef`` corpus map consumes.

        Args:
            corpus_id: the corpus to explore.
            dims: projection target dimensionality — ``2`` or ``3``.
            projection_method: ``"auto"`` / ``"umap"`` / ``"pca"``.
            cluster_method: ``"kmeans"`` / ``"hdbscan"``.
            n_clusters: number of k-means clusters.
            label: when ``True``, also name each cluster with an LLM (needs the
                ``ef[imbed]`` extra and a key); default ``False``.

        Raises:
            KeyError: if ``corpus_id`` is not registered.
            ValueError: if the corpus has fewer than 2 indexed segments.
        """
        return explore(
            self._manager(corpus_id),
            dims=dims,
            projection_method=projection_method,
            cluster_method=cluster_method,
            n_clusters=n_clusters,
            label=label,
        )

    # -- dunders ------------------------------------------------------------

    def __contains__(self, corpus_id: object) -> bool:
        return corpus_id in self._corpora

    def __iter__(self) -> Iterator[str]:
        """Iterate the registered ``corpus_id``\\ s."""
        return iter(self._corpora)

    def __len__(self) -> int:
        return len(self._corpora)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {len(self._corpora)} corpus(es) registered>"

    # -- internals ----------------------------------------------------------

    def _manager(self, corpus_id: str) -> SourceManager:
        """Resolve ``corpus_id`` to its :class:`~ef.source_manager.SourceManager`."""
        try:
            return self._corpora[corpus_id]
        except KeyError:
            raise KeyError(
                f"Unknown corpus_id {corpus_id!r}. Registered: {sorted(self._corpora)}."
            ) from None

    def _info(self, corpus_id: str) -> CorpusInfo:
        """Build the :class:`CorpusInfo` for a registered corpus."""
        manager = self._manager(corpus_id)
        searchable = manager.searchable()
        embedder = searchable.embedder
        return CorpusInfo(
            corpus_id=corpus_id,
            n_sources=len(manager.corpus),
            n_segments=len(searchable.collection),
            embedder=str(getattr(embedder, "model_id", "unknown")),
            dim=int(getattr(embedder, "dim", 0)),
            config_id=str(searchable.config),
        )
