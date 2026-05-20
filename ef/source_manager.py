"""The search facade — one-shot :func:`ingest`, :class:`SearchableCorpus`, :class:`SourceManager`.

This is where ``ef``'s layers come together. Phases 1–4 built the parts — a
:class:`~ef.corpus.Corpus`, a :class:`~ef.segmenters.Segmenter`, an
:class:`~ef.embedders.Embedder`, the content-addressed
:class:`~ef.artifact_graph.ArtifactGraph` — and this module wires
**corpus → segment → embed → vd** into a usable search facade with the
progressive disclosure ``ef`` promises:

- **The light path** — :func:`ingest` — one line. A list of strings (or any
  corpus) in, a :class:`SearchableCorpus` out, ``search(query)`` ready.
- **The heavy path** — :class:`SourceManager` — the same machinery with the
  components named explicitly and **multiple configs** (a config is one
  segmenter+embedder pipeline). Two configs that share a step share that step's
  artifacts in the :class:`~ef.artifact_graph.ArtifactGraph` for free — config
  branching costs only the divergent cone.

How the wiring works (one source, one config):

1. the source is a graph **leaf**, keyed by :func:`~ef.corpus.content_hash`;
2. a ``segment`` :class:`~ef.artifact_graph.ProducerSpec` over that leaf →
   a ``list`` of :class:`~ef.segments.Segment`\\ s;
3. an ``embed`` :class:`~ef.artifact_graph.ProducerSpec` over the segment
   artifact → an ``(n, dim)`` array;
4. each ``(segment, vector)`` pair is written into a ``vd`` collection as a
   :class:`vd.Document`, with ``source_id`` / ``source_hash`` / ``config_hash``
   recorded in its metadata.

``ef`` owns the embedding: it always supplies vectors to ``vd`` (the index is
``vd``'s job, the vectors are ``ef``'s — design notes §7), and at query time it
embeds the query with ``input_type="query"`` and hands ``vd`` the vector.

Example — the light path, end to end, offline:

>>> import numpy as np
>>> from ef import ingest, as_embedder
>>> toy = as_embedder(
...     lambda texts: np.array(
...         [[t.count('a'), t.count('e'), t.count('o')] for t in texts], dtype=float
...     ),
...     model_id='toy-vowels@3',
... )
>>> idx = ingest(['apple orchard', 'ocean breeze', 'green meadow'], embedder=toy)
>>> hits = idx.search('ocean breeze', limit=2)
>>> hits[0].segment['text']
'ocean breeze'
>>> round(hits[0].score, 6)
1.0
>>> hits[0].source_id is not None
True
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from ef.artifact_graph import ArtifactGraph, ArtifactId, producer_spec
from ef.config import ConfigId, PipelineSpec, config_id
from ef.corpus import as_corpus, content_hash
from ef.embedder_adapters import as_embedder
from ef.embedders import Embedder
from ef.ops import embed_step, segment_step
from ef.segmenter_adapters import as_segmenter
from ef.segments import Segment, make_segment

__all__ = [
    "DEFAULT_EMBEDDER",
    "SearchHit",
    "SearchableCorpus",
    "SourceManager",
    "ingest",
]


#: The model name :func:`ingest` / :class:`SourceManager` resolve an absent
#: ``embedder`` to — the local ``sentence-transformers`` default. It needs the
#: ``ef[st]`` extra; pass an explicit ``embedder`` to avoid that dependency.
DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"

#: Metadata keys ``ef`` reserves on every ``vd`` document it writes. Provenance
#: (``source_id`` / ``source_hash`` / ``config_hash`` — the latter two are
#: Phase-6 staleness-filter keys) and the promoted :class:`~ef.segments.Segment`
#: fields that ``vd``'s flat document model cannot hold top-level. A segment's
#: own ``metadata`` must not use these keys — they would not round-trip.
_RESERVED_METADATA_KEYS = (
    "source_id",
    "source_hash",
    "config_hash",
)
_PROMOTED_SEGMENT_FIELDS = ("parent_id", "index", "start", "end", "tokens")


# ---------------------------------------------------------------------------
# The search result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked search result — a :class:`~ef.segments.Segment` and its score.

    :meth:`SearchableCorpus.search` / :meth:`SourceManager.search` return a
    ranked ``list`` of these. Keeping :class:`~ef.segments.Segment` (the
    interchange ``TypedDict``) pure of a result-only ``score`` key, a
    :class:`SearchHit` *wraps* it alongside the similarity ``score`` and the
    ``source_id`` of the document the segment was cut from.

    Attributes:
        segment: the matched segment — the canonical :class:`~ef.segments.Segment`.
        score: the similarity score from the vector store (higher = closer).
        source_id: the corpus key of the source document, if recorded.

    >>> hit = SearchHit(segment={'text': 'hi', 'id': 'x'}, score=0.9, source_id='doc-1')
    >>> hit.segment['text'], hit.score, hit.source_id
    ('hi', 0.9, 'doc-1')
    """

    segment: Segment
    score: float
    source_id: str | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_floats(vector: Any) -> list[float]:
    """Coerce a vector (``ndarray`` row, list, …) into a plain ``list[float]``."""
    tolist = getattr(vector, "tolist", None)
    seq = tolist() if tolist is not None else vector
    return [float(x) for x in seq]


def _coerce_embedder(embedder: Any) -> Embedder:
    """Coerce an embedder argument, defaulting ``None`` to :data:`DEFAULT_EMBEDDER`."""
    if embedder is None:
        return as_embedder(DEFAULT_EMBEDDER)
    return as_embedder(embedder)


def _doc_metadata(
    segment: Segment, *, source_id: str, source_hash: str, config_hash: ConfigId
) -> dict[str, Any]:
    """Build the ``vd`` document metadata for ``segment`` — provenance + promoted fields."""
    meta = dict(segment.get("metadata") or {})
    meta["source_id"] = source_id
    meta["source_hash"] = source_hash
    meta["config_hash"] = config_hash
    for key in _PROMOTED_SEGMENT_FIELDS:
        value = segment.get(key)
        if value is not None:
            meta[key] = value
    return meta


def _hit_from_result(result: Mapping[str, Any]) -> SearchHit:
    """Rebuild a :class:`SearchHit` (segment + score + source) from a ``vd`` result."""
    meta = dict(result.get("metadata") or {})
    source_id = meta.pop("source_id", None)
    meta.pop("source_hash", None)
    meta.pop("config_hash", None)
    promoted = {k: meta.pop(k) for k in _PROMOTED_SEGMENT_FIELDS if k in meta}
    segment = make_segment(
        result["text"], id=result["id"], metadata=meta or None, **promoted
    )
    return SearchHit(segment=segment, score=float(result["score"]), source_id=source_id)


def _search_collection(
    collection: Any,
    embedder: Embedder,
    query: str | Iterable[float],
    *,
    limit: int,
    filter: Mapping[str, Any] | None,
) -> list[SearchHit]:
    """Embed ``query`` (if it is text) and search ``collection`` — return ranked hits."""
    if isinstance(query, str):
        query_vector = _as_floats(embedder([query], input_type="query")[0])
    else:  # a pre-computed query vector
        query_vector = _as_floats(query)
    results = collection.search(query_vector, limit=limit, filter=filter)
    return [_hit_from_result(result) for result in results]


# ---------------------------------------------------------------------------
# The thin facade — what `ingest` returns
# ---------------------------------------------------------------------------


class SearchableCorpus:
    """A ready-to-search view over one indexed config — the thin facade.

    What :func:`ingest` returns and what :meth:`SourceManager.searchable` hands
    out: a small, read-mostly object that knows just enough to answer a query —
    the ``vd`` collection holding the vectors and the :class:`~ef.embedders.Embedder`
    that embeds the query. It does **not** re-index; mutation, refresh and
    multi-config live on :class:`SourceManager`.

    Args:
        collection: the ``vd`` collection holding this config's vectors.
        embedder: the embedder — used to embed queries with ``input_type="query"``.
        config: the :class:`~ef.config.ConfigId` of the indexed pipeline, if known.

    >>> # built for you by ingest(); see the module docstring for a full example.
    """

    def __init__(
        self, collection: Any, embedder: Embedder, *, config: ConfigId | None = None
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.config = config

    def search(
        self,
        query: str | Iterable[float],
        *,
        limit: int = 10,
        filter: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search the indexed corpus — return up to ``limit`` ranked :class:`SearchHit`\\ s.

        ``query`` is text (embedded with ``input_type="query"``) or an already
        computed query vector. ``filter`` is a ``vd`` metadata filter
        (MongoDB-style) — see :mod:`vd.filters`.
        """
        return _search_collection(
            self.collection, self.embedder, query, limit=limit, filter=filter
        )

    def retrieve(
        self,
        query: str | Iterable[float],
        *,
        limit: int = 10,
        filter: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Retrieve ranked segments for ``query`` — the RAG-plug-in surface.

        Identical to :meth:`search` in Phase 5: it is the name an external
        RAG/agent framework reaches for (``ef`` returns ranked segments; it does
        not synthesize answers). Evaluation hookpoints land in a later phase.
        """
        return self.search(query, limit=limit, filter=filter)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {len(self.collection)} segment(s) indexed>"


# ---------------------------------------------------------------------------
# The heavy facade — multi-config indexing
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    """A registered config: its pipeline spec, its embedder, its ``vd`` collection."""

    name: str
    pipeline: PipelineSpec
    embedder: Embedder
    collection: Any


class SourceManager:
    """The heavy search facade — multi-config corpus indexing over one corpus.

    A :class:`SourceManager` holds a corpus, an :class:`~ef.artifact_graph.ArtifactGraph`
    and one or more **configs**. A config is a named segmenter+embedder pipeline
    (:class:`~ef.config.PipelineSpec`); :meth:`register_config` adds one,
    :meth:`materialize` runs the corpus through it into a ``vd`` collection, and
    :meth:`search` queries it. Because every artifact is content-addressed in
    the shared graph, two configs that share a step (e.g. the same segmenter)
    compute that step's artifacts **once** — config branching is free.

    The constructor optionally registers a ``"default"`` config: pass an
    ``embedder`` (and optionally a ``segmenter``) and it is registered eagerly;
    omit ``embedder`` to build every config explicitly via :meth:`register_config`.

    Args:
        corpus: the source corpus — coerced by :func:`~ef.corpus.as_corpus`
            (a mapping, a directory path, an iterable of sources, or ``None``).
        segmenter: the default config's segmenter — coerced by
            :func:`~ef.segmenter_adapters.as_segmenter` (``None`` → the recursive
            default). Used only when ``embedder`` is also given.
        embedder: the default config's embedder — coerced by
            :func:`~ef.embedder_adapters.as_embedder`. If given, a ``"default"``
            config is registered; if ``None``, no default config is registered.
        store: where the vectors go. ``None`` → an in-memory ``vd`` backend; a
            ``vd`` client → collections are created on it; a ``vd`` collection →
            used directly (then only one config is supported); a backend-name
            string → ``vd.connect`` of that backend.
        cache: the :class:`~ef.artifact_graph.ArtifactGraph`'s value cache — any
            ``MutableMapping``; ``None`` → an in-RAM ``dict``.

    >>> import numpy as np
    >>> from ef import as_embedder
    >>> e1 = as_embedder(lambda ts: np.array([[len(t)] for t in ts], float), model_id='len@1')
    >>> sm = SourceManager(['hello', 'hi there'], embedder=e1)
    >>> report = sm.materialize()
    >>> report['segments']
    2
    >>> len(sm.search('hello')) >= 1
    True
    """

    def __init__(
        self,
        corpus: Any = None,
        *,
        segmenter: Any = None,
        embedder: Any = None,
        store: Any = None,
        cache: MutableMapping[ArtifactId, Any] | None = None,
    ) -> None:
        self.corpus = as_corpus(corpus)
        self.graph = ArtifactGraph(store=cache)
        self._client, self._explicit_collection = _open_store(store)
        self._configs: dict[ConfigId, _Config] = {}
        self._names: dict[str, ConfigId] = {}
        self._default_config: ConfigId | None = None
        if embedder is not None:
            self.register_config("default", segmenter=segmenter, embedder=embedder)

    # -- config registration ------------------------------------------------

    def register_config(
        self, name: str = "default", *, segmenter: Any = None, embedder: Any = None
    ) -> ConfigId:
        """Register a named segmenter+embedder pipeline; return its :class:`~ef.config.ConfigId`.

        The segmenter and embedder are coerced through ``ef``'s DI seams
        (:func:`~ef.segmenter_adapters.as_segmenter` /
        :func:`~ef.embedder_adapters.as_embedder`; an absent ``embedder``
        defaults to :data:`DEFAULT_EMBEDDER`). Their ops are registered into the
        shared :class:`~ef.artifact_graph.ArtifactGraph` and a ``vd`` collection
        is created for the config. Registering a config does *not* index it —
        call :meth:`materialize`.

        Re-registering the same ``name`` replaces it. Two configs that resolve
        to the same :class:`~ef.config.PipelineSpec` share one
        :class:`~ef.config.ConfigId` (and one collection).

        >>> import numpy as np
        >>> from ef import as_embedder
        >>> sm = SourceManager(['a', 'b'])
        >>> cid = sm.register_config(
        ...     'mini', embedder=as_embedder(
        ...         lambda ts: np.ones((len(ts), 2)), model_id='ones@2'))
        >>> len(cid)
        64
        """
        segmenter_obj = as_segmenter(segmenter)
        embedder_obj = _coerce_embedder(embedder)
        seg_spec, seg_fn = segment_step(segmenter_obj)
        emb_spec, emb_fn = embed_step(embedder_obj)
        pipeline = PipelineSpec(segment=seg_spec, embed=emb_spec)
        cid = config_id(pipeline)
        self.graph.register_op(seg_spec.op, seg_fn)
        self.graph.register_op(emb_spec.op, emb_fn)
        collection = self._collection_for(name, cid)
        self._configs[cid] = _Config(
            name=name, pipeline=pipeline, embedder=embedder_obj, collection=collection
        )
        self._names[name] = cid
        if name == "default":
            self._default_config = cid
        return cid

    # -- materialization ----------------------------------------------------

    def materialize(self, config: str | ConfigId | None = None) -> dict[str, Any]:
        """Index the corpus through one config (or all) — corpus → segment → embed → ``vd``.

        For each source and each selected config: the source is added to the
        :class:`~ef.artifact_graph.ArtifactGraph` as a content-addressed leaf,
        its ``segment`` and ``embed`` artifacts are :meth:`~ef.artifact_graph.ArtifactGraph.materialize`\\ d
        (a cache hit if already computed — so a shared segment step runs once
        across configs), and every ``(segment, vector)`` pair is written into
        the config's ``vd`` collection. Idempotent: re-materializing rewrites
        identical documents.

        Args:
            config: the config to index — a name, a :class:`~ef.config.ConfigId`,
                or ``None`` for *every* registered config.

        Returns:
            a report ``dict`` with ``configs`` / ``sources`` / ``segments`` counts.
        """
        configs = (
            list(self._configs) if config is None else [self._resolve_config(config)]
        )
        total_segments = 0
        for cid in configs:
            total_segments += self._materialize_config(cid)
        return {
            "configs": len(configs),
            "sources": len(self.corpus),
            "segments": total_segments,
        }

    def _materialize_config(self, cid: ConfigId) -> int:
        """Index every source through config ``cid``; return the segment count written."""
        cfg = self._configs[cid]
        pipeline = cfg.pipeline
        written = 0
        for source_id in self.corpus:
            source = self.corpus[source_id]
            leaf = content_hash(source)
            self.graph.put(leaf, source)
            seg_aid = self.graph.add(
                producer_spec(
                    pipeline.segment.op,
                    leaf,
                    op_version=pipeline.segment.op_version,
                    **pipeline.segment.params,
                )
            )
            emb_aid = self.graph.add(
                producer_spec(
                    pipeline.embed.op,
                    seg_aid,
                    op_version=pipeline.embed.op_version,
                    **pipeline.embed.params,
                )
            )
            segments = self.graph.materialize(seg_aid)
            vectors = self.graph.materialize(emb_aid)
            for segment, vector in zip(segments, vectors):
                self._write_document(
                    cfg.collection, segment, vector, str(source_id), leaf, cid
                )
                written += 1
        return written

    @staticmethod
    def _write_document(
        collection: Any,
        segment: Segment,
        vector: Any,
        source_id: str,
        source_hash: str,
        config_hash: ConfigId,
    ) -> None:
        """Write one ``(segment, vector)`` pair into ``collection`` as a ``vd`` document."""
        from vd import Document

        metadata = _doc_metadata(
            segment,
            source_id=source_id,
            source_hash=source_hash,
            config_hash=config_hash,
        )
        collection[segment["id"]] = Document(
            id=segment["id"],
            text=segment["text"],
            vector=_as_floats(vector),
            metadata=metadata,
        )

    # -- querying -----------------------------------------------------------

    def search(
        self,
        query: str | Iterable[float],
        *,
        config: str | ConfigId | None = None,
        limit: int = 10,
        filter: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search one config's index — return up to ``limit`` ranked :class:`SearchHit`\\ s.

        ``config`` selects which config to query (a name / :class:`~ef.config.ConfigId`,
        or ``None`` for the default or sole config). ``query`` is text or a
        pre-computed query vector; ``filter`` is a ``vd`` metadata filter.
        """
        cid = self._resolve_config(config)
        cfg = self._configs[cid]
        return _search_collection(
            cfg.collection, cfg.embedder, query, limit=limit, filter=filter
        )

    def retrieve(
        self,
        query: str | Iterable[float],
        *,
        config: str | ConfigId | None = None,
        limit: int = 10,
        filter: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Retrieve ranked segments — the RAG-plug-in alias of :meth:`search`."""
        return self.search(query, config=config, limit=limit, filter=filter)

    def searchable(self, config: str | ConfigId | None = None) -> SearchableCorpus:
        """Return a thin :class:`SearchableCorpus` bound to one config's index."""
        cid = self._resolve_config(config)
        cfg = self._configs[cid]
        return SearchableCorpus(cfg.collection, cfg.embedder, config=cid)

    # -- introspection ------------------------------------------------------

    @property
    def configs(self) -> dict[str, ConfigId]:
        """The registered configs as a ``{name: config_id}`` mapping."""
        return dict(self._names)

    def pipeline(self, config: str | ConfigId | None = None) -> PipelineSpec:
        """The :class:`~ef.config.PipelineSpec` of a registered config."""
        return self._configs[self._resolve_config(config)].pipeline

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}: {len(self.corpus)} source(s), "
            f"{len(self._configs)} config(s), {len(self.graph)} artifact(s)>"
        )

    # -- internals ----------------------------------------------------------

    def _resolve_config(self, config: str | ConfigId | None) -> ConfigId:
        """Resolve a config name / id / ``None`` to a registered :class:`~ef.config.ConfigId`."""
        if not self._configs:
            raise ValueError(
                "No config registered. Pass an `embedder` to SourceManager(...) "
                "or call register_config(...) before materializing/searching."
            )
        if config is None:
            if self._default_config is not None:
                return self._default_config
            if len(self._configs) == 1:
                return next(iter(self._configs))
            raise ValueError(
                f"Several configs are registered ({sorted(self._names)}); "
                f"pass config= to choose one."
            )
        if config in self._configs:
            return config  # already a ConfigId
        if config in self._names:
            return self._names[config]
        raise KeyError(f"Unknown config {config!r}. Registered: {sorted(self._names)}.")

    def _collection_for(self, name: str, cid: ConfigId) -> Any:
        """Get or create the ``vd`` collection backing config ``cid``."""
        if self._explicit_collection is not None:
            if self._configs and cid not in self._configs:
                raise ValueError(
                    "This SourceManager was given an explicit `vd` collection, "
                    "which holds one config's vectors; register only one config "
                    "(or pass a `vd` client / store= to branch configs)."
                )
            return self._explicit_collection
        collection_name = f"ef:{name}"
        try:
            return self._client.get_collection(collection_name)
        except KeyError:
            return self._client.create_collection(collection_name)


def _unused_embedding_model(text: str) -> list[float]:
    """A ``vd`` ``embedding_model`` that must never be called.

    ``ef`` always supplies vectors to ``vd`` (documents on write, the query
    vector on search), so ``vd``'s own ``embedding_model`` is dead code in
    ``ef``'s flow. If it *is* reached, that is a bug — fail loudly rather than
    embed silently with the wrong model.
    """
    raise RuntimeError(
        "vd's embedding_model was called, but ef supplies all vectors itself. "
        "This indicates a document or query reached vd without a vector."
    )


def _open_store(store: Any) -> tuple[Any, Any]:
    """Resolve a ``store`` argument into a ``(vd_client, explicit_collection)`` pair.

    Exactly one of the two is non-``None``: a client (collections are created on
    demand, one per config) or a single pre-made collection.
    """
    if store is None or isinstance(store, str):
        import vd

        client = vd.connect(store or "memory", embedding_model=_unused_embedding_model)
        return client, None
    if hasattr(store, "create_collection"):  # a vd client
        return store, None
    if hasattr(store, "search") and hasattr(store, "__setitem__"):  # a vd collection
        return None, store
    raise TypeError(
        f"Cannot interpret {store!r} as a vector store. Pass None, a backend-name "
        f"string, a vd client, or a vd collection."
    )


# ---------------------------------------------------------------------------
# The one-shot light path
# ---------------------------------------------------------------------------


def ingest(
    sources: Any,
    *,
    segmenter: Any = None,
    embedder: Any = None,
    store: Any = None,
    cache: MutableMapping[ArtifactId, Any] | None = None,
) -> SearchableCorpus:
    """Index ``sources`` and return a ready-to-search :class:`SearchableCorpus` — one call.

    The light path: corpus in, ``search``-ready object out, every component
    defaulted. ``ingest(['a', 'b', 'c']).search('query')`` is the whole story.
    It builds a single-config :class:`SourceManager`, :meth:`~SourceManager.materialize`\\ s
    it, and hands back the thin :class:`SearchableCorpus`.

    Args:
        sources: the corpus — coerced by :func:`~ef.corpus.as_corpus` (a mapping,
            a directory path, an iterable of strings/mappings, or ``None``).
        segmenter: the segmenter — :func:`~ef.segmenter_adapters.as_segmenter`
            coerces it (``None`` → the recursive default).
        embedder: the embedder — :func:`~ef.embedder_adapters.as_embedder`
            coerces it; ``None`` → :data:`DEFAULT_EMBEDDER` (needs ``ef[st]``).
            Pass an explicit embedder for an offline / dependency-free run.
        store: the vector store — ``None`` → an in-memory ``vd`` backend. See
            :class:`SourceManager` for the other accepted forms.
        cache: the artifact-graph value cache — any ``MutableMapping``.

    Returns:
        a :class:`SearchableCorpus` over the freshly indexed corpus.

    See the module docstring for a complete offline example.
    """
    manager = SourceManager(
        sources, segmenter=segmenter, embedder=embedder, store=store, cache=cache
    )
    manager.materialize()
    return manager.searchable()
