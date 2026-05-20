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
- **Keeping it fresh** — :meth:`SourceManager.diagnose` reports how far the
  index has drifted from the corpus (the four staleness conditions);
  :meth:`SourceManager.refresh` re-syncs it. Pass ``auto_refresh=True`` to keep
  the index live as the corpus is edited. See :mod:`ef.diagnostics` and
  :mod:`ef.refresh`.

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

And the true light path — no ``embedder`` at all, nothing to install beyond
``ef`` itself (the default is the dependency-free
:class:`~ef.embedders.HashingEmbedder`):

>>> idx = ingest(['apple orchard', 'ocean breeze', 'green meadow'])
>>> idx.search('ocean breeze', limit=1)[0].segment['text']
'ocean breeze'
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from ef.artifact_graph import ArtifactGraph, ArtifactId, producer_spec
from ef.config import ConfigId, PipelineSpec, config_id
from ef.corpus import (
    ChangeDetectingCorpus,
    ChangeEvent,
    CorpusDiff,
    as_corpus,
    content_hash,
)
from ef.diagnostics import StalenessReport, diagnose, indexed_state
from ef.embedder_adapters import as_embedder
from ef.embedders import Embedder
from ef.ops import embed_step, segment_step
from ef.refresh import (
    RefreshMode,
    RefreshReport,
    delete_source_documents,
    plan_refresh,
    prune_dead_leaves,
    refresh_on_change,
)
from ef.segmenter_adapters import as_segmenter
from ef.segments import Segment, make_segment

__all__ = [
    "DEFAULT_EMBEDDER",
    "SearchHit",
    "SearchableCorpus",
    "SourceManager",
    "hits_to_segments",
    "ingest",
]


#: The model name :func:`ingest` / :class:`SourceManager` resolve an absent
#: ``embedder`` to — :class:`~ef.embedders.HashingEmbedder`, the dependency-free
#: default (numpy only, no extra to install). It is a lexical embedder; pass an
#: explicit ``embedder`` (``"st:..."``, ``"openai:..."``, …) for neural
#: semantic quality.
DEFAULT_EMBEDDER = "hashing"

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


def hits_to_segments(hits: Iterable[SearchHit]) -> list[Segment]:
    """Project :class:`SearchHit`\\ s to plain :class:`~ef.segments.Segment`\\ s.

    The shape :meth:`SearchableCorpus.retrieve` / :meth:`SourceManager.retrieve`
    hand to an external RAG/agent framework — the RAG-plug-in surface (design
    notes §F5). A :class:`SearchHit` carries an ``ef``-specific ``score`` and
    ``source_id``; a :class:`~ef.segments.Segment` is a plain ``TypedDict`` an
    external framework already understands (and trivially adapts to a LangChain
    ``Document`` / LlamaIndex ``TextNode`` / a Ragas ``retrieved_contexts``
    ``list[str]`` via ``[s["text"] for s in segments]``).

    The similarity ``score`` is dropped — :class:`~ef.segments.Segment` is the
    interchange type and stays pure of a result-only key (call :meth:`search`
    when the score matters). Provenance is *not* lost: each hit's ``source_id``
    is folded into the segment's ``metadata`` under the conventional ``"source"``
    key (one of :data:`~ef.segments.PROMOTED_METADATA_KEYS`), so a plain segment
    still records which source document it was retrieved from. A segment that
    already carries ``metadata["source"]`` keeps its own value.

    >>> hit = SearchHit(segment={'text': 'hi', 'id': 'x'}, score=0.9, source_id='doc-1')
    >>> segs = hits_to_segments([hit])
    >>> segs[0]['text'], segs[0]['metadata']['source']
    ('hi', 'doc-1')
    >>> 'score' in segs[0]                          # the score is not a Segment key
    False
    """
    segments: list[Segment] = []
    for hit in hits:
        segment: Segment = dict(hit.segment)  # type: ignore[assignment]
        if hit.source_id is not None:
            metadata = dict(segment.get("metadata") or {})
            metadata.setdefault("source", hit.source_id)
            segment["metadata"] = metadata
        segments.append(segment)
    return segments


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
    ) -> list[Segment]:
        """Retrieve ranked :class:`~ef.segments.Segment`\\ s — the RAG-plug-in surface.

        The name an external RAG/agent framework reaches for. Where
        :meth:`search` returns scored :class:`SearchHit`\\ s (for inspecting the
        ranking), ``retrieve`` returns **plain segments** in rank order — clean
        context to hand to an LLM, with no ``ef``-specific type to learn (design
        notes §F5). ``ef`` returns the context; it does *not* synthesize answers.
        See :func:`hits_to_segments` for how the score is dropped and the
        ``source_id`` provenance is preserved in ``metadata["source"]``.

        >>> # idx.retrieve('a query') -> [{'text': ..., 'id': ..., 'metadata': ...}, ...]
        """
        return hits_to_segments(self.search(query, limit=limit, filter=filter))

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
        auto_refresh: when ``True``, the corpus is wrapped in a
            :class:`~ef.corpus.ChangeDetectingCorpus` and every edit made
            *through* it is incrementally re-indexed into each materialized
            config — the index stays live without an explicit
            :meth:`refresh`. :meth:`scan` then also picks up out-of-band edits.

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
        auto_refresh: bool = False,
    ) -> None:
        self.corpus = as_corpus(corpus)
        self.graph = ArtifactGraph(store=cache)
        self._client, self._explicit_collection = _open_store(store)
        self._configs: dict[ConfigId, _Config] = {}
        self._names: dict[str, ConfigId] = {}
        self._default_config: ConfigId | None = None
        #: Configs that have been materialized at least once — auto-refresh
        #: only touches these (a never-indexed config is filled by materialize).
        self._materialized: set[ConfigId] = set()
        self._auto_refresh = False
        if embedder is not None:
            self.register_config("default", segmenter=segmenter, embedder=embedder)
        if auto_refresh:
            self._enable_auto_refresh()

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

    def materialize(
        self,
        config: str | ConfigId | None = None,
        *,
        sources: Iterable[str] | None = None,
    ) -> dict[str, Any]:
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
            sources: an optional subset of source ids to index — ``None`` (the
                default) indexes the whole corpus. :meth:`refresh` uses this to
                re-index only the sources that changed; ids not in the corpus
                are skipped.

        Returns:
            a report ``dict`` with ``configs`` / ``sources`` / ``segments`` counts.
        """
        configs = (
            list(self._configs) if config is None else [self._resolve_config(config)]
        )
        source_ids = None if sources is None else tuple(sources)
        total_segments = 0
        for cid in configs:
            total_segments += self._materialize_config(cid, sources=source_ids)
        return {
            "configs": len(configs),
            "sources": len(self.corpus) if source_ids is None else len(source_ids),
            "segments": total_segments,
        }

    def _materialize_config(
        self, cid: ConfigId, *, sources: Iterable[str] | None = None
    ) -> int:
        """Index the corpus (or a ``sources`` subset) through config ``cid``.

        Returns the segment count written. Records ``cid`` as materialized so
        auto-refresh knows to keep it live.
        """
        cfg = self._configs[cid]
        pipeline = cfg.pipeline
        written = 0
        source_ids = self.corpus if sources is None else sources
        for source_id in source_ids:
            if source_id not in self.corpus:
                continue  # a `sources` entry no longer in the corpus — skip
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
        self._materialized.add(cid)
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

    # -- refresh & diagnostics ----------------------------------------------

    def diagnose(self, config: str | ConfigId | None = None) -> StalenessReport:
        """Report how far one config's index has drifted from the corpus.

        Computes the four staleness conditions — ``orphan`` / ``missing`` /
        ``stale`` / ``misconfigured`` — by comparing the config's ``vd``
        collection against the current corpus (see :mod:`ef.diagnostics`).
        Read-only: it changes nothing. :meth:`refresh` acts on the result.

        >>> sm = SourceManager({'a': 'alpha', 'b': 'beta'}, embedder='hashing')
        >>> _ = sm.materialize()
        >>> bool(sm.diagnose())                  # freshly indexed — in sync
        False
        >>> sm.corpus['a'] = 'alpha rewritten'
        >>> sm.diagnose().stale
        ('a',)
        """
        cid = self._resolve_config(config)
        cfg = self._configs[cid]
        # `diagnose` here is the module-level function from ef.diagnostics.
        return diagnose(self.corpus, cfg.collection, cid)

    def refresh(
        self,
        config: str | ConfigId | None = None,
        *,
        sources: Iterable[str] | None = None,
        mode: RefreshMode = "full",
    ) -> RefreshReport:
        """Re-sync one config's ``vd`` index with the current corpus.

        The explicit-refresh entry point. It :meth:`diagnose`\\ s the config,
        turns the staleness report into a :class:`~ef.refresh.RefreshPlan` for
        the chosen ``mode`` (:func:`~ef.refresh.plan_refresh`), deletes the
        stale/orphan documents, re-:meth:`materialize`\\ s what changed, and
        prunes the dead leaves out of the :class:`~ef.artifact_graph.ArtifactGraph`.
        Because it re-reads and re-hashes the corpus, it catches out-of-band
        edits without any change-detection wrapper.

        Args:
            config: the config to refresh (name / id / ``None`` for the default
                or sole config).
            sources: an optional subset of source ids to refresh — sources
                outside it are left untouched. Pair it with ``mode='scoped_full'``
                (or ``'incremental'``); ``'full'`` rejects it.
            mode: one of :data:`~ef.refresh.REFRESH_MODES` — ``'none'`` (index
                new/changed content, delete nothing), ``'incremental'`` (also
                replace changed sources), ``'full'`` (also delete orphans — the
                corpus is authoritative), ``'scoped_full'`` (``full`` over the
                ``sources`` subset). Default ``'full'``.

        Returns:
            a :class:`~ef.refresh.RefreshReport`.

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
        """
        if mode == "full" and sources is not None:
            raise ValueError(
                "full mode refreshes the whole corpus; pass mode='scoped_full' "
                "to refresh a subset of sources."
            )
        cid = self._resolve_config(config)
        cfg = self._configs[cid]
        report = diagnose(self.corpus, cfg.collection, cid)
        indexed = indexed_state(cfg.collection)
        plan = plan_refresh(report, mode=mode, scope=sources)

        documents_removed = delete_source_documents(
            cfg.collection, indexed, plan.to_delete
        )
        documents_written = 0
        if plan.to_materialize:
            documents_written = self._materialize_config(
                cid, sources=plan.to_materialize
            )
        artifacts_removed = prune_dead_leaves(self.graph, self._live_leaves())

        materialized = set(plan.to_materialize)
        return RefreshReport(
            config=cid,
            mode=mode,
            added=tuple(sorted(set(report.missing) & materialized)),
            modified=tuple(sorted(set(report.needs_reindexing) & materialized)),
            deleted=tuple(sorted(set(report.orphan) & set(plan.to_delete))),
            unchanged=report.fresh,
            documents_written=documents_written,
            documents_removed=documents_removed,
            artifacts_removed=artifacts_removed,
        )

    def rebuild(self, config: str | ConfigId | None = None) -> RefreshReport:
        """Drop one config's index entirely and re-index the whole corpus.

        The heavy hammer: every document in the config's ``vd`` collection is
        deleted, then the whole corpus is re-:meth:`materialize`\\ d from
        scratch. Use it to recover from a config whose ``vd`` documents were
        produced by an earlier pipeline (every source ``misconfigured``), or
        whenever a guaranteed-clean index is wanted over the surgical
        :meth:`refresh`. Every source is reported as ``added`` — the collection
        was emptied and rebuilt.

        Returns:
            a :class:`~ef.refresh.RefreshReport`.
        """
        cid = self._resolve_config(config)
        cfg = self._configs[cid]
        documents_removed = 0
        for doc_id in list(cfg.collection):
            del cfg.collection[doc_id]
            documents_removed += 1
        documents_written = self._materialize_config(cid)
        artifacts_removed = prune_dead_leaves(self.graph, self._live_leaves())
        return RefreshReport(
            config=cid,
            mode="full",
            added=tuple(sorted(str(s) for s in self.corpus)),
            documents_written=documents_written,
            documents_removed=documents_removed,
            artifacts_removed=artifacts_removed,
        )

    def gc_orphans(self, config: str | ConfigId | None = None) -> int:
        """Delete the documents of sources that left the corpus — return the count.

        The garbage-collection slice of :meth:`refresh`: it removes only
        **orphan** documents (a source gone from the corpus) and prunes the
        dead leaves out of the :class:`~ef.artifact_graph.ArtifactGraph`,
        touching nothing else. ``gc_orphans()`` is the orphan-deletion half of
        ``refresh(mode='full')``.
        """
        cid = self._resolve_config(config)
        cfg = self._configs[cid]
        report = diagnose(self.corpus, cfg.collection, cid)
        indexed = indexed_state(cfg.collection)
        removed = delete_source_documents(cfg.collection, indexed, report.orphan)
        prune_dead_leaves(self.graph, self._live_leaves())
        return removed

    def lineage(self, key: ArtifactId) -> frozenset[ArtifactId]:
        """The artifacts ``key`` was produced from — its provenance.

        A thin pass-through to :meth:`ArtifactGraph.ancestors
        <ef.artifact_graph.ArtifactGraph.ancestors>`: "what produced this
        vector / segment?". The leaves of the returned set are the source
        content hashes the artifact ultimately derives from.
        """
        return self.graph.ancestors(key)

    def scan(self) -> CorpusDiff:
        """Re-scan the corpus for out-of-band edits — needs ``auto_refresh=True``.

        Picks up changes made to the backing store *directly* (a file edited on
        disk, an S3 object replaced) rather than through the corpus wrapper.
        When the manager was built with ``auto_refresh=True`` the corpus is a
        :class:`~ef.corpus.ChangeDetectingCorpus`; this calls its
        :meth:`~ef.corpus.ChangeDetectingCorpus.scan`, which fires a
        :class:`~ef.corpus.ChangeEvent` per drift — each one auto-applied to
        every materialized config. Returns the :class:`~ef.corpus.CorpusDiff`.

        For a large bulk of out-of-band edits, prefer :meth:`refresh` (one
        corpus pass) over ``scan`` (one collection pass per changed source).

        Raises:
            TypeError: if the corpus is not change-detecting (the manager was
                not built with ``auto_refresh=True``).
        """
        if not isinstance(self.corpus, ChangeDetectingCorpus):
            raise TypeError(
                "scan() needs a change-detecting corpus — construct the "
                "SourceManager with auto_refresh=True."
            )
        return self.corpus.scan()

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
    ) -> list[Segment]:
        """Retrieve ranked :class:`~ef.segments.Segment`\\ s — the RAG-plug-in surface.

        The :class:`SourceManager` counterpart of
        :meth:`SearchableCorpus.retrieve`: plain segments in rank order, the
        clean context shape an external RAG/agent framework consumes (``ef``
        does not synthesize answers). :meth:`search` is the scored counterpart.
        """
        return hits_to_segments(
            self.search(query, config=config, limit=limit, filter=filter)
        )

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

    def _live_leaves(self) -> set[ArtifactId]:
        """The content hashes of every current corpus source — the live graph leaves."""
        return {content_hash(self.corpus[source_id]) for source_id in self.corpus}

    def _enable_auto_refresh(self) -> None:
        """Wrap the corpus for change detection and wire incremental auto-refresh.

        Called from ``__init__`` when ``auto_refresh=True``. If the corpus is
        already a :class:`~ef.corpus.ChangeDetectingCorpus`, its ``on_change``
        is *chained* (the existing callback still fires); otherwise the corpus
        is wrapped in a fresh one. The handler is :func:`~ef.refresh.refresh_on_change`.
        """
        handler = refresh_on_change(self)
        if isinstance(self.corpus, ChangeDetectingCorpus):
            previous = self.corpus.on_change

            def chained(event: ChangeEvent) -> None:
                previous(event)
                handler(event)

            self.corpus.on_change = chained
        else:
            self.corpus = ChangeDetectingCorpus(self.corpus, on_change=handler)
        self._auto_refresh = True

    def _apply_change(self, event: ChangeEvent) -> None:
        """Incrementally re-index one corpus change across every materialized config.

        The per-event worker behind :func:`~ef.refresh.refresh_on_change` (the
        ``auto_refresh=True`` seam). An ``added`` source is indexed, a
        ``modified`` one is replaced (delete-then-index), a ``deleted`` one is
        removed; a now-dead leaf is then cascaded out of the graph.
        """
        for cid in list(self._materialized):
            cfg = self._configs.get(cid)
            if cfg is None:  # the config was dropped since it was materialized
                continue
            if event.kind in ("modified", "deleted"):
                self._delete_source_from_collection(cfg.collection, event.source_id)
            if event.kind in ("added", "modified") and event.source_id in self.corpus:
                self._materialize_config(cid, sources=[event.source_id])
        if event.kind in ("modified", "deleted") and event.old_hash:
            if event.old_hash not in self._live_leaves():
                self.graph.delete_cascade(event.old_hash)

    @staticmethod
    def _delete_source_from_collection(collection: Any, source_id: str) -> int:
        """Delete every document in ``collection`` whose ``source_id`` matches."""
        removed = 0
        for doc_id in list(collection):
            document = collection[doc_id]
            metadata = getattr(document, "metadata", None) or {}
            if metadata.get("source_id") == source_id:
                del collection[doc_id]
                removed += 1
        return removed

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
            coerces it; ``None`` → :data:`DEFAULT_EMBEDDER`, the dependency-free
            :class:`~ef.embedders.HashingEmbedder` (so this call needs nothing
            beyond ``pip install ef``). Pass an explicit embedder — e.g.
            ``"st:all-MiniLM-L6-v2"`` — for neural semantic quality.
        store: the vector store — ``None`` → an in-memory ``vd`` backend. See
            :class:`SourceManager` for the other accepted forms.
        cache: the artifact-graph value cache — any ``MutableMapping``.

    Returns:
        a :class:`SearchableCorpus` over the freshly indexed corpus.

    See the module docstring for a complete offline example.
    """
    # The light path always indexes, so it always needs a concrete embedder —
    # resolve ``None`` to :data:`DEFAULT_EMBEDDER` here rather than relying on
    # ``SourceManager``, which deliberately registers *no* config when given no
    # embedder (its explicit multi-config mode).
    manager = SourceManager(
        sources,
        segmenter=segmenter,
        embedder=_coerce_embedder(embedder),
        store=store,
        cache=cache,
    )
    manager.materialize()
    return manager.searchable()
