"""Refresh — keeping an indexed corpus in sync as its sources change.

Phase 5 indexes a corpus *once*. This module makes the index **maintainable**:
when sources are added, edited or deleted, refresh brings the ``vd`` index back
into agreement with the corpus. It is the acting counterpart of the read-only
:mod:`ef.diagnostics` — diagnostics says *what drifted*, refresh *fixes it*.

The one idea, restated from :mod:`ef.artifact_graph`:

    Refresh = diff source hashes → delete the stale/orphan documents →
    re-``materialize`` what changed.

The ``vd`` collection is the single source of truth for *what is indexed*; the
:class:`~ef.artifact_graph.ArtifactGraph` is only a *cache*. So refresh has two
kinds of effect, and they are not equally critical:

- **Collection edits are correctness.** Stale and orphaned documents must be
  removed precisely (by ``source_id``), or a search returns wrong hits.
- **Graph pruning is housekeeping.** A content-addressed artifact keyed by a
  now-dead content hash is never *served* again (its source's content — hence
  its leaf id — has moved on); it is merely wasted cache. So :func:`prune_dead_leaves`
  is best-effort and only ever cascades leaves verified dead — no surviving
  source still hashes to them.

The four refresh modes (``ef_design_notes.md`` §8.5, mirroring LangChain's
indexing API) differ only in **how much they delete**:

============= =================================================================
Mode          Behavior
============= =================================================================
``none``      pure dedup — index new/changed content, delete nothing
``incremental`` replace changed sources (delete-then-re-index); keep the rest
``full``      the corpus is authoritative — also delete orphans (absent sources)
``scoped_full`` like ``full``, restricted to a ``sources=`` subset
============= =================================================================

:func:`plan_refresh` turns a :class:`~ef.diagnostics.StalenessReport` + a mode
into a :class:`RefreshPlan` (which sources to (re-)materialize, which documents
to delete) — a pure function, the testable heart of the policy. The orchestration
that applies a plan lives on :class:`~ef.source_manager.SourceManager`
(:meth:`~ef.source_manager.SourceManager.refresh`).

**Explicit vs. auto.** :meth:`SourceManager.refresh
<ef.source_manager.SourceManager.refresh>` is the *explicit* path — it re-reads
and re-hashes the corpus, so it catches out-of-band edits on its own.
:func:`refresh_on_change` is the *auto* path: a handler you hand to a
:class:`~ef.corpus.ChangeDetectingCorpus`'s ``on_change`` so each edit is
incrementally re-indexed as it happens (``SourceManager(auto_refresh=True)``
wires this for you).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ef.config import ConfigId
from ef.diagnostics import IndexedSource, StalenessReport

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from ef.artifact_graph import ArtifactGraph
    from ef.corpus import ChangeEvent

__all__ = [
    "RefreshMode",
    "REFRESH_MODES",
    "RefreshPlan",
    "RefreshReport",
    "plan_refresh",
    "delete_source_documents",
    "prune_dead_leaves",
    "refresh_on_change",
]


#: The four refresh modes — see the module docstring and ``ef_design_notes.md``
#: §8.5. They differ only in how much they delete.
RefreshMode = Literal["none", "incremental", "full", "scoped_full"]

#: The accepted :data:`RefreshMode` values, for validation.
REFRESH_MODES: tuple[RefreshMode, ...] = ("none", "incremental", "full", "scoped_full")


# ---------------------------------------------------------------------------
# The plan and the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefreshPlan:
    """What a refresh will do — :func:`plan_refresh`'s output.

    The two fields are disjoint in *intent* but may overlap in membership: a
    stale source is in **both** — its old documents are deleted *and* it is
    re-materialized. An orphan is only in ``to_delete``; a brand-new source only
    in ``to_materialize``.

    Attributes:
        to_materialize: source ids to (re-)run through the pipeline.
        to_delete: source ids whose currently-indexed documents are removed
            (before re-materializing, for the ones that also reappear above).
    """

    to_materialize: tuple[str, ...] = ()
    to_delete: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """True iff the plan does anything at all."""
        return bool(self.to_materialize or self.to_delete)


@dataclass(frozen=True, slots=True)
class RefreshReport:
    """The outcome of a refresh — what :meth:`SourceManager.refresh` returns.

    ``added`` / ``modified`` / ``deleted`` / ``unchanged`` partition the
    source ids the refresh considered (mirroring :class:`~ef.corpus.CorpusDiff`
    field names): newly indexed, re-indexed (was stale/misconfigured), removed
    (orphan), and already-fresh-left-alone. The three counts report the work
    done. Truthiness reports whether anything changed.

    >>> r = RefreshReport(config='c0', mode='full', added=('a',), modified=('b',))
    >>> bool(r), r.changed
    (True, ('a', 'b'))
    >>> bool(RefreshReport(config='c0', mode='full', unchanged=('a',)))
    False
    """

    config: ConfigId
    mode: RefreshMode
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    documents_written: int = 0
    documents_removed: int = 0
    artifacts_removed: int = 0

    @property
    def changed(self) -> tuple[str, ...]:
        """The source ids that were added, modified or deleted."""
        return self.added + self.modified + self.deleted

    def __bool__(self) -> bool:
        """True iff anything was added, modified or deleted."""
        return bool(self.added or self.modified or self.deleted)


# ---------------------------------------------------------------------------
# The policy — a pure function from (report, mode) to a plan
# ---------------------------------------------------------------------------


def plan_refresh(
    report: StalenessReport,
    *,
    mode: RefreshMode = "full",
    scope: Iterable[str] | None = None,
) -> RefreshPlan:
    """Turn a :class:`~ef.diagnostics.StalenessReport` into a :class:`RefreshPlan`.

    The pure, side-effect-free core of the refresh policy. Every source that is
    ``missing``, ``stale`` or ``misconfigured`` is (re-)materialized; what gets
    *deleted* is what the ``mode`` dictates:

    - ``none`` — delete nothing (stale content is added alongside the old);
    - ``incremental`` — delete the stale/misconfigured sources' old documents;
    - ``full`` / ``scoped_full`` — also delete orphans.

    Args:
        report: the staleness report to act on.
        mode: one of :data:`REFRESH_MODES`.
        scope: an optional subset of source ids the plan is restricted to —
            sources outside it are left entirely alone. ``None`` means the whole
            report. (With ``scope=None``, ``full`` and ``scoped_full`` coincide.)

    Raises:
        ValueError: if ``mode`` is not a recognized :data:`RefreshMode`.

    >>> from ef.diagnostics import StalenessReport
    >>> report = StalenessReport(config='c0', orphan=('x',), missing=('a',),
    ...                          stale=('b',), misconfigured=('c',), fresh=('d',))
    >>> plan_refresh(report, mode='full').to_materialize
    ('a', 'b', 'c')
    >>> plan_refresh(report, mode='full').to_delete
    ('b', 'c', 'x')
    >>> plan_refresh(report, mode='incremental').to_delete
    ('b', 'c')
    >>> plan_refresh(report, mode='none').to_delete
    ()
    >>> plan_refresh(report, mode='scoped_full', scope=['a', 'b']).to_delete
    ('b',)
    """
    if mode not in REFRESH_MODES:
        raise ValueError(f"Unknown refresh mode {mode!r}. Use one of {REFRESH_MODES}.")
    missing = set(report.missing)
    reindex = set(report.stale) | set(report.misconfigured)
    orphan = set(report.orphan)
    if scope is not None:
        in_scope = set(scope)
        missing &= in_scope
        reindex &= in_scope
        orphan &= in_scope

    to_materialize = missing | reindex
    if mode == "none":
        to_delete: set[str] = set()
    elif mode == "incremental":
        to_delete = set(reindex)
    else:  # "full" / "scoped_full"
        to_delete = reindex | orphan

    return RefreshPlan(
        to_materialize=tuple(sorted(to_materialize)),
        to_delete=tuple(sorted(to_delete)),
    )


# ---------------------------------------------------------------------------
# The effects — applied against an explicitly passed collection / graph
# ---------------------------------------------------------------------------


def delete_source_documents(
    collection: Any,
    indexed: dict[str, IndexedSource],
    source_ids: Iterable[str],
) -> int:
    """Delete every ``vd`` document belonging to the given ``source_ids``.

    The correctness half of a refresh: a stale or orphaned source's documents
    must leave the collection. ``indexed`` — an :func:`~ef.diagnostics.indexed_state`
    grouping — supplies each source's document ids, so no per-source scan is
    needed. A document already gone is silently tolerated (idempotent).

    Args:
        collection: the ``vd`` collection (a ``MutableMapping``) to delete from.
        indexed: the ``{source_id: IndexedSource}`` map of what is indexed.
        source_ids: the sources whose documents to remove.

    Returns:
        the number of documents actually deleted.

    >>> from ef.diagnostics import IndexedSource
    >>> collection = {'s1#0': ..., 's1#1': ..., 's2#0': ...}
    >>> indexed = {
    ...     's1': IndexedSource('s1', ('s1#0', 's1#1'), frozenset(), frozenset()),
    ...     's2': IndexedSource('s2', ('s2#0',), frozenset(), frozenset()),
    ... }
    >>> delete_source_documents(collection, indexed, ['s1'])
    2
    >>> sorted(collection)
    ['s2#0']
    """
    removed = 0
    for source_id in source_ids:
        entry = indexed.get(source_id)
        if entry is None:
            continue
        for doc_id in entry.doc_ids:
            try:
                del collection[doc_id]
            except KeyError:
                continue
            removed += 1
    return removed


def prune_dead_leaves(graph: "ArtifactGraph", live_leaves: Iterable[str]) -> int:
    """Cascade out of ``graph`` every leaf no live source still hashes to.

    The housekeeping half of a refresh. A graph *leaf* (a node with a cached
    value but no producer recipe) is an L0 source keyed by its content hash.
    After a refresh, the set of leaves the graph *should* hold is exactly the
    content hashes of the current corpus; any other leaf is dead — the source
    that produced it was edited (new content → new leaf) or deleted. Each dead
    leaf and its whole downstream cone (its segment + embed artifacts) is
    :meth:`~ef.artifact_graph.ArtifactGraph.delete_cascade`\\ d.

    This is purely a cache reclaim: a dead leaf is never *served* (nothing
    addresses it any more), so pruning it can never change a result — only free
    memory. Dead leaves' cones are disjoint (each produced artifact traces to
    exactly one leaf), so the removed-count is exact.

    Args:
        graph: the :class:`~ef.artifact_graph.ArtifactGraph` to prune.
        live_leaves: the content hashes that are still live (the current
            corpus's source hashes).

    Returns:
        the number of artifacts (leaves + produced) removed.

    >>> from ef import ArtifactGraph, producer_spec
    >>> graph = ArtifactGraph()
    >>> _ = graph.register_op('id', lambda x: x)
    >>> _ = graph.put('live', 'L')
    >>> _ = graph.put('dead', 'D')
    >>> artifact = graph.add(producer_spec('id', 'dead', op_version='1'))
    >>> _ = graph.materialize(artifact)
    >>> prune_dead_leaves(graph, ['live'])      # the 'dead' leaf + its artifact
    2
    >>> 'dead' in graph, 'live' in graph
    (False, True)
    """
    live = set(live_leaves)
    dead = [
        key
        for key in list(graph.store)
        if key not in graph.producers and key not in live
    ]
    removed = 0
    for leaf in dead:
        removed += len(graph.delete_cascade(leaf))
    return removed


# ---------------------------------------------------------------------------
# The auto-refresh seam
# ---------------------------------------------------------------------------


def refresh_on_change(manager: Any) -> Callable[["ChangeEvent"], None]:
    """Build a :class:`~ef.corpus.ChangeEvent` handler that auto-refreshes ``manager``.

    The *auto* refresh seam. The returned callable is meant to be a
    :class:`~ef.corpus.ChangeDetectingCorpus`'s ``on_change`` — each detected
    edit (through the wrapper, or surfaced out-of-band by ``scan()``) is
    incrementally applied to every materialized config of ``manager``:
    an ``added`` source is indexed, a ``modified`` one is re-indexed
    (delete-then-add), a ``deleted`` one is removed.

    :class:`~ef.source_manager.SourceManager` wires this automatically when
    constructed with ``auto_refresh=True``; call this directly only to attach a
    manager to a corpus you build yourself.

    Args:
        manager: a :class:`~ef.source_manager.SourceManager` (duck-typed — any
            object with an ``_apply_change(event)`` method).

    Returns:
        a ``Callable[[ChangeEvent], None]`` for ``on_change``.
    """

    def handler(event: "ChangeEvent") -> None:
        manager._apply_change(event)

    return handler
