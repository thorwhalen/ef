"""Staleness diagnostics — has an indexed corpus drifted from its source?

Phase 5 wired ``corpus → segment → embed → vd`` and recorded provenance on
every ``vd`` document: ``source_id``, ``source_hash`` and ``config_hash`` ride
in the document metadata (see :data:`ef.source_manager._RESERVED_METADATA_KEYS`).
This module reads that provenance back and answers a single question — *is the
index still a faithful picture of the corpus?* — by computing the **four
staleness conditions** (``ef_use_cases.md`` §C):

================ =============================================================
Condition        Definition
================ =============================================================
**orphan**       vectors exist, but the source is gone from the corpus
**missing**      the source exists, but nothing is indexed for it
**stale**        the source changed since its vectors were computed
**misconfigured** the vectors were built by a *different* pipeline (config)
================ =============================================================

The detection is a set comparison, exactly as the design intends — the
``vd`` collection is the single source of truth for *what is indexed*, the
corpus for *what should be indexed*, and the current
:class:`~ef.config.ConfigId` for *how*:

- ``orphan`` / ``missing`` partition the symmetric difference of the two id
  sets;
- ``stale`` compares each indexed source's recorded ``source_hash`` against a
  freshly recomputed :func:`~ef.corpus.content_hash`;
- ``misconfigured`` compares each document's recorded ``config_hash`` against
  the config being diagnosed.

This module is **read-only** — it never mutates the corpus, the collection or
the graph. Acting on a :class:`StalenessReport` is :mod:`ef.refresh`'s job.

Example — diagnose a config whose corpus has since drifted:

>>> from ef import SourceManager
>>> sm = SourceManager({'a': 'alpha', 'b': 'beta'}, embedder='hashing')
>>> _ = sm.materialize()
>>> sm.corpus['b'] = 'beta changed'          # an edit, behind the index's back
>>> sm.corpus['c'] = 'gamma'                 # a brand-new source
>>> del sm.corpus['a']                       # and a deletion
>>> report = sm.diagnose()
>>> report.orphan, report.missing, report.stale
(('a',), ('c',), ('b',))
>>> bool(report)                             # True — the index has drifted
True
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ef.config import ConfigId
from ef.corpus import Corpus, content_hash

__all__ = [
    "IndexedSource",
    "StalenessReport",
    "indexed_state",
    "corpus_state",
    "diagnose",
]


# ---------------------------------------------------------------------------
# Reading the index's recorded state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexedSource:
    """The provenance a ``vd`` collection records for one source document.

    A source is segmented into several documents; they all carry the same
    ``source_id`` and *should* carry one ``source_hash`` / ``config_hash`` each.
    :func:`indexed_state` groups a collection's documents into one of these per
    source — collecting the hashes as **sets** so a partially re-indexed source
    (documents disagreeing on a hash) is detectable rather than papered over.

    Attributes:
        source_id: the corpus key of the source these documents were cut from.
        doc_ids: the ids of every ``vd`` document belonging to the source.
        source_hashes: the distinct ``source_hash`` values its documents record
            — a single-element set for a cleanly indexed source.
        config_hashes: likewise the distinct ``config_hash`` values.
    """

    source_id: str
    doc_ids: tuple[str, ...]
    source_hashes: frozenset[str]
    config_hashes: frozenset[str]


def indexed_state(collection: Any) -> dict[str, IndexedSource]:
    """Group a ``vd`` collection's documents by source — its recorded state.

    Iterates the collection (a ``MutableMapping`` of ``vd`` documents) and
    folds every document ``ef`` wrote into an :class:`IndexedSource` keyed by
    its ``source_id``. Documents with no ``source_id`` metadata (not written by
    ``ef``) are skipped. This is the *what is indexed* half of :func:`diagnose`.

    Args:
        collection: the ``vd`` collection backing one config.

    Returns:
        a ``{source_id: IndexedSource}`` mapping.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for doc_id in list(collection):
        document = collection[doc_id]
        metadata = getattr(document, "metadata", None) or {}
        source_id = metadata.get("source_id")
        if source_id is None:
            continue  # not an ef-written document — nothing to diagnose
        bucket = grouped.setdefault(
            source_id, {"doc_ids": [], "source_hashes": set(), "config_hashes": set()}
        )
        bucket["doc_ids"].append(doc_id)
        source_hash = metadata.get("source_hash")
        if source_hash is not None:
            bucket["source_hashes"].add(source_hash)
        config_hash = metadata.get("config_hash")
        if config_hash is not None:
            bucket["config_hashes"].add(config_hash)
    return {
        source_id: IndexedSource(
            source_id=source_id,
            doc_ids=tuple(bucket["doc_ids"]),
            source_hashes=frozenset(bucket["source_hashes"]),
            config_hashes=frozenset(bucket["config_hashes"]),
        )
        for source_id, bucket in grouped.items()
    }


def corpus_state(corpus: Corpus | Mapping[str, Any]) -> dict[str, str]:
    """The current content hash of every source in ``corpus``.

    The *what should be indexed* half of :func:`diagnose`: a fresh
    :func:`~ef.corpus.content_hash` per source — recomputed, not read from any
    registry, so it reflects out-of-band edits to the backing store. It matches
    the ``source_hash`` :class:`~ef.source_manager.SourceManager` writes into
    ``vd`` (both are ``content_hash`` of the source with no ``content_keys``).

    >>> sorted(corpus_state({'a': 'one', 'b': 'two'}))
    ['a', 'b']
    >>> corpus_state({'a': 'x'})['a'] == corpus_state({'a': 'x'})['a']
    True
    """
    return {source_id: content_hash(corpus[source_id]) for source_id in corpus}


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StalenessReport:
    """The four staleness conditions of one config — what :func:`diagnose` returns.

    Each field is a sorted tuple of ``source_id``\\ s, so a report is
    deterministic and JSON-friendly. ``orphan`` and ``missing`` are disjoint
    from the rest (a source absent from one side cannot also be compared); a
    source can be **both** ``stale`` and ``misconfigured`` (its content changed
    *and* the config did) — the conditions are reported independently.
    ``fresh`` is the complement: indexed, current and built by this config.

    Truthiness reports whether *anything is wrong* — ``bool(report)`` is
    ``False`` only for a perfectly in-sync index.

    >>> r = StalenessReport(config='c0', stale=('a',), fresh=('b',))
    >>> bool(r), r.needs_reindexing
    (True, ('a',))
    >>> bool(StalenessReport(config='c0', fresh=('a', 'b')))
    False
    """

    config: ConfigId
    orphan: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    misconfigured: tuple[str, ...] = ()
    fresh: tuple[str, ...] = ()

    @property
    def needs_reindexing(self) -> tuple[str, ...]:
        """Sources whose vectors must be recomputed — ``stale`` ∪ ``misconfigured``."""
        return tuple(sorted(set(self.stale) | set(self.misconfigured)))

    def __bool__(self) -> bool:
        """True iff anything is orphaned, missing, stale or misconfigured."""
        return bool(self.orphan or self.missing or self.stale or self.misconfigured)


def diagnose(
    corpus: Corpus | Mapping[str, Any],
    collection: Any,
    config: ConfigId,
) -> StalenessReport:
    """Compute the four staleness conditions of ``collection`` against ``corpus``.

    The pure core of ``ef``'s diagnostics — no side effects. It reads what the
    ``vd`` ``collection`` records (:func:`indexed_state`), what the ``corpus``
    currently holds (:func:`corpus_state`), and partitions every source id into
    exactly the buckets it belongs in:

    - **orphan** — in the collection, not in the corpus;
    - **missing** — in the corpus, not in the collection;
    - **stale** — in both, but the recorded ``source_hash`` ≠ the current one;
    - **misconfigured** — in both, but a recorded ``config_hash`` ≠ ``config``;
    - **fresh** — in both, content current and config matching.

    Args:
        corpus: the source corpus.
        collection: the ``vd`` collection backing the config being diagnosed.
        config: the :class:`~ef.config.ConfigId` the collection *should* hold
            — documents recording any other ``config_hash`` are misconfigured.

    Returns:
        a :class:`StalenessReport`.
    """
    indexed = indexed_state(collection)
    current = corpus_state(corpus)
    corpus_ids = set(current)
    indexed_ids = set(indexed)

    orphan = sorted(indexed_ids - corpus_ids)
    missing = sorted(corpus_ids - indexed_ids)
    stale: list[str] = []
    misconfigured: list[str] = []
    fresh: list[str] = []
    for source_id in sorted(corpus_ids & indexed_ids):
        entry = indexed[source_id]
        is_stale = entry.source_hashes != {current[source_id]}
        is_misconfigured = entry.config_hashes != {config}
        if is_stale:
            stale.append(source_id)
        if is_misconfigured:
            misconfigured.append(source_id)
        if not is_stale and not is_misconfigured:
            fresh.append(source_id)

    return StalenessReport(
        config=config,
        orphan=tuple(orphan),
        missing=tuple(missing),
        stale=tuple(stale),
        misconfigured=tuple(misconfigured),
        fresh=tuple(fresh),
    )
