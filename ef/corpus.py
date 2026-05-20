"""The ``Corpus`` abstraction — ``ef``'s source layer (L0) — and change detection.

A *corpus* is the entry point of the pipeline: the set of source documents to
be segmented, embedded and indexed. In ``ef`` a corpus is deliberately *not* a
new class — it is just a mapping::

    Corpus = MutableMapping[source_id, Source]

Any ``dol``-style mapping is a corpus: a plain ``dict`` (RAM), a filesystem
folder, an S3 bucket, an API-backed store. The backing store is swappable
because the rest of ``ef`` only ever sees a ``MutableMapping``. This module
provides:

- :data:`Source` / :data:`Corpus` — the type aliases that name the contract.
- :func:`as_corpus` — the dependency-injection seam: ``None`` / a mapping / a
  directory path / an iterable of strings → a corpus. Mirrors
  :func:`~ef.embedder_adapters.as_embedder` and
  :func:`~ef.segmenter_adapters.as_segmenter`.
- :func:`content_hash` — the content hash of a source, the basis of change
  detection. Hashes *normalized* content (via :mod:`ef.hashing`); for a
  structured source it hashes the text only, so metadata-only edits never
  trigger a spurious re-embed.
- :class:`ChangeDetectingCorpus` — a corpus *wrapper* that content-hashes
  values and reports what changed, both through-the-wrapper (via an
  ``on_change`` callback) and out-of-band (via :meth:`~ChangeDetectingCorpus.scan`).

:class:`ChangeDetectingCorpus` is the *"something changed"* half of incremental
refresh; the downstream cascade is the ``ArtifactGraph`` (a later phase). The
``on_change`` callback is the seam those two halves meet at.

Example — detect changes as they are written:

>>> events = []
>>> corpus = ChangeDetectingCorpus(on_change=events.append)
>>> corpus['doc1'] = 'hello world'
>>> corpus['doc1'] = 'hello world'      # identical content — idempotent, no event
>>> corpus['doc1'] = 'hello there'      # a real change
>>> del corpus['doc1']
>>> [(e.source_id, e.kind) for e in events]
[('doc1', 'added'), ('doc1', 'modified'), ('doc1', 'deleted')]
"""

from __future__ import annotations

import os
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from dataclasses import dataclass
from typing import Any, Literal

from ef.hashing import canonical_json, normalize_text, sha256_hex

__all__ = [
    "Source",
    "Corpus",
    "content_hash",
    "as_corpus",
    "ChangeKind",
    "ChangeEvent",
    "CorpusDiff",
    "ChangeDetectingCorpus",
]


# ---------------------------------------------------------------------------
# The contract — type aliases
# ---------------------------------------------------------------------------

#: A source document. Either plain ``text``, or a ``Mapping`` carrying a
#: ``text`` key alongside metadata — the same shape a
#: :class:`~ef.segmenters.Segmenter` accepts as input.
Source = str | Mapping[str, Any]

#: A corpus: a mapping from ``source_id`` to :data:`Source`. ``ef`` never
#: requires a dedicated class — any ``dol``-style ``MutableMapping`` (RAM,
#: filesystem, S3, API) *is* a corpus. :func:`as_corpus` builds one.
Corpus = MutableMapping[str, Source]


#: A sentinel distinct from every real value, including ``None``.
_MISSING: Any = object()


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def _source_field(source: Mapping[str, Any], key: str) -> Any:
    """Look ``key`` up at the top level of ``source``, then inside ``metadata``.

    Returns :data:`_MISSING` if the key is present in neither place — so an
    absent ``content_keys`` entry simply contributes nothing to the hash.
    """
    if key in source:
        return source[key]
    metadata = source.get("metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    return _MISSING


def content_hash(source: Any, *, content_keys: Iterable[str] | None = None) -> str:
    """Content hash of a source document — ``sha256`` of its normalized content.

    This is the basis of change detection: re-reading an unchanged source
    yields an identical hash, so a refresh that finds the same hash does
    nothing. Hashing *normalized* content (NFC, no BOM, ``\\n`` line endings —
    see :mod:`ef.hashing`) means cosmetic encoding differences never look like
    a change.

    The treatment of a ``Mapping`` source is deliberate:

    - a plain ``str`` source — the hash covers the text;
    - a ``Mapping`` with a ``text`` key — the hash covers ``source['text']``
      **only**. A change to title, tags, ACL or any other metadata does *not*
      change the hash, so it never triggers a spurious re-segment/re-embed
      (embedding is the pipeline's most expensive step). Name the metadata keys
      that *do* feed downstream in ``content_keys`` and changes to those keys
      will be detected;
    - a ``Mapping`` *without* a ``text`` key — treated as a structured document
      and hashed via canonical (sorted-key) JSON of the whole mapping;
    - ``bytes`` — hashed raw (no text normalization is possible pre-parse).

    Args:
        source: a ``str``, ``bytes``, or a ``Mapping`` (ideally with a ``text``
            key).
        content_keys: metadata keys whose values join the text in the hash.
            Each key is looked up at the top level of ``source`` or inside a
            ``source['metadata']`` sub-mapping. Use this *only* for metadata
            that genuinely affects the L1–L4 pipeline (e.g. a title prepended
            to the text before embedding); key order does not matter.

    Raises:
        TypeError: if ``source`` is not a ``str``, ``bytes`` or ``Mapping``.

    >>> content_hash('hello') == content_hash('hello')
    True
    >>> content_hash('hello') == content_hash('hello world')
    False
    >>> content_hash('a\\r\\nb') == content_hash('a\\nb')   # line endings normalized
    True
    >>> a = {'text': 'body', 'metadata': {'tags': ['x']}}
    >>> b = {'text': 'body', 'metadata': {'tags': ['y']}}
    >>> content_hash(a) == content_hash(b)                # metadata ignored by default
    True
    >>> content_hash(a, content_keys=['tags']) == content_hash(b, content_keys=['tags'])
    False
    >>> content_hash({'text': 'body'}) == content_hash('body')  # wrapper is transparent
    True
    """
    if isinstance(source, bytes):
        return sha256_hex(source)
    if isinstance(source, str):
        return sha256_hex(normalize_text(source))
    if isinstance(source, Mapping):
        if "text" in source:
            payload = normalize_text(source["text"])
            if content_keys:
                extra = {
                    key: value
                    for key in content_keys
                    if (value := _source_field(source, key)) is not _MISSING
                }
                if extra:
                    payload += canonical_json(extra)
            return sha256_hex(payload)
        # A structured document with no text body — hash it whole, canonically.
        return sha256_hex(canonical_json(dict(source)))
    raise TypeError(
        f"Cannot content-hash a {type(source).__name__}. A source must be a "
        f"str, bytes, or a Mapping (ideally with a 'text' key)."
    )


# ---------------------------------------------------------------------------
# The dependency-injection seam
# ---------------------------------------------------------------------------


def _filesystem_corpus(directory: str, **kwargs: Any) -> MutableMapping[str, str]:
    """A ``dol`` filesystem corpus over the text files under ``directory``."""
    try:
        from dol import TextFiles
    except ImportError as exc:  # pragma: no cover - dol is a core dependency
        raise ImportError(
            "as_corpus needs the `dol` package to build a filesystem corpus. "
            "Install it with: pip install dol"
        ) from exc
    return TextFiles(directory, **kwargs)


def as_corpus(x: Any = None, /, **kwargs: Any):
    """Normalize ``x`` into a corpus (``MutableMapping[source_id, Source]``).

    The single dependency-injection seam through which every ``ef`` entry point
    accepts a user-supplied corpus — the mirror of
    :func:`~ef.embedder_adapters.as_embedder` and
    :func:`~ef.segmenter_adapters.as_segmenter`. Accepts, in order:

    1. ``None`` — a fresh empty in-RAM ``dict``;
    2. a ``Mapping`` — returned **unchanged** (a ``dict``, a ``dol`` store,
       another corpus: it already *is* a corpus);
    3. a directory path (``str`` / ``os.PathLike`` naming an existing
       directory) — a ``dol`` filesystem store of the text files under it
       (``dol`` is imported lazily; only this branch needs it);
    4. an iterable of sources (strings or mappings) — an in-RAM ``dict`` keyed
       by each source's :func:`content_hash` (content-addressed, so duplicate
       sources collapse to one entry).

    Extra ``**kwargs`` are forwarded to the ``dol`` filesystem store (the
    directory branch only) and ignored otherwise.

    Raises:
        TypeError: if ``x`` is a non-directory string, or none of the above.

    >>> as_corpus()
    {}
    >>> as_corpus({'a': 'x'})                 # a mapping passes straight through
    {'a': 'x'}
    >>> sorted(as_corpus(['alpha', 'beta']).values())
    ['alpha', 'beta']
    >>> len(as_corpus(['dup', 'dup']))        # content-addressed: duplicates collapse
    1
    """
    if x is None:
        return {}
    if isinstance(x, Mapping):
        return x
    if isinstance(x, (str, os.PathLike)):
        path = os.fspath(x)
        if os.path.isdir(path):
            return _filesystem_corpus(path, **kwargs)
        raise TypeError(
            f"as_corpus got the path {x!r}, which is not an existing "
            f"directory. Pass a directory path, a Mapping, an iterable of "
            f"sources, or None."
        )
    if isinstance(x, Iterable):
        return {content_hash(source): source for source in x}
    raise TypeError(
        f"Cannot interpret {x!r} as a corpus. Pass a Mapping, a directory "
        f"path, an iterable of sources, or None."
    )


# ---------------------------------------------------------------------------
# Change-detection data model
# ---------------------------------------------------------------------------

#: The kinds of change :class:`ChangeDetectingCorpus` reports.
ChangeKind = Literal["added", "modified", "deleted"]


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """A single detected change to a source in a corpus.

    Emitted by :class:`ChangeDetectingCorpus` to its ``on_change`` callback.
    ``old_hash`` is ``None`` for an ``"added"`` event, ``new_hash`` is ``None``
    for a ``"deleted"`` one.

    Attributes:
        source_id: the key of the source that changed.
        kind: ``"added"``, ``"modified"`` or ``"deleted"``.
        old_hash: the source's previous content hash, if any.
        new_hash: the source's new content hash, if any.
    """

    source_id: str
    kind: ChangeKind
    old_hash: str | None = None
    new_hash: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusDiff:
    """The result of comparing a corpus's content against known content hashes.

    The four fields partition the corpus's source ids; each is a sorted tuple,
    so a :class:`CorpusDiff` is deterministic and JSON-friendly. Truthiness
    reports whether *anything* changed.

    >>> d = CorpusDiff(added=('b',), modified=('a',))
    >>> bool(d), d.changed
    (True, ('b', 'a'))
    >>> bool(CorpusDiff(unchanged=('a', 'b')))
    False
    """

    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()

    @property
    def changed(self) -> tuple[str, ...]:
        """The source ids that are new or modified — the set to (re-)materialize."""
        return self.added + self.modified

    def __bool__(self) -> bool:
        """True iff anything was added, modified or deleted."""
        return bool(self.added or self.modified or self.deleted)


# ---------------------------------------------------------------------------
# The change-detecting wrapper
# ---------------------------------------------------------------------------


def _noop(event: ChangeEvent) -> None:
    """Default ``on_change`` callback — ignore the event."""


class ChangeDetectingCorpus(MutableMapping):
    """A corpus wrapper that content-hashes values and reports what changed.

    Wraps any inner corpus (a ``MutableMapping[source_id, Source]``) and adds
    change detection *without* altering its mapping behavior — reads, writes and
    deletes pass straight through to the inner store. It is to a corpus what
    :class:`~ef.embedder_wrappers.CachedEmbedder` is to an embedder: a
    transparent, composable layer.

    Changes are noticed two ways:

    - **Through the wrapper.** A ``corpus[id] = source`` or ``del corpus[id]``
      that actually changes content fires the injected ``on_change`` callback
      with a :class:`ChangeEvent`. Writing *identical* content is a no-op (no
      event) — the basis of idempotent ingestion.
    - **Out of band.** When the backing store is edited directly (a file
      changed on disk, an S3 object replaced), :meth:`scan` re-reads every
      source, compares hashes, fires events for what differs and returns a
      :class:`CorpusDiff`. :meth:`diff` runs the same comparison with no side
      effects.

    The ``on_change`` callback is the seam the downstream ``ArtifactGraph``
    plugs into: a ``"modified"`` / ``"deleted"`` event is exactly the trigger
    for a cascade invalidation.

    On construction a *baseline* is established: the current content hashes of
    the inner corpus are recorded silently (no events). Pass an already-populated
    ``hashes`` registry to instead resume from a persisted baseline — then call
    :meth:`scan` to reconcile any drift.

    Args:
        corpus: the inner corpus to wrap. Coerced via :func:`as_corpus`, so a
            directory path / ``dict`` / iterable of sources / ``None`` all work.
        on_change: called with each :class:`ChangeEvent`. Defaults to a no-op.
            It should be cheap and not raise; an exception propagates to the
            caller *after* the corpus and registry are already consistent.
        hashes: the hash registry — a ``MutableMapping[source_id, str]`` of the
            last-known content hash per source. Injectable so it can be
            persisted (e.g. a ``dol`` store); defaults to an in-RAM ``dict``.
            If passed already-populated it is trusted as the baseline.
        content_keys: forwarded to :func:`content_hash` — metadata keys that
            participate in the hash.
        fingerprint: an optional cheap ``source_id -> Hashable`` callback (an
            mtime / etag prefilter). When given, :meth:`scan` / :meth:`diff`
            skip re-hashing a source whose fingerprint is unchanged. The
            content hash always remains the source of truth — the fingerprint
            is only a speed optimization, never trusted to *declare* a change.

    >>> backing = {'a': 'one', 'b': 'two'}
    >>> corpus = ChangeDetectingCorpus(backing)   # baseline adopts 'a' and 'b'
    >>> backing['a'] = 'ONE'                      # edited behind the wrapper's back
    >>> backing['c'] = 'three'
    >>> del backing['b']
    >>> d = corpus.scan()
    >>> (d.added, d.modified, d.deleted)
    (('c',), ('a',), ('b',))
    """

    def __init__(
        self,
        corpus: Any = None,
        *,
        on_change: Callable[[ChangeEvent], Any] | None = None,
        hashes: MutableMapping[str, str] | None = None,
        content_keys: Iterable[str] | None = None,
        fingerprint: Callable[[str], Hashable] | None = None,
    ) -> None:
        self.corpus = as_corpus(corpus)
        self.on_change: Callable[[ChangeEvent], Any] = on_change or _noop
        self.hashes: MutableMapping[str, str] = {} if hashes is None else hashes
        self.content_keys: tuple[str, ...] | None = (
            tuple(content_keys) if content_keys else None
        )
        self.fingerprint = fingerprint
        self._fingerprints: dict[str, Hashable] = {}
        if not self.hashes:
            # No persisted baseline — adopt the inner corpus's current content
            # as the baseline, silently (no events for pre-existing sources).
            for source_id in self.corpus:
                self.hashes[source_id] = self._hash(self.corpus[source_id])
                self._refresh_fingerprint(source_id)

    # -- content hashing helpers --------------------------------------------

    def _hash(self, source: Source) -> str:
        """Content hash of ``source`` honoring this corpus's ``content_keys``."""
        return content_hash(source, content_keys=self.content_keys)

    def _emit(self, event: ChangeEvent) -> None:
        """Deliver ``event`` to the ``on_change`` callback."""
        self.on_change(event)

    # -- the mtime/etag prefilter -------------------------------------------

    def _refresh_fingerprint(self, source_id: str) -> None:
        """Store the current cheap fingerprint of ``source_id`` (if any)."""
        if self.fingerprint is None:
            return
        try:
            fp = self.fingerprint(source_id)
        except Exception:  # noqa: BLE001 - a flaky prefilter must not break us
            self._fingerprints.pop(source_id, None)
            return
        if fp is None:
            self._fingerprints.pop(source_id, None)
        else:
            self._fingerprints[source_id] = fp

    def _fingerprint_unchanged(self, source_id: str) -> bool:
        """True iff a cheap fingerprint proves ``source_id`` need not be re-hashed.

        Conservative: a missing prefilter, a missing stored fingerprint, a
        ``None`` reading or any exception all return ``False`` — the content
        hash is then recomputed, because it is the only source of truth.
        """
        if self.fingerprint is None:
            return False
        try:
            fp = self.fingerprint(source_id)
        except Exception:  # noqa: BLE001 - a flaky prefilter must not break us
            return False
        if fp is None:
            return False
        return self._fingerprints.get(source_id, _MISSING) == fp

    # -- MutableMapping interface (pass-through + change detection) ----------

    def __getitem__(self, source_id: str) -> Source:
        return self.corpus[source_id]

    def __setitem__(self, source_id: str, source: Source) -> None:
        new_hash = self._hash(source)
        old_hash = self.hashes.get(source_id)
        self.corpus[source_id] = source  # write through first
        if old_hash is not None and old_hash == new_hash:
            self._refresh_fingerprint(source_id)  # content same — no event
            return
        self.hashes[source_id] = new_hash
        self._refresh_fingerprint(source_id)
        if old_hash is None:
            self._emit(ChangeEvent(source_id, "added", None, new_hash))
        else:
            self._emit(ChangeEvent(source_id, "modified", old_hash, new_hash))

    def __delitem__(self, source_id: str) -> None:
        del self.corpus[source_id]  # raises KeyError if absent
        old_hash = self.hashes.pop(source_id, None)
        self._fingerprints.pop(source_id, None)
        self._emit(ChangeEvent(source_id, "deleted", old_hash, None))

    def __iter__(self) -> Iterator[str]:
        return iter(self.corpus)

    def __len__(self) -> int:
        return len(self.corpus)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self.corpus

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} over {type(self.corpus).__name__} "
            f"tracking {len(self.hashes)} source(s)>"
        )

    # -- out-of-band change detection ---------------------------------------

    def _compute_diff(self) -> tuple[CorpusDiff, dict[str, str]]:
        """Compare current corpus content against the registry — no side effects.

        Returns the :class:`CorpusDiff` and the map of freshly-computed content
        hashes (so :meth:`scan` can update the registry without re-hashing).
        """
        added: list[str] = []
        modified: list[str] = []
        unchanged: list[str] = []
        new_hashes: dict[str, str] = {}
        current_ids = set(self.corpus)
        for source_id in current_ids:
            old_hash = self.hashes.get(source_id)
            if old_hash is not None and self._fingerprint_unchanged(source_id):
                new_hashes[source_id] = old_hash  # prefilter: skip re-hashing
                unchanged.append(source_id)
                continue
            new_hash = self._hash(self.corpus[source_id])
            new_hashes[source_id] = new_hash
            if old_hash is None:
                added.append(source_id)
            elif new_hash != old_hash:
                modified.append(source_id)
            else:
                unchanged.append(source_id)
        deleted = sorted(set(self.hashes) - current_ids)
        diff = CorpusDiff(
            added=tuple(sorted(added)),
            modified=tuple(sorted(modified)),
            deleted=tuple(deleted),
            unchanged=tuple(sorted(unchanged)),
        )
        return diff, new_hashes

    def diff(self) -> CorpusDiff:
        """Compare current corpus content against the registry — **no side effects**.

        Detects out-of-band edits (changes made to the backing store directly)
        without firing events or updating the registry. :meth:`scan` is the
        effectful counterpart.
        """
        return self._compute_diff()[0]

    def scan(self) -> CorpusDiff:
        """Reconcile the registry with the corpus, fire events, return the diff.

        Use this to pick up **out-of-band** edits — changes made to the backing
        store directly rather than through this wrapper. Every detected change
        fires a :class:`ChangeEvent` and the registry (and fingerprints) are
        advanced to the new state, so a second :meth:`scan` with nothing
        further changed is a no-op. :meth:`diff` is the side-effect-free version.
        """
        diff, new_hashes = self._compute_diff()
        for source_id in diff.added:
            self.hashes[source_id] = new_hashes[source_id]
            self._refresh_fingerprint(source_id)
            self._emit(ChangeEvent(source_id, "added", None, new_hashes[source_id]))
        for source_id in diff.modified:
            old_hash = self.hashes.get(source_id)
            self.hashes[source_id] = new_hashes[source_id]
            self._refresh_fingerprint(source_id)
            self._emit(
                ChangeEvent(source_id, "modified", old_hash, new_hashes[source_id])
            )
        for source_id in diff.deleted:
            old_hash = self.hashes.pop(source_id, None)
            self._fingerprints.pop(source_id, None)
            self._emit(ChangeEvent(source_id, "deleted", old_hash, None))
        return diff
