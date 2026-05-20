"""The ``ArtifactGraph`` — ``ef``'s content-addressed corpus-indexing core.

The corpus-indexing engine is **not a pipeline** — a pipeline is the wrong
abstraction. It is a *declared, content-addressed producer graph* (a DAG). The
one idea this module is built on::

    Cascade invalidation and configuration branching are the same operation.

A source edit changes a leaf's content hash; a parameter change changes a leaf's
content hash. Both produce a new ``artifact_id`` whose downstream cone needs
(re-)materialization — and the graph does not distinguish them. "Incremental
refresh", "experiment tracking", "hot reload", "cache invalidation" and
"reactivity" are five names for that one operation.

Every artifact is addressed by **what produced it**::

    artifact_id = H(op, op_version, inputs, params)

so two configs that share an upstream step (same ``op``, same ``params``, same
``inputs``) share the *same* artifact id — each segment is computed once, each
vector once, no matter how many configs branch through the graph. Bumping an
``op_version`` (or any ``param``, or any input) yields a *new* id; the old
artifact keeps its old id and its old cached value, still valid for the config
that produced it. Config branching is therefore free.

The module's surface:

- :class:`ProducerSpec` — the declarative recipe for one produced artifact:
  *which op*, *which version*, *over which inputs*, *with which params*. It is
  fully serializable (string ``op`` key, not a raw callable — see below), so an
  :class:`ArtifactGraph`'s stores can be persisted.
- :func:`artifact_id` — the content-addressed id of a :class:`ProducerSpec`.
- :func:`producer_spec` — an ergonomic :class:`ProducerSpec` constructor
  (positional inputs, keyword params).
- :class:`ArtifactGraph` — the graph itself: four injectable
  ``MutableMapping``\\ s (a content-addressed cache ``store``, the ``producers``
  registry, a reverse-``edges`` index, an ``ops`` registry) and the four
  contract operations :meth:`~ArtifactGraph.materialize` (lazy backward),
  :meth:`~ArtifactGraph.mark_stale` / :meth:`~ArtifactGraph.delete_cascade`
  (forward) and :meth:`~ArtifactGraph.freshness`.

**Why ``op`` is a string, not a callable.** The design wants the dependency
graph to scale to ~10⁷ nodes, which means the ``producers`` registry must be
*persistable* (a ``dol`` store — SQLite-backed for the heavy case). A raw
``Callable`` cannot be serialized; an op *name* can. So :class:`ProducerSpec`
records a string op-key, and the :class:`ArtifactGraph` resolves it to the
actual function through an injected ``ops`` registry at materialize time. Every
store (``store`` / ``producers`` / ``edges``) is a plain ``MutableMapping``;
``ef`` itself takes no SQLite dependency — you swap the backend in.

This module is the *engine*. Wiring it to a corpus — turning a Phase-3
:class:`~ef.corpus.ChangeEvent` into a :meth:`~ArtifactGraph.delete_cascade`,
exposing the sources-and-configs facade — is the job of later phases
(``source_manager.py`` / ``refresh.py``); :meth:`~ArtifactGraph.delete_cascade`
is exactly the cascade a ``"deleted"`` / ``"modified"`` event will trigger.

Example — a two-step graph, a source "change", a recompute:

>>> graph = ArtifactGraph()
>>> _ = graph.register_op('upper', str.upper)
>>> _ = graph.register_op('repeat', lambda s, *, times: s * times)
>>> src = graph.put('src', 'ab')                       # a leaf source
>>> shout = graph.add(producer_spec('upper', src, op_version='1'))
>>> echo = graph.add(producer_spec('repeat', shout, op_version='1', times=3))
>>> graph.materialize(echo)                            # lazy backward compute
'ABABAB'
>>> graph.freshness(echo)
'materialized'
>>> _ = graph.mark_stale(src)                          # the source 'changed'
>>> graph.freshness(echo)                              # downstream invalidated
'stale'
>>> graph.materialize(echo)                            # rebuilt from the recipe
'ABABAB'
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ef.hashing import canonical_json, sha256_hex

__all__ = [
    "ArtifactId",
    "OpKey",
    "Freshness",
    "ProducerSpec",
    "artifact_id",
    "producer_spec",
    "ArtifactGraph",
]


# ---------------------------------------------------------------------------
# The contract — type aliases
# ---------------------------------------------------------------------------

#: A content-addressed artifact identifier — a ``sha256`` hex digest for a
#: produced artifact (:func:`artifact_id`), or a caller-supplied content hash
#: for a leaf (e.g. :func:`ef.corpus.content_hash` of an L0 source).
ArtifactId = str

#: The key an :class:`ArtifactGraph` resolves to a producer callable through its
#: ``ops`` registry. A short, stable operation name, e.g. ``"segment"`` or
#: ``"embed"``.
OpKey = str

#: The state of an artifact, reported by :meth:`ArtifactGraph.freshness`:
#: ``"materialized"`` (a value is cached), ``"stale"`` (a recipe exists but no
#: cached value — :meth:`~ArtifactGraph.materialize` will (re)build it) or
#: ``"unknown"`` (the graph has neither a value nor a recipe for the id).
Freshness = Literal["materialized", "stale", "unknown"]


# ---------------------------------------------------------------------------
# The producer recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProducerSpec:
    """A declarative recipe for one produced artifact — the graph's node type.

    A :class:`ProducerSpec` says: *this artifact is produced by op* ``op`` *at
    version* ``op_version``, *applied to the artifacts* ``inputs`` *(passed
    positionally), with the keyword parameters* ``params``. It does **not**
    hold the producer callable — only the string ``op`` key — so a spec is
    fully serializable (:meth:`as_dict` / :meth:`from_dict`) and the graph's
    ``producers`` registry can be persisted.

    The recipe is the artifact's *identity*: :func:`artifact_id` hashes exactly
    these four fields. Treat an instance as immutable — it is frozen, and
    mutating ``params`` in place would silently desynchronize it from its id.

    Attributes:
        op: the op key, resolved to a callable via the graph's ``ops`` registry.
        op_version: the op's version string. It participates in the id, so a
            behavior change to an op **must** be paired with a version bump —
            reusing a version for changed behavior silently serves stale cached
            values (the graph trusts the version; it cannot inspect the code).
        inputs: the input artifact ids, in the order the op receives them
            positionally. Order is significant — it is part of the id.
        params: the op's keyword arguments. Key order is *not* significant
            (the id is computed over canonical, sorted-key JSON); pass
            defaults-filled, fully-named params so that two semantically equal
            calls hash equal.

    >>> spec = producer_spec('embed', 'seg-1', op_version='3', model='small')
    >>> spec.op, spec.inputs, spec.op_version
    ('embed', ('seg-1',), '3')
    >>> spec.params == {'model': 'small'}
    True
    """

    op: OpKey
    op_version: str
    inputs: tuple[ArtifactId, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-``dict`` form of the spec — JSON-serializable.

        The inverse of :meth:`from_dict`. Use it to persist a ``producers``
        registry to a ``dol`` JSON store.

        >>> producer_spec('embed', 'a', 'b', op_version='1', dim=256).as_dict()
        {'op': 'embed', 'op_version': '1', 'inputs': ['a', 'b'], 'params': {'dim': 256}}
        """
        return {
            "op": self.op,
            "op_version": self.op_version,
            "inputs": list(self.inputs),
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProducerSpec":
        """Rebuild a :class:`ProducerSpec` from its :meth:`as_dict` form.

        >>> spec = producer_spec('embed', 'a', 'b', op_version='1', dim=256)
        >>> ProducerSpec.from_dict(spec.as_dict()) == spec
        True
        """
        return cls(
            op=d["op"],
            op_version=d["op_version"],
            inputs=tuple(d.get("inputs", ())),
            params=dict(d.get("params", {})),
        )


def producer_spec(
    op: OpKey, *inputs: ArtifactId, op_version: str, **params: Any
) -> ProducerSpec:
    """Build a :class:`ProducerSpec` ergonomically — inputs positional, params keyword.

    The natural way to write a recipe: the input artifact ids are positional
    (their order is significant — it is the order the op receives them), and
    every keyword argument is an op parameter. ``op_version`` is keyword-only
    and required — an op without a stated version cannot be content-addressed
    safely.

    >>> spec = producer_spec('segment', 'doc-1', op_version='1', size=512, overlap=64)
    >>> spec.op, spec.inputs
    ('segment', ('doc-1',))
    >>> spec.params == {'size': 512, 'overlap': 64}
    True
    """
    return ProducerSpec(
        op=op, op_version=op_version, inputs=tuple(inputs), params=dict(params)
    )


def artifact_id(spec: ProducerSpec) -> ArtifactId:
    """Content-addressed id of ``spec`` — ``sha256`` over its four fields.

    The id is ``sha256`` of canonical (sorted-key) JSON of
    ``{op, op_version, inputs, params}``. Two specs that describe the same
    computation hash *identically*; any change to the op, its version, its
    inputs (including their order) or its params yields a *different* id. That
    is what makes the graph content-addressed: the id *is* the recipe.

    >>> a = producer_spec('embed', 'seg-1', op_version='2', model='small')
    >>> artifact_id(a) == artifact_id(producer_spec(
    ...     'embed', 'seg-1', op_version='2', model='small'))
    True
    >>> len(artifact_id(a))
    64
    >>> artifact_id(a) == artifact_id(producer_spec(
    ...     'embed', 'seg-1', op_version='3', model='small'))   # version bump
    False
    >>> artifact_id(producer_spec('op', 'x', 'y', op_version='1')) == artifact_id(
    ...     producer_spec('op', 'y', 'x', op_version='1'))      # input order matters
    False
    """
    return sha256_hex(
        canonical_json(
            {
                "op": spec.op,
                "op_version": spec.op_version,
                "inputs": list(spec.inputs),
                "params": dict(spec.params),
            }
        )
    )


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


class ArtifactGraph:
    """A content-addressed producer graph — the corpus-indexing & refresh core.

    The graph holds two kinds of node:

    - **Leaves** — externally content-addressed inputs (an L0 source, hashed by
      :func:`ef.corpus.content_hash`). A leaf has a *value* but no recipe; it is
      supplied via :meth:`put`.
    - **Produced artifacts** — a node with a :class:`ProducerSpec` recipe,
      registered via :meth:`add`. Its id *is* :func:`artifact_id` of its recipe.

    and four small ``MutableMapping`` stores, every one injectable so the graph
    can be backed by RAM (the default) or by a persistent ``dol`` store:

    - ``store`` — the content-addressed cache: ``artifact_id -> value``.
    - ``producers`` — the recipe registry: ``artifact_id -> ProducerSpec``.
    - ``edges`` — the reverse index: ``artifact_id -> (dependent ids, ...)``,
      so the *downstream* cone of any node is one lookup away.
    - ``ops`` — the op registry: ``op_key -> callable``. The one store that is
      *not* persisted (callables are not serializable) — repopulate it with
      :meth:`register_op` each process.

    The four contract operations:

    - :meth:`materialize` — lazy *backward*: compute a node by recursively
      materializing its inputs, then memoize.
    - :meth:`mark_stale` — *forward*: drop the cached values of a node's
      produced downstream cone, keeping the recipes (they will rebuild).
    - :meth:`delete_cascade` — *forward*: remove a node and its whole downstream
      cone entirely (values, recipes and edges).
    - :meth:`freshness` — query a node's state (:data:`Freshness`).

    plus the reachability queries :meth:`descendants` (the downstream cone) and
    :meth:`ancestors` (lineage — what an artifact was produced from).

    Args:
        store: the content-addressed value cache. Defaults to a fresh ``dict``.
        producers: the recipe registry. Defaults to a fresh ``dict``. If passed
            already-populated (a resumed/persisted graph) and ``edges`` is left
            empty, the reverse index is rebuilt from it on construction.
        edges: the reverse-dependency index. Defaults to a fresh ``dict``; if
            passed already-populated it is trusted as-is (not rebuilt).
        ops: the op registry. Defaults to a fresh ``dict``; :meth:`register_op`
            adds to it.

    >>> graph = ArtifactGraph()
    >>> _ = graph.register_op('join', lambda a, b, *, sep: f'{a}{sep}{b}')
    >>> x = graph.put('x', 'hello')
    >>> y = graph.put('y', 'world')
    >>> both = graph.add(producer_spec('join', x, y, op_version='1', sep=', '))
    >>> graph.materialize(both)
    'hello, world'
    >>> sorted(graph.descendants('x'))
    [...]
    >>> graph.delete_cascade('y')                     # the 'y' source is removed
    frozenset(...)
    >>> graph.freshness(both)                         # its consumer is gone too
    'unknown'
    """

    def __init__(
        self,
        *,
        store: MutableMapping[ArtifactId, Any] | None = None,
        producers: MutableMapping[ArtifactId, ProducerSpec] | None = None,
        edges: MutableMapping[ArtifactId, tuple[ArtifactId, ...]] | None = None,
        ops: MutableMapping[OpKey, Callable[..., Any]] | None = None,
    ) -> None:
        self.store: MutableMapping[ArtifactId, Any] = {} if store is None else store
        self.producers: MutableMapping[ArtifactId, ProducerSpec] = (
            {} if producers is None else producers
        )
        self.edges: MutableMapping[ArtifactId, tuple[ArtifactId, ...]] = (
            {} if edges is None else edges
        )
        self.ops: MutableMapping[OpKey, Callable[..., Any]] = {} if ops is None else ops
        if self.producers and not self.edges:
            # A resumed graph with persisted recipes but no persisted reverse
            # index — derive the edges from the recipes (mirrors how
            # ChangeDetectingCorpus adopts a baseline on construction).
            self._rebuild_edges()

    # -- construction helpers -----------------------------------------------

    def _rebuild_edges(self) -> None:
        """Derive the whole reverse-``edges`` index from the ``producers`` recipes."""
        for aid, spec in self.producers.items():
            for inp in spec.inputs:
                self._link(inp, aid)

    def _link(self, input_id: ArtifactId, dependent_id: ArtifactId) -> None:
        """Record that ``dependent_id`` consumes ``input_id`` (a reverse edge)."""
        existing = self.edges.get(input_id, ())
        if dependent_id not in existing:
            self.edges[input_id] = tuple(sorted((*existing, dependent_id)))

    def _unlink(self, input_id: ArtifactId, dependent_id: ArtifactId) -> None:
        """Drop the reverse edge ``input_id -> dependent_id`` (if present)."""
        remaining = tuple(d for d in self.edges.get(input_id, ()) if d != dependent_id)
        if remaining:
            self.edges[input_id] = remaining
        else:
            self.edges.pop(input_id, None)

    # -- registration -------------------------------------------------------

    def register_op(self, name: OpKey, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register the callable ``fn`` under the op key ``name``; return ``fn``.

        :meth:`materialize` resolves a :class:`ProducerSpec`'s string ``op``
        through this registry. The op is called ``fn(*input_values, **params)``:
        the materialized inputs positionally, the spec's ``params`` as keywords.

        >>> graph = ArtifactGraph()
        >>> graph.register_op('upper', str.upper)
        <method 'upper' of 'str' objects>
        """
        self.ops[name] = fn
        return fn

    def put(self, key: ArtifactId, value: Any) -> ArtifactId:
        """Store a **leaf** ``value`` (an external input) under ``key``; return ``key``.

        A leaf has a value but no recipe — typically an L0 source keyed by its
        :func:`ef.corpus.content_hash`. Returns ``key`` so the call composes::

            src = graph.put(content_hash(text), text)

        >>> graph = ArtifactGraph()
        >>> graph.put('doc-1', 'some text')
        'doc-1'
        >>> graph.freshness('doc-1')
        'materialized'
        """
        self.store[key] = value
        return key

    def add(self, spec: ProducerSpec) -> ArtifactId:
        """Register a producer recipe; return its content-addressed id.

        Idempotent: the id *is* the recipe, so adding an equal :class:`ProducerSpec`
        twice (or adding the shared upstream step of two branching configs)
        registers one node and returns the same id — the basis of free config
        branching. Inputs need not exist yet; the graph is declarative and
        :meth:`materialize` resolves them lazily.

        >>> graph = ArtifactGraph()
        >>> a = graph.add(producer_spec('op', 'leaf', op_version='1'))
        >>> a == graph.add(producer_spec('op', 'leaf', op_version='1'))  # idempotent
        True
        >>> len(graph.producers)
        1
        """
        aid = artifact_id(spec)
        if aid not in self.producers:
            self.producers[aid] = spec
            for inp in spec.inputs:
                self._link(inp, aid)
        return aid

    # -- the four contract operations ---------------------------------------

    def materialize(self, key: ArtifactId) -> Any:
        """Compute (or fetch the cached value of) the artifact ``key`` — lazy *backward*.

        If ``key`` is already cached in ``store`` its value is returned
        directly. Otherwise its recipe's inputs are materialized recursively
        (depth-first), the op is called ``op(*input_values, **params)``, and the
        result is memoized in ``store`` before being returned — so a second
        :meth:`materialize` is a cache hit, and a shared upstream artifact is
        computed exactly once however many configs depend on it.

        Raises:
            KeyError: if ``key`` is neither cached nor has a producer recipe
                (an unresolved leaf — supply it with :meth:`put`).
            LookupError: if a recipe's ``op`` is not in the ``ops`` registry.
            ValueError: if the producer graph contains a cycle (a content-
                addressed graph cannot — this guards a hand-corrupted registry).

        >>> graph = ArtifactGraph()
        >>> _ = graph.register_op('double', lambda x: x * 2)
        >>> graph.put('n', 21)
        'n'
        >>> out = graph.add(producer_spec('double', 'n', op_version='1'))
        >>> graph.materialize(out)
        42
        """
        return self._materialize(key, set())

    def _materialize(self, key: ArtifactId, _in_progress: set[ArtifactId]) -> Any:
        """Recursive worker for :meth:`materialize`; ``_in_progress`` guards cycles."""
        if key in self.store:
            return self.store[key]
        if key in _in_progress:
            raise ValueError(
                f"Cycle detected in the artifact graph while materializing "
                f"{key!r} — a content-addressed graph cannot cycle, so the "
                f"`producers` registry has been corrupted."
            )
        spec = self.producers.get(key)
        if spec is None:
            raise KeyError(
                f"Cannot materialize artifact {key!r}: it has no cached value "
                f"and no producer recipe. A leaf artifact must be supplied via "
                f"put(); a produced one via add()."
            )
        fn = self.ops.get(spec.op)
        if fn is None:
            raise LookupError(
                f"No op registered under {spec.op!r} (needed to materialize "
                f"{key!r}). Register it with register_op({spec.op!r}, ...)."
            )
        _in_progress.add(key)
        try:
            input_values = [self._materialize(inp, _in_progress) for inp in spec.inputs]
        finally:
            _in_progress.discard(key)
        value = fn(*input_values, **dict(spec.params))
        self.store[key] = value
        return value

    def mark_stale(self, key: ArtifactId) -> frozenset[ArtifactId]:
        """Invalidate ``key``'s produced downstream cone — *forward*; return that cone.

        Drops the **cached values** of every produced artifact in
        ``{key} ∪ descendants(key)``, *keeping the recipes* — so the next
        :meth:`materialize` recomputes them. Leaf nodes (inputs, no recipe) are
        never touched: they cannot be recomputed, so dropping their value would
        be data loss — use :meth:`delete_cascade` to remove an input.

        This is the invalidation half of a refresh: when an op's implementation
        changes, or to force a recompute. (A *source* change is usually a
        :meth:`delete_cascade` of the old content hash — the new content yields
        a new leaf id.) Returns the produced artifacts that are now stale.

        >>> graph = ArtifactGraph()
        >>> _ = graph.register_op('id', lambda x: x)
        >>> graph.put('s', 'v')
        's'
        >>> a = graph.add(producer_spec('id', 's', op_version='1'))
        >>> _ = graph.materialize(a)
        >>> graph.mark_stale('s') == {a}        # the leaf 's' itself is not "stale"
        True
        >>> graph.freshness(a)
        'stale'
        """
        cone = {key} | self.descendants(key)
        produced = frozenset(n for n in cone if n in self.producers)
        for node in produced:
            self.store.pop(node, None)
        return produced

    def delete_cascade(self, key: ArtifactId) -> frozenset[ArtifactId]:
        """Remove ``key`` and its whole downstream cone — *forward*; return what was removed.

        Deletes every artifact in ``{key} ∪ descendants(key)`` *entirely* —
        cached value, recipe and edges — and detaches the cone from any upstream
        inputs that survive it. Unlike :meth:`mark_stale` this removes the
        recipes too: use it when an artifact can never be rebuilt, the canonical
        case being a **deleted source** (its segments, vectors and index entries
        cascade away with it).

        Returns the ids actually removed (those that had a value or a recipe).

        >>> graph = ArtifactGraph()
        >>> _ = graph.register_op('id', lambda x: x)
        >>> graph.put('s', 'v')
        's'
        >>> a = graph.add(producer_spec('id', 's', op_version='1'))
        >>> graph.delete_cascade('s') == {'s', a}
        True
        >>> 's' in graph, a in graph
        (False, False)
        """
        cone = {key} | self.descendants(key)
        removed = frozenset(n for n in cone if n in self.store or n in self.producers)
        for node in cone:
            spec = self.producers.get(node)
            if spec is not None:
                for inp in spec.inputs:
                    if inp not in cone:  # an upstream input that outlives the cone
                        self._unlink(inp, node)
            self.store.pop(node, None)
            self.producers.pop(node, None)
            self.edges.pop(node, None)
        return removed

    def freshness(self, key: ArtifactId) -> Freshness:
        """Report the state of ``key`` — :data:`Freshness`.

        ``"materialized"`` if a value is cached; ``"stale"`` if a recipe exists
        but no value (a :meth:`materialize` will build it); ``"unknown"`` if the
        graph has neither.

        Because the graph is content-addressed, a *cached* value is always
        correct *for its id* — staleness here means "not yet computed", never
        "computed wrong". The richer staleness conditions (orphan / missing /
        misconfigured, computed against ``vd`` index metadata) belong to a later
        diagnostics phase, not to the graph core.

        >>> graph = ArtifactGraph()
        >>> _ = graph.register_op('id', lambda x: x)
        >>> graph.put('s', 'v')
        's'
        >>> a = graph.add(producer_spec('id', 's', op_version='1'))
        >>> graph.freshness('s'), graph.freshness(a), graph.freshness('?')
        ('materialized', 'stale', 'unknown')
        """
        if key in self.store:
            return "materialized"
        if key in self.producers:
            return "stale"
        return "unknown"

    # -- reachability queries -----------------------------------------------

    def descendants(self, key: ArtifactId) -> frozenset[ArtifactId]:
        """The downstream cone of ``key`` — every artifact transitively produced from it.

        Excludes ``key`` itself. This is the set of artifacts a change to
        ``key`` would invalidate; :meth:`mark_stale` and :meth:`delete_cascade`
        are built on it.

        >>> graph = ArtifactGraph()
        >>> _ = graph.register_op('id', lambda x: x)
        >>> a = graph.add(producer_spec('id', 'src', op_version='1'))
        >>> b = graph.add(producer_spec('id', a, op_version='1'))
        >>> graph.descendants('src') == {a, b}
        True
        """
        seen: set[ArtifactId] = set()
        frontier = list(self.edges.get(key, ()))
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(self.edges.get(node, ()))
        seen.discard(key)
        return frozenset(seen)

    def ancestors(self, key: ArtifactId) -> frozenset[ArtifactId]:
        """The lineage of ``key`` — every artifact it was transitively produced from.

        Excludes ``key`` itself. Answers "what produced this?" — provenance /
        lineage queries. The leaves of the returned set are the original
        sources the artifact ultimately derives from.

        >>> graph = ArtifactGraph()
        >>> _ = graph.register_op('id', lambda x: x)
        >>> a = graph.add(producer_spec('id', 'src', op_version='1'))
        >>> b = graph.add(producer_spec('id', a, op_version='1'))
        >>> graph.ancestors(b) == {'src', a}
        True
        """
        seen: set[ArtifactId] = set()
        spec = self.producers.get(key)
        frontier = list(spec.inputs) if spec is not None else []
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            upstream = self.producers.get(node)
            if upstream is not None:
                frontier.extend(upstream.inputs)
        seen.discard(key)
        return frozenset(seen)

    # -- mapping-ish dunders ------------------------------------------------

    def __contains__(self, key: object) -> bool:
        """True iff ``key`` is a known artifact (has a cached value or a recipe)."""
        return key in self.store or key in self.producers

    def __len__(self) -> int:
        """The number of distinct known artifacts (cached values ∪ recipes)."""
        return len(set(self.store) | set(self.producers))

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}: {len(self)} artifact(s), "
            f"{len(self.store)} materialized, {len(self.ops)} op(s)>"
        )
