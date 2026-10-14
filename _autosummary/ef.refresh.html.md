# ef.refresh

Refresh — keeping an indexed corpus in sync as its sources change.

Phase 5 indexes a corpus *once*. This module makes the index **maintainable**:
when sources are added, edited or deleted, refresh brings the `vd` index back
into agreement with the corpus. It is the acting counterpart of the read-only
[`ef.diagnostics`](ef.diagnostics.html.md#module-ef.diagnostics) — diagnostics says *what drifted*, refresh *fixes it*.

The one idea, restated from [`ef.artifact_graph`](ef.artifact_graph.html.md#module-ef.artifact_graph):

> Refresh = diff source hashes → delete the stale/orphan documents →
> re-`materialize` what changed.

The `vd` collection is the single source of truth for *what is indexed*; the
[`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) is only a *cache*. So refresh has two
kinds of effect, and they are not equally critical:

- **Collection edits are correctness.** Stale and orphaned documents must be
  removed precisely (by `source_id`), or a search returns wrong hits.
- **Graph pruning is housekeeping.** A content-addressed artifact keyed by a
  now-dead content hash is never *served* again (its source’s content — hence
  its leaf id — has moved on); it is merely wasted cache. So [`prune_dead_leaves()`](#ef.refresh.prune_dead_leaves)
  is best-effort and only ever cascades leaves verified dead — no surviving
  source still hashes to them.

The four refresh modes (`ef_design_notes.md` §8.5, mirroring LangChain’s
indexing API) differ only in **how much they delete**:

[`plan_refresh()`](#ef.refresh.plan_refresh) turns a [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport) + a mode
into a [`RefreshPlan`](#ef.refresh.RefreshPlan) (which sources to (re-)materialize, which documents
to delete) — a pure function, the testable heart of the policy. The orchestration
that applies a plan lives on [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager)
([`refresh()`](ef.source_manager.html.md#ef.source_manager.SourceManager.refresh)).

**Explicit vs. auto.** [`SourceManager.refresh`](ef.source_manager.html.md#ef.source_manager.SourceManager.refresh) is the *explicit* path — it re-reads
and re-hashes the corpus, so it catches out-of-band edits on its own.
[`refresh_on_change()`](#ef.refresh.refresh_on_change) is the *auto* path: a handler you hand to a
[`ChangeDetectingCorpus`](ef.corpus.html.md#ef.corpus.ChangeDetectingCorpus)’s `on_change` so each edit is
incrementally re-indexed as it happens (`SourceManager(auto_refresh=True)`
wires this for you).

### Module Attributes

| [`RefreshMode`](#ef.refresh.RefreshMode)   | The four refresh modes — see the module docstring and `ef_design_notes.md` §8.5.                  |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [`REFRESH_MODES`](#ef.refresh.REFRESH_MODES) | The accepted [`RefreshMode`](#ef.refresh.RefreshMode) values, for validation. |

### Functions

| [`plan_refresh`](#ef.refresh.plan_refresh)(report, \*[, mode, scope])           | Turn a [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport) into a [`RefreshPlan`](#ef.refresh.RefreshPlan).   |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`delete_source_documents`](#ef.refresh.delete_source_documents)(collection, indexed, ...) | Delete every `vd` document belonging to the given `source_ids`.                                                                                                               |
| [`prune_dead_leaves`](#ef.refresh.prune_dead_leaves)(graph, live_leaves)             | Cascade out of `graph` every leaf no live source still hashes to.                                                                                                             |
| [`refresh_on_change`](#ef.refresh.refresh_on_change)(manager)                        | Build a [`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent) handler that auto-refreshes `manager`.                                             |

### Classes

| [`RefreshPlan`](#ef.refresh.RefreshPlan)([to_materialize, to_delete])   | What a refresh will do — [`plan_refresh()`](#ef.refresh.plan_refresh)'s output.   |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`RefreshReport`](#ef.refresh.RefreshReport)(config, mode[, added, ...])  | The outcome of a refresh — what `SourceManager.refresh()` returns.                                   |

### ef.refresh.REFRESH_MODES *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['none', 'incremental', 'full', 'scoped_full'], ...]* *= ('none', 'incremental', 'full', 'scoped_full')*

The accepted [`RefreshMode`](#ef.refresh.RefreshMode) values, for validation.

### ef.refresh.RefreshMode

The four refresh modes — see the module docstring and `ef_design_notes.md`
§8.5. They differ only in how much they delete.

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘none’, ‘incremental’, ‘full’, ‘scoped_full’]

### *class* ef.refresh.RefreshPlan(to_materialize=(), to_delete=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What a refresh will do — [`plan_refresh()`](#ef.refresh.plan_refresh)’s output.

The two fields are disjoint in *intent* but may overlap in membership: a
stale source is in **both** — its old documents are deleted *and* it is
re-materialized. An orphan is only in `to_delete`; a brand-new source only
in `to_materialize`.

#### to_materialize

source ids to (re-)run through the pipeline.

#### to_delete

source ids whose currently-indexed documents are removed
(before re-materializing, for the ones that also reappear above).

### *class* ef.refresh.RefreshReport(config, mode, added=(), modified=(), deleted=(), unchanged=(), documents_written=0, documents_removed=0, artifacts_removed=0)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of a refresh — what `SourceManager.refresh()` returns.

`added` / `modified` / `deleted` / `unchanged` partition the
source ids the refresh considered (mirroring [`CorpusDiff`](ef.corpus.html.md#ef.corpus.CorpusDiff)
field names): newly indexed, re-indexed (was stale/misconfigured), removed
(orphan), and already-fresh-left-alone. The three counts report the work
done. Truthiness reports whether anything changed.

```pycon
>>> r = RefreshReport(config='c0', mode='full', added=('a',), modified=('b',))
>>> bool(r), r.changed
(True, ('a', 'b'))
>>> bool(RefreshReport(config='c0', mode='full', unchanged=('a',)))
False
```

#### *property* changed *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

The source ids that were added, modified or deleted.

### ef.refresh.delete_source_documents(collection, indexed, source_ids)

Delete every `vd` document belonging to the given `source_ids`.

The correctness half of a refresh: a stale or orphaned source’s documents
must leave the collection. `indexed` — an [`indexed_state()`](ef.diagnostics.html.md#ef.diagnostics.indexed_state)
grouping — supplies each source’s document ids, so no per-source scan is
needed. A document already gone is silently tolerated (idempotent).

* **Parameters:**
  * **collection** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the `vd` collection (a `MutableMapping`) to delete from.
  * **indexed** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`IndexedSource`](ef.diagnostics.html.md#ef.diagnostics.IndexedSource)]) – the `{source_id: IndexedSource}` map of what is indexed.
  * **source_ids** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the sources whose documents to remove.
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  the number of documents actually deleted.

```pycon
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
```

### ef.refresh.plan_refresh(report, , mode='full', scope=None)

Turn a [`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport) into a [`RefreshPlan`](#ef.refresh.RefreshPlan).

The pure, side-effect-free core of the refresh policy. Every source that is
`missing`, `stale` or `misconfigured` is (re-)materialized; what gets
*deleted* is what the `mode` dictates:

- `none` — delete nothing (stale content is added alongside the old);
- `incremental` — delete the stale/misconfigured sources’ old documents;
- `full` / `scoped_full` — also delete orphans.

* **Parameters:**
  * **report** ([`StalenessReport`](ef.diagnostics.html.md#ef.diagnostics.StalenessReport)) – the staleness report to act on.
  * **mode** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'none'`, `'incremental'`, `'full'`, `'scoped_full'`]) – one of [`REFRESH_MODES`](#ef.refresh.REFRESH_MODES).
  * **scope** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional subset of source ids the plan is restricted to —
    sources outside it are left entirely alone. `None` means the whole
    report. (With `scope=None`, `full` and `scoped_full` coincide.)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `mode` is not a recognized [`RefreshMode`](#ef.refresh.RefreshMode).
* **Return type:**
  [`RefreshPlan`](#ef.refresh.RefreshPlan)

```pycon
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
```

### ef.refresh.prune_dead_leaves(graph, live_leaves)

Cascade out of `graph` every leaf no live source still hashes to.

The housekeeping half of a refresh. A graph *leaf* (a node with a cached
value but no producer recipe) is an L0 source keyed by its content hash.
After a refresh, the set of leaves the graph *should* hold is exactly the
content hashes of the current corpus; any other leaf is dead — the source
that produced it was edited (new content → new leaf) or deleted. Each dead
leaf and its whole downstream cone (its segment + embed artifacts) is
[`delete_cascade()`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph.delete_cascade)d.

This is purely a cache reclaim: a dead leaf is never *served* (nothing
addresses it any more), so pruning it can never change a result — only free
memory. Dead leaves’ cones are disjoint (each produced artifact traces to
exactly one leaf), so the removed-count is exact.

* **Parameters:**
  * **graph** ([`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph)) – the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) to prune.
  * **live_leaves** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the content hashes that are still live (the current
    corpus’s source hashes).
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
* **Returns:**
  the number of artifacts (leaves + produced) removed.

```pycon
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
```

### ef.refresh.refresh_on_change(manager)

Build a [`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent) handler that auto-refreshes `manager`.

The *auto* refresh seam. The returned callable is meant to be a
[`ChangeDetectingCorpus`](ef.corpus.html.md#ef.corpus.ChangeDetectingCorpus)’s `on_change` — each detected
edit (through the wrapper, or surfaced out-of-band by `scan()`) is
incrementally applied to every materialized config of `manager`:
an `added` source is indexed, a `modified` one is re-indexed
(delete-then-add), a `deleted` one is removed.

[`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) wires this automatically when
constructed with `auto_refresh=True`; call this directly only to attach a
manager to a corpus you build yourself.

* **Parameters:**
  **manager** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) (duck-typed — any
  object with an `_apply_change(event)` method).
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]
* **Returns:**
  a `Callable[[ChangeEvent], None]` for `on_change`.
