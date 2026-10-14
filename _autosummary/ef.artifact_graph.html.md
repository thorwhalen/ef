# ef.artifact_graph

The `ArtifactGraph` — `ef`’s content-addressed corpus-indexing core.

The corpus-indexing engine is **not a pipeline** — a pipeline is the wrong
abstraction. It is a *declared, content-addressed producer graph* (a DAG). The
one idea this module is built on:

```default
Cascade invalidation and configuration branching are the same operation.
```

A source edit changes a leaf’s content hash; a parameter change changes a leaf’s
content hash. Both produce a new `artifact_id` whose downstream cone needs
(re-)materialization — and the graph does not distinguish them. “Incremental
refresh”, “experiment tracking”, “hot reload”, “cache invalidation” and
“reactivity” are five names for that one operation.

Every artifact is addressed by **what produced it**:

```default
artifact_id = H(op, op_version, inputs, params)
```

so two configs that share an upstream step (same `op`, same `params`, same
`inputs`) share the *same* artifact id — each segment is computed once, each
vector once, no matter how many configs branch through the graph. Bumping an
`op_version` (or any `param`, or any input) yields a *new* id; the old
artifact keeps its old id and its old cached value, still valid for the config
that produced it. Config branching is therefore free.

The module’s surface:

- [`ProducerSpec`](#ef.artifact_graph.ProducerSpec) — the declarative recipe for one produced artifact:
  *which op*, *which version*, *over which inputs*, *with which params*. It is
  fully serializable (string `op` key, not a raw callable — see below), so an
  [`ArtifactGraph`](#ef.artifact_graph.ArtifactGraph)’s stores can be persisted.
- [`artifact_id()`](#ef.artifact_graph.artifact_id) — the content-addressed id of a [`ProducerSpec`](#ef.artifact_graph.ProducerSpec).
- [`producer_spec()`](#ef.artifact_graph.producer_spec) — an ergonomic [`ProducerSpec`](#ef.artifact_graph.ProducerSpec) constructor
  (positional inputs, keyword params).
- [`ArtifactGraph`](#ef.artifact_graph.ArtifactGraph) — the graph itself: four injectable
  `MutableMapping`s (a content-addressed cache `store`, the `producers`
  registry, a reverse-`edges` index, an `ops` registry) and the four
  contract operations [`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) (lazy backward),
  [`mark_stale()`](#ef.artifact_graph.ArtifactGraph.mark_stale) / [`delete_cascade()`](#ef.artifact_graph.ArtifactGraph.delete_cascade)
  > (forward) and [`freshness()`](#ef.artifact_graph.ArtifactGraph.freshness).

\*\*Why `op` is a string, not a callable.\*\* The design wants the dependency
graph to scale to ~10⁷ nodes, which means the `producers` registry must be
*persistable* (a `dol` store — SQLite-backed for the heavy case). A raw
`Callable` cannot be serialized; an op *name* can. So [`ProducerSpec`](#ef.artifact_graph.ProducerSpec)
records a string op-key, and the [`ArtifactGraph`](#ef.artifact_graph.ArtifactGraph) resolves it to the
actual function through an injected `ops` registry at materialize time. Every
store (`store` / `producers` / `edges`) is a plain `MutableMapping`;
`ef` itself takes no SQLite dependency — you swap the backend in.

This module is the *engine*. Wiring it to a corpus — turning a Phase-3
[`ChangeEvent`](ef.corpus.html.md#ef.corpus.ChangeEvent) into a [`delete_cascade()`](#ef.artifact_graph.ArtifactGraph.delete_cascade),
exposing the sources-and-configs facade — is the job of later phases
(`source_manager.py` / `refresh.py`); [`delete_cascade()`](#ef.artifact_graph.ArtifactGraph.delete_cascade)
is exactly the cascade a `"deleted"` / `"modified"` event will trigger.

Example — a two-step graph, a source “change”, a recompute:

```pycon
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
```

### Module Attributes

| [`ArtifactId`](#ef.artifact_graph.ArtifactId)   | A content-addressed artifact identifier — a `sha256` hex digest for a produced artifact ([`artifact_id()`](#ef.artifact_graph.artifact_id)), or a caller-supplied content hash for a leaf (e.g. [`ef.corpus.content_hash()`](ef.corpus.html.md#ef.corpus.content_hash) of an L0 source).                                    |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`OpKey`](#ef.artifact_graph.OpKey)        | The key an [`ArtifactGraph`](#ef.artifact_graph.ArtifactGraph) resolves to a producer callable through its `ops` registry.                                                                                                                                                                                                                              |
| [`Freshness`](#ef.artifact_graph.Freshness)    | The state of an artifact, reported by [`ArtifactGraph.freshness()`](#ef.artifact_graph.ArtifactGraph.freshness): `"materialized"` (a value is cached), `"stale"` (a recipe exists but no cached value — [`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) will (re)build it) or `"unknown"` (the graph has neither a value nor a recipe for the id). |

### Functions

| [`artifact_id`](#ef.artifact_graph.artifact_id)(spec)                                   | Content-addressed id of `spec` — `sha256` over its four fields.                                                          |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`producer_spec`](#ef.artifact_graph.producer_spec)(op, \*inputs, op_version, \*\*params) | Build a [`ProducerSpec`](#ef.artifact_graph.ProducerSpec) ergonomically — inputs positional, params keyword. |

### Classes

| [`ArtifactId`](#ef.artifact_graph.ArtifactId)                                        | A content-addressed artifact identifier — a `sha256` hex digest for a produced artifact ([`artifact_id()`](#ef.artifact_graph.artifact_id)), or a caller-supplied content hash for a leaf (e.g. [`ef.corpus.content_hash()`](ef.corpus.html.md#ef.corpus.content_hash) of an L0 source).   |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`OpKey`](#ef.artifact_graph.OpKey)                                             | The key an [`ArtifactGraph`](#ef.artifact_graph.ArtifactGraph) resolves to a producer callable through its `ops` registry.                                                                                                                                                                                             |
| [`ProducerSpec`](#ef.artifact_graph.ProducerSpec)(op, op_version[, inputs, params])    | A declarative recipe for one produced artifact — the graph's node type.                                                                                                                                                                                                                                                           |
| [`ArtifactGraph`](#ef.artifact_graph.ArtifactGraph)(\*[, store, producers, edges, ops]) | A content-addressed producer graph — the corpus-indexing & refresh core.                                                                                                                                                                                                                                                          |

### *class* ef.artifact_graph.ArtifactGraph(, store=None, producers=None, edges=None, ops=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A content-addressed producer graph — the corpus-indexing & refresh core.

The graph holds two kinds of node:

- **Leaves** — externally content-addressed inputs (an L0 source, hashed by
  [`ef.corpus.content_hash()`](ef.corpus.html.md#ef.corpus.content_hash)). A leaf has a *value* but no recipe; it is
  > supplied via [`put()`](#ef.artifact_graph.ArtifactGraph.put).
- **Produced artifacts** — a node with a [`ProducerSpec`](#ef.artifact_graph.ProducerSpec) recipe,
  registered via [`add()`](#ef.artifact_graph.ArtifactGraph.add). Its id *is* [`artifact_id()`](#ef.artifact_graph.artifact_id) of its recipe.

and four small `MutableMapping` stores, every one injectable so the graph
can be backed by RAM (the default) or by a persistent `dol` store:

- `store` — the content-addressed cache: `artifact_id -> value`.
- `producers` — the recipe registry: `artifact_id -> ProducerSpec`.
- `edges` — the reverse index: `artifact_id -> (dependent ids, ...)`,
  so the *downstream* cone of any node is one lookup away.
- `ops` — the op registry: `op_key -> callable`. The one store that is
  *not* persisted (callables are not serializable) — repopulate it with
  [`register_op()`](#ef.artifact_graph.ArtifactGraph.register_op) each process.

The four contract operations:

- [`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) — lazy *backward*: compute a node by recursively
  materializing its inputs, then memoize.
- [`mark_stale()`](#ef.artifact_graph.ArtifactGraph.mark_stale) — *forward*: drop the cached values of a node’s
  produced downstream cone, keeping the recipes (they will rebuild).
- [`delete_cascade()`](#ef.artifact_graph.ArtifactGraph.delete_cascade) — *forward*: remove a node and its whole downstream
  cone entirely (values, recipes and edges).
- [`freshness()`](#ef.artifact_graph.ArtifactGraph.freshness) — query a node’s state ([`Freshness`](#ef.artifact_graph.Freshness)).

plus the reachability queries [`descendants()`](#ef.artifact_graph.ArtifactGraph.descendants) (the downstream cone) and
[`ancestors()`](#ef.artifact_graph.ArtifactGraph.ancestors) (lineage — what an artifact was produced from).

* **Parameters:**
  * **store** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the content-addressed value cache. Defaults to a fresh `dict`.
  * **producers** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`ProducerSpec`](#ef.artifact_graph.ProducerSpec)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the recipe registry. Defaults to a fresh `dict`. If passed
    already-populated (a resumed/persisted graph) and `edges` is left
    empty, the reverse index is rebuilt from it on construction.
  * **edges** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the reverse-dependency index. Defaults to a fresh `dict`; if
    passed already-populated it is trusted as-is (not rebuilt).
  * **ops** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the op registry. Defaults to a fresh `dict`; [`register_op()`](#ef.artifact_graph.ArtifactGraph.register_op)
    adds to it.

```pycon
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
```

#### add(spec)

Register a producer recipe; return its content-addressed id.

Idempotent: the id *is* the recipe, so adding an equal [`ProducerSpec`](#ef.artifact_graph.ProducerSpec)
twice (or adding the shared upstream step of two branching configs)
registers one node and returns the same id — the basis of free config
branching. Inputs need not exist yet; the graph is declarative and
[`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) resolves them lazily.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> graph = ArtifactGraph()
>>> a = graph.add(producer_spec('op', 'leaf', op_version='1'))
>>> a == graph.add(producer_spec('op', 'leaf', op_version='1'))  # idempotent
True
>>> len(graph.producers)
1
```

#### ancestors(key)

The lineage of `key` — every artifact it was transitively produced from.

Excludes `key` itself. Answers “what produced this?” — provenance /
lineage queries. The leaves of the returned set are the original
sources the artifact ultimately derives from.

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> a = graph.add(producer_spec('id', 'src', op_version='1'))
>>> b = graph.add(producer_spec('id', a, op_version='1'))
>>> graph.ancestors(b) == {'src', a}
True
```

#### delete_cascade(key)

Remove `key` and its whole downstream cone — *forward*; return what was removed.

Deletes every artifact in `{key} ∪ descendants(key)` *entirely* —
cached value, recipe and edges — and detaches the cone from any upstream
inputs that survive it. Unlike [`mark_stale()`](#ef.artifact_graph.ArtifactGraph.mark_stale) this removes the
recipes too: use it when an artifact can never be rebuilt, the canonical
case being a **deleted source** (its segments, vectors and index entries
cascade away with it).

Returns the ids actually removed (those that had a value or a recipe).

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> graph.put('s', 'v')
's'
>>> a = graph.add(producer_spec('id', 's', op_version='1'))
>>> graph.delete_cascade('s') == {'s', a}
True
>>> 's' in graph, a in graph
(False, False)
```

#### descendants(key)

The downstream cone of `key` — every artifact transitively produced from it.

Excludes `key` itself. This is the set of artifacts a change to
`key` would invalidate; [`mark_stale()`](#ef.artifact_graph.ArtifactGraph.mark_stale) and [`delete_cascade()`](#ef.artifact_graph.ArtifactGraph.delete_cascade)
are built on it.

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> a = graph.add(producer_spec('id', 'src', op_version='1'))
>>> b = graph.add(producer_spec('id', a, op_version='1'))
>>> graph.descendants('src') == {a, b}
True
```

#### freshness(key)

Report the state of `key` — [`Freshness`](#ef.artifact_graph.Freshness).

`"materialized"` if a value is cached; `"stale"` if a recipe exists
but no value (a [`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) will build it); `"unknown"` if the
graph has neither.

Because the graph is content-addressed, a *cached* value is always
correct *for its id* — staleness here means “not yet computed”, never
“computed wrong”. The richer staleness conditions (orphan / missing /
misconfigured, computed against `vd` index metadata) belong to a later
diagnostics phase, not to the graph core.

* **Return type:**
  [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'materialized'`, `'stale'`, `'unknown'`]

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('id', lambda x: x)
>>> graph.put('s', 'v')
's'
>>> a = graph.add(producer_spec('id', 's', op_version='1'))
>>> graph.freshness('s'), graph.freshness(a), graph.freshness('?')
('materialized', 'stale', 'unknown')
```

#### mark_stale(key)

Invalidate `key`’s produced downstream cone — *forward*; return that cone.

Drops the **cached values** of every produced artifact in
`{key} ∪ descendants(key)`, *keeping the recipes* — so the next
[`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) recomputes them. Leaf nodes (inputs, no recipe) are
never touched: they cannot be recomputed, so dropping their value would
be data loss — use [`delete_cascade()`](#ef.artifact_graph.ArtifactGraph.delete_cascade) to remove an input.

This is the invalidation half of a refresh: when an op’s implementation
changes, or to force a recompute. (A *source* change is usually a
[`delete_cascade()`](#ef.artifact_graph.ArtifactGraph.delete_cascade) of the old content hash — the new content yields
a new leaf id.) Returns the produced artifacts that are now stale.

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
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
```

#### materialize(key)

Compute (or fetch the cached value of) the artifact `key` — lazy *backward*.

If `key` is already cached in `store` its value is returned
directly. Otherwise its recipe’s inputs are materialized recursively
(depth-first), the op is called `op(*input_values, **params)`, and the
result is memoized in `store` before being returned — so a second
[`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) is a cache hit, and a shared upstream artifact is
computed exactly once however many configs depend on it.

* **Raises:**
  * [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) – if `key` is neither cached nor has a producer recipe
        (an unresolved leaf — supply it with [`put()`](#ef.artifact_graph.ArtifactGraph.put)).
  * [**LookupError**](https://docs.python.org/3/builtins/exceptions.html#LookupError) – if a recipe’s `op` is not in the `ops` registry.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if the producer graph contains a cycle (a content-
        addressed graph cannot — this guards a hand-corrupted registry).
* **Return type:**
  [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)

```pycon
>>> graph = ArtifactGraph()
>>> _ = graph.register_op('double', lambda x: x * 2)
>>> graph.put('n', 21)
'n'
>>> out = graph.add(producer_spec('double', 'n', op_version='1'))
>>> graph.materialize(out)
42
```

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### put(key, value)

Store a **leaf** `value` (an external input) under `key`; return `key`.

A leaf has a value but no recipe — typically an L0 source keyed by its
[`ef.corpus.content_hash()`](ef.corpus.html.md#ef.corpus.content_hash). Returns `key` so the call composes:

```default
src = graph.put(content_hash(text), text)
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> graph = ArtifactGraph()
>>> graph.put('doc-1', 'some text')
'doc-1'
>>> graph.freshness('doc-1')
'materialized'
```

#### register_op(name, fn)

Register the callable `fn` under the op key `name`; return `fn`.

[`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) resolves a [`ProducerSpec`](#ef.artifact_graph.ProducerSpec)’s string `op`
through this registry. The op is called `fn(*input_values, **params)`:
the materialized inputs positionally, the spec’s `params` as keywords.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> graph = ArtifactGraph()
>>> graph.register_op('upper', str.upper)
<method 'upper' of 'str' objects>
```

### ef.artifact_graph.ArtifactId

A content-addressed artifact identifier — a `sha256` hex digest for a
produced artifact ([`artifact_id()`](#ef.artifact_graph.artifact_id)), or a caller-supplied content hash
for a leaf (e.g. [`ef.corpus.content_hash()`](ef.corpus.html.md#ef.corpus.content_hash) of an L0 source).

### ef.artifact_graph.Freshness

The state of an artifact, reported by [`ArtifactGraph.freshness()`](#ef.artifact_graph.ArtifactGraph.freshness):
`"materialized"` (a value is cached), `"stale"` (a recipe exists but no
cached value — [`materialize()`](#ef.artifact_graph.ArtifactGraph.materialize) will (re)build it) or
`"unknown"` (the graph has neither a value nor a recipe for the id).

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘materialized’, ‘stale’, ‘unknown’]

### ef.artifact_graph.OpKey

The key an [`ArtifactGraph`](#ef.artifact_graph.ArtifactGraph) resolves to a producer callable through its
`ops` registry. A short, stable operation name, e.g. `"segment"` or
`"embed"`.

### *class* ef.artifact_graph.ProducerSpec(op, op_version, inputs=(), params=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A declarative recipe for one produced artifact — the graph’s node type.

A [`ProducerSpec`](#ef.artifact_graph.ProducerSpec) says: *this artifact is produced by op* `op` \*at
version\* `op_version`, *applied to the artifacts* `inputs`  *(passed
positionally), with the keyword parameters* `params`. It does **not**
hold the producer callable — only the string `op` key — so a spec is
fully serializable ([`as_dict()`](#ef.artifact_graph.ProducerSpec.as_dict) / [`from_dict()`](#ef.artifact_graph.ProducerSpec.from_dict)) and the graph’s
`producers` registry can be persisted.

The recipe is the artifact’s *identity*: [`artifact_id()`](#ef.artifact_graph.artifact_id) hashes exactly
these four fields. Treat an instance as immutable — it is frozen, and
mutating `params` in place would silently desynchronize it from its id.

#### op

the op key, resolved to a callable via the graph’s `ops` registry.

#### op_version

the op’s version string. It participates in the id, so a
behavior change to an op **must** be paired with a version bump —
reusing a version for changed behavior silently serves stale cached
values (the graph trusts the version; it cannot inspect the code).

#### inputs

the input artifact ids, in the order the op receives them
positionally. Order is significant — it is part of the id.

#### params

the op’s keyword arguments. Key order is *not* significant
(the id is computed over canonical, sorted-key JSON); pass
defaults-filled, fully-named params so that two semantically equal
calls hash equal.

```pycon
>>> spec = producer_spec('embed', 'seg-1', op_version='3', model='small')
>>> spec.op, spec.inputs, spec.op_version
('embed', ('seg-1',), '3')
>>> spec.params == {'model': 'small'}
True
```

#### as_dict()

Return a plain-`dict` form of the spec — JSON-serializable.

The inverse of [`from_dict()`](#ef.artifact_graph.ProducerSpec.from_dict). Use it to persist a `producers`
registry to a `dol` JSON store.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> producer_spec('embed', 'a', 'b', op_version='1', dim=256).as_dict()
{'op': 'embed', 'op_version': '1', 'inputs': ['a', 'b'], 'params': {'dim': 256}}
```

#### *classmethod* from_dict(d)

Rebuild a [`ProducerSpec`](#ef.artifact_graph.ProducerSpec) from its [`as_dict()`](#ef.artifact_graph.ProducerSpec.as_dict) form.

* **Return type:**
  [`ProducerSpec`](#ef.artifact_graph.ProducerSpec)

```pycon
>>> spec = producer_spec('embed', 'a', 'b', op_version='1', dim=256)
>>> ProducerSpec.from_dict(spec.as_dict()) == spec
True
```

### ef.artifact_graph.artifact_id(spec)

Content-addressed id of `spec` — `sha256` over its four fields.

The id is `sha256` of canonical (sorted-key) JSON of
`{op, op_version, inputs, params}`. Two specs that describe the same
computation hash *identically*; any change to the op, its version, its
inputs (including their order) or its params yields a *different* id. That
is what makes the graph content-addressed: the id *is* the recipe.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
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
```

### ef.artifact_graph.producer_spec(op, \*inputs, op_version, \*\*params)

Build a [`ProducerSpec`](#ef.artifact_graph.ProducerSpec) ergonomically — inputs positional, params keyword.

The natural way to write a recipe: the input artifact ids are positional
(their order is significant — it is the order the op receives them), and
every keyword argument is an op parameter. `op_version` is keyword-only
and required — an op without a stated version cannot be content-addressed
safely.

* **Return type:**
  [`ProducerSpec`](#ef.artifact_graph.ProducerSpec)

```pycon
>>> spec = producer_spec('segment', 'doc-1', op_version='1', size=512, overlap=64)
>>> spec.op, spec.inputs
('segment', ('doc-1',))
>>> spec.params == {'size': 512, 'overlap': 64}
True
```
