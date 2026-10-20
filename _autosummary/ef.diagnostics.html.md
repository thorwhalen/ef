# ef.diagnostics

Staleness diagnostics — has an indexed corpus drifted from its source?

Phase 5 wired `corpus → segment → embed → vd` and recorded provenance on
every `vd` document: `source_id`, `source_hash` and `config_hash` ride
in the document metadata (see `ef.source_manager._RESERVED_METADATA_KEYS`).
This module reads that provenance back and answers a single question — \*is the
index still a faithful picture of the corpus?\* — by computing the \*\*four
staleness conditions\*\* (`ef_use_cases.md` §C):

The detection is a set comparison, exactly as the design intends — the
`vd` collection is the single source of truth for *what is indexed*, the
corpus for *what should be indexed*, and the current
[`ConfigId`](ef.config.html.md#ef.config.ConfigId) for *how*:

- `orphan` / `missing` partition the symmetric difference of the two id
  sets;
- `stale` compares each indexed source’s recorded `source_hash` against a
  freshly recomputed [`content_hash()`](ef.corpus.html.md#ef.corpus.content_hash);
- `misconfigured` compares each document’s recorded `config_hash` against
  the config being diagnosed.

This module is **read-only** — it never mutates the corpus, the collection or
the graph. Acting on a [`StalenessReport`](#ef.diagnostics.StalenessReport) is [`ef.refresh`](ef.refresh.html.md#module-ef.refresh)’s job.

Example — diagnose a config whose corpus has since drifted:

```pycon
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
```

### Functions

| [`indexed_state`](#ef.diagnostics.indexed_state)(collection)            | Group a `vd` collection's documents by source — its recorded state.     |
|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`corpus_state`](#ef.diagnostics.corpus_state)(corpus)                 | The current content hash of every source in `corpus`.                   |
| [`diagnose`](#ef.diagnostics.diagnose)(corpus, collection, config) | Compute the four staleness conditions of `collection` against `corpus`. |

### Classes

| [`IndexedSource`](#ef.diagnostics.IndexedSource)(source_id, doc_ids, ...)          | The provenance a `vd` collection records for one source document.                                                       |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [`StalenessReport`](#ef.diagnostics.StalenessReport)(config[, orphan, missing, ...]) | The four staleness conditions of one config — what [`diagnose()`](#ef.diagnostics.diagnose) returns. |

### *class* ef.diagnostics.IndexedSource(source_id, doc_ids, source_hashes, config_hashes)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The provenance a `vd` collection records for one source document.

A source is segmented into several documents; they all carry the same
`source_id` and *should* carry one `source_hash` / `config_hash` each.
[`indexed_state()`](#ef.diagnostics.indexed_state) groups a collection’s documents into one of these per
source — collecting the hashes as **sets** so a partially re-indexed source
(documents disagreeing on a hash) is detectable rather than papered over.

#### source_id

the corpus key of the source these documents were cut from.

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### doc_ids

the ids of every `vd` document belonging to the source.

* **Type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]

#### source_hashes

the distinct `source_hash` values its documents record
— a single-element set for a cleanly indexed source.

* **Type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### config_hashes

likewise the distinct `config_hash` values.

* **Type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* ef.diagnostics.StalenessReport(config, orphan=(), missing=(), stale=(), misconfigured=(), fresh=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The four staleness conditions of one config — what [`diagnose()`](#ef.diagnostics.diagnose) returns.

Each field is a sorted tuple of `source_id`s, so a report is
deterministic and JSON-friendly. `orphan` and `missing` are disjoint
from the rest (a source absent from one side cannot also be compared); a
source can be **both** `stale` and `misconfigured` (its content changed
*and* the config did) — the conditions are reported independently.
`fresh` is the complement: indexed, current and built by this config.

Truthiness reports whether *anything is wrong* — `bool(report)` is
`False` only for a perfectly in-sync index.

```pycon
>>> r = StalenessReport(config='c0', stale=('a',), fresh=('b',))
>>> bool(r), r.needs_reindexing
(True, ('a',))
>>> bool(StalenessReport(config='c0', fresh=('a', 'b')))
False
```

#### *property* needs_reindexing *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

Sources whose vectors must be recomputed — `stale` ∪ `misconfigured`.

### ef.diagnostics.corpus_state(corpus)

The current content hash of every source in `corpus`.

The *what should be indexed* half of [`diagnose()`](#ef.diagnostics.diagnose): a fresh
[`content_hash()`](ef.corpus.html.md#ef.corpus.content_hash) per source — recomputed, not read from any
registry, so it reflects out-of-band edits to the backing store. It matches
the `source_hash` [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager) writes into
`vd` (both are `content_hash` of the source with no `content_keys`).

```pycon
>>> sorted(corpus_state({'a': 'one', 'b': 'two'}))
['a', 'b']
>>> corpus_state({'a': 'x'})['a'] == corpus_state({'a': 'x'})['a']
True
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ef.diagnostics.diagnose(corpus, collection, config)

Compute the four staleness conditions of `collection` against `corpus`.

The pure core of `ef`’s diagnostics — no side effects. It reads what the
`vd` `collection` records ([`indexed_state()`](#ef.diagnostics.indexed_state)), what the `corpus`
currently holds ([`corpus_state()`](#ef.diagnostics.corpus_state)), and partitions every source id into
exactly the buckets it belongs in:

- **orphan** — in the collection, not in the corpus;
- **missing** — in the corpus, not in the collection;
- **stale** — in both, but the recorded `source_hash` ≠ the current one;
- **misconfigured** — in both, but a recorded `config_hash` ≠ `config`;
- **fresh** — in both, content current and config matching.

* **Parameters:**
  * **corpus** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]] | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the source corpus.
  * **collection** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the `vd` collection backing the config being diagnosed.
  * **config** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the [`ConfigId`](ef.config.html.md#ef.config.ConfigId) the collection *should* hold
    — documents recording any other `config_hash` are misconfigured.
* **Return type:**
  [`StalenessReport`](#ef.diagnostics.StalenessReport)
* **Returns:**
  a [`StalenessReport`](#ef.diagnostics.StalenessReport).

### ef.diagnostics.indexed_state(collection)

Group a `vd` collection’s documents by source — its recorded state.

Iterates the collection (a `MutableMapping` of `vd` documents) and
folds every document `ef` wrote into an [`IndexedSource`](#ef.diagnostics.IndexedSource) keyed by
its `source_id`. Documents with no `source_id` metadata (not written by
`ef`) are skipped. This is the *what is indexed* half of [`diagnose()`](#ef.diagnostics.diagnose).

* **Parameters:**
  **collection** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the `vd` collection backing one config.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`IndexedSource`](#ef.diagnostics.IndexedSource)]
* **Returns:**
  a `{source_id: IndexedSource}` mapping.
