# ef.corpus

The `Corpus` abstraction — `ef`’s source layer (L0) — and change detection.

A *corpus* is the entry point of the pipeline: the set of source documents to
be segmented, embedded and indexed. In `ef` a corpus is deliberately *not* a
new class — it is just a mapping:

```default
Corpus = MutableMapping[source_id, Source]
```

Any `dol`-style mapping is a corpus: a plain `dict` (RAM), a filesystem
folder, an S3 bucket, an API-backed store. The backing store is swappable
because the rest of `ef` only ever sees a `MutableMapping`. This module
provides:

- [`Source`](#ef.corpus.Source) / [`Corpus`](#ef.corpus.Corpus) — the type aliases that name the contract.
- [`as_corpus()`](#ef.corpus.as_corpus) — the dependency-injection seam: `None` / a mapping / a
  directory path / an iterable of strings → a corpus. Mirrors
  [`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder) and
  [`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter).
- [`content_hash()`](#ef.corpus.content_hash) — the content hash of a source, the basis of change
  detection. Hashes *normalized* content (via [`ef.hashing`](ef.hashing.html.md#module-ef.hashing)); for a
  structured source it hashes the text only, so metadata-only edits never
  trigger a spurious re-embed.
- [`ChangeDetectingCorpus`](#ef.corpus.ChangeDetectingCorpus) — a corpus *wrapper* that content-hashes
  values and reports what changed, both through-the-wrapper (via an
  `on_change` callback) and out-of-band (via [`scan()`](#ef.corpus.ChangeDetectingCorpus.scan)).

[`ChangeDetectingCorpus`](#ef.corpus.ChangeDetectingCorpus) is the  *“something changed”* half of incremental
refresh; the downstream cascade is the `ArtifactGraph` (a later phase). The
`on_change` callback is the seam those two halves meet at.

Example — detect changes as they are written:

```pycon
>>> events = []
>>> corpus = ChangeDetectingCorpus(on_change=events.append)
>>> corpus['doc1'] = 'hello world'
>>> corpus['doc1'] = 'hello world'      # identical content — idempotent, no event
>>> corpus['doc1'] = 'hello there'      # a real change
>>> del corpus['doc1']
>>> [(e.source_id, e.kind) for e in events]
[('doc1', 'added'), ('doc1', 'modified'), ('doc1', 'deleted')]
```

### Module Attributes

| [`Source`](#ef.corpus.Source)     | A source document.                                                                                  |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`Corpus`](#ef.corpus.Corpus)     | a mapping from `source_id` to [`Source`](#ef.corpus.Source).              |
| [`ChangeKind`](#ef.corpus.ChangeKind) | The kinds of change [`ChangeDetectingCorpus`](#ef.corpus.ChangeDetectingCorpus) reports. |

### Functions

| [`content_hash`](#ef.corpus.content_hash)(source, \*[, content_keys])   | Content hash of a source document — `sha256` of its normalized content.   |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`as_corpus`](#ef.corpus.as_corpus)([x])                             | Normalize `x` into a corpus (`MutableMapping[source_id, Source]`).        |

### Classes

| [`ChangeEvent`](#ef.corpus.ChangeEvent)(source_id, kind[, old_hash, ...])     | A single detected change to a source in a corpus.                        |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`CorpusDiff`](#ef.corpus.CorpusDiff)([added, modified, deleted, unchanged]) | The result of comparing a corpus's content against known content hashes. |
| [`ChangeDetectingCorpus`](#ef.corpus.ChangeDetectingCorpus)([corpus, on_change, ...])   | A corpus wrapper that content-hashes values and reports what changed.    |

### *class* ef.corpus.ChangeDetectingCorpus(corpus=None, , on_change=None, hashes=None, content_keys=None, fingerprint=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

A corpus wrapper that content-hashes values and reports what changed.

Wraps any inner corpus (a `MutableMapping[source_id, Source]`) and adds
change detection *without* altering its mapping behavior — reads, writes and
deletes pass straight through to the inner store. It is to a corpus what
[`CachedEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.CachedEmbedder) is to an embedder: a
transparent, composable layer.

Changes are noticed two ways:

- **Through the wrapper.** A `corpus[id] = source` or `del corpus[id]`
  that actually changes content fires the injected `on_change` callback
  with a [`ChangeEvent`](#ef.corpus.ChangeEvent). Writing *identical* content is a no-op (no
  event) — the basis of idempotent ingestion.
- **Out of band.** When the backing store is edited directly (a file
  changed on disk, an S3 object replaced), [`scan()`](#ef.corpus.ChangeDetectingCorpus.scan) re-reads every
  source, compares hashes, fires events for what differs and returns a
  [`CorpusDiff`](#ef.corpus.CorpusDiff). [`diff()`](#ef.corpus.ChangeDetectingCorpus.diff) runs the same comparison with no side
  > effects.

The `on_change` callback is the seam the downstream `ArtifactGraph`
plugs into: a `"modified"` / `"deleted"` event is exactly the trigger
for a cascade invalidation.

On construction a *baseline* is established: the current content hashes of
the inner corpus are recorded silently (no events). Pass an already-populated
`hashes` registry to instead resume from a persisted baseline — then call
[`scan()`](#ef.corpus.ChangeDetectingCorpus.scan) to reconcile any drift.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the inner corpus to wrap. Coerced via [`as_corpus()`](#ef.corpus.as_corpus), so a
    directory path / `dict` / iterable of sources / `None` all work.
  * **on_change** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`ChangeEvent`](#ef.corpus.ChangeEvent)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – called with each [`ChangeEvent`](#ef.corpus.ChangeEvent). Defaults to a no-op.
    It should be cheap and not raise; an exception propagates to the
    caller *after* the corpus and registry are already consistent.
  * **hashes** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the hash registry — a `MutableMapping[source_id, str]` of the
    last-known content hash per source. Injectable so it can be
    persisted (e.g. a `dol` store); defaults to an in-RAM `dict`.
    If passed already-populated it is trusted as the baseline.
  * **content_keys** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – forwarded to [`content_hash()`](#ef.corpus.content_hash) — metadata keys that
    participate in the hash.
  * **fingerprint** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Hashable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an optional cheap `source_id -> Hashable` callback (an
    mtime / etag prefilter). When given, [`scan()`](#ef.corpus.ChangeDetectingCorpus.scan) / [`diff()`](#ef.corpus.ChangeDetectingCorpus.diff)
    skip re-hashing a source whose fingerprint is unchanged. The
    content hash always remains the source of truth — the fingerprint
    is only a speed optimization, never trusted to *declare* a change.

```pycon
>>> backing = {'a': 'one', 'b': 'two'}
>>> corpus = ChangeDetectingCorpus(backing)   # baseline adopts 'a' and 'b'
>>> backing['a'] = 'ONE'                      # edited behind the wrapper's back
>>> backing['c'] = 'three'
>>> del backing['b']
>>> d = corpus.scan()
>>> (d.added, d.modified, d.deleted)
(('c',), ('a',), ('b',))
```

#### diff()

Compare current corpus content against the registry — **no side effects**.

Detects out-of-band edits (changes made to the backing store directly)
without firing events or updating the registry. [`scan()`](#ef.corpus.ChangeDetectingCorpus.scan) is the
effectful counterpart.

* **Return type:**
  [`CorpusDiff`](#ef.corpus.CorpusDiff)

#### scan()

Reconcile the registry with the corpus, fire events, return the diff.

Use this to pick up **out-of-band** edits — changes made to the backing
store directly rather than through this wrapper. Every detected change
fires a [`ChangeEvent`](#ef.corpus.ChangeEvent) and the registry (and fingerprints) are
advanced to the new state, so a second [`scan()`](#ef.corpus.ChangeDetectingCorpus.scan) with nothing
further changed is a no-op. [`diff()`](#ef.corpus.ChangeDetectingCorpus.diff) is the side-effect-free version.

* **Return type:**
  [`CorpusDiff`](#ef.corpus.CorpusDiff)

### *class* ef.corpus.ChangeEvent(source_id, kind, old_hash=None, new_hash=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A single detected change to a source in a corpus.

Emitted by [`ChangeDetectingCorpus`](#ef.corpus.ChangeDetectingCorpus) to its `on_change` callback.
`old_hash` is `None` for an `"added"` event, `new_hash` is `None`
for a `"deleted"` one.

#### source_id

the key of the source that changed.

#### kind

`"added"`, `"modified"` or `"deleted"`.

#### old_hash

the source’s previous content hash, if any.

#### new_hash

the source’s new content hash, if any.

### ef.corpus.ChangeKind

The kinds of change [`ChangeDetectingCorpus`](#ef.corpus.ChangeDetectingCorpus) reports.

alias of [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[‘added’, ‘modified’, ‘deleted’]

### ef.corpus.Corpus

a mapping from `source_id` to [`Source`](#ef.corpus.Source). `ef` never
requires a dedicated class — any `dol`-style `MutableMapping` (RAM,
filesystem, S3, API) *is* a corpus. [`as_corpus()`](#ef.corpus.as_corpus) builds one.

* **Type:**
  A corpus

alias of [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### *class* ef.corpus.CorpusDiff(added=(), modified=(), deleted=(), unchanged=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The result of comparing a corpus’s content against known content hashes.

The four fields partition the corpus’s source ids; each is a sorted tuple,
so a [`CorpusDiff`](#ef.corpus.CorpusDiff) is deterministic and JSON-friendly. Truthiness
reports whether *anything* changed.

```pycon
>>> d = CorpusDiff(added=('b',), modified=('a',))
>>> bool(d), d.changed
(True, ('b', 'a'))
>>> bool(CorpusDiff(unchanged=('a', 'b')))
False
```

#### *property* changed *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]*

The source ids that are new or modified — the set to (re-)materialize.

### ef.corpus.Source *= str | collections.abc.Mapping[str, typing.Any]*

A source document. Either plain `text`, or a `Mapping` carrying a
`text` key alongside metadata — the same shape a
[`Segmenter`](ef.segmenters.html.md#ef.segmenters.Segmenter) accepts as input.

### ef.corpus.as_corpus(x=None, , \*\*kwargs)

Normalize `x` into a corpus (`MutableMapping[source_id, Source]`).

The single dependency-injection seam through which every `ef` entry point
accepts a user-supplied corpus — the mirror of
[`as_embedder()`](ef.embedder_adapters.html.md#ef.embedder_adapters.as_embedder) and
[`as_segmenter()`](ef.segmenter_adapters.html.md#ef.segmenter_adapters.as_segmenter). Accepts, in order:

1. `None` — a fresh empty in-RAM `dict`;
2. a `Mapping` — returned **unchanged** (a `dict`, a `dol` store,
   another corpus: it already *is* a corpus);
3. a directory path (`str` / `os.PathLike` naming an existing
   directory) — a `dol` filesystem store of the text files under it
   (`dol` is imported lazily; only this branch needs it);
4. an iterable of sources (strings or mappings) — an in-RAM `dict` keyed
   by each source’s [`content_hash()`](#ef.corpus.content_hash) (content-addressed, so duplicate
   sources collapse to one entry).

Extra `**kwargs` are forwarded to the `dol` filesystem store (the
directory branch only) and ignored otherwise.

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is a non-directory string, or none of the above.

```pycon
>>> as_corpus()
{}
>>> as_corpus({'a': 'x'})                 # a mapping passes straight through
{'a': 'x'}
>>> sorted(as_corpus(['alpha', 'beta']).values())
['alpha', 'beta']
>>> len(as_corpus(['dup', 'dup']))        # content-addressed: duplicates collapse
1
```

### ef.corpus.content_hash(source, , content_keys=None)

Content hash of a source document — `sha256` of its normalized content.

This is the basis of change detection: re-reading an unchanged source
yields an identical hash, so a refresh that finds the same hash does
nothing. Hashing *normalized* content (NFC, no BOM, `\n` line endings —
see [`ef.hashing`](ef.hashing.html.md#module-ef.hashing)) means cosmetic encoding differences never look like
a change.

The treatment of a `Mapping` source is deliberate:

- a plain `str` source — the hash covers the text;
- a `Mapping` with a `text` key — the hash covers `source['text']`
  **only**. A change to title, tags, ACL or any other metadata does *not*
  change the hash, so it never triggers a spurious re-segment/re-embed
  (embedding is the pipeline’s most expensive step). Name the metadata keys
  that *do* feed downstream in `content_keys` and changes to those keys
  will be detected;
- a `Mapping` *without* a `text` key — treated as a structured document
  and hashed via canonical (sorted-key) JSON of the whole mapping;
- `bytes` — hashed raw (no text normalization is possible pre-parse).

* **Parameters:**
  * **source** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – a `str`, `bytes`, or a `Mapping` (ideally with a `text`
    key).
  * **content_keys** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – metadata keys whose values join the text in the hash.
    Each key is looked up at the top level of `source` or inside a
    `source['metadata']` sub-mapping. Use this *only* for metadata
    that genuinely affects the L1–L4 pipeline (e.g. a title prepended
    to the text before embedding); key order does not matter.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `source` is not a `str`, `bytes` or `Mapping`.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> content_hash('hello') == content_hash('hello')
True
>>> content_hash('hello') == content_hash('hello world')
False
>>> content_hash('a\r\nb') == content_hash('a\nb')   # line endings normalized
True
>>> a = {'text': 'body', 'metadata': {'tags': ['x']}}
>>> b = {'text': 'body', 'metadata': {'tags': ['y']}}
>>> content_hash(a) == content_hash(b)                # metadata ignored by default
True
>>> content_hash(a, content_keys=['tags']) == content_hash(b, content_keys=['tags'])
False
>>> content_hash({'text': 'body'}) == content_hash('body')  # wrapper is transparent
True
```
