# ef.ablation

Embedder ablations on a graded benchmark: run named *arms* over one corpus.

An ablation asks “which change to the embedder moved the metric?”. Each **arm**
is one embedding configuration (a query-side embedder, optionally a different
document-side one); every arm ranks the *same* segmented corpus for the *same*
queries and is scored by [`evaluate_retrieval()`](ef.evaluation.html.md#ef.evaluation.evaluate_retrieval), so the only
thing that varies is the embedder.

Simple use, an A/B over any two embedders:

```default
bench = read_graded_cases("cases.jsonl", "named_sets.json")
report = run_arms(corpus, {"A": Arm(emb_a), "B": Arm(emb_b)}, bench)
print(report)
```

The instruction-embedding experiment of ef#34 is one instance:
[`instruction_arms()`](#ef.ablation.instruction_arms) builds its four arms (A: plain model, B: plain model +
instruction, the control, C: instruction-tuned model + instruction, C’: the same
model without it) and [`instruction_verdict()`](#ef.ablation.instruction_verdict) checks its success line.

Ranking is dense and document-level: every segment is embedded once per
*document-side* embedder (`input_type="document"`), a query is embedded with
the arm’s embedder (`input_type="query"`), and a document scores the best
cosine of any of its segments (max-pooling). Arms that share a document-side
embedder share its vectors: pass `doc_vectors` (any `MutableMapping`, e.g. a
`dol` store) to keep them across calls too. Only the query side differs between
an instructed arm and its bare twin, so the corpus is embedded once per model.

The labels format [`read_graded_cases()`](#ef.ablation.read_graded_cases) reads is the one `ir.eval` writes:
a JSONL of `{"artifact_id", "labels": {theme: level}}` cases with an optional
`{"__meta__": {"probes": {theme: query}}}` header, plus an optional
`{theme: {"distractors": [...], "hard_positives": [...]}}` JSON of named sets.

### Module Attributes

| [`DFLT_GRADES`](#ef.ablation.DFLT_GRADES)              | Graded level -> relevance.                                                                                                                                                                                                  |
|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_TOPICAL_INSTRUCTION`](#ef.ablation.DFLT_TOPICAL_INSTRUCTION) | The topical instruction for the instruction-tuned arm, in Qwen3-Embedding's format (`Instruct: <task>\nQuery:` with the query appended directly; E5-instruct wants `Query: `` with a space, so pass ``instruction=` there). |

### Functions

| [`instruction_arms`](#ef.ablation.instruction_arms)(base[, tuned, instruction, ...])   | The four arms of the instruction experiment (A, B, C, C').                |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`instruction_verdict`](#ef.ablation.instruction_verdict)(metrics, \*[, k, fp_cut, ...])  | Check ef#34's success line from `{arm: {"fp_rate@20": ..., ...}}`.        |
| [`read_graded_cases`](#ef.ablation.read_graded_cases)(cases_path[, ...])                | Load an `ir.eval`-style graded case file (and named sets) as a benchmark. |
| [`run_arms`](#ef.ablation.run_arms)(corpus, arms, benchmark, \*[, ...])        | Score every arm on the same segmented corpus; one row per arm.            |
| [`segment_corpus`](#ef.ablation.segment_corpus)(corpus, \*[, segmenter])             | `{doc_id: [segment text, ...]}`: strings are segmented, sequences kept.   |

### Classes

| [`Arm`](#ef.ablation.Arm)(embedder[, doc_embedder])           | One embedding configuration: `embedder` for queries, `doc_embedder` for documents.   |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`GradedBenchmark`](#ef.ablation.GradedBenchmark)(qrels, queries[, ...])  | Queries, graded qrels and the named diagnostic sets, keyed by query id.              |
| [`MaxPoolRetriever`](#ef.ablation.MaxPoolRetriever)(doc_ids, vectors, ...) | Rank documents by the best cosine of any of their segments to the query.             |

### *class* ef.ablation.Arm(embedder, doc_embedder=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One embedding configuration: `embedder` for queries, `doc_embedder` for documents.

`doc_embedder` defaults to `embedder`. Give an instructed arm its bare
twin as `doc_embedder` when the instruction is query-side only, so the
corpus vectors are shared rather than recomputed. Either may be anything
[`ef.as_embedder()`](ef.html.md#ef.as_embedder) accepts (an `Embedder`, a spec string, a callable).

### ef.ablation.DFLT_GRADES *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [float](https://docs.python.org/3/builtins/functions.html#float)]* *= {'core': 2.0, 'strong': 1.0}*

Graded level -> relevance. Levels absent here count as not relevant, so
recall is “core/strong recall”.

### ef.ablation.DFLT_TOPICAL_INSTRUCTION *= 'Instruct: Given a topic, retrieve software packages whose main purpose is that topic, not packages that only mention related words\\nQuery:'*

The topical instruction for the instruction-tuned arm, in Qwen3-Embedding’s
format (`Instruct: <task>\nQuery:` with the query appended directly; E5-instruct
wants `Query: `` with a space, so pass ``instruction=` there). It asks for
what a document is *about*, which a plain model confuses with surface keywords.

### *class* ef.ablation.GradedBenchmark(qrels, queries, distractors=dict(), hard_positives=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Queries, graded qrels and the named diagnostic sets, keyed by query id.

```pycon
>>> b = GradedBenchmark(qrels={'t': {'d1': 2.0, 'd9': 1.0}}, queries={'t': 'x'},
...                     distractors={'t': ('d2', 'd8')})
>>> b.restrict(['d1', 'd2']).qrels, b.restrict(['d1', 'd2']).distractors
({'t': {'d1': 2.0}}, {'t': ('d2',)})
```

#### restrict(doc_ids)

The same benchmark with every judgement limited to `doc_ids`.

Use it when some labelled documents could not be put in the corpus: a
document that cannot be retrieved would otherwise count as a miss for
every arm alike, diluting the differences.

* **Return type:**
  [`GradedBenchmark`](#ef.ablation.GradedBenchmark)

### *class* ef.ablation.MaxPoolRetriever(doc_ids, vectors, query_embedder)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Rank documents by the best cosine of any of their segments to the query.

```pycon
>>> from ef.embedders import HashingEmbedder
>>> emb = HashingEmbedder()
>>> segs = {'fruit': ['red apple', 'ripe pear'], 'car': ['blue car engine']}
>>> r = MaxPoolRetriever.from_segments(segs, emb)
>>> r('apple', limit=1)
['fruit']
```

### ef.ablation.instruction_arms(base, tuned=None, , instruction='Instruct: Given a topic, retrieve software packages whose main purpose is that topic, not packages that only mention related words\\nQuery:', base_instruction=None)

The four arms of the instruction experiment (A, B, C, C’).

- `A`: `base` as is (the plain baseline);
- `B`: `base` with the instruction on queries: the **control**, expected
  to move little, since a plain model was not trained to read instructions;
- `C`: `tuned` (an instruction-tuned model) with the instruction;
- `C'`: `tuned` without it, attributing C’s gain to the instruction
  rather than to the bigger model.

Instructions are query-side only (documents bare), so each model embeds the
corpus once. `base` / `tuned` are embedders or [`ef.as_embedder()`](ef.html.md#ef.as_embedder)
specs; without `tuned` only A and B are built.

* **Parameters:**
  * **base** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the plain embedder (e.g. `all-MiniLM-L6-v2`).
  * **tuned** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – the instruction-tuned embedder (e.g. Qwen3-Embedding-0.6B).
  * **instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the query prefix for C, verbatim (model separator included).
  * **base_instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the query prefix for B; defaults to `instruction`.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Arm`](#ef.ablation.Arm)]

```pycon
>>> from ef.embedders import HashingEmbedder
>>> sorted(instruction_arms(HashingEmbedder(), HashingEmbedder()))
['A', 'B', 'C', "C'"]
```

### ef.ablation.instruction_verdict(metrics, , k=20, fp_cut=0.5, control_tolerance=0.02, control_metrics=('ndcg', 'recall', 'fp_rate'))

Check ef#34’s success line from `{arm: {"fp_rate@20": ..., ...}}`.

Success = C cuts the named-distractor `fp_rate@k` by at least `fp_cut`
(relative) vs A, at equal-or-better `recall@k` (core/strong recall), and
the control B moves every `control_metrics` at `k` by less than
`control_tolerance` (absolute; 0.02 = 2 points). Also reports whether C
beats its bare twin C’ on `fp_rate@k` (the gain is the instruction’s).
If A has no distractor in its top `k` there is nothing to cut:
`fp_baseline_zero` is set and the verdict is not a success.

* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – when an arm (A, B, C) or one of its metrics at `k` is missing.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

Accepts `BenchmarkReport.rows` turned into a mapping, e.g.
`{r['embedder']: r for r in report.rows}`.

```pycon
>>> m = {'A': {'fp_rate@20': .2, 'recall@20': .4, 'ndcg@20': .7},
...      'B': {'fp_rate@20': .2, 'recall@20': .41, 'ndcg@20': .7},
...      'C': {'fp_rate@20': .05, 'recall@20': .5, 'ndcg@20': .8},
...      "C'": {'fp_rate@20': .15, 'recall@20': .45, 'ndcg@20': .75}}
>>> v = instruction_verdict(m)
>>> v['success'], round(v['fp_reduction'], 2), v['instruction_beats_bare']
(True, 0.75, True)
```

### ef.ablation.read_graded_cases(cases_path, named_sets_path=None, , grades={'core': 2.0, 'strong': 1.0}, probes=None)

Load an `ir.eval`-style graded case file (and named sets) as a benchmark.

Each theme is one query: its text comes from `probes` or, failing that, the
`__meta__` header’s `probes`; its qrels are the cases whose level for that
theme is in `grades`. A theme with no such case is left out.

* **Parameters:**
  * **cases_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) – JSONL of `{"artifact_id", "labels": {theme: level}}`.
  * **named_sets_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional JSON `{theme: {"distractors", "hard_positives"}}`.
  * **grades** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – level -> relevance grade; other levels are not relevant.
  * **probes** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `{theme: query text}`, overriding the header’s.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – when a labelled theme has no query text.
* **Return type:**
  [`GradedBenchmark`](#ef.ablation.GradedBenchmark)

### ef.ablation.run_arms(corpus, arms, benchmark, , k_values=(10, 20), segmenter=None, doc_vectors=None)

Score every arm on the same segmented corpus; one row per arm.

* **Parameters:**
  * **corpus** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `{doc_id: text}` (segmented with `segmenter`) or
    `{doc_id: [segment, ...]}` (used as is).
  * **arms** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Arm`](#ef.ablation.Arm) | [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `{name: Arm}`; a bare embedder is taken as `Arm(embedder)`.
  * **benchmark** ([`GradedBenchmark`](#ef.ablation.GradedBenchmark)) – what to score against ([`read_graded_cases()`](#ef.ablation.read_graded_cases)).
  * **k_values** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – rank cutoffs.
  * **segmenter** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – for string documents; see [`segment_corpus()`](#ef.ablation.segment_corpus).
  * **doc_vectors** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `ndarray`] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – cache of segment matrices keyed by
    `"<document-side model_id>/<digest of the segmented corpus>"`, so a
    persistent store (e.g. a `dol` store) can be reused across runs and
    corpora. A fresh dict by default; embedders without a `model_id`
    are never cached.
* **Return type:**
  [`BenchmarkReport`](ef.evaluation.html.md#ef.evaluation.BenchmarkReport)

```pycon
>>> from ef.embedders import HashingEmbedder
>>> bench = GradedBenchmark({'q': {'fruit': 1.0}}, {'q': 'apple'})
>>> rep = run_arms({'fruit': 'red apple', 'car': 'blue car'},
...                {'A': HashingEmbedder()}, bench, k_values=(1,))
>>> rep.rows[0]['embedder'], rep.rows[0]['recall@1']
('A', 1.0)
```

### ef.ablation.segment_corpus(corpus, , segmenter=None)

`{doc_id: [segment text, ...]}`: strings are segmented, sequences kept.

`segmenter` is anything [`ef.as_segmenter()`](ef.html.md#ef.as_segmenter) accepts (default: the
recursive character splitter). Empty segments are dropped; a document left
with none is represented by its id, so it stays rankable.

```pycon
>>> segment_corpus({'a': ['x', ''], 'b': ''})
{'a': ['x'], 'b': ['b']}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
