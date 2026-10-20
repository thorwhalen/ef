# ef.evaluation

Evaluation hookpoints — measuring retrieval and RAG quality.

Phase 7 makes an indexed `ef` corpus **measurable**. It stays a *facade*: it
computes the standard, deterministic metrics and provides the standard data
shapes, but it never synthesizes an answer and never requires an LLM (the §6
boundary). LLM-judged metrics are reached through a thin Ragas bridge.

Two journeys (`ef_use_cases.md` §F7–F8), two evaluation conventions
(`ef_design_notes.md` §6):

- **Retrieval evaluation** — [`evaluate_retrieval()`](#ef.evaluation.evaluate_retrieval), BEIR/MTEB-shaped. The
  inputs are the BEIR triple — a corpus, a set of `queries` and `qrels`
  (graded relevance judgements) — and the primary metric is **NDCG@10**. The
  metric primitives ([`ndcg_at_k()`](#ef.evaluation.ndcg_at_k), [`recall_at_k()`](#ef.evaluation.recall_at_k), …) are pure
  functions, usable on their own. [`read_beir()`](#ef.evaluation.read_beir) / [`write_beir()`](#ef.evaluation.write_beir) move
  the triple to and from the on-disk `corpus.jsonl` / `queries.jsonl` /
  `qrels.tsv` format.
- **RAG evaluation** — [`evaluate_rag()`](#ef.evaluation.evaluate_rag), Ragas `SingleTurnSample`-shaped.
  A sample is `(user_input, response, retrieved_contexts, reference)`;
  [`RagSample`](#ef.evaluation.RagSample) names that shape. [`evaluate_rag()`](#ef.evaluation.evaluate_rag) computes only
  > **deterministic, reference-based lexical metrics** ([`exact_match()`](#ef.evaluation.exact_match),

  [`token_f1()`](#ef.evaluation.token_f1), [`context_recall()`](#ef.evaluation.context_recall), [`context_precision()`](#ef.evaluation.context_precision)) — no LLM.

  For the LLM-judged metrics (faithfulness, answer relevancy, …) hand the
  samples to [`as_ragas_dataset()`](#ef.evaluation.as_ragas_dataset) and run Ragas with your own LLM.

Example — score a retriever against a tiny BEIR-shaped triple:

```pycon
>>> qrels = {'q1': {'d1': 1.0, 'd2': 1.0}, 'q2': {'d3': 1.0}}
>>> queries = {'q1': 'first query', 'q2': 'second query'}
>>> rankings = {'first query': ['d1', 'd9', 'd2'], 'second query': ['d9', 'd3']}
>>> def retriever(query, *, limit=10):
...     return rankings[query][:limit]
>>> report = evaluate_retrieval(retriever, qrels, queries, k_values=(1, 3))
>>> report.metrics['recall@3']
1.0
>>> round(report.primary_at(1), 3)        # NDCG@1, averaged over the two queries
0.5
```

### Module Attributes

| [`Qrels`](#ef.evaluation.Qrels)             | Graded relevance judgements — `{query_id: {doc_id: relevance}}`.                                                                                                                          |
|--------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`RETRIEVAL_METRICS`](#ef.evaluation.RETRIEVAL_METRICS) | The retrieval metrics [`evaluate_retrieval()`](#ef.evaluation.evaluate_retrieval) can compute, each keyed by the short name used in the `metrics=` argument and in report keys. |

### Functions

| [`dcg_at_k`](#ef.evaluation.dcg_at_k)(ranked, relevant, k)                     | Discounted cumulative gain of `ranked` at cutoff `k`.                                                                            |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`ndcg_at_k`](#ef.evaluation.ndcg_at_k)(ranked, relevant, k)                    | Normalized DCG at cutoff `k` — `ef`'s **primary** retrieval metric.                                                              |
| [`recall_at_k`](#ef.evaluation.recall_at_k)(ranked, relevant, k)                  | Fraction of the relevant docs that appear in the top `k` results.                                                                |
| [`precision_at_k`](#ef.evaluation.precision_at_k)(ranked, relevant, k)               | Fraction of the top `k` results that are relevant.                                                                               |
| [`reciprocal_rank`](#ef.evaluation.reciprocal_rank)(ranked, relevant[, k])            | Reciprocal of the rank of the first relevant result (`0` if none).                                                               |
| [`average_precision`](#ef.evaluation.average_precision)(ranked, relevant[, k])          | Average precision of `ranked` — the per-query term of MAP.                                                                       |
| [`fp_rate_at_k`](#ef.evaluation.fp_rate_at_k)(ranked, distractors, k)              | Fraction of the *named distractors* that intrude into the top `k`.                                                               |
| [`hard_positive_recall_at_k`](#ef.evaluation.hard_positive_recall_at_k)(ranked, ...)            | Fraction of the *hard positives* (e.g. thin-description docs) found in the top `k`.                                              |
| [`evaluate_retrieval`](#ef.evaluation.evaluate_retrieval)(retriever, qrels, queries, \*) | Score a retriever against a BEIR-shaped evaluation set.                                                                          |
| [`benchmark_embedders`](#ef.evaluation.benchmark_embedders)(corpus_sources, ...[, ...])   | Re-index the same corpus under each embedder and tabulate the metrics.                                                           |
| [`read_beir`](#ef.evaluation.read_beir)(directory, \*[, qrels])                 | Load a BEIR-format dataset from disk into the in-memory evaluation triple.                                                       |
| [`write_beir`](#ef.evaluation.write_beir)(directory, corpus, queries, qrels)     | Write an evaluation triple to disk in BEIR format — the inverse of [`read_beir()`](#ef.evaluation.read_beir). |
| [`exact_match`](#ef.evaluation.exact_match)(response, reference)                  | `1.0` if `response` equals `reference` after normalization, else `0.0`.                                                          |
| [`token_f1`](#ef.evaluation.token_f1)(response, reference)                     | Token-overlap F1 of `response` against `reference` (the SQuAD F1).                                                               |
| [`context_recall`](#ef.evaluation.context_recall)(reference, retrieved_contexts)     | Lexical recall of the `reference` tokens by the retrieved contexts.                                                              |
| [`context_precision`](#ef.evaluation.context_precision)(reference, retrieved_contexts)  | Fraction of the retrieved contexts that lexically overlap the `reference`.                                                       |
| [`evaluate_rag`](#ef.evaluation.evaluate_rag)(samples, \*[, metrics])              | Score RAG samples with deterministic, reference-based lexical metrics.                                                           |
| [`as_ragas_dataset`](#ef.evaluation.as_ragas_dataset)(samples)                         | Convert [`RagSample`](#ef.evaluation.RagSample)s into a Ragas `EvaluationDataset`.                            |

### Classes

| [`RagSample`](#ef.evaluation.RagSample)                                    | One end-to-end RAG interaction — the unit [`evaluate_rag()`](#ef.evaluation.evaluate_rag) scores.       |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [`RetrievalEvalReport`](#ef.evaluation.RetrievalEvalReport)(metrics, per_query, ...) | The outcome of [`evaluate_retrieval()`](#ef.evaluation.evaluate_retrieval) — metrics over a query set.        |
| [`BenchmarkReport`](#ef.evaluation.BenchmarkReport)(rows, reports)               | One row per embedder: its `{metric: value}` (plus an `embedder` key).                                                   |
| [`RagEvalReport`](#ef.evaluation.RagEvalReport)(metrics, coverage, n_samples)  | The outcome of [`evaluate_rag()`](#ef.evaluation.evaluate_rag) — lexical RAG metrics over a sample set. |

### *class* ef.evaluation.BenchmarkReport(rows, reports)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One row per embedder: its `{metric: value}` (plus an `embedder` key).

`reports` keeps each full [`RetrievalEvalReport`](#ef.evaluation.RetrievalEvalReport) for drill-down
(e.g. `distractor_hits`). `to_frame()` gives a pandas DataFrame
(pandas is imported lazily).

```pycon
>>> r = BenchmarkReport(rows=[{'embedder': 'a', 'ndcg@10': 0.5}], reports={})
>>> r.to_records()
[{'embedder': 'a', 'ndcg@10': 0.5}]
```

### ef.evaluation.Qrels

Graded relevance judgements — `{query_id: {doc_id: relevance}}`. A
`relevance` of `0` means *judged non-relevant*; `> 0` means relevant
(graded judgements feed [`ndcg_at_k()`](#ef.evaluation.ndcg_at_k), binary relevance the rest). This
is the in-memory form of a BEIR `qrels.tsv`.

alias of [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### ef.evaluation.RETRIEVAL_METRICS *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)], [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [float](https://docs.python.org/3/builtins/functions.html#float)], [int](https://docs.python.org/3/builtins/functions.html#int)], [float](https://docs.python.org/3/builtins/functions.html#float)]]* *= {'map': <function average_precision>, 'mrr': <function reciprocal_rank>, 'ndcg': <function ndcg_at_k>, 'precision': <function precision_at_k>, 'recall': <function recall_at_k>}*

The retrieval metrics [`evaluate_retrieval()`](#ef.evaluation.evaluate_retrieval) can compute, each keyed by
the short name used in the `metrics=` argument and in report keys. Public
so downstream packages (e.g. `ir.eval`) can consume this single source of
truth — the name→function registry *and* the set of valid metric names —
instead of hand-mirroring it.

### *class* ef.evaluation.RagEvalReport(metrics, coverage, n_samples)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`evaluate_rag()`](#ef.evaluation.evaluate_rag) — lexical RAG metrics over a sample set.

`metrics` maps each metric to its mean over the samples it applied to;
`coverage` records how many samples that was (a metric needing a
`reference` skips samples without one). `n_samples` is the total scored.

```pycon
>>> report = RagEvalReport(
...     metrics={'exact_match': 0.5}, coverage={'exact_match': 4}, n_samples=4)
>>> report.metrics['exact_match']
0.5
```

### *class* ef.evaluation.RagSample

Bases: [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)

One end-to-end RAG interaction — the unit [`evaluate_rag()`](#ef.evaluation.evaluate_rag) scores.

The field names match Ragas’ `SingleTurnSample` exactly, so a
[`RagSample`](#ef.evaluation.RagSample) is the interchange shape `app_ef` / `srag` / `raglab`
produce and [`as_ragas_dataset()`](#ef.evaluation.as_ragas_dataset) forwards unchanged. `ef` itself fills
only `retrieved_contexts` (from [`retrieve()`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve));
`response` is the *caller’s* LLM output — `ef` does not synthesize it.

Keys:
: user_input: the user’s question.
  response: the generated answer (the caller’s LLM produced it).
  retrieved_contexts: the context passages handed to the LLM — what
  <br/>
  > [`retrieve()`](ef.source_manager.html.md#ef.source_manager.SourceManager.retrieve) returned, as text.
  <br/>
  reference: the ground-truth answer, if known.
  reference_contexts: the ground-truth context passages, if known.

### *class* ef.evaluation.RetrievalEvalReport(metrics, per_query, n_queries, k_values, distractor_hits=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The outcome of [`evaluate_retrieval()`](#ef.evaluation.evaluate_retrieval) — metrics over a query set.

`metrics` maps a `"<name>@<k>"` key (e.g. `"ndcg@10"`) to the mean of
that metric over every scored query; `per_query` keeps the same keys per
query for drill-down. `n_queries` is how many queries were actually
scored (a query with no positive judgement is skipped).

```pycon
>>> report = RetrievalEvalReport(
...     metrics={'ndcg@10': 0.83}, per_query={}, n_queries=12, k_values=(10,))
>>> report.primary
0.83
```

#### distractor_hits *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [int](https://docs.python.org/3/builtins/functions.html#int)]]*

`{query_id: {distractor_id: rank}}` — the named distractors that landed in
the top `max(k_values)` (1-based rank); empty unless `distractors` was given.

#### *property* primary *: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The headline number — mean [NDCG@10](mailto:NDCG@10), or `None` if it was not computed.

#### primary_at(k)

Mean NDCG at cutoff `k` — `None` if NDCG@\`\`k\`\` was not computed.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ef.evaluation.as_ragas_dataset(samples)

Convert [`RagSample`](#ef.evaluation.RagSample)s into a Ragas `EvaluationDataset`.

The hookpoint for the *LLM-judged* RAG metrics `ef` deliberately does not
compute itself (faithfulness, answer relevancy, …). [`RagSample`](#ef.evaluation.RagSample)
already mirrors Ragas’ `SingleTurnSample` field-for-field, so this is a
thin, lossless adapter. Run the evaluation with your own LLM:

```default
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

dataset = as_ragas_dataset(samples)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```

* **Parameters:**
  **samples** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`RagSample`](#ef.evaluation.RagSample) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – an iterable of [`RagSample`](#ef.evaluation.RagSample)s (or plain mappings).
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  a `ragas.EvaluationDataset`.
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – if the optional `ragas` package is not installed.

### ef.evaluation.average_precision(ranked, relevant, k=None)

Average precision of `ranked` — the per-query term of MAP.

The mean of the precision measured at each relevant result’s rank. `k`
optionally caps the ranking.

```pycon
>>> average_precision(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0})
0.8333333333333333
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.benchmark_embedders(corpus_sources, embedders, qrels, queries, , distractors=None, hard_positives=None, k_values=(10, 20), segmenter=None, limit=None)

Re-index the same corpus under each embedder and tabulate the metrics.

For every `name -> spec` in `embedders` (a spec is anything
[`ef.as_embedder()`](ef.html.md#ef.as_embedder) accepts: an `Embedder`, `'st:bge-large-en-v1.5'`,
`'openai:text-embedding-3-large'` …) it runs [`ef.source_manager.ingest()`](ef.source_manager.html.md#ef.source_manager.ingest),
then [`evaluate_retrieval()`](#ef.evaluation.evaluate_retrieval), and stacks one row per embedder. Heavy
adapters are only imported when their spec is resolved. Deterministic
metrics only — no LLM.

```pycon
>>> from ef.embedders import HashingEmbedder
>>> corpus = {'d1': 'red apple fruit', 'd2': 'blue car engine'}
>>> report = benchmark_embedders(
...     corpus, {'hash': HashingEmbedder()},
...     {'q': {'d1': 1.0}}, {'q': 'apple'}, k_values=(1,))
>>> report.rows[0]['embedder'], report.rows[0]['recall@1']
('hash', 1.0)
```

* **Return type:**
  [`BenchmarkReport`](#ef.evaluation.BenchmarkReport)

### ef.evaluation.context_precision(reference, retrieved_contexts)

Fraction of the retrieved contexts that lexically overlap the `reference`.

A deterministic *proxy* for retrieval precision — a context counts as useful
if it shares at least one (normalized) token with the reference answer.

```pycon
>>> context_precision('paris', ['Paris is in France.', 'Unrelated text.'])
0.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.context_recall(reference, retrieved_contexts)

Lexical recall of the `reference` tokens by the retrieved contexts.

The fraction of the (normalized) reference’s distinct tokens that appear
anywhere in `retrieved_contexts` — a deterministic *proxy* for “did
retrieval surface the information the answer needs”. For the LLM-judged
context recall, use [`as_ragas_dataset()`](#ef.evaluation.as_ragas_dataset) and Ragas.

```pycon
>>> context_recall('paris france', ['Paris is the capital of France.'])
1.0
>>> context_recall('paris france', ['London is in England.'])
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.dcg_at_k(ranked, relevant, k)

Discounted cumulative gain of `ranked` at cutoff `k`.

`sum(grade_i / log2(i + 2))` over the top `k` results (`i` 0-indexed),
with `grade_i` the relevance of the `i`-th doc (`0` if unjudged) — the
linear-gain DCG that BEIR / `pytrec_eval` use.

```pycon
>>> round(dcg_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 3), 4)
1.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.evaluate_rag(samples, , metrics=('exact_match', 'token_f1', 'context_recall', 'context_precision', 'non_empty_rate'))

Score RAG samples with deterministic, reference-based lexical metrics.

Each sample is a [`RagSample`](#ef.evaluation.RagSample) — `(user_input, response,
retrieved_contexts, reference)`. `evaluate_rag` computes only metrics
that need **no LLM**: [`exact_match()`](#ef.evaluation.exact_match), [`token_f1()`](#ef.evaluation.token_f1),
[`context_recall()`](#ef.evaluation.context_recall), [`context_precision()`](#ef.evaluation.context_precision) and `non_empty_rate`
(the share of samples that retrieved any context). This keeps `ef` a
facade — it never synthesizes an answer and never calls a model (§6).

For the LLM-judged metrics — faithfulness, answer relevancy, … — pass the
same samples to [`as_ragas_dataset()`](#ef.evaluation.as_ragas_dataset) and run Ragas with your own LLM.

* **Parameters:**
  * **samples** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`RagSample`](#ef.evaluation.RagSample) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – an iterable of [`RagSample`](#ef.evaluation.RagSample)s (or plain mappings of the
    same shape).
  * **metrics** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – which metrics to compute — any of `exact_match` /
    `token_f1` / `context_recall` / `context_precision` /
    `non_empty_rate`. The reference-based ones skip samples that carry
    no `reference`.
* **Return type:**
  [`RagEvalReport`](#ef.evaluation.RagEvalReport)
* **Returns:**
  a [`RagEvalReport`](#ef.evaluation.RagEvalReport).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – for an unknown metric name, or if `samples` is empty.

```pycon
>>> samples = [
...     {'user_input': 'capital of France?', 'response': 'Paris',
...      'reference': 'Paris',
...      'retrieved_contexts': ['Paris is the capital of France.']},
...     {'user_input': '2 + 2?', 'response': 'five', 'reference': 'four',
...      'retrieved_contexts': []},
... ]
>>> report = evaluate_rag(samples)
>>> report.metrics['exact_match']
0.5
>>> report.metrics['non_empty_rate']
0.5
```

### ef.evaluation.evaluate_retrieval(retriever, qrels, queries, , k_values=(1, 5, 10, 100), metrics=('ndcg', 'recall', 'precision', 'mrr', 'map'), limit=None, distractors=None, hard_positives=None)

Score a retriever against a BEIR-shaped evaluation set.

For every query judged in `qrels`, the `retriever` is run, its results
are mapped back to *source document ids* (`_result_doc_id()`,
de-duplicated to the best rank per source), and each requested metric is
computed at each cutoff in `k_values`. The report holds the mean of each
`"<metric>@<k>"` over all scored queries — primary metric **NDCG@10**.

The natural pairing with [`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest):

```default
corpus, queries, qrels = read_beir("path/to/beir/dataset")
index = ingest(corpus, embedder="st:all-MiniLM-L6-v2")
report = evaluate_retrieval(index, qrels, queries)
report.primary            # mean NDCG@10
```

* **Parameters:**
  * **retriever** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – what answers a query — a [`SearchableCorpus`](ef.source_manager.html.md#ef.source_manager.SearchableCorpus),
    a [`SourceManager`](ef.source_manager.html.md#ef.source_manager.SourceManager), or any callable
    `query -> ranked results`. Results may be
    [`SearchHit`](ef.source_manager.html.md#ef.source_manager.SearchHit)s, [`Segment`](ef.segments.html.md#ef.segments.Segment)s
    or bare doc-id strings.
  * **qrels** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – the relevance judgements — `{query_id: {doc_id: relevance}}`.
    Queries with no positive judgement are skipped.
  * **queries** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the query texts — `{query_id: query_text}`. A judged query
    absent here cannot be run and is skipped.
  * **k_values** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – the rank cutoffs to evaluate at.
  * **metrics** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – which metrics to compute — any of `ndcg` / `recall` /
    `precision` / `mrr` / `map`.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – how many results to retrieve per query. `None` uses
    `max(k_values)`; raise it when sources segment into many pieces
    (de-duplication to source level can otherwise shorten the ranking).
  * **distractors** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional `{query_id: named distractor doc ids}`; adds
    `fp_rate@k` ([`fp_rate_at_k()`](#ef.evaluation.fp_rate_at_k)) to the metrics, averaged over the
    queries that name distractors, and fills `report.distractor_hits`.
  * **hard_positives** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional `{query_id: hard-positive doc ids}`; adds
    `hard_recall@k` ([`hard_positive_recall_at_k()`](#ef.evaluation.hard_positive_recall_at_k)), likewise.
* **Return type:**
  [`RetrievalEvalReport`](#ef.evaluation.RetrievalEvalReport)
* **Returns:**
  a [`RetrievalEvalReport`](#ef.evaluation.RetrievalEvalReport).
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – for an unknown metric name, or if no query can be scored.

```pycon
>>> qrels = {'q1': {'d1': 1.0}, 'q2': {'d2': 1.0}}
>>> queries = {'q1': 'alpha', 'q2': 'beta'}
>>> def retriever(query, *, limit=10):
...     return {'alpha': ['d1', 'd2'], 'beta': ['d1', 'd2']}[query][:limit]
>>> report = evaluate_retrieval(retriever, qrels, queries, k_values=(1, 2))
>>> report.metrics['recall@2'], report.n_queries
(1.0, 2)
```

### ef.evaluation.exact_match(response, reference)

`1.0` if `response` equals `reference` after normalization, else `0.0`.

```pycon
>>> exact_match('The Eiffel Tower', 'eiffel tower')
1.0
>>> exact_match('Paris', 'London')
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.fp_rate_at_k(ranked, distractors, k)

Fraction of the *named distractors* that intrude into the top `k`.

A distractor is a document known to be topically near the query but not
relevant; aggregate NDCG can improve while these keep over-ranking. 0.0 is
best.

```pycon
>>> fp_rate_at_k(['au', 'a', 'su', 'b'], ['au', 'su', 'csm'], 3)
0.6666666666666666
>>> fp_rate_at_k(['a', 'b'], ['au'], 2)
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.hard_positive_recall_at_k(ranked, hard_positives, k)

Fraction of the *hard positives* (e.g. thin-description docs) found in the top `k`.

```pycon
>>> hard_positive_recall_at_k(['a', 'imbed', 'b'], ['imbed', 'meshed'], 2)
0.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.ndcg_at_k(ranked, relevant, k)

Normalized DCG at cutoff `k` — `ef`’s **primary** retrieval metric.

[`dcg_at_k()`](#ef.evaluation.dcg_at_k) divided by the ideal DCG (the DCG of the best possible
ranking of the judged docs). `1.0` is a perfect ranking; `0.0` when no
relevant doc was judged.

```pycon
>>> ndcg_at_k(['d1', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)   # perfect order
1.0
>>> round(ndcg_at_k(['x', 'd1'], {'d1': 1.0}, 2), 4)     # relevant doc at rank 2
0.6309
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.precision_at_k(ranked, relevant, k)

Fraction of the top `k` results that are relevant.

```pycon
>>> precision_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)
0.5
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.read_beir(directory, , qrels=None)

Load a BEIR-format dataset from disk into the in-memory evaluation triple.

Reads `corpus.jsonl` (records `{"_id", "title", "text"}`),
`queries.jsonl` (records `{"_id", "text"}`) and a `qrels` TSV
(`query-id`, `corpus-id`, `score` columns). The corpus text is the
record’s `title` and `text` joined — the BEIR convention.

* **Parameters:**
  * **directory** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the dataset directory.
  * **qrels** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the qrels file, relative to `directory`. `None` tries
    `_DEFAULT_QRELS_PATHS` in order.
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]]
* **Returns:**
  a `(corpus, queries, qrels)` triple — `corpus` and `queries` are
  `{id: text}` dicts, `qrels` is `{query_id: {doc_id: relevance}}`.
  Feed it straight to [`evaluate_retrieval()`](#ef.evaluation.evaluate_retrieval) (after indexing `corpus`
  with [`ingest()`](ef.source_manager.html.md#ef.source_manager.ingest)).
* **Raises:**
  [**FileNotFoundError**](https://docs.python.org/3/builtins/exceptions.html#FileNotFoundError) – if `corpus.jsonl`, `queries.jsonl` or the qrels
      file is missing.

### ef.evaluation.recall_at_k(ranked, relevant, k)

Fraction of the relevant docs that appear in the top `k` results.

```pycon
>>> recall_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 2)
0.5
>>> recall_at_k(['d1', 'x', 'd2'], {'d1': 1.0, 'd2': 1.0}, 3)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.reciprocal_rank(ranked, relevant, k=None)

Reciprocal of the rank of the first relevant result (`0` if none).

Averaged over queries this is MRR. `k` optionally caps how far down the
ranking to look.

```pycon
>>> reciprocal_rank(['x', 'd1', 'd2'], {'d1': 1.0})
0.5
>>> reciprocal_rank(['x', 'd1'], {'d1': 1.0}, k=1)       # relevant doc past the cutoff
0.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.token_f1(response, reference)

Token-overlap F1 of `response` against `reference` (the SQuAD F1).

The harmonic mean of token precision and recall over the normalized token
bags — a graded credit where [`exact_match()`](#ef.evaluation.exact_match) is all-or-nothing.

```pycon
>>> token_f1('the quick brown fox', 'a quick brown fox')
1.0
>>> round(token_f1('quick brown fox', 'quick brown dog'), 4)
0.6667
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### ef.evaluation.write_beir(directory, corpus, queries, qrels)

Write an evaluation triple to disk in BEIR format — the inverse of [`read_beir()`](#ef.evaluation.read_beir).

Creates `directory` (and parents) and writes `corpus.jsonl`,
`queries.jsonl` and `qrels.tsv`. Use it to snapshot an `ef` corpus as
a reusable benchmark (`ef_use_cases.md` §H4).

* **Parameters:**
  * **directory** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – the output directory — created if absent.
  * **corpus** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the documents — `{doc_id: text}`.
  * **queries** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the queries — `{query_id: query_text}`.
  * **qrels** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – the relevance judgements — `{query_id: {doc_id: relevance}}`.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)
