# ef.embedder_adapters

Concrete [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) adapters and the DI seam.

Adapters are **lazily-imported factory functions** — heavy dependencies
(`openai`, `sentence-transformers`) are imported inside the factory, never
at module load, and live in optional extras. Importing this module costs only
numpy + the stdlib.

- [`openai_embedder()`](#ef.embedder_adapters.openai_embedder) — OpenAI embeddings (sync + the async Batch API).
  Needs `ef[openai]`.
- [`sentence_transformers_embedder()`](#ef.embedder_adapters.sentence_transformers_embedder) — local PyTorch models. Needs
  `ef[sentence-transformers]`.
- [`http_embedder()`](#ef.embedder_adapters.http_embedder) — any remote HTTP embedding service (TEI, infinity,
  …). Stdlib-only.
- [`cohere_embedder()`](#ef.embedder_adapters.cohere_embedder), [`voyage_embedder()`](#ef.embedder_adapters.voyage_embedder), [`gemini_embedder()`](#ef.embedder_adapters.gemini_embedder) —
  the Cohere / Voyage / Gemini hosted embedding APIs, spoken over their REST
  endpoints directly. Stdlib-only (no SDK dependency); each needs only an API
  key. The whole point of these adapters is translating the canonical
  [`InputType`](ef.embedders.html.md#ef.embedders.InputType) vocabulary to the vendor’s own task name.
- [`as_embedder()`](#ef.embedder_adapters.as_embedder) — the single dependency-injection seam: turns a string
  (model name / `openai:` / `cohere:` / `voyage:` / `gemini:` prefix /
  URL), a bare callable, or an existing embedder into an
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

The **canonical task vocabulary** ([`InputType`](ef.embedders.html.md#ef.embedders.InputType)) is
translated to each vendor’s own name *at the adapter boundary* — that
translation is the whole point of having adapters.

Example — the DI seam in action (no network / no heavy deps needed):

```pycon
>>> import numpy as np
>>> from ef.embedders import Embedder
>>> e = as_embedder(lambda ts: np.ones((len(ts), 5)), model_id='ones@5')
>>> isinstance(e, Embedder)
True
>>> e(['x', 'y']).shape
(2, 5)
>>> as_embedder(e) is e          # an Embedder passes straight through
True
```

### Functions

| [`openai_embedder`](#ef.embedder_adapters.openai_embedder)([model, dim, api_key, ...])       | Build an OpenAI-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).          |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [`sentence_transformers_embedder`](#ef.embedder_adapters.sentence_transformers_embedder)([model_name, ...]) | Build a local sentence-transformers [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder). |
| [`http_embedder`](#ef.embedder_adapters.http_embedder)(url, \*[, model_id, dim, ...])      | Build an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over a remote HTTP service. |
| [`cohere_embedder`](#ef.embedder_adapters.cohere_embedder)([model, dim, api_key, ...])       | Build a Cohere-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).           |
| [`voyage_embedder`](#ef.embedder_adapters.voyage_embedder)([model, dim, api_key, ...])       | Build a Voyage AI [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).                   |
| [`gemini_embedder`](#ef.embedder_adapters.gemini_embedder)([model, dim, api_key, ...])       | Build a Google Gemini [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).               |
| [`as_embedder`](#ef.embedder_adapters.as_embedder)(x, \*\*kwargs)                        | Normalize `x` into an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) — the DI seam. |

### ef.embedder_adapters.as_embedder(x, \*\*kwargs)

Normalize `x` into an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) — the DI seam.

The single place every `ef` entry point coerces a user-supplied embedder
argument. Accepts, in order:

1. an existing [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) — returned unchanged;
2. a URL string (`http://` / `https://`) — [`http_embedder()`](#ef.embedder_adapters.http_embedder);
3. a provider-prefixed string — `"openai:<model>"`, `"cohere:<model>"`,
   `"voyage:<model>"` or `"gemini:<model>"` — the matching hosted-API
   adapter;
4. the bare string `"hashing"` — the dependency-free
   [`HashingEmbedder`](ef.embedders.html.md#ef.embedders.HashingEmbedder) (`ef`’s zero-install default);
5. an `"st:<model>"` string, or any other bare model name —
   [`sentence_transformers_embedder()`](#ef.embedder_adapters.sentence_transformers_embedder) (the local neural default);
6. a bare callable — wrapped in
   [`FunctionEmbedder`](ef.embedders.html.md#ef.embedders.FunctionEmbedder) (the `imbed`-embedder bridge).

Extra `**kwargs` are forwarded to the chosen factory.

* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – if `x` is none of the above.
* **Return type:**
  [*Embedder*](ef.embedders.html.md#ef.embedders.Embedder)

```pycon
>>> import numpy as np
>>> e = as_embedder(lambda ts: np.zeros((len(ts), 2)), model_id='zero@2')
>>> type(e).__name__
'FunctionEmbedder'
```

* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)

### ef.embedder_adapters.cohere_embedder(model='embed-v4.0', , dim=None, api_key=None, batch_size=96, timeout=60.0)

Build a Cohere-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Talks to Cohere’s `/v2/embed` REST endpoint directly — no `cohere` SDK,
no extra dependency. Cohere’s v3+ models *require* a task hint; this adapter
supplies one for every canonical [`InputType`](ef.embedders.html.md#ef.embedders.InputType) and
defaults to the document role when none is given.

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A Cohere embedding model name.
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated width (`embed-v4.0` only). Omit for the
    model’s native width. The dim is part of the embedder’s identity —
    set it once here, not per call.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Cohere API key. Falls back to `CO_API_KEY` / `COHERE_API_KEY`.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at Cohere’s 96 limit).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over the Cohere API.

### ef.embedder_adapters.gemini_embedder(model='gemini-embedding-001', , dim=None, api_key=None, batch_size=100, timeout=60.0)

Build a Google Gemini [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Talks to the Generative Language `batchEmbedContents` REST endpoint
directly — no `google-genai` SDK, no extra dependency.

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A Gemini embedding model name (a leading `models/` is
    tolerated and stripped).
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated width (`gemini-embedding-001` only). Omit
    for the model’s native width. Note that Gemini, unlike OpenAI, does
    *not* re-normalize MRL-truncated output — a truncated embedder
    therefore reports `normalized=False`.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Gemini API key. Falls back to `GEMINI_API_KEY` /
    `GOOGLE_API_KEY`.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at Gemini’s 100 limit).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over the Gemini API.

### ef.embedder_adapters.http_embedder(url, \*, model_id=None, dim=None, normalized=False, batch_size=64, headers=None, timeout=30.0, payload_builder=<function \_tei_payload>, response_parser=<function \_tei_parse>)

Build an [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over a remote HTTP service.

Defaults target the Text-Embeddings-Inference / infinity shape (a POST of
`{"inputs": [...]}` returning a JSON list of vectors). For a different
service, override `payload_builder` and/or `response_parser`.

* **Parameters:**
  * **url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The embedding endpoint.
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Vector-identity string (defaults to `"http:<url>"`).
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Output width; inferred from the first response if omitted.
  * **normalized** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether the service returns unit vectors.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (used with length-sorted batching).
  * **headers** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Extra HTTP headers (e.g. an auth token).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
  * **payload_builder** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]], [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – `texts -> request body bytes`.
  * **response_parser** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – `decoded JSON -> array-like (n, dim)`.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)

Uses only the stdlib — no extra dependency.

### ef.embedder_adapters.openai_embedder(model='text-embedding-3-small', , dim=None, api_key=None, client=None, batch_size=2048, poll_interval=30.0)

Build an OpenAI-embeddings [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – An OpenAI embedding model name.
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated output width (`text-embedding-3-*` only).
    Omit for the model’s native dimension. The dim is part of the
    embedder’s identity — set it once here, not per call.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – OpenAI API key (else the SDK’s usual env-var resolution).
  * **client** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A pre-built `openai.OpenAI` client (overrides `api_key`).
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at OpenAI’s 2048 limit).
  * **poll_interval** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Seconds between polls when blocking on a Batch-API job.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An embedder whose `embed_batch` uses the async OpenAI Batch API.

Requires the `openai` package (`pip install 'ef[openai]'`).

### ef.embedder_adapters.sentence_transformers_embedder(model_name='all-MiniLM-L6-v2', , device=None, normalize=False, batch_size=128, model_path=None, query_prompt=None, document_prompt=None, prompt_names=None, \*\*st_kwargs)

Build a local sentence-transformers [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Instruction-tuned models (bge, gte, e5-instruct, Qwen3-Embedding…) are
conditioned per `input_type`: pass a literal `query_prompt` (and, rarely,
`document_prompt`), or map input types to prompts the model defines itself
with `prompt_names` (e.g. `{"query": "query"}` for Qwen3-Embedding). Those
roles are then advertised in `honored_input_types`, and the prompt text is
baked into `model_id`. Only trust prompts on genuinely instruction-tuned
models; on `all-MiniLM` they are expected to be a near no-op. Prefer these
over wrapping the adapter in [`InstructionEmbedder`](ef.embedder_wrappers.html.md#ef.embedder_wrappers.InstructionEmbedder):
sentence-transformers knows the prompt’s length, so models that pool without
the prompt tokens (INSTRUCTOR, GRIT) stay correct. Needs
sentence-transformers >= 2.3.

* **Parameters:**
  * **model_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A sentence-transformers model name (also used in
    `model_id`).
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Torch device (`"cuda"` / `"cpu"` / `"mps"`); `None`
    auto-selects.
  * **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to L2-normalize outputs. sentence-transformers does
    **not** normalize by default — set this truthfully.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Encoding batch size (smaller on CPU).
  * **model_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A local filesystem path to load from instead of the hub —
    for air-gapped use. Loading still records `model_name` in
    `model_id`.
  * **query_prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prompt prepended to `input_type="query"` texts (passed
    as `encode(prompt=...)`).
  * **document_prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Prompt for `input_type="document"` texts (usually
    none: the asymmetric pattern leaves documents bare). `""` means an
    explicit empty prompt, which stops a model’s `default_prompt_name`
    from being applied to documents.
  * **prompt_names** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `{input_type: name}` of prompts defined in the model’s
    own `prompts` config (passed as `encode(prompt_name=...)`).
    An explicit `prompt=` / `prompt_name=` in a call overrides both.
  * **\*\*st_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Forwarded to the `SentenceTransformer` constructor.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)

Requires `sentence-transformers` (`pip install 'ef[sentence-transformers]'`).

### ef.embedder_adapters.voyage_embedder(model='voyage-3.5', , dim=None, api_key=None, batch_size=1000, timeout=60.0)

Build a Voyage AI [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder).

Talks to Voyage’s `/v1/embeddings` REST endpoint directly — no
`voyageai` SDK, no extra dependency. Voyage embeddings are L2-normalized.
Voyage distinguishes only `query` from `document`; a `classification`
or `clustering` hint is simply not sent.

* **Parameters:**
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A Voyage embedding model name.
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Matryoshka-truncated width (the `voyage-3.5` family and
    `voyage-code-3` only). Omit for the model’s native width.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voyage API key. Falls back to `VOYAGE_API_KEY`.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Texts per request (capped at Voyage’s 1000 limit).
  * **timeout** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Per-request timeout in seconds.
* **Return type:**
  [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder)
* **Returns:**
  An [`Embedder`](ef.embedders.html.md#ef.embedders.Embedder) over the Voyage API.
