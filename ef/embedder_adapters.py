"""Concrete :class:`~ef.embedders.Embedder` adapters and the DI seam.

Adapters are **lazily-imported factory functions** — heavy dependencies
(``openai``, ``sentence-transformers``) are imported inside the factory, never
at module load, and live in optional extras. Importing this module costs only
numpy + the stdlib.

- :func:`openai_embedder` — OpenAI embeddings (sync + the async Batch API).
  Needs ``ef[openai]``.
- :func:`sentence_transformers_embedder` — local PyTorch models. Needs
  ``ef[sentence-transformers]``.
- :func:`http_embedder` — any remote HTTP embedding service (TEI, infinity,
  …). Stdlib-only.
- :func:`cohere_embedder`, :func:`voyage_embedder`, :func:`gemini_embedder` —
  the Cohere / Voyage / Gemini hosted embedding APIs, spoken over their REST
  endpoints directly. Stdlib-only (no SDK dependency); each needs only an API
  key. The whole point of these adapters is translating the canonical
  :data:`~ef.embedders.InputType` vocabulary to the vendor's own task name.
- :func:`as_embedder` — the single dependency-injection seam: turns a string
  (model name / ``openai:`` / ``cohere:`` / ``voyage:`` / ``gemini:`` prefix /
  URL), a bare callable, or an existing embedder into an
  :class:`~ef.embedders.Embedder`.

The **canonical task vocabulary** (:data:`~ef.embedders.InputType`) is
translated to each vendor's own name *at the adapter boundary* — that
translation is the whole point of having adapters.

Example — the DI seam in action (no network / no heavy deps needed):

>>> import numpy as np
>>> from ef.embedders import Embedder
>>> e = as_embedder(lambda ts: np.ones((len(ts), 5)), model_id='ones@5')
>>> isinstance(e, Embedder)
True
>>> e(['x', 'y']).shape
(2, 5)
>>> as_embedder(e) is e          # an Embedder passes straight through
True
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ef.embedders import (
    BaseEmbedder,
    BatchHandle,
    Embedder,
    EmbedderError,
    FunctionEmbedder,
    HashingEmbedder,
    InputType,
    embed_length_sorted,
)

__all__ = [
    "openai_embedder",
    "sentence_transformers_embedder",
    "http_embedder",
    "cohere_embedder",
    "voyage_embedder",
    "gemini_embedder",
    "as_embedder",
]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

#: Native output dimensionality of OpenAI embedding models.
_OPENAI_NATIVE_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
#: Models supporting Matryoshka (MRL) dimension truncation via ``dimensions=``.
_OPENAI_MRL_MODELS = {"text-embedding-3-small", "text-embedding-3-large"}
#: OpenAI's per-request input cap.
_OPENAI_MAX_BATCH = 2048


class _OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings adapter — see :func:`openai_embedder`."""

    honored_input_types: tuple[InputType, ...] = ()  # OpenAI ignores task hints
    normalized = True  # OpenAI L2-normalizes (incl. MRL-truncated outputs)

    def __init__(
        self,
        model: str,
        *,
        dim: int | None,
        client: Any,
        batch_size: int,
        poll_interval: float,
    ) -> None:
        native = _OPENAI_NATIVE_DIMS.get(model)
        if dim is not None and dim != native:
            if model not in _OPENAI_MRL_MODELS:
                raise EmbedderError(
                    f"Model {model!r} does not support a custom dimension; "
                    f"omit `dim` or use a text-embedding-3-* model"
                )
            self._dimensions_kwarg = True
        else:
            self._dimensions_kwarg = False
        self.model = model
        self.dim = dim or native
        self.model_id = f"openai:{model}@{self.dim}"
        self._client = client
        self.batch_size = min(batch_size, _OPENAI_MAX_BATCH)
        self.poll_interval = poll_interval

    def _request_body(self, chunk: Sequence[str]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "input": list(chunk)}
        if self._dimensions_kwarg:
            body["dimensions"] = self.dim
        return body

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        rows: list[Sequence[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(
                **self._request_body(chunk), **backend
            )
            for item in sorted(response.data, key=lambda d: d.index):
                rows.append(item.embedding)
        return np.asarray(rows, dtype=np.float32)

    def embed_batch(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> BatchHandle:
        """Submit an OpenAI Batch-API job (50% cheaper, 24h SLA).

        Returns a genuinely-pending :class:`BatchHandle`: ``poll()`` checks the
        job, ``result()`` blocks until completion then downloads and orders the
        vectors, ``cancel()`` cancels the job.
        """
        import io

        texts = list(texts)
        chunks = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        lines = [
            json.dumps(
                {
                    "custom_id": f"chunk-{ci}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": self._request_body(chunk),
                }
            )
            for ci, chunk in enumerate(chunks)
        ]
        upload = self._client.files.create(
            file=io.BytesIO("\n".join(lines).encode("utf-8")), purpose="batch"
        )
        job = self._client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        client, model_id, dim = self._client, self.model_id, self.dim or 0
        batch_size, poll_interval = self.batch_size, self.poll_interval
        n_texts = len(texts)

        def _status(state: str) -> str:
            if state == "completed":
                return "done"
            if state in ("failed", "expired", "cancelled", "cancelling"):
                return "failed"
            return "pending"

        def poll() -> str:
            return _status(client.batches.retrieve(job.id).status)

        def result() -> np.ndarray:
            while True:
                current = client.batches.retrieve(job.id)
                status = _status(current.status)
                if status == "done":
                    break
                if status == "failed":
                    raise EmbedderError(
                        f"OpenAI batch {job.id} ended as {current.status!r}"
                    )
                time.sleep(poll_interval)
            content = client.files.content(current.output_file_id).text
            out = np.empty((n_texts, dim), dtype=np.float32)
            for line in content.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_idx = int(record["custom_id"].split("-")[1])
                data = record["response"]["body"]["data"]
                base = chunk_idx * batch_size
                for item in data:
                    out[base + item["index"]] = item["embedding"]
            return out

        def cancel() -> None:
            client.batches.cancel(job.id)

        return BatchHandle(poll=poll, result=result, cancel=cancel)


def openai_embedder(
    model: str = "text-embedding-3-small",
    *,
    dim: int | None = None,
    api_key: str | None = None,
    client: Any | None = None,
    batch_size: int = _OPENAI_MAX_BATCH,
    poll_interval: float = 30.0,
) -> Embedder:
    """Build an OpenAI-embeddings :class:`~ef.embedders.Embedder`.

    Args:
        model: An OpenAI embedding model name.
        dim: Matryoshka-truncated output width (``text-embedding-3-*`` only).
            Omit for the model's native dimension. The dim is part of the
            embedder's identity — set it once here, not per call.
        api_key: OpenAI API key (else the SDK's usual env-var resolution).
        client: A pre-built ``openai.OpenAI`` client (overrides ``api_key``).
        batch_size: Texts per request (capped at OpenAI's 2048 limit).
        poll_interval: Seconds between polls when blocking on a Batch-API job.

    Returns:
        An embedder whose ``embed_batch`` uses the async OpenAI Batch API.

    Requires the ``openai`` package (``pip install 'ef[openai]'``).
    """
    if client is None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - import-guard
            raise ImportError(
                "openai_embedder needs the `openai` package. "
                "Install it with: pip install 'ef[openai]'"
            ) from exc
        client = openai.OpenAI(api_key=api_key)
    return _OpenAIEmbedder(
        model,
        dim=dim,
        client=client,
        batch_size=batch_size,
        poll_interval=poll_interval,
    )


# ---------------------------------------------------------------------------
# sentence-transformers (local)
# ---------------------------------------------------------------------------


class _SentenceTransformerEmbedder(BaseEmbedder):
    """Local sentence-transformers adapter — see :func:`sentence_transformers_embedder`."""

    honored_input_types: tuple[InputType, ...] = ()

    def __init__(
        self,
        model_name: str,
        *,
        model: Any,
        normalize: bool,
        batch_size: int,
    ) -> None:
        self._model = model
        self.dim = int(model.get_sentence_embedding_dimension())
        self.model_id = f"st:{model_name}@{self.dim}"
        self.normalized = normalize
        self.batch_size = batch_size

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        # SentenceTransformer.encode already sorts by length internally.
        embedded = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalized,
            convert_to_numpy=True,
            show_progress_bar=False,
            **backend,
        )
        return np.asarray(embedded, dtype=np.float32)


def sentence_transformers_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    *,
    device: str | None = None,
    normalize: bool = False,
    batch_size: int = 128,
    model_path: str | None = None,
    **st_kwargs: Any,
) -> Embedder:
    """Build a local sentence-transformers :class:`~ef.embedders.Embedder`.

    Args:
        model_name: A sentence-transformers model name (also used in
            ``model_id``).
        device: Torch device (``"cuda"`` / ``"cpu"`` / ``"mps"``); ``None``
            auto-selects.
        normalize: Whether to L2-normalize outputs. sentence-transformers does
            **not** normalize by default — set this truthfully.
        batch_size: Encoding batch size (smaller on CPU).
        model_path: A local filesystem path to load from instead of the hub —
            for air-gapped use. Loading still records ``model_name`` in
            ``model_id``.
        **st_kwargs: Forwarded to the ``SentenceTransformer`` constructor.

    Requires ``sentence-transformers`` (``pip install 'ef[sentence-transformers]'``).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - import-guard
        raise ImportError(
            "sentence_transformers_embedder needs the `sentence-transformers` "
            "package. Install it with: pip install 'ef[sentence-transformers]'"
        ) from exc
    model = SentenceTransformer(model_path or model_name, device=device, **st_kwargs)
    return _SentenceTransformerEmbedder(
        model_name, model=model, normalize=normalize, batch_size=batch_size
    )


# ---------------------------------------------------------------------------
# Remote HTTP service
# ---------------------------------------------------------------------------


def _tei_payload(texts: Sequence[str]) -> bytes:
    """Default request body: ``{"inputs": [...]}`` (TEI / infinity shape)."""
    return json.dumps({"inputs": list(texts)}).encode("utf-8")


def _tei_parse(response: Any) -> Any:
    """Default response parser: a bare JSON list of vectors."""
    return response


class _HttpEmbedder(BaseEmbedder):
    """Remote HTTP-service adapter — see :func:`http_embedder`."""

    honored_input_types: tuple[InputType, ...] = ()

    def __init__(
        self,
        url: str,
        *,
        model_id: str,
        dim: int | None,
        normalized: bool,
        batch_size: int,
        headers: dict[str, str],
        timeout: float,
        payload_builder: Callable[[Sequence[str]], bytes],
        response_parser: Callable[[Any], Any],
    ) -> None:
        self.url = url
        self.model_id = model_id
        self.dim = dim
        self.normalized = normalized
        self.batch_size = batch_size
        self._headers = headers
        self._timeout = timeout
        self._payload_builder = payload_builder
        self._response_parser = response_parser

    def _encode(self, batch: Sequence[str]) -> np.ndarray:
        request = urllib.request.Request(
            self.url,
            data=self._payload_builder(batch),
            headers={"Content-Type": "application/json", **self._headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        vectors = np.asarray(self._response_parser(decoded), dtype=np.float32)
        if vectors.ndim != 2:
            raise EmbedderError(
                f"{self.model_id}: expected a 2-D response, got shape "
                f"{vectors.shape} — supply a `response_parser`"
            )
        if self.dim is None:
            self.dim = int(vectors.shape[1])
        return vectors

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        return embed_length_sorted(texts, self._encode, batch_size=self.batch_size)


def http_embedder(
    url: str,
    *,
    model_id: str | None = None,
    dim: int | None = None,
    normalized: bool = False,
    batch_size: int = 64,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    payload_builder: Callable[[Sequence[str]], bytes] = _tei_payload,
    response_parser: Callable[[Any], Any] = _tei_parse,
) -> Embedder:
    """Build an :class:`~ef.embedders.Embedder` over a remote HTTP service.

    Defaults target the Text-Embeddings-Inference / infinity shape (a POST of
    ``{"inputs": [...]}`` returning a JSON list of vectors). For a different
    service, override ``payload_builder`` and/or ``response_parser``.

    Args:
        url: The embedding endpoint.
        model_id: Vector-identity string (defaults to ``"http:<url>"``).
        dim: Output width; inferred from the first response if omitted.
        normalized: Whether the service returns unit vectors.
        batch_size: Texts per request (used with length-sorted batching).
        headers: Extra HTTP headers (e.g. an auth token).
        timeout: Per-request timeout in seconds.
        payload_builder: ``texts -> request body bytes``.
        response_parser: ``decoded JSON -> array-like (n, dim)``.

    Uses only the stdlib — no extra dependency.
    """
    return _HttpEmbedder(
        url,
        model_id=model_id or f"http:{url}",
        dim=dim,
        normalized=normalized,
        batch_size=batch_size,
        headers=headers or {},
        timeout=timeout,
        payload_builder=payload_builder,
        response_parser=response_parser,
    )


# ---------------------------------------------------------------------------
# Hosted REST embedding APIs — Cohere, Voyage, Gemini
# ---------------------------------------------------------------------------
#
# Unlike the OpenAI adapter (which keeps its SDK for the genuinely-async Batch
# API), these three speak their providers' REST endpoints directly over
# ``urllib`` — no SDK, no extra dependency. The provider JSON contracts are
# public and versioned. What each adapter really earns its keep doing is
# translating ``ef``'s canonical InputType vocabulary to the vendor's own task
# name (Cohere ``input_type``, Voyage ``input_type``, Gemini ``taskType``).


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
) -> Any:
    """POST ``payload`` as JSON to ``url``; return the decoded JSON response.

    The single HTTP seam shared by the REST provider adapters
    (:func:`cohere_embedder`, :func:`voyage_embedder`, :func:`gemini_embedder`).
    Isolating it here keeps the adapters offline-testable — a test monkeypatches
    this one function instead of the network.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network error path
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise EmbedderError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _resolve_api_key(api_key: str | None, env_vars: Sequence[str], factory: str) -> str:
    """Return ``api_key`` if truthy, else the first set environment variable.

    Raises:
        EmbedderError: if no key is given and none of ``env_vars`` is set —
            immediate, actionable feedback instead of an opaque HTTP 401 later.
    """
    if api_key:
        return api_key
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            return value
    raise EmbedderError(
        f"{factory} needs an API key — pass `api_key=` or set "
        f"{' or '.join(env_vars)} in the environment."
    )


def _resolve_dim(
    model: str,
    dim: int | None,
    native_dims: dict[str, int],
    mrl_models: frozenset[str],
    provider: str,
) -> tuple[int | None, bool]:
    """Resolve an embedder's output width and whether to ask for MRL truncation.

    Returns ``(dim, truncated)``. ``dim`` is the explicit value if given, else
    the model's known native width, else ``None`` (an unknown model with no
    ``dim`` — the width is then inferred from the first response). ``truncated``
    is ``True`` only when a non-native width is requested from a model that
    supports Matryoshka (MRL) truncation.

    Raises:
        EmbedderError: if a non-native ``dim`` is asked of a model that cannot
            truncate.

    >>> _resolve_dim('m', 256, {'m': 1024}, frozenset({'m'}), 'X')
    (256, True)
    >>> _resolve_dim('m', None, {'m': 1024}, frozenset(), 'X')
    (1024, False)
    >>> _resolve_dim('unknown', 700, {}, frozenset(), 'X')
    (700, False)
    """
    native = native_dims.get(model)
    if dim is None:
        return native, False
    if native is None or dim == native:
        return dim, False
    if model in mrl_models:
        return dim, True
    raise EmbedderError(
        f"{provider} model {model!r} does not support a custom dimension "
        f"({dim} != native {native}); omit `dim` or choose an MRL-capable model."
    )


# Cohere ---------------------------------------------------------------------

#: Cohere's text-embeddings endpoint (API v2).
_COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"
#: Native output widths of the Cohere embedding models.
_COHERE_NATIVE_DIMS = {
    "embed-v4.0": 1536,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}
#: Cohere models that support Matryoshka (``output_dimension``) truncation.
_COHERE_MRL_MODELS = frozenset({"embed-v4.0"})
#: Cohere's per-request texts cap.
_COHERE_MAX_BATCH = 96
#: Canonical :data:`~ef.embedders.InputType` → Cohere's ``input_type`` name.
_COHERE_INPUT_TYPE: dict[InputType, str] = {
    "query": "search_query",
    "document": "search_document",
    "classification": "classification",
    "clustering": "clustering",
}


def _cohere_floats(response: Any) -> list[list[float]]:
    """Pull the float vectors out of a Cohere ``/v2/embed`` JSON response."""
    embeddings = response.get("embeddings") if isinstance(response, dict) else None
    if isinstance(embeddings, dict):
        floats = embeddings.get("float", embeddings.get("float_"))
        if floats is not None:
            return floats
    raise EmbedderError(
        "Cohere: no float embeddings in the response — expected "
        "{'embeddings': {'float': [...]}}"
    )


class _CohereEmbedder(BaseEmbedder):
    """Cohere embeddings adapter — see :func:`cohere_embedder`."""

    #: Cohere honors the task hint for every canonical InputType.
    honored_input_types: tuple[InputType, ...] = (
        "query",
        "document",
        "classification",
        "clustering",
    )
    normalized = False  # Cohere does not guarantee unit-norm float vectors

    def __init__(
        self,
        model: str,
        *,
        dim: int | None,
        truncated: bool,
        api_key: str,
        batch_size: int,
        timeout: float,
    ) -> None:
        self.model = model
        self.dim = dim
        self._truncated = truncated
        self.model_id = f"cohere:{model}@{dim}" if dim else f"cohere:{model}"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.batch_size = min(batch_size, _COHERE_MAX_BATCH)
        self._timeout = timeout

    def _encode(self, batch: Sequence[str], *, input_type: InputType | None) -> Any:
        # Cohere v3+ *requires* input_type; default to the document/index role.
        vendor_type = (
            _COHERE_INPUT_TYPE[input_type] if input_type else "search_document"
        )
        body: dict[str, Any] = {
            "model": self.model,
            "texts": list(batch),
            "input_type": vendor_type,
            "embedding_types": ["float"],
        }
        if self._truncated:
            body["output_dimension"] = self.dim
        response = _post_json(
            _COHERE_EMBED_URL, body, headers=self._headers, timeout=self._timeout
        )
        floats = _cohere_floats(response)
        if self.dim is None and floats:
            self.dim = len(floats[0])
        return floats

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)

        def encode(batch: Sequence[str]) -> Any:
            return self._encode(batch, input_type=input_type)

        return embed_length_sorted(texts, encode, batch_size=self.batch_size)


def cohere_embedder(
    model: str = "embed-v4.0",
    *,
    dim: int | None = None,
    api_key: str | None = None,
    batch_size: int = _COHERE_MAX_BATCH,
    timeout: float = 60.0,
) -> Embedder:
    """Build a Cohere-embeddings :class:`~ef.embedders.Embedder`.

    Talks to Cohere's ``/v2/embed`` REST endpoint directly — no ``cohere`` SDK,
    no extra dependency. Cohere's v3+ models *require* a task hint; this adapter
    supplies one for every canonical :data:`~ef.embedders.InputType` and
    defaults to the document role when none is given.

    Args:
        model: A Cohere embedding model name.
        dim: Matryoshka-truncated width (``embed-v4.0`` only). Omit for the
            model's native width. The dim is part of the embedder's identity —
            set it once here, not per call.
        api_key: Cohere API key. Falls back to ``CO_API_KEY`` / ``COHERE_API_KEY``.
        batch_size: Texts per request (capped at Cohere's 96 limit).
        timeout: Per-request timeout in seconds.

    Returns:
        An :class:`~ef.embedders.Embedder` over the Cohere API.
    """
    key = _resolve_api_key(api_key, ("CO_API_KEY", "COHERE_API_KEY"), "cohere_embedder")
    resolved_dim, truncated = _resolve_dim(
        model, dim, _COHERE_NATIVE_DIMS, _COHERE_MRL_MODELS, "Cohere"
    )
    return _CohereEmbedder(
        model,
        dim=resolved_dim,
        truncated=truncated,
        api_key=key,
        batch_size=batch_size,
        timeout=timeout,
    )


# Voyage ---------------------------------------------------------------------

#: Voyage AI's text-embeddings endpoint.
_VOYAGE_EMBED_URL = "https://api.voyageai.com/v1/embeddings"
#: Native output widths of the Voyage embedding models.
_VOYAGE_NATIVE_DIMS = {
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-large-2": 1536,
    "voyage-2": 1024,
}
#: Voyage models that support ``output_dimension`` (MRL) truncation.
_VOYAGE_MRL_MODELS = frozenset(
    {"voyage-3.5", "voyage-3.5-lite", "voyage-3-large", "voyage-code-3"}
)
#: Voyage's per-request texts cap.
_VOYAGE_MAX_BATCH = 1000
#: Canonical InputType → Voyage's ``input_type`` name. Voyage only distinguishes
#: query from document; classification / clustering map to no hint at all (the
#: request then omits ``input_type``).
_VOYAGE_INPUT_TYPE: dict[InputType, str] = {
    "query": "query",
    "document": "document",
}


class _VoyageEmbedder(BaseEmbedder):
    """Voyage AI embeddings adapter — see :func:`voyage_embedder`."""

    #: Voyage honors a query/document hint; it has no classification/clustering role.
    honored_input_types: tuple[InputType, ...] = ("query", "document")
    normalized = True  # Voyage embeddings are L2-normalized

    def __init__(
        self,
        model: str,
        *,
        dim: int | None,
        truncated: bool,
        api_key: str,
        batch_size: int,
        timeout: float,
    ) -> None:
        self.model = model
        self.dim = dim
        self._truncated = truncated
        self.model_id = f"voyage:{model}@{dim}" if dim else f"voyage:{model}"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.batch_size = min(batch_size, _VOYAGE_MAX_BATCH)
        self._timeout = timeout

    def _encode(self, batch: Sequence[str], *, input_type: InputType | None) -> Any:
        body: dict[str, Any] = {"model": self.model, "input": list(batch)}
        vendor_type = _VOYAGE_INPUT_TYPE.get(input_type) if input_type else None
        if vendor_type is not None:
            body["input_type"] = vendor_type
        if self._truncated:
            body["output_dimension"] = self.dim
        response = _post_json(
            _VOYAGE_EMBED_URL, body, headers=self._headers, timeout=self._timeout
        )
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, list):
            raise EmbedderError("Voyage: no 'data' array in the response")
        rows = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        if self.dim is None and rows:
            self.dim = len(rows[0])
        return rows

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)

        def encode(batch: Sequence[str]) -> Any:
            return self._encode(batch, input_type=input_type)

        return embed_length_sorted(texts, encode, batch_size=self.batch_size)


def voyage_embedder(
    model: str = "voyage-3.5",
    *,
    dim: int | None = None,
    api_key: str | None = None,
    batch_size: int = _VOYAGE_MAX_BATCH,
    timeout: float = 60.0,
) -> Embedder:
    """Build a Voyage AI :class:`~ef.embedders.Embedder`.

    Talks to Voyage's ``/v1/embeddings`` REST endpoint directly — no
    ``voyageai`` SDK, no extra dependency. Voyage embeddings are L2-normalized.
    Voyage distinguishes only ``query`` from ``document``; a ``classification``
    or ``clustering`` hint is simply not sent.

    Args:
        model: A Voyage embedding model name.
        dim: Matryoshka-truncated width (the ``voyage-3.5`` family and
            ``voyage-code-3`` only). Omit for the model's native width.
        api_key: Voyage API key. Falls back to ``VOYAGE_API_KEY``.
        batch_size: Texts per request (capped at Voyage's 1000 limit).
        timeout: Per-request timeout in seconds.

    Returns:
        An :class:`~ef.embedders.Embedder` over the Voyage API.
    """
    key = _resolve_api_key(api_key, ("VOYAGE_API_KEY",), "voyage_embedder")
    resolved_dim, truncated = _resolve_dim(
        model, dim, _VOYAGE_NATIVE_DIMS, _VOYAGE_MRL_MODELS, "Voyage"
    )
    return _VoyageEmbedder(
        model,
        dim=resolved_dim,
        truncated=truncated,
        api_key=key,
        batch_size=batch_size,
        timeout=timeout,
    )


# Gemini ---------------------------------------------------------------------

#: Base URL of the Gemini (Generative Language) embedding models.
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
#: Native output widths of the Gemini embedding models.
_GEMINI_NATIVE_DIMS = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,
    "embedding-001": 768,
}
#: Gemini models that support ``outputDimensionality`` (MRL) truncation.
_GEMINI_MRL_MODELS = frozenset({"gemini-embedding-001"})
#: Gemini's per-request batch cap for ``batchEmbedContents``.
_GEMINI_MAX_BATCH = 100
#: Canonical InputType → Gemini's ``taskType`` name.
_GEMINI_TASK_TYPE: dict[InputType, str] = {
    "query": "RETRIEVAL_QUERY",
    "document": "RETRIEVAL_DOCUMENT",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}


class _GeminiEmbedder(BaseEmbedder):
    """Google Gemini embeddings adapter — see :func:`gemini_embedder`."""

    #: Gemini honors the task hint for every canonical InputType.
    honored_input_types: tuple[InputType, ...] = (
        "query",
        "document",
        "classification",
        "clustering",
    )

    def __init__(
        self,
        model: str,
        *,
        dim: int | None,
        truncated: bool,
        api_key: str,
        batch_size: int,
        timeout: float,
    ) -> None:
        self.model = model
        self.dim = dim
        self._truncated = truncated
        # Gemini's full-width output is unit-norm; MRL-truncated output is not.
        self.normalized = not truncated
        self.model_id = f"gemini:{model}@{dim}" if dim else f"gemini:{model}"
        self._url = f"{_GEMINI_BASE_URL}/{model}:batchEmbedContents"
        self._model_path = f"models/{model}"
        self._headers = {"x-goog-api-key": api_key}
        self.batch_size = min(batch_size, _GEMINI_MAX_BATCH)
        self._timeout = timeout

    def _encode(self, batch: Sequence[str], *, input_type: InputType | None) -> Any:
        task_type = _GEMINI_TASK_TYPE.get(input_type) if input_type else None
        requests: list[dict[str, Any]] = []
        for text in batch:
            request: dict[str, Any] = {
                "model": self._model_path,
                "content": {"parts": [{"text": text}]},
            }
            if task_type is not None:
                request["taskType"] = task_type
            if self._truncated:
                request["outputDimensionality"] = self.dim
            requests.append(request)
        response = _post_json(
            self._url,
            {"requests": requests},
            headers=self._headers,
            timeout=self._timeout,
        )
        embeddings = response.get("embeddings") if isinstance(response, dict) else None
        if not isinstance(embeddings, list):
            raise EmbedderError("Gemini: no 'embeddings' array in the response")
        rows = [item["values"] for item in embeddings]
        if self.dim is None and rows:
            self.dim = len(rows[0])
        return rows

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim or 0), dtype=np.float32)

        def encode(batch: Sequence[str]) -> Any:
            return self._encode(batch, input_type=input_type)

        return embed_length_sorted(texts, encode, batch_size=self.batch_size)


def gemini_embedder(
    model: str = "gemini-embedding-001",
    *,
    dim: int | None = None,
    api_key: str | None = None,
    batch_size: int = _GEMINI_MAX_BATCH,
    timeout: float = 60.0,
) -> Embedder:
    """Build a Google Gemini :class:`~ef.embedders.Embedder`.

    Talks to the Generative Language ``batchEmbedContents`` REST endpoint
    directly — no ``google-genai`` SDK, no extra dependency.

    Args:
        model: A Gemini embedding model name (a leading ``models/`` is
            tolerated and stripped).
        dim: Matryoshka-truncated width (``gemini-embedding-001`` only). Omit
            for the model's native width. Note that Gemini, unlike OpenAI, does
            *not* re-normalize MRL-truncated output — a truncated embedder
            therefore reports ``normalized=False``.
        api_key: Gemini API key. Falls back to ``GEMINI_API_KEY`` /
            ``GOOGLE_API_KEY``.
        batch_size: Texts per request (capped at Gemini's 100 limit).
        timeout: Per-request timeout in seconds.

    Returns:
        An :class:`~ef.embedders.Embedder` over the Gemini API.
    """
    model = model.removeprefix("models/")
    key = _resolve_api_key(
        api_key, ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "gemini_embedder"
    )
    resolved_dim, truncated = _resolve_dim(
        model, dim, _GEMINI_NATIVE_DIMS, _GEMINI_MRL_MODELS, "Gemini"
    )
    return _GeminiEmbedder(
        model,
        dim=resolved_dim,
        truncated=truncated,
        api_key=key,
        batch_size=batch_size,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# as_embedder — the dependency-injection seam
# ---------------------------------------------------------------------------


def as_embedder(x: Any, **kwargs: Any) -> Embedder:
    """Normalize ``x`` into an :class:`~ef.embedders.Embedder` — the DI seam.

    The single place every ``ef`` entry point coerces a user-supplied embedder
    argument. Accepts, in order:

    1. an existing :class:`~ef.embedders.Embedder` — returned unchanged;
    2. a URL string (``http://`` / ``https://``) — :func:`http_embedder`;
    3. a provider-prefixed string — ``"openai:<model>"``, ``"cohere:<model>"``,
       ``"voyage:<model>"`` or ``"gemini:<model>"`` — the matching hosted-API
       adapter;
    4. the bare string ``"hashing"`` — the dependency-free
       :class:`~ef.embedders.HashingEmbedder` (``ef``'s zero-install default);
    5. an ``"st:<model>"`` string, or any other bare model name —
       :func:`sentence_transformers_embedder` (the local neural default);
    6. a bare callable — wrapped in
       :class:`~ef.embedders.FunctionEmbedder` (the ``imbed``-embedder bridge).

    Extra ``**kwargs`` are forwarded to the chosen factory.

    Raises:
        TypeError: if ``x`` is none of the above.

    >>> import numpy as np
    >>> e = as_embedder(lambda ts: np.zeros((len(ts), 2)), model_id='zero@2')
    >>> type(e).__name__
    'FunctionEmbedder'
    """
    if isinstance(x, Embedder):
        return x
    if isinstance(x, str):
        if x.startswith(("http://", "https://")):
            return http_embedder(x, **kwargs)
        if x.startswith("openai:"):
            return openai_embedder(x[len("openai:") :], **kwargs)
        if x.startswith("cohere:"):
            return cohere_embedder(x[len("cohere:") :], **kwargs)
        if x.startswith("voyage:"):
            return voyage_embedder(x[len("voyage:") :], **kwargs)
        if x.startswith("gemini:"):
            return gemini_embedder(x[len("gemini:") :], **kwargs)
        if x == "hashing":
            return HashingEmbedder(**kwargs)
        if x.startswith("st:"):
            return sentence_transformers_embedder(x[len("st:") :], **kwargs)
        return sentence_transformers_embedder(x, **kwargs)
    if callable(x):
        return FunctionEmbedder(x, **kwargs)
    raise TypeError(
        f"Cannot interpret {x!r} as an Embedder. Pass an Embedder, a model-name "
        f"string ('all-MiniLM-L6-v2', 'openai:text-embedding-3-small'), a URL, "
        f"or a callable mapping texts to vectors."
    )
