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
- :func:`as_embedder` — the single dependency-injection seam: turns a string
  (model name / ``openai:`` prefix / URL), a bare callable, or an existing
  embedder into an :class:`~ef.embedders.Embedder`.

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
import time
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
# as_embedder — the dependency-injection seam
# ---------------------------------------------------------------------------


def as_embedder(x: Any, **kwargs: Any) -> Embedder:
    """Normalize ``x`` into an :class:`~ef.embedders.Embedder` — the DI seam.

    The single place every ``ef`` entry point coerces a user-supplied embedder
    argument. Accepts, in order:

    1. an existing :class:`~ef.embedders.Embedder` — returned unchanged;
    2. a URL string (``http://`` / ``https://``) — :func:`http_embedder`;
    3. an ``"openai:<model>"`` string — :func:`openai_embedder`;
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
