"""Composition wrappers for :class:`~ef.embedders.Embedder`.

Each wrapper *takes an inner embedder and is itself an embedder* — they compose
freely (``CachedEmbedder(RetryingEmbedder(NormalizingEmbedder(inner)), store)``).
This keeps the core protocol tiny: caching, retries, routing and normalization
are opt-in layers, not protocol obligations.

- :class:`CachedEmbedder` — memoize vectors in any ``MutableMapping`` store.
- :class:`RetryingEmbedder` — exponential backoff + jitter, retryable-aware.
- :class:`MultiEmbedder` — route texts to different embedders by a predicate.
- :class:`NormalizingEmbedder` — L2-normalize an inner embedder's output.

All four delegate their metadata (``model_id`` / ``dim`` / ``normalized`` /
``honored_input_types``) to the inner embedder, except where the wrapper changes
the vector (:class:`NormalizingEmbedder` reports ``normalized=True``).

The inner argument must already be an :class:`~ef.embedders.Embedder`; use
:func:`ef.embedder_adapters.as_embedder` to coerce a string / callable first.

Example — cache an embedder so repeats are free:

>>> import numpy as np
>>> from ef.embedders import FunctionEmbedder
>>> calls = []
>>> def toy(texts):
...     calls.extend(texts)
...     return np.ones((len(texts), 3))
>>> cached = CachedEmbedder(FunctionEmbedder(toy, model_id='toy@3'), store={})
>>> _ = cached(['a', 'b'])
>>> _ = cached(['b', 'c'])   # 'b' served from cache
>>> calls                    # 'b' embedded once
['a', 'b', 'c']
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, MutableMapping

import numpy as np

from ef.embedders import (
    BaseEmbedder,
    Embedder,
    EmbedderError,
    InputType,
    cache_key,
)

__all__ = [
    "CachedEmbedder",
    "RetryPolicy",
    "RetryingEmbedder",
    "MultiEmbedder",
    "NormalizingEmbedder",
]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return ``vectors`` scaled to unit L2 norm (zero vectors left untouched)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# CachedEmbedder
# ---------------------------------------------------------------------------


class CachedEmbedder(BaseEmbedder):
    """Memoize embeddings in a ``MutableMapping`` store.

    On each call, texts already in the store are served from it; only the misses
    are forwarded to the inner embedder (in a single batch) and then written
    back. The store can be any ``dol``-style mapping — an in-RAM ``dict``, a
    directory of ``.npy`` files, a key-value DB.

    Document embeddings are cached unconditionally; *query* embeddings are
    skipped by default (they are rarely repeated) — pass ``cache_queries=True``
    to cache them too.

    Args:
        inner: The embedder to memoize.
        store: A ``MutableMapping[str, ndarray]`` keyed by :func:`cache_key`.
        cache_queries: Whether to also cache ``input_type="query"`` calls.
    """

    def __init__(
        self,
        inner: Embedder,
        store: MutableMapping[str, np.ndarray],
        *,
        cache_queries: bool = False,
    ) -> None:
        self.inner = inner
        self.store = store
        self.cache_queries = cache_queries

    @property
    def model_id(self) -> str:  # noqa: D102 — delegated metadata
        return self.inner.model_id

    @property
    def dim(self) -> int | None:  # noqa: D102
        return self.inner.dim

    @property
    def normalized(self) -> bool:  # noqa: D102
        return self.inner.normalized

    @property
    def honored_input_types(self) -> tuple[InputType, ...]:  # noqa: D102
        return self.inner.honored_input_types

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        bypass = input_type == "query" and not self.cache_queries
        if bypass or not texts:
            return self.inner(texts, input_type=input_type, **backend)

        keys = [cache_key(self.inner, t, input_type) for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)
        miss_texts: list[str] = []
        miss_positions: list[int] = []
        for i, key in enumerate(keys):
            try:
                results[i] = np.asarray(self.store[key], dtype=np.float32)
            except KeyError:
                miss_texts.append(texts[i])
                miss_positions.append(i)

        if miss_texts:
            fresh = np.asarray(
                self.inner(miss_texts, input_type=input_type, **backend),
                dtype=np.float32,
            )
            for j, pos in enumerate(miss_positions):
                vector = fresh[j]
                results[pos] = vector
                self.store[keys[pos]] = vector

        return np.stack(results).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# RetryingEmbedder
# ---------------------------------------------------------------------------


def _default_is_retryable(exc: BaseException) -> bool:
    """Heuristic: retry transient failures (429, 5xx, connection/timeout).

    Per-request 400-class errors are *not* retried — the same request would fail
    identically. Recognizes the ``status_code``/``status`` attribute carried by
    most HTTP-client SDK exceptions (incl. the OpenAI SDK).
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return "connection" in name or "timeout" in name


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry policy for :class:`RetryingEmbedder`.

    Attributes:
        max_attempts: Total tries (the first call counts as attempt 1).
        base_delay: Seconds before the first retry; doubles each attempt.
        max_delay: Upper bound on any single backoff sleep.
        jitter: Random fraction (``0..1``) of the delay added on top, to
            de-correlate concurrent clients.
        retry_on: Predicate deciding whether an exception is worth retrying.
    """

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.3
    retry_on: Callable[[BaseException], bool] = field(default=_default_is_retryable)


class RetryingEmbedder(BaseEmbedder):
    """Retry an inner embedder's calls on transient failures.

    Args:
        inner: The embedder to guard.
        policy: A :class:`RetryPolicy` (sensible defaults if omitted).
        sleep: The sleep function — injectable so tests run instantly.
    """

    def __init__(
        self,
        inner: Embedder,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.policy = policy or RetryPolicy()
        self._sleep = sleep

    @property
    def model_id(self) -> str:  # noqa: D102 — delegated metadata
        return self.inner.model_id

    @property
    def dim(self) -> int | None:  # noqa: D102
        return self.inner.dim

    @property
    def normalized(self) -> bool:  # noqa: D102
        return self.inner.normalized

    @property
    def honored_input_types(self) -> tuple[InputType, ...]:  # noqa: D102
        return self.inner.honored_input_types

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        attempt = 0
        while True:
            try:
                return self.inner(texts, input_type=input_type, **backend)
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                attempt += 1
                exhausted = attempt >= self.policy.max_attempts
                if exhausted or not self.policy.retry_on(exc):
                    raise
                delay = min(
                    self.policy.max_delay,
                    self.policy.base_delay * (2 ** (attempt - 1)),
                )
                delay += random.uniform(0.0, self.policy.jitter * delay)
                self._sleep(delay)


# ---------------------------------------------------------------------------
# MultiEmbedder
# ---------------------------------------------------------------------------


class MultiEmbedder(BaseEmbedder):
    """Route each text to one of several embedders by a predicate.

    Useful for domain-specific models (e.g. a code embedder for code, a prose
    embedder for prose). All routes must share the same ``dim`` — the output is
    a single ``(n, dim)`` array, with each row produced by its text's route.

    Args:
        routes: ``{route_key: Embedder}``.
        predicate: ``text -> route_key`` — picks a route per text.
        default: Embedder for texts whose ``route_key`` is not in ``routes``
            (otherwise an unknown key raises :class:`EmbedderError`).

    >>> import numpy as np
    >>> from ef.embedders import FunctionEmbedder
    >>> short = FunctionEmbedder(lambda ts: np.zeros((len(ts), 2)), model_id='s@2')
    >>> long = FunctionEmbedder(lambda ts: np.ones((len(ts), 2)), model_id='l@2')
    >>> m = MultiEmbedder({'s': short, 'l': long},
    ...                   predicate=lambda t: 's' if len(t) < 4 else 'l')
    >>> m(['hi', 'hello']).tolist()
    [[0.0, 0.0], [1.0, 1.0]]
    """

    def __init__(
        self,
        routes: Mapping[Hashable, Embedder],
        predicate: Callable[[str], Hashable],
        *,
        default: Embedder | None = None,
    ) -> None:
        if not routes:
            raise EmbedderError("MultiEmbedder needs at least one route")
        self.routes = dict(routes)
        self.predicate = predicate
        self.default = default
        embedders = list(self.routes.values())
        if default is not None:
            embedders.append(default)
        dims = {e.dim for e in embedders if e.dim is not None}
        if len(dims) > 1:
            raise EmbedderError(
                f"MultiEmbedder routes have differing dims: {sorted(dims)}"
            )
        self.dim = dims.pop() if len(dims) == 1 else None
        self.normalized = all(e.normalized for e in embedders)
        honored = [set(e.honored_input_types) for e in embedders]
        self.honored_input_types = tuple(set.intersection(*honored)) if honored else ()
        self.model_id = "multi(" + "|".join(sorted(e.model_id for e in embedders)) + ")"

    def _route_for(self, route_key: Hashable) -> Embedder:
        embedder = self.routes.get(route_key, self.default)
        if embedder is None:
            raise EmbedderError(
                f"No route for key {route_key!r} and no default embedder set"
            )
        return embedder

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        groups: dict[Hashable, tuple[list[int], list[str]]] = {}
        for i, text in enumerate(texts):
            route_key = self.predicate(text)
            positions, grp_texts = groups.setdefault(route_key, ([], []))
            positions.append(i)
            grp_texts.append(text)

        out: np.ndarray | None = None
        for route_key, (positions, grp_texts) in groups.items():
            embedded = np.asarray(
                self._route_for(route_key)(grp_texts, input_type=input_type, **backend),
                dtype=np.float32,
            )
            if out is None:
                out = np.empty((len(texts), embedded.shape[1]), dtype=np.float32)
            for j, pos in enumerate(positions):
                out[pos] = embedded[j]

        if out is None:
            return np.empty((0, self.dim or 0), dtype=np.float32)
        return out


# ---------------------------------------------------------------------------
# NormalizingEmbedder
# ---------------------------------------------------------------------------


class NormalizingEmbedder(BaseEmbedder):
    """L2-normalize an inner embedder's output.

    Use this for MRL-truncated vectors (which lose unit norm when sliced to a
    smaller dimension) or any backend that does not normalize by construction.
    Reports ``normalized=True``; :func:`cache_key` keys on ``normalized``, so a
    normalized and an un-normalized view of the same model never collide in a
    shared cache.

    Args:
        inner: The embedder whose output to renormalize.
    """

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner

    @property
    def model_id(self) -> str:  # noqa: D102 — delegated metadata
        return self.inner.model_id

    @property
    def dim(self) -> int | None:  # noqa: D102
        return self.inner.dim

    normalized = True

    @property
    def honored_input_types(self) -> tuple[InputType, ...]:  # noqa: D102
        return self.inner.honored_input_types

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        vectors = np.asarray(
            self.inner(texts, input_type=input_type, **backend), dtype=np.float32
        )
        if vectors.size == 0:
            return vectors
        return _l2_normalize(vectors)
