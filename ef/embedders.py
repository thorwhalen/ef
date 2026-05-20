"""The ``Embedder`` facade — ``ef``'s embed layer (L3).

An *embedder* is, fundamentally, a batch callable::

    Iterable[str] -> ndarray of shape (n, dim)

Everything else — batching, async/batch-APIs, normalization, task hints,
caching, retries — is *layered scaffolding* over that one contract.

This module defines the small, structural core of the facade:

- :class:`Embedder` — the ``@runtime_checkable`` protocol (a callable plus four
  introspectable metadata attributes). It is a protocol, **not** an ABC: any
  object with the right shape *is* an embedder.
- :class:`BatchHandle` / :func:`ready_handle` — a uniform poll/result shape that
  collapses synchronous embedding and provider Batch APIs into one thing.
- :class:`BaseEmbedder` — an implementation convenience base (supplies a default
  synchronous :meth:`~BaseEmbedder.embed_batch` and a ``repr``). Subclassing it
  is optional; satisfying :class:`Embedder` structurally is what matters.
- :class:`FunctionEmbedder` — wraps a bare ``texts -> vectors`` callable into a
  full :class:`Embedder` (the bridge for e.g. ``imbed``'s embedder functions).
- :func:`cache_key` — a deterministic key pinning everything that changes a
  vector, for :class:`~ef.embedder_wrappers.CachedEmbedder`.
- :func:`embed_length_sorted` — length-sorted batching with order
  reconstruction; a 5–10× throughput win that ``ef`` owns so users never have
  to think about padding or sort order.

Composition wrappers live in :mod:`ef.embedder_wrappers`; concrete provider /
local adapters and the :func:`~ef.embedder_adapters.as_embedder` dependency
-injection seam live in :mod:`ef.embedder_adapters`.

Example — wrap a plain function and embed two strings:

>>> import numpy as np
>>> def toy(texts):  # a bare callable: list[str] -> array(n, 3)
...     return np.array([[len(t), t.count(' '), 1.0] for t in texts])
>>> embedder = FunctionEmbedder(toy, model_id='toy@3')
>>> vectors = embedder(['hello', 'a b c'])
>>> vectors.shape
(2, 3)
>>> embedder.dim  # inferred on first call
3
>>> isinstance(embedder, Embedder)
True
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

__all__ = [
    "InputType",
    "BatchStatus",
    "BatchHandle",
    "ready_handle",
    "Embedder",
    "BaseEmbedder",
    "FunctionEmbedder",
    "EmbedderError",
    "cache_key",
    "embed_length_sorted",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Canonical embedding-task vocabulary. Every provider names the same
#: distinction differently (``input_type`` / ``task_type`` / ``task`` /
#: ``prompt_name``); ``ef`` defines *one* canonical set and each adapter
#: translates it at its boundary. This is the facade's single most valuable
#: normalization. Backends that ignore the hint advertise an empty
#: ``honored_input_types``.
InputType = Literal["query", "document", "classification", "clustering"]

#: The lifecycle states of a :class:`BatchHandle`.
BatchStatus = Literal["pending", "done", "failed"]


class EmbedderError(Exception):
    """Raised for embedder-specific failures (shape mismatch, batch job failure)."""


# ---------------------------------------------------------------------------
# BatchHandle — sync and provider Batch APIs, one shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchHandle:
    """A poll/result handle over an in-flight (or already-finished) embed job.

    Batch-and-poll is just a special case of async. A synchronous backend
    returns a *ready* handle (via :func:`ready_handle`); a provider Batch API
    returns a genuinely pending one. Either way the caller sees one shape.

    Attributes:
        poll: ``() -> BatchStatus`` — non-blocking status check.
        result: ``() -> np.ndarray`` — blocks until done, returns ``(n, dim)``.
        cancel: ``() -> None`` — best-effort cancellation.
    """

    poll: Callable[[], BatchStatus]
    result: Callable[[], np.ndarray]
    cancel: Callable[[], None] = field(default=lambda: None)


def ready_handle(array: Any) -> BatchHandle:
    """Wrap an already-computed array in a finished :class:`BatchHandle`.

    >>> import numpy as np
    >>> h = ready_handle(np.zeros((2, 3)))
    >>> h.poll()
    'done'
    >>> h.result().shape
    (2, 3)
    """
    arr = np.asarray(array, dtype=np.float32)
    return BatchHandle(poll=lambda: "done", result=lambda: arr)


# ---------------------------------------------------------------------------
# The Embedder protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Structural type for an embedder: a batch callable plus metadata.

    An embedder maps ``Iterable[str]`` to an ``ndarray`` of shape ``(n, dim)``.
    The four metadata attributes are the single source of truth *on the object
    itself* — no global registry, no config singleton.

    Attributes:
        model_id: Identity that bakes in everything affecting the vector —
            ``"openai:text-embedding-3-large@1024"`` (``provider:model@dim``).
            Works directly as a cache namespace.
        dim: Output dimensionality. May be ``None`` until first call for
            lazily-probed embedders (e.g. a bare callable of unknown width).
        normalized: ``True`` iff ``||v|| == 1`` by construction.
        honored_input_types: The :data:`InputType` values this embedder acts on
            (empty if it ignores the task hint).
    """

    model_id: str
    dim: int | None
    normalized: bool
    honored_input_types: tuple[InputType, ...]

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        """Embed ``texts`` synchronously, returning ``(n, dim)`` float32."""
        ...

    def embed_batch(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> BatchHandle:
        """Submit an embed job, returning a :class:`BatchHandle`."""
        ...


class BaseEmbedder:
    """Implementation base for embedders (optional — :class:`Embedder` is structural).

    Subclasses set ``model_id`` / ``dim`` / ``normalized`` /
    ``honored_input_types`` and implement ``__call__``; they then satisfy
    :class:`Embedder` structurally. In return they get:

    - a default *synchronous* :meth:`embed_batch` (``ready_handle(self(...))``) —
      override it only for a genuinely async provider Batch API;
    - a readable ``repr``.
    """

    model_id: str
    dim: int | None = None
    normalized: bool = False
    honored_input_types: tuple[InputType, ...] = ()

    def embed_batch(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> BatchHandle:
        """Default: embed synchronously and return a finished handle."""
        return ready_handle(self(texts, input_type=input_type, **backend))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"model_id={getattr(self, 'model_id', None)!r}, "
            f"dim={getattr(self, 'dim', None)})"
        )


# ---------------------------------------------------------------------------
# FunctionEmbedder — wrap a bare callable
# ---------------------------------------------------------------------------


class FunctionEmbedder(BaseEmbedder):
    """Promote a bare ``texts -> vectors`` callable to a full :class:`Embedder`.

    This is the bridge for embedder *functions* that carry no metadata — most
    importantly ``imbed``'s registered embedders. The wrapped callable is asked
    only for ``func(texts)`` (it knows nothing of ``input_type``); the wrapper
    advertises ``honored_input_types=()`` accordingly.

    Args:
        func: A callable mapping a list of strings to an array-like of shape
            ``(n, dim)`` (a numpy array or a nested sequence — both accepted).
        model_id: Vector-identity string. Defaults to ``"function:<name>"``.
        dim: Output width. If ``None`` it is inferred (and frozen) on the first
            call.
        normalized: Whether ``func`` already returns unit vectors.
        honored_input_types: Task hints ``func`` honors (usually none).
        pass_backend: If ``True`` (default), forward any ``**backend`` kwargs to
            ``func``; set ``False`` for callables with a strict ``(texts,)``
            signature.

    >>> import numpy as np
    >>> e = FunctionEmbedder(lambda ts: np.ones((len(ts), 4)), model_id='ones@4')
    >>> e(['a', 'b', 'c']).shape
    (3, 4)
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        model_id: str | None = None,
        dim: int | None = None,
        normalized: bool = False,
        honored_input_types: Iterable[InputType] = (),
        pass_backend: bool = True,
    ) -> None:
        if not callable(func):
            raise TypeError(f"FunctionEmbedder needs a callable, got {type(func)}")
        self._func = func
        self.model_id = model_id or f"function:{getattr(func, '__name__', repr(func))}"
        self.dim = dim
        self.normalized = normalized
        self.honored_input_types = tuple(honored_input_types)
        self._pass_backend = pass_backend

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
        raw = (
            self._func(texts, **backend)
            if (self._pass_backend and backend)
            else self._func(texts)
        )
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2:
            raise EmbedderError(
                f"{self.model_id}: expected a 2-D (n, dim) output, "
                f"got an array of shape {arr.shape}"
            )
        if arr.shape[0] != len(texts):
            raise EmbedderError(
                f"{self.model_id}: got {arr.shape[0]} vectors for "
                f"{len(texts)} input texts"
            )
        if self.dim is None:
            self.dim = int(arr.shape[1])
        elif arr.shape[1] != self.dim:
            raise EmbedderError(
                f"{self.model_id}: expected dim {self.dim}, got {arr.shape[1]}"
            )
        return arr


# ---------------------------------------------------------------------------
# Caching primitive
# ---------------------------------------------------------------------------


def cache_key(
    embedder: Embedder, text: str, input_type: InputType | None = None
) -> str:
    """Deterministic cache key for one ``(embedder, text, input_type)`` triple.

    The key pins everything that changes the resulting vector. ``model_id`` is
    the embedder-identity SSOT — it already bakes in provider, model and ``dim``
    — so ``dim`` is *not* keyed separately (keying it would also break
    :class:`FunctionEmbedder`'s lazy ``dim`` inference). ``normalized`` is keyed
    because a :class:`~ef.embedder_wrappers.NormalizingEmbedder` shares its
    inner's ``model_id`` yet yields different vectors.

    Returns ``"<model_id>/<sha256>"`` — a namespaced key safe for any
    ``MutableMapping`` store.

    >>> class _E:
    ...     model_id, normalized = 'm@8', False
    >>> k1 = cache_key(_E(), 'hello', 'document')
    >>> k2 = cache_key(_E(), 'hello', 'query')
    >>> k1 == k2  # input_type participates
    False
    >>> k1.startswith('m@8/')
    True
    """
    payload = f"{embedder.model_id}|{embedder.normalized}|{input_type}|{text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{embedder.model_id}/{digest}"


# ---------------------------------------------------------------------------
# Throughput: length-sorted batching
# ---------------------------------------------------------------------------


def embed_length_sorted(
    texts: Sequence[str],
    encode: Callable[[Sequence[str]], Any],
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode ``texts`` in length-sorted batches; return vectors in caller order.

    Transformer self-attention is O(n²) in sequence length and a batch is padded
    to its longest member, so grouping similar-length texts together avoids
    wasted compute — a 5–10× speedup on skewed corpora. ``ef`` owns this so
    callers never sort, pad, or un-sort themselves.

    Args:
        texts: The strings to embed.
        encode: Maps a list of strings to an array-like of shape ``(k, dim)``.
        batch_size: Texts per call to ``encode``.

    Returns:
        A ``float32`` array of shape ``(len(texts), dim)`` in the *original*
        order of ``texts``.

    >>> import numpy as np
    >>> def enc(batch):  # toy encoder: vector = [length]
    ...     return np.array([[len(t)] for t in batch], dtype=np.float32)
    >>> out = embed_length_sorted(['aaa', 'a', 'aaaaa', 'aa'], enc, batch_size=2)
    >>> out.ravel().tolist()  # order preserved despite internal sorting
    [3.0, 1.0, 5.0, 2.0]
    """
    items = list(enumerate(texts))
    if not items:
        return np.empty((0, 0), dtype=np.float32)
    items.sort(key=lambda it: len(it[1]))
    vectors: list[np.ndarray | None] = [None] * len(items)
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        encoded = np.asarray(encode([t for _, t in batch]), dtype=np.float32)
        if encoded.shape[0] != len(batch):
            raise EmbedderError(
                f"encode returned {encoded.shape[0]} vectors for "
                f"{len(batch)} texts"
            )
        for (orig_idx, _), vector in zip(batch, encoded):
            vectors[orig_idx] = vector
    return np.stack(vectors).astype(np.float32, copy=False)
