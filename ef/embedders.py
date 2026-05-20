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
- :class:`HashingEmbedder` — a dependency-free embedder (the feature-hashing
  trick, numpy only); ``ef``'s zero-install default when no embedder is given.
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
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
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
    "HashingEmbedder",
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
# HashingEmbedder — the dependency-free default
# ---------------------------------------------------------------------------

#: Algorithm version of :class:`HashingEmbedder`. It is baked into the
#: ``model_id`` — and therefore into every cached vector and every
#: :class:`~ef.artifact_graph.ArtifactGraph` artifact id. **Bump it on any
#: change** to how a vector is produced (tokenization, hashing, weighting) so
#: stale vectors from an older algorithm are never silently reused.
_HASHING_EMBEDDER_VERSION = 1

#: Default output width of :class:`HashingEmbedder` — wide enough that token
#: collisions are rare on ordinary corpora, small enough to stay cheap.
_HASHING_DEFAULT_DIM = 512

#: Splits text into word tokens for :class:`HashingEmbedder`: Unicode-aware
#: runs of "word" characters (the caller lowercases first).
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_ngrams(text: str, ngram_range: tuple[int, int]) -> Iterator[str]:
    """Yield the lowercased word n-grams of ``text`` for ``ngram_range``.

    >>> list(_word_ngrams('The quick fox', (1, 2)))
    ['the', 'quick', 'fox', 'the quick', 'quick fox']
    >>> list(_word_ngrams('one two', (2, 2)))
    ['one two']
    """
    words = _WORD_RE.findall(text.lower())
    n_min, n_max = ngram_range
    for n in range(n_min, n_max + 1):
        if n == 1:
            yield from words
        else:
            for i in range(len(words) - n + 1):
                yield " ".join(words[i : i + n])


def _hash_bucket(token: str, dim: int) -> tuple[int, float]:
    """Hash ``token`` to a ``(bucket, sign)`` pair — the feature-hashing trick.

    Uses ``blake2b`` rather than the builtin ``hash()``: the latter is salted
    per process (``PYTHONHASHSEED``), but ``ef``'s vectors must be byte-identical
    across runs and machines for content addressing to hold. The low bit is the
    sign — signed hashing makes bucket collisions cancel in expectation rather
    than compound (Weinberger et al., 2009).
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    h = int.from_bytes(digest, "big")
    return (h >> 1) % dim, (1.0 if h & 1 else -1.0)


def _hashing_model_id(
    dim: int, ngram_range: tuple[int, int], sublinear_tf: bool
) -> str:
    """Build a :class:`HashingEmbedder` ``model_id`` — ``"hashing:<name>@<dim>"``.

    Every vector-affecting parameter is encoded, so two embedders with the same
    ``model_id`` truly produce the same vectors. The default configuration
    yields the clean ``"hashing:v1@512"``; each non-default appends a marker.

    >>> _hashing_model_id(512, (1, 2), True)
    'hashing:v1@512'
    >>> _hashing_model_id(256, (1, 3), False)
    'hashing:v1-ng1_3-tflin@256'
    """
    name = f"v{_HASHING_EMBEDDER_VERSION}"
    if ngram_range != (1, 2):
        name += f"-ng{ngram_range[0]}_{ngram_range[1]}"
    if not sublinear_tf:
        name += "-tflin"
    return f"hashing:{name}@{dim}"


class HashingEmbedder(BaseEmbedder):
    """A dependency-free embedder — the feature-hashing trick, numpy only.

    :class:`HashingEmbedder` is ``ef``'s **zero-install default**: the embedder
    :func:`~ef.source_manager.ingest` resolves to when none is given, so the
    headline ``ingest([...]).search(query)`` works on a bare ``pip install ef``
    — no torch, no model download, no network.

    It is a *lexical* embedder, not a neural one. Each text is tokenized into
    lowercased word n-grams; every token is hashed into one of ``dim`` buckets
    with a sign (the hashing trick — no vocabulary, no fitting, fixed memory);
    the signed, sublinear-tf-weighted bucket counts are L2-normalized. The
    cosine similarity of two vectors then ranks by shared vocabulary — enough
    for demos, doctests and offline tests, and a sane fallback in production.
    For genuine semantic quality install a real embedder (``ef[sentence-transformers]``,
    ``ef[openai]``) and pass it explicitly.

    Every parameter that changes a vector is baked into :attr:`model_id`, so a
    :class:`HashingEmbedder`'s artifacts stay correctly content-addressed in the
    :class:`~ef.artifact_graph.ArtifactGraph`.

    Args:
        dim: Output width. Wider → fewer hash collisions, larger vectors.
        ngram_range: Inclusive ``(min, max)`` word-n-gram sizes. The default
            ``(1, 2)`` mixes single words with adjacent pairs — a little phrase
            sensitivity at no real cost.
        sublinear_tf: Weight a token by ``1 + log(count)`` rather than its raw
            ``count``, damping a word repeated many times in one text.

    >>> e = HashingEmbedder()
    >>> e.model_id
    'hashing:v1@512'
    >>> import numpy as np
    >>> v = e(['the quick brown fox'])
    >>> v.shape
    (1, 512)
    >>> bool(np.isclose(np.linalg.norm(v[0]), 1.0))  # rows are L2-normalized
    True
    >>> a, b = e(['hello world']), e(['hello world'])
    >>> round(float(a[0] @ b[0]), 5)  # identical text → cosine similarity 1.0
    1.0
    >>> isinstance(e, Embedder)
    True
    """

    normalized = True
    honored_input_types: tuple[InputType, ...] = ()  # a lexical, task-agnostic model

    def __init__(
        self,
        *,
        dim: int = _HASHING_DEFAULT_DIM,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        n_min, n_max = ngram_range
        if n_min < 1 or n_max < n_min:
            raise ValueError(
                f"ngram_range must satisfy 1 <= min <= max, got {ngram_range!r}"
            )
        if dim < 1:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        self.dim = int(dim)
        self.ngram_range = (int(n_min), int(n_max))
        self.sublinear_tf = bool(sublinear_tf)
        self.model_id = _hashing_model_id(self.dim, self.ngram_range, self.sublinear_tf)

    def __call__(
        self,
        texts: Iterable[str],
        *,
        input_type: InputType | None = None,
        **backend: Any,
    ) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            self._vectorize_into(text, out[row])
        return out

    def _vectorize_into(self, text: str, row: np.ndarray) -> None:
        """Fill ``row`` in place with the L2-normalized hashed vector of ``text``.

        An empty (or token-free) text leaves ``row`` a zero vector — it has no
        unit direction, and a zero vector is the honest representation.
        """
        counts = Counter(_word_ngrams(text, self.ngram_range))
        for token, count in counts.items():
            bucket, sign = _hash_bucket(token, self.dim)
            weight = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            row[bucket] += sign * weight
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            row /= norm


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
                f"encode returned {encoded.shape[0]} vectors for {len(batch)} texts"
            )
        for (orig_idx, _), vector in zip(batch, encoded):
            vectors[orig_idx] = vector
    return np.stack(vectors).astype(np.float32, copy=False)
