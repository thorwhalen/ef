"""Reranking — a precision pass over retrieved segments (use case F6).

First-stage retrieval (a dense ANN query over ``vd``) is tuned for *recall*: it
casts a wide, cheap net. A **reranker** is the precision counterpart — a slower,
more accurate model that re-scores a handful of candidates against the query.
The standard two-stage pattern is "retrieve ``fetch_k`` cheaply, rerank, keep
the top ``limit``".

This module keeps reranking a *decorator over a retriever*, never a new pipeline
stage — it composes with the existing surface rather than complicating it:

- :class:`Reranker` — the structural protocol: a callable
  ``(query, segments) -> one score per segment``. Any plain function of that
  shape *is* a reranker; no base class to subclass.
- :func:`rerank` — the pure operation: re-score and re-order a list of
  :class:`~ef.segments.Segment`\\ s. The reranker's score is folded into each
  segment's ``metadata["rerank_score"]``.
- :func:`with_reranker` — the decorator: wrap a retrieve-style callable (such as
  :meth:`SourceManager.retrieve <ef.source_manager.SourceManager.retrieve>`) so
  every call over-fetches, reranks, and trims — transparently.
- :func:`cross_encoder_reranker` — a ready reranker backed by a
  sentence-transformers ``CrossEncoder`` (an optional heavy dependency, imported
  lazily — the protocol and :func:`rerank` cost nothing).

Example — rerank three segments by a toy length score, longest first:

>>> segments = [
...     {'text': 'cat', 'id': '1'},
...     {'text': 'dog', 'id': '2'},
...     {'text': 'cathedral', 'id': '3'},
... ]
>>> def by_length(query, segments):
...     return [len(segment['text']) for segment in segments]
>>> [segment['text'] for segment in rerank('a query', segments, by_length, limit=2)]
['cathedral', 'cat']
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from ef.segments import Segment

__all__ = [
    "Reranker",
    "rerank",
    "with_reranker",
    "cross_encoder_reranker",
]


@runtime_checkable
class Reranker(Protocol):
    """A callable that re-scores ``(query, segments)`` — higher means more relevant.

    The structural contract a reranker satisfies: given the query and a list of
    candidate :class:`~ef.segments.Segment`\\ s, return **one score per
    segment**, in the same order. :func:`rerank` sorts by these scores. It is a
    ``@runtime_checkable Protocol``, not a base class — a plain function of the
    right shape already *is* a :class:`Reranker`.

    >>> def by_length(query, segments):
    ...     return [len(s['text']) for s in segments]
    >>> isinstance(by_length, Reranker)
    True
    """

    def __call__(
        self, query: str, segments: Sequence[Segment], /
    ) -> Sequence[float]: ...


def rerank(
    query: str,
    segments: Sequence[Segment],
    reranker: Reranker,
    *,
    limit: int | None = None,
) -> list[Segment]:
    """Re-score ``segments`` against ``query`` with ``reranker`` and re-order them.

    The pure heart of the module — no I/O, no retrieval. The ``reranker`` is
    called once with the whole candidate list; the segments are returned sorted
    by descending score (ties keep their incoming order), optionally trimmed to
    ``limit``. Each returned segment is a copy carrying the reranker's score in
    ``metadata["rerank_score"]`` — so the new ranking is auditable.

    Args:
        query: the search query the segments are scored against.
        segments: the candidate segments (from a first-stage retriever).
        reranker: a :class:`Reranker` — a callable ``(query, segments) -> scores``.
        limit: keep only the top ``limit`` after reranking; ``None`` keeps all.

    Returns:
        the reranked segments — copies, best first.

    Raises:
        ValueError: if the reranker returns a number of scores that does not
            match the number of segments.

    >>> segs = [{'text': 'a', 'id': '1'}, {'text': 'bbb', 'id': '2'}]
    >>> ranked = rerank('q', segs, lambda q, s: [len(x['text']) for x in s])
    >>> [r['text'] for r in ranked]
    ['bbb', 'a']
    >>> ranked[0]['metadata']['rerank_score']
    3.0
    """
    candidates = list(segments)
    if not candidates:
        return []
    scores = list(reranker(query, candidates))
    if len(scores) != len(candidates):
        raise ValueError(
            f"Reranker returned {len(scores)} score(s) for {len(candidates)} "
            f"segment(s) — it must return exactly one score per segment."
        )
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    reranked: list[Segment] = []
    for i in order if limit is None else order[:limit]:
        segment: Segment = dict(candidates[i])  # type: ignore[assignment]
        metadata = dict(segment.get("metadata") or {})
        metadata["rerank_score"] = float(scores[i])
        segment["metadata"] = metadata
        reranked.append(segment)
    return reranked


def with_reranker(
    retriever: Callable[..., Sequence[Segment]],
    reranker: Reranker,
    *,
    fetch_k: int = 50,
) -> Callable[..., list[Segment]]:
    """Decorate a retrieve-style callable with a two-stage reranking pass.

    Wraps ``retriever`` (e.g. :meth:`SourceManager.retrieve
    <ef.source_manager.SourceManager.retrieve>`, or any callable
    ``query -> list[Segment]``) so every call over-fetches ``fetch_k``
    candidates cheaply, :func:`rerank`\\ s them, and returns the top ``limit``.
    The decorated callable keeps the retriever's interface — ``query`` plus a
    keyword-only ``limit`` — so it is a drop-in replacement, including as the
    ``retriever`` argument of :func:`~ef.evaluation.evaluate_retrieval`.

    Args:
        retriever: the first-stage retriever — a callable
            ``retriever(query, *, limit=...) -> segments``.
        reranker: the :class:`Reranker` applied to the over-fetched candidates.
        fetch_k: how many candidates to retrieve before reranking. Larger trades
            latency for recall headroom; it should comfortably exceed the
            ``limit`` callers ask for.

    Returns:
        a new callable ``(query, *, limit=10, **kwargs) -> list[Segment]`` —
        extra keyword arguments (e.g. ``filter``) pass straight through to
        ``retriever``.

    >>> pool = [{'text': t, 'id': t} for t in ['ab', 'abcd', 'abc']]
    >>> def base(query, *, limit=10):
    ...     return pool[:limit]
    >>> retrieve = with_reranker(
    ...     base, lambda q, segs: [len(s['text']) for s in segs], fetch_k=10)
    >>> [s['text'] for s in retrieve('a query', limit=2)]
    ['abcd', 'abc']
    """

    def reranked_retrieve(
        query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[Segment]:
        """Over-fetch, rerank, and return the top ``limit`` segments."""
        candidates = retriever(query, limit=fetch_k, **kwargs)
        return rerank(query, candidates, reranker, limit=limit)

    return reranked_retrieve


def cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    *,
    device: str | None = None,
    batch_size: int = 32,
    **predict_kwargs: Any,
) -> Reranker:
    """Build a :class:`Reranker` backed by a sentence-transformers ``CrossEncoder``.

    A cross-encoder scores a ``(query, passage)`` pair *jointly* — it is the
    accurate, slower second stage two-stage retrieval is built for. The model is
    a heavy, optional dependency: it is imported only when this factory is
    called, so importing :mod:`ef.reranking` itself stays cheap.

    Example::

        from ef import ingest, with_reranker, cross_encoder_reranker

        index = ingest(corpus, embedder="st:all-MiniLM-L6-v2")
        retrieve = with_reranker(index.retrieve, cross_encoder_reranker())
        segments = retrieve("a query", limit=5)

    Args:
        model_name: a sentence-transformers ``CrossEncoder`` model name.
        device: torch device (``"cuda"`` / ``"cpu"`` / ``"mps"``); ``None``
            auto-selects.
        batch_size: how many pairs to score per forward pass.
        predict_kwargs: extra keyword arguments forwarded to ``CrossEncoder.predict``.

    Returns:
        a :class:`Reranker` — ``(query, segments) -> scores``.

    Raises:
        ImportError: if the optional ``sentence-transformers`` package is absent.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "cross_encoder_reranker needs the `sentence-transformers` package. "
            "Install it with: pip install 'ef[sentence-transformers]'"
        ) from exc

    model = CrossEncoder(model_name, device=device)

    def reranker(query: str, segments: Sequence[Segment]) -> list[float]:
        """Score each segment's text against ``query`` with the cross-encoder."""
        if not segments:
            return []
        pairs = [(query, segment["text"]) for segment in segments]
        scores = model.predict(pairs, batch_size=batch_size, **predict_kwargs)
        return [float(score) for score in scores]

    reranker.model_name = model_name  # type: ignore[attr-defined]
    return reranker
