"""The ``Segmenter`` facade — ``ef``'s segment layer (L2).

A *segmenter* turns a document into a stream of :class:`~ef.segments.Segment`
pieces. Like the embedder facade, its core is a tiny structural contract::

    Segmenter = Callable[[str | Mapping], Iterable[Segment]]

Everything else — overlap, hierarchy, batching, the recursive splitting
algorithm — is *layered scaffolding* over that one shape.

This module defines:

- :class:`Segmenter` — the ``@runtime_checkable`` protocol (a callable). It is
  a protocol, **not** an ABC: any matching callable *is* a segmenter.
- :class:`BatchedSegmenter` — an optional optimization trait (a ``batch``
  method) for segmenters that can process many documents at once.
- :class:`BaseSegmenter` — an implementation-convenience base (a default
  :meth:`~BaseSegmenter.batch` and a ``repr``). Subclassing is optional.
- :class:`RecursiveCharacterSegmenter` — the **default segmenter**: recursive
  character splitting at ~512 tokens with ~12.5% overlap. The honest default —
  within a point or two of fancier methods on realistic corpora.
- :func:`line_segmenter` — a builtin that splits a document into its non-empty
  lines (a plain function — proof that a bare callable *is* a segmenter).
- :class:`FunctionSegmenter` — wraps a bare ``text -> pieces`` callable into a
  full segmenter; the bridge for ``imbed``'s registered segmenters.
- Composition helpers — :func:`with_overlap`, :func:`hierarchical`,
  :func:`materialise`.

Streaming is first-class: a segmenter returns an *iterable*, not a list, so the
heavy case (a huge corpus) never has to fit every segment in memory at once.

The dependency-injection seam :func:`~ef.segmenter_adapters.as_segmenter` and
the ``imbed`` registry bridge live in :mod:`ef.segmenter_adapters`.

Example — split a document into ~4-token chunks with one token of overlap:

>>> seg = RecursiveCharacterSegmenter(chunk_size=4, chunk_overlap=1)
>>> [c['text'] for c in seg('alpha beta gamma delta epsilon')]
['alpha beta gamma delta', 'delta epsilon']
>>> isinstance(seg, Segmenter)
True
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ef.segments import Segment, as_segment, make_segment, segment_id

__all__ = [
    "Segmenter",
    "BatchedSegmenter",
    "BaseSegmenter",
    "FunctionSegmenter",
    "RecursiveCharacterSegmenter",
    "line_segmenter",
    "with_overlap",
    "hierarchical",
    "materialise",
    "approx_token_count",
    "APPROX_TOKENIZER",
    "DEFAULT_SEPARATORS",
]


# ---------------------------------------------------------------------------
# Tokenization heuristic
# ---------------------------------------------------------------------------

#: Identifier recorded as the ``tokenizer`` metadata key when token counts come
#: from :func:`approx_token_count`. Recording *which* tokenizer counted a
#: segment is mandatory — a ``tokens`` number is meaningless without it.
APPROX_TOKENIZER = "char-approx"


def approx_token_count(text: str) -> int:
    """Rough token count — the common ``~4 characters per token`` heuristic.

    A dependency-free default so ``ef`` can talk in "tokens" without pulling in
    ``tiktoken`` or a model tokenizer. Inject a real counter (and its name) into
    a segmenter when an exact count matters.

    >>> approx_token_count('a' * 40)
    10
    >>> approx_token_count('')
    0
    """
    return max(1, round(len(text) / 4)) if text else 0


# ---------------------------------------------------------------------------
# The protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Segmenter(Protocol):
    """Structural type for a segmenter: ``str | Mapping -> Iterable[Segment]``.

    A segmenter is *just a callable*. It accepts a document — a bare ``str`` or
    a ``Mapping`` with a ``text`` key (and optionally an ``id``, which becomes
    the emitted segments' ``parent_id``) — and yields :class:`~ef.segments.Segment`
    pieces. It returns an **iterable**, not a list: streaming is the default so
    the heavy case scales.

    Because the protocol's only member is ``__call__``, *every* callable
    satisfies ``isinstance(x, Segmenter)``. Treat it as documentation of intent,
    not a discriminating check — :func:`~ef.segmenter_adapters.as_segmenter` is
    the seam that turns an arbitrary callable into a *well-behaved* segmenter.
    """

    def __call__(self, doc: str | Mapping[str, Any], /) -> Iterable[Segment]:
        """Segment ``doc`` into an iterable of :class:`~ef.segments.Segment`."""
        ...


@runtime_checkable
class BatchedSegmenter(Protocol):
    """Optional trait: a segmenter that can process many documents at once.

    Plain :class:`Segmenter` handles one document per call. A segmenter that can
    amortize per-call cost across documents (a shared model load, a vectorized
    pass) also offers ``batch``. :class:`BaseSegmenter` supplies a correct — if
    un-optimized — default, so every :class:`BaseSegmenter` subclass is already
    a :class:`BatchedSegmenter`.
    """

    def batch(
        self, docs: Iterable[str | Mapping[str, Any]], /
    ) -> Iterable[Iterable[Segment]]:
        """Segment each document in ``docs``, yielding one segment-stream each."""
        ...


class BaseSegmenter:
    """Implementation base for segmenters (optional — :class:`Segmenter` is structural).

    A subclass implements ``__call__``; in return it gets a default
    :meth:`batch` (one-document-at-a-time — override it for a genuinely batched
    backend) and a readable ``repr``. Subclassing is purely a convenience;
    satisfying :class:`Segmenter` structurally is what matters.
    """

    def batch(
        self, docs: Iterable[str | Mapping[str, Any]], /
    ) -> Iterator[Iterable[Segment]]:
        """Default: segment each document independently."""
        for doc in docs:
            yield self(doc)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Document unpacking
# ---------------------------------------------------------------------------


def _unpack_doc(
    doc: str | Mapping[str, Any],
) -> tuple[str, str | None, dict[str, Any]]:
    """Normalize a document into ``(text, parent_id, base_metadata)``.

    A bare ``str`` is its own text with no parent and no metadata. A ``Mapping``
    must carry a ``text`` key; its ``id`` (if any) becomes the ``parent_id`` of
    every segment cut from it, and its remaining keys — flattened with an
    explicit ``metadata`` sub-mapping — become each segment's base metadata.
    """
    if isinstance(doc, str):
        return doc, None, {}
    if isinstance(doc, Mapping):
        if "text" not in doc:
            raise ValueError("a document mapping must have a 'text' key")
        text = doc["text"]
        if not isinstance(text, str):
            raise TypeError(f"document 'text' must be a str, got {type(text).__name__}")
        parent_id = doc.get("id")
        explicit = doc.get("metadata") or {}
        other = {
            k: v
            for k, v in doc.items()
            if k not in ("text", "id", "metadata") and v is not None
        }
        return text, parent_id, {**explicit, **other}
    raise TypeError(f"a document must be a str or a Mapping, got {type(doc).__name__}")


# ---------------------------------------------------------------------------
# line_segmenter — the simplest builtin
# ---------------------------------------------------------------------------


def line_segmenter(doc: str | Mapping[str, Any], /) -> Iterator[Segment]:
    """Segment a document into its non-empty lines.

    Each line is stripped of surrounding whitespace; blank lines are dropped.
    Character offsets point at the *stripped* content within the source.

    A plain function — and therefore already a :class:`Segmenter`. It carries
    the marker :func:`~ef.segmenter_adapters.as_segmenter` uses to recognize a
    ready-made segmenter.

    >>> [s['text'] for s in line_segmenter('  hello  \\n\\nworld\\n')]
    ['hello', 'world']
    """
    text, parent_id, base_metadata = _unpack_doc(doc)
    metadata = {**base_metadata, "tokenizer": APPROX_TOKENIZER}
    index = 0
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        stripped = raw_line.strip()
        if stripped:
            start = offset + raw_line.find(stripped)
            yield make_segment(
                stripped,
                parent_id=parent_id,
                start=start,
                end=start + len(stripped),
                index=index,
                tokens=approx_token_count(stripped),
                metadata=metadata,
            )
            index += 1
        offset += len(raw_line)


line_segmenter._ef_segmenter = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# FunctionSegmenter — wrap a bare callable
# ---------------------------------------------------------------------------


class FunctionSegmenter(BaseSegmenter):
    """Promote a bare ``text -> pieces`` callable to a full :class:`Segmenter`.

    The bridge for segmenter *functions* that carry no segment schema — most
    importantly ``imbed``'s registered segmenters, which map text to an iterable
    of plain strings. Each piece the callable yields is normalized:

    - a ``str`` piece becomes a :class:`~ef.segments.Segment`; its character
      offsets are recovered by locating it in the source text, and a token
      count plus ``tokenizer`` metadata are attached;
    - a ``Mapping`` / :class:`~ef.segments.SegmentRecord` piece is coerced with
      :func:`~ef.segments.as_segment`, gaining only an ``index`` / ``parent_id``
      where it lacks them.

    Args:
        func: The callable to wrap. By default it receives the document's
            *text*; with ``pass_doc=True`` it receives the whole document.
        pass_doc: Pass the raw document instead of its text.
        count_tokens: Token counter for ``str`` pieces (``None`` disables the
            ``tokens`` / ``tokenizer`` annotation).
        tokenizer: Name recorded as the ``tokenizer`` metadata key.

    >>> fs = FunctionSegmenter(lambda text: text.split(','))
    >>> [s['text'] for s in fs('a,b,c')]
    ['a', 'b', 'c']
    >>> s = list(fs('a,b,c'))[1]
    >>> (s['start'], s['end'], s['index'])
    (2, 3, 1)
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        pass_doc: bool = False,
        count_tokens: Callable[[str], int] | None = approx_token_count,
        tokenizer: str = APPROX_TOKENIZER,
    ) -> None:
        if not callable(func):
            raise TypeError(
                f"FunctionSegmenter needs a callable, got {type(func).__name__}"
            )
        self._func = func
        self._pass_doc = pass_doc
        self._count_tokens = count_tokens
        self.tokenizer = tokenizer

    def __repr__(self) -> str:
        name = getattr(self._func, "__name__", repr(self._func))
        return f"{type(self).__name__}({name})"

    def __call__(self, doc: str | Mapping[str, Any], /) -> Iterator[Segment]:
        text, parent_id, base_metadata = _unpack_doc(doc)
        raw = self._func(doc if self._pass_doc else text)
        if isinstance(raw, (str, Mapping)):
            raw = [raw]
        cursor = 0
        index = 0
        for item in raw:
            if isinstance(item, str):
                if not item:
                    continue
                metadata = dict(base_metadata)
                kwargs: dict[str, Any] = {"index": index, "parent_id": parent_id}
                located = text.find(item, cursor)
                if located >= 0:
                    kwargs["start"] = located
                    kwargs["end"] = located + len(item)
                    cursor = located + len(item)
                if self._count_tokens is not None:
                    kwargs["tokens"] = self._count_tokens(item)
                    metadata["tokenizer"] = self.tokenizer
                kwargs["metadata"] = metadata or None
                yield make_segment(item, **kwargs)
            else:
                yield as_segment(
                    item,
                    index=index,
                    parent_id=parent_id,
                    metadata=base_metadata or None,
                )
            index += 1


# ---------------------------------------------------------------------------
# RecursiveCharacterSegmenter — the default
# ---------------------------------------------------------------------------

#: Separators tried, in order, by :class:`RecursiveCharacterSegmenter`. Earlier
#: (coarser) boundaries are preferred; the empty string is the last resort,
#: splitting between characters so an oversized run is never left unsplit.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", ", ", " ", "")


def _split_with_offsets(
    text: str,
    base: int,
    separators: Sequence[str],
    count_tokens: Callable[[str], int],
    chunk_size: int,
) -> list[tuple[str, int]]:
    """Recursively split ``text`` into ``(piece, absolute_offset)`` pairs.

    The first separator that occurs in ``text`` is used; pieces still larger
    than ``chunk_size`` tokens are split again with the remaining (finer)
    separators. Every piece is small enough to merge — or atomic (no separator
    left). Offsets are absolute (``base`` is the offset of ``text`` itself).
    """
    separator = separators[-1]
    rest: Sequence[str] = ()
    for i, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator = candidate
            rest = separators[i + 1 :]
            break

    raw_pieces: list[tuple[str, int]] = []
    if separator == "":
        raw_pieces = [(ch, base + i) for i, ch in enumerate(text)]
    else:
        cursor = 0
        for part in text.split(separator):
            raw_pieces.append((part, base + cursor))
            cursor += len(part) + len(separator)

    result: list[tuple[str, int]] = []
    for piece, offset in raw_pieces:
        if not piece:
            continue
        if not rest or count_tokens(piece) <= chunk_size:
            result.append((piece, offset))
        else:
            result.extend(
                _split_with_offsets(piece, offset, rest, count_tokens, chunk_size)
            )
    return result


def _merge_pieces(
    pieces: Sequence[tuple[str, int]],
    *,
    count_tokens: Callable[[str], int],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """Merge consecutive pieces into ``(start, end)`` chunk spans.

    A chunk's text is recovered as ``source[start:end]`` — so the separators
    *between* merged pieces are preserved verbatim, while leading/trailing
    separators at chunk boundaries are trimmed. Each emitted chunk carries at
    most ``chunk_overlap`` tokens of trailing context into the next, and a span
    that would consist *only* of carried-over overlap is never emitted.
    """

    def piece_end(piece: tuple[str, int]) -> int:
        return piece[1] + len(piece[0])

    spans: list[tuple[int, int]] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    carried = 0  # leading pieces of `current` that are carried-over overlap

    for piece in pieces:
        piece_tokens = count_tokens(piece[0])
        if current and current_tokens + piece_tokens > chunk_size:
            if carried < len(current):  # skip a pure-overlap span
                spans.append((current[0][1], piece_end(current[-1])))
            current, current_tokens = _overlap_tail(
                current, count_tokens, chunk_overlap
            )
            carried = len(current)
        current.append(piece)
        current_tokens += piece_tokens

    if current and carried < len(current):
        spans.append((current[0][1], piece_end(current[-1])))
    return spans


def _overlap_tail(
    current: list[tuple[str, int]],
    count_tokens: Callable[[str], int],
    chunk_overlap: int,
) -> tuple[list[tuple[str, int]], int]:
    """Trailing pieces of ``current`` to seed the next chunk's overlap.

    Returns at most ``len(current) - 1`` pieces (always leaving the next chunk
    strictly ahead of this one, so merging cannot stall) totalling roughly
    ``chunk_overlap`` tokens.
    """
    if chunk_overlap <= 0 or len(current) <= 1:
        return [], 0
    tail: list[tuple[str, int]] = []
    tail_tokens = 0
    for piece in reversed(current):
        if len(tail) >= len(current) - 1:
            break
        piece_tokens = count_tokens(piece[0])
        if tail and tail_tokens + piece_tokens > chunk_overlap:
            break
        tail.insert(0, piece)
        tail_tokens += piece_tokens
    return tail, tail_tokens


class RecursiveCharacterSegmenter(BaseSegmenter):
    """The default segmenter — recursive character splitting.

    Splits a document on a descending ladder of separators (paragraph → line →
    sentence → clause → word → character), merging the resulting pieces into
    chunks of about ``chunk_size`` tokens with ``chunk_overlap`` tokens shared
    between neighbours. This "good enough" default is, on realistic corpora,
    within a point or two of far more elaborate methods — upgrade only if you
    can measure the win.

    Sizes are measured with ``count_tokens``. The default counter is the
    dependency-free :func:`approx_token_count`; inject a real tokenizer (and its
    ``tokenizer`` name) when an exact count matters. Every emitted segment
    records the tokenizer in its metadata and carries exact ``start`` / ``end``
    character offsets into the source.

    Args:
        chunk_size: Target chunk size, in ``count_tokens`` units.
        chunk_overlap: Tokens of overlap between consecutive chunks (must be
            smaller than ``chunk_size``).
        separators: The separator ladder, coarsest first; the last should be
            ``""`` so an unsplittable run is still broken between characters.
        count_tokens: Maps a string to a token count.
        tokenizer: Name recorded as the ``tokenizer`` metadata key.

    >>> seg = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=1)
    >>> [c['text'] for c in seg('one two three four')]
    ['one two three', 'three four']
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        count_tokens: Callable[[str], int] = approx_token_count,
        tokenizer: str = APPROX_TOKENIZER,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)
        self._count_tokens = count_tokens
        self.tokenizer = tokenizer

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )

    def __call__(self, doc: str | Mapping[str, Any], /) -> Iterator[Segment]:
        text, parent_id, base_metadata = _unpack_doc(doc)
        if not text:
            return
        pieces = _split_with_offsets(
            text, 0, self.separators, self._count_tokens, self.chunk_size
        )
        spans = _merge_pieces(
            pieces,
            count_tokens=self._count_tokens,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        metadata = {**base_metadata, "tokenizer": self.tokenizer}
        for index, (start, end) in enumerate(spans):
            chunk_text = text[start:end]
            yield make_segment(
                chunk_text,
                parent_id=parent_id,
                start=start,
                end=end,
                index=index,
                tokens=self._count_tokens(chunk_text),
                metadata=metadata,
            )


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def with_overlap(
    segmenter: Segmenter,
    n_chars: int,
    /,
    *,
    count_tokens: Callable[[str], int] = approx_token_count,
    tokenizer: str = APPROX_TOKENIZER,
) -> Segmenter:
    """Wrap ``segmenter`` so consecutive segments overlap by ``n_chars`` characters.

    Each segment is extended *forward* by up to ``n_chars`` characters (clamped
    to the source length), so segment *K* shares its tail with segment *K+1*'s
    head. A segment whose text changes has its ``id`` re-derived and its
    ``tokens`` recounted; segments lacking ``start`` / ``end`` offsets pass
    through untouched.

    Overlap is expressed in *characters* here (this is a generic post-hoc
    helper) — for token-measured overlap use a segmenter's own ``chunk_overlap``.

    >>> base = RecursiveCharacterSegmenter(chunk_size=3, chunk_overlap=0)
    >>> [c['text'] for c in base('one two three four')]
    ['one two three', 'four']
    >>> [c['text'] for c in with_overlap(base, 100)('one two three four')]
    ['one two three four', 'four']
    """
    if n_chars < 0:
        raise ValueError("n_chars must be non-negative")

    def overlapping(doc: str | Mapping[str, Any], /) -> Iterator[Segment]:
        text, _, _ = _unpack_doc(doc)
        for seg in segmenter(doc):
            start = seg.get("start")
            end = seg.get("end")
            if n_chars == 0 or start is None or end is None:
                yield seg
                continue
            new_end = min(len(text), end + n_chars)
            if new_end == end:
                yield seg
                continue
            new_text = text[start:new_end]
            metadata = dict(seg.get("metadata") or {})
            updated: Segment = dict(seg)  # type: ignore[assignment]
            updated["text"] = new_text
            updated["end"] = new_end
            if "tokens" in seg:
                updated["tokens"] = count_tokens(new_text)
                metadata["tokenizer"] = tokenizer
            if metadata:
                updated["metadata"] = metadata
            updated["id"] = segment_id(new_text, metadata or None)
            yield updated

    overlapping._ef_segmenter = True  # type: ignore[attr-defined]
    return overlapping


def hierarchical(segmenters: Sequence[Any], /) -> Segmenter:
    """Compose segmenters into levels — each segments the previous level's output.

    Level 0 applies ``segmenters[0]`` to the document. Level *k* applies
    ``segmenters[k]`` to every level-*(k-1)* segment's text, setting each
    child's ``parent_id`` to that segment's ``id`` and shifting its offsets to
    stay absolute. The result is a flat stream of every segment of every level
    (level 0 first), with the tree expressed purely through ``parent_id``
    pointers.

    Each entry of ``segmenters`` is passed through
    :func:`~ef.segmenter_adapters.as_segmenter`, so strings (e.g. ``"recursive"``)
    and bare callables are accepted alongside ready-made segmenters.
    """
    from ef.segmenter_adapters import as_segmenter

    resolved = [as_segmenter(s) for s in segmenters]
    if not resolved:
        raise ValueError("hierarchical needs at least one segmenter")

    def hierarchical_segmenter(doc: str | Mapping[str, Any], /) -> Iterator[Segment]:
        level = list(resolved[0](doc))
        yield from level
        for seg_fn in resolved[1:]:
            next_level: list[Segment] = []
            for parent in level:
                offset = parent.get("start") or 0
                child_doc = {"text": parent["text"], "id": parent["id"]}
                for child in seg_fn(child_doc):
                    child = dict(child)  # type: ignore[assignment]
                    if "start" in child:
                        child["start"] += offset
                    if "end" in child:
                        child["end"] += offset
                    next_level.append(child)
            yield from next_level
            level = next_level

    hierarchical_segmenter._ef_segmenter = True  # type: ignore[attr-defined]
    return hierarchical_segmenter


def materialise(x: Any) -> Any:
    """Force a streaming segmenter (or segment stream) to a concrete list.

    Streaming is the default, but the light case often wants everything in
    hand. :func:`materialise` is polymorphic:

    - given a **segmenter** (a callable), it returns a new segmenter whose
      output is a ``list`` rather than a lazy iterator;
    - given an **iterable of segments**, it returns a ``list`` of them.

    >>> eager = materialise(line_segmenter)
    >>> out = eager('a\\nb')
    >>> isinstance(out, list)
    True
    >>> [s['text'] for s in out]
    ['a', 'b']
    >>> materialise(s for s in [{'text': 'x'}])
    [{'text': 'x'}]
    """
    if callable(x):

        def materialised(doc: str | Mapping[str, Any], /) -> list[Segment]:
            return list(x(doc))

        materialised._ef_segmenter = True  # type: ignore[attr-defined]
        return materialised
    return list(x)
