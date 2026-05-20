"""The segmenter dependency-injection seam and the ``imbed`` registry bridge.

:func:`as_segmenter` is the single place ``ef`` coerces a user-supplied
segmenter argument into a well-behaved :class:`~ef.segmenters.Segmenter` — the
exact mirror of :func:`~ef.embedder_adapters.as_embedder` for the embed layer.

:func:`imbed_segmenter` builds a segmenter from ``imbed``'s segmenter registry
(``imbed.components.segmentation``); ``imbed`` is imported lazily, so importing
this module costs nothing extra and ``ef`` works without ``imbed`` installed.

Example — the DI seam in action (no ``imbed`` needed):

>>> from ef.segmenters import Segmenter, RecursiveCharacterSegmenter
>>> seg = as_segmenter('recursive', chunk_size=256)
>>> isinstance(seg, RecursiveCharacterSegmenter)
True
>>> as_segmenter(seg) is seg          # a ready segmenter passes straight through
True
>>> custom = as_segmenter(lambda text: text.split('|'))
>>> [s['text'] for s in custom('a|b|c')]
['a', 'b', 'c']
"""

from __future__ import annotations

from typing import Any

from ef.segmenters import (
    BaseSegmenter,
    FunctionSegmenter,
    RecursiveCharacterSegmenter,
    Segmenter,
    line_segmenter,
)

__all__ = [
    "as_segmenter",
    "imbed_segmenter",
]

#: String aliases routed to the default recursive splitter.
_RECURSIVE_ALIASES = frozenset({"default", "recursive", "recursive-character"})
#: String aliases routed to :func:`~ef.segmenters.line_segmenter`.
_LINE_ALIASES = frozenset({"lines", "line", "line-segmenter"})


def imbed_segmenter(name: str = "default", /, **kwargs: Any) -> Segmenter:
    """Build a :class:`~ef.segmenters.Segmenter` from ``imbed``'s segmenter registry.

    ``imbed``'s registered segmenters are bare ``text -> Iterable[str]``
    callables; this wraps the named one in a :class:`~ef.segmenters.FunctionSegmenter`
    so it emits well-formed :class:`~ef.segments.Segment` pieces.

    Args:
        name: A key in ``imbed.components.segmentation.segmenters`` (``"default"``
            resolves to ``imbed``'s configured default segmenter).
        **kwargs: Forwarded to :class:`~ef.segmenters.FunctionSegmenter`.

    Raises:
        ImportError: if ``imbed`` is not installed.
        ValueError: if ``name`` is not a registered segmenter.

    Requires the ``imbed`` package (``pip install 'ef[imbed]'``).
    """
    try:
        from imbed.components.segmentation import segmenters as registry
    except ImportError as exc:  # pragma: no cover - import-guard
        raise ImportError(
            "imbed_segmenter needs the `imbed` package. "
            "Install it with: pip install 'ef[imbed]'"
        ) from exc
    try:
        func = registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown imbed segmenter {name!r}. Registered: {sorted(registry)}"
        ) from None
    return FunctionSegmenter(func, **kwargs)


def as_segmenter(x: Any = None, /, **kwargs: Any) -> Segmenter:
    """Normalize ``x`` into a :class:`~ef.segmenters.Segmenter` — the DI seam.

    The single place every ``ef`` entry point coerces a user-supplied segmenter
    argument. Accepts, in order:

    1. ``None`` or ``"default"`` / ``"recursive"`` — a
       :class:`~ef.segmenters.RecursiveCharacterSegmenter` (the default);
    2. ``"lines"`` — :func:`~ef.segmenters.line_segmenter`;
    3. any other string — looked up in ``imbed``'s registry via
       :func:`imbed_segmenter`;
    4. a ready-made segmenter (a :class:`~ef.segmenters.BaseSegmenter` or a
       callable marked by ``ef``) — returned unchanged;
    5. a bare callable — wrapped in :class:`~ef.segmenters.FunctionSegmenter`.

    Extra ``**kwargs`` are forwarded to the chosen factory (ignored when ``x``
    is already a segmenter).

    Raises:
        TypeError: if ``x`` is none of the above.

    >>> seg = as_segmenter(chunk_size=256, chunk_overlap=32)
    >>> (seg.chunk_size, seg.chunk_overlap)
    (256, 32)
    """
    if x is None:
        return RecursiveCharacterSegmenter(**kwargs)
    if isinstance(x, str):
        key = x.strip().lower()
        if key in _RECURSIVE_ALIASES:
            return RecursiveCharacterSegmenter(**kwargs)
        if key in _LINE_ALIASES:
            return line_segmenter
        return imbed_segmenter(x, **kwargs)
    if isinstance(x, BaseSegmenter) or getattr(x, "_ef_segmenter", False):
        return x
    if callable(x):
        return FunctionSegmenter(x, **kwargs)
    raise TypeError(
        f"Cannot interpret {x!r} as a Segmenter. Pass a Segmenter, a name "
        f"string ('recursive', 'lines', or an imbed segmenter name), or a "
        f"callable mapping a document to text pieces."
    )
