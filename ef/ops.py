"""The pipeline ops — the callables an :class:`~ef.artifact_graph.ArtifactGraph` runs.

The :class:`~ef.artifact_graph.ArtifactGraph` resolves a :class:`ProducerSpec
<ef.artifact_graph.ProducerSpec>`'s string ``op`` key to a callable through its
injectable ``ops`` registry, then calls ``op(*input_values, **params)``. This
module supplies the two ops Phase 5's pipeline needs and the machinery that
turns a *live component* (a :class:`~ef.segmenters.Segmenter` /
:class:`~ef.embedders.Embedder`) into a serializable :class:`~ef.config.TransformSpec`
*plus* the bound callable to register.

The split is deliberate:

- a :class:`~ef.config.TransformSpec` is **data** — an op *key*, a version,
  params — so it can be persisted (the recipe is content-addressed);
- the callable is **not** data — it closes over a live component — so it is
  re-registered into the graph's ``ops`` registry each process.

The op key carries the **component identity** (:func:`segmenter_identity`,
``embedder.model_id``): ``"segment:<identity>"`` / ``"embed:<model_id>"``.
Because ``op`` participates in :func:`~ef.artifact_graph.artifact_id`, two
configs with different components get disjoint artifact cones automatically,
and two configs with the *same* component share every upstream artifact — which
is what makes config branching free (design notes §3.1).

>>> import numpy as np
>>> from ef import as_embedder, as_segmenter
>>> seg = as_segmenter('recursive', chunk_size=128)
>>> seg_spec, seg_fn = segment_step(seg)
>>> seg_spec.op.startswith('segment:recursive-char')
True
>>> [s['text'] for s in seg_fn('hello world')]
['hello world']
>>> emb = as_embedder(lambda ts: np.ones((len(ts), 4)), model_id='ones@4')
>>> emb_spec, emb_fn = embed_step(emb)
>>> emb_spec.op, emb_spec.params
('embed:ones@4', {'input_type': 'document'})
>>> emb_fn([{'text': 'a'}, {'text': 'b'}]).shape
(2, 4)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ef.config import TransformSpec, step_params
from ef.embedders import Embedder
from ef.hashing import canonical_json
from ef.segmenters import (
    FunctionSegmenter,
    RecursiveCharacterSegmenter,
    Segmenter,
)
from ef.segments import Segment

__all__ = [
    "SEGMENT_OP_VERSION",
    "EMBED_OP_VERSION",
    "segmenter_identity",
    "embedder_identity",
    "make_segment_op",
    "make_embed_op",
    "segment_step",
    "embed_step",
]


#: Version of the :func:`make_segment_op` wrapper. It participates in every
#: downstream :func:`~ef.artifact_graph.artifact_id`, so **bump it on any
#: behavior change** to how a segment artifact is produced — the graph trusts
#: the version and cannot inspect the code (a silent change serves stale cache).
SEGMENT_OP_VERSION = "1"

#: Version of the :func:`make_embed_op` wrapper. Bump on any behavior change —
#: see :data:`SEGMENT_OP_VERSION`. (This is the *wrapper* version; the embedding
#: *model*'s identity already lives in the op key via ``embedder.model_id``.)
EMBED_OP_VERSION = "1"


# ---------------------------------------------------------------------------
# Component identity — the content-addressing key for an op
# ---------------------------------------------------------------------------


def _callable_identity(fn: Callable[..., Any]) -> str:
    """A best-effort stable identity string for a plain callable.

    Uses ``module.qualname`` when available. An anonymous ``lambda`` has the
    qualname ``"<lambda>"`` — distinct lambdas then collide, so a lambda
    segmenter is *not* safely content-addressable across definitions; pass a
    named function (or a :class:`~ef.segmenters.Segmenter` with an explicit
    ``segmenter_id``) when persistence/branching correctness matters.
    """
    module = getattr(fn, "__module__", None) or "?"
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    return f"{module}.{qualname}" if qualname else repr(fn)


def segmenter_identity(segmenter: Segmenter) -> str:
    """A stable identity string for a segmenter — the content-addressing key.

    The segmenter's counterpart of :attr:`Embedder.model_id
    <ef.embedders.Embedder.model_id>`. Two segmenters that segment identically
    must return the *same* identity; any difference that changes the segments
    (chunk size, overlap, separators, tokenizer) must change it. Resolution
    order:

    1. an explicit ``segmenter_id`` attribute, if the segmenter advertises one;
    2. a :class:`~ef.segmenters.RecursiveCharacterSegmenter` — identified by its
       config-bearing attributes (``chunk_size`` / ``chunk_overlap`` /
       ``separators`` / ``tokenizer``);
    3. a :class:`~ef.segmenters.FunctionSegmenter` — by its wrapped callable;
    4. otherwise the segmenter's own ``__qualname__`` / ``__name__`` / type name.

    >>> from ef import as_segmenter
    >>> a = segmenter_identity(as_segmenter('recursive', chunk_size=256))
    >>> b = segmenter_identity(as_segmenter('recursive', chunk_size=512))
    >>> a == b                              # different chunk_size → different id
    False
    >>> a == segmenter_identity(as_segmenter('recursive', chunk_size=256))
    True
    """
    explicit = getattr(segmenter, "segmenter_id", None)
    if explicit:
        return str(explicit)
    if isinstance(segmenter, RecursiveCharacterSegmenter):
        return "recursive-char:" + canonical_json(
            {
                "chunk_size": segmenter.chunk_size,
                "chunk_overlap": segmenter.chunk_overlap,
                "separators": list(segmenter.separators),
                "tokenizer": segmenter.tokenizer,
            }
        )
    if isinstance(segmenter, FunctionSegmenter):
        return "function:" + _callable_identity(segmenter._func)
    name = getattr(segmenter, "__qualname__", None) or getattr(
        segmenter, "__name__", None
    )
    return str(name) if name else type(segmenter).__name__


def embedder_identity(embedder: Embedder) -> str:
    """The content-addressing identity of an embedder — its ``model_id``.

    :attr:`Embedder.model_id <ef.embedders.Embedder.model_id>` already bakes in
    everything that changes the vector (provider, model, dimensionality), so it
    *is* the identity — this function just names that fact for symmetry with
    :func:`segmenter_identity`.

    >>> from ef import as_embedder
    >>> import numpy as np
    >>> embedder_identity(as_embedder(lambda ts: np.ones((len(ts), 2)), model_id='m@2'))
    'm@2'
    """
    return embedder.model_id


# ---------------------------------------------------------------------------
# The ops — bound callables the ArtifactGraph runs
# ---------------------------------------------------------------------------


def make_segment_op(segmenter: Segmenter) -> Callable[[Any], list[Segment]]:
    """Build the **segment** op — a leaf source → its list of :class:`Segment`\\ s.

    The returned callable closes over ``segmenter``; it is what gets registered
    into the :class:`~ef.artifact_graph.ArtifactGraph`'s ``ops`` registry under
    the key :func:`segment_step` computes. It materializes the (streaming)
    segmenter output into a ``list`` — a graph artifact must be a concrete,
    re-readable value.

    >>> from ef import as_segmenter
    >>> op = make_segment_op(as_segmenter('lines'))
    >>> [s['text'] for s in op('one\\ntwo')]
    ['one', 'two']
    """

    def segment_op(source: Any) -> list[Segment]:
        """Segment one source document into a list of segments."""
        return list(segmenter(source))

    return segment_op


def make_embed_op(
    embedder: Embedder,
) -> Callable[[Sequence[Segment]], Any]:
    """Build the **embed** op — a list of :class:`Segment`\\ s → an ``(n, dim)`` array.

    The returned callable closes over ``embedder`` and embeds the segments'
    ``text`` in caller order; row ``i`` of the result is the vector of segment
    ``i``. ``input_type`` is a hashable step param (it changes the vector for
    task-aware embedders) — it defaults to ``"document"``, the indexing role.

    >>> import numpy as np
    >>> from ef import as_embedder
    >>> op = make_embed_op(as_embedder(lambda ts: np.ones((len(ts), 3)), model_id='m@3'))
    >>> op([{'text': 'a'}, {'text': 'b'}]).shape
    (2, 3)
    >>> op([]).shape[0]                     # an empty doc embeds to zero rows
    0
    """

    def embed_op(segments: Sequence[Segment], *, input_type: str = "document") -> Any:
        """Embed a list of segments into an ``(n, dim)`` array."""
        texts = [segment["text"] for segment in segments]
        return embedder(texts, input_type=input_type)  # type: ignore[arg-type]

    return embed_op


# ---------------------------------------------------------------------------
# Step builders — a live component → (TransformSpec, bound callable)
# ---------------------------------------------------------------------------


def segment_step(segmenter: Segmenter) -> tuple[TransformSpec, Callable[[Any], Any]]:
    """A segmenter → its :class:`~ef.config.TransformSpec` and bound op callable.

    The :class:`~ef.config.TransformSpec` is serializable data (op key + version
    + params); the callable closes over the live ``segmenter`` and is registered
    into the graph's ``ops`` registry under ``spec.op``. Pair them: persist the
    spec, re-register the callable each process.

    >>> from ef import as_segmenter
    >>> spec, fn = segment_step(as_segmenter('recursive', chunk_size=128))
    >>> spec.op_version, spec.params
    ('1', {})
    """
    op_callable = make_segment_op(segmenter)
    op_key = f"segment:{segmenter_identity(segmenter)}"
    spec = TransformSpec(
        op=op_key,
        op_version=SEGMENT_OP_VERSION,
        params=step_params(op_callable, n_inputs=1),
    )
    return spec, op_callable


def embed_step(embedder: Embedder) -> tuple[TransformSpec, Callable[[Any], Any]]:
    """An embedder → its :class:`~ef.config.TransformSpec` and bound op callable.

    The mirror of :func:`segment_step` for the embed layer. The op key is
    ``"embed:<model_id>"`` — the model identity already content-addresses the
    vector — and ``params`` carries the defaults-filled ``input_type``.

    >>> import numpy as np
    >>> from ef import as_embedder
    >>> spec, fn = embed_step(as_embedder(lambda ts: np.ones((len(ts), 2)), model_id='m@2'))
    >>> spec.op, spec.params
    ('embed:m@2', {'input_type': 'document'})
    """
    op_callable = make_embed_op(embedder)
    op_key = f"embed:{embedder_identity(embedder)}"
    spec = TransformSpec(
        op=op_key,
        op_version=EMBED_OP_VERSION,
        params=step_params(op_callable, n_inputs=1),
    )
    return spec, op_callable
