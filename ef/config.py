"""The config layer — declarative, content-addressable pipeline specs.

A *pipeline* in ``ef`` is not an object graph you wire by hand — it is **data**.
A :class:`PipelineSpec` declares the segment→embed chain; each step is a
:class:`TransformSpec` (an op key, an op version, and the keyword params that
feed the content hash). Because a spec is fully serializable and content-hashed
(:func:`config_id`), two configs that share a step share that step's artifacts
in the :class:`~ef.artifact_graph.ArtifactGraph` for free — config branching
costs nothing beyond the divergent cone (design notes §3.1).

This module is the *bridge* between an ordinary Python call and a content-
addressed graph node. For content addressing to be **correct**, two
semantically-equal calls must produce identical params — so ``op(text)`` and
``op(text, size=512)`` (``512`` being the default) must hash the same. The
canonical ``i2`` idiom does exactly that normalization:

- :func:`full_kwargs` — the raw recipe (design notes §3.8): a call ``(args,
  kwargs)`` → a fully-named, defaults-filled kwargs dict.
- :func:`step_params` — its spec-building application: the canonical *keyword*
  params of an op, with the leading positional *input* slots (the upstream
  artifact ids) dropped, since those become :attr:`ProducerSpec.inputs
  <ef.artifact_graph.ProducerSpec.inputs>`, not params.

Phase 4's :class:`~ef.artifact_graph.ArtifactGraph` takes :class:`ProducerSpec
<ef.artifact_graph.ProducerSpec>`\\ s as given; this module (and
:mod:`ef.ops` / :mod:`ef.source_manager`) is where ordinary calls *become*
those specs.

Example — two pipelines, equal segment step, divergent embed step:

>>> seg = TransformSpec(op='segment:recursive', op_version='1', params={})
>>> embed_a = TransformSpec(op='embed:m-a', op_version='1', params={})
>>> embed_b = TransformSpec(op='embed:m-b', op_version='1', params={})
>>> pa = PipelineSpec(segment=seg, embed=embed_a)
>>> pb = PipelineSpec(segment=seg, embed=embed_b)
>>> config_id(pa) == config_id(pb)        # different embedders → different config
False
>>> config_id(pa) == config_id(PipelineSpec.from_dict(pa.as_dict()))
True
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from ef.hashing import canonical_json, sha256_hex

__all__ = [
    "ConfigId",
    "full_kwargs",
    "step_params",
    "TransformSpec",
    "PipelineSpec",
    "config_id",
]


#: A content-addressed pipeline identifier — the ``sha256`` hex digest of a
#: :class:`PipelineSpec` (:func:`config_id`). It is what ``ef`` writes into
#: ``vd`` index metadata as ``config_hash`` so a later phase can tell which
#: pipeline produced a vector.
ConfigId = str


# ---------------------------------------------------------------------------
# Call normalization — turning an op call into hashable params
# ---------------------------------------------------------------------------


def full_kwargs(
    op: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a call into a canonical, fully-named, defaults-filled kwargs dict.

    The ``i2.Sig`` recipe of design notes §3.8. A call expressed as positional
    ``args`` + keyword ``kwargs`` is mapped onto ``op``'s signature: every
    parameter is named, and any parameter the call left implicit is filled with
    its default. The result is therefore *stable* — ``op(x)`` and
    ``op(x, size=512)`` (``512`` the default) yield the **same** dict, which is
    what makes a content hash built from it correct.

    Args:
        op: the callable whose signature defines the parameter names/defaults.
        args: the call's positional arguments — for an :mod:`ef.ops` op these
            are the upstream artifact ids; :func:`step_params` drops them again.
        kwargs: the call's keyword arguments.

    >>> def embed(segments, *, input_type='document', dim=None): ...
    >>> full_kwargs(embed, ('seg-1',), {'input_type': 'query'})
    {'segments': 'seg-1', 'input_type': 'query', 'dim': None}
    >>> full_kwargs(embed, ('seg-1',))                 # defaults filled in
    {'segments': 'seg-1', 'input_type': 'document', 'dim': None}
    """
    from i2 import Sig

    return dict(
        Sig(op).map_arguments(
            tuple(args),
            dict(kwargs or {}),
            apply_defaults=True,
            allow_partial=True,
            allow_excess=True,
        )
    )


def step_params(
    op: Callable[..., Any],
    raw_params: Mapping[str, Any] | None = None,
    *,
    n_inputs: int = 1,
) -> dict[str, Any]:
    """The canonical *keyword* params of an op step — its inputs dropped.

    An :mod:`ef.ops` op is called ``op(*input_values, **params)``: the first
    ``n_inputs`` positional parameters are upstream artifacts (they belong in
    :attr:`ProducerSpec.inputs <ef.artifact_graph.ProducerSpec.inputs>`, not in
    the params), and every remaining parameter is a hashable step param. This
    runs :func:`full_kwargs` with placeholder inputs, then drops those input
    slots — leaving exactly the params, defaults filled, ready for a
    :class:`TransformSpec`.

    Args:
        op: the op callable whose signature is the param contract.
        raw_params: the keyword params the caller actually supplied; anything
            omitted is defaulted.
        n_inputs: how many leading positional parameters are upstream inputs.

    >>> def embed(segments, *, input_type='document'): ...
    >>> step_params(embed, {'input_type': 'query'})
    {'input_type': 'query'}
    >>> step_params(embed)                              # default filled in
    {'input_type': 'document'}
    >>> def segment(source): ...
    >>> step_params(segment)                            # no params at all
    {}
    """
    from i2 import Sig

    input_names = [p.name for p in Sig(op).parameters.values()][:n_inputs]
    placeholders = tuple(None for _ in range(n_inputs))
    mapped = full_kwargs(op, placeholders, raw_params)
    return {k: v for k, v in mapped.items() if k not in input_names}


# ---------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """One step of a pipeline — a serializable recipe for an op application.

    A :class:`TransformSpec` names *which op* (``op``, a key resolved through
    the :class:`~ef.artifact_graph.ArtifactGraph`'s ``ops`` registry), *at which
    version* (``op_version`` — it participates in every downstream hash), and
    *with which keyword params* (``params``). It does **not** record the step's
    inputs: those are wired structurally by the :class:`PipelineSpec` (the embed
    step consumes the segment step's output). It is the per-step counterpart of
    a :class:`~ef.artifact_graph.ProducerSpec`.

    The op key carries the *component identity* — e.g. ``"embed:openai:text-
    embedding-3-large@1024"`` or ``"segment:<segmenter-identity>"`` (see
    :mod:`ef.ops`) — so two configs with different components get disjoint
    artifact cones automatically.

    >>> ts = TransformSpec(op='embed:m@8', op_version='2', params={'input_type': 'document'})
    >>> ts.op, ts.op_version
    ('embed:m@8', '2')
    >>> TransformSpec.from_dict(ts.as_dict()) == ts
    True
    """

    op: str
    op_version: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-``dict`` (JSON-serializable) form of the spec.

        >>> TransformSpec(op='segment:r', op_version='1', params={}).as_dict()
        {'op': 'segment:r', 'op_version': '1', 'params': {}}
        """
        return {
            "op": self.op,
            "op_version": self.op_version,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TransformSpec":
        """Rebuild a :class:`TransformSpec` from its :meth:`as_dict` form."""
        return cls(
            op=d["op"],
            op_version=d["op_version"],
            params=dict(d.get("params", {})),
        )


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """A whole indexing pipeline as data — the segment→embed chain.

    A :class:`PipelineSpec` is the declarative description of one *config*: a
    :class:`TransformSpec` for the segmenter and one for the embedder. It is
    fully serializable (:meth:`as_dict` / :meth:`from_dict`) and content-hashed
    by :func:`config_id`. Registering a second :class:`PipelineSpec` that
    differs only in its embed step shares the segment artifacts in the
    :class:`~ef.artifact_graph.ArtifactGraph` — that is config branching, and
    the graph gives it for free.

    The ``segment``→``embed`` ordering *is* the input wiring: the embed step
    consumes the segment step's output. (Phase 5 fixes this two-step shape; a
    later phase may generalize it to an arbitrary step list.)

    >>> seg = TransformSpec(op='segment:r', op_version='1', params={})
    >>> emb = TransformSpec(op='embed:m@8', op_version='1', params={'input_type': 'document'})
    >>> spec = PipelineSpec(segment=seg, embed=emb)
    >>> spec.embed.params['input_type']
    'document'
    >>> PipelineSpec.from_dict(spec.as_dict()) == spec
    True
    """

    segment: TransformSpec
    embed: TransformSpec

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-``dict`` (JSON-serializable) form of the pipeline.

        >>> seg = TransformSpec(op='segment:r', op_version='1', params={})
        >>> emb = TransformSpec(op='embed:m', op_version='1', params={})
        >>> PipelineSpec(segment=seg, embed=emb).as_dict()['segment']['op']
        'segment:r'
        """
        return {"segment": self.segment.as_dict(), "embed": self.embed.as_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PipelineSpec":
        """Rebuild a :class:`PipelineSpec` from its :meth:`as_dict` form."""
        return cls(
            segment=TransformSpec.from_dict(d["segment"]),
            embed=TransformSpec.from_dict(d["embed"]),
        )


def config_id(spec: PipelineSpec) -> ConfigId:
    """Content-addressed id of a :class:`PipelineSpec` — ``sha256`` over its steps.

    The id is ``sha256`` of the canonical (sorted-key) JSON of
    :meth:`PipelineSpec.as_dict`. Two pipelines that describe the same
    computation hash identically; any change to a step's op, version or params
    yields a different id. ``ef`` writes this digest into ``vd`` index metadata
    as ``config_hash``.

    >>> seg = TransformSpec(op='segment:r', op_version='1', params={})
    >>> emb = TransformSpec(op='embed:m@8', op_version='1', params={})
    >>> cid = config_id(PipelineSpec(segment=seg, embed=emb))
    >>> len(cid)
    64
    >>> cid == config_id(PipelineSpec(segment=seg, embed=emb))
    True
    """
    return sha256_hex(canonical_json(spec.as_dict()))
