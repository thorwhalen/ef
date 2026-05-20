"""Tests for the ``ef`` config layer (Phase 5).

All tests run offline — exercising :func:`~ef.config.full_kwargs`,
:func:`~ef.config.step_params`, :class:`~ef.config.TransformSpec`,
:class:`~ef.config.PipelineSpec` and :func:`~ef.config.config_id`.
"""

import pytest

from ef import PipelineSpec, TransformSpec, config_id, full_kwargs, step_params


# ---------------------------------------------------------------------------
# full_kwargs — the i2.Sig call-hashing recipe
# ---------------------------------------------------------------------------


def _op(segments, *, input_type="document", dim=None):
    """A reference op used to probe signature normalization."""


def test_full_kwargs_names_positional_and_fills_defaults():
    assert full_kwargs(_op, ("seg-1",)) == {
        "segments": "seg-1",
        "input_type": "document",
        "dim": None,
    }


def test_full_kwargs_keyword_overrides_default():
    assert full_kwargs(_op, ("seg-1",), {"input_type": "query"}) == {
        "segments": "seg-1",
        "input_type": "query",
        "dim": None,
    }


def test_full_kwargs_semantically_equal_calls_match():
    # op(x) and op(x, input_type='document') — 'document' is the default
    assert full_kwargs(_op, ("x",)) == full_kwargs(
        _op, ("x",), {"input_type": "document"}
    )


def test_full_kwargs_tolerates_missing_required_and_excess():
    # no positional 'segments' supplied (allow_partial); a junk kwarg (allow_excess)
    out = full_kwargs(_op, (), {"input_type": "query", "junk": 1})
    assert out == {"input_type": "query", "dim": None}


# ---------------------------------------------------------------------------
# step_params — canonical keyword params, inputs dropped
# ---------------------------------------------------------------------------


def test_step_params_drops_the_input_slot():
    assert step_params(_op, {"input_type": "query"}) == {
        "input_type": "query",
        "dim": None,
    }


def test_step_params_fills_defaults():
    assert step_params(_op) == {"input_type": "document", "dim": None}


def test_step_params_no_params_op():
    def segment(source):
        pass

    assert step_params(segment) == {}


def test_step_params_multiple_inputs():
    def join(a, b, *, sep=","):
        pass

    assert step_params(join, n_inputs=2) == {"sep": ","}


# ---------------------------------------------------------------------------
# TransformSpec
# ---------------------------------------------------------------------------


def test_transform_spec_construction():
    ts = TransformSpec(
        op="embed:m@8", op_version="2", params={"input_type": "document"}
    )
    assert ts.op == "embed:m@8"
    assert ts.op_version == "2"
    assert ts.params == {"input_type": "document"}


def test_transform_spec_is_frozen():
    ts = TransformSpec(op="op", op_version="1", params={})
    with pytest.raises((AttributeError, TypeError)):
        ts.op = "other"  # type: ignore[misc]


def test_transform_spec_equality():
    a = TransformSpec(op="op", op_version="1", params={"k": 1})
    b = TransformSpec(op="op", op_version="1", params={"k": 1})
    c = TransformSpec(op="op", op_version="2", params={"k": 1})
    assert a == b
    assert a != c


def test_transform_spec_roundtrip():
    ts = TransformSpec(op="segment:r", op_version="1", params={"x": 2})
    assert TransformSpec.from_dict(ts.as_dict()) == ts


def test_transform_spec_as_dict_is_plain():
    d = TransformSpec(op="o", op_version="1", params={"a": 1}).as_dict()
    assert d == {"op": "o", "op_version": "1", "params": {"a": 1}}


# ---------------------------------------------------------------------------
# PipelineSpec
# ---------------------------------------------------------------------------


def _pipeline(embed_op="embed:m@8"):
    seg = TransformSpec(op="segment:r", op_version="1", params={})
    emb = TransformSpec(op=embed_op, op_version="1", params={"input_type": "document"})
    return PipelineSpec(segment=seg, embed=emb)


def test_pipeline_spec_construction():
    spec = _pipeline()
    assert spec.segment.op == "segment:r"
    assert spec.embed.params == {"input_type": "document"}


def test_pipeline_spec_roundtrip():
    spec = _pipeline()
    assert PipelineSpec.from_dict(spec.as_dict()) == spec


def test_pipeline_spec_is_frozen():
    spec = _pipeline()
    with pytest.raises((AttributeError, TypeError)):
        spec.embed = spec.segment  # type: ignore[misc]


# ---------------------------------------------------------------------------
# config_id
# ---------------------------------------------------------------------------


def test_config_id_is_sha256_hex():
    cid = config_id(_pipeline())
    assert len(cid) == 64
    assert all(c in "0123456789abcdef" for c in cid)


def test_config_id_is_deterministic():
    assert config_id(_pipeline()) == config_id(_pipeline())


def test_config_id_distinguishes_configs():
    # two pipelines that differ only in the embed op → different config ids
    assert config_id(_pipeline("embed:a")) != config_id(_pipeline("embed:b"))


def test_config_id_survives_roundtrip():
    spec = _pipeline()
    assert config_id(spec) == config_id(PipelineSpec.from_dict(spec.as_dict()))
