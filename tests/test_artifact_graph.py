"""Tests for the ``ef`` artifact-graph core (Phase 4).

All tests run offline — no network, no heavy deps — exercising
:func:`~ef.artifact_graph.artifact_id`, :func:`~ef.artifact_graph.producer_spec`,
:class:`~ef.artifact_graph.ProducerSpec` and the
:class:`~ef.artifact_graph.ArtifactGraph` and its four contract operations.
"""

import pytest

from ef import (
    ArtifactGraph,
    ProducerSpec,
    artifact_id,
    producer_spec,
)


# ---------------------------------------------------------------------------
# ProducerSpec & producer_spec
# ---------------------------------------------------------------------------


def test_producer_spec_construction():
    spec = producer_spec("segment", "doc-1", op_version="1", size=512, overlap=64)
    assert spec.op == "segment"
    assert spec.op_version == "1"
    assert spec.inputs == ("doc-1",)
    assert spec.params == {"size": 512, "overlap": 64}


def test_producer_spec_multiple_inputs_keep_order():
    spec = producer_spec("join", "a", "b", "c", op_version="1")
    assert spec.inputs == ("a", "b", "c")


def test_producer_spec_defaults():
    spec = ProducerSpec(op="leaf-op", op_version="1")
    assert spec.inputs == ()
    assert spec.params == {}


def test_producer_spec_is_frozen():
    spec = producer_spec("op", op_version="1")
    with pytest.raises((AttributeError, TypeError)):
        spec.op = "other"  # type: ignore[misc]


def test_producer_spec_equality():
    a = producer_spec("op", "x", op_version="1", k=1)
    b = producer_spec("op", "x", op_version="1", k=1)
    c = producer_spec("op", "x", op_version="2", k=1)
    assert a == b
    assert a != c


def test_producer_spec_as_dict_from_dict_roundtrip():
    spec = producer_spec("embed", "a", "b", op_version="3", dim=256, model="m")
    d = spec.as_dict()
    assert d == {
        "op": "embed",
        "op_version": "3",
        "inputs": ["a", "b"],
        "params": {"dim": 256, "model": "m"},
    }
    assert ProducerSpec.from_dict(d) == spec


def test_producer_spec_from_dict_tolerates_missing_optionals():
    spec = ProducerSpec.from_dict({"op": "x", "op_version": "1"})
    assert spec.inputs == ()
    assert spec.params == {}


# ---------------------------------------------------------------------------
# artifact_id — content addressing
# ---------------------------------------------------------------------------


def test_artifact_id_is_deterministic():
    a = producer_spec("embed", "seg", op_version="1", model="small")
    b = producer_spec("embed", "seg", op_version="1", model="small")
    assert artifact_id(a) == artifact_id(b)


def test_artifact_id_is_a_sha256_hex():
    aid = artifact_id(producer_spec("op", op_version="1"))
    assert len(aid) == 64
    assert all(c in "0123456789abcdef" for c in aid)


def test_artifact_id_param_key_order_does_not_matter():
    a = producer_spec("op", "x", op_version="1", alpha=1, beta=2)
    b = producer_spec("op", "x", op_version="1", beta=2, alpha=1)
    assert artifact_id(a) == artifact_id(b)


def test_artifact_id_input_order_matters():
    a = producer_spec("op", "x", "y", op_version="1")
    b = producer_spec("op", "y", "x", op_version="1")
    assert artifact_id(a) != artifact_id(b)


def test_artifact_id_sensitive_to_every_field():
    base = producer_spec("op", "x", op_version="1", k=1)
    base_id = artifact_id(base)
    assert base_id != artifact_id(producer_spec("OP", "x", op_version="1", k=1))
    assert base_id != artifact_id(producer_spec("op", "x", op_version="2", k=1))
    assert base_id != artifact_id(producer_spec("op", "z", op_version="1", k=1))
    assert base_id != artifact_id(producer_spec("op", "x", op_version="1", k=2))
    assert base_id != artifact_id(producer_spec("op", "x", op_version="1"))


def test_artifact_id_handles_nested_params():
    a = producer_spec("op", op_version="1", cfg={"a": 1, "b": 2})
    b = producer_spec("op", op_version="1", cfg={"b": 2, "a": 1})
    assert artifact_id(a) == artifact_id(b)  # nested keys canonicalized too


# ---------------------------------------------------------------------------
# ArtifactGraph — put, add, materialize
# ---------------------------------------------------------------------------


def test_put_stores_a_leaf():
    graph = ArtifactGraph()
    assert graph.put("doc", "the text") == "doc"
    assert graph.materialize("doc") == "the text"
    assert graph.freshness("doc") == "materialized"


def test_add_returns_content_addressed_id():
    graph = ArtifactGraph()
    spec = producer_spec("op", "leaf", op_version="1")
    assert graph.add(spec) == artifact_id(spec)


def test_add_is_idempotent():
    graph = ArtifactGraph()
    a = graph.add(producer_spec("op", "leaf", op_version="1"))
    b = graph.add(producer_spec("op", "leaf", op_version="1"))
    assert a == b
    assert len(graph.producers) == 1


def test_materialize_computes_a_chain():
    graph = ArtifactGraph()
    graph.register_op("upper", str.upper)
    graph.register_op("repeat", lambda s, *, times: s * times)
    graph.put("src", "ab")
    shout = graph.add(producer_spec("upper", "src", op_version="1"))
    echo = graph.add(producer_spec("repeat", shout, op_version="1", times=3))
    assert graph.materialize(echo) == "ABABAB"


def test_materialize_memoizes():
    graph = ArtifactGraph()
    calls = []
    graph.register_op("count", lambda x: calls.append(x) or x)
    graph.put("n", 7)
    out = graph.add(producer_spec("count", "n", op_version="1"))
    graph.materialize(out)
    graph.materialize(out)
    assert calls == [7]  # second call was a cache hit


def test_materialize_computes_shared_upstream_once():
    # Config branching: two downstream artifacts share one upstream node.
    graph = ArtifactGraph()
    seg_calls = []
    graph.register_op("segment", lambda doc: seg_calls.append(doc) or f"seg({doc})")
    graph.register_op("embed", lambda seg, *, model: f"{model}:{seg}")
    graph.put("doc", "DOC")
    segs = graph.add(producer_spec("segment", "doc", op_version="1"))
    cfg_a = graph.add(producer_spec("embed", segs, op_version="1", model="small"))
    cfg_b = graph.add(producer_spec("embed", segs, op_version="1", model="large"))
    assert graph.materialize(cfg_a) == "small:seg(DOC)"
    assert graph.materialize(cfg_b) == "large:seg(DOC)"
    assert seg_calls == ["DOC"]  # the shared segment op ran exactly once


def test_materialize_missing_leaf_raises_keyerror():
    graph = ArtifactGraph()
    graph.register_op("op", lambda x: x)
    out = graph.add(producer_spec("op", "absent-leaf", op_version="1"))
    with pytest.raises(KeyError, match="no cached value"):
        graph.materialize(out)


def test_materialize_unknown_id_raises_keyerror():
    graph = ArtifactGraph()
    with pytest.raises(KeyError):
        graph.materialize("nothing-here")


def test_materialize_unregistered_op_raises_lookuperror():
    graph = ArtifactGraph()
    graph.put("leaf", "v")
    out = graph.add(producer_spec("not-registered", "leaf", op_version="1"))
    with pytest.raises(LookupError, match="not-registered"):
        graph.materialize(out)


def test_materialize_detects_a_corrupted_cycle():
    # A content-addressed graph cannot cycle; a hand-corrupted registry can.
    graph = ArtifactGraph()
    graph.register_op("op", lambda x: x)
    graph.producers["A"] = ProducerSpec(op="op", op_version="1", inputs=("B",))
    graph.producers["B"] = ProducerSpec(op="op", op_version="1", inputs=("A",))
    with pytest.raises(ValueError, match="[Cc]ycle"):
        graph.materialize("A")


# ---------------------------------------------------------------------------
# descendants / ancestors
# ---------------------------------------------------------------------------


def _diamond():
    """A diamond: src -> (a, b) -> sink.  Returns (graph, a, b, sink)."""
    graph = ArtifactGraph()
    graph.register_op("op", lambda *xs, **kw: xs)  # **kw absorbs the `branch` tag
    graph.put("src", "s")
    a = graph.add(producer_spec("op", "src", op_version="1", branch="a"))
    b = graph.add(producer_spec("op", "src", op_version="1", branch="b"))
    sink = graph.add(producer_spec("op", a, b, op_version="1"))
    return graph, a, b, sink


def test_descendants_is_the_transitive_downstream_cone():
    graph, a, b, sink = _diamond()
    assert graph.descendants("src") == {a, b, sink}
    assert graph.descendants(a) == {sink}
    assert graph.descendants(sink) == frozenset()


def test_descendants_of_unknown_key_is_empty():
    assert ArtifactGraph().descendants("nope") == frozenset()


def test_ancestors_is_the_transitive_lineage():
    graph, a, b, sink = _diamond()
    assert graph.ancestors(sink) == {a, b, "src"}
    assert graph.ancestors(a) == {"src"}
    assert graph.ancestors("src") == frozenset()


# ---------------------------------------------------------------------------
# mark_stale
# ---------------------------------------------------------------------------


def test_mark_stale_drops_cached_values_of_the_produced_cone():
    graph, a, b, sink = _diamond()
    graph.materialize(sink)
    invalidated = graph.mark_stale("src")
    assert invalidated == {a, b, sink}  # the leaf 'src' is not "stale"
    assert graph.freshness("src") == "materialized"  # leaf value kept
    for key in (a, b, sink):
        assert graph.freshness(key) == "stale"


def test_mark_stale_keeps_recipes_so_materialize_rebuilds():
    graph = ArtifactGraph()
    graph.register_op("up", str.upper)
    graph.put("s", "hi")
    out = graph.add(producer_spec("up", "s", op_version="1"))
    graph.materialize(out)
    graph.mark_stale("s")
    assert out in graph.producers  # recipe survived
    assert graph.materialize(out) == "HI"  # rebuilt


def test_mark_stale_never_drops_a_leaf_value():
    graph = ArtifactGraph()
    graph.put("leaf", "precious")
    assert graph.mark_stale("leaf") == frozenset()  # nothing produced downstream
    assert graph.materialize("leaf") == "precious"  # leaf intact


# ---------------------------------------------------------------------------
# delete_cascade
# ---------------------------------------------------------------------------


def test_delete_cascade_removes_the_whole_cone():
    graph, a, b, sink = _diamond()
    removed = graph.delete_cascade("src")
    assert removed == {"src", a, b, sink}
    for key in ("src", a, b, sink):
        assert key not in graph


def test_delete_cascade_keeps_upstream_and_detaches_edges():
    graph, a, b, sink = _diamond()
    removed = graph.delete_cascade(a)
    assert removed == {a, sink}  # a and its only descendant
    assert "src" in graph  # upstream survives
    assert b in graph  # the sibling branch survives
    # 'src' no longer lists the deleted 'a' as a dependent, but still lists 'b'
    assert graph.descendants("src") == {b}


def test_delete_cascade_of_unknown_key_removes_nothing():
    graph = ArtifactGraph()
    assert graph.delete_cascade("ghost") == frozenset()


def test_delete_cascade_then_freshness_is_unknown():
    graph = ArtifactGraph()
    graph.register_op("op", lambda x: x)
    graph.put("s", "v")
    out = graph.add(producer_spec("op", "s", op_version="1"))
    graph.delete_cascade("s")
    assert graph.freshness(out) == "unknown"


# ---------------------------------------------------------------------------
# freshness
# ---------------------------------------------------------------------------


def test_freshness_states():
    graph = ArtifactGraph()
    graph.register_op("op", lambda x: x)
    graph.put("leaf", "v")
    out = graph.add(producer_spec("op", "leaf", op_version="1"))
    assert graph.freshness("leaf") == "materialized"
    assert graph.freshness(out) == "stale"
    assert graph.freshness("unknown-id") == "unknown"
    graph.materialize(out)
    assert graph.freshness(out) == "materialized"


# ---------------------------------------------------------------------------
# dunders
# ---------------------------------------------------------------------------


def test_contains_and_len():
    graph = ArtifactGraph()
    graph.put("leaf", "v")
    out = graph.add(producer_spec("op", "leaf", op_version="1"))
    assert "leaf" in graph
    assert out in graph
    assert "missing" not in graph
    assert len(graph) == 2


def test_repr_reports_counts():
    graph = ArtifactGraph()
    graph.register_op("op", lambda x: x)
    graph.put("leaf", "v")
    text = repr(graph)
    assert "ArtifactGraph" in text
    assert "1 op(s)" in text


# ---------------------------------------------------------------------------
# dependency injection — swappable backends
# ---------------------------------------------------------------------------


def test_stores_are_injectable():
    store: dict = {}
    producers: dict = {}
    graph = ArtifactGraph(store=store, producers=producers)
    graph.register_op("op", lambda x: x)
    graph.put("leaf", "v")
    out = graph.add(producer_spec("op", "leaf", op_version="1"))
    graph.materialize(out)
    assert store["leaf"] == "v"  # the caller's own dict was used
    assert out in producers


def test_resumed_graph_rebuilds_edges_from_producers():
    # Persist only the recipes; on reload the reverse index is derived.
    original = ArtifactGraph()
    original.register_op("op", lambda x: x)
    original.put("src", "v")
    out = original.add(producer_spec("op", "src", op_version="1"))

    resumed = ArtifactGraph(producers=dict(original.producers))
    assert resumed.descendants("src") == {out}  # edges were rebuilt


def test_injected_ops_registry():
    graph = ArtifactGraph(ops={"op": str.upper})
    graph.put("s", "hi")
    out = graph.add(producer_spec("op", "s", op_version="1"))
    assert graph.materialize(out) == "HI"


# ---------------------------------------------------------------------------
# end-to-end: refresh / config-branching scenario
# ---------------------------------------------------------------------------


def test_refresh_scenario_old_source_cascade_deleted_new_one_materialized():
    """A source edit: delete the cone of the old content hash, build the new."""
    graph = ArtifactGraph()
    graph.register_op("embed", lambda text, *, model: f"{model}({text})")

    # Index v1 of a source (keyed by its content hash — here a stand-in).
    graph.put("hash-v1", "old text")
    vec_v1 = graph.add(producer_spec("embed", "hash-v1", op_version="1", model="m"))
    assert graph.materialize(vec_v1) == "m(old text)"

    # The source changes -> a new content hash -> the old cone is now garbage.
    graph.delete_cascade("hash-v1")
    assert vec_v1 not in graph

    # Index v2.
    graph.put("hash-v2", "new text")
    vec_v2 = graph.add(producer_spec("embed", "hash-v2", op_version="1", model="m"))
    assert graph.materialize(vec_v2) == "m(new text)"


def test_config_branching_shares_upstream_artifacts():
    """Two embedder configs over one corpus share every segment artifact."""
    graph = ArtifactGraph()
    segment_runs = []
    graph.register_op(
        "segment", lambda doc: segment_runs.append(doc) or [doc[:2], doc[2:]]
    )
    graph.register_op("embed", lambda segs, *, model: [f"{model}:{s}" for s in segs])

    graph.put("doc", "abcd")
    segs = graph.add(producer_spec("segment", "doc", op_version="1"))

    # Config A and Config B differ only in the embedder model.
    cfg_a = graph.add(producer_spec("embed", segs, op_version="1", model="A"))
    cfg_b = graph.add(producer_spec("embed", segs, op_version="1", model="B"))
    assert cfg_a != cfg_b  # divergent leaves

    graph.materialize(cfg_a)
    graph.materialize(cfg_b)
    assert segment_runs == ["abcd"]  # segmentation shared, ran once
