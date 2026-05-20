"""Tests for ``ef.explore`` — the layer-L5 project / cluster / label surface.

The numpy-only paths (PCA projection, k-means clustering) are tested directly.
UMAP, HDBSCAN and ``imbed`` are optional: those paths are ``importorskip``-
guarded, exactly as the ``ragas`` path is in ``test_evaluation.py``.
"""

import numpy as np
import pytest

import ef
from ef.explore import (
    _choose_projection,
    _kmeans,
    _l2_normalize,
    _pca,
    _resolve_vectors,
    _segment_text,
)


# ---------------------------------------------------------------------------
# Fixtures — three well-separated blobs so clustering is unambiguous
# ---------------------------------------------------------------------------


@pytest.fixture
def blobs():
    """45 vectors in 3 separated Gaussian blobs in 8-D."""
    rng = np.random.RandomState(0)
    centers = np.array([[5.0] * 8, [-5.0] * 8, [0.0] * 4 + [5.0] * 4])
    return np.vstack([rng.randn(15, 8) * 0.3 + c for c in centers])


# ---------------------------------------------------------------------------
# project()
# ---------------------------------------------------------------------------


def test_project_pca_shape(blobs):
    coords = ef.project(blobs, dims=2, method="pca")
    assert coords.shape == (45, 2)
    assert coords.dtype == float


def test_project_pca_dims_3(blobs):
    assert ef.project(blobs, dims=3, method="pca").shape == (45, 3)


def test_project_pca_is_deterministic(blobs):
    a = ef.project(blobs, dims=2, method="pca")
    b = ef.project(blobs, dims=2, method="pca")
    np.testing.assert_array_equal(a, b)


def test_project_from_texts():
    coords = ef.project(["alpha beta", "gamma delta", "epsilon zeta"], method="pca")
    assert coords.shape == (3, 2)


def test_project_needs_two_samples():
    with pytest.raises(ValueError, match=">= 2 samples"):
        ef.project(np.zeros((1, 8)), method="pca")


def test_project_rejects_bad_dims(blobs):
    with pytest.raises(ValueError, match="dims must be >= 1"):
        ef.project(blobs, dims=0)


def test_project_auto_falls_back_to_pca_when_tiny():
    """Too few samples for UMAP -> PCA, with a warning."""
    with pytest.warns(UserWarning, match="using PCA"):
        coords = ef.project(np.random.RandomState(1).rand(3, 8), method="auto")
    assert coords.shape == (3, 2)


def test_project_from_searchable_corpus():
    idx = ef.ingest(["cats and dogs", "machine learning", "neural networks", "kittens"])
    coords = ef.project(idx, dims=2, method="pca")
    assert coords.shape == (4, 2)


def test_project_umap_path(blobs):
    """The real PCA -> UMAP recipe — skipped unless umap-learn is importable."""
    pytest.importorskip("umap")
    coords = ef.project(blobs, dims=2, method="umap", random_state=42)
    assert coords.shape == (45, 2)


# ---------------------------------------------------------------------------
# cluster()
# ---------------------------------------------------------------------------


def test_cluster_kmeans_recovers_blobs(blobs):
    labels = ef.cluster(blobs, method="kmeans", n_clusters=3, random_state=0)
    assert labels.shape == (45,)
    assert set(labels.tolist()) == {0, 1, 2}
    # each of the three contiguous 15-vector blobs lands in one cluster
    for start in (0, 15, 30):
        block = labels[start : start + 15]
        assert len(set(block.tolist())) == 1


def test_cluster_kmeans_is_deterministic(blobs):
    a = ef.cluster(blobs, method="kmeans", n_clusters=3, random_state=0)
    b = ef.cluster(blobs, method="kmeans", n_clusters=3, random_state=0)
    np.testing.assert_array_equal(a, b)


def test_cluster_kmeans_clamps_n_clusters():
    """More clusters requested than samples -> at most n_samples labels."""
    labels = ef.cluster(np.random.RandomState(0).rand(4, 8), n_clusters=10)
    assert len(set(labels.tolist())) <= 4


def test_cluster_without_normalize(blobs):
    labels = ef.cluster(blobs, method="kmeans", n_clusters=3, normalize=False)
    assert labels.shape == (45,)


def test_cluster_rejects_unknown_method(blobs):
    with pytest.raises(ValueError, match="unknown clustering method"):
        ef.cluster(blobs, method="spectral")


def test_cluster_hdbscan_path(blobs):
    """HDBSCAN — skipped unless the hdbscan package is installed."""
    pytest.importorskip("hdbscan")
    labels = ef.cluster(blobs, method="hdbscan", min_cluster_size=5)
    assert labels.shape == (45,)
    assert labels.dtype == int


# ---------------------------------------------------------------------------
# label_clusters()
# ---------------------------------------------------------------------------


def test_label_clusters_length_mismatch():
    with pytest.raises(ValueError, match="differ in length"):
        ef.label_clusters(["a", "b", "c"], [0, 1])


def test_label_clusters_wires_imbed(monkeypatch):
    """label_clusters builds the (segment, cluster_idx) frame and int-keys the result.

    The LLM-backed ``imbed.tools.ClusterLabeler`` is replaced by a stub, so the
    wiring is tested without a network call or an API key.
    """
    pd = pytest.importorskip("pandas")
    seen = {}

    class StubLabeler:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def label_clusters(self, frame):
            seen["columns"] = list(frame.columns)
            seen["rows"] = len(frame)
            return {cid: f"cluster-{cid}" for cid in frame["cluster_idx"].unique()}

    monkeypatch.setattr(
        ef.explore, "_import_cluster_labeler", lambda: (pd, StubLabeler)
    )
    titles = ef.label_clusters(
        ["neural nets", "kittens", "gradient descent", "puppies"],
        np.array([0, 1, 0, 1]),
        context="ML and pets",
        n_words=3,
    )
    assert titles == {0: "cluster-0", 1: "cluster-1"}
    assert all(isinstance(k, int) for k in titles)
    assert seen["columns"] == ["segment", "cluster_idx"]
    assert seen["rows"] == 4
    assert seen["kwargs"]["context"] == "ML and pets"
    assert seen["kwargs"]["n_words"] == 3


def test_label_clusters_accepts_segment_mappings(monkeypatch):
    pd = pytest.importorskip("pandas")

    class StubLabeler:
        def __init__(self, **kwargs):
            pass

        def label_clusters(self, frame):
            return dict.fromkeys(frame["cluster_idx"].unique(), "x")

    monkeypatch.setattr(
        ef.explore, "_import_cluster_labeler", lambda: (pd, StubLabeler)
    )
    segments = [{"text": "hi", "id": "1"}, {"text": "yo", "id": "2"}]
    titles = ef.label_clusters(segments, [0, 0])
    assert titles == {0: "x"}


# ---------------------------------------------------------------------------
# _resolve_vectors() — the corpus | vectors -> ndarray seam
# ---------------------------------------------------------------------------


def test_resolve_vectors_ndarray_passthrough():
    array = np.random.RandomState(0).rand(5, 4)
    np.testing.assert_array_equal(_resolve_vectors(array), array)


def test_resolve_vectors_rejects_1d_array():
    with pytest.raises(ValueError, match="2-D array"):
        _resolve_vectors(np.zeros(8))


def test_resolve_vectors_nested_lists():
    resolved = _resolve_vectors([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert resolved.shape == (3, 2)


def test_resolve_vectors_embeds_texts():
    resolved = _resolve_vectors(["hello world", "goodbye world"])
    assert resolved.ndim == 2
    assert resolved.shape[0] == 2


def test_resolve_vectors_corpus_mapping():
    resolved = _resolve_vectors({"a": "first source", "b": "second source"})
    assert resolved.shape[0] == 2


def test_resolve_vectors_empty():
    with pytest.raises(ValueError, match="empty corpus"):
        _resolve_vectors([])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_pca_zero_pads_when_rank_deficient():
    """Requesting more components than exist pads with zero columns."""
    vectors = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    scores = _pca(vectors, n_components=4)
    assert scores.shape == (3, 4)
    assert np.allclose(scores[:, 1:], 0.0)


def test_l2_normalize_unit_rows_and_safe_on_zero():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]])
    normed = _l2_normalize(vectors)
    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    assert np.allclose(normed[1], 0.0)  # zero row stays zero, no divide-by-zero


def test_kmeans_single_cluster(blobs):
    labels = _kmeans(blobs, n_clusters=1, random_state=0)
    assert set(labels.tolist()) == {0}


def test_choose_projection_explicit_methods():
    assert _choose_projection("pca", n_samples=100) == "pca"
    assert _choose_projection("umap", n_samples=100) == "umap"


def test_choose_projection_rejects_unknown():
    with pytest.raises(ValueError, match="unknown projection method"):
        _choose_projection("tsne", n_samples=100)


def test_segment_text_handles_str_and_mapping():
    assert _segment_text("plain") == "plain"
    assert _segment_text({"text": "wrapped", "id": "1"}) == "wrapped"
