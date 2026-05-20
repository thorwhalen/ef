"""Tests for :class:`ef.embedders.HashingEmbedder` — the zero-install default.

All tests run offline with no dependency beyond numpy: that is the whole point
of the hashing embedder (see ``thorwhalen/ef#12``). They cover the embedder in
isolation; the ``ingest``-with-no-embedder integration lives in
``test_source_manager.py``.
"""

import numpy as np
import pytest

from ef import Embedder, HashingEmbedder, as_embedder


# ---------------------------------------------------------------------------
# shape, dtype, normalization
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype():
    out = HashingEmbedder()(["alpha", "beta gamma", "delta"])
    assert out.shape == (3, 512)
    assert out.dtype == np.float32


def test_dim_is_respected():
    assert HashingEmbedder(dim=64)(["abc def"]).shape == (1, 64)


def test_rows_are_l2_normalized():
    out = HashingEmbedder()(["the quick brown fox", "a b c d e f"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_empty_text_is_a_zero_vector():
    # an empty / token-free text has no unit direction — zeros is the honest answer
    out = HashingEmbedder()([""])
    assert out.shape == (1, 512)
    assert not out.any()


def test_empty_input_list():
    out = HashingEmbedder()([])
    assert out.shape == (0, 512)


# ---------------------------------------------------------------------------
# determinism — the reason content addressing holds
# ---------------------------------------------------------------------------


def test_deterministic_across_instances():
    # blake2b (not the salted builtin hash()) → identical vectors every run
    a = HashingEmbedder()(["hello world", "foo bar baz"])
    b = HashingEmbedder()(["hello world", "foo bar baz"])
    np.testing.assert_array_equal(a, b)


def test_input_type_is_ignored():
    e = HashingEmbedder()
    as_query = e(["ocean breeze"], input_type="query")
    as_document = e(["ocean breeze"], input_type="document")
    np.testing.assert_array_equal(as_query, as_document)


# ---------------------------------------------------------------------------
# lexical similarity — what the embedder is for
# ---------------------------------------------------------------------------


def test_shared_vocabulary_scores_higher():
    e = HashingEmbedder()
    v = e(["the quick brown fox", "the quick brown dog", "zebras yonder weave"])
    shared = float(v[0] @ v[1])  # 3 of 4 words in common
    disjoint = float(v[0] @ v[2])  # no words in common
    assert shared > disjoint
    assert shared > 0.0


def test_identical_text_is_self_similar():
    e = HashingEmbedder()
    v = e(["ocean breeze"])
    assert round(float(v[0] @ v[0]), 5) == 1.0


def test_bigrams_capture_word_order():
    # same unigrams, different bigrams → distinct vectors under the default range
    v = HashingEmbedder()(["red car", "car red"])
    assert not np.array_equal(v[0], v[1])
    # with unigrams only, word order is invisible
    v1 = HashingEmbedder(ngram_range=(1, 1))(["red car", "car red"])
    np.testing.assert_array_equal(v1[0], v1[1])


def test_sublinear_tf_changes_the_vector():
    text = ["spam spam spam eggs"]  # a repeated word is where the weighting bites
    linear = HashingEmbedder(sublinear_tf=False)(text)
    sublinear = HashingEmbedder(sublinear_tf=True)(text)
    assert not np.array_equal(linear, sublinear)


# ---------------------------------------------------------------------------
# identity — model_id bakes in every vector-affecting parameter
# ---------------------------------------------------------------------------


def test_default_model_id_is_clean():
    assert HashingEmbedder().model_id == "hashing:v1@512"


def test_model_id_encodes_non_default_params():
    assert HashingEmbedder(dim=128).model_id == "hashing:v1@128"
    assert HashingEmbedder(ngram_range=(1, 3)).model_id == "hashing:v1-ng1_3@512"
    assert HashingEmbedder(sublinear_tf=False).model_id == "hashing:v1-tflin@512"
    assert (
        HashingEmbedder(dim=128, ngram_range=(1, 3), sublinear_tf=False).model_id
        == "hashing:v1-ng1_3-tflin@128"
    )


def test_different_params_give_different_model_ids():
    # distinct model_id → disjoint artifact cones in the ArtifactGraph
    assert HashingEmbedder(dim=128).model_id != HashingEmbedder(dim=256).model_id
    assert (
        HashingEmbedder(ngram_range=(1, 1)).model_id
        != HashingEmbedder(ngram_range=(1, 2)).model_id
    )


# ---------------------------------------------------------------------------
# protocol conformance & validation
# ---------------------------------------------------------------------------


def test_satisfies_the_embedder_protocol():
    e = HashingEmbedder()
    assert isinstance(e, Embedder)
    assert e.normalized is True
    assert e.honored_input_types == ()


def test_embed_batch_returns_a_ready_handle():
    handle = HashingEmbedder().embed_batch(["a", "b"])
    assert handle.poll() == "done"
    assert handle.result().shape == (2, 512)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": -8},
        {"ngram_range": (2, 1)},
        {"ngram_range": (0, 2)},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        HashingEmbedder(**kwargs)


# ---------------------------------------------------------------------------
# the as_embedder DI seam
# ---------------------------------------------------------------------------


def test_as_embedder_resolves_the_hashing_string():
    assert isinstance(as_embedder("hashing"), HashingEmbedder)


def test_as_embedder_forwards_kwargs():
    e = as_embedder("hashing", dim=64)
    assert isinstance(e, HashingEmbedder)
    assert e.dim == 64
    assert e.model_id == "hashing:v1@64"
