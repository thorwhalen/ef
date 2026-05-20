"""Tests for the ``ef`` embedder facade (Phase 1).

All tests run offline — no network, no torch — using stub embedders built on
:class:`ef.embedders.FunctionEmbedder`.
"""

import numpy as np
import pytest

from ef import (
    BaseEmbedder,
    BatchHandle,
    CachedEmbedder,
    Embedder,
    EmbedderError,
    FunctionEmbedder,
    MultiEmbedder,
    NormalizingEmbedder,
    RetryPolicy,
    RetryingEmbedder,
    as_embedder,
    cache_key,
    embed_length_sorted,
    http_embedder,
    openai_embedder,
    ready_handle,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _counting_embedder(dim=4):
    """A FunctionEmbedder plus the list of every text it was asked to embed."""
    seen = []

    def func(texts):
        seen.extend(texts)
        return np.array([[float(len(t))] * dim for t in texts], dtype=np.float32)

    return FunctionEmbedder(func, model_id=f"count@{dim}"), seen


# --------------------------------------------------------------------------
# FunctionEmbedder & the protocol
# --------------------------------------------------------------------------


def test_function_embedder_numpy_output():
    e = FunctionEmbedder(lambda ts: np.ones((len(ts), 3)), model_id="ones@3")
    out = e(["a", "b"])
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    assert e.dim == 3


def test_function_embedder_accepts_nested_lists():
    e = FunctionEmbedder(lambda ts: [[1, 2], [3, 4]], model_id="lst@2")
    out = e(["x", "y"])
    assert out.shape == (2, 2)
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_function_embedder_infers_dim_lazily():
    e = FunctionEmbedder(lambda ts: np.zeros((len(ts), 7)), model_id="z")
    assert e.dim is None
    e(["one"])
    assert e.dim == 7


def test_function_embedder_rejects_wrong_count():
    e = FunctionEmbedder(lambda ts: np.zeros((1, 3)), model_id="bad")
    with pytest.raises(EmbedderError):
        e(["a", "b"])


def test_function_embedder_rejects_dim_mismatch():
    e = FunctionEmbedder(lambda ts: np.zeros((len(ts), 3)), model_id="m", dim=5)
    with pytest.raises(EmbedderError):
        e(["a"])


def test_empty_input_returns_empty_array():
    e = FunctionEmbedder(lambda ts: np.zeros((len(ts), 3)), model_id="m", dim=3)
    out = e([])
    assert out.shape == (0, 3)


def test_protocol_isinstance():
    e, _ = _counting_embedder()
    assert isinstance(e, Embedder)
    # a bare function is NOT an Embedder (no metadata attrs)
    assert not isinstance(lambda ts: ts, Embedder)


# --------------------------------------------------------------------------
# BatchHandle
# --------------------------------------------------------------------------


def test_ready_handle():
    h = ready_handle(np.zeros((2, 3)))
    assert isinstance(h, BatchHandle)
    assert h.poll() == "done"
    assert h.result().shape == (2, 3)


def test_base_embedder_default_embed_batch_is_synchronous():
    e, _ = _counting_embedder()
    handle = e.embed_batch(["a", "b"])
    assert handle.poll() == "done"
    assert handle.result().shape == (2, 4)


# --------------------------------------------------------------------------
# length-sorted batching
# --------------------------------------------------------------------------


def test_embed_length_sorted_preserves_order():
    def enc(batch):
        return np.array([[len(t)] for t in batch], dtype=np.float32)

    texts = ["aaa", "a", "aaaaa", "aa", "aaaa"]
    out = embed_length_sorted(texts, enc, batch_size=2)
    assert out.ravel().tolist() == [3.0, 1.0, 5.0, 2.0, 4.0]


def test_embed_length_sorted_empty():
    out = embed_length_sorted([], lambda b: np.zeros((0, 3)))
    assert out.shape == (0, 0)


# --------------------------------------------------------------------------
# cache_key
# --------------------------------------------------------------------------


def test_cache_key_deterministic_and_namespaced():
    e, _ = _counting_embedder()
    k1 = cache_key(e, "hello", "document")
    k2 = cache_key(e, "hello", "document")
    assert k1 == k2
    assert k1.startswith(e.model_id + "/")


def test_cache_key_distinguishes_input_type_and_text():
    e, _ = _counting_embedder()
    assert cache_key(e, "x", "query") != cache_key(e, "x", "document")
    assert cache_key(e, "x", "query") != cache_key(e, "y", "query")


# --------------------------------------------------------------------------
# CachedEmbedder
# --------------------------------------------------------------------------


def test_cached_embedder_serves_repeats_from_store():
    inner, seen = _counting_embedder()
    cached = CachedEmbedder(inner, store={})
    first = cached(["a", "b"])
    second = cached(["b", "c"])
    assert seen == ["a", "b", "c"]  # 'b' embedded exactly once
    np.testing.assert_array_equal(first[1], second[0])  # same vector for 'b'


def test_cached_embedder_skips_query_by_default():
    inner, seen = _counting_embedder()
    cached = CachedEmbedder(inner, store={})
    cached(["q"], input_type="query")
    cached(["q"], input_type="query")
    assert seen == ["q", "q"]  # not cached


def test_cached_embedder_caches_query_when_opted_in():
    inner, seen = _counting_embedder()
    cached = CachedEmbedder(inner, store={}, cache_queries=True)
    cached(["q"], input_type="query")
    cached(["q"], input_type="query")
    assert seen == ["q"]


def test_cached_embedder_delegates_metadata():
    inner, _ = _counting_embedder(dim=4)
    inner(["x"])  # resolve lazy dim
    cached = CachedEmbedder(inner, store={})
    assert cached.model_id == inner.model_id
    assert cached.dim == 4
    assert isinstance(cached, Embedder)


# --------------------------------------------------------------------------
# RetryingEmbedder
# --------------------------------------------------------------------------


class _Boom(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_retrying_embedder_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky(texts):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _Boom(503)
        return np.ones((len(texts), 2))

    inner = FunctionEmbedder(flaky, model_id="flaky@2")
    retrying = RetryingEmbedder(
        inner, RetryPolicy(max_attempts=5, base_delay=0.0, jitter=0.0), sleep=lambda d: None
    )
    out = retrying(["a"])
    assert out.shape == (1, 2)
    assert attempts["n"] == 3


def test_retrying_embedder_does_not_retry_client_error():
    attempts = {"n": 0}

    def always_400(texts):
        attempts["n"] += 1
        raise _Boom(400)

    inner = FunctionEmbedder(always_400, model_id="bad@2")
    retrying = RetryingEmbedder(inner, sleep=lambda d: None)
    with pytest.raises(_Boom):
        retrying(["a"])
    assert attempts["n"] == 1  # 400 is not retryable


def test_retrying_embedder_gives_up_after_max_attempts():
    attempts = {"n": 0}

    def always_503(texts):
        attempts["n"] += 1
        raise _Boom(503)

    inner = FunctionEmbedder(always_503, model_id="down@2")
    retrying = RetryingEmbedder(
        inner, RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0.0), sleep=lambda d: None
    )
    with pytest.raises(_Boom):
        retrying(["a"])
    assert attempts["n"] == 3


# --------------------------------------------------------------------------
# MultiEmbedder
# --------------------------------------------------------------------------


def test_multi_embedder_routes_by_predicate():
    short = FunctionEmbedder(lambda ts: np.zeros((len(ts), 2)), model_id="s@2")
    long = FunctionEmbedder(lambda ts: np.ones((len(ts), 2)), model_id="l@2")
    multi = MultiEmbedder(
        {"s": short, "l": long}, predicate=lambda t: "s" if len(t) < 4 else "l"
    )
    out = multi(["hi", "hello", "ab", "world"])
    assert out.tolist() == [[0, 0], [1, 1], [0, 0], [1, 1]]


def test_multi_embedder_rejects_mismatched_dims():
    a = FunctionEmbedder(lambda ts: np.zeros((len(ts), 2)), model_id="a@2", dim=2)
    b = FunctionEmbedder(lambda ts: np.zeros((len(ts), 3)), model_id="b@3", dim=3)
    with pytest.raises(EmbedderError):
        MultiEmbedder({"a": a, "b": b}, predicate=lambda t: "a")


def test_multi_embedder_unknown_route_raises():
    a = FunctionEmbedder(lambda ts: np.zeros((len(ts), 2)), model_id="a@2")
    multi = MultiEmbedder({"a": a}, predicate=lambda t: "missing")
    with pytest.raises(EmbedderError):
        multi(["x"])


# --------------------------------------------------------------------------
# NormalizingEmbedder
# --------------------------------------------------------------------------


def test_normalizing_embedder_yields_unit_vectors():
    inner = FunctionEmbedder(
        lambda ts: np.array([[3.0, 4.0]] * len(ts)), model_id="raw@2"
    )
    norm = NormalizingEmbedder(inner)
    out = norm(["a", "b"])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], rtol=1e-6)
    assert norm.normalized is True
    assert norm.model_id == inner.model_id


# --------------------------------------------------------------------------
# as_embedder
# --------------------------------------------------------------------------


def test_as_embedder_passthrough():
    e, _ = _counting_embedder()
    assert as_embedder(e) is e


def test_as_embedder_wraps_callable():
    e = as_embedder(lambda ts: np.ones((len(ts), 5)), model_id="ones@5")
    assert isinstance(e, FunctionEmbedder)
    assert e(["a", "b"]).shape == (2, 5)


def test_as_embedder_url_builds_http_embedder_without_calling():
    e = as_embedder("http://localhost:8080/embed")
    assert isinstance(e, Embedder)
    assert e.model_id == "http:http://localhost:8080/embed"


def test_as_embedder_rejects_unknown_type():
    with pytest.raises(TypeError):
        as_embedder(42)


def test_http_embedder_is_an_embedder():
    e = http_embedder("https://example.com/embed", dim=8)
    assert isinstance(e, Embedder)
    assert e.dim == 8


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def test_wrappers_compose():
    inner, seen = _counting_embedder(dim=2)
    stacked = CachedEmbedder(NormalizingEmbedder(inner), store={})
    out1 = stacked(["a", "b"])
    out2 = stacked(["a", "b"])
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_allclose(np.linalg.norm(out1, axis=1), [1.0, 1.0], rtol=1e-6)
    assert seen == ["a", "b"]  # second call fully cached
    assert isinstance(stacked, Embedder)


# --------------------------------------------------------------------------
# OpenAI adapter — exercised against a fake client (no network)
# --------------------------------------------------------------------------


class _FakeOpenAIClient:
    """Minimal stand-in for ``openai.OpenAI`` — embeds text as ``[len(t)] * dim``."""

    def __init__(self):
        self.embeddings = self._Embeddings()
        self.files = self._Files()
        self.batches = self._Batches(self.files)

    @staticmethod
    def _vec(text, dim):
        return [float(len(text))] * dim

    class _Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Embeddings:
        def create(self, *, model, input, dimensions=None, **kw):
            dim = dimensions or 1536
            data = [
                _FakeOpenAIClient._Obj(
                    index=i, embedding=_FakeOpenAIClient._vec(t, dim)
                )
                for i, t in enumerate(input)
            ]
            return _FakeOpenAIClient._Obj(data=data)

    class _Files:
        def __init__(self):
            self._store = {}
            self._n = 0

        def create(self, *, file, purpose):
            self._n += 1
            fid = f"file-{self._n}"
            self._store[fid] = file.read()
            return _FakeOpenAIClient._Obj(id=fid)

        def put(self, content):
            self._n += 1
            fid = f"file-{self._n}"
            self._store[fid] = content
            return fid

        def content(self, fid):
            return _FakeOpenAIClient._Obj(text=self._store[fid])

    class _Batches:
        def __init__(self, files):
            self._files = files
            self._jobs = {}
            self._n = 0

        def create(self, *, input_file_id, endpoint, completion_window):
            self._n += 1
            jid = f"batch-{self._n}"
            # immediately "complete" the job: compute its output file
            import json

            out_lines = []
            for line in self._files._store[input_file_id].decode().splitlines():
                req = json.loads(line)
                body = req["body"]
                dim = body.get("dimensions", 1536)
                data = [
                    {"index": i, "embedding": _FakeOpenAIClient._vec(t, dim)}
                    for i, t in enumerate(body["input"])
                ]
                out_lines.append(
                    json.dumps(
                        {
                            "custom_id": req["custom_id"],
                            "response": {"body": {"data": data}},
                        }
                    )
                )
            out_fid = self._files.put("\n".join(out_lines))
            self._jobs[jid] = out_fid
            return _FakeOpenAIClient._Obj(id=jid)

        def retrieve(self, jid):
            return _FakeOpenAIClient._Obj(
                id=jid, status="completed", output_file_id=self._jobs[jid]
            )

        def cancel(self, jid):  # pragma: no cover - not exercised
            pass


def test_openai_embedder_sync_call():
    e = openai_embedder("text-embedding-3-small", dim=8, client=_FakeOpenAIClient())
    assert e.model_id == "openai:text-embedding-3-small@8"
    out = e(["aa", "bbbb"])
    assert out.shape == (2, 8)
    assert out[0].tolist() == [2.0] * 8
    assert out[1].tolist() == [4.0] * 8


def test_openai_embedder_batch_api():
    e = openai_embedder(
        "text-embedding-3-small", dim=8, client=_FakeOpenAIClient(), batch_size=2
    )
    handle = e.embed_batch(["aa", "bbbb", "c"])  # 3 texts -> 2 chunks
    assert handle.poll() == "done"
    out = handle.result()
    assert out.shape == (3, 8)
    assert out[0].tolist() == [2.0] * 8
    assert out[1].tolist() == [4.0] * 8
    assert out[2].tolist() == [1.0] * 8  # second chunk, index 0


def test_openai_embedder_rejects_dim_on_non_mrl_model():
    with pytest.raises(EmbedderError):
        openai_embedder("text-embedding-ada-002", dim=512, client=_FakeOpenAIClient())
