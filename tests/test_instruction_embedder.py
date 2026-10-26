"""Instruction-conditioned embedding (ef#34): InstructionEmbedder + ST prompts.

No model is downloaded: the sentence-transformers adapter is exercised with a
fake ``SentenceTransformer``-shaped object that records what ``encode`` receives.
"""

import numpy as np
import pytest

from ef import CachedEmbedder, InstructionEmbedder
from ef.embedder_adapters import _SentenceTransformerEmbedder
from ef.embedders import BaseEmbedder, FunctionEmbedder, cache_key


def _recording_embedder(seen, *, model_id="toy@2"):
    def toy(texts):
        seen.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    return FunctionEmbedder(toy, model_id=model_id)


# --- InstructionEmbedder ------------------------------------------------------


def test_prefixes_only_the_instructed_role():
    seen = []
    emb = InstructionEmbedder(_recording_embedder(seen), query_instruction="Q: ")
    emb(["graphs"], input_type="query")
    emb(["a DAG lib"], input_type="document")
    emb(["plain"])
    assert seen == [["Q: graphs"], ["a DAG lib"], ["plain"]]


def test_each_role_gets_its_own_instruction():
    seen = []
    emb = InstructionEmbedder(
        _recording_embedder(seen),
        query_instruction="Q: ",
        document_instruction="D: ",
        classification_instruction="C: ",
        clustering_instruction="K: ",
    )
    for role in ("query", "document", "classification", "clustering"):
        emb(["x"], input_type=role)
    assert seen == [["Q: x"], ["D: x"], ["C: x"], ["K: x"]]
    assert set(emb.honored_input_types) == {
        "query",
        "document",
        "classification",
        "clustering",
    }


def test_honored_input_types_extends_the_inner():
    class _Inner(BaseEmbedder):
        model_id, dim, honored_input_types = "inner@2", 2, ("document",)

        def __call__(self, texts, *, input_type=None, **backend):
            return np.zeros((len(list(texts)), 2), dtype=np.float32)

    emb = InstructionEmbedder(_Inner(), query_instruction="Q: ")
    assert emb.honored_input_types == ("document", "query")


def test_model_id_bakes_in_the_instructions():
    base = _recording_embedder([])
    a = InstructionEmbedder(base, query_instruction="Find packages about: ")
    b = InstructionEmbedder(base, query_instruction="Find packages about: ")
    c = InstructionEmbedder(base, query_instruction="Find modules about: ")
    d = InstructionEmbedder(base, classification_instruction="Find packages about: ")
    assert a.model_id == b.model_id
    assert len({a.model_id, c.model_id, d.model_id, base.model_id}) == 4
    assert a.model_id.startswith(base.model_id + "+instruct:")
    # so a shared cache never mixes vectors made with different instructions
    assert cache_key(a, "x", "query") != cache_key(c, "x", "query")


def test_metadata_is_delegated():
    base = FunctionEmbedder(lambda ts: np.ones((len(ts), 3)), model_id="m@3", dim=3)
    emb = InstructionEmbedder(base, query_instruction="Q: ")
    assert emb.dim == 3
    assert emb.normalized is base.normalized


def test_composes_with_cache():
    seen = []
    emb = InstructionEmbedder(_recording_embedder(seen), query_instruction="Q: ")
    cached = CachedEmbedder(emb, store={}, cache_queries=True)
    v1 = cached(["graphs"], input_type="query")
    v2 = cached(["graphs"], input_type="query")
    assert seen == [["Q: graphs"]]  # second call served from the cache
    np.testing.assert_array_equal(v1, v2)


def test_embed_batch_is_conditioned_too():
    seen = []
    emb = InstructionEmbedder(_recording_embedder(seen), query_instruction="Q: ")
    handle = emb.embed_batch(["graphs"], input_type="query")
    assert handle is not None
    assert seen == [["Q: graphs"]]


def test_input_type_is_forwarded():
    got = {}

    class _Inner(BaseEmbedder):
        model_id, dim = "inner@2", 2

        def __call__(self, texts, *, input_type=None, **backend):
            got["input_type"] = input_type
            return np.zeros((len(list(texts)), 2), dtype=np.float32)

    InstructionEmbedder(_Inner(), query_instruction="Q: ")(["x"], input_type="query")
    assert got["input_type"] == "query"


def test_requires_at_least_one_instruction():
    with pytest.raises(ValueError, match="at least one instruction"):
        InstructionEmbedder(_recording_embedder([]))


# --- sentence-transformers prompts ----------------------------------------------


class _FakeST:
    """Duck-typed stand-in for ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, prompts=None):
        self.prompts = prompts or {}
        self.calls = []

    def get_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        return np.zeros((len(texts), 4), dtype=np.float32)


def _st(model=None, **kwargs):
    return _SentenceTransformerEmbedder(
        "fake-model",
        model=model or _FakeST(),
        normalize=False,
        batch_size=8,
        **kwargs,
    )


def test_st_default_is_unchanged():
    emb = _st()
    assert emb.honored_input_types == ()
    assert emb.model_id == "st:fake-model@4"
    emb(["x"], input_type="query")
    assert "prompt" not in emb._model.calls[0]
    assert "prompt_name" not in emb._model.calls[0]


def test_st_query_prompt_applies_to_queries_only():
    emb = _st(prompts={"query": "query: ", "document": None})
    assert emb.honored_input_types == ("query",)
    emb(["x"], input_type="query")
    emb(["x"], input_type="document")
    emb(["x"])
    calls = emb._model.calls
    assert calls[0]["prompt"] == "query: "
    assert "prompt" not in calls[1] and "prompt" not in calls[2]


def test_st_prompt_names_use_the_models_own_prompts():
    model = _FakeST(prompts={"query": "Instruct: find\nQuery: "})
    emb = _st(model=model, prompt_names={"query": "query"})
    emb(["x"], input_type="query")
    assert model.calls[0]["prompt_name"] == "query"
    assert emb.honored_input_types == ("query",)


def test_st_model_id_bakes_in_the_prompt_text():
    a = _st(prompts={"query": "query: "})
    b = _st(prompts={"query": "search_query: "})
    same_text = _st(
        model=_FakeST(prompts={"q": "query: "}), prompt_names={"query": "q"}
    )
    assert a.model_id != b.model_id
    assert a.model_id.startswith("st:fake-model@4+prompts:")
    # a literal prompt and a model-defined prompt with the same text are the same vectors
    assert a.model_id == same_text.model_id


def test_st_explicit_backend_prompt_wins():
    emb = _st(prompts={"query": "query: "})
    emb(["x"], input_type="query", prompt="custom: ")
    assert emb._model.calls[0]["prompt"] == "custom: "


def test_st_rejects_unknown_prompt_name():
    with pytest.raises(ValueError, match="not defined by model"):
        _st(prompt_names={"query": "nope"})


def test_st_rejects_prompt_and_prompt_name_for_same_role():
    model = _FakeST(prompts={"query": "q: "})
    with pytest.raises(ValueError, match="pick one"):
        _st(model=model, prompts={"query": "q: "}, prompt_names={"query": "query"})


def test_st_rejects_unknown_input_type():
    with pytest.raises(ValueError, match="unknown input_type"):
        _st(prompts={"queries": "q: "})


def test_st_empty_document_prompt_is_an_explicit_no_prompt():
    emb = _st(prompts={"query": "query: ", "document": ""})
    emb(["x"], input_type="document")
    assert emb._model.calls[0]["prompt"] == ""  # overrides a default_prompt_name
    assert emb.honored_input_types == ("query",)


# --- dtype defaults (#40) -----------------------------------------------------


def _capture_st_kwargs(monkeypatch, **call_kwargs):
    import sys
    import types

    seen = {}

    class _Fake:
        def __init__(self, name, **kw):
            seen.update(kw)

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_Fake),
    )
    from ef.embedder_adapters import sentence_transformers_embedder

    sentence_transformers_embedder("m", **call_kwargs)
    return seen


def test_cpu_defaults_to_float32(monkeypatch):
    import torch

    from ef.embedder_adapters import _dtype_key

    seen = _capture_st_kwargs(monkeypatch, device="cpu")
    assert seen["model_kwargs"][_dtype_key()] == "float32"


def test_explicit_dtype_wins(monkeypatch):
    import torch

    from ef.embedder_adapters import _dtype_key

    seen = _capture_st_kwargs(monkeypatch, device="cpu", dtype="bfloat16")
    assert seen["model_kwargs"][_dtype_key()] == "bfloat16"
    seen = _capture_st_kwargs(
        monkeypatch, device="cpu", model_kwargs={"torch_dtype": torch.float16}
    )
    assert seen["model_kwargs"]["torch_dtype"] is torch.float16


def test_non_cpu_keeps_checkpoint_dtype(monkeypatch):
    seen = _capture_st_kwargs(monkeypatch, device="cuda")
    assert "model_kwargs" not in seen


def test_model_kwargs_dtype_key_is_respected(monkeypatch):
    import torch

    seen = _capture_st_kwargs(
        monkeypatch, device="cpu", model_kwargs={"dtype": torch.float16}
    )
    assert seen["model_kwargs"] == {"dtype": torch.float16}


def test_auto_device_defers_to_sentence_transformers(monkeypatch):
    import sys
    import types

    util = types.SimpleNamespace(get_device_name=lambda: "xpu")
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util)
    assert "model_kwargs" not in _capture_st_kwargs(monkeypatch)


def test_dtype_auto_is_passed_through(monkeypatch):
    from ef.embedder_adapters import _dtype_key

    seen = _capture_st_kwargs(monkeypatch, device="cpu", dtype="auto")
    assert seen["model_kwargs"] == {_dtype_key(): "auto"}


@pytest.mark.parametrize("version,key", [("4.55.9", "torch_dtype"), ("4.56.0", "dtype")])
def test_dtype_key_follows_transformers_version(monkeypatch, version, key):
    import transformers

    from ef.embedder_adapters import _dtype_key

    monkeypatch.setattr(transformers, "__version__", version)
    assert _dtype_key() == key


def test_auto_device_cpu_gets_float32(monkeypatch):
    import sys
    import types

    from ef.embedder_adapters import _dtype_key

    util = types.SimpleNamespace(get_device_name=lambda: "cpu")
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util)
    seen = _capture_st_kwargs(monkeypatch)
    assert seen["model_kwargs"] == {_dtype_key(): "float32"}
