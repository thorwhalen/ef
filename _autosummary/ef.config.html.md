# ef.config

The config layer — declarative, content-addressable pipeline specs.

A *pipeline* in `ef` is not an object graph you wire by hand — it is **data**.
A [`PipelineSpec`](#ef.config.PipelineSpec) declares the segment→embed chain; each step is a
[`TransformSpec`](#ef.config.TransformSpec) (an op key, an op version, and the keyword params that
feed the content hash). Because a spec is fully serializable and content-hashed
([`config_id()`](#ef.config.config_id)), two configs that share a step share that step’s artifacts
in the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) for free — config branching
costs nothing beyond the divergent cone (design notes §3.1).

This module is the *bridge* between an ordinary Python call and a content-
addressed graph node. For content addressing to be **correct**, two
semantically-equal calls must produce identical params — so `op(text)` and
`op(text, size=512)` (`512` being the default) must hash the same. The
canonical `i2` idiom does exactly that normalization:

- [`full_kwargs()`](#ef.config.full_kwargs) — the raw recipe (design notes §3.8): a call `(args,
  kwargs)` → a fully-named, defaults-filled kwargs dict.
- [`step_params()`](#ef.config.step_params) — its spec-building application: the canonical *keyword*
  params of an op, with the leading positional *input* slots (the upstream
  artifact ids) dropped, since those become [`ProducerSpec.inputs`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec.inputs), not params.

Phase 4’s [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) takes [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec)s as given; this module (and
[`ef.ops`](ef.ops.html.md#module-ef.ops) / [`ef.source_manager`](ef.source_manager.html.md#module-ef.source_manager)) is where ordinary calls *become*
those specs.

Example — two pipelines, equal segment step, divergent embed step:

```pycon
>>> seg = TransformSpec(op='segment:recursive', op_version='1', params={})
>>> embed_a = TransformSpec(op='embed:m-a', op_version='1', params={})
>>> embed_b = TransformSpec(op='embed:m-b', op_version='1', params={})
>>> pa = PipelineSpec(segment=seg, embed=embed_a)
>>> pb = PipelineSpec(segment=seg, embed=embed_b)
>>> config_id(pa) == config_id(pb)        # different embedders → different config
False
>>> config_id(pa) == config_id(PipelineSpec.from_dict(pa.as_dict()))
True
```

### Module Attributes

| [`ConfigId`](#ef.config.ConfigId)   | A content-addressed pipeline identifier — the `sha256` hex digest of a [`PipelineSpec`](#ef.config.PipelineSpec) ([`config_id()`](#ef.config.config_id)).   |
|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`full_kwargs`](#ef.config.full_kwargs)(op[, args, kwargs])         | Normalize a call into a canonical, fully-named, defaults-filled kwargs dict.                                       |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`step_params`](#ef.config.step_params)(op[, raw_params, n_inputs]) | The canonical *keyword* params of an op step — its inputs dropped.                                                 |
| [`config_id`](#ef.config.config_id)(spec)                         | Content-addressed id of a [`PipelineSpec`](#ef.config.PipelineSpec) — `sha256` over its steps. |

### Classes

| [`ConfigId`](#ef.config.ConfigId)                                | A content-addressed pipeline identifier — the `sha256` hex digest of a [`PipelineSpec`](#ef.config.PipelineSpec) ([`config_id()`](#ef.config.config_id)).   |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`TransformSpec`](#ef.config.TransformSpec)(op, op_version[, params]) | One step of a pipeline — a serializable recipe for an op application.                                                                                                                                  |
| [`PipelineSpec`](#ef.config.PipelineSpec)(segment, embed)            | A whole indexing pipeline as data — the segment→embed chain.                                                                                                                                           |

### ef.config.ConfigId

A content-addressed pipeline identifier — the `sha256` hex digest of a
[`PipelineSpec`](#ef.config.PipelineSpec) ([`config_id()`](#ef.config.config_id)). It is what `ef` writes into
`vd` index metadata as `config_hash` so a later phase can tell which
pipeline produced a vector.

### *class* ef.config.PipelineSpec(segment, embed)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A whole indexing pipeline as data — the segment→embed chain.

A [`PipelineSpec`](#ef.config.PipelineSpec) is the declarative description of one *config*: a
[`TransformSpec`](#ef.config.TransformSpec) for the segmenter and one for the embedder. It is
fully serializable ([`as_dict()`](#ef.config.PipelineSpec.as_dict) / [`from_dict()`](#ef.config.PipelineSpec.from_dict)) and content-hashed
by [`config_id()`](#ef.config.config_id). Registering a second [`PipelineSpec`](#ef.config.PipelineSpec) that
differs only in its embed step shares the segment artifacts in the
[`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph) — that is config branching, and
the graph gives it for free.

The `segment``→``embed` ordering *is* the input wiring: the embed step
consumes the segment step’s output. (Phase 5 fixes this two-step shape; a
later phase may generalize it to an arbitrary step list.)

```pycon
>>> seg = TransformSpec(op='segment:r', op_version='1', params={})
>>> emb = TransformSpec(op='embed:m@8', op_version='1', params={'input_type': 'document'})
>>> spec = PipelineSpec(segment=seg, embed=emb)
>>> spec.embed.params['input_type']
'document'
>>> PipelineSpec.from_dict(spec.as_dict()) == spec
True
```

#### as_dict()

Return a plain-`dict` (JSON-serializable) form of the pipeline.

```pycon
>>> seg = TransformSpec(op='segment:r', op_version='1', params={})
>>> emb = TransformSpec(op='embed:m', op_version='1', params={})
>>> PipelineSpec(segment=seg, embed=emb).as_dict()['segment']['op']
'segment:r'
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *classmethod* from_dict(d)

Rebuild a [`PipelineSpec`](#ef.config.PipelineSpec) from its [`as_dict()`](#ef.config.PipelineSpec.as_dict) form.

* **Return type:**
  [`PipelineSpec`](#ef.config.PipelineSpec)

### *class* ef.config.TransformSpec(op, op_version, params=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One step of a pipeline — a serializable recipe for an op application.

A [`TransformSpec`](#ef.config.TransformSpec) names *which op* (`op`, a key resolved through
the [`ArtifactGraph`](ef.artifact_graph.html.md#ef.artifact_graph.ArtifactGraph)’s `ops` registry), \*at which
version\* (`op_version` — it participates in every downstream hash), and
*with which keyword params* (`params`). It does **not** record the step’s
inputs: those are wired structurally by the [`PipelineSpec`](#ef.config.PipelineSpec) (the embed
step consumes the segment step’s output). It is the per-step counterpart of
a [`ProducerSpec`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec).

The op key carries the *component identity* — e.g. `"embed:openai:text-
embedding-3-large@1024"` or `"segment:<segmenter-identity>"` (see
[`ef.ops`](ef.ops.html.md#module-ef.ops)) — so two configs with different components get disjoint
artifact cones automatically.

```pycon
>>> ts = TransformSpec(op='embed:m@8', op_version='2', params={'input_type': 'document'})
>>> ts.op, ts.op_version
('embed:m@8', '2')
>>> TransformSpec.from_dict(ts.as_dict()) == ts
True
```

#### as_dict()

Return a plain-`dict` (JSON-serializable) form of the spec.

```pycon
>>> TransformSpec(op='segment:r', op_version='1', params={}).as_dict()
{'op': 'segment:r', 'op_version': '1', 'params': {}}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *classmethod* from_dict(d)

Rebuild a [`TransformSpec`](#ef.config.TransformSpec) from its [`as_dict()`](#ef.config.TransformSpec.as_dict) form.

* **Return type:**
  [`TransformSpec`](#ef.config.TransformSpec)

### ef.config.config_id(spec)

Content-addressed id of a [`PipelineSpec`](#ef.config.PipelineSpec) — `sha256` over its steps.

The id is `sha256` of the canonical (sorted-key) JSON of
[`PipelineSpec.as_dict()`](#ef.config.PipelineSpec.as_dict). Two pipelines that describe the same
computation hash identically; any change to a step’s op, version or params
yields a different id. `ef` writes this digest into `vd` index metadata
as `config_hash`.

```pycon
>>> seg = TransformSpec(op='segment:r', op_version='1', params={})
>>> emb = TransformSpec(op='embed:m@8', op_version='1', params={})
>>> cid = config_id(PipelineSpec(segment=seg, embed=emb))
>>> len(cid)
64
>>> cid == config_id(PipelineSpec(segment=seg, embed=emb))
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ef.config.full_kwargs(op, args=(), kwargs=None)

Normalize a call into a canonical, fully-named, defaults-filled kwargs dict.

The `i2.Sig` recipe of design notes §3.8. A call expressed as positional
`args` + keyword `kwargs` is mapped onto `op`’s signature: every
parameter is named, and any parameter the call left implicit is filled with
its default. The result is therefore *stable* — `op(x)` and
`op(x, size=512)` (`512` the default) yield the **same** dict, which is
what makes a content hash built from it correct.

* **Parameters:**
  * **op** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the callable whose signature defines the parameter names/defaults.
  * **args** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]) – the call’s positional arguments — for an [`ef.ops`](ef.ops.html.md#module-ef.ops) op these
    are the upstream artifact ids; [`step_params()`](#ef.config.step_params) drops them again.
  * **kwargs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the call’s keyword arguments.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> def embed(segments, *, input_type='document', dim=None): ...
>>> full_kwargs(embed, ('seg-1',), {'input_type': 'query'})
{'segments': 'seg-1', 'input_type': 'query', 'dim': None}
>>> full_kwargs(embed, ('seg-1',))                 # defaults filled in
{'segments': 'seg-1', 'input_type': 'document', 'dim': None}
```

### ef.config.step_params(op, raw_params=None, , n_inputs=1)

The canonical *keyword* params of an op step — its inputs dropped.

An [`ef.ops`](ef.ops.html.md#module-ef.ops) op is called `op(*input_values, **params)`: the first
`n_inputs` positional parameters are upstream artifacts (they belong in
[`ProducerSpec.inputs`](ef.artifact_graph.html.md#ef.artifact_graph.ProducerSpec.inputs), not in
the params), and every remaining parameter is a hashable step param. This
runs [`full_kwargs()`](#ef.config.full_kwargs) with placeholder inputs, then drops those input
slots — leaving exactly the params, defaults filled, ready for a
[`TransformSpec`](#ef.config.TransformSpec).

* **Parameters:**
  * **op** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – the op callable whose signature is the param contract.
  * **raw_params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the keyword params the caller actually supplied; anything
    omitted is defaulted.
  * **n_inputs** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – how many leading positional parameters are upstream inputs.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> def embed(segments, *, input_type='document'): ...
>>> step_params(embed, {'input_type': 'query'})
{'input_type': 'query'}
>>> step_params(embed)                              # default filled in
{'input_type': 'document'}
>>> def segment(source): ...
>>> step_params(segment)                            # no params at all
{}
```
