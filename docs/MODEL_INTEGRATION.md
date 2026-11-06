# Adding a model to CDRSeqGen

[Back to docs](README.md) · [Model specifications](MODELS.md) ·
[Method workflow](../cdrseqgen/methods/common/README.md) ·
[Benchmark adapters](../benchmark/adapters/README.md)

Implementation reference: **6 November 2026**. IgLM demonstrates a dedicated
model-family adapter: reuse the upstream architecture and preserve its native
tokenizer, infilling format, checkpoint loading and sampling behavior.

## Choose the integration level

| What is changing? | Where it belongs |
|---|---|
| A compatible model ID within an already supported family or fallback | Select it through `model_type`; verify tokenization, input layout and output/loss contracts. |
| A model family with custom loading, tokenization or decoding, within an existing generation approach | Add `methods/<method>/models/<family>.py` and a registration in that method's `backbones.py`. |
| An objective, data view or generation/evaluation procedure that the existing method contracts cannot serve | Add a method package and register it in `methods/registry.py`; provide its CLI, dataset and execution contracts. |
| An independent generator needed only for benchmark comparison | Add a benchmark adapter and external registration; native training integration is a separate choice. |

One family module can support several sizes and checkpoints, as `iglm.py` supports
both IgLM and IgLM-S. A different architecture alone does not require a new method.

MLM has a generic `hf_encoder` fallback; AR has `hf_decoder`. Matching an ID does
not establish compatibility. The generic AR path may add missing infilling tokens
and resize embeddings; the pretrained model still needs to learn that format.
S2S currently has no generic full encoder-decoder fallback: its pretrained support
combines supported PLM encoders with a new residue decoder.

## Keep the responsibilities separate

```text
config.model_type
  -> methods/<method>/backbones.py: select a registered family
  -> models/<family>.py: construct/load the tokenizer and model
  -> existing method training, evaluation and generation workflows
```

- **Model module:** architecture/factory for a custom model, or an adapter around an
  existing implementation; family-specific source checks, tokenization and native protocol.
- **Method registry:** matching predicates, loader wiring, input conventions and validation.
- **Common code:** registry mechanics, reusable loading helpers and shared lifecycles.
- **Method workflow:** objective, dataset/label alignment, generation and scoring strategy.

The `models/` directory holds integration code. Checkpoint weights are selected by
source ID/local path and stored separately as model inputs or saved run artifacts.

Keep heavy optional imports inside their loaders when practical. Registration and
family matching should not instantiate models or load checkpoint weights.

## Hooks for a dedicated model module

The two AR modules [transformer.py](../cdrseqgen/methods/ar/models/transformer.py)
and [iglm.py](../cdrseqgen/methods/ar/models/iglm.py) use these names:

| Hook | Responsibility |
|---|---|
| `load_tokenizer(source, position_type, uses_special_tokens)` | Return a tokenizer adapter with the required vocabulary and input conventions. |
| `load_training_model(source, **options)` | Construct fresh weights or load pretrained weights for fine-tuning. |
| `load_pretrained_model(source)` | Restore a checkpoint for evaluation/generation. |
| `validate_training_args(model_args, data_args)` | Optional family checks before run creation, including unsupported context or input budgets. |

Training and inference may share one loader, as IgLM does. A custom model may need
separate construction and strict checkpoint restoration paths.

This is the convention for dedicated adapters, not a claim that every existing
MLM/S2S module already has the same organization. Some loaders still live in their
method's `backbones.py`. The shared requirement is the
[Backbone descriptor contract](../cdrseqgen/methods/common/backbones.py).

## Inference contracts

- Default token samplers require residue-compatible tokenization and logits shaped
  `[batch, sequence, vocabulary]`. Dictionaries, `.logits` outputs and `(logits, ...)`
  tuples are supported when no labels are passed to the model.
- EOS samplers return one record per attempt, preserving order, duplicates and
  failures. `loop` is a string; `complete`, and optional `valid`/`terminated`, are
  coherent booleans. Contradictory records are rejected before counting.
- Self/judge scorers return one scalar or `None` per candidate in the same order.
  Wrong counts fail explicitly; NaN/Inf scores become unavailable before means and
  coverage are computed. This also applies to supplied-case self-scoring.
- Declare context capacities through the tokenizer/model metadata. MLM checks masks,
  wrappers and reserved lengths before forwarding. Its length head converts frozen
  backbone features to its own device/dtype, including FP16/BF16 inputs.
- Both standalone commands validate numeric sampling options before model loading.
  Native adapters retain additional family checks, such as IgLM's prompt format,
  single-chain context and decoding policy.

## Register the family

For example, place this **illustrative** block in AR's `backbones.py`, before its
generic `hf_decoder` fallback. It assumes a new `models/new_family.py` with a cheap
`matches()` predicate and the hooks above; that example module does not exist yet.

```python
def _family():
    from cdrseqgen.methods.ar.models import new_family
    return new_family


register(CausalBackbone(
    id="new_family",
    matches=lambda source: _family().matches(source),
    position_type="absolute",       # declare the family's actual input conventions
    uses_special_tokens=False,
    load_tokenizer=lambda source, pos, wrap: _family().load_tokenizer(source, pos, wrap),
    load_training_model=lambda source, **kw: _family().load_training_model(source, **kw),
    load_pretrained_model=lambda path: _family().load_pretrained_model(path),
    validate_model_args=lambda model, data: _family().validate_training_args(model, data),
))
```

The constructor field is **`validate_model_args`**; the descriptor's public method
and the model-module hook are named **`validate_training_args`**. Omit the optional
validator when the family needs no additional checks.

The registry takes the first matching entry, requires unique IDs, and keeps any
generic fallback last. Matching must cover intended source IDs and saved checkpoints
so restored runs reach the same family. `position_type` and `uses_special_tokens`
describe tokenizer/input-layout conventions; they do not configure model positional embeddings.
The CLI files continue calling the same shared workflows.

## Preserve native protocols

IgLM needs more integration code than an ordinary compatible HF model because it
preserves a native vocabulary, chain/species conditioning, span layout, source-format
recognition and sampling restrictions. Future adapters implement only what their family needs.

For native AR infilling:

- Attach a `span_layout` to the tokenizer adapter. Prompt construction, target labels
  and the answer terminator must agree between training and generation.
- The current answer contract is the original target IDs followed by exactly one
  terminator; layout overhead counts prompt markers plus that terminator. Wrapped or
  reordered answers need a dataset-contract extension, beyond a custom sampler.
- If decoding differs, attach `causal_sampler` and `causal_sampling_metadata`;
  preserve the fixed attempt budget, duplicates and failures.
- Persist vocabulary, layout and sampling metadata with saved checkpoints. Reject
  incompatible recorded contracts rather than silently changing their interpretation.
- Use the optional registration field `native_checkpoint` only with a format-recognition
  predicate for original releases lacking CDRSeqGen layout metadata. Missing metadata
  alone is insufficient; format recognition does not authenticate weight bytes.
- Reject unsupported partner/antigen context or input budgets before a run starts.

Use IgLM's [implementation](../cdrseqgen/methods/ar/models/iglm.py) and
[documented contract](MODELS.md#pretrained-ar-backbone-iglm) as a worked example,
with the method's own conventions taking precedence over copied defaults.

## Verify through the public workflows

Use tiny synthetic CPU fixtures to check tokenizer IDs, prompt/label alignment,
loss and self-scoring, a real parameter update, and save/reload agreement. Exercise
the method's train/evaluate/generate entry points and every requested attempt,
including empty, invalid and incomplete draws. Native adapters also need parity
checks against their original implementation under matched inputs/settings.

Existing examples live in `test_common_backbones`, `test_model_roundtrip`,
`test_training_pipeline_contracts`, `test_span_layout` and the `test_iglm*` modules;
see the [test guide](../unit_tests/README.md). Tiny CPU checks do not establish
published-checkpoint/GPU correctness; the real-checkpoint smoke/replay stage remains user-run.

Benchmark integration has its own capability contract. The current benchmark runner
serves scratch checkpoints for all three methods and pretrained AR/S2S; it rejects
pretrained MLM arms at planning time. Verify that path separately before treating a new backbone as
a comparable benchmark arm. Benchmark-only externals follow the
[adapter guide](../benchmark/adapters/README.md) and
[external registry](../benchmark/externals.py), potentially in their own environment.
