# Custom Transformer specifications

[Back to docs](README.md) · [Architecture](ARCHITECTURE.md) ·
[Arguments](ARGUMENTS.md) · [Method workflow](../cdrseqgen/methods/common/README.md)

Implementation snapshot: **3 November 2026**. This reference describes the current
from-scratch MLM, AR and S2S models and distinguishes builder defaults from the
saved configurations used in the main validation benchmark. S2S's
[PLM-backed composite](../cdrseqgen/methods/s2s/models/plm.py) is a separate model.

## AR model-module interface

[transformer.py](../cdrseqgen/methods/ar/models/transformer.py) and
[iglm.py](../cdrseqgen/methods/ar/models/iglm.py) expose the same registry hooks:

| Hook | Custom Transformer | IgLM |
|---|---|---|
| `load_tokenizer(source, position_type, uses_special_tokens)` | Fixed causal vocabulary and default span layout | Native vocabulary, conditioning, span layout and sampler |
| `load_training_model(source, **options)` | Create fresh weights through `build_model()` | Restore weights through `load_model()` for fine-tuning |
| `load_pretrained_model(source)` | Check layout and strictly restore custom weights | Same strict native loader used for fine-tuning |
| `validate_training_args(model_args, data_args)` | Check architecture dimensions | Check supported context and checkpoint capacity |

The registry selects a module and calls these hooks lazily; model-specific construction and
restoration live in the model module. Existing `transformer.build_model()` and `iglm.load_model()`
remain available. IgLM retains standard Hugging Face model outputs; the custom model returns
its dictionary contract. Their module docstrings use Components, Flow and Contract sections.

## Current implementation and builder defaults

Sources: [shared components](../cdrseqgen/methods/common/transformer.py),
[MLM](../cdrseqgen/methods/mlm/models/transformer.py),
[AR](../cdrseqgen/methods/ar/models/transformer.py),
[S2S](../cdrseqgen/methods/s2s/models/transformer.py).

The common module supplies reusable components. Each custom implementation defines
its config, model and `build_model()` factory; the factory constructs `Config → Model`.

| Specification | Common components | MLM | AR | S2S |
|---|---|---|---|---|
| Structure | Reusable validation/layers/helpers | Bidirectional encoder | Causal decoder | Encoder + decoder |
| Self-attention | Rotary; causal or bidirectional | PyTorch TransformerEncoder | Shared causal attention | Shared bidirectional encoder / causal decoder attention |
| Cross-attention | Method-owned | — | — | Decoder → encoder; no rotary positions |
| Positions | Rotary helpers | Learned absolute table | Rotary | Rotary on self-attention |
| Normalization | Method-owned | Post-norm; no extra final encoder norm | Pre-norm + final norm | Pre-norm + final norm per stream |
| FFN activation | GELU | ReLU, from the PyTorch layer default | Shared GELU | Shared GELU |
| Dropout | Attention 0.0; no FFN dropout | 0.1 attention/FFN/residual dropout | None | None |
| Default depth | No model stack | 4 encoder blocks | 4 causal blocks | 4 encoder + 4 decoder blocks |
| Default vocabulary | Method-owned | 25 | 27 | 27 |
| Output projection | Method-owned | Linear with bias | Linear without bias | Linear without bias |
| Output/embedding weight tying | Method-owned | Untied | Untied | Untied; encoder/decoder share one input embedding |
| Label alignment | Method-owned | Same-position reconstruction | Internal next-token shift | Dataset already offsets decoder targets |
| Shared helpers used | Provides helpers | Validation + initialization | Validation + initialization + attention + FFN | Same as AR |
| Default sequence-model parameters | No standalone model | **865,049** | **800,256** | **1,858,816** |

These counts use builder defaults: **hidden size 128, 8 heads, FFN width 512** and
the depths above. The builders resolve `intermediate_size=None` to
`4 × hidden_size`. MLM allocates `max(max_len, 512)` learned position slots; the
default count includes 512. AR/S2S compute rotary positions from sequence length
with `rope_theta=10000` and have no learned position table.

The shared initializer sets Linear/Embedding weights to normal std 0.02 and
linear biases to zero. MLM's fused attention input projection retains PyTorch's
initialization; its position table starts at zero. Sharing the helper therefore
does not make every parameter's initialization identical across architectures.
Normalization, activation, dropout and capacity differences remain part of the
implemented baselines.

## Forward and loss contracts

All three return dictionaries with `loss` and `logits`; logits have shape
`[batch, sequence, vocabulary]`. Loss is None without labels. Optional hidden
states expose the final representation; S2S can also expose encoder states and
reuse them through `encode()`. `return_dict` is accepted for HF compatibility;
these custom implementations always return dictionaries.

```text
MLM: logits[t] -> labels[t]       supervised masked positions only
AR:  logits[t] -> labels[t + 1]   shift performed inside the model
S2S: decoder input  [BOS, A, C]
     target labels [A,   C, EOS]  already offset by the dataset; no model shift
```

Labels equal to -100 are excluded from loss. MLM training uses
[MLMLossTrainer](../cdrseqgen/methods/mlm/losses.py) for its configured external
objective; the model's own CE supports direct calls. AR/S2S normally return their
own CE. With a Trainer-supplied `num_items_in_batch`, they divide summed token
loss by the supervised-token count across the accumulation window; direct calls
and evaluation use the token mean.

## Main validation benchmark: saved architecture settings

All **21 scratch checkpoints** (seeds 42–48 for each method) were checked against
their saved run/model configs and safetensors shape headers. All use **hidden size
128, 8 heads, FFN width 512 and input cap max_len=384**. The recorded
`max_position_embeddings` is 512.

| System | Saved stack depths | Sequence parameters | Length-head parameters | Total system parameters |
|---|---|---:|---:|---:|
| MLM | 4 encoder blocks | 865,049 | 4,900 | **869,949** |
| AR | 4 causal blocks | 800,256 | 0 | **800,256** |
| S2S | 2 encoder + 2 decoder blocks | 933,120 | 0 | **933,120** |

Sources: [original checkpoint recipe](../benchmark/recipes/oas4_h3_length_free.json),
[additional-seed plan](../results/benchmark/20261009_weekend_seeds/plan.json) and
[saved capacity export](../results/benchmark/oas4_h3_followup_20261008/capacity.json).
Saved result links refer to locally retained research artifacts.

Use these saved settings for the main benchmark's architecture/capacity table.
The default S2S count in the first table is a 4+4 model; the benchmark used 2+2.
MLM's sequence-model count excludes its separately fitted length head; report
both counts for length-free generation. The seven-seed analysis is linked from
the [results inventory](RESULTS.md#completed-validation-evidence).

Equal `max_len` values do not imply identical context budgets: MLM includes its
mask slots, AR reserves answer space in the same stream, and S2S caps the encoder
stream separately. Report input layout and length policy alongside the cap.

## Pretrained AR backbone: IgLM

The AR registry also holds the released IgLM models, adapted by
[ar/models/iglm.py](../cdrseqgen/methods/ar/models/iglm.py). They are pretrained GPT-2
language models, not custom Transformers, so nothing above describes their layers.

| Release | Width | Layers | Heads | FFN | Positions | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| IgLM | 512 | 4 | 8 | 2048 | 512 | 12,889,600 |
| IgLM-S | 192 | 3 | 6 | 768 | 512 | 1,439,616 |

Both use the 33-token vocabulary of the release (ids unchanged: padding, unknown, `[CLS]`,
`[SEP]`, `[MASK]`, 20 residues, six species, `[HEAVY]`, `[LIGHT]`). The weights load unchanged;
no token is added and no embedding is resized.

**Format.** `[chain][species] left [MASK] right [SEP]` is the prompt and the loop plus `[CLS]` is
the answer (span layout `iglm_infill`, five layout tokens). The chain token comes from
`target_region` (`h*` → `[HEAVY]`, `l*` → `[LIGHT]`); the species is the record's `species` or
`species_token`, `[HUMAN]` when absent. Generation takes the region from `--target_loop`, else the
checkpoint's, and a test-file case may carry its own `target_region`, `species` or
`species_token`; the chain and species tokens used are recorded in each generated case.

**Selecting it.** `model_type` is `iglm:/path/to/IgLM-S`, or `IgLM` / `IgLM-S` when the `iglm`
package is installed in the same environment (nothing is downloaded). A checkpoint saved by a
CDRSeqGen run is recognised from its config. The unmodified release has no CDRSeqGen metadata, so
it is accepted without layout metadata only when its config and vocabulary match a recognized
release signature. This recognizes the format, not the authenticity of its weights.
`evaluate.py --config` can score it zero-shot; the config must declare `method: ar`.
For a release without `run_config.json`, use `generate.py --method ar --model_type iglm:/path/to/IgLM-S`
and supply `--target_loop h3` (or explicit chain information in the case JSON).
A saved CDRSeqGen run retains its declared method; `--method` cannot override it.
In the benchmark, a fine-tuned IgLM run is a local
arm like any pretrained AR checkpoint (the adapter passes the manifest's chain and species tokens);
the released weights are scored by the separate external adapter, an independent implementation.

**Limits.** Single-chain flanks only: partner-chain and antigen context are rejected rather than
ignored. `max_len` must lie in [101, 512] (96-residue answer reserve, five layout tokens, 512
positions), and the flanks are capped to `max_len − 101` tokens. H3 flanks in the OAS data are 107
residues (median; maximum 112), so `max_len` 192 would truncate every heavy-chain prompt and 256 is
safe (213 is the minimum for the longest).

**Sampling** follows the fixed-attempt protocol shared by AR: every attempt is kept, duplicates
and invalid draws included. It keeps IgLM's decoding restrictions (padding, unknown and
conditioning tokens banned; `[MASK]`, `[SEP]` and `[CLS]` stay reachable and are checked
afterwards) but differs from the upstream convenience wrapper in two ways: no refill to a set of
unique sequences, and no forced `[CLS]` at a length cap (an unfinished draw counts as
incomplete). The wrapper's implicit `top_k` of 50 does not matter: it never truncates a
33-token vocabulary (same-seed draws are identical with `top_k` 50 and 0). The external
benchmark adapter (`benchmark/adapters/iglm.py`) is a separate implementation and is unchanged.

The native sampler isolates its generation configuration, so saved repetition penalties,
minimum lengths and output modes cannot silently change this protocol. Both training and
inference check recorded layout/policy metadata and reject missing learned weights. Original
GPT-2 attention-mask buffers are the only permitted legacy extra keys. The tokenizer loader
supports the `vocab_file`/`vocab` constructor difference between Transformers 4 and 5.
Generated/evaluated artifacts record the protocol, actual conditioning and every attempt,
including malformed terminated outputs separately from missing terminators. Pretrained
benchmark arms require their saved tokenizer and hash its files and generation configuration.

**Checked on the released weights** (CPU, 120 test frameworks per release across H3, L3 and H2):
`perplexities` agrees with `exp(-IgLM.log_likelihood)` of the upstream package to a maximum
relative difference of 2.6e-6, and greedy decoding with the upstream `generate` arguments gives
identical token sequences in every case. These are prior implementation checks, not new
comparative results. They used some test-split records; final held-out comparative evaluation
remains pending, but the test split has not been wholly unseen during development. Further
implementation checks should use synthetic or validation examples.

**For the paper.** IgLM was pretrained on OAS and is the §11 baseline. Scoring its released weights
on the shared frameworks is a measurement of that model; fine-tuning them on OAS is not a matched
comparison with the from-scratch arms (different pretraining data and capacity), so report it as a
separate arm and state the pretraining.

## Using this reference in the paper

The Methods section should describe the shared framework and each model's
attention, positional encoding, normalization, activation, dropout and loss
alignment. An architecture table should report the actual run's dimensions,
stack depths, vocabulary, parameter counts and any auxiliary heads.

Keep the training/evaluation protocol alongside that table: corpus/split/context,
masking or decoder policy, optimizer/schedule, batch size and accumulation,
seeds, stopping/checkpoint selection, candidate budget and scoring populations.
The [argument reference](ARGUMENTS.md) explains the fields; saved
`run_config.json`, model `config.json`, source/environment provenance and
auxiliary-head artifacts establish the configuration for a particular result.

The first table records current implementation details, including implicit
PyTorch defaults that model configs do not all persist. Before final publication,
check those details against the source revision and library versions recorded
with the reported runs. The saved architecture snapshot above is not a new
experiment or a replacement for the full protocol, and unequal capacities and
layer choices limit conclusions about an isolated generation-objective effect.
