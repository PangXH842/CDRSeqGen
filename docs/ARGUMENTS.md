# Training arguments

[Back to docs](README.md) · [Method workflow map](../cdrseqgen/methods/common/README.md)

Every training config key and every `train.py` flag is a field of one of three argument groups,
plus Hugging Face's `TrainingArguments` (learning rate, epochs, batch size, evaluation and save
strategy, seed, and so on), which all methods take unchanged. A config JSON names fields directly;
trailing command-line flags override it.

Only explicitly supplied flags replace JSON values. A JSON null stays null, and empty lists
and dictionaries are preserved when an unrelated flag is added. Omitted fields use dataclass
defaults. For example, `{"gen_judge": null, "label_names": []}` plus `--seed 7` keeps the
judge disabled and the empty list intact.

| Group | Defined in | Holds |
|---|---|---|
| Model | `BaseModelArguments`, extended by each method's `ModelArguments` | backbone, scratch architecture, anchors, early stopping, method-specific objective or encoder |
| Data | `BaseDataArguments`, extended by `DataArguments` | corpus, target loop, split, context, input length cap |
| Generation | `BaseGenerationArguments`, extended by `GenerationArguments` | post-training sampling, judge, decoder and length settings |

The base classes live in [common/arguments.py](../cdrseqgen/methods/common/arguments.py); each method adds
only its own fields in `methods/<method>/arguments.py`. The **Where** column says who has a field:
`all` is the shared base, otherwise the method that adds it. Defaults and rules were read from the code;
`unit_tests/test_arguments_contracts.py` fails when a field is missing from this page.

## Model

| Argument | Where | Default | What it does and the rule |
|---|---|---|---|
| `method` | all | `mlm` / `ar` / `s2s` | Each method's own default. `train.py` reads it to choose the package. |
| `model_type` | all | `scratch` | `scratch`, a hub id, or a path. Must exist in that method's backbone registry (checked early). |
| `hidden_size` | all | 128 | Scratch width. In S2S it is also the width of the decoder around a PLM. Must be positive. |
| `num_layers` | all | 4 | Scratch depth. In S2S it is the default stack depth. |
| `num_attention_heads` | all | 8 | Must divide `hidden_size`. AR and S2S also need an even head width (rotary positions). Checked for scratch models, and for every S2S backbone. |
| `intermediate_size` | all | `None` | FFN width. `None` means 4 × `hidden_size`. |
| `anchor_trim` | all | 2 | Residues at each loop end counted as anchors in the core/anchor split. Non-negative integer. |
| `early_stopping_patience` | all | 0 | 0 is off. Above 0 it needs `load_best_model_at_end`. |
| `early_stopping_threshold` | all | 0.0 | Minimum improvement. |
| `loss_type` | mlm | `cross_entropy` | Or `weighted_cross_entropy`. |
| `core_loss_weight` | mlm | 1.0 | Core-residue weight under the weighted loss; only the ratio to the anchor weight matters. |
| `anchor_loss_weight` | mlm | 1.0 | Anchor-residue weight under the weighted loss. |
| `encoder_init` | s2s | `pretrained` | `pretrained` or `random` (a same-architecture control). PLM backbones only. |
| `freeze_encoder` | s2s | `False` | Fixes the PLM encoder's weights and dropout. PLM backbones only. |
| `num_encoder_layers` | s2s | `None` | Scratch encoder depth. Leave unset for PLMs. |
| `num_decoder_layers` | s2s | `None` | Decoder depth. `None` means `num_layers`. |
| `conditioning_probe_steps` | s2s | 0 | Fixed validation conditioning probes. 0 is off. Needs `eval_split=validation`. |
| `diagnostic_stop_epoch` | s2s | 0.0 | Stops at this epoch without shortening the LR schedule. 0 is off. |

AR adds no model arguments.

## Data

| Argument | Where | Default | What it does and the rule |
|---|---|---|---|
| `data_path` | all | `data/processed/processed_oas_multi.json` | Processed corpus JSON. |
| `target_region` | all | `h3` | `h1`–`h3` or `l1`–`l3`. The aliases `cdr1`–`cdr3` and `lcdr1`–`lcdr3` work in any case. Rejected early if unknown. |
| `eval_split` | all | `""` | Named split for early stopping and best-checkpoint choice. Required if the data has a `test` partition. |
| `test_size` | all | 0.1 | Random fallback split for unlabelled data. |
| `max_len` | all | 256 | Cap on the assembled sequence. AR also reserves 96 positions for its answer; S2S caps only the encoder stream. |
| `include_partner_chain` | all | `False` | Adds the paired chain (light for an `h*` target, heavy for `l*`) to the context. |
| `include_antigen` | all | `False` | Adds the antigen sequence to the context. |
| `context_dropout` | all | 0.0 | Fraction of examples dropping each context kind, drawn independently. Between 0 and 1. |
| `masking_policy` | mlm | `full_span` | Or `mixed` (fully masked loops plus partially revealed ones). |
| `mixed_full_span_probability` | mlm | 0.5 | Under `mixed`, the chance that an example still uses the full-span mask. |
| `train_length_predictor` | mlm | `True` | Fits the length head after training. |
| `length_predictor_epochs` | mlm | 20 | Length-head epochs. Must be positive. |
| `length_predictor_learning_rate` | mlm | 0.01 | Length-head learning rate. Must be positive. |
| `length_predictor_batch_size` | mlm | 16 | Length-head batch size. Must be positive. |

AR and S2S add no data arguments.

## Generation

| Argument | Where | Default | What it does and the rule |
|---|---|---|---|
| `run_generative_eval` | all | `True` | Samples and scores candidates for held-out frameworks after training. |
| `gen_eval_context_ablation` | all | `False` | Also evaluates without partner context. Needs a partner chain, generative evaluation and at least one example. |
| `gen_num_examples` | all | 150 | Frameworks to evaluate. `-1` means all. |
| `gen_num_samples` | all | 20 | Candidates per framework. Must be positive. |
| `gen_temperature` | all | 1.0 | Softmax temperature. 0 is greedy. Cannot be negative. |
| `gen_top_k` | all | 0 | Keep only the top-k logits. 0 is off. |
| `gen_top_p` | all | 1.0 | Nucleus cutoff, in (0, 1]. 1.0 is off. |
| `gen_seed` | all | 42 | Sampling seed. |
| `gen_stratify` | all | `False` | Balances the draw across length buckets. |
| `gen_skip_perplexity` | all | `False` | Skips self-scoring (pseudo-perplexity for MLM, perplexity for AR and S2S). |
| `gen_judge` | all | `Exscientia/IgBert` | Independent judge. `""` or JSON `null` disables it. Compare judge scores with the same judge and context; generator self-perplexities are not comparable across models. Sequence metrics need no judge. |
| `gen_judge_partner_context` | all | `trained` | `trained`, `with` or `without`; independent of the generator's context. |
| `gen_judge_path` | all | `None` | Local judge weights, if not the hub id. |
| `gen_decoding` | mlm | `independent` | One pass for all masks, or `iterative` for one residue per pass. An explicit iterative request is allowed for either masking policy. |
| `gen_length_mode` | mlm | `fixed_native` | Or `bounded` or `inferred`. |
| `gen_min_length` | mlm | `None` | Lower length bound for `bounded`. |
| `gen_max_length` | mlm | `None` | Upper length bound for `bounded`. |
| `gen_length_temperature` | mlm | 1.0 | Temperature for sampling the learned length. Cannot be negative. |
| `gen_max_new_tokens` | ar, s2s | 97 | Decode steps including `<eos>`. 97 is the maximum. A draw that never emits `<eos>` within the cap is dropped, not scored. |

## What is checked, and where

Checks run in `setup_args()` before the run directory is created, so a bad config leaves nothing behind.

| Scope | Checks |
|---|---|
| All methods | The config's `method` matches the entrypoint. `target_region` resolves. `context_dropout`, `gen_judge_partner_context`, `anchor_trim`, `gen_num_examples`, `gen_num_samples`, `gen_temperature`, `gen_top_k` and `gen_top_p` are in range. The context ablation has its prerequisites. `early_stopping_patience` is not given without `load_best_model_at_end`. `model_type` is in the method's registry. |
| MLM | The loss settings, `masking_policy`, `mixed_full_span_probability`, the length-head settings, `gen_decoding`, the length policy (`gen_length_mode` with the bounds) and `gen_length_temperature`. Heads divide the width (scratch only). |
| AR | `gen_max_new_tokens` within its limit. Heads divide the width and are of even width (scratch only). `label_smoothing_factor` is rejected. |
| S2S | `gen_max_new_tokens` within its limit. Probe and stop-epoch settings valid. Scratch models reject the encoder controls; PLM backbones validate them. Layer counts are positive integers or unset. Heads divide the width and are of even width (every backbone). `label_smoothing_factor` is rejected. |

Failures that need the data (a `max_len` too small for the target, a mistyped `eval_split`) still surface
after the run directory exists.

## Standalone flags

`evaluate.py` and `generate.py` take the sampling settings as flags rather than `gen_*` fields. The flags
they share are defined once in `add_sampling_arguments()` and carry the training default wherever the
field has one.

| Flag | Training field | Commands |
|---|---|---|
| `--num_samples`, `--temperature`, `--top_k`, `--top_p`, `--seed`, `--skip_perplexity` | `gen_num_samples`, `gen_temperature`, `gen_top_k`, `gen_top_p`, `gen_seed`, `gen_skip_perplexity` | both |
| `--decoding`, `--max_new_tokens` | `gen_decoding`, `gen_max_new_tokens` | both |
| `--min_length`, `--max_length`, `--length_temperature` | `gen_min_length`, `gen_max_length`, `gen_length_temperature` | both |
| `--num_examples`, `--stratify` | `gen_num_examples`, `gen_stratify` | evaluate |
| `--length_mode` | `gen_length_mode` | evaluate |
| `--judge`, `--judge_path`, `--judge_partner_context` | `gen_judge`, `gen_judge_path`, `gen_judge_partner_context` | evaluate |

Three defaults differ on purpose. `--decoding` and `--max_new_tokens` default to `None` and take the value
recorded in the checkpoint's run configuration. `--length_mode` defaults to `None` and then to the method's
own default (`fixed_native` for MLM). `--judge` defaults to no judge, while `gen_judge` defaults to IgBert.
`--partner_context` (evaluate only) has no training field: it forces the generator's context for one
evaluation.
