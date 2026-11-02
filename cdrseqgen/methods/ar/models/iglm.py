"""Adapt released IgLM checkpoints to AR span infilling.

Native format:
  [HEAVY] [HUMAN] left [MASK] right [SEP] target [CLS]
  |---------------- prompt ----------------| |-- answer --|

Entry points (registered by methods/ar/backbones.py):
  * matches / resolve_source -- IgLM, IgLM-S, iglm:/local/path, or a saved IgLM directory
  * is_native_checkpoint     -- the two original releases, which carry no span-layout metadata
  * load_tokenizer           -- the native 33 token ids, with layout, conditioning and sampler attached
  * load_model               -- GPT-2 weights unchanged; native EOS/PAD metadata repaired
  * validate_training_args   -- reject unsupported context and oversized inputs before a run exists

Infilling:
  * resolve_conditioning -- target chain and species tokens (human when omitted)
  * sample_loops         -- IgLM's token restrictions, one draw per attempt, invalid draws kept

Notes:
  * Sampling retains every attempt, including duplicates and invalid outputs. Unlike IgLM's
    convenience wrapper it does not refill a set of unique sequences or force an ending token at
    the cap. (The wrapper's implicit top-k of 50 never truncates a 33-token vocabulary, so top-k
    does not distinguish the two.) This is the fixed-attempt CDRSeqGen infill protocol; the
    external benchmark adapter remains an independent implementation.
  * The released models know only a chain, a species and one gap, so partner-chain and antigen
    context is rejected rather than ignored, and a prompt needs a target chain.
  * Whole-chain generation and whole-chain likelihood are outside this adapter's task contract.
  * IgLM was pretrained on OAS and is the Findings §11 baseline. Scoring its released weights here
    applies this protocol to that model; fine-tuning them on OAS is not a matched comparison with
    the from-scratch arms.
"""
from contextlib import contextmanager
import importlib.util
import json
import math
import inspect
from pathlib import Path

import torch

from cdrseqgen.datasets.dataset_ar import (
    LOOP_LENGTH_RESERVE, SpanLayout, validate_causal_decode_steps,
)
from cdrseqgen.datasets.dataset_common import TARGET_REGIONS, TokenizerAdapter


PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = range(5)
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"
SPECIES_TOKENS = ("[CAMEL]", "[HUMAN]", "[MOUSE]", "[RABBIT]", "[RAT]", "[RHESUS]")
CHAIN_TOKENS = ("[HEAVY]", "[LIGHT]")
NATIVE_VOCAB = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *RESIDUES,
                *SPECIES_TOKENS, *CHAIN_TOKENS)
TOKEN_IDS = {token: index for index, token in enumerate(NATIVE_VOCAB)}
# The species names IgLM's own tokenizer table maps to its six tokens (iglm/model/tokens.py).
SPECIES_NAMES = {
    "Camel": "[CAMEL]", "human": "[HUMAN]", "HIS-mouse": "[MOUSE]", "mouse_Balb/c": "[MOUSE]",
    "mouse_BALB/c": "[MOUSE]", "mouse_C57BL/6": "[MOUSE]", "mouse_C57BL/6J": "[MOUSE]",
    "mouse_Ighe/e": "[MOUSE]", "mouse_Ighg/g": "[MOUSE]", "mouse_Igh/wt": "[MOUSE]",
    "mouse_outbred": "[MOUSE]", "mouse_outbred/C57BL/6": "[MOUSE]",
    "mouse_RAG2-GFP/129Sve": "[MOUSE]", "mouse_Swiss-Webster": "[MOUSE]", "rabbit": "[RABBIT]",
    "rat": "[RAT]", "rat_SD": "[RAT]", "rhesus": "[RHESUS]",
}
# IgLM bans the padding, unknown and conditioning tokens while decoding, and leaves [SEP], [MASK]
# and [CLS] reachable; a repeated marker is caught afterwards, when the draw is validated.
BAD_WORD_IDS = [[PAD_ID], [UNK_ID], *[[i] for i in range(TOKEN_IDS["[CAMEL]"], len(NATIVE_VOCAB))]]
SAMPLING_METADATA = {
    "id": "iglm_infill", "version": 1,
    "bad_token_ids": [entry[0] for entry in BAD_WORD_IDS],
    "forced_eos": False, "collection": "fixed_attempts_with_duplicates",
}
# The two original releases, by the name recorded in their config and their architecture.
RELEASES = {
    "trained_models/IgLM_SM_chain_species_95/checkpoint-400000/": (512, 4, 8, 2048),
    "trained_models/IgLM_XS_chain_species_95/checkpoint-180000/": (192, 3, 6, 768),
}


def _config(root):
    with (Path(root) / "config.json").open() as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"IgLM config must be an object: {root}")
    return config


def _vocab_path(root):
    """The original vocab.txt, beside the weights or (as released) in their parent; else None."""
    root = Path(root)
    for candidate in (root / "vocab.txt", root.parent / "vocab.txt"):
        if candidate.is_file():
            return candidate
    return None


def _vocab_txt_is_native(root):
    path = _vocab_path(root)
    return path is not None and tuple(path.read_text().splitlines()) == NATIVE_VOCAB


def _backend(root):
    """The native-vocabulary BertTokenizerFast: a saved tokenizer if present, else vocab.txt."""
    from transformers import BertTokenizerFast

    root = Path(root)
    if (root / "tokenizer_config.json").is_file():
        backend = BertTokenizerFast.from_pretrained(root, local_files_only=True)
    else:
        if not _vocab_txt_is_native(root):
            raise ValueError(f"IgLM needs a saved tokenizer, or the unchanged native 33-token "
                             f"vocab.txt in {root} or its parent")
        # Transformers 5 renamed vocab_file to vocab; the old keyword can
        # silently construct a five-token default tokenizer in that version.
        argument = "vocab" if "vocab" in inspect.signature(BertTokenizerFast).parameters else "vocab_file"
        backend = BertTokenizerFast(**{argument: str(_vocab_path(root))}, do_lower_case=False)
    if backend.get_vocab() != TOKEN_IDS:
        raise ValueError("IgLM tokenizer changes native vocabulary ids or adds tokens")
    return backend


def _check_config(config):
    """Return the checkpoint's context capacity after confirming it is IgLM-shaped GPT-2."""
    if config.get("model_type") != "gpt2" or config.get("vocab_size") != len(NATIVE_VOCAB):
        raise ValueError("IgLM requires a GPT-2 checkpoint with the native 33-token vocabulary")
    capacity = config.get("n_positions")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("IgLM config must declare a positive n_positions context capacity")
    return capacity


def is_native_checkpoint(path):
    """Recognize the two original releases before accepting absent layout metadata.

    A saved CDRSeqGen checkpoint must retain its recorded layout. Merely sharing
    GPT-2's class and vocabulary is insufficient to claim original provenance.
    """
    root = Path(path)
    try:
        config = _config(root)
    except (OSError, ValueError, TypeError):
        return False
    if not _vocab_txt_is_native(root):
        return False
    if ("cdrseqgen_backbone" in config or "span_layout" in config
            or (root / "span_layout.json").exists() or (root / "run_config.json").exists()):
        return False
    dimensions = RELEASES.get(config.get("_name_or_path"))
    return bool(
        dimensions is not None
        and tuple(config.get(key) for key in ("n_embd", "n_layer", "n_head", "n_inner")) == dimensions
        and config.get("model_type") == "gpt2" and config.get("vocab_size") == len(NATIVE_VOCAB)
        and config.get("n_positions") == 512
        and config.get("condition_on_chain") is True
        and config.get("condition_on_species") is True
        and config.get("bos_token_id") == 50256 and config.get("eos_token_id") == 50256)


def matches(model_type):
    """Recognize explicit names and self-describing local checkpoints without importing IgLM."""
    source = str(model_type)
    if source.lower() in ("iglm", "iglm-s") or source.lower().startswith("iglm:"):
        return True
    try:
        root = Path(source).expanduser()
        if not root.is_dir():
            return False
        return _config(root).get("cdrseqgen_backbone") == "iglm" or is_native_checkpoint(root)
    except (OSError, ValueError, TypeError):
        return False


def resolve_source(source):
    """Return a local checkpoint directory; never download or instantiate IgLM's own wrapper."""
    name = str(source)
    if name.lower() in ("iglm", "iglm-s"):
        try:
            spec = importlib.util.find_spec("iglm")
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            raise FileNotFoundError(
                "IgLM is not installed in this environment. Use model_type='iglm:/absolute/path' "
                "to its local released or converted checkpoint; no download is performed.")
        model_name = "IgLM-S" if name.lower() == "iglm-s" else "IgLM"
        root = Path(next(iter(spec.submodule_search_locations))) / "trained_models" / model_name
    else:
        root = Path(name[5:] if name.lower().startswith("iglm:") else name).expanduser()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise FileNotFoundError(f"IgLM checkpoint directory needs config.json: {root}")
    _check_config(_config(root))
    return str(root.resolve())


def _species_token(value):
    """A native token ([MOUSE], mouse) or one of IgLM's species names; anything else is an error."""
    text = str(value).strip()
    token = SPECIES_NAMES.get(text) or f"[{text.strip('[]').upper()}]"
    if token not in SPECIES_TOKENS:
        raise ValueError(f"IgLM species must be one of {SPECIES_TOKENS} or a known species name, "
                         f"got {value!r}")
    return token


def resolve_conditioning(options):
    """Return native chain/species tokens, rejecting missing or conflicting chain labels.

    target_region=h3 -> [HEAVY]; l3 -> [LIGHT]. An explicit chain_token must agree with it.
    species_token accepts a native token or a species name; omitted means the declared [HUMAN]
    default. No native residues or target length are read.
    """
    region = options.get("target_region")
    inferred = None
    if region:
        try:
            inferred = "[HEAVY]" if TARGET_REGIONS[str(region).lower()][0] == "heavy" else "[LIGHT]"
        except KeyError as exc:
            raise ValueError(f"Unsupported IgLM target_region: {region!r}") from exc
    chain = options.get("chain_token")
    if chain is not None:
        chain = f"[{str(chain).strip().strip('[]').upper()}]"
        if chain not in CHAIN_TOKENS:
            raise ValueError(f"IgLM chain_token must be one of {CHAIN_TOKENS}")
    if chain and inferred and chain != inferred:
        raise ValueError(f"IgLM chain_token {chain} conflicts with target_region={region!r}")
    chain = chain or inferred
    if chain is None:
        raise ValueError("IgLM infilling requires target_region or an explicit chain_token")
    species = options.get("species_token")
    return {"chain_token": chain,
            "species_token": "[HUMAN]" if species is None else _species_token(species)}


def _prompt(tokenizer, left, right, options):
    conditioning = resolve_conditioning(options)
    return [TOKEN_IDS[conditioning["chain_token"]], TOKEN_IDS[conditioning["species_token"]],
            *left, MASK_ID, *right, SEP_ID]


IGLM_SPAN = SpanLayout(
    id="iglm_infill", n_layout_tokens=5, build_prompt=_prompt,
    build_answer=lambda tokenizer, target: [*target, CLS_ID],
    terminator_id=lambda tokenizer: CLS_ID,
    public_fields=("target_region", "chain_token", "species_token"),
)
MIN_MAX_LEN = LOOP_LENGTH_RESERVE + IGLM_SPAN.n_layout_tokens  # fixed answer reserve + layout


def load_tokenizer(source, position_type="absolute", uses_special_tokens=False):
    """Preserve native ids and attach the infilling layout, conditioning and sampling hooks.

    position_type and uses_special_tokens are fixed by the checkpoint and accepted only so the
    signature matches Backbone.load_tokenizer.
    """
    root = resolve_source(source)
    backend = _backend(root)

    def encode(sequence):
        if not isinstance(sequence, str) or any(residue not in RESIDUES for residue in sequence):
            raise ValueError("IgLM expects canonical uppercase amino-acid strings")
        return [TOKEN_IDS[residue] for residue in sequence]

    adapter = TokenizerAdapter(
        encode=encode, decode=lambda ids: backend.decode(ids).replace(" ", ""),
        vocab_size=len(NATIVE_VOCAB), pad_id=PAD_ID, mask_id=MASK_ID, cls_id=None, end_id=None,
        chain_sep_id=SEP_ID, special_ids={*range(5), *range(TOKEN_IDS["[CAMEL]"], len(NATIVE_VOCAB))},
        uses_special_tokens=False, position_type="absolute",
    )
    adapter.eos_id = CLS_ID
    adapter.bos_id = None
    adapter.backend_tokenizer = backend
    adapter.save_pretrained = backend.save_pretrained
    adapter.span_layout = IGLM_SPAN
    adapter.native_context_only = True
    adapter.max_positions = _check_config(_config(root))
    adapter.layout_probe_options = {"chain_token": "[HEAVY]", "species_token": "[HUMAN]"}
    adapter.resolve_conditioning = resolve_conditioning
    adapter.causal_sampler = sample_loops
    adapter.causal_sampling_metadata = dict(SAMPLING_METADATA)
    adapter.checkpoint_source = root
    return adapter


def load_model(source, *, vocab_size=None, max_len=None, **_scratch_options):
    """Load the existing GPT-2 architecture without token additions or embedding resizing.

    The scratch-model options (width, depth, heads) are accepted and ignored, as for any
    pretrained backbone: the checkpoint fixes its own architecture.
    """
    from transformers import GenerationConfig, GPT2LMHeadModel
    from cdrseqgen.methods.ar.backbones import validate_checkpoint_layout

    root = resolve_source(source)
    capacity = _check_config(_config(root))
    if vocab_size is not None and vocab_size != len(NATIVE_VOCAB):
        raise ValueError("IgLM model and tokenizer must both use the native 33-token vocabulary")
    if max_len is not None and max_len > capacity:
        raise ValueError(f"IgLM max_len={max_len} exceeds checkpoint capacity {capacity}")
    # Fine-tuning a saved checkpoint must obey the same contract as inference.
    tokenizer = load_tokenizer(root)
    validate_checkpoint_layout(root, tokenizer, native_checkpoint=is_native_checkpoint(root))
    model, loading = GPT2LMHeadModel.from_pretrained(
        root, local_files_only=True, output_loading_info=True)
    # Older GPT-2 checkpoints also stored attention masks as buffers. Modern
    # models reconstruct them; they are not missing or extra learned weights.
    legacy_buffers = {f"transformer.h.{layer}.attn.{name}"
                      for layer in range(model.config.n_layer)
                      for name in ("bias", "masked_bias")}
    loading["unexpected_keys"] = [key for key in loading.get("unexpected_keys", [])
                                  if key not in legacy_buffers]
    failures = {key: loading.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading.get(key)}
    if failures:
        raise ValueError(f"IgLM checkpoint weights do not match its architecture: {failures}")
    model.config.cdrseqgen_backbone = "iglm"
    model.config.bos_token_id = None
    model.config.eos_token_id = CLS_ID
    model.config.pad_token_id = PAD_ID
    model.config.forced_eos_token_id = None
    # The registered protocol owns generation settings. A saved repetition
    # penalty, minimum length or forced token must not silently change it.
    model.generation_config = GenerationConfig(
        bos_token_id=None, eos_token_id=CLS_ID, pad_token_id=PAD_ID,
        forced_eos_token_id=None)
    model.eval()
    return model


def validate_training_args(model_args, data_args):
    """Reject unsupported context and budgets before a training run is initialized."""
    if data_args.include_partner_chain or data_args.include_antigen:
        raise ValueError("Native IgLM infilling supports single-chain flanks, "
                         "not partner/antigen context")
    capacity = _check_config(_config(resolve_source(model_args.model_type)))
    if not MIN_MAX_LEN <= data_args.max_len <= capacity:
        raise ValueError(f"IgLM max_len must be in [{MIN_MAX_LEN}, {capacity}] for the fixed AR "
                         "answer reserve and the checkpoint's context capacity")


@contextmanager
def _sampling_rng(generator, device):
    """Make Hugging Face generate() follow an explicit generator without moving the global RNG.

    generate() samples from the ambient RNG. One draw from the supplied generator seeds the
    ambient CPU RNG (and the model's CUDA device) inside a forked scope that is restored on exit,
    so the generator advances by exactly that draw and nothing else is disturbed.
    """
    if generator is None:
        yield
        return
    seed = int(torch.randint(0, 2 ** 62, (1,), generator=generator, device=generator.device).item())
    device = torch.device(device)
    cuda = ([device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda):
        torch.default_generator.manual_seed(seed)
        if cuda:
            with torch.cuda.device(cuda[0]):
                torch.cuda.manual_seed(seed)
        yield


def _check_prompt(prompt):
    if (len(prompt) < 4 or prompt[0] not in (TOKEN_IDS[t] for t in CHAIN_TOKENS)
            or prompt[1] not in (TOKEN_IDS[t] for t in SPECIES_TOKENS)
            or prompt.count(MASK_ID) != 1 or prompt.count(SEP_ID) != 1
            or prompt[-1] != SEP_ID or CLS_ID in prompt
            or any(token not in range(5, 25) for token in prompt[2:-1] if token != MASK_ID)):
        raise ValueError("IgLM requires [chain][species] left[MASK]right[SEP] as its infill prompt")


def _draw_record(tokens, prompt, tokenizer):
    """Parse one raw draw: exactly one [MASK], [SEP] and closing [CLS], residues in between."""
    continuation = tokens[len(prompt):]
    terminated = CLS_ID in continuation
    valid_layout = (tokens[:len(prompt)] == prompt and tokens.count(MASK_ID) == 1
                    and tokens.count(SEP_ID) == 1 and tokens.count(CLS_ID) == 1
                    and tokens[-1] == CLS_ID)
    answer = continuation[:-1] if valid_layout else []
    valid = valid_layout and all(token in range(5, 25) for token in answer)
    return {"loop": tokenizer.decode(answer) if valid else "", "length": len(answer),
            "complete": valid, "valid": valid, "terminated": terminated,
            "failure": (None if valid else
                        "invalid_token_layout" if terminated else "missing_terminator"),
            "raw_token_ids": tokens}


@torch.no_grad()
def sample_loops(model, tokenizer, prompt_ids, *, device, num_samples=1, temperature=1.0,
                 top_k=0, top_p=1.0, max_new_tokens=97, generator=None):
    """Make one native-decoder draw per attempt, retaining duplicates and failures.

    [MASK]/[SEP] remain eligible during decoding, as in IgLM. Repeated markers invalidate the
    result afterward. A capped draw must emit its own [CLS]; the cap never fabricates completion.
    Temperature zero returns one greedy draw.
    """
    validate_causal_decode_steps(max_new_tokens)
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    prompt = list(prompt_ids)
    _check_prompt(prompt)
    capacity = getattr(model.config, "n_positions", tokenizer.max_positions)
    if len(prompt) + max_new_tokens > capacity:
        raise ValueError(f"IgLM prompt plus decode cap exceeds checkpoint capacity {capacity}")
    model.eval()
    inputs = torch.tensor([prompt], dtype=torch.long, device=device)
    greedy = temperature == 0
    settings = {"max_new_tokens": max_new_tokens, "pad_token_id": PAD_ID,
                "eos_token_id": CLS_ID, "forced_eos_token_id": None,
                "bad_words_ids": BAD_WORD_IDS, "do_sample": not greedy,
                "num_beams": 1, "num_return_sequences": 1}
    if not greedy:
        settings.update(temperature=temperature, top_k=top_k, top_p=top_p)
    from transformers import GenerationConfig
    generation_config = GenerationConfig(bos_token_id=None, **settings)
    records = []
    previous_config = getattr(model, "generation_config", None)
    had_config = hasattr(model, "generation_config")
    # Transformers 5 fills even an explicit config's None fields from the
    # model's defaults. Scope both sources to this protocol and restore the
    # caller's settings afterwards, including on a failed draw.
    model.generation_config = generation_config
    try:
        with _sampling_rng(generator, device):
            for _ in range(1 if greedy else num_samples):
                output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                        generation_config=generation_config)
                records.append(_draw_record(output[0].detach().cpu().tolist(), prompt, tokenizer))
    finally:
        if had_config:
            model.generation_config = previous_config
        else:
            del model.generation_config
    return records
