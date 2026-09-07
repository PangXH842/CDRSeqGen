#!/usr/bin/env bash
#
# Unified hyperparameter sweep for CDRSeqGen. Replaces sweep_h3.sh and sweep_scratch.sh.
#
# Usage:  ./scripts/sweep.sh <gpu> [preset] [base_config]      (default preset: h3)
#
# Presets (one at a time; every dimension is env-overridable, see below):
#   h3       validation-safe screen: 4 LR x 2 warmup x 2 weight-decay x 2 seeds   (32 trials)
#   lr       focused LR screen at the settled warmup/decay, 2 seeds              (6 trials)
#            the screen a new method needs before its first comparison, e.g.
#              ./scripts/sweep.sh 4 lr configs/h3_s2s_scratch.json
#            Fixed 30-epoch scheduler horizon — the budget is deliberately NOT swept, because
#            changing num_train_epochs also reshapes the linear schedule, which would confound
#            "more epochs" with "different schedule".
#   pilot    the 2026-07 question-answering grid: 3 LR x 2 epoch budgets          (6 trials)
#            Kept because it answers a different question — is the budget itself enough? — at
#            the cost of confounding budget with schedule. Use h3 to select hyperparameters.
#   arch     architecture screen: 3 widths x 2 depths, FFN tracks width           (6 trials)
#            Only valid since 2026-07-27, when num_attention_heads / intermediate_size became
#            ModelArguments; before that the FFN ratio silently changed with hidden_size.
#   loss     objective screen: 3 DISTINCT objectives x 3 seeds                   (9 trials)
#            Generative eval is ON and trials rank by judge perplexity, because ranking on
#            core accuracy would let a core-weighted objective grade its own homework.
#            Deduped on the core:anchor RATIO, which is all the loss normaliser preserves:
#            24 nominal -> 9 distinct (weighted 1/1 == cross_entropy; 3/0 == 1/0). Checkpoints
#            select on whole-span accuracy, not core, for the same non-circularity reason.
#   quick    2 trials, 2 epochs — for checking this script runs, not for results.
#
# Every preset selects on the named validation split and never touches test. The dataset gate
# below refuses a two-way train/test file outright.
#
# Overrides (any preset):
#   LRS="3e-4 1e-3" SEEDS="42 43 44" ./scripts/sweep.sh 0 h3
#   MAX_EPOCHS=40 PATIENCE=5 ./scripts/sweep.sh 0 h3
#   SWEEP_DIR=results/runs/my_sweep ./scripts/sweep.sh 0 h3     # resume into the same directory
#   DRY_RUN=1 ./scripts/sweep.sh 0 loss                         # print the grid, run nothing
#   SELECT_METRIC=accuracy ./scripts/sweep.sh 0 h3              # which metric picks each
#                                                               #   trial's best checkpoint
#   DATA_PATH=path/to/three_way.json ./scripts/sweep.sh 0 h3
#
# Before the first run, create a three-way clustered dataset:
#   python -m cdrseqgen.data.preprocess.oas_paired \
#     --input_file dataset/data/raw/1287169_1_Paired_All.csv \
#     --output_file dataset/data/processed/processed_oas_tuning.json \
#     --split cluster --test_size 0.1 --validation_size 0.1 --seed 42

set -uo pipefail

GPU=${1:-0}
PRESET=${2:-h3}
BASE=${3:-configs/h3_scratch.json}
PYTHON=${PYTHON:-python}
DATA_PATH=${DATA_PATH:-dataset/data/processed/processed_oas_multi.json}
# Which pipeline the base config selects. The `arch` preset needs it: an encoder-decoder has two
# depths, and the flag that controls them is not the one the other methods use.
BASE_METHOD=$("$PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1])).get('method','mlm'))" "$BASE" 2>/dev/null || echo mlm)

# --- Preset grids. Env vars win, so a preset only supplies defaults. --------------------------
case "$PRESET" in
  h3)
    LRS=${LRS:-"1e-4 3e-4 1e-3 3e-3"}; WARMUPS=${WARMUPS:-"0.0 0.06"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0 0.01"}; SEEDS=${SEEDS:-"42 43"}
    EPOCH_LIST=${EPOCH_LIST:-"30"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    ;;
  pilot)
    LRS=${LRS:-"1e-4 3e-4 1e-3"}; WARMUPS=${WARMUPS:-"0.0"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0"}; SEEDS=${SEEDS:-"42"}
    EPOCH_LIST=${EPOCH_LIST:-"10 20"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    ;;
  arch)
    LRS=${LRS:-"3e-4"}; WARMUPS=${WARMUPS:-"0.06"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.01"}; SEEDS=${SEEDS:-"42"}
    EPOCH_LIST=${EPOCH_LIST:-"30"}
    HIDDEN_SIZES=${HIDDEN_SIZES:-"128 192 256"}; NUM_LAYERS=${NUM_LAYERS:-"4 6"}
    ;;
  loss)
    # One hyperparameter fixed at the h3 screen's winner; only the objective varies. Ranks on
    # judge perplexity because core recovery is what a core-weighted loss optimises — ranking on
    # it would just rediscover the weighting. Generative eval is therefore ON.
    LRS=${LRS:-"1e-3"}; WARMUPS=${WARMUPS:-"0.06"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.01"}; SEEDS=${SEEDS:-"42 43 44"}
    EPOCH_LIST=${EPOCH_LIST:-"30"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    LOSS_TYPES=${LOSS_TYPES:-"cross_entropy weighted_cross_entropy"}
    CORE_WEIGHTS=${CORE_WEIGHTS:-"1.0 3.0"}; ANCHOR_WEIGHTS=${ANCHOR_WEIGHTS:-"1.0 0.0"}
    GEN_EVAL=${GEN_EVAL:-true}; RANK_METRIC=${RANK_METRIC:-judge_ppl}
    # Checkpoint selection has the same circularity as ranking: core_accuracy is what a core-
    # weighted objective optimises, so selecting on it hands those settings an advantage at the
    # per-epoch level even when the final ranking is judged independently. Whole-span is the least
    # circular per-epoch metric available. eval_loss is NOT an option here — it is computed with
    # each trial's OWN loss function, so its scale differs between objectives and is not comparable.
    SELECT_METRIC=${SELECT_METRIC:-accuracy}
    # Generative eval runs once per trial, so the 150x20 default would dominate the sweep. 100x10
    # still gives ~1000 candidates per trial, which is ample for a stable judge-perplexity mean.
    GEN_EXAMPLES=${GEN_EXAMPLES:-100}; GEN_SAMPLES=${GEN_SAMPLES:-10}
    ;;
  lr)
    # Six trials: the learning rate alone, at the warmup/decay the `h3` screen already settled,
    # two seeds so a ranking that flips between them is visibly noise rather than a result.
    # This is the screen a NEW METHOD needs before it appears in any comparison -- inheriting
    # another method's optimum and then losing establishes nothing (Findings §9).
    LRS=${LRS:-"1e-4 3e-4 1e-3"}; WARMUPS=${WARMUPS:-"0.06"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.01"}; SEEDS=${SEEDS:-"42 43"}
    EPOCH_LIST=${EPOCH_LIST:-"20"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    ;;
  quick)
    LRS=${LRS:-"1e-4 1e-3"}; WARMUPS=${WARMUPS:-"0.0"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0"}; SEEDS=${SEEDS:-"42"}
    EPOCH_LIST=${EPOCH_LIST:-"2"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    ;;
  *)
    echo "ERROR: unknown preset '$PRESET'. Use: h3 lr pilot arch loss quick"; exit 1 ;;
esac

# Loss dims default to "whatever the base config says", so non-loss presets are unchanged.
LOSS_TYPES=${LOSS_TYPES:-"__keep__"}
CORE_WEIGHTS=${CORE_WEIGHTS:-"__keep__"}
ANCHOR_WEIGHTS=${ANCHOR_WEIGHTS:-"__keep__"}
GEN_EVAL=${GEN_EVAL:-false}
RANK_METRIC=${RANK_METRIC:-core}
# Per-epoch metric that picks each trial's best checkpoint (load_best_model_at_end).
SELECT_METRIC=${SELECT_METRIC:-core_accuracy}
# "__keep__" leaves the base config's generative-eval size alone.
GEN_EXAMPLES=${GEN_EXAMPLES:-__keep__}
GEN_SAMPLES=${GEN_SAMPLES:-__keep__}
case "$SELECT_METRIC" in
  core_accuracy|accuracy) ;;
  *) echo "ERROR: SELECT_METRIC must be core_accuracy|accuracy, got '$SELECT_METRIC'"; exit 1 ;;
esac
case "$RANK_METRIC" in
  core|whole|judge_ppl|diversity|gen_core|length_mae|length_match|valid_rate) ;;
  *) echo "ERROR: RANK_METRIC must be core|whole|judge_ppl|diversity|gen_core|length_mae|length_match|valid_rate,"
     echo "       got '$RANK_METRIC'"
     exit 1 ;;
esac
if [[ "$RANK_METRIC" =~ ^(judge_ppl|diversity|gen_core|length_mae|length_match|valid_rate)$ ]] && [[ "$GEN_EVAL" != "true" ]]; then
  echo "ERROR: RANK_METRIC=$RANK_METRIC needs generation_eval.json — set GEN_EVAL=true"; exit 1
fi

# Architecture dims are optional; an empty list means "leave the config's value alone".
[[ -z "$HIDDEN_SIZES" ]] && HIDDEN_SIZES="__keep__"
[[ -z "$NUM_LAYERS" ]] && NUM_LAYERS="__keep__"

PATIENCE=${PATIENCE:-5}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.0001}
SCHEDULER=${SCHEDULER:-linear}

[[ -f "$BASE" ]] || { echo "ERROR: no base config: $BASE"; exit 1; }
[[ -f "$DATA_PATH" ]] || {
  echo "ERROR: no three-way tuning dataset: $DATA_PATH"
  echo "Create it with the preprocessing command in this script's header."
  exit 1
}

# --- Dataset gate: refuse to tune on a two-way file or to select the final test split ----------
"$PYTHON" - "$BASE" "$DATA_PATH" <<'PY'
import json, sys
from collections import Counter

base_path, data_path = sys.argv[1:]
with open(base_path) as f:
    base = json.load(f)
with open(data_path) as f:
    data = json.load(f)

counts = Counter(row.get("split") for row in data)
missing = {"train", "validation", "test"} - set(counts)
if missing:
    raise SystemExit(
        f"ERROR: {data_path} has split counts {dict(counts)}; missing {sorted(missing)}. "
        "Do not tune on the historical two-way train/test file."
    )
print(f"Dataset gate passed: train={counts['train']}, validation={counts['validation']}, "
      f"test={counts['test']} (test will not be selected).")
print(f"Target loop from base config: {base.get('target_region', 'h3')}")
PY
[[ $? -eq 0 ]] || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"
export CDRSEQGEN_SKIP_LATEST=1          # an exploratory sweep must not steal results/runs/latest

STAMP=$(date +'%Y%m%d_%H%M%S')
OUT=${SWEEP_DIR:-"results/runs/${STAMP}_sweep_${PRESET}"}
mkdir -p "$OUT"
export CDRSEQGEN_RESULTS_ROOT="$OUT"

# Per-trial stdout/stderr goes in its own subdirectory. It is mostly a superset of the trial's own
# train.log (same logger lines, plus transformers warnings and tqdm redraws) — but only mostly, and
# the difference is what matters: a trial that dies before logging attaches (bad config, import
# error, OOM at startup) writes no train.log and may leave no run dir, so this capture is the only
# record of the failure. Kept, therefore, but out of the sweep root, where 32 near-identical
# filenames buried runs.tsv and summary.tsv.
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

MAP="$OUT/runs.tsv"
if [[ ! -f "$MAP" ]]; then
  printf 'lr\twarmup\tweight_decay\tepochs\thidden\tlayers\tloss\tcore_w\tanchor_w\tseed\tscheduler\tstatus\trun_dir\n' > "$MAP"
fi

# The grid is materialised once, one trial per TSV line. Ten nested bash loops were becoming
# unreadable and silently mis-ordered columns; a flat list keeps adding a dimension to one place.
GRID_FILE="$OUT/grid.tsv"
"$PYTHON" - "$GRID_FILE" "$LRS" "$WARMUPS" "$WEIGHT_DECAYS" "$EPOCH_LIST" "$HIDDEN_SIZES" \
                         "$NUM_LAYERS" "$LOSS_TYPES" "$CORE_WEIGHTS" "$ANCHOR_WEIGHTS" "$SEEDS" <<'PY'
import itertools, sys
out, *dims = sys.argv[1:]


def objective_key(loss, cw, aw):
    """The equivalence class of an objective, so two spellings of one loss are trained once.

    weighted_cross_entropy is a weighted MEAN over supervised tokens, so the normaliser cancels any
    common factor and only the core:anchor RATIO survives. Three collapses follow, and the 08-25
    screen ran all three before anyone noticed (README Findings §13):
      * cross_entropy ignores the weights entirely
      * equal weights reduce to plain cross_entropy       -> weighted 1/1 == cross_entropy
      * a common factor cancels                            -> weighted 3/0 == weighted 1/0
    """
    if loss in ("__keep__", "cross_entropy"):
        return (loss,)
    core, anchor = float(cw), float(aw)
    if anchor == 0:
        return ("core_only",)
    if core == 0:
        return ("anchor_only",)
    return ("cross_entropy",) if core == anchor else ("ratio", round(core / anchor, 6))


seen, rows, collapsed = set(), [], 0
for lr, wu, wd, ep, hs, nl, loss, cw, aw, seed in itertools.product(*[d.split() for d in dims]):
    if loss == "cross_entropy":
        cw = aw = "1.0"
    key = (lr, wu, wd, ep, hs, nl, objective_key(loss, cw, aw), seed)
    if key in seen:
        collapsed += 1
        continue
    seen.add(key)
    rows.append((lr, wu, wd, ep, hs, nl, loss, cw, aw, seed))
with open(out, "w") as handle:
    for row in rows:
        handle.write("\t".join(row) + "\n")
if collapsed:
    print(f"Grid: collapsed {collapsed} trial(s) that retrain an objective the grid already covers "
          f"(only the core:anchor ratio survives the loss normaliser; see Findings §13).")
PY
total=$(wc -l < "$GRID_FILE")

if [[ -n "${DRY_RUN:-}" ]]; then
  # Print the grid and stop. Checking a preset should never cost GPU time.
  echo "DRY_RUN — $total trials would run:"
  printf 'lr\twu\twd\tep\thidden\tlayers\tloss\tcore_w\tanchor_w\tseed\n'
  cat "$GRID_FILE"
  exit 0
fi

echo "======================================================================"
echo "CDRSeqGen sweep | preset=$PRESET | GPU $GPU | base=$BASE | trials=$total"
echo "data=$DATA_PATH (selecting on validation)"
echo "LR=[$LRS] warmup=[$WARMUPS] weight_decay=[$WEIGHT_DECAYS] epochs=[$EPOCH_LIST] seeds=[$SEEDS]"
[[ "$HIDDEN_SIZES" != "__keep__" ]] && echo "hidden=[$HIDDEN_SIZES] layers=[$NUM_LAYERS]"
[[ "$LOSS_TYPES" != "__keep__" ]] && echo "loss=[$LOSS_TYPES] core_w=[$CORE_WEIGHTS] anchor_w=[$ANCHOR_WEIGHTS]"
echo "generative_eval=$GEN_EVAL  ranked_by=$RANK_METRIC  checkpoints_selected_on=$SELECT_METRIC"
[[ "$GEN_EVAL" == "true" && "$GEN_EXAMPLES" != "__keep__" ]] && \
  echo "gen_eval size: ${GEN_EXAMPLES} frameworks x ${GEN_SAMPLES} candidates per trial"
echo "patience=$PATIENCE threshold=$STOP_THRESHOLD scheduler=$SCHEDULER"
echo "output=$OUT"
echo "======================================================================"

i=0
ROUND_FAILURES=0
while IFS=$'\t' read -r lr wu wd ep hs nl loss cw aw seed; do
  i=$((i + 1))

  # Resume only from a completed run whose artifacts satisfy this invocation. In particular, an
  # older training-only run must be revisited when a generation-based ranking is requested later.
  COMPLETED_RUN=$(awk -F '\t' -v a="$lr" -v b="$wu" -v c="$wd" -v d="$ep" -v e="$hs" -v f="$nl" \
         -v g="$loss" -v h="$cw" -v j="$aw" -v k="$seed" \
    'NR > 1 && $1==a && $2==b && $3==c && $4==d && $5==e && $6==f && $7==g && $8==h \
     && $9==j && $10==k && $12=="completed" {run=$13} END {print run}' "$MAP")
  if [[ -n "$COMPLETED_RUN" && -f "$COMPLETED_RUN/epoch_metrics.csv" &&
        ( "$GEN_EVAL" != "true" || -f "$COMPLETED_RUN/generation_eval.json" ) ]]; then
    echo "[$i/$total] SKIP completed: lr=$lr wu=$wu wd=$wd ep=$ep loss=$loss cw=$cw aw=$aw seed=$seed"
    continue
  fi

  tag="lr${lr}_wu${wu}_wd${wd}_ep${ep}_s${seed}"
  [[ "$hs" != "__keep__" ]] && tag="${tag}_h${hs}l${nl}"
  [[ "$loss" != "__keep__" ]] && tag="${tag}_${loss%%_*}${cw}-${aw}"
  export CDRSEQGEN_RUN_SUFFIX="$tag"
  LOG="$LOG_DIR/${tag}.log"

  EXTRA_ARGS=()
  if [[ "$hs" != "__keep__" ]]; then
    # intermediate_size defaults to 4 x hidden_size, so the FFN ratio tracks width and the
    # comparison is not confounded by a fixed FFN. Heads must divide hidden_size.
    EXTRA_ARGS+=(--hidden_size "$hs" --num_layers "$nl" --num_attention_heads 8)
    # An s2s config PINS num_encoder_layers/num_decoder_layers, and those win over num_layers --
    # so sweeping num_layers alone would run every arch trial at the config's fixed depth and
    # report the resulting noise as an architecture result. Sweep both stacks symmetrically.
    if [[ "$BASE_METHOD" == "s2s" ]]; then
      EXTRA_ARGS+=(--num_encoder_layers "$nl" --num_decoder_layers "$nl")
    fi
  fi
  if [[ "$loss" != "__keep__" ]]; then
    EXTRA_ARGS+=(--loss_type "$loss" --core_loss_weight "$cw" --anchor_loss_weight "$aw")
  fi
  [[ "$GEN_EXAMPLES" != "__keep__" ]] && EXTRA_ARGS+=(--gen_num_examples "$GEN_EXAMPLES")
  [[ "$GEN_SAMPLES" != "__keep__" ]] && EXTRA_ARGS+=(--gen_num_samples "$GEN_SAMPLES")

  echo "[$i/$total] RUN  lr=$lr wu=$wu wd=$wd ep=$ep hs=$hs nl=$nl loss=$loss cw=$cw aw=$aw seed=$seed"
  if "$PYTHON" train.py "$BASE" \
    --data_path "$DATA_PATH" \
    --eval_split validation \
    --learning_rate "$lr" \
    --warmup_ratio "$wu" \
    --weight_decay "$wd" \
    --num_train_epochs "$ep" \
    --lr_scheduler_type "$SCHEDULER" \
    --early_stopping_patience "$PATIENCE" \
    --early_stopping_threshold "$STOP_THRESHOLD" \
    --metric_for_best_model "$SELECT_METRIC" \
    --greater_is_better true \
    --load_best_model_at_end true \
    --save_total_limit 1 \
    --seed "$seed" \
    --data_seed "$seed" \
    --run_generative_eval "$GEN_EVAL" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG" 2>&1; then
    RUNDIR=$(grep -oE "Workspace initialized at: .*" "$LOG" | tail -1 | sed 's/.*at: //')
    if [[ -n "$RUNDIR" && -f "$RUNDIR/epoch_metrics.csv" &&
          ( "$GEN_EVAL" != "true" || -f "$RUNDIR/generation_eval.json" ) ]]; then
      STATUS=completed
      echo "          OK: $RUNDIR"
    else
      STATUS=incomplete
      ROUND_FAILURES=$((ROUND_FAILURES + 1))
      echo "          INCOMPLETE: training exited 0 but a required run/metric artifact is missing; see $LOG"
    fi
  else
    STATUS=failed; RUNDIR=""
    ROUND_FAILURES=$((ROUND_FAILURES + 1))
    echo "          FAILED: see $LOG"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lr" "$wu" "$wd" "$ep" "$hs" "$nl" "$loss" "$cw" "$aw" "$seed" "$SCHEDULER" \
    "$STATUS" "${RUNDIR:-}" >> "$MAP"
done < "$GRID_FILE"

# --- Reporting: summary table + comparison figure, both beside the runs ------------------------
# These live in cdrseqgen/visualization/ rather than in a heredoc here, so an interrupted sweep can
# still be summarised and plotted from whatever finished — which is exactly when you want to see
# where it got to. Neither step can fail the sweep: the trials are already on disk.
"$PYTHON" -m cdrseqgen.visualization.sweep_tables --sweep "$OUT" --rank_metric "$RANK_METRIC" \
  || echo "⚠️ Summary step failed; the runs are intact — rerun it against $OUT"
"$PYTHON" -m cdrseqgen.visualization.sweep_plots --sweep "$OUT" \
  || echo "⚠️ Plot step failed; the runs are intact — rerun it against $OUT"

if [[ $ROUND_FAILURES -gt 0 ]]; then
  echo ""
  echo "❌ Sweep finished with $ROUND_FAILURES failed/incomplete trial(s). Artifacts are in $OUT/"
  exit 1
fi

echo ""
echo "🎉 Sweep complete. Everything for this sweep is in $OUT/"
echo "   runs.tsv · summary.tsv · figures/ · logs/ · one folder per trial"
