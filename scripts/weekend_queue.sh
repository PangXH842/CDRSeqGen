#!/usr/bin/env bash
# Unattended single-GPU experiment queue. Resumable: re-running skips finished stages and resumes unfinished batches.
#
#   ./scripts/weekend_queue.sh dry-run    # plan every stage on the CPU and print the schedule; runs nothing
#   ./scripts/weekend_queue.sh start      # run it (use nohup/tmux; see below)
#   ./scripts/weekend_queue.sh status     # what is done, running or failed
#
#   nohup setsid ./scripts/weekend_queue.sh start > /dev/null 2>&1 &     # survives closing the terminal
#
# Stages, in order of value so a partial weekend is still useful:
#   blind_all        information-budget test for the seven existing adapters (GPU records)
#   igcraft_t1       IgCraft at temperature 1.0, the sensitivity batch
#   seeds            scratch MLM/AR/S2S, seeds 44-48 (three methods train side by side), then benchmark them
#   repeats          seed 42 retrained twice per method with an identical config: fixed-seed run-to-run variation
#   capacity         medium / large / extra-large scratch models, seeds 42-43, then benchmark them
# Training and benchmark batches land in results/runs and results/benchmark as 20261009_weekend_<name>.
# A failed stage is logged and retried; later stages still run. The source tree is frozen at start: edit
# anything under benchmark/ or cdrseqgen/ and the queue stops rather than mix code versions.
set -u -o pipefail
cd "$(dirname "$0")/.."

TAG=${QUEUE_TAG:-20261009_weekend}
Q=${QUEUE_DIR:-results/queue/$TAG}
GPU=${GPU:-0}
CDR=${CDR:-/home/pang/miniconda3/envs/cdrprj/bin/python}
IGC=${IGC:-/mnt/hdd/pang/envs/igcraft/bin/python}
SPECS=configs/experiments/weekend_20261009
ORIGINAL_PLAN=results/benchmark/oas4_h3_length_free_20261007/plan.json
BLIND_LABELS=(mlm_s42 mlm_s43 ar_s42 ar_s43 s2s_s42 s2s_s43 iglm)
METHODS=(mlm ar s2s)
export PYTHON=$CDR HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

log() { mkdir -p "$Q"; printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$Q/queue.log"; }
source_id() { "$CDR" -c "import sys; sys.path.insert(0, '.'); from benchmark._common import execution_source_identity as f; print(f()['sha256'])"; }
stub_call() {  # QUEUE_STUB=1 test mode: the caller returns this status and never reaches its real commands
    echo "STUB $*"
    [[ -n ${STUB_FAIL:-} && "$*" == *"$STUB_FAIL"* ]] && return 1
    return 0
}
real() {  # every heavy command goes through here; in test mode reaching one is a bug, so refuse instead of running it
    if [[ -n ${QUEUE_STUB:-} ]]; then echo "REFUSED real command in stub mode: $*" >&2; exit 99; fi
    "$@"
}
retry() { local n=$1 i rc=1; shift; for ((i = 1; i <= n; i++)); do "$@" && return 0; rc=$?; echo "attempt $i/$n failed (exit $rc)"; sleep "${RETRY_SLEEP:-30}"; done; return $rc; }

stage() {  # stage NAME COMMAND...: skips finished stages, isolates failures, stops if the source tree moved
    local name=$1; shift
    [[ -f $Q/done/$name ]] && { log "skip   $name (done)"; return 0; }
    if [[ -z ${QUEUE_STUB:-} && $(source_id) != "$(cat "$Q/source.sha")" ]]; then
        log "ABORT  source tree changed since the queue started; restore it or start a new queue"; exit 3
    fi
    log "start  $name"; touch "$Q/running/$name"
    "$@" > "$Q/logs/$name.log" 2>&1
    local rc=$?; rm -f "$Q/running/$name"
    if [[ $rc -eq 0 ]]; then rm -f "$Q/failed/$name"; touch "$Q/done/$name"; log "done   $name"
    else echo "$rc" > "$Q/failed/$name"; log "FAILED $name (exit $rc), see $Q/logs/$name.log"; fi
    return $rc
}

group() {  # group NAME STEM...: one training batch per stem, side by side on the same GPU
    local name=$1 stem pids=(); shift
    for stem in "$@"; do stage "${name}_$stem" retry "${RETRIES:-2}" train_batch "$stem" & pids+=($!); done
    wait "${pids[@]}"
}

train_batch() {  # a training batch: resume if it started, else run; then require every trial completed
    [[ -n ${QUEUE_STUB:-} ]] && { stub_call train_batch "$1"; return; }
    local out=results/runs/${TAG}_$1
    [[ -d $out && ! -f $out/plan.json ]] && rm -rf "$out"
    if [[ -f $out/plan.json ]]; then real ./scripts/run.sh resume "$out" --gpus "$GPU"
    else real ./scripts/run.sh run "$SPECS/$1.json" --gpus "$GPU" --output "$out"; fi
    real "$CDR" -c "import sys; sys.path.insert(0, '.'); from benchmark.arms_recipe import batch_arms; print(len(batch_arms(sys.argv[1])), 'trials completed')" "$out"
}

bench() {  # bench NAME STEM...: benchmark every finished checkpoint of those training batches
    local name=$1 stem args=(); shift
    [[ -n ${QUEUE_STUB:-} ]] && { stub_call bench "$name" "$@"; return; }
    local recipe=benchmark/recipes/generated/$name.json out=results/benchmark/${TAG}_$name
    for stem in "$@"; do [[ -f results/runs/${TAG}_$stem/plan.json ]] && args+=(--batch "results/runs/${TAG}_$stem"); done
    [[ ${#args[@]} -gt 0 ]] || { echo "no training batch exists for $name"; return 1; }
    real "$CDR" benchmark/arms_recipe.py --name "$name" --output "$recipe" --skip-incomplete "${args[@]}" || return 1
    if [[ -f $out/plan.json ]]; then real ./scripts/run.sh benchmark resume "$out" --gpu "$GPU"
    else [[ -d $out ]] && rm -rf "$out"; real ./scripts/run.sh benchmark run --recipe "$recipe" --gpu "$GPU" --output "$out"; fi
}

igcraft_t1() {
    [[ -n ${QUEUE_STUB:-} ]] && { stub_call igcraft_t1; return; }
    local out=results/benchmark/${TAG}_igcraft_t1
    if [[ -f $out/plan.json ]]; then real ./scripts/run.sh benchmark resume "$out" --gpu "$GPU"
    else [[ -d $out ]] && rm -rf "$out"
        real ./scripts/run.sh benchmark run --recipe benchmark/recipes/oas4_h3_igcraft_t1.json \
            --external-python igcraft_t1="$IGC" --gpu "$GPU" --output "$out"; fi
}

blind_all() {  # the GPU record of the information-budget test for each existing adapter
    [[ -n ${QUEUE_STUB:-} ]] && { stub_call blind_all; return; }
    local label rc=0 dir
    for label in "${BLIND_LABELS[@]}"; do
        dir=results/benchmark/blind_checks/$label
        if [[ -f $dir/blind_check.json ]] && "$CDR" -c "import json, sys; sys.exit(json.load(open(sys.argv[1]))['verdict'] != 'pass')" "$dir/blind_check.json"; then continue; fi
        rm -rf "$dir"
        CUDA_VISIBLE_DEVICES=$GPU real "$CDR" benchmark/blind_check.py --manifest benchmark/manifests/oas4_h3_validation_length_free_smoke_v1.json \
            --limit 4 --workdir "$dir" --plan "$ORIGINAL_PLAN" --label "$label" --num-samples 5 || rc=1
    done
    return $rc
}

preflight() {
    local problem=0 used free
    [[ -x $CDR ]] || { echo "missing $CDR"; problem=1; }
    [[ -x $IGC ]] || { echo "missing $IGC"; problem=1; }
    "$CDR" scripts/queue_specs.py > /dev/null || problem=1
    used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1)
    [[ ${used:-0} -lt 4000 ]] || echo "WARNING: GPU $GPU already uses ${used} MiB; IgCraft and the large models may run out of memory"
    free=$(df --output=avail -BG results | tail -1 | tr -dc 0-9)
    [[ ${free:-0} -gt 150 ]] || { echo "only ${free} GB free under results/; need more than 150"; problem=1; }
    return $problem
}

summary() {
    log "---- summary ----"
    local name stem out done_n total failed=0
    for stem in "${STEMS[@]}"; do
        out=results/runs/${TAG}_$stem
        [[ -f $out/plan.json ]] || { log "train  $stem: not started"; continue; }
        read -r done_n total < <("$CDR" -c "
import csv, json, sys
plan = json.load(open('$out/plan.json')); rows = {r['trial_id']: r for r in csv.DictReader(open('$out/runs.tsv'), delimiter='\t')}
print(sum(rows.get(t['trial_id'], {}).get('status') == 'completed' for t in plan['trials']), len(plan['trials']))")
        log "train  $stem: $done_n/$total trials completed"
    done
    for name in blind_all igcraft_t1 bench_seeds bench_repeats bench_capacity; do
        if [[ -f $Q/done/$name ]]; then log "ok     $name"; else log "NOT DONE $name"; failed=1; fi
    done
    ls "$Q/failed" 2> /dev/null | sed 's/^/FAILED stage: /' | tee -a "$Q/queue.log"
    [[ -z $(ls "$Q/failed" 2> /dev/null) ]] || failed=1
    return $failed
}

STEMS=()
for m in "${METHODS[@]}"; do STEMS+=("seeds_$m" "repeat_${m}_r1" "repeat_${m}_r2" "capacity_$m"); done

case "${1:-}" in
    dry-run)
        preflight || exit 1
        for stem in "${STEMS[@]}"; do ./scripts/run.sh plan "$SPECS/$stem.json" --gpus "$GPU" 2>&1 | sed -n '1p'; done
        ./scripts/run.sh benchmark plan --recipe benchmark/recipes/oas4_h3_igcraft_t1.json --stage full \
            --external-python igcraft_t1="$IGC" 2>&1 | sed -n '1p'
        "$CDR" benchmark/blind_check.py --help > /dev/null && echo "blind_check available for: ${BLIND_LABELS[*]}"
        cat << 'EOF'

Schedule (estimates; one GPU, three methods side by side while training):
  blind_all    ~0.2 h   seven adapters x 3 runs x 4 frameworks
  igcraft_t1   ~1.6 h   150 frameworks x 20 attempts, 1.4 h generation + judging
  seeds        ~4 h     15 trainings (5 per method in parallel) + 15 arms generated and judged (~8 min each)
  repeats      ~1.5 h   6 trainings in two rounds + 6 arms
  capacity     ~6-10 h  18 trainings (the large ones dominate) + 18 arms
  total        ~13-17 h of the ~60 available
EOF
        ;;
    start)
        mkdir -p "$Q"/{logs,done,failed,running}
        exec 9> "$Q/lock"; flock -n 9 || { echo "a queue is already running in $Q"; exit 1; }
        preflight || exit 1
        [[ -f $Q/source.sha ]] || source_id > "$Q/source.sha"
        log "queue start: GPU $GPU, source $(head -c 12 "$Q/source.sha"), stages resume where they stopped"
        stage blind_all blind_all
        stage igcraft_t1 igcraft_t1
        group train_seeds seeds_mlm seeds_ar seeds_s2s
        stage bench_seeds bench seeds seeds_mlm seeds_ar seeds_s2s
        group train_rep1 repeat_mlm_r1 repeat_ar_r1 repeat_s2s_r1
        group train_rep2 repeat_mlm_r2 repeat_ar_r2 repeat_s2s_r2
        stage bench_repeats bench repeats repeat_mlm_r1 repeat_ar_r1 repeat_s2s_r1 repeat_mlm_r2 repeat_ar_r2 repeat_s2s_r2
        group train_capacity capacity_mlm capacity_ar capacity_s2s
        stage bench_capacity bench capacity capacity_mlm capacity_ar capacity_s2s
        summary; rc=$?
        log "queue finished (exit $rc); send back the summary above"
        exit $rc
        ;;
    status)
        [[ -d $Q ]] || { echo "no queue at $Q"; exit 1; }
        echo "running: $(ls "$Q/running" 2> /dev/null | tr '\n' ' ')"
        echo "done:    $(ls "$Q/done" 2> /dev/null | tr '\n' ' ')"
        echo "failed:  $(ls "$Q/failed" 2> /dev/null | tr '\n' ' ')"
        echo; tail -n "${2:-15}" "$Q/queue.log"
        ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
