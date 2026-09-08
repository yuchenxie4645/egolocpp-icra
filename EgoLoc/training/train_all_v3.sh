#!/usr/bin/env bash
#
# Train all six v3 adapters sequentially on GPU 1.
#
#   1. v3-sft/contact_adapter                 SFT only
#   2. v3-sft/separation_adapter              SFT only
#   3. v3-grpo/contact_base_grpo_adapter      GRPO only (fresh LoRA on base)
#   4. v3-grpo/separation_base_grpo_adapter   GRPO only
#   5. v3-grpo/contact_sft_grpo_adapter       SFT + GRPO
#   6. v3-grpo/separation_sft_grpo_adapter    SFT + GRPO
#
# Stages 5-6 continue the adapters produced by stages 1-2, so ordering matters.
#
# Usage:
#   ./train_all_v3.sh                 # run every stage that is not done yet
#   ./train_all_v3.sh --overwrite     # rebuild everything from scratch
#   ./train_all_v3.sh --smoke         # 1 step / 8 rows per stage, for a dry run
#
# Completed stages are skipped on re-run unless --overwrite is passed, so an
# interrupted run can be resumed by re-invoking the same command.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=1

# The base weights are already in the local HF cache and this host cannot
# reach huggingface.co; without offline mode every load spends minutes in
# connection-timeout retries and then dies.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SFT_ROOT="$SCRIPT_DIR/v3-sft"
GRPO_ROOT="$SCRIPT_DIR/v3-grpo"
LOG_DIR="$SCRIPT_DIR/v3-logs"

OVERWRITE=0
SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --overwrite) OVERWRITE=1 ;;
        --smoke)     SMOKE=1 ;;
        -h|--help)   sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR"

SFT_FLAGS=()
GRPO_FLAGS=()
if [[ $OVERWRITE -eq 1 ]]; then
    SFT_FLAGS+=(--overwrite)
    GRPO_FLAGS+=(--overwrite)
fi
if [[ $SMOKE -eq 1 ]]; then
    SFT_FLAGS+=(--epochs 0.01)
    GRPO_FLAGS+=(--max_steps 1 --limit 8)
fi

STAGE_NUM=0
TOTAL_STAGES=6
STARTED_AT=$(date +%s)

banner() {
    echo
    echo "########################################################################"
    echo "# [$(date '+%H:%M:%S')]  stage $STAGE_NUM/$TOTAL_STAGES  $1"
    echo "########################################################################"
}

# run_stage <adapter_path> <log_name> <description> <command...>
run_stage() {
    local adapter="$1"; shift
    local log_name="$1"; shift
    local description="$1"; shift

    STAGE_NUM=$((STAGE_NUM + 1))

    if [[ -d "$adapter" && $OVERWRITE -eq 0 ]]; then
        echo
        echo "[$(date '+%H:%M:%S')]  stage $STAGE_NUM/$TOTAL_STAGES  SKIP (exists): $description"
        echo "                     $adapter"
        return 0
    fi

    banner "$description"
    local log_path="$LOG_DIR/$log_name.log"
    local stage_start
    stage_start=$(date +%s)

    if ! "$@" 2>&1 | tee "$log_path"; then
        echo "STAGE $STAGE_NUM FAILED: $description (see $log_path)" >&2
        exit 1
    fi

    if [[ ! -d "$adapter" ]]; then
        echo "STAGE $STAGE_NUM produced no adapter at $adapter" >&2
        exit 1
    fi

    echo "[$(date '+%H:%M:%S')]  stage $STAGE_NUM done in $(( ($(date +%s) - stage_start) / 60 )) min -> $adapter"
}

echo "Training 6 v3 adapters sequentially on GPU ${CUDA_VISIBLE_DEVICES}"
echo "  SFT output:  $SFT_ROOT"
echo "  GRPO output: $GRPO_ROOT"
echo "  logs:        $LOG_DIR"
[[ $SMOKE -eq 1 ]] && echo "  MODE: smoke test (1 step / 8 rows per stage)"
[[ $OVERWRITE -eq 1 ]] && echo "  MODE: overwrite (rebuilding all stages)"

# ---------------------------------------------------------------- SFT only ---
run_stage "$SFT_ROOT/contact_adapter" "1_sft_contact" \
    "SFT only - contact" \
    python train_qlora.py --tasks contact --output_dir "$SFT_ROOT" "${SFT_FLAGS[@]}"

run_stage "$SFT_ROOT/separation_adapter" "2_sft_separation" \
    "SFT only - separation" \
    python train_qlora.py --tasks separation --output_dir "$SFT_ROOT" "${SFT_FLAGS[@]}"

# --------------------------------------------------------------- GRPO only ---
run_stage "$GRPO_ROOT/contact_base_grpo_adapter" "3_grpo_only_contact" \
    "GRPO only - contact (fresh LoRA on base)" \
    python train_grpo.py --tasks contact --init_from base \
        --output_root "$GRPO_ROOT" "${GRPO_FLAGS[@]}"

run_stage "$GRPO_ROOT/separation_base_grpo_adapter" "4_grpo_only_separation" \
    "GRPO only - separation (fresh LoRA on base)" \
    python train_grpo.py --tasks separation --init_from base \
        --output_root "$GRPO_ROOT" "${GRPO_FLAGS[@]}"

# --------------------------------------------------------------- SFT + GRPO ---
run_stage "$GRPO_ROOT/contact_sft_grpo_adapter" "5_sft_grpo_contact" \
    "SFT + GRPO - contact" \
    python train_grpo.py --tasks contact --init_from sft \
        --sft_adapter_root "$SFT_ROOT" --output_root "$GRPO_ROOT" "${GRPO_FLAGS[@]}"

run_stage "$GRPO_ROOT/separation_sft_grpo_adapter" "6_sft_grpo_separation" \
    "SFT + GRPO - separation" \
    python train_grpo.py --tasks separation --init_from sft \
        --sft_adapter_root "$SFT_ROOT" --output_root "$GRPO_ROOT" "${GRPO_FLAGS[@]}"

# ------------------------------------------------------------------ summary ---
echo
echo "########################################################################"
echo "# All 6 adapters complete in $(( ($(date +%s) - STARTED_AT) / 60 )) min"
echo "########################################################################"
for adapter in \
    "$SFT_ROOT/contact_adapter" \
    "$SFT_ROOT/separation_adapter" \
    "$GRPO_ROOT/contact_base_grpo_adapter" \
    "$GRPO_ROOT/separation_base_grpo_adapter" \
    "$GRPO_ROOT/contact_sft_grpo_adapter" \
    "$GRPO_ROOT/separation_sft_grpo_adapter"
do
    if [[ -d "$adapter" ]]; then
        echo "  OK      ${adapter#$SCRIPT_DIR/}"
    else
        echo "  MISSING ${adapter#$SCRIPT_DIR/}"
    fi
done
