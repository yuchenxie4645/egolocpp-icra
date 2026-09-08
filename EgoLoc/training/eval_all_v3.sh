#!/usr/bin/env bash
#
# Evaluate all six v3 adapters as three pairs, sequentially.
#
#   pair 1  SFT only    v3-sft/{contact,separation}_adapter
#   pair 2  GRPO only   v3-grpo/{contact,separation}_base_grpo_adapter
#   pair 3  SFT + GRPO  v3-grpo/{contact,separation}_sft_grpo_adapter
#
# Each pair is one infer_eval.py invocation that attaches both of its adapters
# and evaluates contact and separation against the held-out validation split
# (10 videos per source, identical rows across every pair).
#
# Each pair's report is appended to v3-eval-results.txt the moment that pair
# finishes, so partial results survive an interruption and you can watch the
# file grow.
#
# Usage:
#   ./eval_all_v3.sh                 # run the three pairs
#   ./eval_all_v3.sh --limit 20      # quick smoke: 20 rows per task
#   ./eval_all_v3.sh --with-base     # also evaluate the raw 4-bit base first
#   ./eval_all_v3.sh --gpu 0         # run on GPU 0 (default 1)
#   ./eval_all_v3.sh --fresh         # start a new results file
#
# Prompt note: the SFT and GRPO datasets carry different prompt wording
# ("closest to the moment" vs "corresponding to the exact moment"). Each pair
# is evaluated against the dataset it was trained on so every adapter sees its
# native prompt. Pass --same-prompt to force all pairs onto the SFT dataset
# instead, which controls for prompt at the cost of penalising the GRPO pairs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="/home/data_labeling/data"
SFT_ROOT="$SCRIPT_DIR/v3-sft"
GRPO_ROOT="$SCRIPT_DIR/v3-grpo"
LOG_DIR="$SCRIPT_DIR/v3-eval-logs"
RESULTS="$SCRIPT_DIR/v3-eval-results.txt"

GPU=1
LIMIT=0
WITH_BASE=0
SAME_PROMPT=0
FRESH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU="$2"; shift 2 ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        --with-base)   WITH_BASE=1; shift ;;
        --same-prompt) SAME_PROMPT=1; shift ;;
        --fresh)       FRESH=1; shift ;;
        -h|--help)     sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
# Base weights are cached locally and this host cannot reach huggingface.co.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$LOG_DIR"

# flash_attention_4 fails to JIT-compile against Qwen3.5's GQA layout on this
# Blackwell card, which is why training used SDPA. infer_eval.py still defaults
# to FA4, so force SDPA here to match.
COMMON_FLAGS=(--attn_impl sdpa --vision_attn_impl sdpa)
[[ $LIMIT -gt 0 ]] && COMMON_FLAGS+=(--limit "$LIMIT")

# SFT-flavoured prompts vs GRPO-flavoured prompts.
SFT_CONTACT="$DATA_DIR/v3-contact_dataset"
SFT_SEPARATION="$DATA_DIR/v3-separation_dataset"
GRPO_CONTACT="$DATA_DIR/v3-contact_grpo_dataset"
GRPO_SEPARATION="$DATA_DIR/v3-separation_grpo_dataset"
if [[ $SAME_PROMPT -eq 1 ]]; then
    GRPO_CONTACT="$SFT_CONTACT"
    GRPO_SEPARATION="$SFT_SEPARATION"
fi

if [[ $FRESH -eq 1 && -f "$RESULTS" ]]; then
    mv "$RESULTS" "$RESULTS.$(date +%Y%m%d-%H%M%S).bak"
fi

{
    echo
    echo "########################################################################"
    echo "# v3 adapter evaluation - started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "#   GPU=$GPU  limit=${LIMIT:-all}  same_prompt=$SAME_PROMPT"
    echo "########################################################################"
} >> "$RESULTS"

STARTED_AT=$(date +%s)

# Pull infer_eval.py's summary table out of a run log.
extract_report() {
    local log="$1"
    local start
    start=$(grep -n "^task/source" "$log" | tail -1 | cut -d: -f1 || true)
    if [[ -n "$start" ]]; then
        sed -n "$((start - 1)),\$p" "$log"
    else
        echo "  (no report produced - tail of log follows)"
        tail -n 15 "$log" | sed 's/^/  | /'
    fi
}

# run_pair <label> <log_name> <output_dir> <adapter_suffix> <contact_root> <separation_root>
run_pair() {
    local label="$1" log_name="$2" out_dir="$3" suffix="$4"
    local contact_root="$5" separation_root="$6"
    local log_path="$LOG_DIR/$log_name.log"
    local start_ts; start_ts=$(date +%s)

    echo
    echo "########################################################################"
    echo "# [$(date '+%H:%M:%S')]  $label"
    echo "#   adapters: $out_dir/{contact,separation}$suffix"
    echo "########################################################################"

    local extra=()
    [[ -n "$out_dir" ]] && extra+=(--output_dir "$out_dir" --adapter_suffix "$suffix")
    [[ -n "$contact_root" ]] && extra+=(--contact_root "$contact_root")
    [[ -n "$separation_root" ]] && extra+=(--separation_root "$separation_root")

    local status="OK"
    if ! python infer_eval.py "${extra[@]}" "${COMMON_FLAGS[@]}" 2>&1 | tee "$log_path"; then
        status="FAILED"
    fi

    {
        echo
        echo "------------------------------------------------------------------------"
        echo "$label"
        if [[ -n "$out_dir" ]]; then
            echo "  adapters : ${out_dir#$SCRIPT_DIR/}/{contact,separation}$suffix"
        else
            echo "  adapters : none (raw 4-bit base)"
        fi
        echo "  contact  : ${contact_root:-<saved val jsonl>}"
        echo "  separate : ${separation_root:-<saved val jsonl>}"
        echo "  status   : $status   ($(( ($(date +%s) - start_ts) / 60 )) min)"
        echo "------------------------------------------------------------------------"
        extract_report "$log_path"
    } >> "$RESULTS"

    echo "[$(date '+%H:%M:%S')]  $label -> $status, appended to ${RESULTS#$SCRIPT_DIR/}"
}

echo "Evaluating v3 adapters on GPU $GPU"
echo "  results : $RESULTS"
echo "  logs    : $LOG_DIR"
[[ $LIMIT -gt 0 ]] && echo "  limit   : $LIMIT rows per task"

# Fail early on a missing adapter rather than 40 minutes in.
missing=0
for a in "$SFT_ROOT/contact_adapter" "$SFT_ROOT/separation_adapter" \
         "$GRPO_ROOT/contact_base_grpo_adapter" "$GRPO_ROOT/separation_base_grpo_adapter" \
         "$GRPO_ROOT/contact_sft_grpo_adapter" "$GRPO_ROOT/separation_sft_grpo_adapter"; do
    if [[ ! -f "$a/adapter_model.safetensors" ]]; then
        echo "  WARNING: missing adapter -> ${a#$SCRIPT_DIR/}"
        missing=1
    fi
done
[[ $missing -eq 1 ]] && echo "  (pairs containing a missing adapter will report it and continue)"

if [[ $WITH_BASE -eq 1 ]]; then
    echo
    echo "########################################################################"
    echo "# [$(date '+%H:%M:%S')]  baseline - raw 4-bit base, no LoRA"
    echo "########################################################################"
    log_path="$LOG_DIR/0_base.log"
    status="OK"
    if ! python infer_eval.py --base_only \
            --contact_root "$SFT_CONTACT" --separation_root "$SFT_SEPARATION" \
            "${COMMON_FLAGS[@]}" 2>&1 | tee "$log_path"; then
        status="FAILED"
    fi
    {
        echo
        echo "------------------------------------------------------------------------"
        echo "baseline - raw 4-bit base (no LoRA)"
        echo "  status   : $status"
        echo "------------------------------------------------------------------------"
        extract_report "$log_path"
    } >> "$RESULTS"
fi

run_pair "pair 1/3 - SFT only" "1_sft_only" \
    "$SFT_ROOT" "_adapter" "$SFT_CONTACT" "$SFT_SEPARATION"

run_pair "pair 2/3 - GRPO only (fresh LoRA on base)" "2_grpo_only" \
    "$GRPO_ROOT" "_base_grpo_adapter" "$GRPO_CONTACT" "$GRPO_SEPARATION"

run_pair "pair 3/3 - SFT + GRPO" "3_sft_grpo" \
    "$GRPO_ROOT" "_sft_grpo_adapter" "$GRPO_CONTACT" "$GRPO_SEPARATION"

{
    echo
    echo "########################################################################"
    echo "# all pairs complete in $(( ($(date +%s) - STARTED_AT) / 60 )) min"
    echo "########################################################################"
} >> "$RESULTS"

echo
echo "########################################################################"
echo "# Done in $(( ($(date +%s) - STARTED_AT) / 60 )) min"
echo "# Results: $RESULTS"
echo "########################################################################"
