# Locked protocol — OccluBench Trial-2-final sampling ablation

## Scope

- Dataset: all 327 local OccluBench episode sequences.
- Available labels: 326 contact and 325 separation events.
- Ignore without imputation: contact `000098`; separation `000098`, `000238`.
- Conditions: `speed`, `pinch`, `speed_pinch`.
- Trials: deterministic paired N=3.
- Expected immutable result keys: 651 × 3 × 3 = 5,859.
- This experiment does not include DeskTIL, EgoPAT3D, Greedy VLM, GPT-4o,
  VideoChat, or any other model comparison.

## Final localization implementation

- Use only `/home/EgoLoc/trial_2_final.py`.
- Do not modify or import `trial2.py` or `trial3.py`.
- Trial-2 cue aggregation: center the grid on the rounded mean of selected
  cue frames.
- Every localization grid contains exactly nine unique consecutive source
  frames, clamped to the video bounds.
- Remove the contact-first-half / separation-second-half restriction.
- Common exact prompt:
  - contact: exact earliest frame where a stable grasp is established;
  - separation: exact earliest frame where visible release occurs.
- Retain the `-1` option.
- Primary prediction: closed-loop result after one visual-feedback round
  (`max_feedbacks=1`). Persist the initial prediction separately.
- Report initial candidate-grid recall separately from conditional localization
  error. In a consecutive grid, cell distance equals source-frame distance.

## Missing-hand corrections

- Speed is Euclidean hand-center displacement divided by elapsed source-frame
  gap. Frames without detections retain zero/unavailable speed and are not
  selected as nonzero speed cues.
- Pinch distance is thumb-tip to index-tip distance.
- Split detected pinch observations into contiguous frame-index segments.
  Compute Trial-2 local minima independently per segment, then add ±1 source
  frame neighbors within video bounds.
- Speed and pinch share one detector/ViTPose pass per video. Cache metadata must
  identify schema `trial-2-final-gap-and-segment-v1`; legacy caches are invalid.

## Mode isolation

- `speed`: top four lowest valid nonzero gap-normalized speeds; no pinch read;
  speed feedback enabled.
- `pinch`: contiguous-segment pinch minima ±1; no speed read or speed feedback;
  chronological signal-neutral cue fallback.
- `speed_pinch`: deduplicated speed-first union of both cue sets; speed feedback
  enabled.
- If a condition has fewer than four cue frames, fill cue-center inputs from
  the condition-allowed fallback pool. The displayed grid remains consecutive.

## Fixed model and inference

- Base: `Qwen/Qwen3.5-9B`.
- Adapters:
  - `/home/EgoLoc/training/v3-grpo/contact_sft_grpo_adapter`
  - `/home/EgoLoc/training/v3-grpo/separation_sft_grpo_adapter`
- Both adapters are rank 16, alpha 32, attached once and never merged.
- Transformers only; no vLLM/OpenAI server.
- NF4, double quantization, bfloat16 compute/model dtype, SDPA.
- One persistent warm model/processor process; task-major execution.
- Decoding: sampling enabled, temperature 0.1, top-p 0.5,
  max-new-tokens 200, thinking disabled.
- Paired seed depends only on `(episode, task, trial)`, never mode.

## Data and persistence

- Labels: `/home/data_labeling/data/OccluBench/label.xlsx`.
- Score local indices (`contact_local`, `separate_local`) only.
- Convert ordered PNG sequences to verified mp4v MP4s.
- Persist conversion, cache, request, result, error, latency, cue/fallback,
  prompt, grid, adapter, NF4, seed, and code provenance.
- Append and fsync each trial immediately.
- Resume by `{episode}|{task}|{mode}|{trial}` and never overwrite completed
  success or terminal-failure records.

## Analysis

- Aggregate each event/mode using `int(round(mean(valid N=3 predictions)))`.
- Report per task/mode and combined: available count, success/failure,
  candidate-grid recall, mean/median frame MAE, normalized MAE, and paired
  bootstrap 95% confidence intervals.
- Run paired Wilcoxon tests for every mode pair on common successful events,
  one declared Holm family, with matched rank-biserial effect sizes.
- Claim superiority only when direction, paired CI, and corrected test agree.
- Outputs: JSON, CSV, paper-style Markdown table/report, plots, audit, and full
  configuration/provenance.
