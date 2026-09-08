# SPEC — EgoLoc Sampling Ablation on OccluBench (ablation #2, narrowed scope)

## Locked protocol (do not deviate without asking the user)

- **Dataset**: OccluBench only. 327 episode PNG sequences under
  `/home/data_labeling/data/OccluBench/<NNNNNN>/rgb/` (files `000000.png`..`{num_frames-1:06d}.png`).
- **Labels**: `/home/data_labeling/data/OccluBench/label.xlsx`, sheet `Sheet1`, 327 data rows,
  columns: `episode, start_frame, end_frame, num_frames, source_folder, episode_folder,
  created_at, contact_local, separate_local, contact_frame, seperate_frame`.
  Treat non-numeric `"x"` labels as explicitly unavailable and ignore them
  without imputation: contact for episode `000098`, and separation for
  episodes `000098` and `000238`. This leaves 326 contact and 325 separation
  events (651 available events total).
  **Ground truth for scoring is the LOCAL indices** (`contact_local`, `separate_local`),
  which index the re-indexed episode frames (local 0..num_frames-1), exactly what our MP4s use.
  `contact_frame`/`seperate_frame` are global source-video indices (start_frame + local); record
  them in the manifest for provenance but do NOT score against them.
- **Model/eval constants (held fixed across all conditions)**:
  - Base: `Qwen/Qwen3.5-9B` (local HF cache; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
  - Adapters: `/home/EgoLoc/training/v3-grpo/contact_sft_grpo_adapter` and
    `.../separation_sft_grpo_adapter` (r=16, alpha=32, peft 0.19.1). Served aliases: `contact`, `separation`.
  - Inference is **Transformers-only** in one long-lived Python process. Quantization is
    bitsandbytes NF4 (double quant, compute dtype bfloat16); text and vision attention use SDPA.
    No vLLM preflight/server and no OpenAI-compatible endpoint are part of this ablation.
  - Grid: trial3 "direct candidate grid", 3x3 (`grid_size=3`), prompts unchanged from trial3.py,
    start/end anchoring unchanged, max_feedbacks=1 (one visual-feedback step), N=3 trials.
- **Modes** (single shared code path):
  - `speed`: candidates = top-4 lowest hand-speed frames from speed JSON; fallback pool =
    speed-ranked valid frames; feedback = visual (`determine_by_state`) + speed check
    (`determine_by_speed`). NO pinch JSON reads anywhere.
  - `pinch`: candidates = pinch relative minima ±1 neighbors (`select_pinch_keyframes`); visual
    feedback stays; NO speed candidates, NO speed checks, NO speed-ranked fallback — the
    signal-neutral fallback pool is **chronological ascending frames** (anchor-respecting fill
    handled by the existing two-pass grid builder). NO speed JSON reads anywhere.
  - `speed_pinch`: exact current trial3 behavior (speed∪pinch union, speed-ranked fallback pool,
    visual + speed feedback).
- **Seeds**: deterministic, PAIRED across modes — seed depends only on (episode, task, trial),
  never on mode. Same seed for all three modes at the same (episode, task, trial). Thread the
  seed into Transformers generation (`torch.Generator`, with locked global torch/CUDA seed
  fallback only if the remote model rejects the generator keyword).
- **Equal budgets**: every mode fills the same 9-slot grid; cue coverage and fallback usage are
  recorded per trial (counts of speed/pinch/neutral-fill frames in the final grid).
- **Leakage note**: v3-contact/v3-separation SFT+GRPO training datasets contain NO OccluBench
  (sources: egopat3d, egodex, desktil, dyngrasp). OccluBench is held-out w.r.t. the adapters.
  Record this in the manifest/report; no separate held-out sensitivity needed beyond documenting it.

## Scale / resume requirements

- Event-level trials: 651 available events × 3 modes × 3 trials = **5,859** before feedback;
  roughly 12k–25k model requests with feedback. Every stage MUST be resumable and persist each
  result immediately (append-safe JSONL, fsync per line). Resume by immutable result key:
  `{episode}|{task}|{mode}|{trial}` — never overwrite a completed key.

## Code layout (this folder: /home/EgoLoc/ablation-tests)

Plain modules imported via sys.path insertion (hyphen in dir name — no package import):

- `spec.md` — this file
- `manifest.py` — build versioned manifest JSON from label.xlsx + segments.csv:
  per-episode {episode, num_frames, contact_local, separate_local, global labels, label paths,
  mtime, availability/unavailable-marker provenance,
  v3_train_membership="not_in_v3_train"} + expected key enumeration (5859).
- `convert.py` — PNG→MP4 (`cv2.VideoWriter`, mp4v, fps=30) per episode into
  `/home/EgoLoc/ablation-tests/data/videos/<NNNNNN>.mp4`. Verify: PNG names contiguous 0..n-1,
  every frame readable, output FRAME_COUNT == num_frames, dimensions match, re-read spot checks.
  Per-episode conversion manifest JSON for resume/provenance.
- `signals.py` — speed + pinch extraction per episode MP4 using trial3.py's
  `build_detector`/`build_vitpose`/`extract_2d_speed_and_visualize`/`extract_pinch_distance_and_visualize`
  (HaMeR cache at /home/EgoLoc/hamer_cache, GPU 0). Cache dir
  `/home/EgoLoc/ablation-tests/data/signals/<NNNNNN>/metrics/`. Validation: speed JSON keys ==
  0..num_frames-1; pinch JSON parses with total_frames == num_frames. Regenerate only invalid/missing.
- `backend.py` — one persistent `TransformersBackend` returning `(parsed_point, raw_text)`.
  It follows the proven path from `/home/EgoLoc/training/infer_eval.py`: NF4
  `BitsAndBytesConfig`, both PEFT adapters attached once and unmerged, `set_adapter(task)` only
  when the task changes, SDPA for both towers (FA4 fails on Blackwell), and the legacy rope /
  processor shims. Decoding is locked to `do_sample=True`, temperature 0.1, top_p 0.5,
  max_new_tokens 200, disabled frequency/presence penalties, and `enable_thinking=False`.
  The model and processor stay warm for all 5,859 trials. This module has no OpenAI or vLLM
  dependency.
- `runner.py` — orchestrates evaluate stage: for each key not already in results JSONL:
  run one trial through the trial3 mode-aware API with max_feedbacks=1, collect trace, append
  record, fsync. Record fields: key, episode, task, mode, trial, seed, prediction_frame,
  fallback_used (+which stage), cue_coverage {speed_n, pinch_n, neutral_n}, used_frame_indices,
  raw_response, parse_error, error, latency_s, backend_used, adapter_alias, config_hash, ts.
  Bounded retries (3, exponential backoff) on transient errors; terminal failures recorded.
- `stats.py` — aggregation + paired bootstrap 95% CIs (resample episodes), Wilcoxon paired tests
  for all mode pairs on common successful samples + Holm correction, paired effect sizes,
  sample-weighted and dataset-macro summaries, JSON/CSV/Markdown/plots outputs into `results/`.
- `run_full.py` — stage CLI: `manifest|convert|signals|evaluate|audit|stats|all`, resume-safe,
  tmux-friendly logging to `logs/`.
- `tests/` — pytest suite (no GPU required): condition isolation (speed mode never reads pinch
  JSON and vice versa — use sentinels/monkeypatched readers), candidate budget/order/anchors
  (9 slots, chronological, anchor region respected with documented relaxations), no-speed pinch
  fallback (chronological pool), deterministic paired seeds, missing-label path, cache validation
  logic, MP4 conversion on a tiny synthetic PNG dir (index preservation), resume/idempotence with
  a fake backend, metric calculations (MAE, CI, Holm).

## trial3.py refactor contract (Builder 1 deliverable)

- Add `SamplingMode = Literal["speed", "pinch", "speed_pinch"]` + validation helper.
- Centralize candidate construction: `build_mode_candidates(mode, speed_json_path,
  pinch_json_path, n_candidates=4)` → dict {candidates, total_pool, coverage} where total_pool
  semantics: speed→speed-ranked valid frames; pinch→chronological ascending valid frames;
  speed_pinch→speed-ranked (current). This is the ONLY place signals are selected.
- Thread `mode` keyword through: `process_task`, `determine_by_state`, `determine_by_speed`,
  `feedback_contact`, `feedback_separation`, `convert_video` — default `mode="speed_pinch"`.
  All existing signatures otherwise unchanged; all existing callers keep current behavior
  (mode="speed_pinch" must be byte-identical behavior to today's code path).
- In `pinch` mode: `determine_by_speed` is never called (feedback_contact/separation branch on
  mode); no code path may open the speed JSON. In `speed` mode: no code path may open the pinch
  JSON (pinch_folder ignored).
- Backend abstraction: `vlm_request(image_bgr, prompt, task, backend, seed)` accepts a persistent
  backend object exposing `generate(...)`. The ablation runner always passes its single
  `TransformersBackend` object and never instantiates or reloads inside `trial3.py`.
  `scene_understanding` and the legacy string `backend="vllm"` remain only for backward
  compatibility outside this ablation; there is no `"auto"` mode or server preflight.
- Trace collection: optional `trace_out: dict | None = None` param on process_task; when provided,
  fill {stage, candidates, total_pool, used_frame_indices, raw_response, latency_s, backend_used,
  seed, mode}. Backward compatible when None.
- Keep the `__main__` demo CLI working unchanged.

## Environment notes for builders

- All commands run INSIDE container via `docker exec xyc bash -c '...'` (Shell tool,
  required_permissions ["all"]). Python: `conda run -n egolocxyc python ...`.
- NEVER request sandbox permissions for host-side file operations. Code files in
  /home/EgoLoc/ablation-tests (= workspace /home/xieyuchen/data0/EgoLoc/ablation-tests) are
  edited with the normal file tools (folder is chmod 777 per user's instruction).
- Trial3.py is untracked in a dirty repo: DO NOT run any git command. Do not touch auth.env,
  script/, hamer/, training/ (read-only reference), or any unrelated files.
- The host cannot reach huggingface.co — keep offline env vars for model loads.
- Tests must pass with: `docker exec xyc bash -c 'cd /home/EgoLoc/ablation-tests && conda run -n egolocxyc python -m pytest tests -q'`
  (pytest may need `pip install pytest` inside egolocxyc if missing; check first).
