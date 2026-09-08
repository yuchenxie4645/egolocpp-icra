# OccluBench EgoLoc sampling ablation

This pipeline evaluates only the 327 OccluBench sequences and only three
sampling modes: hand speed, pinch, and hand speed + pinch. Ground truth is the
local episode frame index. Three unavailable labels are ignored without
imputation: contact for `000098`, and separation for `000098` and `000238`.
The complete run therefore has 651 available events and 5,859 trials
(651 × 3 modes × N=3).

Inference is Transformers-only: one warm `Qwen/Qwen3.5-9B` NF4 model, with the
contact and separation PEFT adapters attached once and switched task-major.
Nothing here requires vLLM, an OpenAI client, or port 8000.

Run inside `xyc`:

```bash
cd /home/EgoLoc/ablation-tests
conda run -n egolocxyc python manifest.py build
conda run -n egolocxyc python manifest.py validate
conda run -n egolocxyc python convert.py --workers 2
CUDA_VISIBLE_DEVICES=0 conda run -n egolocxyc python signals.py --device cuda:0
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run -n egolocxyc python runner.py --resume
conda run -n egolocxyc python audit.py
conda run -n egolocxyc python stats.py
```

Equivalent sequential driver:

```bash
conda run -n egolocxyc python run_full.py all
```

An explicitly fake, separately stored CPU smoke evaluation is available after
conversion/signals:

```bash
conda run -n egolocxyc python runner.py \
  --dry-run --limit-episodes 1 --tasks contact --trials 0
```

It writes `results/dry_run_smoke.jsonl`, never `results/trials.jsonl`.

Useful resume/subset flags are `--episodes`, `--limit-episodes`, `--tasks`,
`--modes`, `--trials`, `--results`, and `--resume/--no-resume`. A production
results file refuses a mismatched protocol/config/code hash; use a new path
after any locked change.

Tests and syntax checks:

```bash
conda run -n egolocxyc python -m pytest tests -q
conda run -n egolocxyc python -m py_compile \
  /home/EgoLoc/trial3.py ./*.py
```

For a full unattended run from the host, invoke `./launch_tmux.sh`. It creates
the `egoloc-occlubench` session: preparation uses GPU 0; evaluation waits for
signals and then uses GPU 1. The launcher is not run automatically.

Expected successful audit cardinality:

- 327 unique sequences
- 651 available labeled events; 3 unavailable labels explicitly ignored
- 5,859 unique immutable trial keys
- 326 contact and 325 separation records in each mode × trial stratum
- paired seeds identical across modes
- at most two adapter activations/switches in clean task-major execution
