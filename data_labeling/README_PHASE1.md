# Phase 1 Preparation

This folder is the Phase 1 workspace for preparing official USST, EgoPAT3D-DT, EgoPAT3D, and DeskTIL resources for a future VLM/GRPO temporal interaction localization dataset.

Phase 1 only prepares, downloads, organizes, and verifies data/code. It does not create GRPO JSONL, label examples, filter examples, or upload anything to Hugging Face.

## Layout

- `external/`: official source repositories cloned from GitHub.
- `data/raw_downloads/`: original archives or cloud-file downloads.
- `data/EgoPAT3D/EgoPAT3D-postproc/`: expected USST EgoPAT3D-DT extracted layout.
- `data/EgoPAT3D/raw/`: official EgoPAT3D raw Hugging Face snapshot, if downloaded.
- `data/DeskTIL/`: sorted DeskTIL videos and annotations.
- `data/EgoLoc_samples/`: EgoLoc sample packages after extraction.
- `reports/`: manifests, inventories, blockers, and final summary.

## Reproduce

From this directory:

```bash
bash scripts/setup_env.sh
bash scripts/download_sources.sh
source .venv/bin/activate
python scripts/inspect_data.py
python scripts/make_manifest.py
tree -L 5 -h . > reports/tree.txt
```

To opt into the full official EgoPAT3D Hugging Face snapshot after checking disk capacity:

```bash
FULL_EGOPAT3D_HF_DOWNLOAD=1 bash scripts/download_sources.sh
```

Cloud-file blockers are documented in `reports/download_blockers.md`.
