#!/usr/bin/env python3
"""Print read-only dataset facts requested for the ablation report."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATA = Path("/home/data_labeling/data")


def _metadata_rows(root):
    rows = []
    for split in ("train", "validation"):
        path = root / split / "metadata.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            metadata = raw.get("metadata", raw)
            rows.append((split, metadata))
    return rows


def main():
    occlu = DATA / "OccluBench"
    labels = pd.read_excel(occlu / "label.xlsx")
    segments = pd.read_csv(occlu / "segments.csv")
    meta_keys = Counter()
    for path in sorted(occlu.glob("[0-9]" * 6 + "/meta.json")):
        meta_keys.update(json.loads(path.read_text()).keys())

    sources = {}
    all_ids = defaultdict(set)
    task_ids = defaultdict(lambda: defaultdict(set))
    row_counts = defaultdict(Counter)
    for task, directory in (
        ("contact", "v3-contact_dataset"),
        ("separation", "v3-separation_dataset"),
    ):
        for split, row in _metadata_rows(DATA / directory):
            source = str(row["video_source"])
            video_id = int(row["video_index"])
            all_ids[source].add(video_id)
            task_ids[source][task].add(video_id)
            row_counts[source][f"{task}_{split}_grid_rows"] += 1
    for source in sorted(all_ids):
        sources[source] = {
            "unique_video_indices_union": len(all_ids[source]),
            "contact_unique_video_indices": len(task_ids[source]["contact"]),
            "separation_unique_video_indices": len(
                task_ids[source]["separation"]
            ),
            "grid_rows": dict(row_counts[source]),
        }

    grpo_path = DATA / "v3-contact_grpo_dataset" / "train" / "metadata.jsonl"
    grpo_row = json.loads(grpo_path.open().readline())
    gt = int(grpo_row["gt_cell"])
    adjacent = None
    if gt < len(grpo_row["cell_rewards"]):
        adjacent = grpo_row["cell_rewards"][gt]
    elif gt > 1:
        adjacent = grpo_row["cell_rewards"][gt - 2]

    output = {
        "occlubench": {
            "label_columns": list(labels.columns),
            "sequence_rows": len(labels),
            "unique_source_folders": int(segments["source_folder"].nunique()),
            "episodes_per_source_folder": {
                str(key): int(value)
                for key, value in segments["source_folder"]
                .value_counts()
                .sort_index()
                .items()
            },
            "meta_keys_and_episode_counts": dict(meta_keys),
            "saved_occlusion_type_column": any(
                "occlu" in str(column).lower()
                and str(column).lower() not in {"episode_folder"}
                for column in labels.columns
            ),
        },
        "v3_training_metadata": sources,
        "grpo": {
            "example_file": str(grpo_path),
            "reward_sigma": grpo_row.get("reward_sigma"),
            "gt_cell": gt,
            "one_cell_away_reward": adjacent,
            "cell_rewards": grpo_row["cell_rewards"],
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
