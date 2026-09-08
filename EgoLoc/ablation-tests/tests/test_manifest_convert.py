import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import convert
import manifest


def make_dataset(root, episodes=2, frames=12, missing_label=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    label_rows = []
    segment_rows = []
    for episode_index in range(episodes):
        episode = f"{episode_index:06d}"
        start = 100 + episode_index * 100
        end = start + frames - 1
        episode_dir = root / episode
        rgb_dir = episode_dir / "rgb"
        rgb_dir.mkdir(parents=True)
        for frame_index in range(frames):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 0] = (frame_index * 17) % 255
            image[:, :, 1] = (episode_index * 50 + frame_index * 7) % 255
            cv2.putText(
                image,
                str(frame_index),
                (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            assert cv2.imwrite(str(rgb_dir / f"{frame_index:06d}.png"), image)
        meta = {
            "episode": episode,
            "source_folder": f"/source/{episode}",
            "start_frame": start,
            "end_frame": end,
            "num_frames": frames,
        }
        (episode_dir / "meta.json").write_text(json.dumps(meta))
        contact_local = "x" if missing_label and episode_index == 0 else 2
        separate_local = frames - 3
        row = {
            "episode": episode_index,
            "start_frame": start,
            "end_frame": end,
            "num_frames": frames,
            "source_folder": f"/source/{episode}",
            "episode_folder": f"/original/{episode}",
            "created_at": "2026-08-06 00:00:00",
            "contact_local": contact_local,
            "separate_local": separate_local,
            "contact_frame": (
                "x" if contact_local == "x" else start + contact_local
            ),
            "seperate_frame": start + separate_local,
        }
        label_rows.append(row)
        segment_rows.append(
            {
                key: row[key]
                for key in (
                    "episode",
                    "start_frame",
                    "end_frame",
                    "num_frames",
                    "source_folder",
                    "episode_folder",
                    "created_at",
                )
            }
        )
        segment_rows[-1]["episode"] = episode
    labels = pd.DataFrame(label_rows, columns=manifest.EXPECTED_COLUMNS)
    labels.to_excel(root / "label.xlsx", sheet_name="Sheet1", index=False)
    pd.DataFrame(segment_rows).to_csv(root / "segments.csv", index=False)
    return root


def build_fixture_manifest(tmp_path, episodes=2, frames=12):
    dataset = make_dataset(tmp_path / "dataset", episodes=episodes, frames=frames)
    path = tmp_path / "manifest.json"
    document = manifest.build_manifest(
        output=path,
        label_path=dataset / "label.xlsx",
        segments_path=dataset / "segments.csv",
        dataset_root=dataset,
        expected_episodes=episodes,
    )
    return dataset, path, document


def test_manifest_workbook_count_labels_and_idempotent_validation(tmp_path):
    _, path, document = build_fixture_manifest(tmp_path)

    assert document["expected"] == {
        "sequences": 2,
        "label_slots": 4,
        "events": 4,
        "contact_events": 2,
        "separation_events": 2,
        "trials": 36,
        "unavailable_labels": 0,
    }
    assert document["missing_labels"]["contact"] == []
    assert document["missing_labels"]["total_count"] == 0
    assert document["episodes"][0]["labels"]["contact_local"] == 2
    assert document["episodes"][0]["labels"]["contact_global"] == 102
    assert document["episodes"][0]["v3_train_membership"] is False
    assert document["episodes"][0]["held_out"] is True
    first = manifest.validate_manifest(path)
    second = manifest.validate_manifest(path)
    assert first == second
    assert first["episodes"] == 2


def test_manifest_records_and_ignores_unavailable_label(tmp_path):
    dataset = make_dataset(tmp_path / "dataset", missing_label=True)
    document = manifest.build_manifest(
        output=tmp_path / "manifest.json",
        label_path=dataset / "label.xlsx",
        segments_path=dataset / "segments.csv",
        dataset_root=dataset,
        expected_episodes=2,
    )

    assert document["expected"] == {
        "sequences": 2,
        "label_slots": 4,
        "events": 3,
        "contact_events": 1,
        "separation_events": 2,
        "trials": 27,
        "unavailable_labels": 1,
    }
    assert document["missing_labels"]["contact"] == ["000000"]
    assert document["missing_labels"]["separation"] == []
    assert document["episodes"][0]["labels"]["contact_local"] is None
    assert document["episodes"][0]["availability"]["contact"] is False
    assert document["episodes"][0]["unavailable_labels"][0]["local_value"] == "x"


def test_png_to_mp4_preserves_count_indices_and_resumes(tmp_path):
    _, manifest_path, document = build_fixture_manifest(
        tmp_path, episodes=1, frames=12
    )
    video_dir = tmp_path / "videos"
    status_path = tmp_path / "conversion.jsonl"

    first = convert.convert_episodes(
        manifest_path=manifest_path,
        output_dir=video_dir,
        status_path=status_path,
        workers=1,
    )
    assert first[0]["status"] == "converted"
    video_path = video_dir / "000000.mp4"
    validated = convert.validate_video(
        video_path, document["episodes"][0]
    )
    assert validated["frame_count"] == 12
    assert validated["labels_aligned"] is True
    assert set(validated["spot_similarity"]) >= {"0", "6", "11", "2", "9"}

    second = convert.convert_episodes(
        manifest_path=manifest_path,
        output_dir=video_dir,
        status_path=status_path,
        workers=1,
    )
    assert second[0]["status"] == "reused"
    lines = status_path.read_text().strip().splitlines()
    assert [json.loads(line)["status"] for line in lines] == [
        "converted",
        "reused",
    ]
