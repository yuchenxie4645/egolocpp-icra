import json
from pathlib import Path

import convert
import runner
from tests.test_manifest_convert import build_fixture_manifest
from tests.test_signals_cache import write_valid_caches


def prepare_pipeline_fixture(tmp_path, episodes=2):
    _, manifest_path, document = build_fixture_manifest(
        tmp_path, episodes=episodes, frames=12
    )
    video_dir = tmp_path / "videos"
    converted = convert.convert_episodes(
        manifest_path=manifest_path,
        output_dir=video_dir,
        status_path=tmp_path / "conversion.jsonl",
        workers=1,
    )
    assert all(item["status"] == "converted" for item in converted)
    signal_dir = tmp_path / "signals"
    for record in document["episodes"]:
        write_valid_caches(
            signal_dir,
            episode=record["episode"],
            frames=record["num_frames"],
        )
    return manifest_path, video_dir, signal_dir


def read_records(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def test_task_major_runner_reuses_one_backend_and_pairs_seeds(tmp_path):
    manifest_path, video_dir, signal_dir = prepare_pipeline_fixture(tmp_path)
    backend = runner.DryRunBackend()
    requested_path = tmp_path / "production-name.jsonl"

    summary = runner.run_evaluation(
        manifest_path=manifest_path,
        video_dir=video_dir,
        signal_dir=signal_dir,
        results_path=requested_path,
        dry_run=True,
        backend=backend,
    )

    assert summary["written"] == 36
    assert summary["backend_load_count"] == 1
    assert summary["adapter_switch_count"] <= 2
    assert summary["backend_request_count"] >= 72
    assert summary["results_path"] != str(requested_path)
    assert "dry-run" in Path(summary["results_path"]).name
    records = read_records(summary["results_path"])
    assert len(records) == 36
    groups = {}
    for record in records:
        group = (record["episode"], record["task"], record["trial"])
        groups.setdefault(group, set()).add(record["seed"])
        assert record["config_snapshot"]["backend"] == "dry-run-fake"
        assert len(record["request_traces"][0]["used_frame_indices"]) == 9
        if record["mode"] == "speed":
            assert record["signal_reads"] == {"speed": True, "pinch": False}
            assert record["cue_coverage"]["pinch_n"] == 0
        if record["mode"] == "pinch":
            assert record["signal_reads"] == {"speed": False, "pinch": True}
            assert record["cue_coverage"]["speed_n"] == 0
            assert not any(
                "speed_feedback" in trace["stage"]
                for trace in record["request_traces"]
            )
    assert all(len(seeds) == 1 for seeds in groups.values())

    resumed = runner.run_evaluation(
        manifest_path=manifest_path,
        video_dir=video_dir,
        signal_dir=signal_dir,
        results_path=summary["results_path"],
        dry_run=True,
        backend=runner.DryRunBackend(),
    )
    assert resumed["written"] == 0
    assert resumed["skipped"] == 36
    assert len(read_records(summary["results_path"])) == 36


def test_cli_dry_run_uses_separate_smoke_path(tmp_path):
    manifest_path, video_dir, signal_dir = prepare_pipeline_fixture(
        tmp_path, episodes=1
    )
    requested = tmp_path / "trials.jsonl"

    exit_code = runner.main(
        [
            "--manifest",
            str(manifest_path),
            "--video-dir",
            str(video_dir),
            "--signal-dir",
            str(signal_dir),
            "--results",
            str(requested),
            "--dry-run",
            "--tasks",
            "contact",
            "--trials",
            "0",
        ]
    )

    assert exit_code == 0
    assert not requested.exists()
    smoke = tmp_path / "trials.dry-run.jsonl"
    assert smoke.exists()
    assert len(read_records(smoke)) == 3
