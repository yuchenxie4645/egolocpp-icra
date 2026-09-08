import json

import audit
import runner
from manifest import load_manifest
from results_store import ResultsStore, result_key, stable_config_hash
from tests.test_manifest_convert import build_fixture_manifest


def complete_record(record, task, mode, trial, config, config_hash):
    episode = record["episode"]
    seed = runner.trial2_final.derive_seed(episode, task, trial)
    prediction = 2 if task == "contact" else 9
    if mode == "speed":
        coverage = {"speed_n": 4, "pinch_n": 0, "neutral_n": 5}
        reads = {"speed": True, "pinch": False}
        sources = {"speed": [1, 2, 3, 4], "pinch": []}
    elif mode == "pinch":
        coverage = {"speed_n": 0, "pinch_n": 3, "neutral_n": 6}
        reads = {"speed": False, "pinch": True}
        sources = {"speed": [], "pinch": [1, 2, 3]}
    else:
        coverage = {"speed_n": 4, "pinch_n": 2, "neutral_n": 3}
        reads = {"speed": True, "pinch": True}
        sources = {"speed": [1, 2, 3, 4], "pinch": [5, 6]}
    coverage.update(
        {
            "overlap_n": 0,
            "speed_frames": [],
            "pinch_frames": [],
            "neutral_frames": [],
        }
    )
    trace_base = {
        "mode": mode,
        "raw_response": '{"points": [1]}',
        "latency_s": 0.01,
        "backend_used": "transformers-nf4",
        "seed": seed,
        "task": task,
        "selected_frame": prediction,
        "saved_grid_path": "/tmp/grid.png",
        "midpoint_restriction": False,
    }
    traces = [
        {
            **trace_base,
            "stage": f"{task}_initial",
            "candidates": sorted(set(sources["speed"] + sources["pinch"])),
            "total_pool": list(range(12)),
            "used_frame_indices": list(range(9)),
            "flag": None,
        },
        {
            **trace_base,
            "stage": f"{task}_visual_feedback_00",
            "candidates": [prediction],
            "total_pool": [prediction],
            "used_frame_indices": [prediction],
            "flag": "feedback",
            "raw_response": "1",
        },
    ]
    return {
        "key": result_key(episode, task, mode, trial),
        "episode": episode,
        "task": task,
        "mode": mode,
        "trial": trial,
        "seed": seed,
        "adapter_alias": task,
        "ground_truth_local": record["labels"][
            "contact_local" if task == "contact" else "separate_local"
        ],
        "num_frames": record["num_frames"],
        "status": "success",
        "prediction_frame": prediction,
        "signal_reads": reads,
        "cue_coverage": coverage,
        "fallback": {
            "used": True,
            "count": coverage["neutral_n"],
            "frames": list(range(coverage["neutral_n"])),
            "kind": "chronological" if mode == "pinch" else "speed_ranked",
            "source_stage": f"{task}_initial",
        },
        "anchor": {
            "kind": "none",
            "task_anchor": "start" if task == "contact" else "end",
            "midpoint_restriction": False,
        },
        "candidate_grid_recall": True,
        "candidate_grid_distance_to_ground_truth": 0,
        "primary_output": "closed_loop",
        "request_traces": traces,
        "config_hash": config_hash,
        "config_snapshot": config,
    }


def write_complete_results(manifest_path, results_path):
    manifest_doc = load_manifest(manifest_path)
    config = runner.build_evaluation_config(manifest_doc)
    config_hash = stable_config_hash(config)
    store = ResultsStore(results_path, config_hash)
    for episode_record in manifest_doc["episodes"]:
        for task in audit.TASKS:
            for mode in audit.MODES:
                for trial in audit.TRIALS:
                    store.append(
                        complete_record(
                            episode_record,
                            task,
                            mode,
                            trial,
                            config,
                            config_hash,
                        )
                    )
    return config_hash


def test_full_audit_validates_all_strata_and_traces(tmp_path):
    _, manifest_path, _ = build_fixture_manifest(
        tmp_path, episodes=1, frames=12
    )
    results_path = tmp_path / "results.jsonl"
    write_complete_results(manifest_path, results_path)

    report = audit.audit_results(
        manifest_path=manifest_path,
        results_path=results_path,
        output_json=tmp_path / "audit.json",
        output_markdown=tmp_path / "audit.md",
    )

    assert report["success"] is True
    assert report["expected_keys"] == 18
    assert report["unique_keys"] == 18
    assert report["error_count"] == 0
    assert json.loads((tmp_path / "audit.json").read_text())["complete"] is True


def test_partial_audit_reports_missing_without_success_claim(tmp_path):
    _, manifest_path, _ = build_fixture_manifest(
        tmp_path, episodes=1, frames=12
    )
    results_path = tmp_path / "results.jsonl"
    write_complete_results(manifest_path, results_path)
    lines = results_path.read_text().splitlines()
    results_path.write_text("\n".join(lines[:-1]) + "\n")

    report = audit.audit_results(
        manifest_path=manifest_path,
        results_path=results_path,
        partial=True,
        output_json=tmp_path / "partial.json",
        output_markdown=tmp_path / "partial.md",
    )

    assert report["valid"] is True
    assert report["complete"] is False
    assert report["success"] is False
    assert report["missing_count"] == 1
    assert "NOT A SUCCESS CLAIM" in (tmp_path / "partial.md").read_text()
