#!/usr/bin/env python3
"""Audit completeness, isolation, traces, and protocol identity."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from manifest import load_manifest
from results_store import read_jsonl_tolerant, result_key, stable_config_hash

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "data" / "manifest.json"
DEFAULT_RESULTS = HERE / "results" / "trials.jsonl"
DEFAULT_JSON = HERE / "results" / "audit.json"
DEFAULT_MARKDOWN = HERE / "results" / "AUDIT.md"
TASKS = ("contact", "separation")
MODES = ("speed", "pinch", "speed_pinch")
TRIALS = (0, 1, 2)
TRACE_KEYS = {
    "stage",
    "mode",
    "candidates",
    "total_pool",
    "used_frame_indices",
    "raw_response",
    "latency_s",
    "backend_used",
    "seed",
    "task",
    "flag",
    "selected_frame",
    "saved_grid_path",
}


def expected_keys(manifest):
    return {
        result_key(record["episode"], task, mode, trial)
        for record in manifest["episodes"]
        for task in TASKS
        for mode in MODES
        for trial in TRIALS
    }


def _error(errors, key, message):
    errors.append({"key": key, "message": message})


def _validate_config(record, errors):
    key = record.get("key", "?")
    config = record.get("config_snapshot")
    if not isinstance(config, dict):
        _error(errors, key, "missing config_snapshot")
        return
    if stable_config_hash(config) != record.get("config_hash"):
        _error(errors, key, "config_snapshot hash mismatch")
    if config.get("backend") != "transformers-nf4":
        _error(errors, key, "backend is not transformers-nf4")
    quant = config.get("quantization", {})
    expected_quant = {
        "implementation": "bitsandbytes",
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
    }
    if quant != expected_quant:
        _error(errors, key, f"NF4 config differs: {quant!r}")
    adapters = config.get("adapters", {})
    for task in TASKS:
        adapter = adapters.get(task, {})
        if (
            adapter.get("rank") != 16
            or adapter.get("alpha") != 32
            or adapter.get("merged") is not False
        ):
            _error(errors, key, f"{task} adapter identity/config differs")


def _validate_trace(record, errors):
    key = record.get("key", "?")
    traces = record.get("request_traces")
    if not isinstance(traces, list):
        _error(errors, key, "request_traces is not a list")
        return
    if record.get("status") == "success" and len(traces) < 2:
        _error(errors, key, "successful trial has fewer than two requests")
    initial_stage = f"{record.get('task')}_initial"
    initial_count = 0
    for index, trace in enumerate(traces):
        if not isinstance(trace, dict):
            _error(errors, key, f"trace {index} is not an object")
            continue
        missing = TRACE_KEYS - set(trace)
        if missing:
            _error(errors, key, f"trace {index} missing {sorted(missing)}")
        if trace.get("stage") == initial_stage:
            initial_count += 1
        if trace.get("mode") != record.get("mode"):
            _error(errors, key, f"trace {index} mode mismatch")
        if trace.get("task") != record.get("task"):
            _error(errors, key, f"trace {index} task mismatch")
        if trace.get("seed") != record.get("seed"):
            _error(errors, key, f"trace {index} seed mismatch")
        if trace.get("backend_used") != "transformers-nf4":
            _error(errors, key, f"trace {index} backend mismatch")
        used = trace.get("used_frame_indices")
        if not isinstance(used, list):
            _error(errors, key, f"trace {index} used frames not a list")
            continue
        if any(
            not isinstance(frame, int)
            or frame < 0
            or frame >= int(record.get("num_frames", 0))
            for frame in used
        ):
            _error(errors, key, f"trace {index} has out-of-range frame")
        if len(used) not in (1, 9):
            _error(errors, key, f"trace {index} grid has {len(used)} frames")
        if len(used) == 1 and "visual_feedback" not in str(trace.get("stage")):
            _error(errors, key, f"trace {index} unexpected one-frame request")
        if len(used) == 9 and len(set(used)) != 9:
            _error(errors, key, f"trace {index} grid frames are not unique")
        latency = trace.get("latency_s")
        if not isinstance(latency, (int, float)) or latency < 0:
            _error(errors, key, f"trace {index} invalid latency")
    if record.get("status") == "success" and initial_count != 1:
        _error(errors, key, f"expected one initial trace, found {initial_count}")


def _validate_mode_isolation(record, errors):
    key = record.get("key", "?")
    mode = record.get("mode")
    reads = record.get("signal_reads", {})
    coverage = record.get("cue_coverage", {})
    speed_n = coverage.get("speed_n")
    pinch_n = coverage.get("pinch_n")
    neutral_n = coverage.get("neutral_n")
    if not all(isinstance(value, int) for value in (speed_n, pinch_n, neutral_n)):
        _error(errors, key, "cue coverage counts missing/non-integer")
    elif speed_n + pinch_n + neutral_n != 9:
        _error(errors, key, "cue coverage does not sum to nine")
    stages = [
        str(trace.get("stage", ""))
        for trace in record.get("request_traces", [])
        if isinstance(trace, dict)
    ]
    if mode == "speed":
        if reads != {"speed": True, "pinch": False}:
            _error(errors, key, "speed mode signal reads leaked")
        if pinch_n != 0:
            _error(errors, key, "speed mode has pinch cue frames")
    elif mode == "pinch":
        if reads != {"speed": False, "pinch": True}:
            _error(errors, key, "pinch mode signal reads leaked")
        if speed_n != 0:
            _error(errors, key, "pinch mode has speed cue frames")
        if any("speed_feedback" in stage for stage in stages):
            _error(errors, key, "pinch mode contains speed feedback request")
    elif mode == "speed_pinch":
        if reads != {"speed": True, "pinch": True}:
            _error(errors, key, "speed_pinch mode read provenance differs")
    fallback = record.get("fallback")
    if not isinstance(fallback, dict):
        _error(errors, key, "missing fallback provenance")
    else:
        expected_kind = "chronological" if mode == "pinch" else "speed_ranked"
        if fallback.get("kind") != expected_kind:
            _error(errors, key, "fallback kind differs from mode protocol")
        if fallback.get("count") != len(fallback.get("frames", [])):
            _error(errors, key, "fallback count/frame list mismatch")
    anchor = record.get("anchor")
    if not isinstance(anchor, dict) or anchor.get("kind") not in ("start", "end"):
        _error(errors, key, "missing anchor provenance")


def audit_results(
    manifest_path=DEFAULT_MANIFEST,
    results_path=DEFAULT_RESULTS,
    partial=False,
    output_json=DEFAULT_JSON,
    output_markdown=DEFAULT_MARKDOWN,
):
    manifest = load_manifest(manifest_path, validate=True, check_sources=False)
    records, torn_final = read_jsonl_tolerant(results_path)
    expected = expected_keys(manifest)
    keys = [record.get("key") for record in records]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    actual = {key for key in keys if isinstance(key, str)}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    errors = []
    for key in duplicates:
        _error(errors, key, f"duplicate key appears {counts[key]} times")
    for key in unexpected:
        _error(errors, key, "unexpected result key")

    config_hashes = {
        record.get("config_hash")
        for record in records
        if record.get("config_hash") is not None
    }
    if len(config_hashes) > 1:
        _error(errors, "*", "multiple config hashes in one results file")

    strata = defaultdict(int)
    seed_groups = defaultdict(dict)
    status_counts = Counter()
    by_episode = {record["episode"]: record for record in manifest["episodes"]}
    for record in records:
        key = record.get("key", "?")
        task = record.get("task")
        mode = record.get("mode")
        trial = record.get("trial")
        episode = record.get("episode")
        status = record.get("status")
        status_counts[status] += 1
        if status not in ("success", "terminal_failure"):
            _error(errors, key, f"invalid status {status!r}")
        if episode not in by_episode:
            _error(errors, key, "episode absent from manifest")
            continue
        if task not in TASKS or mode not in MODES or trial not in TRIALS:
            _error(errors, key, "task/mode/trial identity differs")
        if key != result_key(episode, task, mode, trial):
            _error(errors, key, "immutable key does not match fields")
        if record.get("adapter_alias") != task:
            _error(errors, key, "task alias does not match adapter identity")
        expected_label = int(
            by_episode[episode]["labels"][
                "contact_local" if task == "contact" else "separate_local"
            ]
        )
        if record.get("ground_truth_local") != expected_label:
            _error(errors, key, "ground-truth local label mismatch")
        if status == "success":
            prediction = record.get("prediction_frame")
            if (
                not isinstance(prediction, int)
                or prediction < 0
                or prediction >= int(record.get("num_frames", 0))
            ):
                _error(errors, key, "successful prediction is out of range")
        elif record.get("prediction_frame") is not None:
            _error(errors, key, "terminal failure contains a prediction")
        strata[(task, mode, trial)] += 1
        seed_groups[(episode, task, trial)][mode] = record.get("seed")
        _validate_config(record, errors)
        _validate_trace(record, errors)
        _validate_mode_isolation(record, errors)

    for group, mode_seeds in seed_groups.items():
        if len(mode_seeds) == 3 and len(set(mode_seeds.values())) != 1:
            _error(errors, "|".join(map(str, group)), "paired seeds differ by mode")

    expected_per_stratum = len(manifest["episodes"])
    bad_strata = {
        f"{task}|{mode}|{trial}": count
        for (task, mode, trial), count in sorted(strata.items())
        if count != expected_per_stratum
    }
    complete = not missing and not unexpected and not duplicates and not torn_final
    valid = not errors
    report = {
        "dataset": "OccluBench",
        "partial_mode": bool(partial),
        "complete": complete,
        "valid": valid,
        "success": bool(complete and valid and not partial),
        "expected_keys": len(expected),
        "records": len(records),
        "unique_keys": len(actual),
        "missing_count": len(missing),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "duplicates": duplicates,
        "torn_final_line_ignored": torn_final,
        "status_counts": dict(status_counts),
        "bad_strata": bad_strata,
        "config_hashes": sorted(config_hashes),
        "error_count": len(errors),
        "errors": errors,
    }
    output_json = Path(output_json)
    output_markdown = Path(output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = (
        "COMPLETE AND VALID"
        if report["success"]
        else "PARTIAL — NOT A SUCCESS CLAIM"
        if partial and valid
        else "INVALID OR INCOMPLETE"
    )
    markdown = [
        "# OccluBench ablation audit",
        "",
        f"**State:** {state}",
        "",
        f"- Expected keys: {len(expected)}",
        f"- Unique keys: {len(actual)}",
        f"- Missing: {len(missing)}",
        f"- Duplicate keys: {len(duplicates)}",
        f"- Terminal failures: {status_counts.get('terminal_failure', 0)}",
        f"- Validation errors: {len(errors)}",
        f"- Torn final line ignored: {torn_final}",
    ]
    if missing:
        markdown.extend(
            ["", "## Missing keys", "", *[f"- `{key}`" for key in missing[:100]]]
        )
        if len(missing) > 100:
            markdown.append(f"- … and {len(missing) - 100} more")
    if errors:
        markdown.extend(
            [
                "",
                "## Validation errors",
                "",
                *[
                    f"- `{item['key']}`: {item['message']}"
                    for item in errors[:100]
                ],
            ]
        )
    output_markdown.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--partial", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    report = audit_results(
        manifest_path=args.manifest,
        results_path=args.results,
        partial=args.partial,
        output_json=args.json,
        output_markdown=args.markdown,
    )
    print(json.dumps(report, sort_keys=True))
    if args.partial:
        return 0 if report["valid"] else 1
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
