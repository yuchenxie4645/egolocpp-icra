#!/usr/bin/env python3
"""Run the task-major, persistent-backend OccluBench sampling ablation."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Must precede trial3 -> torch import in this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
EGOLOC_ROOT = HERE.parent
for _path in (str(EGOLOC_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import trial3  # noqa: E402
from manifest import load_manifest  # noqa: E402
from results_store import (  # noqa: E402
    ResultsStore,
    result_key,
    stable_config_hash,
)

DEFAULT_MANIFEST = HERE / "data" / "manifest.json"
DEFAULT_VIDEO_DIR = HERE / "data" / "videos"
DEFAULT_SIGNAL_DIR = HERE / "data" / "signals"
DEFAULT_RESULTS = HERE / "results" / "trials.jsonl"
DEFAULT_DRY_RESULTS = HERE / "results" / "dry_run_smoke.jsonl"
TASKS = ("contact", "separation")
MODES = ("speed", "pinch", "speed_pinch")
TRIALS = (0, 1, 2)
ACTION = "Grasping the object"


class PredictionUnavailable(RuntimeError):
    """The model did not return a valid local frame prediction."""


class DryRunBackend:
    """Deterministic fake backend used only for explicit smoke runs."""

    name = "dry-run-fake"

    def __init__(self):
        self.load_count = 1
        self.request_count = 0
        self.adapter_switch_count = 0
        self.current_task = None

    def generate(
        self,
        image_bgr,
        prompt_message,
        task,
        seed=None,
        flag=None,
    ):
        if task != self.current_task:
            self.current_task = task
            self.adapter_switch_count += 1
        self.request_count += 1
        if flag is not None:
            return "1"
        cells = max(1, min(9, int(image_bgr.shape[0] > 0) * 9))
        point = (int(seed or 0) % cells) + 1
        raw = (
            "Deterministic dry-run response. "
            f'{{"points": [{point}]}}'
        )
        return point, raw


def _is_transient(exc):
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    markers = (
        "temporarily unavailable",
        "timed out",
        "timeout",
        "connection reset",
        "rate limit",
        "try again",
    )
    return any(marker in message for marker in markers)


class RetryingBackend:
    """Retry only transient exceptions at the model-call boundary."""

    def __init__(self, backend, attempts=3, initial_delay=1.0):
        self.backend = backend
        self.name = str(
            getattr(backend, "name", None) or backend.__class__.__name__
        )
        self.attempts = int(attempts)
        self.initial_delay = float(initial_delay)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, image_bgr, prompt_message, task, seed=None, flag=None):
        for attempt in range(1, self.attempts + 1):
            try:
                return self.backend.generate(
                    image_bgr,
                    prompt_message,
                    task,
                    seed=seed,
                    flag=flag,
                )
            except Exception as exc:
                if attempt >= self.attempts or not _is_transient(exc):
                    raise
                time.sleep(self.initial_delay * (2 ** (attempt - 1)))
        raise AssertionError("unreachable")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_evaluation_config(manifest, backend_name="transformers-nf4"):
    """Return the exact production protocol snapshot hashed into every row."""
    code_paths = {
        "trial3.py": EGOLOC_ROOT / "trial3.py",
        "backend.py": HERE / "backend.py",
        "runner.py": HERE / "runner.py",
    }
    return {
        "protocol_version": manifest["protocol_version"],
        "dataset": "OccluBench",
        "manifest_content_hash": manifest["manifest_content_hash"],
        "backend": backend_name,
        "model": "Qwen/Qwen3.5-9B",
        "adapters": {
            "contact": {
                "path": "/home/EgoLoc/training/v3-grpo/contact_sft_grpo_adapter",
                "rank": 16,
                "alpha": 32,
                "merged": False,
            },
            "separation": {
                "path": "/home/EgoLoc/training/v3-grpo/separation_sft_grpo_adapter",
                "rank": 16,
                "alpha": 32,
                "merged": False,
            },
        },
        "quantization": {
            "implementation": "bitsandbytes",
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
        },
        "attention": {"text": "sdpa", "vision": "sdpa"},
        "decode": {
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "max_new_tokens": 200,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "enable_thinking": False,
        },
        "grid": {
            "size": 3,
            "frames": 9,
            "max_feedbacks": 1,
            "action": ACTION,
        },
        "trials": [0, 1, 2],
        "paired_seed_fields": ["episode", "task", "trial"],
        "mode_semantics": {
            "speed": {
                "candidates": "top4_lowest_nonzero_hand_speed",
                "fallback": "speed_ranked",
                "visual_feedback": True,
                "speed_feedback": True,
                "reads": ["speed"],
            },
            "pinch": {
                "candidates": "relative_minima_plus_minus_1",
                "fallback": "chronological",
                "visual_feedback": True,
                "speed_feedback": False,
                "reads": ["pinch"],
            },
            "speed_pinch": {
                "candidates": "speed_union_pinch",
                "fallback": "speed_ranked",
                "visual_feedback": True,
                "speed_feedback": True,
                "reads": ["speed", "pinch"],
            },
        },
        "code_sha256": {
            name: _sha256_file(path) for name, path in code_paths.items()
        },
        "held_out": {
            "dataset": "OccluBench",
            "v3_train_membership": False,
        },
    }


def _normalize_choice(values, allowed, label):
    if values is None:
        return list(allowed)
    tokens = []
    for value in values:
        tokens.extend(token.strip() for token in str(value).split(","))
    tokens = [token for token in tokens if token]
    unknown = [token for token in tokens if token not in allowed]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; expected {list(allowed)}")
    return list(dict.fromkeys(tokens))


def _normalize_trials(values):
    if values is None:
        return list(TRIALS)
    trials = []
    for value in values:
        trials.extend(
            int(token) for token in str(value).split(",") if token.strip()
        )
    if any(trial not in TRIALS for trial in trials):
        raise ValueError(f"trials must be a subset of {list(TRIALS)}")
    return list(dict.fromkeys(trials))


def _select_episodes(manifest, episodes=None, limit=None):
    records = list(manifest["episodes"])
    if episodes:
        wanted = []
        for value in episodes:
            wanted.extend(
                f"{int(token):06d}"
                for token in str(value).split(",")
                if token.strip()
            )
        wanted = list(dict.fromkeys(wanted))
        by_id = {record["episode"]: record for record in records}
        unknown = [episode for episode in wanted if episode not in by_id]
        if unknown:
            raise ValueError(f"episodes absent from manifest: {unknown}")
        records = [by_id[episode] for episode in wanted]
    if limit is not None:
        records = records[: int(limit)]
    return records


def _validate_inputs(records, video_dir, signal_dir, modes):
    from convert import validate_video
    from signals import (
        signal_paths,
        validate_pinch_cache,
        validate_speed_cache,
    )

    need_speed = any(mode in ("speed", "speed_pinch") for mode in modes)
    need_pinch = any(mode in ("pinch", "speed_pinch") for mode in modes)
    for record in records:
        if int(record["num_frames"]) < 9:
            raise ValueError(
                f"episode {record['episode']} has fewer than nine frames"
            )
        video_path = Path(video_dir) / f"{record['episode']}.mp4"
        validate_video(video_path, record)
        paths = signal_paths(signal_dir, record["episode"])
        if need_speed:
            validate_speed_cache(paths["speed"], int(record["num_frames"]))
        if need_pinch:
            validate_pinch_cache(paths["pinch"], int(record["num_frames"]))


def _cue_and_fallback(initial_trace, mode, total_frames, anchor):
    used = [int(value) for value in initial_trace.get("used_frame_indices", [])]
    sources = initial_trace.get("candidate_sources", {})
    speed = {int(value) for value in sources.get("speed", [])}
    pinch_all = {int(value) for value in sources.get("pinch", [])}
    # The union is constructed speed-first; overlapping cues are attributed to
    # speed so speed_n + pinch_n + neutral_n always equals the nine-frame budget.
    pinch = pinch_all - speed
    speed_frames = [frame for frame in used if frame in speed]
    pinch_frames = [frame for frame in used if frame in pinch]
    neutral_frames = [
        frame for frame in used if frame not in speed and frame not in pinch
    ]
    midpoint = int(total_frames) // 2
    if anchor == "start":
        outside = [frame for frame in used if frame >= midpoint]
    else:
        outside = [frame for frame in used if frame < midpoint]
    return {
        "cue_coverage": {
            "speed_n": len(speed_frames),
            "pinch_n": len(pinch_frames),
            "neutral_n": len(neutral_frames),
            "overlap_n": len(set(used) & speed & pinch_all),
            "speed_frames": speed_frames,
            "pinch_frames": pinch_frames,
            "neutral_frames": neutral_frames,
        },
        "fallback": {
            "used": bool(neutral_frames),
            "count": len(neutral_frames),
            "frames": neutral_frames,
            "kind": "chronological" if mode == "pinch" else "speed_ranked",
            "source_stage": initial_trace.get("stage"),
        },
        "anchor": {
            "kind": anchor,
            "midpoint": midpoint,
            "outside_preferred_region": outside,
            "relaxed": bool(outside),
        },
    }


def _trace_raw_response(traces):
    for trace in reversed(traces):
        if "raw_response" in trace:
            return trace["raw_response"]
    return None


def _run_trial(
    record,
    task,
    mode,
    trial,
    backend,
    video_dir,
    signal_dir,
):
    episode = record["episode"]
    total_frames = int(record["num_frames"])
    video_path = str(Path(video_dir) / f"{episode}.mp4")
    signal_root = str(Path(signal_dir) / episode)
    seed = trial3.derive_seed(episode, task, trial)
    anchor = "start" if task == "contact" else "end"
    traces = []
    started = time.monotonic()
    base = {
        "key": result_key(episode, task, mode, trial),
        "episode": episode,
        "task": task,
        "mode": mode,
        "trial": int(trial),
        "seed": int(seed),
        "adapter_alias": task,
        "ground_truth_local": int(
            record["labels"][
                "contact_local" if task == "contact" else "separate_local"
            ]
        ),
        "num_frames": total_frames,
        "video_path": video_path,
        "signal_reads": {
            "speed": mode in ("speed", "speed_pinch"),
            "pinch": mode in ("pinch", "speed_pinch"),
        },
        "request_traces": traces,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        initial = trial3.process_task(
            {},
            video_path,
            ACTION,
            3,
            total_frames,
            anchor,
            signal_root,
            pinch_folder=signal_root,
            trial_idx=trial,
            stage=f"{task}_initial",
            mode=mode,
            trial_seed=seed,
            trace_log=traces,
            inference_backend=backend,
        )
        initial_trace = next(
            (
                trace
                for trace in traces
                if trace.get("stage") == f"{task}_initial"
            ),
            {},
        )
        base.update(_cue_and_fallback(initial_trace, mode, total_frames, anchor))
        if initial is None:
            raise PredictionUnavailable("initial localization returned no frame")

        feedback = (
            trial3.feedback_contact
            if task == "contact"
            else trial3.feedback_separation
        )
        frame_kw = (
            {"frame_start": initial}
            if task == "contact"
            else {"frame_end": initial}
        )
        final = feedback(
            credentials={},
            video_path=video_path,
            action=ACTION,
            grid_size=3,
            total_frames=total_frames,
            max_feedbacks=1,
            search_anchor=anchor,
            speed_folder=signal_root,
            pinch_folder=signal_root,
            trial_idx=trial,
            mode=mode,
            trial_seed=seed,
            trace_log=traces,
            inference_backend=backend,
            **frame_kw,
        )
        if final is None or int(final) < 0 or int(final) >= total_frames:
            raise PredictionUnavailable(f"invalid final prediction: {final!r}")
        base.update(
            status="success",
            initial_prediction_frame=int(initial),
            prediction_frame=int(final),
            raw_response=_trace_raw_response(traces),
            parse_error=False,
            error=None,
        )
    except Exception as exc:
        if "cue_coverage" not in base:
            initial_trace = next(
                (
                    trace
                    for trace in traces
                    if trace.get("stage") == f"{task}_initial"
                ),
                {},
            )
            base.update(
                _cue_and_fallback(initial_trace, mode, total_frames, anchor)
            )
        base.update(
            status="terminal_failure",
            prediction_frame=None,
            raw_response=_trace_raw_response(traces),
            parse_error=isinstance(exc, PredictionUnavailable),
            error={
                "class": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    base["latency_s"] = time.monotonic() - started
    return base


def _create_backend(dry_run=False):
    if dry_run:
        return DryRunBackend()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError(
            "Production evaluation is locked to CUDA_VISIBLE_DEVICES=1"
        )
    from backend import TransformersBackend

    return TransformersBackend()


def _safe_dry_path(path):
    path = Path(path) if path else DEFAULT_DRY_RESULTS
    lowered = path.name.lower()
    if "dry" in lowered or "smoke" in lowered:
        return path
    suffix = path.suffix or ".jsonl"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}.dry-run{suffix}")


def run_evaluation(
    manifest_path=DEFAULT_MANIFEST,
    video_dir=DEFAULT_VIDEO_DIR,
    signal_dir=DEFAULT_SIGNAL_DIR,
    results_path=None,
    episodes=None,
    limit_episodes=None,
    tasks=None,
    modes=None,
    trials=None,
    resume=True,
    dry_run=False,
    backend=None,
    validate_inputs=True,
):
    manifest = load_manifest(manifest_path, validate=True, check_sources=False)
    tasks = _normalize_choice(tasks, TASKS, "tasks")
    modes = _normalize_choice(modes, MODES, "modes")
    trials = _normalize_trials(trials)
    records = _select_episodes(
        manifest, episodes=episodes, limit=limit_episodes
    )
    if validate_inputs:
        _validate_inputs(records, video_dir, signal_dir, modes)
    if backend is None:
        backend = _create_backend(dry_run=dry_run)
    if int(getattr(backend, "load_count", 1)) != 1:
        raise RuntimeError("persistent backend must report load_count == 1")
    retrying_backend = RetryingBackend(backend)

    if dry_run:
        results_path = _safe_dry_path(results_path)
    else:
        results_path = Path(results_path or DEFAULT_RESULTS)
    if not resume and Path(results_path).exists() and Path(results_path).stat().st_size:
        raise RuntimeError(
            f"results already exist at {results_path}; use --resume or a new path"
        )
    config = build_evaluation_config(
        manifest,
        backend_name=(
            "dry-run-fake"
            if dry_run
            else "transformers-nf4"
        ),
    )
    config_hash = stable_config_hash(config)
    store = ResultsStore(results_path, config_hash=config_hash)

    planned = 0
    written = 0
    skipped = 0
    ignored_unavailable_events = 0
    status_counts = {"success": 0, "terminal_failure": 0}
    # Locked task-major order minimizes PEFT set_adapter calls.
    for task in tasks:
        for record in records:
            label_key = (
                "contact_local" if task == "contact" else "separate_local"
            )
            if record["labels"].get(label_key) is None:
                ignored_unavailable_events += 1
                continue
            for trial in trials:
                for mode in modes:
                    planned += 1
                    key = result_key(record["episode"], task, mode, trial)
                    if key in store:
                        skipped += 1
                        continue
                    trial_record = _run_trial(
                        record,
                        task,
                        mode,
                        trial,
                        retrying_backend,
                        video_dir,
                        signal_dir,
                    )
                    trial_record["config_hash"] = config_hash
                    trial_record["config_snapshot"] = config
                    store.append(trial_record)
                    status_counts[trial_record["status"]] += 1
                    written += 1

    return {
        "results_path": str(results_path),
        "planned": planned,
        "written": written,
        "skipped": skipped,
        "ignored_unavailable_events": ignored_unavailable_events,
        "completed_in_file": len(store.completed_keys),
        "statuses_written": status_counts,
        "config_hash": config_hash,
        "backend": str(
            getattr(backend, "name", None) or backend.__class__.__name__
        ),
        "backend_load_count": int(getattr(backend, "load_count", 1)),
        "backend_request_count": int(getattr(backend, "request_count", 0)),
        "adapter_switch_count": int(
            getattr(backend, "adapter_switch_count", 0)
        ),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--signal-dir", default=str(DEFAULT_SIGNAL_DIR))
    parser.add_argument("--results")
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--modes", nargs="*")
    parser.add_argument("--trials", nargs="*")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backend-load-smoke",
        action="store_true",
        help="load the persistent backend once, print provenance, and exit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.backend_load_smoke:
        backend = _create_backend(dry_run=args.dry_run)
        print(
            json.dumps(
                {
                    "backend": backend.name,
                    "load_count": backend.load_count,
                    "request_count": backend.request_count,
                    "adapter_switch_count": backend.adapter_switch_count,
                    "load_provenance": getattr(
                        backend, "load_provenance", None
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    summary = run_evaluation(
        manifest_path=args.manifest,
        video_dir=args.video_dir,
        signal_dir=args.signal_dir,
        results_path=args.results,
        episodes=args.episodes,
        limit_episodes=args.limit_episodes,
        tasks=args.tasks,
        modes=args.modes,
        trials=args.trials,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, sort_keys=True))
    if not args.dry_run and summary["adapter_switch_count"] > 2:
        raise RuntimeError(
            "task-major adapter switch invariant violated (>2 switches)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
