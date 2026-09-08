#!/usr/bin/env python3
"""Precompute and validate hand-speed and pinch signals once per video."""

import argparse
import fcntl
import json
import math
import os
import shutil
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "data" / "manifest.json"
DEFAULT_VIDEO_DIR = HERE / "data" / "videos"
DEFAULT_SIGNAL_DIR = HERE / "data" / "signals"
DEFAULT_STATUS = HERE / "data" / "signals_status.jsonl"


class SignalError(RuntimeError):
    """A cached signal does not match the canonical schema."""


def signal_paths(signal_root, episode):
    episode_root = Path(signal_root) / episode
    metrics = episode_root / "metrics"
    return {
        "root": episode_root,
        "metrics": metrics,
        "speed": metrics / f"{episode}_hand_speed.json",
        "pinch": metrics / f"{episode}_pinch_distance.json",
        "debug": episode_root / "debug",
        "plots": episode_root / "plots",
    }


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_speed_cache(path, num_frames):
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SignalError(f"invalid speed JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SignalError(f"{path}: speed cache must be an object")
    expected = {str(idx) for idx in range(int(num_frames))}
    if set(data) != expected:
        raise SignalError(
            f"{path}: speed frame keys do not exactly cover 0..{num_frames - 1}"
        )
    nonzero = 0
    for frame, value in data.items():
        if not _finite_number(value) or float(value) < 0:
            raise SignalError(f"{path}: invalid speed at frame {frame}: {value!r}")
        if float(value) != 0.0:
            nonzero += 1
    return {
        "valid": True,
        "frames": len(data),
        "nonzero_frames": nonzero,
        "path": str(path),
    }


def validate_pinch_cache(path, num_frames):
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SignalError(f"invalid pinch JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SignalError(f"{path}: pinch cache must be an object")
    if int(data.get("total_frames", -1)) != int(num_frames):
        raise SignalError(
            f"{path}: total_frames={data.get('total_frames')} != {num_frames}"
        )
    frames = data.get("frames")
    if not isinstance(frames, dict):
        raise SignalError(f"{path}: missing frames object")
    expected = {str(idx) for idx in range(int(num_frames))}
    if set(frames) != expected:
        raise SignalError(
            f"{path}: pinch frame keys do not exactly cover 0..{num_frames - 1}"
        )
    detected_count = 0
    for frame, record in frames.items():
        if not isinstance(record, dict):
            raise SignalError(f"{path}: frame {frame} is not an object")
        required = {
            "detected",
            "detection_status",
            "pinch_distance_px",
            "is_relative_minimum",
        }
        if not required.issubset(record):
            raise SignalError(
                f"{path}: frame {frame} missing {sorted(required - set(record))}"
            )
        if not isinstance(record["detected"], bool):
            raise SignalError(f"{path}: frame {frame} detected must be bool")
        if not isinstance(record["detection_status"], str):
            raise SignalError(f"{path}: frame {frame} status must be a string")
        if not isinstance(record["is_relative_minimum"], bool):
            raise SignalError(f"{path}: frame {frame} minimum flag must be bool")
        neighbor = record.get("is_relative_minimum_neighbor", False)
        if not isinstance(neighbor, bool):
            raise SignalError(f"{path}: frame {frame} neighbor flag must be bool")
        distance = record["pinch_distance_px"]
        if record["detected"]:
            detected_count += 1
            if not _finite_number(distance) or float(distance) < 0:
                raise SignalError(
                    f"{path}: detected frame {frame} has invalid distance"
                )
        elif distance is not None and not _finite_number(distance):
            raise SignalError(
                f"{path}: undetected frame {frame} has invalid distance"
            )

    selection = data.get("candidate_selection")
    if not isinstance(selection, dict):
        raise SignalError(f"{path}: missing candidate_selection")
    if selection.get("neighborhood") != [-1, 0, 1]:
        raise SignalError(f"{path}: candidate neighborhood must be [-1,0,1]")
    minima = selection.get("relative_minimum_frames", [])
    neighborhood = selection.get("relative_minimum_neighborhood_frames", [])
    for name, values in (("minima", minima), ("neighborhood", neighborhood)):
        if not isinstance(values, list):
            raise SignalError(f"{path}: {name} must be a list")
        clean = [int(value) for value in values]
        if clean != sorted(set(clean)):
            raise SignalError(f"{path}: {name} must be sorted and unique")
        if any(value < 0 or value >= int(num_frames) for value in clean):
            raise SignalError(f"{path}: {name} contains out-of-range frames")
    minima_set = {int(value) for value in minima}
    neighborhood_set = {int(value) for value in neighborhood}
    expected_neighborhood = {
        neighbor
        for minimum in minima_set
        for neighbor in (minimum - 1, minimum, minimum + 1)
        if 0 <= neighbor < int(num_frames)
    }
    if neighborhood_set != expected_neighborhood:
        raise SignalError(
            f"{path}: candidate neighborhood is inconsistent with minima"
        )
    for minimum in minima_set:
        record = frames[str(minimum)]
        if not record["detected"] or not record["is_relative_minimum"]:
            raise SignalError(f"{path}: minimum frame {minimum} is not marked")
    return {
        "valid": True,
        "frames": len(frames),
        "detected_frames": detected_count,
        "minima": len(minima_set),
        "candidate_frames": len(neighborhood_set),
        "path": str(path),
    }


def validate_episode_cache(signal_root, episode, num_frames):
    paths = signal_paths(signal_root, episode)
    return {
        "speed": validate_speed_cache(paths["speed"], num_frames),
        "pinch": validate_pinch_cache(paths["pinch"], num_frames),
    }


def _is_valid(validator, path, num_frames):
    try:
        return True, validator(path, num_frames)
    except (SignalError, OSError, ValueError, TypeError) as exc:
        return False, {"valid": False, "error": str(exc), "path": str(path)}


def append_status(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _select_records(manifest, episodes=None, limit=None):
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
            raise SignalError(f"episodes absent from manifest: {unknown}")
        records = [by_id[episode] for episode in wanted]
    if limit is not None:
        records = records[: int(limit)]
    return records


def _needs(record, signal_root, force=False):
    paths = signal_paths(signal_root, record["episode"])
    n = int(record["num_frames"])
    speed_ok, speed_meta = _is_valid(validate_speed_cache, paths["speed"], n)
    pinch_ok, pinch_meta = _is_valid(validate_pinch_cache, paths["pinch"], n)
    return {
        "paths": paths,
        "speed": force or not speed_ok,
        "pinch": force or not pinch_ok,
        "existing": {"speed": speed_meta, "pinch": pinch_meta},
    }


def precompute_signals(
    manifest_path=DEFAULT_MANIFEST,
    video_dir=DEFAULT_VIDEO_DIR,
    signal_root=DEFAULT_SIGNAL_DIR,
    status_path=DEFAULT_STATUS,
    episodes=None,
    limit=None,
    device="cuda:0",
    debug=False,
    force=False,
    retries=2,
):
    """Process selected videos while reusing one detector and one ViTPose."""
    from manifest import load_manifest

    manifest = load_manifest(manifest_path, validate=True, check_sources=False)
    records = _select_records(manifest, episodes=episodes, limit=limit)
    pending = []
    results = []
    for record in records:
        state = _needs(record, signal_root, force=force)
        video_path = Path(video_dir) / f"{record['episode']}.mp4"
        if not video_path.is_file():
            raise SignalError(f"missing converted video: {video_path}")
        if not state["speed"] and not state["pinch"]:
            result = {
                "episode": record["episode"],
                "status": "reused",
                "signals": state["existing"],
                "timestamp_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "elapsed_s": 0.0,
            }
            append_status(status_path, result)
            results.append(result)
        else:
            pending.append((record, state, video_path))

    if not pending:
        return results

    import sys

    egoloc_root = str(HERE.parent)
    if egoloc_root not in sys.path:
        sys.path.insert(0, egoloc_root)
    import torch
    import trial3

    trial3.setup_hamer_cache()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SignalError(f"CUDA device requested but unavailable: {device}")
    torch_device = torch.device(device)
    detector = trial3.build_detector()
    vitpose = trial3.build_vitpose(torch_device)

    for record, initial_state, video_path in pending:
        started = time.monotonic()
        episode = record["episode"]
        result = {
            "episode": episode,
            "video_path": str(video_path),
            "device": str(device),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempts": 0,
            "computed": [],
        }
        last_error = None
        for attempt in range(1, int(retries) + 2):
            result["attempts"] = attempt
            state = _needs(record, signal_root, force=(force and attempt == 1))
            paths = state["paths"]
            paths["root"].mkdir(parents=True, exist_ok=True)
            try:
                if state["speed"]:
                    trial3.extract_2d_speed_and_visualize(
                        str(video_path),
                        str(paths["root"]),
                        detector,
                        vitpose,
                    )
                    result["computed"].append("speed")
                if state["pinch"]:
                    trial3.extract_pinch_distance_and_visualize(
                        str(video_path),
                        str(paths["root"]),
                        detector,
                        vitpose,
                        str(paths["debug"]),
                    )
                    result["computed"].append("pinch")
                result["signals"] = validate_episode_cache(
                    signal_root, episode, int(record["num_frames"])
                )
                result["status"] = "computed"
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                result["last_error_class"] = type(exc).__name__
                result["last_error"] = str(exc)
                result["traceback"] = traceback.format_exc()
                if attempt > int(retries):
                    break
        if last_error is not None:
            result["status"] = "error"
        if not debug:
            shutil.rmtree(signal_paths(signal_root, episode)["debug"], ignore_errors=True)
            shutil.rmtree(signal_paths(signal_root, episode)["plots"], ignore_errors=True)
        result["elapsed_s"] = time.monotonic() - started
        append_status(status_path, result)
        results.append(result)
    return results


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--signal-dir", default=str(DEFAULT_SIGNAL_DIR))
    parser.add_argument("--status", default=str(DEFAULT_STATUS))
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    results = precompute_signals(
        manifest_path=args.manifest,
        video_dir=args.video_dir,
        signal_root=args.signal_dir,
        status_path=args.status,
        episodes=args.episodes,
        limit=args.limit,
        device=args.device,
        debug=args.debug,
        force=args.force,
        retries=args.retries,
    )
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(json.dumps({"episodes": len(results), "statuses": counts}, sort_keys=True))
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
