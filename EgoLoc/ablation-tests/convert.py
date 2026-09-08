#!/usr/bin/env python3
"""Lossy-safe, resumable OccluBench PNG-to-MP4 conversion."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "data" / "manifest.json"
DEFAULT_VIDEO_DIR = HERE / "data" / "videos"
DEFAULT_STATUS = HERE / "data" / "conversion_manifest.jsonl"


class ConversionError(RuntimeError):
    """A source sequence or converted video failed validation."""


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_frames(rgb_dir, num_frames, compute_hash=True):
    """Validate every source PNG and return ordered paths plus metadata."""
    import cv2

    rgb_dir = Path(rgb_dir)
    names = sorted(
        entry.name
        for entry in os.scandir(rgb_dir)
        if entry.is_file() and entry.name.lower().endswith(".png")
    )
    expected = [f"{idx:06d}.png" for idx in range(int(num_frames))]
    if names != expected:
        raise ConversionError(
            f"{rgb_dir}: expected exact contiguous PNG names "
            f"000000.png..{int(num_frames) - 1:06d}.png"
        )
    digest = hashlib.sha256()
    dimensions = None
    total_bytes = 0
    paths = []
    for name in names:
        path = rgb_dir / name
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ConversionError(f"unreadable source PNG: {path}")
        current = (int(frame.shape[1]), int(frame.shape[0]))
        if dimensions is None:
            dimensions = current
        elif current != dimensions:
            raise ConversionError(
                f"dimension mismatch at {path}: {current} != {dimensions}"
            )
        stat = path.stat()
        total_bytes += int(stat.st_size)
        if compute_hash:
            digest.update(name.encode("ascii"))
            with open(path, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        paths.append(path)
    if dimensions is None:
        raise ConversionError(f"no source frames in {rgb_dir}")
    return {
        "paths": paths,
        "count": len(paths),
        "width": dimensions[0],
        "height": dimensions[1],
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest() if compute_hash else None,
    }


def _similarity(source, decoded):
    import cv2
    import numpy as np

    if source.shape != decoded.shape:
        return 0.0
    # mp4v is lossy. A normalized mean absolute similarity is stable for both
    # textured and nearly uniform frames, unlike correlation alone.
    mae = float(
        np.mean(
            cv2.absdiff(
                source.astype(np.uint8),
                decoded.astype(np.uint8),
            )
        )
    )
    return max(0.0, 1.0 - mae / 255.0)


def _validate_labels(record):
    num_frames = int(record["num_frames"])
    labels = record["labels"]
    for name in ("contact_local", "separate_local"):
        raw_value = labels[name]
        if raw_value is None:
            continue
        value = int(raw_value)
        if not 0 <= value < num_frames:
            raise ConversionError(
                f"episode {record['episode']}: {name}={value} outside video"
            )
    start = int(record["start_frame"])
    if (
        labels["contact_local"] is not None
        and int(labels["contact_global"]) != start + int(labels["contact_local"])
    ):
        raise ConversionError("contact label does not align local/global indices")
    if (
        labels["separate_local"] is not None
        and int(labels["separation_global"])
        != start + int(labels["separate_local"])
    ):
        raise ConversionError("separation label does not align local/global indices")


def validate_video(
    video_path,
    record,
    source_info=None,
    min_similarity=0.60,
):
    """Reopen and fully validate a converted MP4."""
    import cv2

    video_path = Path(video_path)
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise ConversionError(f"missing/empty converted video: {video_path}")
    _validate_labels(record)
    num_frames = int(record["num_frames"])
    if source_info is None:
        source_info = validate_source_frames(
            record["paths"]["rgb_dir"], num_frames, compute_hash=False
        )
    expected_dims = (int(source_info["width"]), int(source_info["height"]))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ConversionError(f"cannot open converted video: {video_path}")
    property_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    property_dims = (
        int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if property_count != num_frames:
        cap.release()
        raise ConversionError(
            f"{video_path}: frame count property {property_count} != {num_frames}"
        )
    if property_dims != expected_dims:
        cap.release()
        raise ConversionError(
            f"{video_path}: dimensions {property_dims} != {expected_dims}"
        )

    spot_indices = {0, num_frames // 2, num_frames - 1}
    spot_indices.update(
        int(value)
        for value in (
            record["labels"]["contact_local"],
            record["labels"]["separate_local"],
        )
        if value is not None
    )
    decoded_spots = {}
    decoded_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (int(frame.shape[1]), int(frame.shape[0])) != expected_dims:
            cap.release()
            raise ConversionError(
                f"{video_path}: decoded frame {decoded_count} changed dimensions"
            )
        if decoded_count in spot_indices:
            decoded_spots[decoded_count] = frame
        decoded_count += 1
    cap.release()
    if decoded_count != num_frames:
        raise ConversionError(
            f"{video_path}: readable frame count {decoded_count} != {num_frames}"
        )
    if set(decoded_spots) != spot_indices:
        raise ConversionError(f"{video_path}: not all alignment spots decoded")

    similarities = {}
    for idx in sorted(spot_indices):
        source = cv2.imread(
            str(Path(record["paths"]["rgb_dir"]) / f"{idx:06d}.png"),
            cv2.IMREAD_COLOR,
        )
        score = _similarity(source, decoded_spots[idx])
        similarities[str(idx)] = score
        if score < float(min_similarity):
            raise ConversionError(
                f"{video_path}: frame {idx} similarity {score:.4f} "
                f"< {min_similarity:.4f}"
            )
    return {
        "frame_count": decoded_count,
        "width": expected_dims[0],
        "height": expected_dims[1],
        "fps": float(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS)),
        "spot_similarity": similarities,
        "labels_aligned": True,
    }


def _write_video(temp_path, source_info, fps):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(temp_path),
        fourcc,
        float(fps),
        (int(source_info["width"]), int(source_info["height"])),
    )
    if not writer.isOpened():
        raise ConversionError(f"cv2 VideoWriter failed for {temp_path}")
    try:
        for path in source_info["paths"]:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ConversionError(f"source became unreadable: {path}")
            writer.write(frame)
    finally:
        writer.release()


def convert_episode(record, output_dir, fps=30.0, force=False):
    started = time.monotonic()
    episode = record["episode"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{episode}.mp4"
    temp_path = output_dir / f".{episode}.{os.getpid()}.{time.time_ns()}.mp4"
    status = {
        "episode": episode,
        "input_path": record["paths"]["rgb_dir"],
        "output_path": str(output_path),
        "fps": float(fps),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        _validate_labels(record)
        source_info = validate_source_frames(
            record["paths"]["rgb_dir"],
            int(record["num_frames"]),
            compute_hash=True,
        )
        status["input"] = {
            key: value
            for key, value in source_info.items()
            if key != "paths"
        }
        if output_path.exists() and not force:
            output_meta = validate_video(output_path, record, source_info)
            status.update(
                status="reused",
                output={
                    **output_meta,
                    "size": int(output_path.stat().st_size),
                    "sha256": sha256_file(output_path),
                },
            )
            return status

        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        _write_video(temp_path, source_info, fps)
        output_meta = validate_video(temp_path, record, source_info)
        with open(temp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        output_meta = validate_video(output_path, record, source_info)
        status.update(
            status="converted",
            output={
                **output_meta,
                "size": int(output_path.stat().st_size),
                "sha256": sha256_file(output_path),
            },
        )
        return status
    except Exception as exc:
        status.update(
            status="error",
            error_class=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return status
    finally:
        status["elapsed_s"] = time.monotonic() - started
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


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


def _normalize_episodes(values):
    if not values:
        return None
    result = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                result.append(f"{int(token):06d}")
    return list(dict.fromkeys(result))


def select_records(manifest, episodes=None, limit=None):
    selected_ids = _normalize_episodes(episodes)
    records = list(manifest["episodes"])
    if selected_ids is not None:
        by_id = {record["episode"]: record for record in records}
        unknown = [episode for episode in selected_ids if episode not in by_id]
        if unknown:
            raise ConversionError(f"episodes absent from manifest: {unknown}")
        records = [by_id[episode] for episode in selected_ids]
    if limit is not None:
        records = records[: int(limit)]
    return records


def convert_episodes(
    manifest_path=DEFAULT_MANIFEST,
    output_dir=DEFAULT_VIDEO_DIR,
    status_path=DEFAULT_STATUS,
    episodes=None,
    limit=None,
    force=False,
    workers=2,
    fps=30.0,
):
    from manifest import load_manifest

    manifest = load_manifest(manifest_path, validate=True, check_sources=False)
    records = select_records(manifest, episodes=episodes, limit=limit)
    results = []
    if int(workers) <= 1:
        for record in records:
            result = convert_episode(record, output_dir, fps=fps, force=force)
            append_status(status_path, result)
            results.append(result)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers)
        ) as executor:
            future_map = {
                executor.submit(
                    convert_episode, record, output_dir, fps, force
                ): record["episode"]
                for record in records
            }
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                append_status(status_path, result)
                results.append(result)
        results.sort(key=lambda item: item["episode"])
    return results


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--status", default=str(DEFAULT_STATUS))
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    results = convert_episodes(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        status_path=args.status,
        episodes=args.episodes,
        limit=args.limit,
        force=args.force,
        workers=args.workers,
        fps=args.fps,
    )
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    summary = {"episodes": len(results), "statuses": counts}
    print(json.dumps(summary, sort_keys=True))
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
