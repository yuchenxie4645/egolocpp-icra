#!/usr/bin/env python3
"""Build and validate the immutable OccluBench ablation manifest."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/home/data_labeling/data/OccluBench")
DEFAULT_LABEL_PATH = DEFAULT_DATASET_ROOT / "label.xlsx"
DEFAULT_SEGMENTS_PATH = DEFAULT_DATASET_ROOT / "segments.csv"
DEFAULT_OUTPUT = HERE / "data" / "manifest.json"
PROTOCOL_VERSION = "occlubench-egoloc-trial2-final-ablation-v2"
EXPECTED_COLUMNS = [
    "episode",
    "start_frame",
    "end_frame",
    "num_frames",
    "source_folder",
    "episode_folder",
    "created_at",
    "contact_local",
    "separate_local",
    "contact_frame",
    "seperate_frame",
]
MODEL_ID = "Qwen/Qwen3.5-9B"
ADAPTERS = {
    "contact": "/home/EgoLoc/training/v3-grpo/contact_sft_grpo_adapter",
    "separation": "/home/EgoLoc/training/v3-grpo/separation_sft_grpo_adapter",
}
UNAVAILABLE_MARKERS = {"", "x", "na", "n/a", "none", "null"}


class ManifestError(ValueError):
    """The source dataset or manifest violates the locked protocol."""


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path, include_hash=True):
    path = Path(path)
    stat = path.stat()
    record = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash and path.is_file():
        record["sha256"] = _sha256_file(path)
    return record


def _as_int(value, field, episode=None):
    prefix = f"episode {episode}: " if episode is not None else ""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ManifestError(f"{prefix}missing {field}")
    if isinstance(value, bool):
        raise ManifestError(f"{prefix}{field} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManifestError(f"{prefix}{field} must be an integer: {value!r}") from exc
    try:
        if float(value) != converted:
            raise ManifestError(f"{prefix}{field} is not integral: {value!r}")
    except (TypeError, ValueError):
        pass
    return converted


def _as_optional_int(value, field, episode=None):
    """Parse an integer label, returning ``None`` for explicit unavailability."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and value.strip().lower() in UNAVAILABLE_MARKERS:
        return None
    return _as_int(value, field, episode)


def _raw_label(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _manifest_hash(document):
    stable = dict(document)
    stable.pop("manifest_content_hash", None)
    stable.pop("created_at_utc", None)
    return hashlib.sha256(_canonical_json(stable).encode("utf-8")).hexdigest()


def _atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _read_sources(label_path, segments_path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "manifest.py requires pandas and openpyxl in egolocxyc"
        ) from exc
    labels = pd.read_excel(label_path, sheet_name="Sheet1")
    if list(labels.columns) != EXPECTED_COLUMNS:
        raise ManifestError(
            f"label columns differ: expected {EXPECTED_COLUMNS}, "
            f"found {list(labels.columns)}"
        )
    segments = pd.read_csv(segments_path, dtype={"episode": str})
    required_segments = [
        "episode",
        "start_frame",
        "end_frame",
        "num_frames",
        "source_folder",
        "episode_folder",
        "created_at",
    ]
    if list(segments.columns) != required_segments:
        raise ManifestError(
            f"segments columns differ: expected {required_segments}, "
            f"found {list(segments.columns)}"
        )
    return labels, segments


def build_manifest(
    output=DEFAULT_OUTPUT,
    label_path=DEFAULT_LABEL_PATH,
    segments_path=DEFAULT_SEGMENTS_PATH,
    dataset_root=DEFAULT_DATASET_ROOT,
    expected_episodes=327,
):
    """Build a deterministic protocol manifest and write it atomically."""
    label_path = Path(label_path)
    segments_path = Path(segments_path)
    dataset_root = Path(dataset_root)
    labels, segments = _read_sources(label_path, segments_path)
    if len(labels) != expected_episodes:
        raise ManifestError(
            f"expected {expected_episodes} label rows, found {len(labels)}"
        )
    if len(segments) != expected_episodes:
        raise ManifestError(
            f"expected {expected_episodes} segment rows, found {len(segments)}"
        )

    label_ids = [_as_int(v, "episode") for v in labels["episode"].tolist()]
    if len(set(label_ids)) != expected_episodes:
        raise ManifestError("label.xlsx episode identifiers are not unique")
    expected_names = {f"{episode:06d}" for episode in label_ids}
    actual_names = {
        child.name
        for child in dataset_root.iterdir()
        if child.is_dir() and len(child.name) == 6 and child.name.isdigit()
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ManifestError(
            f"episode directories differ; missing={missing}, extra={extra}"
        )

    segment_by_episode = {}
    for _, row in segments.iterrows():
        raw = str(row["episode"]).strip()
        try:
            episode_name = f"{int(raw):06d}"
        except ValueError as exc:
            raise ManifestError(f"invalid segments episode {raw!r}") from exc
        if episode_name in segment_by_episode:
            raise ManifestError(f"duplicate segments episode {episode_name}")
        segment_by_episode[episode_name] = row
    if set(segment_by_episode) != expected_names:
        raise ManifestError("segments.csv episodes do not match label.xlsx")

    records = []
    for _, row in labels.sort_values("episode").iterrows():
        episode_index = _as_int(row["episode"], "episode")
        episode = f"{episode_index:06d}"
        start = _as_int(row["start_frame"], "start_frame", episode)
        end = _as_int(row["end_frame"], "end_frame", episode)
        num_frames = _as_int(row["num_frames"], "num_frames", episode)
        if num_frames != end - start + 1:
            raise ManifestError(
                f"episode {episode}: num_frames={num_frames} but "
                f"inclusive range has {end - start + 1}"
            )
        contact_local = _as_optional_int(
            row["contact_local"], "contact_local", episode
        )
        separate_local = _as_optional_int(
            row["separate_local"], "separate_local", episode
        )
        for field, value in (
            ("contact_local", contact_local),
            ("separate_local", separate_local),
        ):
            if value is not None and not 0 <= value < num_frames:
                raise ManifestError(
                    f"episode {episode}: {field}={value} outside "
                    f"[0,{num_frames})"
                )
        contact_global = _as_optional_int(
            row["contact_frame"], "contact_frame", episode
        )
        separation_global = _as_optional_int(
            row["seperate_frame"], "seperate_frame", episode
        )
        if (contact_local is None) != (contact_global is None):
            raise ManifestError(
                f"episode {episode}: contact local/global availability mismatch"
            )
        if (separate_local is None) != (separation_global is None):
            raise ManifestError(
                f"episode {episode}: separation local/global availability mismatch"
            )
        if contact_local is not None and contact_global != start + contact_local:
            raise ManifestError(
                f"episode {episode}: contact global/local mismatch"
            )
        if (
            separate_local is not None
            and separation_global != start + separate_local
        ):
            raise ManifestError(
                f"episode {episode}: separation global/local mismatch"
            )

        segment = segment_by_episode[episode]
        for field, expected in (
            ("start_frame", start),
            ("end_frame", end),
            ("num_frames", num_frames),
        ):
            found = _as_int(segment[field], field, episode)
            if found != expected:
                raise ManifestError(
                    f"episode {episode}: segments {field}={found}, "
                    f"label has {expected}"
                )

        episode_dir = dataset_root / episode
        rgb_dir = episode_dir / "rgb"
        meta_path = episode_dir / "meta.json"
        if not rgb_dir.is_dir() or not meta_path.is_file():
            raise ManifestError(f"episode {episode}: missing rgb/ or meta.json")
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        meta_expected = {
            "episode": episode,
            "start_frame": start,
            "end_frame": end,
            "num_frames": num_frames,
        }
        for field, expected in meta_expected.items():
            found = meta.get(field)
            if field != "episode":
                found = _as_int(found, f"meta.{field}", episode)
            else:
                found = str(found)
            if found != expected:
                raise ManifestError(
                    f"episode {episode}: meta {field}={found!r}, "
                    f"expected {expected!r}"
                )

        png_names = sorted(
            entry.name
            for entry in os.scandir(rgb_dir)
            if entry.is_file() and entry.name.lower().endswith(".png")
        )
        expected_pngs = [f"{idx:06d}.png" for idx in range(num_frames)]
        if png_names != expected_pngs:
            raise ManifestError(
                f"episode {episode}: PNG range/count is not contiguous "
                f"000000.png..{num_frames - 1:06d}.png"
            )

        availability = {
            "contact": contact_local is not None,
            "separation": separate_local is not None,
        }
        unavailable_labels = []
        if not availability["contact"]:
            unavailable_labels.append(
                {
                    "task": "contact",
                    "local_column": "contact_local",
                    "local_value": _raw_label(row["contact_local"]),
                    "global_column": "contact_frame",
                    "global_value": _raw_label(row["contact_frame"]),
                }
            )
        if not availability["separation"]:
            unavailable_labels.append(
                {
                    "task": "separation",
                    "local_column": "separate_local",
                    "local_value": _raw_label(row["separate_local"]),
                    "global_column": "seperate_frame",
                    "global_value": _raw_label(row["seperate_frame"]),
                }
            )

        records.append(
            {
                "episode": episode,
                "episode_index": episode_index,
                "start_frame": start,
                "end_frame": end,
                "num_frames": num_frames,
                "labels": {
                    "contact_local": contact_local,
                    "separate_local": separate_local,
                    "contact_global": contact_global,
                    "separation_global": separation_global,
                },
                "availability": availability,
                "unavailable_labels": unavailable_labels,
                "paths": {
                    "episode_dir": str(episode_dir),
                    "rgb_dir": str(rgb_dir),
                    "meta_json": str(meta_path),
                },
                "source_folder": str(row["source_folder"]),
                "original_episode_folder": str(row["episode_folder"]),
                "source_records": {
                    "meta_json": _source_record(meta_path),
                    "rgb_dir": _source_record(rgb_dir, include_hash=False),
                    "png_count": len(png_names),
                    "first_png": png_names[0],
                    "last_png": png_names[-1],
                },
                "v3_train_membership": False,
                "held_out": True,
            }
        )

    unavailable_contact = [
        record["episode"]
        for record in records
        if not record["availability"]["contact"]
    ]
    unavailable_separation = [
        record["episode"]
        for record in records
        if not record["availability"]["separation"]
    ]
    contact_events = expected_episodes - len(unavailable_contact)
    separation_events = expected_episodes - len(unavailable_separation)
    available_events = contact_events + separation_events
    unavailable_count = len(unavailable_contact) + len(unavailable_separation)
    expected_trials = available_events * 3 * 3
    document = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "OccluBench",
        "label_space": "local_episode_frame_index",
        "source": {
            "dataset_root": str(dataset_root),
            "label_workbook": _source_record(label_path),
            "segments_csv": _source_record(segments_path),
        },
        "model": {
            "base": MODEL_ID,
            "backend": "transformers-nf4",
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "double_quant": True,
            },
            "adapters": {
                task: {
                    "path": path,
                    "rank": 16,
                    "alpha": 32,
                    "merged": False,
                }
                for task, path in ADAPTERS.items()
            },
            "v3_training_sources": [
                "egopat3d",
                "egodex",
                "desktil",
                "dyngrasp",
            ],
            "v3_train_membership": False,
            "held_out": True,
            "held_out_note": (
                "OccluBench is absent from the v3 SFT and GRPO sources."
            ),
        },
        "protocol": {
            "tasks": ["contact", "separation"],
            "modes": ["speed", "pinch", "speed_pinch"],
            "trials": [0, 1, 2],
            "grid_size": 3,
            "grid_frames": 9,
            "grid_topology": "consecutive_average_centered",
            "midpoint_restriction": False,
            "max_feedbacks": 1,
            "primary_output": "closed_loop",
            "prompt_semantics": (
                "exact earliest stable grasp / exact earliest visible release"
            ),
            "negative_option": -1,
            "gap_normalized_speed": True,
            "contiguous_segment_pinch_minima": True,
            "action": "Grasping the object",
            "paired_seed_fields": ["episode", "task", "trial"],
        },
        "expected": {
            "sequences": expected_episodes,
            "label_slots": expected_episodes * 2,
            "events": available_events,
            "contact_events": contact_events,
            "separation_events": separation_events,
            "trials": expected_trials,
            "unavailable_labels": unavailable_count,
        },
        "missing_labels": {
            "contact": unavailable_contact,
            "separation": unavailable_separation,
            "contact_count": len(unavailable_contact),
            "separation_count": len(unavailable_separation),
            "total_count": unavailable_count,
        },
        "episodes": records,
    }
    document["manifest_content_hash"] = _manifest_hash(document)
    validate_manifest_document(document, check_sources=True)
    _atomic_json(output, document)
    return document


def validate_manifest_document(document, check_sources=True):
    """Validate a loaded manifest without mutating it."""
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ManifestError("unsupported or missing protocol_version")
    episodes = document.get("episodes")
    if not isinstance(episodes, list):
        raise ManifestError("episodes must be a list")
    expected = document.get("expected", {})
    expected_n = _as_int(expected.get("sequences"), "expected.sequences")
    if len(episodes) != expected_n:
        raise ManifestError(
            f"manifest has {len(episodes)} episodes, expected {expected_n}"
        )
    identifiers = [record.get("episode") for record in episodes]
    if len(set(identifiers)) != len(identifiers):
        raise ManifestError("manifest episode identifiers are not unique")
    if identifiers != sorted(identifiers):
        raise ManifestError("manifest episodes are not deterministically sorted")
    if expected.get("label_slots") != expected_n * 2:
        raise ManifestError("expected label-slot count is inconsistent")
    missing = document.get("missing_labels", {})
    missing_contact = missing.get("contact", [])
    missing_separation = missing.get("separation", [])
    if not isinstance(missing_contact, list) or not isinstance(
        missing_separation, list
    ):
        raise ManifestError("missing-label episode lists must be arrays")
    unavailable_count = len(missing_contact) + len(missing_separation)
    if (
        missing.get("contact_count") != len(missing_contact)
        or missing.get("separation_count") != len(missing_separation)
        or missing.get("total_count") != unavailable_count
        or expected.get("unavailable_labels") != unavailable_count
    ):
        raise ManifestError("unavailable-label counts are inconsistent")
    contact_events = expected_n - len(missing_contact)
    separation_events = expected_n - len(missing_separation)
    available_events = contact_events + separation_events
    if (
        expected.get("events") != available_events
        or expected.get("contact_events") != contact_events
        or expected.get("separation_events") != separation_events
        or expected.get("trials") != available_events * 3 * 3
    ):
        raise ManifestError("available event/trial counts are inconsistent")
    if document.get("manifest_content_hash") != _manifest_hash(document):
        raise ManifestError("manifest_content_hash does not match content")

    for record in episodes:
        episode = record.get("episode")
        if not isinstance(episode, str) or len(episode) != 6 or not episode.isdigit():
            raise ManifestError(f"invalid episode identifier {episode!r}")
        num_frames = _as_int(record.get("num_frames"), "num_frames", episode)
        labels = record.get("labels", {})
        contact = _as_optional_int(
            labels.get("contact_local"), "contact_local", episode
        )
        separation = _as_optional_int(
            labels.get("separate_local"), "separate_local", episode
        )
        availability = record.get("availability", {})
        if availability != {
            "contact": contact is not None,
            "separation": separation is not None,
        }:
            raise ManifestError(f"episode {episode}: availability flags differ")
        start = _as_int(record.get("start_frame"), "start_frame", episode)
        for task, local, global_key in (
            ("contact", contact, "contact_global"),
            ("separation", separation, "separation_global"),
        ):
            global_value = _as_optional_int(
                labels.get(global_key), global_key, episode
            )
            if (local is None) != (global_value is None):
                raise ManifestError(
                    f"episode {episode}: {task} local/global availability mismatch"
                )
            if local is not None:
                if not 0 <= local < num_frames:
                    raise ManifestError(
                        f"episode {episode}: {task} local label out of range"
                    )
                if global_value != start + local:
                    raise ManifestError(
                        f"episode {episode}: {task} global mismatch"
                    )
        if record.get("v3_train_membership") is not False:
            raise ManifestError(f"episode {episode}: training membership must be false")
        if record.get("held_out") is not True:
            raise ManifestError(f"episode {episode}: held_out must be true")
        if check_sources:
            rgb_dir = Path(record["paths"]["rgb_dir"])
            meta_path = Path(record["paths"]["meta_json"])
            if not rgb_dir.is_dir() or not meta_path.is_file():
                raise ManifestError(f"episode {episode}: source path disappeared")
            png_names = sorted(
                entry.name
                for entry in os.scandir(rgb_dir)
                if entry.is_file() and entry.name.lower().endswith(".png")
            )
            if png_names != [
                f"{idx:06d}.png" for idx in range(num_frames)
            ]:
                raise ManifestError(f"episode {episode}: PNG range changed")
    actual_missing_contact = [
        record["episode"]
        for record in episodes
        if not record["availability"]["contact"]
    ]
    actual_missing_separation = [
        record["episode"]
        for record in episodes
        if not record["availability"]["separation"]
    ]
    if (
        missing_contact != actual_missing_contact
        or missing_separation != actual_missing_separation
    ):
        raise ManifestError("missing-label lists differ from episode records")

    return {
        "valid": True,
        "episodes": len(episodes),
        "events": expected["events"],
        "trials": expected["trials"],
        "manifest_content_hash": document["manifest_content_hash"],
    }


def load_manifest(path=DEFAULT_OUTPUT, validate=True, check_sources=False):
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if validate:
        validate_manifest_document(document, check_sources=check_sources)
    return document


def validate_manifest(path=DEFAULT_OUTPUT, check_sources=True):
    document = load_manifest(path, validate=False)
    return validate_manifest_document(document, check_sources=check_sources)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--label", default=str(DEFAULT_LABEL_PATH))
    parser.add_argument("--segments", default=str(DEFAULT_SEGMENTS_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--expected-episodes", type=int, default=327)
    parser.add_argument(
        "--no-source-check",
        action="store_true",
        help="validate manifest structure without rescanning source PNG names",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.command == "build":
        document = build_manifest(
            output=args.output,
            label_path=args.label,
            segments_path=args.segments,
            dataset_root=args.dataset_root,
            expected_episodes=args.expected_episodes,
        )
        result = validate_manifest_document(
            document, check_sources=not args.no_source_check
        )
    else:
        result = validate_manifest(
            args.output, check_sources=not args.no_source_check
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
