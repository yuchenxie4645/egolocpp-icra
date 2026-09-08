import json

import pytest

import signals


def write_valid_caches(root, episode="000000", frames=12):
    paths = signals.signal_paths(root, episode)
    paths["metrics"].mkdir(parents=True)
    speed = {str(index): float(index) for index in range(frames)}
    paths["speed"].write_text(json.dumps(speed))
    minima = [5]
    neighborhood = [4, 5, 6]
    pinch_frames = {}
    for index in range(frames):
        detected = index in (4, 5, 6)
        pinch_frames[str(index)] = {
            "detected": detected,
            "detection_status": "detected" if detected else "no_hand_detected",
            "pinch_distance_px": float(abs(index - 5) + 1) if detected else None,
            "is_relative_minimum": index == 5,
            "is_relative_minimum_neighbor": index in (4, 6),
        }
    pinch = {
        "video_name": episode,
        "total_frames": frames,
        "candidate_selection": {
            "strategy": "relative_minimum_pinch_distance_with_neighbors",
            "contiguous_detection_segments": True,
            "neighborhood": [-1, 0, 1],
            "relative_minimum_frames": minima,
            "relative_minimum_neighborhood_frames": neighborhood,
        },
        "frames": pinch_frames,
    }
    paths["pinch"].write_text(json.dumps(pinch))
    paths["metadata"].write_text(
        json.dumps(
            {
                "schema_version": signals.SIGNAL_SCHEMA_VERSION,
                "total_frames": frames,
                "speed": (
                    "euclidean_hand_center_displacement_per_elapsed_frame"
                ),
                "pinch": (
                    "relative_minima_within_contiguous_detection_segments"
                ),
            }
        )
    )
    return paths


def test_valid_speed_and_pinch_cache_schema(tmp_path):
    paths = write_valid_caches(tmp_path)

    speed = signals.validate_speed_cache(paths["speed"], 12)
    pinch = signals.validate_pinch_cache(paths["pinch"], 12)
    metadata = signals.validate_signal_metadata(paths["metadata"], 12)

    assert speed["frames"] == 12
    assert speed["nonzero_frames"] == 11
    assert pinch["frames"] == 12
    assert pinch["minima"] == 1
    assert pinch["candidate_frames"] == 3
    assert metadata["schema_version"] == signals.SIGNAL_SCHEMA_VERSION


def test_speed_cache_requires_exact_frame_keys(tmp_path):
    paths = write_valid_caches(tmp_path)
    data = json.loads(paths["speed"].read_text())
    data.pop("11")
    paths["speed"].write_text(json.dumps(data))

    with pytest.raises(signals.SignalError, match="exactly cover"):
        signals.validate_speed_cache(paths["speed"], 12)


def test_pinch_cache_rejects_out_of_range_candidate(tmp_path):
    paths = write_valid_caches(tmp_path)
    data = json.loads(paths["pinch"].read_text())
    data["candidate_selection"]["relative_minimum_neighborhood_frames"].append(12)
    paths["pinch"].write_text(json.dumps(data))

    with pytest.raises(signals.SignalError, match="out-of-range"):
        signals.validate_pinch_cache(paths["pinch"], 12)
