"""Mode isolation: each sampling mode reads only its own signal JSON.

The "poisoned file" technique: the JSON of the signal a mode must NOT touch
contains syntactically invalid JSON, so any attempt to open and parse it
raises json.JSONDecodeError and fails the test. A clean run proves isolation.
"""

import pytest

import trial3

SPEED_TOP4 = [1, 4, 8, 3]
SPEED_POOL = [1, 4, 8, 3, 7, 6, 2, 0, 9, 5]
PINCH_CANDIDATES = [9, 10, 11, 24, 25, 26]


def test_speed_mode_never_opens_pinch_json(episode):
    episode.write_speed_json()
    episode.write_pinch_json(corrupt=True)

    result = trial3.build_mode_candidates(
        "speed", episode.speed_path, episode.pinch_path,
        total_frames=episode.total_frames,
    )

    assert result["mode"] == "speed"
    assert result["candidates"] == SPEED_TOP4
    assert result["total_pool"] == SPEED_POOL
    assert result["invalid_list"] == []
    assert result["coverage"] == {"speed": 4, "pinch": 0, "neutral": 0}


def test_process_task_speed_mode_ignores_poisoned_pinch_folder(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json(corrupt=True)
    episode.make_video()

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        pinch_folder=episode.root, trace_out=trace, mode="speed",
    )

    assert selected == trace["used_frame_indices"][0]
    assert trace["total_pool"] == SPEED_POOL
    assert trace["used_frame_indices"] == [0, 1, 2, 3, 4, 6, 7, 8, 9]


def test_pinch_mode_never_opens_speed_json(episode):
    episode.write_speed_json(corrupt=True)
    episode.write_pinch_json()

    result = trial3.build_mode_candidates(
        "pinch", episode.speed_path, episode.pinch_path,
        total_frames=episode.total_frames,
    )

    assert result["mode"] == "pinch"
    assert result["candidates"] == PINCH_CANDIDATES
    assert result["total_pool"] == list(range(40))
    assert result["invalid_list"] == []
    assert result["coverage"] == {"speed": 0, "pinch": 6, "neutral": 0}


def test_process_task_pinch_mode_ignores_poisoned_speed_folder(episode, stub_vlm):
    episode.write_speed_json(corrupt=True)
    episode.write_pinch_json()
    episode.make_video()

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        pinch_folder=episode.root, trace_out=trace, mode="pinch",
    )

    assert selected == trace["used_frame_indices"][0]
    assert trace["candidates"] == PINCH_CANDIDATES
    assert trace["total_pool"] == list(range(40))


def test_speed_pinch_matches_legacy_merge_and_golden_grids(episode):
    episode.write_speed_json()
    episode.write_pinch_json()

    result = trial3.build_mode_candidates(
        "speed_pinch", episode.speed_path, episode.pinch_path,
        total_frames=episode.total_frames,
    )

    # Golden values for this fixture: speed top-4 plus the pinch minima
    # neighborhoods, deduped in speed-then-pinch order; speed-ranked pool.
    assert result["candidates"] == [1, 4, 8, 3, 9, 10, 11, 24, 25, 26]
    assert result["total_pool"] == SPEED_POOL
    assert result["coverage"] == {"speed": 4, "pinch": 6, "neutral": 0}

    def grid_for(mode):
        data = trial3.build_mode_candidates(
            mode, episode.speed_path, episode.pinch_path,
            total_frames=episode.total_frames,
        )
        return trial3.build_direct_candidate_grid(
            data["candidates"], data["total_pool"], 3, episode.total_frames,
            "start", invalid_list=data["invalid_list"], min_index=None,
        )

    assert grid_for("speed") == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert grid_for("pinch") == [0, 1, 2, 3, 4, 5, 9, 10, 11]
    assert grid_for("speed_pinch") == [1, 3, 4, 6, 7, 8, 9, 10, 11]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        trial3.validate_sampling_mode("greedy")
    with pytest.raises(ValueError):
        trial3.build_mode_candidates("bogus", None, None, total_frames=10)
    assert trial3.validate_sampling_mode("speed_pinch") == "speed_pinch"
