"""Missing-label and missing-file behavior.

select_pinch_keyframes tolerates a missing pinch JSON (returns []); the
speed JSON path raises the same error type as before the refactor (the open
of a nonexistent file); build_mode_candidates tolerates pinch_json_path=None
in speed mode.
"""

import pytest

import trial3


def test_select_pinch_keyframes_missing_file_returns_empty():
    assert trial3.select_pinch_keyframes("/nonexistent/path/pinch.json") == []


def test_select_top_n_missing_speed_json_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        trial3.select_top_n_frames_from_json("/nonexistent/path/speed.json", 4)


def test_build_mode_candidates_speed_missing_json_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        trial3.build_mode_candidates(
            "speed", "/nonexistent/path/speed.json", None, total_frames=10
        )


def test_speed_mode_tolerates_none_pinch_path(episode):
    episode.write_speed_json()

    result = trial3.build_mode_candidates(
        "speed", episode.speed_path, None, total_frames=episode.total_frames
    )

    assert result["candidates"] == [1, 4, 8, 3]
    assert result["invalid_list"] == []


def test_pinch_mode_tolerates_none_paths():
    result = trial3.build_mode_candidates("pinch", None, None, total_frames=12)

    assert result["candidates"] == []
    assert result["total_pool"] == list(range(12))
    assert result["invalid_list"] == []


def test_speed_pinch_tolerates_none_pinch_path(episode):
    episode.write_speed_json()

    result = trial3.build_mode_candidates(
        "speed_pinch", episode.speed_path, None, total_frames=episode.total_frames
    )

    # No pinch JSON available: the merge degenerates to speed-only candidates.
    assert result["candidates"] == [1, 4, 8, 3]


def test_process_task_speed_mode_without_pinch_folder(episode, stub_vlm):
    episode.write_speed_json()
    episode.make_video()

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        trace_out=trace, mode="speed",
    )

    assert selected is not None
    assert selected == trace["used_frame_indices"][0]
