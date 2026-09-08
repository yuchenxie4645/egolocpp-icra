"""Backward compatibility of the mode-aware refactor.

The pre-refactor candidate algorithm (top-4 speed frames union pinch
keyframes, fed into build_direct_candidate_grid with per-branch invalid_list
and min_index) is transcribed inline below and compared against the
refactored process_task running in its default mode="speed_pinch".
"""

import inspect

import pytest

import trial3

# Legacy signatures: everything before the appended mode-aware keywords must
# stay byte-compatible so existing positional/keyword call sites keep working.
LEGACY_POSITIONAL = {
    "process_task": [
        "credentials", "video_path", "action", "grid_size", "total_frames",
        "search_anchor", "speed_folder", "frame_index", "flag",
        "pinch_folder", "trial_idx", "stage",
    ],
    "determine_by_state": [
        "credentials", "video_path", "action", "grid_size", "total_frames",
        "frame_index", "search_anchor", "speed_folder", "pinch_folder",
        "trial_idx", "stage",
    ],
    "determine_by_speed": [
        "credentials", "video_path", "action", "grid_size", "total_frames",
        "frame_index", "search_anchor", "speed_folder", "pinch_folder",
        "trial_idx", "stage",
    ],
    "feedback_contact": [
        "credentials", "video_path", "action", "grid_size", "total_frames",
        "frame_start", "max_feedbacks", "search_anchor", "speed_folder",
        "pinch_folder", "trial_idx",
    ],
    "feedback_separation": [
        "credentials", "video_path", "action", "grid_size", "total_frames",
        "frame_end", "max_feedbacks", "search_anchor", "speed_folder",
        "pinch_folder", "trial_idx",
    ],
    "convert_video": [
        "video_path", "action", "credentials", "grid_size", "speed_folder",
        "max_feedbacks", "repeat_times", "pinch_folder",
    ],
}

NEW_KEYWORDS = {
    "process_task": [
        "mode", "trial_seed", "trace_out", "trace_log", "inference_backend",
    ],
    "determine_by_state": [
        "mode", "trial_seed", "trace_log", "inference_backend",
    ],
    "determine_by_speed": [
        "mode", "trial_seed", "trace_log", "inference_backend",
    ],
    "feedback_contact": [
        "mode", "trial_seed", "trace_log", "inference_backend",
    ],
    "feedback_separation": [
        "mode", "trial_seed", "trace_log", "inference_backend",
    ],
    "convert_video": [
        "mode", "trial_seed", "trace_log", "inference_backend",
    ],
}


def legacy_candidates(json_path, pinch_json_path, total_frames, anchor, frame_index=None, flag=None):
    """Transcription of the pre-refactor process_task candidate logic."""
    if frame_index is None:
        speed_selected = trial3.select_top_n_frames_from_json(json_path, 4)
        total_indices = trial3.select_top_n_frames_from_json(json_path, total_frames)
        invalid_list = []
        min_index = None
    elif flag == "feedback":
        invalid_list, speed_selected = trial3.select_top_n_frames_from_json(
            json_path, 4, frame_index, flag, receive_flag="right"
        )
        total_indices = trial3.select_top_n_frames_from_json(
            json_path, total_frames, frame_index, flag
        )
        min_index = frame_index
    elif flag == "speed":
        invalid_list, speed_selected = trial3.select_top_n_frames_from_json(
            json_path, 4, frame_index, flag, receive_flag="right"
        )
        total_indices = trial3.select_top_n_frames_from_json(
            json_path, total_frames, frame_index, flag
        )
        min_index = None
    else:
        invalid_list, speed_selected = trial3.select_top_n_frames_from_json(
            json_path, 4, frame_index, receive_flag="right"
        )
        total_indices = trial3.select_top_n_frames_from_json(
            json_path, total_frames, frame_index, flag
        )
        min_index = None

    pinch_selected = trial3.select_pinch_keyframes(pinch_json_path)
    selected = trial3._merge_speed_and_pinch(speed_selected, pinch_selected)
    grid = trial3.build_direct_candidate_grid(
        selected, total_indices, 3, total_frames, anchor,
        invalid_list=invalid_list, min_index=min_index,
    )
    return selected, total_indices, grid


def test_initial_localization_matches_legacy(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    legacy_selected, legacy_pool, legacy_grid = legacy_candidates(
        episode.speed_path, episode.pinch_path, episode.total_frames, "start"
    )

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        pinch_folder=episode.root, trace_out=trace,
    )

    assert trace["candidates"] == legacy_selected
    assert trace["total_pool"] == legacy_pool
    assert trace["used_frame_indices"] == legacy_grid
    assert selected == legacy_grid[0]


def test_visual_feedback_matches_legacy(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    legacy_selected, legacy_pool, legacy_grid = legacy_candidates(
        episode.speed_path, episode.pinch_path, episode.total_frames,
        "start", frame_index=7, flag="feedback",
    )

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        frame_index=7, flag="feedback", pinch_folder=episode.root,
        trace_out=trace,
    )

    assert trace["candidates"] == legacy_selected
    assert trace["total_pool"] == legacy_pool
    assert trace["used_frame_indices"] == legacy_grid
    assert selected == legacy_grid[0]


def test_speed_feedback_matches_legacy(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    legacy_selected, legacy_pool, legacy_grid = legacy_candidates(
        episode.speed_path, episode.pinch_path, episode.total_frames,
        "start", frame_index=2.5, flag="speed",
    )

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        frame_index=2.5, flag="speed", pinch_folder=episode.root,
        trace_out=trace,
    )

    assert trace["candidates"] == legacy_selected
    assert trace["total_pool"] == legacy_pool
    assert trace["used_frame_indices"] == legacy_grid
    assert selected == legacy_grid[0]


def test_fault_feedback_matches_legacy(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    legacy_selected, legacy_pool, legacy_grid = legacy_candidates(
        episode.speed_path, episode.pinch_path, episode.total_frames,
        "start", frame_index=7, flag="fault",
    )

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        frame_index=7, flag="fault", pinch_folder=episode.root,
        trace_out=trace,
    )

    assert trace["candidates"] == legacy_selected
    assert trace["total_pool"] == legacy_pool
    assert trace["used_frame_indices"] == legacy_grid
    assert selected == legacy_grid[0]


def test_new_params_are_optional_and_appended():
    for fn_name, extras in NEW_KEYWORDS.items():
        fn = getattr(trial3, fn_name)
        names = list(inspect.signature(fn).parameters)
        assert names[-len(extras):] == extras
        for name in extras:
            param = inspect.signature(fn).parameters[name]
            assert param.default is not inspect.Parameter.empty
            expected = {
                "mode": "speed_pinch",
                "inference_backend": "vllm",
            }.get(name)
            assert param.default == expected
        assert names[-1] == "inference_backend"


def test_legacy_positional_prefix_unchanged():
    for fn_name, expected in LEGACY_POSITIONAL.items():
        names = list(inspect.signature(getattr(trial3, fn_name)).parameters)
        assert names[:len(expected)] == expected


def test_scene_understanding_signature_unchanged():
    names = list(inspect.signature(trial3.scene_understanding).parameters)

    assert names == ["credentials", "frame", "prompt_message", "flag", "task"]


def test_scene_understanding_delegates_to_vlm_request(monkeypatch):
    captured = {}

    def fake_vlm_request(image_bgr, prompt_message, task, backend="vllm", seed=None, flag=None, trace_out=None):
        captured.update(
            backend=backend, seed=seed, flag=flag, task=task,
            prompt_message=prompt_message,
        )
        return 3, "raw text"

    monkeypatch.setattr(trial3, "vlm_request", fake_vlm_request)

    point, reason = trial3.scene_understanding({}, "img", "prompt", flag=None, task="separation")

    assert (point, reason) == (3, "raw text")
    assert captured["backend"] == "vllm"
    assert captured["seed"] is None
    assert captured["flag"] is None
    assert captured["task"] == "separation"
    assert captured["prompt_message"] == "prompt"


def test_transformers_backend_import_is_lazy_and_parser_is_wired():
    import backend

    assert hasattr(backend, "TransformersBackend")
    assert backend.TransformersBackend.name == "transformers-nf4"
    assert backend.parse_point('analysis {"points": [4]}') == 4
    assert backend.parse_point("not JSON") == -1
