"""Trace collection contract for process_task / vlm_request.

When trace_out is provided, the documented keys are populated, the recorded
selection matches the returned frame, and separate calls never write into
each other's dicts. Passing trace_out=None changes nothing vs the baseline.
"""

import numpy as np

import trial3

DOCUMENTED_KEYS = {
    "stage",
    "mode",
    "candidates",
    "total_pool",
    "used_frame_indices",
    "raw_response",
    "latency_s",
    "backend_used",
    "seed",
    "flag",
}


def test_trace_keys_and_selection_match(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        pinch_folder=episode.root, trace_out=trace, trial_seed=777,
    )

    assert DOCUMENTED_KEYS.issubset(trace.keys())
    assert trace["stage"] == "start_initial"
    assert trace["mode"] == "speed_pinch"
    assert trace["candidates"] == [1, 4, 8, 3, 9, 10, 11, 24, 25, 26]
    assert trace["total_pool"] == [1, 4, 8, 3, 7, 6, 2, 0, 9, 5]
    grid = trace["used_frame_indices"]
    assert grid == sorted(grid)
    assert len(grid) == 9
    assert selected in grid
    assert selected == trace["used_frame_indices"][0]
    assert trace["raw_response"] == stub_vlm.content
    assert trace["backend_used"] == "vllm"
    assert trace["seed"] == 777
    assert trace["flag"] is None
    assert trace["latency_s"] == 0.0


def test_trace_stage_and_flag_on_feedback(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        frame_index=7, flag="feedback", pinch_folder=episode.root,
        trace_out=trace,
    )

    assert trace["stage"] == "start_visual_feedback"
    assert trace["flag"] == "feedback"
    # Speed candidates after frame 7 are [8, 9] (only frames > 7 carry
    # non-zero speed); merged with the pinch neighborhoods, deduped.
    assert trace["candidates"] == [8, 9, 10, 11, 24, 25, 26]
    # Frames 24+ fall outside the "start" anchor on the first evidence pass
    # (which was non-empty, so no relaxation), leaving [8, 9, 10, 11] as
    # evidence; the rest fills by ascending context rings around them.
    assert trace["used_frame_indices"] == [8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert selected == trace["used_frame_indices"][0]


def test_trace_dicts_are_isolated_per_call(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    trace_a, trace_b = {}, {}
    selected_a = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        trace_out=trace_a, trial_seed=1,
    )
    selected_b = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        trace_out=trace_b, trial_seed=2,
    )

    assert trace_a["seed"] == 1
    assert trace_b["seed"] == 2
    assert trace_a is not trace_b
    assert selected_a == selected_b


def test_trace_none_matches_baseline(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    traced = {}
    with_trace = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root, trace_out=traced,
    )
    without_trace = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
    )

    assert with_trace == without_trace
    assert traced["used_frame_indices"] == [0, 1, 2, 3, 4, 6, 7, 8, 9]


def test_persistent_backend_object_propagates_and_has_stable_trace_name():
    class RecordingBackend:
        name = "recording-persistent"

        def __init__(self):
            self.calls = []

        def generate(self, image_bgr, prompt_message, task, seed=None, flag=None):
            self.calls.append((task, seed, flag))
            if flag is not None:
                return "1"
            return 2, 'analysis {"points": [2]}'

    backend = RecordingBackend()
    trace = {}
    result = trial3.vlm_request(
        np.zeros((8, 8, 3), dtype=np.uint8),
        "prompt",
        "contact",
        backend=backend,
        seed=99,
        trace_out=trace,
    )

    assert result == (2, 'analysis {"points": [2]}')
    assert backend.calls == [("contact", 99, None)]
    assert trace["backend_used"] == "recording-persistent"
    assert trace["seed"] == 99


def test_feedback_captures_visual_and_correction_requests(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()
    traces = []

    result = trial3.feedback_contact(
        {},
        episode.video_path,
        "Grasping the object",
        3,
        episode.total_frames,
        1,
        1,
        "start",
        episode.root,
        pinch_folder=episode.root,
        trial_idx=0,
        trial_seed=123,
        trace_log=traces,
    )

    assert result is not None
    assert len(traces) == 2
    assert traces[0]["used_frame_indices"] == [1]
    assert len(traces[1]["used_frame_indices"]) == 9
    assert traces[0] is not traces[1]
    assert all(trace["seed"] == 123 for trace in traces)


def test_process_task_forwards_backend_object_unchanged(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()
    backend = object()

    trial3.process_task(
        {},
        episode.video_path,
        "Grasping the object",
        3,
        episode.total_frames,
        "start",
        episode.root,
        pinch_folder=episode.root,
        inference_backend=backend,
    )

    assert stub_vlm.calls[0]["backend"] is backend
