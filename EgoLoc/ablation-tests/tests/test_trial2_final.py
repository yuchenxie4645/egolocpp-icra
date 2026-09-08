import json

import pytest

import trial_2_final as trial2_final


class RecordingBackend:
    name = "recording-transformers"

    def __init__(self, point=5, visual="1"):
        self.point = point
        self.visual = visual
        self.calls = []

    def generate(
        self,
        image_bgr,
        prompt_message,
        task,
        seed=None,
        flag=None,
    ):
        self.calls.append(
            {
                "prompt": prompt_message,
                "task": task,
                "seed": seed,
                "flag": flag,
            }
        )
        if flag is not None:
            return self.visual
        return self.point, f'analysis {{"points": [{self.point}]}}'


def test_exact_prompt_retains_negative_option():
    contact = trial2_final.build_exact_localization_prompt(
        "start", "Grasping the object"
    )
    separation = trial2_final.build_exact_localization_prompt(
        "end", "Grasping the object"
    )
    assert "exact earliest frame" in contact
    assert "stable grasp" in contact
    assert "visible release" in separation
    assert "number -1" in contact
    assert "number -1" in separation
    assert "closest to the moment" not in contact + separation


def test_gap_normalized_speed():
    centers = {0: (0.0, 0.0), 1: (3.0, 4.0), 6: (18.0, 24.0)}
    speed = trial2_final.compute_gap_normalized_speed(centers, 7)
    assert speed[0] == 0.0
    assert speed[1] == 5.0
    assert speed[2] == 0.0
    assert speed[6] == 5.0


def test_pinch_minima_split_at_missing_hand_gaps():
    values = [(1, 3.0), (2, 1.0), (7, 2.0), (8, 1.5)]
    assert trial2_final._relative_minimum_frame_indices(values) == [2, 8]
    values = [(1, 3.0), (2, 1.0), (3, 2.0), (8, 4.0)]
    assert trial2_final._relative_minimum_frame_indices(values) == [2, 8]


def test_consecutive_window_is_unique_and_clamped():
    middle, meta = trial2_final.build_consecutive_candidate_grid(
        [18, 20], 3, 40
    )
    assert middle == list(range(15, 24))
    assert meta["topology"] == "consecutive"
    assert meta["midpoint_restriction"] is False

    left, _ = trial2_final.build_consecutive_candidate_grid([0], 3, 40)
    right, _ = trial2_final.build_consecutive_candidate_grid([39], 3, 40)
    assert left == list(range(9))
    assert right == list(range(31, 40))
    assert len(set(left)) == len(set(right)) == 9


def test_feedback_window_honors_minimum_or_records_relaxation():
    frames, meta = trial2_final.build_consecutive_candidate_grid(
        [20], 3, 40, min_index=20
    )
    assert frames == list(range(21, 30))
    assert meta["min_index_relaxed"] is False

    frames, meta = trial2_final.build_consecutive_candidate_grid(
        [38], 3, 40, min_index=38
    )
    assert frames == list(range(31, 40))
    assert meta["min_index_relaxed"] is True


def test_mode_isolation_and_no_midpoint_filter(episode):
    speed_path = episode.write_speed_json()
    pinch_path = episode.write_pinch_json()

    speed = trial2_final.build_mode_candidates(
        "speed",
        speed_path,
        "/path/that/must/not/be/read.json",
        total_frames=episode.total_frames,
    )
    pinch = trial2_final.build_mode_candidates(
        "pinch",
        "/path/that/must/not/be/read.json",
        pinch_path,
        total_frames=episode.total_frames,
    )
    both = trial2_final.build_mode_candidates(
        "speed_pinch",
        speed_path,
        pinch_path,
        total_frames=episode.total_frames,
    )

    assert speed["candidate_sources"]["pinch"] == []
    assert pinch["candidate_sources"]["speed"] == []
    assert speed["signal_reads"] == {"speed": True, "pinch": False}
    assert pinch["signal_reads"] == {"speed": False, "pinch": True}
    assert 25 in pinch["candidates"]
    assert set(speed["cue_candidates"]).issubset(both["cue_candidates"])
    assert set(pinch["cue_candidates"]).issubset(both["cue_candidates"])


def test_process_task_uses_same_unrestricted_window_for_both_tasks(episode):
    video = episode.make_video()
    episode.write_speed_json()
    backend = RecordingBackend(point=5)
    contact_traces = []
    separation_traces = []
    common = dict(
        credentials={},
        video_path=video,
        action="Grasping the object",
        grid_size=3,
        total_frames=episode.total_frames,
        speed_folder=episode.root,
        mode="speed",
        trial_seed=123,
        inference_backend=backend,
    )
    contact = trial2_final.process_task(
        search_anchor="start", trace_log=contact_traces, **common
    )
    separation = trial2_final.process_task(
        search_anchor="end", trace_log=separation_traces, **common
    )

    assert contact == separation
    contact_frames = contact_traces[0]["used_frame_indices"]
    separation_frames = separation_traces[0]["used_frame_indices"]
    assert contact_frames == separation_frames
    assert all(
        right - left == 1
        for left, right in zip(contact_frames, contact_frames[1:])
    )
    assert contact_traces[0]["midpoint_restriction"] is False
    assert backend.calls[0]["task"] == "contact"
    assert backend.calls[1]["task"] == "separation"


def test_closed_loop_visual_feedback_and_pinch_skips_speed(episode):
    video = episode.make_video()
    episode.write_pinch_json()
    backend = RecordingBackend(visual="1")
    traces = []

    final = trial2_final.feedback_contact(
        {},
        video,
        "Grasping the object",
        3,
        episode.total_frames,
        frame_start=12,
        max_feedbacks=1,
        search_anchor="start",
        speed_folder=episode.root,
        pinch_folder=episode.root,
        mode="pinch",
        trial_seed=99,
        trace_log=traces,
        inference_backend=backend,
    )

    assert final == 12
    assert len(backend.calls) == 1
    assert backend.calls[0]["flag"] == "feedback"
    assert traces[0]["stage"].startswith("contact_visual_feedback")
    assert traces[0]["used_frame_indices"] == [12]


def test_paired_seed_has_no_mode_argument():
    seed = trial2_final.derive_seed("000001", "contact", 0)
    assert seed == trial2_final.derive_seed("000001", "contact", 0)
    assert seed != trial2_final.derive_seed("000001", "contact", 1)
    assert seed != trial2_final.derive_seed("000001", "separation", 0)
    with pytest.raises(ValueError):
        trial2_final.validate_sampling_mode("unknown")


def test_speed_mode_does_not_open_pinch(episode):
    video = episode.make_video()
    episode.write_speed_json()
    poison = episode.metrics_dir + f"/{episode.name}_pinch_distance.json"
    with open(poison, "w") as handle:
        handle.write("{not valid json")
    backend = RecordingBackend()
    traces = []
    result = trial2_final.process_task(
        {},
        video,
        "Grasping the object",
        3,
        episode.total_frames,
        "start",
        episode.root,
        pinch_folder=episode.root,
        mode="speed",
        trial_seed=1,
        trace_log=traces,
        inference_backend=backend,
    )
    assert result is not None
    assert traces[0]["signal_reads"] == {"speed": True, "pinch": False}


def test_pinch_mode_does_not_open_speed(episode):
    video = episode.make_video()
    episode.write_pinch_json()
    poison = episode.metrics_dir + f"/{episode.name}_hand_speed.json"
    with open(poison, "w") as handle:
        handle.write("{not valid json")
    backend = RecordingBackend()
    traces = []
    result = trial2_final.process_task(
        {},
        video,
        "Grasping the object",
        3,
        episode.total_frames,
        "start",
        episode.root,
        pinch_folder=episode.root,
        mode="pinch",
        trial_seed=1,
        trace_log=traces,
        inference_backend=backend,
    )
    assert result is not None
    assert traces[0]["signal_reads"] == {"speed": False, "pinch": True}
