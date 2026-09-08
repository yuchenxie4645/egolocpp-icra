"""Pinch-mode fallback: the signal-neutral pool is chronological.

Pinch mode never opens the speed JSON, so its fallback pool is the plain
ascending frame range. Every grid slot that is not a pinch candidate must
come from that chronological fill — never from a speed-ranked ordering.
"""

import trial3


def test_pinch_mode_pool_is_chronological(episode):
    episode.write_speed_json()
    episode.write_pinch_json(minima=[20])

    result = trial3.build_mode_candidates(
        "pinch", episode.speed_path, episode.pinch_path,
        total_frames=episode.total_frames,
    )

    assert result["candidates"] == [19, 20, 21]
    assert result["total_pool"] == list(range(40))


def test_pinch_grid_fills_chronologically_around_candidates(episode):
    episode.write_pinch_json(minima=[20])

    result = trial3.build_mode_candidates(
        "pinch", episode.speed_path, episode.pinch_path,
        total_frames=episode.total_frames,
    )
    grid = trial3.build_direct_candidate_grid(
        result["candidates"], result["total_pool"], 3, episode.total_frames, "start"
    )

    # Chronological transcription: 19 anchors the grid in the start half,
    # then slots fill from ascending frames 0..7 (frames 20/21 are out of the
    # anchor region on the first pass and the evidence pass was non-empty, so
    # no relaxation fires).
    assert grid == [0, 1, 2, 3, 4, 5, 6, 7, 19]

    non_candidates = [f for f in grid if f not in result["candidates"]]
    assert non_candidates == [0, 1, 2, 3, 4, 5, 6, 7]
    assert all(0 <= f < episode.total_frames for f in non_candidates)


def test_decoy_speed_ranked_pool_produces_a_different_grid():
    # A speed-ranked pool (descending "speed" order) would place late frames
    # first; the chronological policy must not reproduce it.
    decoy_pool = list(range(39, -1, -1))
    grid = trial3.build_direct_candidate_grid(
        [19, 20, 21], decoy_pool, 3, 40, "start"
    )

    assert grid == [11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert grid != [0, 1, 2, 3, 4, 5, 6, 7, 19]


def test_process_task_pinch_fallback_uses_chronological_pool(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json(minima=[20])
    episode.make_video()

    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        pinch_folder=episode.root, trace_out=trace, mode="pinch",
    )

    assert trace["total_pool"] == list(range(episode.total_frames))
    assert trace["used_frame_indices"] == [0, 1, 2, 3, 4, 5, 6, 7, 19]
    assert selected == trace["used_frame_indices"][0]
