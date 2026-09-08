"""Grid builder budget/order/anchor contract.

build_direct_candidate_grid must produce exactly grid_size**2 sorted
chronological frames whenever the signal is rich enough, must respect the
start/end anchor on the first pass (with the documented relaxation only when
the first pass would come up empty), and must exclude frames at or below
min_index.
"""

import trial3


def test_grid_returns_nine_sorted_chronological_frames():
    grid = trial3.build_direct_candidate_grid(
        [9, 10, 11, 24, 25, 26], list(range(40)), 3, 40, "start"
    )

    assert len(grid) == 9
    assert grid == sorted(grid)
    assert len(set(grid)) == 9
    assert grid == [0, 1, 2, 3, 4, 5, 9, 10, 11]


def test_anchor_start_respected_when_pool_is_rich():
    grid = trial3.build_direct_candidate_grid(
        [9, 10, 11, 24, 25, 26], list(range(40)), 3, 40, "start"
    )

    assert all(frame < 20 for frame in grid)


def test_anchor_end_respected_when_pool_is_rich():
    grid = trial3.build_direct_candidate_grid(
        [9, 10, 11, 24, 25, 26], list(range(40)), 3, 40, "end"
    )

    assert len(grid) == 9
    assert grid == sorted(grid)
    assert all(frame >= 20 for frame in grid)
    assert grid == [20, 21, 22, 23, 24, 25, 26, 27, 28]


def test_anchor_relaxed_only_when_first_pass_empty():
    # All evidence sits in the second half while anchoring to "start": the
    # first evidence pass would come up empty, so the documented relaxation
    # kicks in and the grid is still filled to nine frames.
    grid = trial3.build_direct_candidate_grid(
        [30, 31, 32], list(range(40)), 3, 40, "start"
    )

    assert len(grid) == 9
    assert grid == sorted(grid)
    assert grid == [0, 1, 2, 3, 4, 5, 30, 31, 32]


def test_min_index_excludes_frames_at_or_below():
    grid = trial3.build_direct_candidate_grid(
        [9, 10, 11, 24, 25, 26], list(range(40)), 3, 40, "start", min_index=15
    )

    assert len(grid) == 9
    assert all(frame > 15 for frame in grid)
    assert grid == [16, 17, 18, 19, 20, 21, 24, 25, 26]


def test_exactly_nine_slots_with_duplicate_candidates():
    grid = trial3.build_direct_candidate_grid(
        [1, 1, 4, 8, 8, 3, 3, 3], list(range(40)), 3, 40, "start"
    )

    assert len(grid) == 9
    assert grid == sorted(grid)
    assert len(set(grid)) == 9


def test_invalid_list_frames_are_excluded():
    grid = trial3.build_direct_candidate_grid(
        [1, 4, 8, 3], [1, 4, 8, 3, 7, 6, 2, 0, 9, 5], 3, 40, "start", invalid_list=[7]
    )

    assert len(grid) == 9
    assert 7 not in grid
    assert grid == [0, 1, 2, 3, 4, 5, 6, 8, 9]


def test_short_video_cannot_fill_nine_slots():
    grid = trial3.build_direct_candidate_grid(
        [], [], 3, 4, "start"
    )

    assert grid == [0, 1, 2, 3]
