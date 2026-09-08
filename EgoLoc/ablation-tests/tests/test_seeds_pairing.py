"""Deterministic, mode-independent paired seeds.

derive_seed(episode, task, trial) must be a pure function of its tuple —
identical across calls and processes, never dependent on the sampling mode —
and the seed must reach the VLM request (observable via the trace).
"""

import hashlib
import inspect

import trial3


def test_derive_seed_is_deterministic():
    first = trial3.derive_seed("ep042", "contact", 0)
    second = trial3.derive_seed("ep042", "contact", 0)

    assert first == second
    assert isinstance(first, int)
    assert 0 <= first < 2 ** 32


def test_derive_seed_varies_with_tuple_components():
    base = trial3.derive_seed("ep042", "contact", 0)

    assert base != trial3.derive_seed("ep043", "contact", 0)
    assert base != trial3.derive_seed("ep042", "separation", 0)
    assert base != trial3.derive_seed("ep042", "contact", 1)


def test_derive_seed_uses_a_process_stable_hash():
    # Pin the documented formula (SHA-256 of "episode|task|trial"): the
    # builtin hash() is process-randomized and would break seed pairing
    # across runs, so the stable-hash property is asserted explicitly.
    expected = int(hashlib.sha256(b"ep042|contact|0").hexdigest()[:8], 16)

    assert trial3.derive_seed("ep042", "contact", 0) == expected


def test_derive_seed_signature_takes_no_mode():
    params = list(inspect.signature(trial3.derive_seed).parameters)

    assert params == ["episode", "task", "trial"]


def test_vlm_request_receives_trial_seed(episode, stub_vlm):
    episode.write_speed_json()
    episode.write_pinch_json()
    episode.make_video()

    seed = trial3.derive_seed("ep001", "contact", 0)
    trace = {}
    selected = trial3.process_task(
        {}, episode.video_path, "Grasping the object", 3,
        episode.total_frames, "start", episode.root,
        trace_out=trace, trial_seed=seed,
    )

    assert stub_vlm.calls, "stub VLM was not called"
    assert stub_vlm.calls[0]["seed"] == seed
    assert trace["seed"] == seed
    assert selected == trace["used_frame_indices"][0]
