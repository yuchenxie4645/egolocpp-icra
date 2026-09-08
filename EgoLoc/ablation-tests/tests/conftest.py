"""Shared fixtures for the Trial-2-final protocol tests.

Everything here runs on CPU only: synthetic MJPG videos, canned signal
JSONs, and a recording stub that replaces Trial-2-final VLM calls so no VLM
server or model is ever contacted.
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.dirname(TESTS_DIR)
EGOLOC_DIR = os.path.dirname(ABLATION_DIR)

for _path in (EGOLOC_DIR, ABLATION_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import trial_2_final as trial2_final  # noqa: E402

# Shared golden fixture: 40 frames; a sparse hand-speed signal where only
# frames 0..9 have non-zero speeds.
DEFAULT_TOTAL_FRAMES = 40
DEFAULT_SPEEDS = {
    0: 5.0,
    1: 0.2,
    2: 3.0,
    3: 0.9,
    4: 0.4,
    5: 7.0,
    6: 2.0,
    7: 1.0,
    8: 0.8,
    9: 6.0,
}
DEFAULT_MINIMA = [10, 25]


class Episode:
    """Builder for one synthetic episode: video plus signal JSONs."""

    def __init__(self, root, name="ep001", total_frames=DEFAULT_TOTAL_FRAMES):
        self.root = str(root)
        self.name = name
        self.total_frames = total_frames
        self.metrics_dir = os.path.join(self.root, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.video_path = None
        self.speed_path = None
        self.pinch_path = None

    def make_video(self, n_frames=None):
        n_frames = self.total_frames if n_frames is None else n_frames
        self.video_path = os.path.join(self.root, f"{self.name}.avi")
        writer = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48)
        )
        assert writer.isOpened(), "cv2.VideoWriter failed to open"
        for i in range(n_frames):
            frame = np.full((48, 64, 3), (i * 6) % 255, dtype=np.uint8)
            frame[:, :, i % 3] = (i * 11) % 255
            writer.write(frame)
        writer.release()
        cap = cv2.VideoCapture(self.video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert count == n_frames, f"video round-trip: {count} != {n_frames} frames"
        return self.video_path

    def write_speed_json(self, speeds=None, corrupt=False):
        self.speed_path = os.path.join(self.metrics_dir, f"{self.name}_hand_speed.json")
        if corrupt:
            with open(self.speed_path, "w") as f:
                f.write("{this is not valid json")
        else:
            speeds = DEFAULT_SPEEDS if speeds is None else speeds
            data = {str(i): 0.0 for i in range(self.total_frames)}
            for frame_idx, speed in speeds.items():
                data[str(frame_idx)] = float(speed)
            with open(self.speed_path, "w") as f:
                json.dump(data, f)
        return self.speed_path

    def write_pinch_json(self, neighborhood=None, minima=None, corrupt=False):
        self.pinch_path = os.path.join(self.metrics_dir, f"{self.name}_pinch_distance.json")
        if corrupt:
            with open(self.pinch_path, "w") as f:
                f.write("{broken json!!")
        else:
            minima = DEFAULT_MINIMA if minima is None else minima
            if neighborhood is None:
                neighborhood = sorted(
                    {
                        n
                        for m in minima
                        for n in (m - 1, m, m + 1)
                        if 0 <= n < self.total_frames
                    }
                )
            data = {
                "video_name": self.name,
                "total_frames": self.total_frames,
                "pinch_joint_pair": {"thumb_tip": 4, "index_tip": 8},
                "candidate_selection": {
                    "strategy": "relative_minimum_pinch_distance_with_neighbors",
                    "contiguous_detection_segments": True,
                    "neighborhood": [-1, 0, 1],
                    "relative_minimum_frames": sorted(minima),
                    "relative_minimum_neighborhood_frames": sorted(neighborhood),
                },
                "frames": {},
            }
            with open(self.pinch_path, "w") as f:
                json.dump(data, f)
        return self.pinch_path


class StubVLM:
    """Recording drop-in replacement for Trial-2-final ``vlm_request``."""

    def __init__(self, content='{"points": [1]} canned analysis'):
        self.content = content
        self.calls = []

    def __call__(self, image_bgr, prompt_message, task, backend="vllm", seed=None, flag=None, trace_out=None):
        self.calls.append({"task": task, "backend": backend, "seed": seed, "flag": flag})
        if trace_out is not None:
            trace_out["raw_response"] = self.content
            trace_out["latency_s"] = 0.0
            trace_out["backend_used"] = backend
            trace_out["seed"] = seed
            trace_out["task"] = task
            trace_out["flag"] = flag
        if flag is None:
            response = trial2_final.extract_json_part(self.content)
            point = -1
            if response is not None:
                points = json.loads(response).get("points", [])
                point = points[0] if points else -1
            return point, self.content
        return self.content


@pytest.fixture
def episode(tmp_path):
    """Fresh synthetic episode inside a tmp dir (video not created yet)."""
    return Episode(tmp_path)


@pytest.fixture
def stub_vlm(monkeypatch):
    """Monkeypatch Trial-2-final ``vlm_request`` with a recording stub."""
    stub = StubVLM()
    monkeypatch.setattr(trial2_final, "vlm_request", stub)
    return stub
