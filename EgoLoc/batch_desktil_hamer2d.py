#!/usr/bin/env python3
"""
Batch runner that drives the pinch-augmented 2D EgoLoc pipeline
(`hamer_pinch_2d.py`) over a directory of videos.

Per video it:
  1. Runs HaMeR 2D hand detection + ViTPose to compute hand speed.
  2. Runs ViTPose 21-keypoint hand inference to compute pinch distance.
  3. Calls `convert_video` (with `pinch_folder` set) so the LLM candidate
     pool merges speed-based and relative-minimum pinch candidates.
  4. Saves the contact/separation frames and appends a row to the CSV.

Outputs (relative to `--output_root`, default `/home/data_labeling/data/DeskTIL`):
  egoloc_hamer2d_results.csv
  egoloc_hamer2d_frames/video<idx>_contact.png
  egoloc_hamer2d_frames/video<idx>_separation.png
  per_video/video<idx>/metrics/<video>_hand_speed.json
  per_video/video<idx>/metrics/<video>_pinch_distance.json
  per_video/video<idx>/plots/<video>_hand_speed_curve.png
  per_video/video<idx>/plots/<video>_pinch_distance_curve.png
  per_video/video<idx>/localization_grids/trial_<trial>/<video>_<stage>_<anchor>_selected_f<frame>.png
  per_video/video<idx>/debug/pinch_distance_detected_frames/<video>/<video>_pinch_detected_f<idx:04d>.png
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trial3 import (
    build_detector,
    build_vitpose,
    convert_video,
    extract_2d_speed_and_visualize,
    extract_pinch_distance_and_visualize,
    setup_hamer_cache,
    visualize_frame,
)
from script.config import Config


def video_index(path):
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse video index from {path.name}")
    return int(match.group(1))


def iter_videos(video_dir, start, end, limit):
    videos = [p for p in video_dir.glob("*.mp4") if start <= video_index(p) <= end]
    videos = sorted(videos, key=video_index)
    return videos[:limit] if limit else videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="/home/data_labeling/data/DeskTIL/videos")
    parser.add_argument("--output_root", default="/home/data_labeling/data/DeskTIL",
                        help="Root folder for CSV, frames dir, and per-video outputs.")
    parser.add_argument("--csv_name", default="egoloc_hamer2d_results.csv")
    parser.add_argument("--frames_dir_name", default="egoloc_hamer2d_frames")
    parser.add_argument("--per_video_dir_name", default="per_video")
    parser.add_argument("--credentials", default="/home/EgoLoc/auth.env")
    parser.add_argument("--start", type=int, default=51)
    parser.add_argument("--end", type=int, default=180)
    parser.add_argument("--grid_size", type=int, default=3)
    parser.add_argument("--max_feedbacks", type=int, default=1)
    parser.add_argument("--repeat_times", type=int, default=3)
    parser.add_argument("--action", default="Grasping the object")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    setup_hamer_cache()

    creds = Config.load_credentials(args.credentials)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_detector()
    vitpose = build_vitpose(device)

    output_root = Path(args.output_root)
    csv_path = output_root / args.csv_name
    frames_dir = output_root / args.frames_dir_name
    per_video_root = output_root / args.per_video_dir_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    per_video_root.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(Path(args.video_dir), args.start, args.end, args.limit)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video Index", "Total frames", "Contact", "Separation"])

        for video in videos:
            idx = video_index(video)
            video_name = video.stem
            print(f"\n=== video{idx} ===")

            out_dir = per_video_root / f"video{idx}"
            debug_dir = out_dir / "debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)

            extract_2d_speed_and_visualize(str(video), str(out_dir), detector, vitpose)
            pinch_json_path, pinch_vis_path = extract_pinch_distance_and_visualize(
                str(video), str(out_dir), detector, vitpose, str(debug_dir)
            )

            contact, separation = convert_video(
                str(video), args.action, creds, args.grid_size,
                str(out_dir), args.max_feedbacks,
                repeat_times=args.repeat_times, pinch_folder=str(out_dir),
            )

            writer.writerow([idx, int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)), contact, separation])
            f.flush()
            visualize_frame(str(video), contact, str(frames_dir / f"video{idx}_contact.png"), label="Contact")
            visualize_frame(str(video), separation, str(frames_dir / f"video{idx}_separation.png"), label="Separation")
            print(f"saved row to {csv_path}")
            print(f"contact={contact}, separation={separation}")


if __name__ == "__main__":
    main()
