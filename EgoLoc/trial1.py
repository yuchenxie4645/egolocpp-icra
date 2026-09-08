import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import dotenv
import base64
import math
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = os.path.dirname(os.path.abspath(__file__))
HAMER_ROOT = os.path.join(ROOT, "hamer")
VITPOSE_ROOT = os.path.join(HAMER_ROOT, "third-party", "ViTPose")
HAMER_CACHE = os.path.join(ROOT, "hamer_cache")

for path_entry in [ROOT, HAMER_ROOT, VITPOSE_ROOT]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)


def copy_cache_tree(src, dst):
    if not os.path.exists(src):
        return

    import shutil
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        return

    for current_root, _, files in os.walk(src):
        rel = os.path.relpath(current_root, src)
        target_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target_root, exist_ok=True)
        for name in files:
            target = os.path.join(target_root, name)
            if not os.path.exists(target):
                shutil.copy2(os.path.join(current_root, name), target)


def setup_hamer_cache():
    os.makedirs(HAMER_CACHE, exist_ok=True)
    copy_cache_tree(os.path.join(HAMER_ROOT, "_DATA"), HAMER_CACHE)
    copy_cache_tree(os.path.join(HAMER_ROOT, "models"), os.path.join(HAMER_CACHE, "models"))
    copy_cache_tree("/home/hamer/models", os.path.join(HAMER_CACHE, "models"))
    os.environ.setdefault("TORCH_HOME", os.path.join(HAMER_CACHE, "torch"))
    os.environ.setdefault("FVCORE_CACHE", os.path.join(HAMER_CACHE, "fvcore_cache"))
    print(f"Using HaMeR cache: {HAMER_CACHE}")


def build_detector():
    import hamer
    from detectron2.config import LazyConfig
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    cfg_path = os.path.join(os.path.dirname(hamer.__file__), "configs", "cascade_mask_rcnn_vitdet_h_75ep.py")
    cfg = LazyConfig.load(cfg_path)

    local_ckpts = [
        os.path.join(HAMER_CACHE, "models", "model_final_f05665.pkl"),
        os.path.join(HAMER_CACHE, "model_final_f05665.pkl"),
        "/home/hamer/models/model_final_f05665.pkl",
        os.path.join(HAMER_ROOT, "models", "model_final_f05665.pkl"),
        os.path.join(HAMER_ROOT, "_DATA", "model_final_f05665.pkl"),
    ]
    for ckpt in local_ckpts:
        if os.path.exists(ckpt):
            cfg.train.init_checkpoint = ckpt
            print(f"Using Detectron2 checkpoint: {ckpt}")
            break
    else:
        raise FileNotFoundError("Missing model_final_f05665.pkl for HaMeR body detection")

    for predictor in cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = 0.25

    return DefaultPredictor_Lazy(cfg)


def build_vitpose(device):
    import vitpose_model as vpm

    vpm.ROOT_DIR = HAMER_ROOT
    vpm.VIT_DIR = VITPOSE_ROOT
    cfg_rel = os.path.join(
        "configs", "wholebody", "2d_kpt_sview_rgb_img", "topdown_heatmap",
        "coco-wholebody", "ViTPose_huge_wholebody_256x192.py",
    )
    cached_ckpt = os.path.join(HAMER_CACHE, "vitpose_ckpts", "vitpose+_huge", "wholebody.pth")
    original_ckpt = os.path.join(HAMER_ROOT, "_DATA", "vitpose_ckpts", "vitpose+_huge", "wholebody.pth")
    if os.path.exists(cached_ckpt):
        vitpose_ckpt = cached_ckpt
    elif os.path.exists(original_ckpt):
        vitpose_ckpt = original_ckpt
    else:
        raise FileNotFoundError("Missing ViTPose checkpoint wholebody.pth for HaMeR 2D detection")

    print(f"Using ViTPose checkpoint: {vitpose_ckpt}")
    for item in vpm.ViTPoseModel.MODEL_DICT.values():
        item["config"] = os.path.join(VITPOSE_ROOT, cfg_rel)
        item["model"] = vitpose_ckpt

    return vpm.ViTPoseModel(device)


def hand_center_2d(frame_bgr, detector, vitpose):
    det_out = detector(frame_bgr)
    inst = det_out["instances"]
    valid = (inst.pred_classes == 0) & (inst.scores > 0.5)
    boxes = inst.pred_boxes.tensor[valid].cpu().numpy()
    scores = inst.scores[valid].cpu().numpy()

    if len(boxes) == 0:
        return None

    det_results = [np.concatenate([boxes, scores[:, None]], axis=1)]
    poses = vitpose.predict_pose(frame_bgr[:, :, ::-1], det_results)

    candidates = []
    for pose in poses:
        keypoints = pose["keypoints"]
        for hand_kpts in [keypoints[-42:-21], keypoints[-21:]]:
            good = hand_kpts[:, 2] > 0.5
            if good.sum() <= 3:
                continue
            xy = hand_kpts[good, :2]
            score = float(hand_kpts[good, 2].mean())
            center = (float(xy[:, 0].mean()), float(xy[:, 1].mean()))
            candidates.append((score, center))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]

def extract_2d_speed_and_visualize(video_path, output_dir, detector, vitpose):
    """
    Track the hand in every frame with HaMeR's detector and ViTPose hand keypoints,
    compute its 2D speed, and save results.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    metrics_dir = os.path.join(output_dir, "metrics")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_bounding_box_centers = {}

    for idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        center = hand_center_2d(frame, detector, vitpose)
        if center is not None:
            all_bounding_box_centers[idx] = center

        if (idx + 1) % 50 == 0 or idx + 1 == total_frames:
            print(f"{video_name}: processed {idx + 1}/{total_frames}")

    cap.release()
    speed_dict = {}
    prev_center = None
    for idx in range(total_frames):
        center = all_bounding_box_centers.get(idx, None)
        if center is None:
            speed = 0.0
        else:
            if prev_center is None:
                speed = 0.0
            else:
                speed = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
            prev_center = center
        speed_dict[idx] = speed
    speed_json_path = os.path.join(metrics_dir, f"{video_name}_hand_speed.json")
    with open(speed_json_path, 'w') as f:
        json.dump(speed_dict, f, indent=2)
    plt.figure(figsize=(12, 4))
    plt.plot(list(speed_dict.keys()), list(speed_dict.values()), label='2D Hand Speed')
    plt.xlabel('Frame Index')
    plt.ylabel('Speed')
    plt.title('2D Hand Speed Curve')
    plt.legend()
    plt.tight_layout()
    speed_vis_path = os.path.join(plots_dir, f"{video_name}_hand_speed_curve.png")
    plt.savefig(speed_vis_path)
    plt.close()
    return speed_json_path, speed_vis_path, all_bounding_box_centers

def _detect_hands_for_pinch(frame_bgr, detector, vitpose):
    """
    Run Detectron2 + ViTPose on a single frame and return 21-point hand
    keypoints plus right/left flags for every detected hand.

    Returns:
        Tuple[list[np.ndarray], np.ndarray]: (hand_keypoints, is_right), where
        each hand_keypoints entry is a (21, 3) array in full-frame pixels with
        confidence in the third column. Empty arrays if no hands are detected.
    """
    det_out = detector(frame_bgr)
    inst = det_out["instances"]
    valid = (inst.pred_classes == 0) & (inst.scores > 0.5)
    boxes = inst.pred_boxes.tensor[valid].cpu().numpy()
    scores = inst.scores[valid].cpu().numpy()

    if len(boxes) == 0:
        return [], np.zeros((0,), dtype=np.int32)

    det_results = [np.concatenate([boxes, scores[:, None]], axis=1)]
    poses = vitpose.predict_pose(frame_bgr[:, :, ::-1], det_results)

    hand_keypoints = []
    is_right = []
    for pose in poses:
        keypoints = pose["keypoints"]
        for hand_kpts, side_flag in [(keypoints[-42:-21], 0), (keypoints[-21:], 1)]:
            good = hand_kpts[:, 2] > 0.5
            if good.sum() <= 3:
                continue
            hand_keypoints.append(hand_kpts.astype(np.float32))
            is_right.append(side_flag)

    if not hand_keypoints:
        return [], np.zeros((0,), dtype=np.int32)
    return hand_keypoints, np.array(is_right, dtype=np.int32)


def _draw_pinch_debug(frame_bgr, kpts_2d_full, pinch_dist):
    """
    Annotate a frame with the 21 hand keypoints and the pinch line.

    All 21 keypoints are drawn green; points 4 (thumb tip) and 8 (index tip)
    are drawn black and connected by a black line labeled with the pixel
    distance at the midpoint.

    Args:
        frame_bgr (np.ndarray): BGR frame to annotate (will be copied).
        kpts_2d_full (np.ndarray): Shape (21, 2) full-frame keypoints.
        pinch_dist (float): Pixel distance between points 4 and 8.

    Returns:
        np.ndarray: Annotated BGR frame.
    """
    img = frame_bgr.copy()
    for idx, (x, y) in enumerate(kpts_2d_full):
        x_i, y_i = int(round(float(x))), int(round(float(y)))
        if idx in (4, 8):
            cv2.circle(img, (x_i, y_i), 6, (0, 0, 0), -1)
        else:
            cv2.circle(img, (x_i, y_i), 4, (0, 255, 0), -1)

    p4 = kpts_2d_full[4]
    p8 = kpts_2d_full[8]
    cv2.line(
        img,
        (int(round(float(p4[0]))), int(round(float(p4[1])))),
        (int(round(float(p8[0]))), int(round(float(p8[1])))),
        (0, 0, 0),
        2,
    )
    mid = ((p4[0] + p8[0]) / 2.0, (p4[1] + p8[1]) / 2.0)
    cv2.putText(
        img,
        f"{pinch_dist:.1f}px",
        (int(round(float(mid[0]))), int(round(float(mid[1])))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    return img


def extract_pinch_distance_and_visualize(video_path, output_dir, detector, vitpose, debug_dir, percentile=15):
    """
    Run the HaMeR 2D detector + ViTPose per frame, compute the pinch distance
    between thumb tip (joint 4) and index tip (joint 8), and persist the
    per-frame distance plus a curve plot. Pinch keyframes are the detected
    frames below the requested distance percentile. Annotated debug PNGs are
    written only for frames where a hand is detected; frames without detections
    are still represented in the JSON.

    This uses ViTPose's 21 hand keypoints and does not load the HaMeR MANO
    model or require MANO mesh assets.

    Args:
        video_path (str): Input video.
        output_dir (str): Per-video folder where metrics and plots are saved.
        detector: Detectron2 predictor (from *build_detector*).
        vitpose: ViTPose model (from *build_vitpose*).
        debug_dir (str): Folder under which pinch debug PNGs
            annotated PNGs are written. Created if missing.
        percentile (int, optional): Distance percentile used for pinch
            keyframe selection. Default 15.

    Returns:
        Tuple[str, str]: (pinch_json_path, pinch_vis_path).
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    metrics_dir = os.path.join(output_dir, "metrics")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    debug_video_dir = os.path.join(debug_dir, "pinch_distance_detected_frames", video_name)
    os.makedirs(debug_video_dir, exist_ok=True)

    pinch_frames = {}

    for idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            pinch_frames[str(idx)] = {
                "detected": False,
                "detection_status": "frame_read_failed",
                "pinch_distance_px": None,
                "is_bottom_percentile": False,
            }
            continue

        hand_keypoints, is_right = _detect_hands_for_pinch(frame, detector, vitpose)
        if len(hand_keypoints) == 0:
            pinch_frames[str(idx)] = {
                "detected": False,
                "detection_status": "no_hand_detected",
                "pinch_distance_px": None,
                "is_bottom_percentile": False,
            }
            continue

        per_hand_distances = []
        per_hand_keypoints = []
        for kpts in hand_keypoints:
            d = float(np.linalg.norm(kpts[4, :2] - kpts[8, :2]))
            per_hand_distances.append(d)
            per_hand_keypoints.append(kpts[:, :2])

        if not per_hand_distances:
            pinch_frames[str(idx)] = {
                "detected": False,
                "detection_status": "no_valid_hand_keypoints",
                "pinch_distance_px": None,
                "is_bottom_percentile": False,
            }
            continue

        best = int(np.argmin(per_hand_distances))
        pinch_distance = float(per_hand_distances[best])
        pinch_frames[str(idx)] = {
            "detected": True,
            "detection_status": "detected",
            "pinch_distance_px": pinch_distance,
            "is_bottom_percentile": False,
            "detected_hand_count": int(len(hand_keypoints)),
            "selected_hand_index": best,
        }
        annotated = _draw_pinch_debug(frame, per_hand_keypoints[best], per_hand_distances[best])
        cv2.imwrite(os.path.join(debug_video_dir, f"{video_name}_pinch_detected_f{idx:04d}.png"), annotated)

        if (idx + 1) % 50 == 0 or idx + 1 == total_frames:
            print(f"{video_name}: pinch processed {idx + 1}/{total_frames}")

    cap.release()

    detected_items = [
        (int(frame_idx), record["pinch_distance_px"])
        for frame_idx, record in pinch_frames.items()
        if record["detected"] and record["pinch_distance_px"] is not None
    ]
    for record in pinch_frames.values():
        record.setdefault("is_bottom_percentile", False)

    if detected_items:
        distances = np.array([distance for _, distance in detected_items], dtype=float)
        threshold = float(np.percentile(distances, percentile))
        bottom_percentile_frames = sorted(
            frame_idx for frame_idx, distance in detected_items if float(distance) < threshold
        )
    else:
        threshold = None
        bottom_percentile_frames = []

    for frame_idx in bottom_percentile_frames:
        pinch_frames[str(frame_idx)]["is_bottom_percentile"] = True

    pinch_json = {
        "video_name": video_name,
        "total_frames": total_frames,
        "pinch_joint_pair": {
            "thumb_tip": 4,
            "index_tip": 8,
        },
        "candidate_selection": {
            "strategy": "bottom_percentile_pinch_distance",
            "percentile": percentile,
            "threshold_px": threshold,
            "ignored_statuses": [
                "frame_read_failed",
                "no_hand_detected",
                "no_valid_hand_keypoints",
            ],
            "bottom_percentile_frames": bottom_percentile_frames,
        },
        "frames": pinch_frames,
    }

    pinch_json_path = os.path.join(metrics_dir, f"{video_name}_pinch_distance.json")
    with open(pinch_json_path, 'w') as f:
        json.dump(pinch_json, f, indent=2)

    if detected_items:
        items = sorted(detected_items)
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        plt.figure(figsize=(12, 4))
        plt.plot(xs, ys, label='Pinch distance (thumb tip - index tip)')
        plt.axhline(
            threshold,
            color='r',
            linestyle='--',
            label=f'{percentile}th percentile = {threshold:.1f}px',
        )
        bottom_y = [
            pinch_frames[str(frame_idx)]["pinch_distance_px"]
            for frame_idx in bottom_percentile_frames
        ]
        if bottom_percentile_frames:
            plt.scatter(
                bottom_percentile_frames, bottom_y,
                color='g', s=24, zorder=3,
                label='bottom-percentile pinch keyframes',
            )
        plt.xlabel('Frame Index')
        plt.ylabel('Pinch distance (px)')
        plt.title('Pinch Distance Curve')
        plt.legend()
        plt.tight_layout()
    else:
        plt.figure(figsize=(12, 4))
        plt.title('Pinch Distance Curve (no detections)')
        plt.tight_layout()
    pinch_vis_path = os.path.join(plots_dir, f"{video_name}_pinch_distance_curve.png")
    plt.savefig(pinch_vis_path)
    plt.close()

    return pinch_json_path, pinch_vis_path

def visualize_frame(video_path, frame_idx, output_path, label=None):
    """
    Save one frame from a video, with an optional text label.

    Args:
        video_path (str): Video file.
        frame_idx (int): Index of the frame to save.
        output_path (str): Where to write the image file.
        label (str, optional): If given, draw this text on the frame.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Frame {frame_idx} not found for visualization.")
        return
    if label is not None:
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
    cv2.imwrite(output_path, frame)
    cap.release()

def feedback_contact(credentials, video_path, action, grid_size, total_frames, frame_start, max_feedbacks, search_anchor, speed_folder, pinch_folder=None, trial_idx=None):
    """
    Improves the first-contact frame by asking the vision-language model again.

    The function keeps querying until either:
        • The model agrees the frame is contact, or
        • The maximum number of feedback loops is reached, or
        • No valid frame is returned.

    Args:
        credentials (dict): API keys for GPT-4V.
        video_path (str): Input video.
        action (str): Action description, e.g. "Grasping the object".
        grid_size (int): Grid size used to build query images.
        total_frames (int): Total frame count of the video.
        frame_start (int): Initial contact guess.
        max_feedbacks (int): How many times to re-ask.
        search_anchor (str): "start" anchor for contact.
        speed_folder (str): Folder holding metrics/*_hand_speed.json files.
        pinch_folder (str | None): Folder holding metrics/*_pinch_distance.json files. When
            provided, pinch keyframes are merged with the speed keyframes in
            the candidate pool.

    Returns:
        int | None: Final contact frame index or None if not found.
    """
    feedback_count = 0
    frame_contact = frame_start
    while feedback_count < max_feedbacks:
        corrected_frame = determine_by_state(
            credentials, video_path, action, grid_size, total_frames,
            frame_contact, search_anchor, speed_folder, pinch_folder=pinch_folder,
            trial_idx=trial_idx, stage=f"contact_visual_feedback_{feedback_count:02d}",
        )
        if corrected_frame is None:
            return None
        if corrected_frame == frame_contact:
            corrected_speed_frame = determine_by_speed(
                credentials, video_path, action, grid_size, total_frames,
                frame_contact, search_anchor, speed_folder, pinch_folder=pinch_folder,
                trial_idx=trial_idx, stage=f"contact_speed_feedback_{feedback_count:02d}",
            )
            if corrected_speed_frame is None:
                return None
            if corrected_speed_frame == frame_contact:
                return frame_contact
            else:
                frame_contact = corrected_speed_frame
        else:
            frame_contact = corrected_frame
        feedback_count += 1
    return frame_contact

def feedback_separation(credentials, video_path, action, grid_size, total_frames, frame_end, max_feedbacks, search_anchor, speed_folder, pinch_folder=None, trial_idx=None):
    """
    Improve the separation (end) frame using the same feedback logic as
    *feedback_contact* but looking for the end of the action.

    Args:
        credentials (dict): API keys for GPT-4V.
        video_path (str): Input video.
        action (str): Action description.
        grid_size (int): Grid size for grid image.
        total_frames (int): Frame count.
        frame_end (int): Initial separation guess.
        max_feedbacks (int): How many feedback loops.
        search_anchor (str): "end" anchor for separation.
        speed_folder (str): Folder with speed JSONs.
        pinch_folder (str | None): Folder with pinch JSONs. Forwarded to
            *process_task* so pinch keyframes can be merged into the
            candidate pool.

    Returns:
        int | None: Final separation frame index or None if not found.
    """
    feedback_count = 0
    frame_separate = frame_end
    while feedback_count < max_feedbacks:
        corrected_frame = determine_by_state(
            credentials, video_path, action, grid_size, total_frames,
            frame_separate, search_anchor, speed_folder, pinch_folder=pinch_folder,
            trial_idx=trial_idx, stage=f"separation_visual_feedback_{feedback_count:02d}",
        )
        if corrected_frame is None:
            return None
        if corrected_frame == frame_separate:
            corrected_speed_frame = determine_by_speed(
                credentials, video_path, action, grid_size, total_frames,
                frame_separate, search_anchor, speed_folder, pinch_folder=pinch_folder,
                trial_idx=trial_idx, stage=f"separation_speed_feedback_{feedback_count:02d}",
            )
            if corrected_speed_frame is None:
                return None
            if corrected_speed_frame == frame_separate:
                return frame_separate
            else:
                frame_separate = corrected_speed_frame
        else:
            frame_separate = corrected_frame
        feedback_count += 1
    return frame_separate

def convert_video(video_path, action, credentials, grid_size, speed_folder, max_feedbacks, repeat_times=5, pinch_folder=None):
    """
    Run several trials to localize contact and separation frames, then average.

    For *repeat_times* iterations the function:
        1. Calls *process_task* to guess the start and end of the action.
        2. Refines those guesses with *feedback_contact* / *feedback_separation*.
        3. Stores the results.

    After all trials it removes *None* or *-1* results and returns the mean
    (rounded) of what is left; if nothing is left the value -1 is returned.

    Args:
        video_path (str): Path to the input video.
        action (str): Action description shown to the model.
        credentials (dict): API keys for GPT-4V.
        grid_size (int): Size of the frame grid fed to the model.
        speed_folder (str): Folder containing metrics/*_hand_speed.json files.
        max_feedbacks (int): Max fk loops per trial.
        repeat_times (int, optional): How many trials to run. Default is 5.
        pinch_folder (str | None): Folder containing metrics/*_pinch_distance.json files. When
            provided, pinch keyframes are merged with the speed keyframes in
            the candidate pool.

    Returns:
        Tuple[int, int]: (contact_frame, separation_frame). Each is a frame
        index or -1 if the event could not be determined.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    contact_list = []
    separate_list = []
    for trial_idx in range(repeat_times):

        print("----------> trial_idx: ", trial_idx)

        frame_start = process_task(
            credentials, video_path, action, grid_size, total_frames,
            'start', speed_folder, pinch_folder=pinch_folder,
            trial_idx=trial_idx, stage="contact_initial",
        )
        frame_end = process_task(
            credentials, video_path, action, grid_size, total_frames,
            'end', speed_folder, pinch_folder=pinch_folder,
            trial_idx=trial_idx, stage="separation_initial",
        )
        frame_contact = feedback_contact(
            credentials, video_path, action, grid_size, total_frames,
            frame_start, max_feedbacks, 'start', speed_folder,
            pinch_folder=pinch_folder, trial_idx=trial_idx,
        )
        frame_separate = feedback_separation(
            credentials, video_path, action, grid_size, total_frames,
            frame_end, max_feedbacks, 'end', speed_folder,
            pinch_folder=pinch_folder, trial_idx=trial_idx,
        )
        contact_list.append(frame_contact)
        separate_list.append(frame_separate)
    
    contact_list = [x for x in contact_list if x is not None and x != -1]
    separate_list = [x for x in separate_list if x is not None and x != -1]
    if len(contact_list) == 0:
        final_contact = -1
    else:
        final_contact = int(round(np.mean(contact_list)))
    if len(separate_list) == 0:
        final_separate = -1
    else:
        final_separate = int(round(np.mean(separate_list)))
    return final_contact, final_separate

def select_top_n_frames_from_json(json_path, n, frame_index=None, flag=None, receive_flag=None):
    """
    Select the *n* frames with the lowest hand speed from a JSON file.

    The JSON maps frame index → speed.  Depending on *flag* and
    *frame_index*, some frames are ignored so that feedback focuses on new
    candidates:

        • flag == "feedback": keep frames *after* frame_index.
        • flag == "speed":   keep frames with speed *below* frame_index
          (here frame_index represents a speed threshold).
        • otherwise:         exclude *frame_index* itself.

    If *receive_flag* is "right" the function also returns an *invalid_list*
    containing the discarded frames.

    Args:
        json_path (str): Path to the hand-speed JSON file.
        n (int): Number of frames to return.
        frame_index (int, optional): Reference index or threshold.
        flag (str, optional): Behaviour switch ("feedback" or "speed").
        receive_flag (str, optional): If set, also return invalid frames.

    Returns:
        list[int] | Tuple[list[int], list[int]]: Top-n frame indices, with an
        optional list of invalid indices first when *receive_flag* is given.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    items = list(data.items())
    if frame_index is None:
        valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed)]
    else:
        if flag == "feedback":
            valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) > frame_index]
            invalid_list = [int(index) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) <= frame_index]
        elif flag == "speed":
            valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed) and speed < frame_index]
            invalid_list = [int(index) for index, speed in items if speed != 0.0 and not math.isnan(speed) and speed >= frame_index]
        else:
            valid_frames = [(int(index), speed) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) != frame_index]
            invalid_list = [int(index) for index, speed in items if speed != 0.0 and not math.isnan(speed) and int(index) == frame_index]
    sorted_frames = sorted(valid_frames, key=lambda x: x[1])
    top_n_frames = [frame[0] for frame in sorted_frames[:n]]
    if receive_flag is None:
        return top_n_frames
    else:
        return invalid_list, top_n_frames

def _detected_pinch_items_from_json(data):
    """Return ``[(frame_idx, pinch_distance_px), ...]`` from new or legacy JSON."""
    if "frames" in data:
        items = []
        for frame_idx, record in data["frames"].items():
            if not isinstance(record, dict):
                continue
            distance = record.get("pinch_distance_px")
            if record.get("detected") and distance is not None:
                items.append((int(frame_idx), float(distance)))
        return sorted(items)

    return sorted((int(k), float(v)) for k, v in data.items() if v is not None)


def select_pinch_keyframes(pinch_json_path, percentile=15):
    """
    Select pinch keyframes from the bottom percentile of detected pinch sizes.

    Frames without a detected hand are ignored for the percentile calculation.
    Structured JSON and legacy ``{frame: distance}`` JSON are both supported.

    Args:
        pinch_json_path (str): Path to the pinch-distance JSON.
        percentile (int, optional): Percentile cutoff. Default 15.

    Returns:
        list[int]: Sorted frame indices whose pinch distance is below the
        percentile threshold. Empty list if the JSON is missing, empty, or has
        no detections.
    """
    if not os.path.exists(pinch_json_path):
        return []
    with open(pinch_json_path, 'r') as f:
        data = json.load(f)
    if not data:
        return []
    candidate_selection = data.get("candidate_selection") if isinstance(data, dict) else None
    if isinstance(candidate_selection, dict) and candidate_selection.get("strategy") == "bottom_percentile_pinch_distance":
        frames = candidate_selection.get("bottom_percentile_frames", [])
        return sorted(int(frame_idx) for frame_idx in frames)

    detected_items = _detected_pinch_items_from_json(data)
    if not detected_items:
        return []

    distances = np.array([distance for _, distance in detected_items], dtype=float)
    threshold = float(np.percentile(distances, percentile))
    return sorted(frame_idx for frame_idx, distance in detected_items if float(distance) < threshold)

def _merge_speed_and_pinch(speed_indices, pinch_indices, n=None):
    """
    Union of speed-based and pinch-based keyframes, deduped.

    Order is preserved (speed indices first, then any pinch indices not already
    present). No weighting is applied — duplicates are simply removed.

    Args:
        speed_indices (list[int]): Keyframes derived from the speed signal.
        pinch_indices (list[int]): Keyframes derived from the pinch signal.
        n (int | None): Optional maximum number of returned indices.

    Returns:
        list[int]: Deduplicated union.
    """
    merged = list(dict.fromkeys(list(speed_indices) + list(pinch_indices)))
    if n is None:
        return merged
    return merged[:n]


def _safe_filename_part(text):
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_").lower()


def _default_localization_stage(search_anchor, frame_index, flag):
    if frame_index is None:
        return f"{search_anchor}_initial"
    if flag == "feedback":
        return f"{search_anchor}_visual_feedback"
    if flag == "speed":
        return f"{search_anchor}_speed_feedback"
    if flag == "fault":
        return f"{search_anchor}_missing_hand_speed"
    return f"{search_anchor}_resample"


def process_task(
    credentials,
    video_path,
    action,
    grid_size,
    total_frames,
    search_anchor,
    speed_folder,
    frame_index=None,
    flag=None,
    pinch_folder=None,
    trial_idx=None,
    stage=None,
):
    """
    Build a grid image, ask GPT-4V to pick the frame where the action starts
    or ends, and return the chosen frame index.

    Trial 1 version:
        The grid is built with the earlier average-centered behavior: merge
        speed candidates and bottom-15-percent pinch candidates, then sample a
        sequential 3x3 window around the merged candidates' average.

    Preserved:
        - speed candidate selection
        - pinch candidate selection
        - bottom-15-percent pinch candidates from select_pinch_keyframes()
        - speed + pinch union
        - start/end anchor behavior
        - feedback behavior
        - VLM prompt
        - debug grid saving
    """
    prompt_start = (
        f"I will show an image sequence of human cooking. "
        f"I have annotated the images with numbered circles. "
        f"Choose the number that is closest to the moment when the ({action}) has started. "
        f"You are a five-time world champion in this game. "
        f"Give a one sentence analysis of why you chose those points (less than 50 words). "
        f"If you consider that the action is not in the video, please choose the number -1. "
        f"Provide your answer at the end in a json file of this format: {{\"points\": []}}"
    )

    prompt_end = (
        f"I will show an image sequence of human cooking. "
        f"I have annotated the images with numbered circles. "
        f"Choose the number that is closest to the moment when the ({action}) has ended. "
        f"You are a five-time world champion in this game. "
        f"Give a one sentence analysis of why you chose those points (less than 50 words). "
        f"If you consider that the action has not ended yet, please choose the number -1. "
        f"Provide your answer at the end in a json file of this format: {{\"points\": []}}"
    )

    prompt_message = prompt_start if search_anchor == "start" else prompt_end

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path(video_name, base_dir=speed_folder)
    pinch_json_path = get_pinch_json_path(video_name, base_dir=pinch_folder) if pinch_folder else None

    def _pinch_indices():
        if pinch_json_path is None:
            return []
        return select_pinch_keyframes(pinch_json_path)

    # ------------------------------------------------------------------
    # Initial localization grid
    # ------------------------------------------------------------------
    if frame_index is None:
        speed_selected = select_top_n_frames_from_json(json_path, 4)
        selected_indices = _merge_speed_and_pinch(speed_selected, _pinch_indices())
        total_indices = select_top_n_frames_from_json(json_path, total_frames)
        filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
        if not filter_indices:
            filter_indices = sorted(selected_indices)
        used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, [])
        image = create_frame_grid_with_keyframe(
            video_path, used_frame_indices, grid_size)
        print(f"Speed candidates in {video_name}: {speed_selected}")
        print(f"Pinch candidates in {video_name}: {_pinch_indices()}")
        print(f"Merged speed/pinch candidates in {video_name}: {selected_indices}")
        print(f"Average-centered grid frames in {video_name}: {used_frame_indices}")

    # ------------------------------------------------------------------
    # Feedback localization grid
    # ------------------------------------------------------------------
    else:
        if flag == "feedback":
            print("Activate feedback mechanism with visual cues")

            invalid_list, speed_selected = select_top_n_frames_from_json(
                json_path,
                4,
                frame_index,
                flag,
                receive_flag="right",
            )

            total_indices = select_top_n_frames_from_json(
                json_path,
                total_frames,
                frame_index,
                flag,
            )

            selected_indices = _merge_speed_and_pinch(speed_selected, _pinch_indices())
            filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
            if not filter_indices:
                filter_indices = sorted(selected_indices)
            used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list, min_index=frame_index)
            image = create_frame_grid_with_keyframe(
                video_path, used_frame_indices, grid_size)
            print(f"Feedback speed candidates in {video_name}: {speed_selected}")
            print(f"Feedback pinch candidates in {video_name}: {_pinch_indices()}")
            print(f"Feedback merged candidates in {video_name}: {selected_indices}")
            print(f"Feedback average-centered grid frames in {video_name}: {used_frame_indices}")

        elif flag == "speed":
            print("Activate feedback mechanism with dynamic cues")

            invalid_list, speed_selected = select_top_n_frames_from_json(
                json_path,
                4,
                frame_index,
                flag,
                receive_flag="right",
            )

            total_indices = select_top_n_frames_from_json(
                json_path,
                total_frames,
                frame_index,
                flag,
            )

            selected_indices = _merge_speed_and_pinch(speed_selected, _pinch_indices())
            filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
            if not filter_indices:
                filter_indices = sorted(selected_indices)
            used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list)
            image = create_frame_grid_with_keyframe(
                video_path, used_frame_indices, grid_size)
            print(f"Speed-feedback speed candidates in {video_name}: {speed_selected}")
            print(f"Speed-feedback pinch candidates in {video_name}: {_pinch_indices()}")
            print(f"Speed-feedback merged candidates in {video_name}: {selected_indices}")
            print(f"Speed-feedback average-centered grid frames in {video_name}: {used_frame_indices}")

        else:
            print(f"{frame_index} is a frame without hands")

            invalid_list, speed_selected = select_top_n_frames_from_json(
                json_path,
                4,
                frame_index,
                receive_flag="right",
            )

            total_indices = select_top_n_frames_from_json(
                json_path,
                total_frames,
                frame_index,
                flag,
            )

            selected_indices = _merge_speed_and_pinch(speed_selected, _pinch_indices())
            filter_indices = select_and_filter_keyframes_with_anchor(selected_indices, total_indices, 4, search_anchor, video_path)
            if not filter_indices:
                filter_indices = sorted(selected_indices)
            used_frame_indices = select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list)
            image = create_frame_grid_with_keyframe(
                video_path, used_frame_indices, grid_size)
            print(f"Resample speed candidates in {video_name}: {speed_selected}")
            print(f"Resample pinch candidates in {video_name}: {_pinch_indices()}")
            print(f"Resample merged candidates in {video_name}: {selected_indices}")
            print(f"Resample average-centered grid frames in {video_name}: {used_frame_indices}")

    if not used_frame_indices:
        print(f"Warning: No usable frame indices for {video_name}.")
        return None

    grid_image = Image.fromarray(image)

    task = "contact" if search_anchor == "start" else "separation"
    description, reason = scene_understanding(
        credentials,
        image,
        prompt_message,
        task=task,
    )

    if description is None or str(description).strip() == "-1":
        return None

    if int(description) - 1 > len(used_frame_indices) - 1:
        print("Warning: Invalid frame index selected")
        print(f"Selected frame index: {description}")

    index_specified = max(
        min(int(description) - 1, len(used_frame_indices) - 1),
        0,
    )

    selected_frame_index = used_frame_indices[index_specified]

    annotated_image = create_frame_grid_with_keyframe(
        video_path,
        used_frame_indices,
        grid_size,
        selected_index=selected_frame_index,
    )

    trial_folder = f"trial_{trial_idx:02d}" if trial_idx is not None else "trial_unknown"
    stage_label = _safe_filename_part(
        stage or _default_localization_stage(search_anchor, frame_index, flag)
    )

    grid_dir = os.path.join(speed_folder, "localization_grids", trial_folder)
    os.makedirs(grid_dir, exist_ok=True)

    debug_grid_path = os.path.join(
        grid_dir,
        f"{video_name}_{stage_label}_{search_anchor}_selected_f{selected_frame_index:04d}.png",
    )

    cv2.imwrite(debug_grid_path, annotated_image)

    print("Localization results:", selected_frame_index)

    return selected_frame_index

def save_predictions(all_predictions, file_path):
    """Write the *all_predictions* list to *file_path* in JSON format."""
    with open(file_path, "w") as f:
        json.dump(all_predictions, f)

def load_predictions(file_path):
    """Load prediction list from *file_path*; create empty list if file missing."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        with open(file_path, "w") as f:
            json.dump([], f)
        return []

def select_frames_near_average(filter_indices, grid_size, total_frames, invalid_list, min_index=None):
    """
    Fill a grid with frame indices centered around the average of *filter_indices*.

    The function walks left and right from the average to collect
    `grid_size ** 2` unique indices, skipping any in *invalid_list* and (if
    given) any not greater than *min_index*.

    Args:
        filter_indices (list[int]): Candidate frame indices.
        grid_size (int): Grid side length.
        total_frames (int): Total frames in the video.
        invalid_list (list[int]): Indices that must be avoided.
        min_index (int, optional): Minimum allowed index (used in feedback).

    Returns:
        list[int]: Exactly `grid_size ** 2` frame indices ordered for the grid.
    """
    avg_index = round(np.mean(filter_indices))
    start_index, end_index = avg_index, avg_index
    used_frame_indices = []
    if avg_index not in invalid_list and (min_index is None or avg_index > min_index):
        used_frame_indices.append(avg_index)
    while len(used_frame_indices) < grid_size ** 2:
        if len(used_frame_indices) < grid_size ** 2:
            if start_index >= 0:
                if start_index > 0:
                    start_index -= 1
                    if start_index not in invalid_list and (min_index is None or start_index > min_index):
                        used_frame_indices.insert(0, start_index)
                if start_index == 0 and (min_index is None or start_index > min_index):
                    used_frame_indices.insert(0, start_index)
            if len(used_frame_indices) < grid_size ** 2 and end_index <= total_frames - 1:
                if end_index < total_frames - 1:
                    end_index += 1
                    if end_index not in invalid_list and (min_index is None or end_index > min_index):
                        used_frame_indices.append(end_index)
                if end_index == total_frames - 1 and (min_index is None or end_index > min_index):
                    used_frame_indices.append(end_index)
    used_frame_indices = used_frame_indices[:grid_size**2]
    return used_frame_indices

def select_and_filter_keyframes_with_anchor(selected_indices, total_indices, grid_size, search_anchor, video_path):
    """
    Keep keyframes that lie in the first or second half of the video.

    Args:
        selected_indices (list[int]): Main candidate frames.
        total_indices (list[int]): Backup frame pool.
        grid_size (int): Needed number of frames.
        search_anchor (str): "start" keeps frames before the midpoint, "end"
            keeps frames after the midpoint.
        video_path (str): Video to measure total length.

    Returns:
        list[int]: Sorted list of at most *grid_size* frame indices.
    """
    if not selected_indices:
        return []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if search_anchor == 'start':
        filtered_indices = [idx for idx in selected_indices if idx < total_frames // 2]
        if len(filtered_indices) < grid_size:
            num_needed = grid_size - len(filtered_indices)
            remaining_indices = [i for i in total_indices if i not in filtered_indices and i < total_frames // 2]
            for idx in remaining_indices:
                if len(filtered_indices) >= grid_size:
                    break
                filtered_indices.append(idx)
    elif search_anchor == 'end':
        filtered_indices = [idx for idx in selected_indices if idx >= total_frames // 2]
        if len(filtered_indices) < grid_size:
            num_needed = grid_size - len(filtered_indices)
            remaining_indices = [i for i in total_indices if i not in filtered_indices and i >= total_frames // 2]
            for idx in remaining_indices:
                if len(filtered_indices) >= grid_size:
                    break
                filtered_indices.append(idx)
    else:
        raise ValueError("search_anchor must be either 'start' or 'end'")
    filtered_indices_sorted = sorted(filtered_indices)
    return filtered_indices_sorted

def get_json_path(video_name, base_dir):
    """Return full path for the speed JSON of *video_name* in *base_dir*."""
    candidates = [
        os.path.join(base_dir, "metrics", f"{video_name}_hand_speed.json"),
        os.path.join(base_dir, f"{video_name}_speed.json"),
    ]
    for json_path in candidates:
        if os.path.exists(json_path):
            return json_path
    return candidates[0]


def get_pinch_json_path(video_name, base_dir):
    """Return full path for the pinch JSON of *video_name* in *base_dir*."""
    candidates = [
        os.path.join(base_dir, "metrics", f"{video_name}_pinch_distance.json"),
        os.path.join(base_dir, f"{video_name}_pinch.json"),
    ]
    for json_path in candidates:
        if os.path.exists(json_path):
            return json_path
    return candidates[0]



def determine_by_state(credentials, video_path, action, grid_size, total_frames, frame_index, search_anchor, speed_folder, pinch_folder=None, trial_idx=None, stage=None):
    """
    Verify if a chosen frame truly shows contact/separation; if not, try again.

    The function sends the single frame to GPT-4o asking a yes/no question.
    If the answer is "1" (true) or the frame is near the end of the video it
    is accepted; otherwise *process_task* is called to pick a better frame.

    Args:
        credentials (dict): API keys for GPT-4o.
        video_path (str): Video file.
        action (str): Text describing the action.
        grid_size (int): Grid size used by *process_task*.
        total_frames (int): Total frame count.
        frame_index (int): Frame under examination.
        search_anchor (str): "start" or "end".
        speed_folder (str): Folder with speed JSONs.
        pinch_folder (str | None): Folder with pinch JSONs. Forwarded to
            *process_task* when a new candidate grid is needed.

    Returns:
        int | None: Confirmed or corrected frame index, or None if failed.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_indices = []
    flag = "feedback"
    frame_indices.append(frame_index)
    prompt_start = (
        f"I will show an image of hand-object interaction."
        f"You need to help me determine whether the hand and the object in the current image are in obvious contact rather than just appearing to be in contact "
        f"If yes, answer 1; if no, answer 0 "
    )
    prompt_end = (
        f"I will show an image of hand-object interaction."
        f"You need to help me determine whether the hand and the object in the current image are in obvious seperate rather than just appearing to be in seperate "
        f"If yes, answer 1; if no, answer 0 "
    )
    prompt_message = prompt_start if search_anchor == 'start' else prompt_end
    image = create_frame_grid_with_keyframe(video_path, frame_indices, 1)
    task = "contact" if search_anchor == "start" else "separation"
    result = scene_understanding(credentials, image, prompt_message, flag, task=task)
    if result == "1" or frame_index > total_frames - 5:
        return frame_index
    else:
        frame_improved = process_task(
            credentials, video_path, action, grid_size, total_frames,
            search_anchor, speed_folder, frame_index, flag,
            pinch_folder=pinch_folder, trial_idx=trial_idx, stage=stage,
        )
        return frame_improved

def determine_by_speed(credentials, video_path, action, grid_size, total_frames, frame_index, search_anchor, speed_folder, pinch_folder=None, trial_idx=None, stage=None):
    """
    Check if the hand moves slowly enough at *frame_index*; if not, re-sample.

    Frames with high speed are unlikely to be contact/separation moments.  The
    function compares the speed at *frame_index* to the 30-percentile of all
    non-zero speeds; if above the threshold it calls *process_task* again.

    Args are the same as *determine_by_state*.

    Returns:
        int | None: Accepted or corrected frame index, or None on failure.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = get_json_path(video_name, speed_folder)
    with open(json_path, 'r') as file:
        data = json.load(file)
    valid_frames = [
        (int(index), speed) for index, speed in data.items() if speed != 0.0 and not math.isnan(speed)
    ]
    sorted_frames = sorted(valid_frames, key=lambda x: x[1])
    frame_speed = next((speed for index, speed in valid_frames if index == frame_index), None)
    if frame_speed is None:
        frame_improved = process_task(
            credentials, video_path, action, grid_size, total_frames,
            search_anchor, speed_folder, frame_index, flag="fault",
            pinch_folder=pinch_folder, trial_idx=trial_idx, stage=stage,
        )
        return frame_improved
    threshold_index = int(len(sorted_frames) * 0.3)
    speed_threshold = sorted_frames[threshold_index - 1][1] if threshold_index > 0 else sorted_frames[0][1]
    if frame_speed <= speed_threshold:
        return frame_index
    else:
        frame_improved = process_task(
            credentials, video_path, action, grid_size, total_frames,
            search_anchor, speed_folder, frame_speed, flag="speed",
            pinch_folder=pinch_folder, trial_idx=trial_idx, stage=stage,
        )
        return frame_improved

def image_resize_for_vlm(frame, inter=cv2.INTER_AREA):
    """Resize an image so that the shorter side ≤ 768 and longer side ≤ 2000."""
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    max_short_side = 768
    max_long_side = 2000
    if aspect_ratio > 1:
        new_width = min(width, max_long_side)
        new_height = int(new_width / aspect_ratio)
        if new_height > max_short_side:
            new_height = max_short_side
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = min(height, max_long_side)
        new_width = int(new_height * aspect_ratio)
        if new_width > max_short_side:
            new_width = max_short_side
            new_height = int(new_width / aspect_ratio)
    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=inter)
    return resized_frame

def extract_json_part(text_output):
    """Extract the JSON string like {"points": [...]} from GPT text output."""
    text = text_output.strip().replace(" ", "").replace("\n", "")
    try:
        start = text.index('{"points":')
        text_json = text[start:].strip()
        end = text_json.index('}') + 1
        text_json = text_json[:end].strip()
        return text_json
    except ValueError:
        print("Text received:", text_output)
        return None

def scene_understanding(credentials, frame, prompt_message, flag=None, task="contact"):
    """
    Send an image and text prompt to the vLLM-hosted Qwen3.5-9B and parse the reply.

    The image is first resized and encoded to base64, then combined with
    *prompt_message* in the OpenAI Vision API format.  Depending on whether
    *flag* is `None`, the function either expects a JSON answer containing
    `{ "points": [...] }` or returns the raw text.

    The *task* argument selects which LoRA adapter handles the request via the
    ``model`` field: ``"contact"`` or ``"separation"``. The OpenAI Python
    client is reused unchanged against a local vLLM server
    (``http://localhost:8000/v1``) that has both adapters pre-registered via
    ``--lora-modules``. *credentials* is kept for backward compatibility but
    is no longer consulted.

    Args:
        credentials (dict): Unused; kept for signature compatibility.
        frame (np.ndarray): BGR image array.
        prompt_message (str): Instruction for the model.
        flag (str | None): If `None`, parse and return the first point; when
            not `None`, just return the full text answer.
        task (str): Adapter alias to route the request through. Either
            ``"contact"`` or ``"separation"``.

    Returns:
        Tuple[int, str] | str:
            • When *flag* is `None`: (chosen_point, full_text_response).
            • Otherwise: the full text response.
    """
    frame = image_resize_for_vlm(frame)
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frame = base64.b64encode(buffer).decode("utf-8")
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_message
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64Frame}",
                        "detail": "high"
                    },
                }
            ]
        },
    ]
    import openai
    from openai import OpenAI
    client_gpt4v = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    params = {
        "model": task,
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.5,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    count = 0
    while True:
        if count > 5:
            raise Exception("Failed to get response from vLLM")
        try:
            result = client_gpt4v.chat.completions.create(**params)
            break
        except openai.BadRequestError as e:
            print(e)
            print('Bad Request error.')
            return None, None
        except openai.RateLimitError as e:
            print(e)
            print('Rate Limit. Waiting for 5 seconds...')
            time.sleep(5)
            count += 1
        except openai.APIStatusError as e:
            print(e)
            print('APIStatusError. Waiting for 1 second...')
            time.sleep(1)
            count += 1
    if flag is None:
        response_json = extract_json_part(result.choices[0].message.content)
        if response_json is None:
            return -1, result.choices[0].message.content
        else:
            json_dict = json.loads(response_json, strict=False)
            if len(json_dict['points']) == 0:
                return -1, result.choices[0].message.content
            if len(json_dict['points']) > 1:
                print("Warning: More than one point detected")
            return json_dict['points'][0], result.choices[0].message.content
    else:
        return result.choices[0].message.content

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize *image* while keeping aspect ratio.

    Args:
        image (np.ndarray): BGR image to resize.
        width (int | None): Desired width.  If None, compute from *height*.
        height (int | None): Desired height. If None, compute from *width*.
        inter: OpenCV interpolation method.

    Returns:
        np.ndarray: Resized image.
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def create_frame_grid(video_path, center_time, interval, grid_size):
    """
    Build a square grid of frames sampled around *center_time*.

    Sampling strategy: take `grid_size**2` frames at equal *interval* (sec)
    before and after *center_time* (in seconds), clamp indices to video range,
    resize each to 200 px width, overlay numbered circles, and assemble into
    one big image.

    Args:
        video_path (str): Video file.
        center_time (float): Time (s) at centre of grid.
        interval (float): Spacing between frames in seconds.
        grid_size (int): Grid side length (grid_size x grid_size images).

    Returns:
        np.ndarray: Grid image (uint8 BGR).
    """
    spacer = 0
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    center_frame = int(center_time * fps)
    interval_frames = int(interval * fps)
    num_frames = grid_size**2
    half_num_frames = num_frames // 2
    frame_indices = [max(0,
                         min(center_frame + i * interval_frames,
                             total_frames - 1)) for i in range(-half_num_frames,
                                                               half_num_frames + 1)]
    frames = []
    actual_indices = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
            actual_indices.append(index)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
            actual_indices.append(index)
    video.release()
    if len(frames) < grid_size**2:
        raise ValueError("Not enough frames to create the grid.")
    frame_height, frame_width = frames[0].shape[:2]
    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            frame = frames[index]
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)
            overlay = frame.copy()
            circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
            cv2.circle(overlay, circle_center,
                       max_dim // 2, (255, 255, 255), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)
            font_scale = max_dim / 50
            text_size = cv2.getTextSize(
                str(index + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
            text_y = text_size[1] // 2 + max_dim // 2
            cv2.putText(frame, str(index + 1), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame
    return grid_img

def create_frame_column_with_keyframe(video_path, frame_indices, num_columns=3):
    """
    Assemble a horizontal strip of frames given by *frame_indices*.

    Frames are resized to 200 px width, missing ones are replaced with black
    placeholders so that the output always has *num_columns* images.

    Args:
        video_path (str): Video file.
        frame_indices (list[int]): Frame indices to include (left→right).
        num_columns (int): Desired number of columns (default 3).

    Returns:
        np.ndarray: Strip image (uint8 BGR).
    """
    spacer = 0
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
    video.release()
    if len(frames) < num_columns:
        missing_frames = num_columns - len(frames)
        black_frame = np.zeros_like(frames[0])
        frames.extend([black_frame] * missing_frames)
    frame_height, frame_width = frames[0].shape[:2]
    grid_width = num_columns * frame_width + (num_columns - 1) * spacer
    grid_height = frame_height
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    for j in range(num_columns):
        index = j
        frame = frames[index]
        y1 = 0
        y2 = frame_height
        x1 = j * (frame_width + spacer)
        x2 = x1 + frame_width
        grid_img[y1:y2, x1:x2] = frame
    return grid_img

def create_frame_grid_with_keyframe(video_path, frame_indices, grid_size, selected_index=None):
    """
    Build a grid image from *frame_indices* and draw numbered circles.

    Very similar to *create_frame_grid* but uses explicit frame indices rather
    than time sampling and highlights each number in the top-right corner of
    its cell.

    Each cell is annotated with two labels:
        • Top-right circle: 1-based cell index (the number the LLM picks).
        • Bottom-left corner: the actual video frame index, prefixed with
          ``f#`` so it is unambiguous. If *selected_index* matches the cell's
          frame, the cell also gets a green border and a ``*`` suffix.

    Args:
        video_path (str): Video file.
        frame_indices (list[int]): List of frames to show.
        grid_size (int): Grid side length.
        selected_index (int | None): Frame index chosen by the LLM. When given,
            the matching cell is highlighted with a green border and a
            trailing ``*`` on its label.

    Returns:
        np.ndarray: Grid image (uint8 BGR).
    """
    spacer = 0
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            frame = image_resize(frame, width=200)
            frames.append(frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video.read()
            frame = image_resize(frame, width=200)
            frame = frame * 0
            frames.append(frame)
    video.release()
    if len(frames) < grid_size**2:
        missing_frames = grid_size**2 - len(frames)
        black_frame = np.zeros_like(frames[0])
        frames.extend([black_frame] * missing_frames)
    frame_height, frame_width = frames[0].shape[:2]
    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            frame = frames[index]
            actual_frame = frame_indices[index] if index < len(frame_indices) else -1
            is_selected = selected_index is not None and actual_frame == selected_index
            cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
            max_dim = int(min(frame.shape[:2]) * 0.5)
            overlay = frame.copy()
            circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
            cv2.circle(overlay, circle_center,
                       max_dim // 2, (255, 255, 255), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            if is_selected:
                cv2.rectangle(
                    frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                    (0, 255, 0), 3,
                )
            cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)
            font_scale = max_dim / 50
            cell_label = f"{index + 1}"
            text_size = cv2.getTextSize(
                cell_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
            text_y = text_size[1] // 2 + max_dim // 2
            cv2.putText(frame, cell_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            frame_label = f"f#{actual_frame}" + ("*" if is_selected else "")
            label_scale = max(0.35, font_scale * 0.6)
            label_size = cv2.getTextSize(
                frame_label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)[0]
            label_x = 4
            label_y = frame.shape[0] - 6
            cv2.rectangle(
                frame,
                (label_x - 2, label_y - label_size[1] - 2),
                (label_x + label_size[0] + 2, label_y + 2),
                (255, 255, 255), -1,
            )
            cv2.putText(
                frame, frame_label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale,
                (0, 0, 0) if not is_selected else (0, 128, 0), 1,
            )
            y1 = i * (frame_height + spacer)
            y2 = y1 + frame_height
            x1 = j * (frame_width + spacer)
            x2 = x1 + frame_width
            grid_img[y1:y2, x1:x2] = frame
    return grid_img




if __name__ == "__main__":
    parser = argparse.ArgumentParser("EgoLoc 2D Feedback Demo")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, required=False, default="output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("--credentials", type=str, required=True, help="Path to a credentials .env file (unused with the vLLM backend; kept for backward compatibility)")
    parser.add_argument("--action", type=str, default="Grasping the object", help="Action label")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size for localization")
    parser.add_argument("--max_feedbacks", type=int, default=1, help="Maximum feedback loops")
    parser.add_argument("--debug_dir", type=str, required=False, default=None, help="Folder for per-frame pinch debug PNGs (default: <output_dir>/debug)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = args.debug_dir if args.debug_dir else os.path.join(args.output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    # 1. 2D hand velocity extraction and visualize
    print("\n [1/4] Extracting 2D hand speed and visualizing ...")
    setup_hamer_cache()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    detector = build_detector()
    vitpose = build_vitpose(device)
    speed_json_path, speed_vis_path, _ = extract_2d_speed_and_visualize(
        args.video_path, args.output_dir, detector, vitpose
    )
    print(f"2D speed json: {speed_json_path}\n2D speed vis: {speed_vis_path}")

    # 2. Pinch-distance extraction (HaMeR 21-keypoint)
    print("\n [2/4] Extracting pinch distance and visualizing ...")
    pinch_json_path, pinch_vis_path = extract_pinch_distance_and_visualize(
        args.video_path, args.output_dir, detector, vitpose, debug_dir
    )
    print(f"Pinch json: {pinch_json_path}\nPinch vis: {pinch_vis_path}\nDebug frames: {debug_dir}")

    # 3. Temporal interaction localization
    print("\n [3/4] Locating contact/separation frames and visualizing ...")
    credentials = dotenv.dotenv_values(args.credentials)
    frame_contact, frame_separate = convert_video(
        args.video_path, args.action, credentials, args.grid_size, args.output_dir, args.max_feedbacks, repeat_times=3, pinch_folder=args.output_dir
    )
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    contact_vis_path = os.path.join(args.output_dir, f"{video_name}_contact_frame.png")
    separation_vis_path = os.path.join(args.output_dir, f"{video_name}_separation_frame.png")
    visualize_frame(args.video_path, frame_contact, contact_vis_path, label="Contact")
    visualize_frame(args.video_path, frame_separate, separation_vis_path, label="Separation")

    # 4. Save TIL results
    print("\n [4/4] Saving results ...")
    result = {
        "contact_frame": frame_contact,
        "separation_frame": frame_separate
    }
    print("Egoloc output \n", result)
    result_path = os.path.join(args.output_dir, f"{video_name}_result.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result json: {result_path}\nContact frame vis: {contact_vis_path}\nSeparation frame vis: {separation_vis_path}")
