"""
Skill Reuse Service — Compute offset-adjusted trajectories for new target objects.

Given:
  - The original skill_reuse_traj (from the pipeline DMP step)
  - The 3D position of the originally-tracked object (anchor)
  - The 3D position of a new target object (from detection + depth)
  - A user-specified release/goal position

Produces:
  - A new skill_reuse_<label>.csv with the trajectory shifted by the 3D offset
  - Joint trajectory via IK for push-to-robot
"""
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """A single object detection with 3D position."""
    label: str
    confidence: float
    bbox_xyxy: np.ndarray        # [x1, y1, x2, y2]
    center_uv: Tuple[int, int]   # pixel center
    xyz: Optional[np.ndarray]    # 3D camera-frame position (may be None if no depth)


def load_intrinsics_dict(path: str) -> dict:
    """Load intrinsics from .npy dict format used throughout the pipeline."""
    raw = np.load(path, allow_pickle=True)
    if hasattr(raw, "item"):
        try:
            return raw.item()
        except ValueError:
            return {k: raw[k] for k in raw.files}
    return dict(raw)


def _pixel_to_xyz(u: int, v: int, depth_m: np.ndarray, intr: dict) -> Optional[np.ndarray]:
    """Back-project pixel + depth to 3D camera coordinates."""
    h, w = depth_m.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    half = 2
    x0, x1 = max(0, u - half), min(w, u + half + 1)
    y0, y1 = max(0, v - half), min(h, v + half + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    z = float(np.median(valid))
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def pixel_to_xyz(u: int, v: int, depth_m: np.ndarray, intr: dict) -> Optional[np.ndarray]:
    """Public wrapper for converting image pixel coordinates into 3D camera coordinates."""
    return _pixel_to_xyz(u, v, depth_m, intr)


def _next_available_stem(output_dir: Path, base_stem: str) -> str:
    """Return a non-colliding stem by appending an incrementing suffix when needed."""
    if not (output_dir / f"{base_stem}.npy").exists():
        return base_stem

    idx = 1
    while True:
        candidate = f"{base_stem}_{idx}"
        if not (output_dir / f"{candidate}.npy").exists():
            return candidate
        idx += 1


def detect_objects_on_frame(
    frame_path: str,
    depth_path: Optional[str],
    intrinsics: dict,
    model_path: str = "yolov8x.pt",
    conf: float = 0.05,
    max_area_pct: float = 15.0,
    min_area_pct: float = 0.0001,
) -> List[DetectedObject]:
    """Run YOLO on a single frame and produce DetectedObject list with 3D positions."""
    from ultralytics import YOLO

    bgr = cv2.imread(frame_path)
    if bgr is None:
        return []

    depth_m = None
    if depth_path and Path(depth_path).exists():
        depth_m = np.load(depth_path)

    model = YOLO(model_path)
    results = model.predict(bgr, verbose=False, conf=conf, imgsz=1280, max_det=100, agnostic_nms=True)
    if not results:
        return []

    fr_h, fr_w = bgr.shape[:2]
    r0 = results[0]
    names = r0.names or {}
    detections: List[DetectedObject] = []

    has_obb = r0.obb is not None and len(r0.obb) > 0
    has_boxes = r0.boxes is not None and len(r0.boxes) > 0

    raw_boxes = []  # list of (xyxy, conf, label)
    if has_obb:
        for obox in r0.obb:
            c = float(obox.conf[0])
            cls_id = int(obox.cls[0]) if obox.cls is not None else None
            label = names.get(cls_id) if cls_id is not None else None
            xywhr = obox.xywhr[0].cpu().numpy()
            rect = ((xywhr[0], xywhr[1]), (xywhr[2], xywhr[3]), float(np.degrees(xywhr[4])))
            pts = cv2.boxPoints(rect)
            x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
            x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
            raw_boxes.append((np.array([x1, y1, x2, y2], dtype=np.float32), c, label))
    elif has_boxes:
        for b, bc, bcls in zip(r0.boxes.xyxy, r0.boxes.conf, r0.boxes.cls):
            bn = b.cpu().numpy().astype(np.float32)
            label = names.get(int(bcls)) if names else None
            raw_boxes.append((bn, float(bc), label))

    for xyxy, conf_val, label in raw_boxes:
        bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        area = bw * bh
        if area > (fr_h * fr_w * (max_area_pct / 100.0)):
            continue
        if area < (fr_h * fr_w * (min_area_pct / 100.0)):
            continue

        cu, cv_coord = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
        xyz = None
        if depth_m is not None:
            xyz = _pixel_to_xyz(cu, cv_coord, depth_m, intrinsics)

        detections.append(DetectedObject(
            label=label or "unknown",
            confidence=conf_val,
            bbox_xyxy=xyxy,
            center_uv=(cu, cv_coord),
            xyz=xyz,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def compute_skill_reuse_for_target(
    original_skill_reuse_path: str,
    anchor_xyz: np.ndarray,
    target_xyz: np.ndarray,
    release_xyz: Optional[np.ndarray],
    seg_dir: str,
    output_dir: str,
    target_label: str = "object",
) -> dict:
    """Compute a new skill_reuse trajectory shifted by the 3D offset from anchor to target.

    The anchor is the object that was tracked in the original pipeline.
    The target is the new object the user wants to apply the skill to.

    If release_xyz is provided, the release position in the trajectory is also
    adjusted to match the user's desired goal position.

    Returns dict with keys: csv_path, npy_path, trajectory, offset, grasp_idx, release_idx
    """
    seg = Path(seg_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    original_traj = np.load(original_skill_reuse_path)
    offset_3d = target_xyz - anchor_xyz

    # Load segmentation info to know grasp/release indices
    reach_path = seg / "reach_traj.npy"
    move_path = seg / "move_traj.npy"
    reach_len = len(np.load(str(reach_path))) if reach_path.exists() else 0
    move_len = len(np.load(str(move_path))) if move_path.exists() else 0
    grasp_idx = reach_len
    release_idx = reach_len + 1 + move_len

    # Shift entire trajectory by the 3D offset
    new_traj = original_traj.copy() + offset_3d

    # If user specified a release position, adjust the move phase goal + post-release
    if release_xyz is not None and release_idx < len(new_traj):
        # Compute delta from current release pos to desired release pos
        current_release = new_traj[release_idx]
        release_delta = release_xyz - current_release
        # Shift the move phase linearly ramp from 0 at grasp to full delta at release
        if release_idx > grasp_idx + 1:
            move_start = grasp_idx + 1
            move_end = release_idx
            n_move = move_end - move_start
            for i in range(n_move):
                alpha = (i + 1) / (n_move + 1)
                new_traj[move_start + i] += release_delta * alpha
        # Set release position exactly
        new_traj[release_idx] = release_xyz
        # Shift post-release hold to new release position
        if release_idx + 1 < len(new_traj):
            new_traj[release_idx + 1:] = release_xyz

    # Save
    safe_label = "".join(c for c in target_label if c.isalnum() or c in ("_", "-")).lower()
    safe_label = safe_label or "object"
    base_stem = f"skill_reuse_{safe_label}"
    final_stem = _next_available_stem(out, base_stem)

    npy_path = out / f"{final_stem}.npy"
    csv_path = out / f"{final_stem}.csv"
    json_path = out / f"{final_stem}.json"

    np.save(str(npy_path), new_traj)
    np.savetxt(str(csv_path), new_traj, delimiter=",", header="x,y,z", comments="")
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(new_traj.tolist(), f)

    # Save metadata
    meta = {
        "trajectory_name": final_stem,
        "target_label": target_label,
        "anchor_xyz": anchor_xyz.tolist(),
        "target_xyz": target_xyz.tolist(),
        "offset_3d": offset_3d.tolist(),
        "release_xyz": release_xyz.tolist() if release_xyz is not None else None,
        "grasp_idx": int(grasp_idx),
        "release_idx": int(release_idx),
        "num_points": len(new_traj),
    }
    with open(str(out / f"{final_stem}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "csv_path": str(csv_path),
        "npy_path": str(npy_path),
        "trajectory": new_traj,
        "offset": offset_3d,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "meta": meta,
    }


def compute_ik_for_reuse_traj(
    skill_reuse_npy: str,
    urdf_path: str,
    robot_config: dict,
    grasp_idx: int,
    release_idx: int,
) -> dict:
    """Run the full robot IK pipeline on a reuse trajectory.

    Returns the same structure as handle_svo_robot step_results["robot"].
    """
    from src.streamlit_template.core.Common.robot_playback import (
        dmp_xyz_to_cartesian,
        compute_ik_trajectory,
    )

    traj = np.load(skill_reuse_npy)
    target_len = len(traj)

    default_offset = robot_config.get("dmp_offset", [0.4, 0.0, 0.2])
    default_scale = robot_config.get("dmp_scale", [0.5, 0.5, 0.5])
    dmp_rot_z = robot_config.get("dmp_rotation_z", 90.0)
    dmp_flip_z = robot_config.get("flip_z", False)
    dmp_arm_reach = robot_config.get("arm_reach", 0.0)

    cart = dmp_xyz_to_cartesian(
        dmp_npy=skill_reuse_npy,
        scale_xyz=tuple(default_scale),
        offset_xyz=tuple(default_offset),
        flip_y=False,
        flip_z=dmp_flip_z,
        rotate_z=dmp_rot_z,
        add_arch=False,
        target_frames=target_len,
        arm_reach=dmp_arm_reach,
    )
    cart_path = cart["cartesian_path"]

    ik = compute_ik_trajectory(urdf_path=urdf_path, cartesian_path=cart_path)
    q_traj = ik["q_traj"]

    video_fps = 15.0
    num_frames = len(q_traj)
    frame_timestamps = []
    sr_release = min(release_idx, num_frames - 1)
    n_pre = min(sr_release + 1, num_frames)
    n_post = num_frames - n_pre

    for i in range(num_frames):
        if i < n_pre:
            frame_timestamps.append(i / video_fps)
        else:
            post_i = i - n_pre
            frame_timestamps.append((n_pre + post_i * max(1, num_frames - n_pre) / max(n_post, 1)) / video_fps)

    return {
        "q_traj": q_traj,
        "cart_path": cart_path,
        "frame_timestamps": frame_timestamps,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "num_frames": num_frames,
    }
