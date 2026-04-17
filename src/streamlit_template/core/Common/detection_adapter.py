# Purpose: Bridge between the class-agnostic ObjectDetector and the existing
#          object trajectory pipeline (03_extract_object_trajectory.py).
#
# The existing tracker expects per-frame: a pixel center (u, v) of the
# highest-confidence detection, and a depth map to project into 3D.
# This adapter runs the agnostic detector, picks the best detection,
# and produces the same output arrays the downstream pipeline consumes.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.streamlit_template.core.Common.object_detector import ObjectDetector
from src.streamlit_template.core.SVO._paper_utils import (
    CameraIntrinsics,
    load_intrinsics,
    median_valid_depth,
    to_camera_xyz,
)


def run_agnostic_detection_pipeline(
    rgb_dir: Path,
    depth_dir: Path,
    intrinsics_path: Path,
    output_dir: Path,
    plots_dir: Optional[Path] = None,
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    frame_skip: int = 1,
    device: Optional[str] = None,
) -> np.ndarray:
    """Run class-agnostic detection and produce 3D object trajectory.

    This is a drop-in replacement for the detection portion of
    ``03_extract_object_trajectory.py``, but uses the agnostic detector
    instead of the OBB model.

    Parameters
    ----------
    rgb_dir : Path
        Directory with RGB frame images (png/jpg).
    depth_dir : Path
        Directory with depth ``.npy`` files (float32, meters).
    intrinsics_path : Path
        Path to camera intrinsics ``.npy`` file.
    output_dir : Path
        Where to save ``object_3d_raw.npy`` and ``object_center_uv_raw.npy``.
    plots_dir : Path | None
        Where to save trajectory plot (optional).
    model_path : str
        YOLO model name or path (default ``yolov8n.pt``).
    confidence_threshold : float
        Minimum confidence for detections.
    frame_skip : int
        Process 1 out of every ``frame_skip`` frames. Skipped frames
        repeat the previous detection (hold-last-value).
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto).

    Returns
    -------
    np.ndarray
        3D trajectory array of shape ``(N, 3)`` in camera frame (meters).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)

    intr = load_intrinsics(intrinsics_path)
    detector = ObjectDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    rgb_files = sorted(
        p.name for p in Path(rgb_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    traj: list[np.ndarray] = []
    uv_centers: list[list[int]] = []

    for idx, fname in enumerate(tqdm(rgb_files, desc="Agnostic detection")):
        # Frame-skip: hold last value for skipped frames
        if idx % max(frame_skip, 1) != 0:
            if traj:
                traj.append(traj[-1].copy())
                uv_centers.append(uv_centers[-1].copy())
            else:
                traj.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                uv_centers.append([-1, -1])
            continue

        frame = cv2.imread(str(Path(rgb_dir) / fname))
        depth_path = Path(depth_dir) / Path(fname).with_suffix(".npy").name
        if not depth_path.exists():
            # Fallback: try matching index pattern
            depth_path = Path(depth_dir) / f"frame_{idx:06d}.npy"

        depth = np.load(str(depth_path)) if depth_path.exists() else None

        dets = detector.detect_frame(frame)

        found = False
        if dets and depth is not None:
            # Pick highest-confidence detection (same strategy as the OBB tracker)
            best = max(dets, key=lambda d: d.confidence)
            cx, cy = best.bbox_xywh[0], best.bbox_xywh[1]
            u_i, v_i = int(round(float(cx))), int(round(float(cy)))
            z = median_valid_depth(depth, u=u_i, v=v_i, half=2)
            if z is not None:
                traj.append(to_camera_xyz(u_i, v_i, z, intr))
                uv_centers.append([u_i, v_i])
                found = True

        if not found:
            if traj:
                traj.append(traj[-1].copy())
                uv_centers.append(uv_centers[-1].copy())
            else:
                traj.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                uv_centers.append([-1, -1])

    traj_np = np.asarray(traj, dtype=np.float32)
    uv_np = np.asarray(uv_centers, dtype=np.int32)

    np.save(output_dir / "object_3d_raw.npy", traj_np)
    np.save(output_dir / "object_center_uv_raw.npy", uv_np)

    if plots_dir:
        from src.streamlit_template.core.SVO._paper_utils import save_plot_xyz
        save_plot_xyz(
            plots_dir / "object_trajectory_raw.png",
            traj_np,
            "Object Trajectory — Agnostic Detection",
        )

    print(f"Agnostic detection complete. Shape: {traj_np.shape}")
    return traj_np
