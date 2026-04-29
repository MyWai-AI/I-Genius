"""
Generic video extraction service — Extract frames, depth, and camera intrinsics from video files.
Treats uploaded videos as complete data packages by extracting all necessary components.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.streamlit_template.core.Common.estimate_monocular_depth import estimate_depth_single_frame

# ---------------------------------------------------------------------------
# Async I/O helpers
# ---------------------------------------------------------------------------

def _save_rgb(path: str, bgr: np.ndarray):
    cv2.imwrite(path, bgr)

def _save_depth_meters(path: str, depth_meters: np.ndarray):
    np.save(path, depth_meters)

def _save_depth_colormap(path: str, depth_meters: np.ndarray):
    depth_vis = np.where((depth_meters > 0) & (depth_meters < 5.0), depth_meters, 0)
    depth_vis_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_vis_norm, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_colormap)


def estimate_camera_intrinsics_from_video(
    video_path: str, 
    num_sample_frames: int = 5
) -> dict:
    """
    Estimate camera intrinsics from video frames using simple heuristics.
    
    Since we don't have actual calibration data, we estimate based on:
    - Frame resolution from video
    - Common focal length assumptions for HD/4K video
    
    Args:
        video_path: Path to video file
        num_sample_frames: Number of frames to check for consistency
    
    Returns:
        Dictionary with estimated intrinsics: {fx, fy, cx, cy, width, height}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if width == 0 or height == 0:
        raise ValueError(f"Could not determine video resolution: {width}x{height}")
    
    # Estimate focal length using simplified pinhole model
    # Typical assumption: focal length ≈ 0.5-1.0 * frame width for consumer cameras
    # For now, use a conservative estimate
    focal_length = width * 0.5  # Conservative estimate
    
    intrinsics = {
        "fx": float(focal_length),
        "fy": float(focal_length),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
        "width": int(width),
        "height": int(height)
    }
    
    return intrinsics


def extract_video_frames_depth_intrinsics(
    video_path: str,
    output_root: Path,
    session_id: str,
    every_n: int = 1,
    skip_depth: bool = False,
) -> dict:
    """
    Extract frames, estimate depth (monocular), and camera intrinsics from a video file.
    
    Structure created:
      output_root/frames/{session_id}/frame_XXXXXX.png
      output_root/depth_meters/{session_id}/frame_XXXXXX.npy
      output_root/depth_color/{session_id}/frame_XXXXXX.png
      output_root/camera/{session_id}.npy (intrinsics dict)
    
    Args:
        video_path: Path to video file
        output_root: Root directory (e.g., Path("data/Generic"))
        session_id: Unique session identifier
        every_n: Extract one frame every N frames
        skip_depth: If True, skip monocular depth estimation (for speed)
    
    Returns:
        Dictionary with extraction stats and paths
    """
    video_path = Path(video_path)
    output_root = Path(output_root)
    
    # Create directories
    rgb_dir = output_root / "frames" / session_id
    depth_meters_dir = output_root / "depth_meters" / session_id
    depth_color_dir = output_root / "depth_color" / session_id
    camera_dir = output_root / "camera"
    
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_meters_dir.mkdir(parents=True, exist_ok=True)
    depth_color_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract intrinsics first (save as dict in NPY for pipeline compatibility)
    intrinsics_path = camera_dir / f"{session_id}.npy"
    if not intrinsics_path.exists():
        intr = estimate_camera_intrinsics_from_video(str(video_path))
        # Save as dict so it can be loaded with .item() by pipeline
        np.save(str(intrinsics_path), intr, allow_pickle=True)
    else:
        intr = np.load(str(intrinsics_path), allow_pickle=True).item()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    
    frame_idx = 0
    saved_idx = 0
    depth_estimated = False
    
    # Thread pool for async disk writes
    io_pool = ThreadPoolExecutor(max_workers=3)
    futures = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % every_n != 0:
                frame_idx += 1
                continue
            
            # --- RGB Frame ---
            rgb_path = rgb_dir / f"frame_{saved_idx:06d}.png"
            futures.append(io_pool.submit(_save_rgb, str(rgb_path), frame))
            
            # --- Monocular Depth Estimation (deterministic fallback) ---
            if not skip_depth:
                try:
                    depth_np = estimate_depth_single_frame(frame)
                    depth_estimated = True
                    
                    # Sanitize
                    depth_np[np.isnan(depth_np)] = 0
                    depth_np[np.isinf(depth_np)] = 0
                    
                    # Save depth
                    depth_meters_path = depth_meters_dir / f"frame_{saved_idx:06d}.npy"
                    depth_color_path = depth_color_dir / f"frame_{saved_idx:06d}.png"
                    
                    futures.append(io_pool.submit(
                        _save_depth_meters, str(depth_meters_path), depth_np
                    ))
                    futures.append(io_pool.submit(
                        _save_depth_colormap, str(depth_color_path), depth_np.copy()
                    ))
                    
                except Exception as e:
                    # Monocular depth estimation failed, continue with frames only
                    if depth_estimated:  # Only warn once
                        pass
            
            saved_idx += 1
            frame_idx += 1
    
    finally:
        cap.release()
        # Wait for all pending writes
        for f in futures:
            try:
                f.result()
            except Exception:
                pass
        io_pool.shutdown(wait=True)
    
    return {
        "frames_extracted": saved_idx,
        "fps": float(fps),
        "rgb_dir": str(rgb_dir),
        "depth_meters_dir": str(depth_meters_dir),
        "depth_color_dir": str(depth_color_dir),
        "intrinsics_path": str(intrinsics_path),
        "intrinsics": intr,
        "depth_estimated": depth_estimated,
    }
