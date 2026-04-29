"""Monocular depth estimation helpers for extracted RGB frames.

This module is intentionally lightweight and dependency-friendly so the
pipeline can always produce depth `.npy` files even when a learned depth model
is not available in the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


def estimate_depth_single_frame(frame: np.ndarray) -> np.ndarray:
    """Estimate a pseudo depth map for a single BGR/RGB frame."""

    if frame is None:
        raise ValueError("frame must not be None")

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    normalized = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth = 1.0 - normalized
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=3.0)

    depth_meters = 0.25 + (depth * 4.75)
    return depth_meters.astype(np.float32)


def _save_depth_preview(depth: np.ndarray, preview_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        d_min = float(valid.min())
        d_max = float(valid.max())
        if d_max - d_min < 1e-6:
            normalized = np.zeros_like(depth, dtype=np.uint8)
        else:
            normalized = ((np.clip(depth, d_min, d_max) - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imwrite(str(preview_path), color)


def estimate_depth_for_frames(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    preview_dir: str | Path | None = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Run pseudo-depth estimation on a directory of RGB frames."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if preview_dir is not None:
        preview_dir = Path(preview_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frame_files = sorted(
        list(input_dir.glob("*.png"))
        + list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg")),
        key=lambda p: p.name,
    )
    if not frame_files:
        raise FileNotFoundError(f"No frame images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for frame_path in frame_files:
        depth_path = output_dir / f"{frame_path.stem}.npy"
        preview_path = (preview_dir / f"{frame_path.stem}.png") if preview_dir is not None else None

        if depth_path.exists() and not overwrite:
            continue

        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        depth = estimate_depth_single_frame(frame)
        depth[np.isnan(depth)] = 0.0
        depth[np.isinf(depth)] = 0.0

        np.save(str(depth_path), depth)
        if preview_path is not None:
            _save_depth_preview(depth, preview_path)
        saved_count += 1

    return {
        "count": saved_count,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "preview_dir": str(preview_dir) if preview_dir is not None else None,
        "frame_count": len(frame_files),
    }


def _find_latest_upload(uploads_dir: str = "data/Generic/uploads") -> Optional[Path]:
    """Return the most recently modified file in uploads_dir, or None."""

    upload_dir = Path(uploads_dir)
    if not upload_dir.exists():
        return None

    files = [path for path in upload_dir.iterdir() if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def run_default(
    uploads_dir: str = "data/Generic/uploads",
    frames_dir: str = "data/Generic/frames",
    depth_dir: str = "data/Generic/depth_meters",
) -> Dict[str, Any]:
    """Convenience runner that processes the latest uploaded video."""

    latest = _find_latest_upload(uploads_dir)
    if latest is None:
        raise FileNotFoundError(f"No files found in {uploads_dir}. Please upload a video first.")

    import hashlib

    session_id = hashlib.md5(str(latest).encode()).hexdigest()[:8]
    frames_out = Path(frames_dir) / session_id
    depth_out = Path(depth_dir) / session_id
    preview_out = Path("data/Generic/depth_color") / session_id

    return estimate_depth_for_frames(
        input_dir=frames_out,
        output_dir=depth_out,
        preview_dir=preview_out,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate pseudo depth for extracted RGB frames.")
    parser.add_argument("--input", type=str, default="data/Generic/frames", help="Input frame directory")
    parser.add_argument("--output", type=str, default="data/Generic/depth_meters", help="Output depth directory")
    parser.add_argument("--preview", type=str, default="data/Generic/depth_color", help="Preview output directory")
    args = parser.parse_args()

    result = estimate_depth_for_frames(args.input, args.output, preview_dir=args.preview)
    print(f"Saved {result['count']} depth maps to {result['output_dir']}")