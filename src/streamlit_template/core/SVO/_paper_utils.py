# Purpose: Shared geometry, smoothing, projection, and plotting helpers for SVO pipeline.
# Notes: Structured stage with stable defaults for repeatable runs.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def load_intrinsics(path: Path) -> CameraIntrinsics:
    """Load camera intrinsics from numpy archive formats."""
    data_raw = np.load(path, allow_pickle=True)
    if hasattr(data_raw, "item"):
        try:
            data = data_raw.item()
        except ValueError:
            data = {k: data_raw[k] for k in data_raw.files}
    else:
        data = data_raw
    return CameraIntrinsics(
        fx=float(data['fx']),
        fy=float(data['fy']),
        cx=float(data['cx']),
        cy=float(data['cy']),
        width=int(data['width']) if 'width' in data else -1,
        height=int(data['height']) if 'height' in data else -1,
    )


def to_camera_xyz(u: float, v: float, z: float, intr: CameraIntrinsics) -> np.ndarray:
    """Project a pixel and depth value into camera-frame XYZ."""
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.array([x, y, z], dtype=np.float32)


def median_valid_depth(depth_m: np.ndarray, u: int, v: int, half: int = 2) -> float | None:
    """Return robust median depth around a pixel from a local patch."""
    h, w = depth_m.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    x0, x1 = max(0, u - half), min(w, u + half + 1)
    y0, y1 = max(0, v - half), min(h, v + half + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def kalman_smooth_3d(traj: np.ndarray, q: float, r: float) -> np.ndarray:
    """Smooth a 3D trajectory with a constant-position Kalman filter."""
    # Simple constant-position Kalman model applied independently per axis.
    traj = np.asarray(traj, dtype=np.float32)
    out = np.zeros_like(traj)
    for axis in range(traj.shape[1]):
        z = traj[:, axis].astype(np.float64)
        finite = np.isfinite(z)
        if not np.any(finite):
            out[:, axis] = 0.0
            continue

        # Fill NaNs by linear interpolation and edge clamping.
        idx = np.arange(len(z), dtype=np.float64)
        z_filled = np.interp(idx, idx[finite], z[finite])

        x_est = float(z_filled[0])
        p_est = 1.0
        for i, zi in enumerate(z_filled):
            p_pred = p_est + q
            k = p_pred / (p_pred + r)
            x_est = x_est + k * (float(zi) - x_est)
            p_est = (1.0 - k) * p_pred
            out[i, axis] = x_est
    return out


def auto_kalman_params(traj: np.ndarray, base_q: float, base_r: float) -> tuple[float, float]:
    """Estimate stable Kalman Q/R values from motion jitter."""
    traj = np.asarray(traj, dtype=np.float64)
    if len(traj) < 6:
        return base_q, base_r

    vel = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    finite = np.isfinite(vel)
    if not np.any(finite):
        return base_q, base_r
    vel = vel[finite]
    p50 = float(np.percentile(vel, 50))
    p90 = float(np.percentile(vel, 90))
    if p50 < 1e-9:
        return base_q, base_r

    jitter_ratio = p90 / p50
    # Higher jitter => trust measurements less (higher R), use slightly lower Q.
    r_scale = float(np.clip(jitter_ratio / 2.5, 0.8, 3.0))
    q_scale = float(np.clip(1.0 / r_scale, 0.35, 1.25))
    return base_q * q_scale, base_r * r_scale


def stable_hand_bbox_center(landmarks, width: int, height: int) -> tuple[int, int]:
    """Compute a stable hand center from landmark extents."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    u = int(round(((min(xs) + max(xs)) * 0.5) * width))
    v = int(round(((min(ys) + max(ys)) * 0.5) * height))
    return u, v


def save_plot_xyz(path: Path, traj: np.ndarray, title: str, xlabel: str = 'Timestamp [frame]') -> None:
    """Save an XYZ-vs-time line plot for a trajectory."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4.5))
    plt.plot(traj[:, 0], label='X')
    plt.plot(traj[:, 1], label='Y')
    plt.plot(traj[:, 2], label='Z')
    plt.xlabel(xlabel)
    plt.ylabel('Position [m]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
