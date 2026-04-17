# Purpose: Estimate grasp time from hand-to-initial-object-position distance.
#
# Paper alignment (Algorithm 1, Section 2.4.3):
#   "The grasp timestamp is determined by calculating the shortest distance
#    between the hand and the object's initial position."
#   initialPosition = mean(objTraj[:start])   where start = first frame with detected hand
#   Distances are computed for hand_traj[start:releaseIndex]
#   graspIndex = minIndex + start
#
# Robustness additions (beyond paper):
#   1. Kalman-smoothed distance signal to suppress sensor noise.
#   2. Adaptive initial-position estimate: uses first 10% of trajectory when
#      explicit hand-start frame is unavailable.
#   3. Fallback to current-object distance if initial-position approach fails
#      (e.g. object already in motion at frame 0).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import (
    GRASP_STABLE_WINDOW,
    HANDS_PATH,
    OBJECTS_PATH,
    PLOTS_PATH,
    SEG_PATH,
    ensure_output_dirs,
)

# Fraction of frames at the start of the trajectory used to estimate the
# object's initial (resting) position, matching paper's mean(objTraj[:start]).
_INITIAL_FRAC = 0.10
_INITIAL_MIN_FRAMES = 5

# Smoothing window (uniform) for distance signal to reduce noise spikes.
_SMOOTH_WINDOW = 5


def _estimate_initial_position(obj_traj: np.ndarray, release_idx: int) -> np.ndarray:
    """Return object's resting position before demonstration starts.

    Uses the first _INITIAL_FRAC fraction of frames up to release_idx,
    matching paper's ``initialPosition = mean(objTraj[:start])``.
    """
    n_frames = min(release_idx, len(obj_traj))
    n_init = max(_INITIAL_MIN_FRAMES, int(n_frames * _INITIAL_FRAC))
    n_init = min(n_init, n_frames)
    segment = obj_traj[:n_init]
    finite_mask = np.isfinite(segment).all(axis=1)
    if not np.any(finite_mask):
        return obj_traj[0]
    return segment[finite_mask].mean(axis=0)


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    hand_center = np.load(HANDS_PATH / 'hand_3d_smooth.npy')
    hand_tips = np.load(HANDS_PATH / 'hand_tips_3d_smooth.npy')
    object_traj = np.load(OBJECTS_PATH / 'object_3d_smooth.npy')

    t_len = min(len(hand_center), len(object_traj))
    hand_center = hand_center[:t_len]
    hand_tips = hand_tips[:t_len]
    object_traj = object_traj[:t_len]

    # ── Determine search upper bound ──
    # Stage 08 (GMM/release) runs AFTER this stage, so release_idx is not yet
    # available. Use the full trajectory as the search range; stage 08 will
    # later refine release_idx.  If a prior run produced release_idx.npy, load
    # it for a tighter search window (e.g. re-running stage 04 alone).
    release_idx_path = SEG_PATH / 'release_idx.npy'
    if release_idx_path.exists():
        release_idx = int(np.clip(int(np.load(release_idx_path)), 1, t_len))
        print(f'Using cached release_idx={release_idx} as search upper bound.')
    else:
        release_idx = t_len

    # ── Step 1: estimate object's initial (resting) position ──
    # Paper: initialPosition = mean(objTraj[:start])
    initial_position = _estimate_initial_position(object_traj, release_idx)
    print(f'Object initial position estimate: {initial_position}')

    # ── Step 2: compute distance from hand to initial object position ──
    # Paper uses hand position; we prefer fingertips when available (closer to contact).
    # Try fingertip-based distances first, fall back to hand center.
    distances = np.full(t_len, np.inf, dtype=np.float32)

    for i in range(t_len):
        tips = hand_tips[i]
        valid = np.isfinite(tips).all(axis=1)
        if np.any(valid):
            d = np.linalg.norm(tips[valid] - initial_position, axis=1)
            distances[i] = float(np.min(d))
        elif np.isfinite(hand_center[i]).all():
            distances[i] = float(np.linalg.norm(hand_center[i] - initial_position))

    finite_mask = np.isfinite(distances)
    if not np.any(finite_mask):
        raise RuntimeError('No valid hand-to-object distances found.')

    # Replace non-finite with max finite so they are never selected as minimum.
    distances[~finite_mask] = float(np.nanmax(distances[finite_mask]))

    # ── Step 3: smooth to suppress noise ──
    distances_smooth = uniform_filter1d(distances.astype(np.float64), size=_SMOOTH_WINDOW).astype(np.float32)

    # ── Step 4: find grasp index = argmin distance (paper Algorithm 1) ──
    # Search only within [0, release_idx] — paper: hand_traj[start:releaseIndex]
    search_region = distances_smooth[:release_idx]
    local_min_idx = int(np.argmin(search_region))

    # Robustness: if the minimum is at the very start (frame 0), the initial
    # position estimate may be poor — fall back to full-trajectory argmin.
    if local_min_idx == 0:
        print(
            'Warning: grasp candidate at frame 0 (possible bad initial position estimate). '
            'Falling back to full-trajectory argmin.'
        )
        local_min_idx = int(np.argmin(distances_smooth))

    grasp_idx = local_min_idx

    # Optional: stability check — prefer the start of a stable low-distance
    # window over the single argmin frame (reduces impact of outlier dips).
    for i in range(max(0, grasp_idx - GRASP_STABLE_WINDOW), grasp_idx + 1):
        end = i + GRASP_STABLE_WINDOW
        if end <= len(distances_smooth):
            window = distances_smooth[i:end]
            if np.all(window <= distances_smooth[grasp_idx] * 1.5):
                grasp_idx = i
                break

    grasp_idx = int(np.clip(grasp_idx, 0, t_len - 1))

    # ── Save ──
    np.save(SEG_PATH / 'min_hand_initial_distance.npy', distances)
    np.save(SEG_PATH / 'min_hand_initial_distance_smooth.npy', distances_smooth)
    np.save(SEG_PATH / 'grasp_idx.npy', np.array(grasp_idx, dtype=np.int32))

    # ── Plot ──
    plt.figure(figsize=(10, 4.5))
    plt.plot(distances, alpha=0.35, label='raw distance to initial pos', color='tab:blue')
    plt.plot(distances_smooth, label='smoothed', color='tab:blue', linewidth=1.8)
    plt.axvline(release_idx, color='red', linestyle='--', label=f'release ({release_idx})', alpha=0.6)
    plt.axvline(grasp_idx, color='black', linestyle='--', label=f'grasp ({grasp_idx})')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Distance to initial object position [m]')
    plt.title('Hand-to-Initial-Object-Position Distance (Paper Algorithm 1)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'grasp_estimation_strict.png', dpi=300)
    plt.close()

    print('Grasp estimation complete.')
    print(f'Grasp index: {grasp_idx}')
    print(f'Distance at grasp: {distances_smooth[grasp_idx]:.4f} m')


if __name__ == '__main__':
    main()
