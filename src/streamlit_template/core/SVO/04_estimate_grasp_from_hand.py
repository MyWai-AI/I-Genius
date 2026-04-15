# Purpose: Estimate grasp time using strict minimum 3D fingertip-object distance.
# Notes: Structured stage with stable defaults for repeatable runs.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import GRASP_DISTANCE_THRESHOLD_M, GRASP_STABLE_WINDOW, HANDS_PATH, OBJECTS_PATH, PLOTS_PATH, SEG_PATH, ensure_output_dirs


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    hand_tips = np.load(HANDS_PATH / 'hand_tips_3d_smooth.npy')
    object_traj = np.load(OBJECTS_PATH / 'object_3d_smooth.npy')
    hand_center = np.load(HANDS_PATH / 'hand_3d_smooth.npy')

    t_len = min(len(hand_tips), len(object_traj))
    hand_tips = hand_tips[:t_len]
    object_traj = object_traj[:t_len]

    distances = np.full(t_len, np.inf, dtype=np.float32)
    for i in range(t_len):
        tips = hand_tips[i]
        valid = np.isfinite(tips).all(axis=1)
        if np.any(valid):
            d = np.linalg.norm(tips[valid] - object_traj[i], axis=1)
            distances[i] = float(np.min(d))

    finite = np.isfinite(distances)
    if not np.any(finite):
        center_finite = np.isfinite(hand_center[:t_len]).all(axis=1)
        if np.any(center_finite):
            distances = np.linalg.norm(hand_center[:t_len] - object_traj[:t_len], axis=1).astype(np.float32)
            print('Warning: no valid fingertip distances; used hand-center distances as fallback.')
            finite = np.isfinite(distances)
        else:
            raise RuntimeError('No valid fingertip-object distances found.')

    max_finite = np.nanmax(distances[finite])
    distances[~finite] = max_finite

    grasp_idx = None
    for i in range(0, t_len - GRASP_STABLE_WINDOW + 1):
        if np.all(distances[i : i + GRASP_STABLE_WINDOW] <= GRASP_DISTANCE_THRESHOLD_M):
            grasp_idx = i
            break
    if grasp_idx is None:
        grasp_idx = int(np.argmin(distances))

    np.save(SEG_PATH / 'min_fingertip_distance.npy', distances)
    np.save(SEG_PATH / 'grasp_idx.npy', np.array(grasp_idx, dtype=np.int32))

    plt.figure(figsize=(10, 4.5))
    plt.plot(distances, label='min fingertip-object distance')
    plt.axhline(GRASP_DISTANCE_THRESHOLD_M, color='tab:orange', linestyle='--', label='threshold')
    plt.axvline(grasp_idx, color='black', linestyle='--', label='grasp index')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Distance [m]')
    plt.title('Strict 3D Fingertip-Object Distance for Grasp Detection')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'grasp_estimation_strict.png', dpi=300)
    plt.close()

    print('Grasp estimation complete.')
    print(f'Grasp index: {grasp_idx}')


if __name__ == '__main__':
    main()
