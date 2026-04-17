# Purpose: Apply Kalman smoothing to raw hand center and fingertip trajectories.
# Notes: Structured stage with stable defaults for repeatable runs.

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import HANDS_PATH, KALMAN_AUTO_TUNE, KALMAN_Q, KALMAN_R, PLOTS_PATH, ensure_output_dirs
from _paper_utils import auto_kalman_params, kalman_smooth_3d, save_plot_xyz


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    raw = np.load(HANDS_PATH / 'hand_3d_raw.npy')
    tips_raw = np.load(HANDS_PATH / 'hand_tips_3d_raw.npy')

    q, r = (KALMAN_Q, KALMAN_R)
    if KALMAN_AUTO_TUNE:
        q, r = auto_kalman_params(raw, base_q=KALMAN_Q, base_r=KALMAN_R)

    smooth = kalman_smooth_3d(raw, q=q, r=r)
    tips_smooth = tips_raw.copy()
    for i in range(tips_smooth.shape[1]):
        tips_smooth[:, i, :] = kalman_smooth_3d(tips_raw[:, i, :], q=q, r=r)

    np.save(HANDS_PATH / 'hand_3d_smooth.npy', smooth)
    np.save(HANDS_PATH / 'hand_tips_3d_smooth.npy', tips_smooth)

    save_plot_xyz(PLOTS_PATH / 'hand_trajectory_smooth.png', smooth, 'Hand Trajectory in Camera Frame (Kalman)')

    print('Hand Kalman smoothing complete.')
    print(f'Shape: {smooth.shape}')
    print(f'Kalman params: q={q:.2e}, r={r:.2e}')


if __name__ == '__main__':
    main()
