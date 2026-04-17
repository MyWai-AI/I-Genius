# Purpose: Apply Kalman smoothing to raw object trajectory.
# Notes: Structured stage with stable defaults for repeatable runs.

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import KALMAN_AUTO_TUNE, KALMAN_Q, KALMAN_R, OBJECTS_PATH, PLOTS_PATH, ensure_output_dirs
from _paper_utils import auto_kalman_params, kalman_smooth_3d, save_plot_xyz


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()
    raw = np.load(OBJECTS_PATH / 'object_3d_raw.npy')

    q, r = (KALMAN_Q, KALMAN_R)
    if KALMAN_AUTO_TUNE:
        q, r = auto_kalman_params(raw, base_q=KALMAN_Q, base_r=KALMAN_R)

    smooth = kalman_smooth_3d(raw, q=q, r=r)

    np.save(OBJECTS_PATH / 'object_3d_smooth.npy', smooth)
    save_plot_xyz(PLOTS_PATH / 'object_trajectory_smooth.png', smooth, 'Object Trajectory in Camera Frame (Kalman)')

    print('Object Kalman smoothing complete.')
    print(f'Shape: {smooth.shape}')
    print(f'Kalman params: q={q:.2e}, r={r:.2e}')


if __name__ == '__main__':
    main()
