# Purpose: Reconstruct carried-object motion after grasp using rigid hand-object offset.
# Notes: Structured stage with stable defaults for repeatable runs.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import HANDS_PATH, OBJECTS_PATH, PLOTS_PATH, SEG_PATH, ensure_output_dirs
from _paper_utils import save_plot_xyz


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    hand = np.load(HANDS_PATH / 'hand_3d_smooth.npy')
    obj = np.load(OBJECTS_PATH / 'object_3d_smooth.npy')
    grasp_idx = int(np.load(SEG_PATH / 'grasp_idx.npy'))

    t_len = min(len(hand), len(obj))
    hand = hand[:t_len]
    obj = obj[:t_len]
    grasp_idx = int(np.clip(grasp_idx, 0, t_len - 1))

    reconstructed = obj.copy()
    offset = obj[grasp_idx] - hand[grasp_idx]
    reconstructed[grasp_idx:] = hand[grasp_idx:] + offset

    np.save(OBJECTS_PATH / 'object_3d_reconstructed.npy', reconstructed)

    save_plot_xyz(
        PLOTS_PATH / 'object_trajectory_reconstructed.png',
        reconstructed,
        'Object Trajectory Reconstructed from Hand Motion After Grasp',
    )

    plt.figure(figsize=(10, 4.5))
    plt.plot(np.linalg.norm(np.diff(reconstructed, axis=0), axis=1), label='|velocity|')
    plt.axvline(max(0, grasp_idx - 1), color='black', linestyle='--', label='grasp-1')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Velocity [m/frame]')
    plt.title('Reconstructed Object Velocity Profile')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'object_reconstructed_velocity.png', dpi=300)
    plt.close()

    print('Object reconstruction complete.')
    print(f'Grasp index: {grasp_idx}')


if __name__ == '__main__':
    main()
