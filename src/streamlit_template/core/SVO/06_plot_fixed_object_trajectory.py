# Purpose: Plot reconstructed-vs-smoothed object trajectory for visual sanity checking.
# Notes: Structured stage with stable defaults for repeatable runs.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import OBJECTS_PATH, PLOTS_PATH, SEG_PATH, ensure_output_dirs


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    raw = np.load(OBJECTS_PATH / 'object_3d_smooth.npy')
    recon = np.load(OBJECTS_PATH / 'object_3d_reconstructed.npy')
    grasp_idx = int(np.load(SEG_PATH / 'grasp_idx.npy'))
    release_idx = int(np.load(SEG_PATH / 'release_idx.npy')) if (SEG_PATH / 'release_idx.npy').exists() else None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for d, name in enumerate(['X', 'Y', 'Z']):
        axes[d].plot(raw[:, d], label='Smooth object', alpha=0.5)
        axes[d].plot(recon[:, d], label='Reconstructed', linewidth=1.8)
        axes[d].axvline(grasp_idx, color='black', linestyle='--', linewidth=1)
        if release_idx is not None:
            axes[d].axvline(release_idx, color='red', linestyle='--', linewidth=1)
        axes[d].set_title(name)
        axes[d].set_xlabel('Frame')
        axes[d].grid(True)

    axes[0].set_ylabel('Position [m]')
    axes[0].legend()
    plt.suptitle('Object Trajectory Reconstruction Check')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'object_reconstruction_validation.png', dpi=300)
    plt.close()
    print('Saved object reconstruction validation plot.')


if __name__ == '__main__':
    main()
