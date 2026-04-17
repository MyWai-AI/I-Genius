# Purpose: Validate learned DMP reproduction quality and report axis-wise RMSE.
# Notes: Structured stage with stable defaults for repeatable runs.

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import DMP_PATH, PLOTS_PATH, TRAJ_PATH, ensure_output_dirs
from _paper_dmp import load_model, rollout_dmp


def evaluate_phase(name: str, demo: np.ndarray) -> dict[str, float]:
    """Run DMP rollout for one phase and compute reproduction metrics."""
    model = load_model(str(DMP_PATH / f'{name}_dmp.npz'))
    out = rollout_dmp(model, timesteps=len(demo))
    y = out['y']

    err = demo - y
    rmse = np.sqrt(np.mean(err**2, axis=0))

    plt.figure(figsize=(10, 4.5))
    for d, axis_name in enumerate(['X', 'Y', 'Z']):
        plt.plot(demo[:, d], label=f'{axis_name} demo', alpha=0.5)
        plt.plot(y[:, d], '--', label=f'{axis_name} repro')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Position [m]')
    plt.title(f'{name.capitalize()} DMP Reproduction (Axis-wise)')
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / f'{name}_dmp_reproduction_axes.png', dpi=300)
    plt.close()

    return {'rmse_x': float(rmse[0]), 'rmse_y': float(rmse[1]), 'rmse_z': float(rmse[2])}


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    reach = np.load(TRAJ_PATH / 'reach_traj.npy')
    move = np.load(TRAJ_PATH / 'move_traj.npy')

    report = {'reach': evaluate_phase('reach', reach)}
    move_model = DMP_PATH / 'move_dmp.npz'
    if move_model.exists() and len(move) > 0:
        report['move'] = evaluate_phase('move', move)
    else:
        report['move'] = {'skipped': True}

    with open(DMP_PATH / 'reproduction_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('DMP reproduction validation complete.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
