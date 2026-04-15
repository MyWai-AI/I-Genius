# Purpose: Train reference-style DMPs for Reach/Move and generate phase/forcing/reproduction plots.
# Notes: Structured stage with stable defaults for repeatable runs.

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import (
    DMP_ALPHA_S,
    DMP_ALPHA_Z,
    DMP_BETA_Z,
    DMP_LAMBDA_CANDIDATES,
    DMP_N_BFS,
    DMP_PATH,
    DMP_REG_LAMBDA,
    DMP_TUNE_SMOOTHNESS_WEIGHT,
    PLOTS_PATH,
    TRAJ_PATH,
    ensure_output_dirs,
)
from _paper_dmp import learn_dmp, rollout_dmp, save_model


def plot_phase(path: Path, s: np.ndarray, title: str) -> None:
    """Plot the canonical phase variable over time."""
    plt.figure(figsize=(8, 4))
    plt.plot(s, color='tab:blue')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Phase s')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_forcing(path: Path, forcing_target: np.ndarray, forcing_rollout: np.ndarray, title: str) -> None:
    """Plot target and learned forcing terms for each axis."""
    plt.figure(figsize=(10, 4.5))
    for d, axis in enumerate(['X', 'Y', 'Z']):
        plt.plot(forcing_target[:, d], label=f'{axis} target', alpha=0.5)
        plt.plot(forcing_rollout[:, d], '--', label=f'{axis} learned')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Forcing term')
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_reproduction(path: Path, demo: np.ndarray, reproduction: np.ndarray, title: str) -> None:
    """Plot demonstration and reproduced trajectories in 3D."""
    fig = plt.figure(figsize=(6.8, 5.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(demo[:, 0], demo[:, 1], demo[:, 2], label='Demonstration', color='tab:blue')
    ax.plot(reproduction[:, 0], reproduction[:, 1], reproduction[:, 2], '--', label='Reproduction', color='tab:orange')
    ax.scatter(*demo[0], marker='+', s=120, color='black', label='Start')
    ax.scatter(*demo[-1], marker='*', s=140, color='red', label='Goal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def train_phase(name: str, traj: np.ndarray) -> dict[str, float]:
    """Train DMP for one phase and save diagnostics."""
    candidates = sorted(set([float(DMP_REG_LAMBDA), *[float(x) for x in DMP_LAMBDA_CANDIDATES]]))
    best, best_pack = None, None
    for reg_lambda in candidates:
        model, diag = learn_dmp(
            y_demo=traj,
            n_bfs=DMP_N_BFS,
            alpha_z=DMP_ALPHA_Z,
            beta_z=DMP_BETA_Z,
            alpha_s=DMP_ALPHA_S,
            reg_lambda=reg_lambda,
        )
        rollout = rollout_dmp(model, timesteps=len(traj))
        y_rep = rollout['y']
        rmse = float(np.sqrt(np.mean((traj - y_rep) ** 2)))
        jerk = np.diff(y_rep, n=2, axis=0) if len(y_rep) > 2 else np.zeros((1, 3))
        smoothness = float(np.mean(np.linalg.norm(jerk, axis=1)))
        score = rmse + DMP_TUNE_SMOOTHNESS_WEIGHT * smoothness
        if best is None or score < best['score']:
            best = {'score': score, 'rmse': rmse, 'smoothness': smoothness, 'lambda': reg_lambda}
            best_pack = (model, diag, rollout)

    assert best_pack is not None
    model, diag, rollout = best_pack
    y_rep = rollout['y']
    save_model(str(DMP_PATH / f'{name}_dmp.npz'), model)

    np.save(DMP_PATH / f'{name}_reproduction.npy', y_rep)
    np.save(DMP_PATH / f'{name}_forcing_target.npy', diag['f_target'])
    np.save(DMP_PATH / f'{name}_forcing_rollout.npy', rollout['f'])
    np.save(DMP_PATH / f'{name}_phase.npy', diag['s'])

    plot_phase(PLOTS_PATH / f'{name}_phase_variable.png', diag['s'], f'{name.capitalize()} Canonical Phase Variable')
    plot_forcing(PLOTS_PATH / f'{name}_forcing_term.png', diag['f_target'], rollout['f'], f'{name.capitalize()} DMP Forcing Term')
    plot_reproduction(PLOTS_PATH / f'{name}_dmp_reproduction.png', traj, y_rep, f'{name.capitalize()} DMP Demonstration vs Reproduction')

    return {
        'rmse': float(best['rmse']),
        'lambda': float(best['lambda']),
        'smoothness': float(best['smoothness']),
        'score': float(best['score']),
    }


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    reach = np.load(TRAJ_PATH / 'reach_traj.npy')
    move = np.load(TRAJ_PATH / 'move_traj.npy')

    metrics = {}
    if len(reach) >= 5:
        metrics['reach'] = train_phase('reach', reach)
    else:
        raise RuntimeError('Reach trajectory too short for DMP training.')

    if len(move) >= 5:
        metrics['move'] = train_phase('move', move)
    else:
        print('Warning: move trajectory too short for DMP training; skipping move phase.')

    with open(DMP_PATH / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('DMP learning complete.')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
