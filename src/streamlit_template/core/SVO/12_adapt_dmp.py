# Purpose: Adapt learned DMPs to new start/goal poses and generate reference-like adaptation figures.
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
from _paper_config import DMP_ALPHA_S, DMP_ALPHA_Z, DMP_BETA_Z, DMP_N_BFS
from _paper_dmp import learn_dmp, load_model, rollout_dmp


def _style_3d_axis(ax, points: np.ndarray) -> None:
    """Apply consistent 3D axis scaling and view settings."""
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.6 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)


def plot_adaptation(path: Path, demo: np.ndarray, repro: np.ndarray, adapted: np.ndarray, new_start: np.ndarray, new_goal: np.ndarray, title: str) -> None:
    """Plot demonstration, reproduction, and adaptation trajectories in 3D."""
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(6.8, 5.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(demo[:, 0], demo[:, 1], demo[:, 2], label='Demonstration', alpha=0.8)
    ax.plot(repro[:, 0], repro[:, 1], repro[:, 2], '--', label='Reproduction', alpha=0.9)
    ax.plot(adapted[:, 0], adapted[:, 1], adapted[:, 2], label='Adaptation', linewidth=2)
    ax.scatter(*new_start, marker='+', s=120, color='black', label='New Start')
    ax.scatter(*new_goal, marker='*', s=150, color='red', label='New Goal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    points = np.vstack([demo, repro, adapted, new_start[None, :], new_goal[None, :]])
    _style_3d_axis(ax, points)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_reference_lambda(
    name: str,
    demo: np.ndarray,
    new_start: np.ndarray,
    new_goal: np.ndarray,
    reg_lambda: float,
    out_name: str,
) -> None:
    """Create adaptation plots for a fixed regularization value."""
    model, _ = learn_dmp(
        y_demo=demo,
        n_bfs=DMP_N_BFS,
        alpha_z=DMP_ALPHA_Z,
        beta_z=DMP_BETA_Z,
        alpha_s=DMP_ALPHA_S,
        reg_lambda=reg_lambda,
    )
    repro = rollout_dmp(model, timesteps=len(demo))['y']
    adapted = rollout_dmp(model, timesteps=len(demo), y0=new_start, goal=new_goal)['y']
    plot_adaptation(
        PLOTS_PATH / out_name,
        demo=demo,
        repro=repro,
        adapted=adapted,
        new_start=new_start,
        new_goal=new_goal,
        title=f'Learned and Updated DMP for {name.capitalize()} with lambda={reg_lambda}',
    )


def adapt_phase(name: str, demo: np.ndarray, new_start: np.ndarray, new_goal: np.ndarray) -> np.ndarray:
    """Adapt one learned DMP to a new start and goal."""
    model = load_model(str(DMP_PATH / f'{name}_dmp.npz'))
    repro = rollout_dmp(model, timesteps=len(demo))['y']
    adapted = rollout_dmp(model, timesteps=len(demo), y0=new_start, goal=new_goal)['y']

    np.save(DMP_PATH / f'{name}_adapted.npy', adapted)
    plot_adaptation(PLOTS_PATH / f'{name}_adaptation.png', demo, repro, adapted, new_start, new_goal, f'Learned and Updated DMP for {name.capitalize()}')
    return adapted


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    reach_demo = np.load(TRAJ_PATH / 'reach_traj.npy')
    move_demo = np.load(TRAJ_PATH / 'move_traj.npy')

    pre_grasp = np.load(TRAJ_PATH / 'pre_grasp_pos.npy')
    post_grasp = np.load(TRAJ_PATH / 'post_grasp_pos.npy')
    pre_release = np.load(TRAJ_PATH / 'pre_release_pos.npy')

    reach_new_start = reach_demo[0] + np.array([0.03, -0.01, 0.00], dtype=np.float32)
    reach_new_goal = pre_grasp
    move_new_start = post_grasp
    move_new_goal = pre_release

    reach_adapted = adapt_phase('reach', reach_demo, reach_new_start, reach_new_goal)
    plot_reference_lambda('reach', reach_demo, reach_new_start, reach_new_goal, reg_lambda=0.0, out_name='reach_adaptation_lambda_0.png')
    plot_reference_lambda('reach', reach_demo, reach_new_start, reach_new_goal, reg_lambda=0.1, out_name='reach_adaptation_lambda_0p1.png')

    move_model = DMP_PATH / 'move_dmp.npz'
    if move_model.exists() and len(move_demo) >= 5:
        move_adapted = adapt_phase('move', move_demo, move_new_start, move_new_goal)
        plot_reference_lambda('move', move_demo, move_new_start, move_new_goal, reg_lambda=0.1, out_name='move_adaptation_lambda_0p1.png')
    else:
        move_adapted = np.empty((0, 3), dtype=np.float64)
        np.save(DMP_PATH / 'move_adapted.npy', move_adapted)
        print('Warning: move DMP unavailable; move adaptation skipped.')

    meta = {
        'reach_new_start': reach_new_start.tolist(),
        'reach_new_goal': reach_new_goal.tolist(),
        'move_new_start': move_new_start.tolist(),
        'move_new_goal': move_new_goal.tolist(),
        'reach_len': int(len(reach_adapted)),
        'move_len': int(len(move_adapted)),
    }
    with open(DMP_PATH / 'adaptation_config.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('DMP adaptation complete for reach and move.')


if __name__ == '__main__':
    main()
