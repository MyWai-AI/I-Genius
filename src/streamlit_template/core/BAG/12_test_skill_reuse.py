# Purpose: Concatenate adapted skills and export final robot-ready reusable trajectory.
# Notes: Structured stage with stable defaults for repeatable runs.

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import DMP_PATH, PLOTS_PATH, SEG_PATH, ensure_output_dirs


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


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    reach = np.load(DMP_PATH / 'reach_adapted.npy')
    move_path = DMP_PATH / 'move_adapted.npy'
    move = np.load(move_path) if move_path.exists() else np.empty((0, 3), dtype=np.float64)
    grasp_pos = np.load(SEG_PATH / 'grasp_pos.npy')
    release_pos = np.load(SEG_PATH / 'release_pos.npy')
    post_release = np.load(SEG_PATH / 'post_release_pos.npy')

    # Discrete events as duplicated points in time sequence.
    grasp_event = grasp_pos[None, :]
    release_event = release_pos[None, :]

    parts = [reach, grasp_event]
    if len(move) > 0:
        parts.append(move)
    parts.extend([release_event, post_release[None, :]])
    skill = np.vstack(parts)

    np.save(DMP_PATH / 'skill_reuse_traj.npy', skill)
    np.savetxt(DMP_PATH / 'skill_reuse_traj.csv', skill, delimiter=',', header='x,y,z', comments='')
    with open(DMP_PATH / 'skill_reuse_traj.json', 'w', encoding='utf-8') as f:
        json.dump(skill.tolist(), f)

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(skill[:, 0], skill[:, 1], skill[:, 2], label='Skill Reuse Trajectory')
    ax.scatter(*skill[0], marker='+', s=120, color='black', label='Start')
    ax.scatter(*grasp_pos, marker='o', s=80, color='tab:orange', label='Grasp')
    ax.scatter(*release_pos, marker='o', s=80, color='tab:red', label='Release')
    ax.scatter(*skill[-1], marker='*', s=150, color='green', label='Final')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Full Skill Reuse from Reach/Move DMPs')
    _style_3d_axis(ax, np.vstack([skill, grasp_pos[None, :], release_pos[None, :]]))
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'skill_reuse_final.png', dpi=300)
    plt.close()

    print('Skill reuse test complete.')
    print('Saved: skill_reuse_traj.npy/.csv/.json')


if __name__ == '__main__':
    main()
