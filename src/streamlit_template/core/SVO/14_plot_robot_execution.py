# Purpose: Export an execution-style waypoint plot and optional robot-frame trajectory.
# Notes: Keeps the camera-frame skill reuse plot intact and adds a second plot for execution review.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import DMP_PATH, PLOTS_PATH, ensure_output_dirs


def _style_3d_axis(ax, points: np.ndarray) -> None:
    """Apply equal-ish 3D scaling for waypoint review."""
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)


def _load_transform(path: Path) -> np.ndarray:
    """Load a 4x4 homogeneous transform from disk."""
    suffix = path.suffix.lower()
    if suffix == '.npy':
        mat = np.load(path)
    elif suffix == '.npz':
        data = np.load(path)
        for key in ('transform', 'matrix', 'T'):
            if key in data:
                mat = data[key]
                break
        else:
            raise KeyError(f'No transform key found in: {path}')
    elif suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            for key in ('transform', 'matrix', 'T'):
                if key in data:
                    data = data[key]
                    break
        mat = np.asarray(data, dtype=np.float64)
    elif suffix in {'.txt', '.dat'}:
        mat = np.loadtxt(path, dtype=np.float64)
    else:
        raise ValueError(f'Unsupported transform format: {path}')

    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f'Expected 4x4 transform, got {mat.shape} from: {path}')
    return mat


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to Nx3 points."""
    homo = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (transform @ homo.T).T[:, :3]


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    parser = argparse.ArgumentParser(description='Generate an execution-style plot from the final skill reuse trajectory.')
    parser.add_argument('--transform-path', default=os.environ.get('VILMA_EXEC_TRANSFORM_PATH', ''), help='Optional 4x4 transform file (.npy/.npz/.json/.txt).')
    parser.add_argument('--unit-scale', type=float, default=float(os.environ.get('VILMA_EXEC_UNIT_SCALE', '1.0')), help='Scale applied after the optional transform.')
    parser.add_argument('--unit-label', default=os.environ.get('VILMA_EXEC_UNIT_LABEL', 'm'), help='Axis unit label for the execution plot.')
    parser.add_argument('--frame-name', default=os.environ.get('VILMA_EXEC_FRAME_NAME', 'camera'), help='Frame label used in the title and metadata.')
    args = parser.parse_args()

    skill_path = DMP_PATH / 'skill_reuse_traj.npy'
    if not skill_path.exists():
        raise FileNotFoundError(f'Missing final skill reuse trajectory: {skill_path}')

    skill = np.load(skill_path).astype(np.float64)
    exec_points = skill.copy()
    transform_path = Path(args.transform_path).expanduser().resolve() if args.transform_path else None
    if transform_path is not None:
        exec_points = _apply_transform(exec_points, _load_transform(transform_path))

    exec_points = exec_points * args.unit_scale

    np.save(DMP_PATH / 'skill_reuse_execution_traj.npy', exec_points)
    np.savetxt(
        DMP_PATH / 'skill_reuse_execution_traj.csv',
        exec_points,
        delimiter=',',
        header='x,y,z',
        comments='',
    )
    with open(DMP_PATH / 'skill_reuse_execution_traj.json', 'w', encoding='utf-8') as f:
        json.dump(exec_points.tolist(), f)

    meta = {
        'source_skill_path': str(skill_path),
        'transform_path': str(transform_path) if transform_path is not None else '',
        'frame_name': args.frame_name,
        'unit_scale': args.unit_scale,
        'unit_label': args.unit_label,
        'point_count': int(len(exec_points)),
    }
    with open(DMP_PATH / 'skill_reuse_execution_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        exec_points[:, 0],
        exec_points[:, 1],
        exec_points[:, 2],
        color='tab:blue',
        linewidth=1.3,
        marker='o',
        markersize=1.8,
        label='Execution Waypoints',
    )
    ax.scatter(*exec_points[0], marker='+', s=140, color='black', label='Start')
    ax.scatter(*exec_points[-1], marker='*', s=170, color='green', label='Final')
    ax.set_xlabel(f'X [{args.unit_label}]')
    ax.set_ylabel(f'Y [{args.unit_label}]')
    ax.set_zlabel(f'Z [{args.unit_label}]')
    ax.set_title(f'Robot Execution Trajectory ({args.frame_name} frame)')
    _style_3d_axis(ax, exec_points)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'skill_reuse_robot_execution.png', dpi=300)
    plt.close()

    print('Robot execution plot complete.')
    print('Saved: skill_reuse_execution_traj.npy/.csv/.json')
    print('Saved: skill_reuse_robot_execution.png')


if __name__ == '__main__':
    main()
