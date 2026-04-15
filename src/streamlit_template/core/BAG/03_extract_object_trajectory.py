# Purpose: Detect object per frame and reconstruct raw 3D object trajectory in camera frame.
# Notes: Structured stage with stable defaults for repeatable runs.

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import (
    DEPTH_M_PATH,
    INTRINSICS_PATH,
    OBJECT_LABELS,
    OBJECT_MODEL_PATH,
    OBJECTS_PATH,
    PLOTS_PATH,
    RGB_PATH,
    ensure_output_dirs,
)
from _paper_utils import load_intrinsics, median_valid_depth, save_plot_xyz, to_camera_xyz


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    intr = load_intrinsics(INTRINSICS_PATH)
    model = YOLO(str(OBJECT_MODEL_PATH))

    rgb_files = sorted(p.name for p in RGB_PATH.glob('*.png'))

    traj = []
    uv_centers = []

    for fname in tqdm(rgb_files, desc='Object detection'):
        rgb = cv2.imread(str(RGB_PATH / fname))
        depth = np.load(DEPTH_M_PATH / fname.replace('.png', '.npy'))

        res = model(rgb, verbose=False)[0]
        if res.obb is not None and len(res.obb) > 0:
            confs = res.obb.conf.detach().cpu().numpy()
            idx = int(np.argmax(confs))
            u, v = res.obb.xywhr[idx].detach().cpu().numpy()[:2]
            u_i, v_i = int(round(float(u))), int(round(float(v)))
            z = median_valid_depth(depth, u=u_i, v=v_i, half=2)
            if z is not None:
                traj.append(to_camera_xyz(u_i, v_i, z, intr))
                uv_centers.append([u_i, v_i])
                continue

        if traj:
            traj.append(traj[-1].copy())
            uv_centers.append(uv_centers[-1].copy())
        else:
            traj.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            uv_centers.append([-1, -1])

    traj_np = np.asarray(traj, dtype=np.float32)
    uv_np = np.asarray(uv_centers, dtype=np.int32)

    np.save(OBJECTS_PATH / 'object_3d_raw.npy', traj_np)
    np.save(OBJECTS_PATH / 'object_center_uv_raw.npy', uv_np)
    with open(OBJECTS_PATH / 'object_labels.json', 'w', encoding='utf-8') as f:
        json.dump(OBJECT_LABELS, f)

    save_plot_xyz(PLOTS_PATH / 'object_trajectory_raw.png', traj_np, 'Object Trajectory in Camera Frame (Raw)')

    print('Object trajectory extraction complete.')
    print(f'Shape: {traj_np.shape}')


if __name__ == '__main__':
    main()
