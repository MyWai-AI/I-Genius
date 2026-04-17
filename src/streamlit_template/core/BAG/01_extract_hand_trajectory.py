# Purpose: Detect hand landmarks, recover 3D hand center/tips, and save raw hand trajectory.
# Notes: Structured stage with stable defaults for repeatable runs.

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import (
    DEPTH_M_PATH,
    HAND_MODEL_PATH,
    HANDS_PATH,
    INTRINSICS_PATH,
    PLOTS_PATH,
    RGB_PATH,
    ensure_output_dirs,
)
from _paper_utils import (
    load_intrinsics,
    median_valid_depth,
    save_plot_xyz,
    stable_hand_bbox_center,
    to_camera_xyz,
)

TIP_IDS = [4, 8, 12, 16, 20]


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    intr = load_intrinsics(INTRINSICS_PATH)

    rgb_files = sorted(p.name for p in RGB_PATH.glob('*.png'))
    depth_files = sorted(p.name for p in DEPTH_M_PATH.glob('*.npy'))
    if len(rgb_files) != len(depth_files):
        raise RuntimeError('RGB and depth counts do not match.')

    base_options = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    trajectory = []
    tips_3d = []
    centers_uv = []

    with vision.HandLandmarker.create_from_options(options) as detector:
        for fname in tqdm(rgb_files, desc='Hand detection'):
            rgb_bgr = cv2.imread(str(RGB_PATH / fname))
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth = np.load(DEPTH_M_PATH / fname.replace('.png', '.npy'))
            h, w, _ = rgb.shape

            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                u, v = stable_hand_bbox_center(lms, width=w, height=h)
                z = median_valid_depth(depth, u=u, v=v, half=2)

                tip_xyz = np.full((len(TIP_IDS), 3), np.nan, dtype=np.float32)
                for j, tip_id in enumerate(TIP_IDS):
                    lm = lms[tip_id]
                    tu = int(round(lm.x * w))
                    tv = int(round(lm.y * h))
                    tz = median_valid_depth(depth, u=tu, v=tv, half=2)
                    if tz is not None:
                        tip_xyz[j] = to_camera_xyz(tu, tv, tz, intr)

                if z is None:
                    valid_tip_xyz = tip_xyz[np.isfinite(tip_xyz).all(axis=1)]
                    if valid_tip_xyz.size > 0:
                        center_xyz = np.median(valid_tip_xyz, axis=0).astype(np.float32)
                    else:
                        center_xyz = None
                else:
                    center_xyz = to_camera_xyz(u, v, z, intr)

                if center_xyz is not None:
                    trajectory.append(center_xyz)
                    tips_3d.append(tip_xyz)
                    centers_uv.append([u, v])
                    continue

            if trajectory:
                trajectory.append(trajectory[-1].copy())
                tips_3d.append(tips_3d[-1].copy())
                centers_uv.append(centers_uv[-1].copy())
            else:
                trajectory.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                tips_3d.append(np.full((len(TIP_IDS), 3), np.nan, dtype=np.float32))
                centers_uv.append([-1, -1])

    traj_np = np.asarray(trajectory, dtype=np.float32)
    tips_np = np.asarray(tips_3d, dtype=np.float32)
    centers_np = np.asarray(centers_uv, dtype=np.int32)

    np.save(HANDS_PATH / 'hand_3d_raw.npy', traj_np)
    np.save(HANDS_PATH / 'hand_tips_3d_raw.npy', tips_np)
    np.save(HANDS_PATH / 'hand_center_uv_raw.npy', centers_np)

    save_plot_xyz(PLOTS_PATH / 'hand_trajectory_raw.png', traj_np, 'Hand Trajectory in Camera Frame (Raw)')

    print('Hand trajectory extraction complete.')
    print(f'Shape: {traj_np.shape}')
    print(f'NaN count: {int(np.isnan(traj_np).sum())}')


if __name__ == '__main__':
    main()
