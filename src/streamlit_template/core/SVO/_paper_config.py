# Purpose: Centralized SVO pipeline constants, paths, and tuning knobs.
# Notes: Structured stage with stable defaults for repeatable runs.

from __future__ import annotations

import os
from pathlib import Path

BASE_PATH = Path(os.environ.get('VILMA_BASE_PATH', 'data\Debug'))

HAND_MODEL_PATH = Path('data/Common/ai_model/hand/hand_landmarker.task')
_DEFAULT_OBJ_MODEL = 'data/Common/ai_model/object/best-obb.pt'
_FALLBACK_OBJ_MODEL = '/home/vvijaykumar/vilma-agent/src/streamlit_template/models/blueball.pt'
OBJECT_MODEL_PATH = Path(
    os.environ.get(
        'VILMA_OBJECT_MODEL',
        _DEFAULT_OBJ_MODEL if Path(_DEFAULT_OBJ_MODEL).exists() else _FALLBACK_OBJ_MODEL,
    )
)

RGB_PATH = BASE_PATH / 'rgb'
DEPTH_PATH = BASE_PATH / 'depth'
HANDS_PATH = BASE_PATH / 'hands'
OBJECTS_PATH = BASE_PATH / 'objects'
SEG_PATH = BASE_PATH / 'segmentation'
DMP_PATH = BASE_PATH / 'dmp'
PLOTS_PATH = BASE_PATH / 'plots'
TRAJ_PATH = BASE_PATH / 'trajectories'

INTRINSICS_NPY_PATH = BASE_PATH / 'camera_intrinsics.npy'
INTRINSICS_NPZ_PATH = BASE_PATH / 'camera_intrinsics.npz'

OBJECT_LABELS = ['shaft']

GMM_COMPONENTS = 3
KALMAN_Q = 1e-4
KALMAN_R = 2e-3
KALMAN_AUTO_TUNE = True
GRASP_DISTANCE_THRESHOLD_M = 0.05
GRASP_STABLE_WINDOW = 5
RELEASE_MIN_MOVE_FRAMES = 15
RELEASE_STABLE_WINDOW = 5
RELEASE_STABLE_WINDOW_MAX_FRAC = 0.1

DMP_N_BFS = 20
DMP_ALPHA_Z = 25.0
DMP_BETA_Z = DMP_ALPHA_Z / 4.0
DMP_ALPHA_S = 4.0
DMP_REG_LAMBDA = 0.1
DMP_LAMBDA_CANDIDATES = (0.02, 0.05, 0.1, 0.2, 0.35)
DMP_TUNE_SMOOTHNESS_WEIGHT = 0.15
PREPOST_DELTA_P = 0.02


def intrinsics_path() -> Path:
    """Return the available intrinsics file path."""
    if INTRINSICS_NPY_PATH.exists():
        return INTRINSICS_NPY_PATH
    return INTRINSICS_NPZ_PATH


def ensure_output_dirs() -> None:
    """Create required output directories if missing."""
    for p in (
        BASE_PATH,
        HANDS_PATH,
        OBJECTS_PATH,
        SEG_PATH,
        DMP_PATH,
        PLOTS_PATH,
        TRAJ_PATH,
    ):
        p.mkdir(parents=True, exist_ok=True)
