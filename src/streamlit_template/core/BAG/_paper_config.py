# Purpose: Centralized BAG pipeline constants, paths, and tuning knobs.
# Notes: Structured stage with stable defaults for repeatable runs.

from __future__ import annotations

import os
from pathlib import Path

# Dataset paths
BASE_PATH = Path(os.environ.get('VILMA_BASE_PATH', '/home/vvijaykumar/vilma-agent/data/BAG Data'))
BAG_FILE = Path(os.environ.get('VILMA_BAG_FILE', str(BASE_PATH / 'demo.bag')))

# Models
HAND_MODEL_PATH = Path('/home/vvijaykumar/vilma-agent/src/streamlit_template/models/hand_landmarker.task')
OBJECT_MODEL_PATH = Path('/home/vvijaykumar/vilma-agent/src/streamlit_template/models/blueball.pt')

# I/O folders
RGB_PATH = BASE_PATH / 'rgb'
DEPTH_RAW_PATH = BASE_PATH / 'depth_raw'
DEPTH_M_PATH = BASE_PATH / 'depth_meters'
HANDS_PATH = BASE_PATH / 'hands'
OBJECTS_PATH = BASE_PATH / 'objects'
SEG_PATH = BASE_PATH / 'segmentation'
DMP_PATH = BASE_PATH / 'dmp'
PLOTS_PATH = BASE_PATH / 'plots'

INTRINSICS_PATH = BASE_PATH / 'camera_intrinsics.npy'

# Object labels are a list for multi-object extension.
OBJECT_LABELS = ['blueball']

# Frame extraction
TARGET_FPS = 10
SOURCE_FPS = 30
SUBSAMPLE = max(1, SOURCE_FPS // TARGET_FPS)

# Segmentation and filtering constants
GMM_COMPONENTS = 3
KALMAN_Q = 1e-4
KALMAN_R = 2e-3
KALMAN_AUTO_TUNE = True
GRASP_DISTANCE_THRESHOLD_M = 0.05
GRASP_STABLE_WINDOW = 5
RELEASE_MIN_MOVE_FRAMES = 15
RELEASE_STABLE_WINDOW = 5
RELEASE_STABLE_WINDOW_MAX_FRAC = 0.1

# DMP constants used in these experiments
DMP_N_BFS = 20
DMP_ALPHA_Z = 25.0
DMP_BETA_Z = DMP_ALPHA_Z / 4.0
DMP_ALPHA_S = 4.0
DMP_REG_LAMBDA = 0.1
DMP_LAMBDA_CANDIDATES = (0.02, 0.05, 0.1, 0.2, 0.35)
DMP_TUNE_SMOOTHNESS_WEIGHT = 0.15
PREPOST_DELTA_P = 0.02


def ensure_output_dirs() -> None:
    """Create required output directories if missing."""
    for p in (
        BASE_PATH,
        RGB_PATH,
        DEPTH_RAW_PATH,
        DEPTH_M_PATH,
        HANDS_PATH,
        OBJECTS_PATH,
        SEG_PATH,
        DMP_PATH,
        PLOTS_PATH,
    ):
        p.mkdir(parents=True, exist_ok=True)
