# Purpose: Run the full SVO pipeline end-to-end with optional cleanup and model override.
# Notes: Structured stage with stable defaults for repeatable runs.

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

STAGES = [
    '01_extract_hand_trajectory.py',
    '02_smooth_hand_trajectory.py',
    '03_extract_object_trajectory.py',
    '07_smooth_object_trajectory.py',
    '04_estimate_grasp_from_hand.py',
    '05_reconstruct_object_trajectory.py',
    '08_gmm_segmentation.py',
    '09_extract_skills.py',
    '10_learn_dmp.py',
    '11_test_dmp_reproduction.py',
    '12_adapt_dmp.py',
    '13_test_skill_reuse.py',
]


def clean_outputs(base_path: Path) -> None:
    """Remove generated output folders before a new run."""
    dirs = ['hands', 'objects', 'segmentation', 'dmp', 'plots', 'trajectories']
    for d in dirs:
        p = base_path / d
        if p.exists():
            shutil.rmtree(p)


def run_stage(script_dir: Path, script_name: str, env: dict[str, str]) -> None:
    """Execute one pipeline stage script in a subprocess."""
    cmd = [sys.executable, str(script_dir / script_name)]
    print(f'\n=== Running {script_name} ===')
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    """Main entry point for this stage."""
    parser = argparse.ArgumentParser(description='Run full reference-style SVO pipeline.')
    parser.add_argument('--base-path', default='data\Debug', help='Data root containing rgb/depth/intrinsics.')
    parser.add_argument('--object-model', default='', help='Optional object detection model path override.')
    parser.add_argument('--no-clean', action='store_true', help='Keep existing outputs in base path.')
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    if not (base_path / 'rgb').exists() or not (base_path / 'depth').exists():
        raise FileNotFoundError(f'Missing rgb/depth folders in: {base_path}')

    if not args.no_clean:
        clean_outputs(base_path)

    env = os.environ.copy()
    env['VILMA_BASE_PATH'] = str(base_path)
    if args.object_model:
        env['VILMA_OBJECT_MODEL'] = str(Path(args.object_model).resolve())

    script_dir = Path(__file__).resolve().parent
    for stage in STAGES:
        run_stage(script_dir, stage, env)

    print('\nSVO pipeline finished successfully.')
    print(f'Base path: {base_path}')


if __name__ == '__main__':
    main()
