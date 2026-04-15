# Purpose: Run the full BAG pipeline end-to-end with optional cleanup and environment overrides.
# Notes: Structured stage with stable defaults for repeatable runs.

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

STAGES = [
    '00_extract_bag_frames.py',
    '01_extract_hand_trajectory.py',
    '02_smooth_hand_trajectory.py',
    '03_extract_object_trajectory.py',
    '04_smooth_object_trajectory.py',
    '05_estimate_grasp_from_hand.py',
    '06_reconstruct_object_trajectory.py',
    '07_gmm_segmentation.py',
    '08_extract_skill_phases.py',
    '09_learn_dmp.py',
    '10_test_dmp_reproduction.py',
    '11_adapt_dmp.py',
    '12_test_skill_reuse.py',
]


def clean_outputs(base_path: Path) -> None:
    """Remove generated output folders before a new run."""
    dirs = [
        'rgb',
        'depth_raw',
        'depth_meters',
        'hands',
        'objects',
        'segmentation',
        'dmp',
        'plots',
    ]
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
    parser = argparse.ArgumentParser(description='Run full reference-style bag pipeline for one demo.')
    parser.add_argument('--base-path', required=True, help='Output folder for this demo run.')
    parser.add_argument('--bag-file', required=True, help='Path to input RealSense .bag file.')
    parser.add_argument('--no-clean', action='store_true', help='Keep existing outputs in base path.')
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    bag_file = Path(args.bag_file).resolve()
    if not bag_file.exists():
        raise FileNotFoundError(f'Bag file not found: {bag_file}')

    base_path.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        clean_outputs(base_path)

    env = os.environ.copy()
    env['VILMA_BASE_PATH'] = str(base_path)
    env['VILMA_BAG_FILE'] = str(bag_file)

    script_dir = Path(__file__).resolve().parent
    for stage in STAGES:
        run_stage(script_dir, stage, env)

    print('\nPipeline finished successfully.')
    print(f'Base path: {base_path}')


if __name__ == '__main__':
    main()
