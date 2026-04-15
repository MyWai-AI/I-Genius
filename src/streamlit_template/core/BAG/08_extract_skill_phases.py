# Purpose: Split hand trajectory into Reach/Grasp/Move/Release skill phases and key poses.
# Notes: Structured stage with stable defaults for repeatable runs.

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import (
    HANDS_PATH,
    OBJECTS_PATH,
    PREPOST_DELTA_P,
    RELEASE_MIN_MOVE_FRAMES,
    SEG_PATH,
    ensure_output_dirs,
)


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    hand = np.load(HANDS_PATH / 'hand_3d_smooth.npy')
    obj = np.load(OBJECTS_PATH / 'object_3d_reconstructed.npy')
    grasp_idx = int(np.load(SEG_PATH / 'grasp_idx.npy'))
    release_idx = int(np.load(SEG_PATH / 'release_idx.npy'))

    t_len = min(len(hand), len(obj))
    hand = hand[:t_len]
    obj = obj[:t_len]

    grasp_idx = int(np.clip(grasp_idx, 1, t_len - 2))
    release_idx = int(np.clip(release_idx, grasp_idx + 1, t_len - 1))
    min_release = min(t_len - 1, grasp_idx + RELEASE_MIN_MOVE_FRAMES)
    if release_idx < min_release:
        release_idx = min_release

    # Skill decomposition used in this workflow.
    reach = hand[:grasp_idx]                # [P(t1), ..., P(tg-1)]
    move = hand[grasp_idx + 1 : release_idx]  # [P(tg+1), ..., P(tr-1)]

    grasp_pos = hand[grasp_idx]
    release_pos = hand[release_idx]

    pre_grasp = grasp_pos + np.array([0.0, 0.0, PREPOST_DELTA_P], dtype=np.float32)
    post_grasp = pre_grasp.copy()
    pre_release = release_pos + np.array([0.0, 0.0, PREPOST_DELTA_P], dtype=np.float32)
    post_release = pre_release.copy()

    np.save(SEG_PATH / 'reach_traj.npy', reach)
    np.save(SEG_PATH / 'move_traj.npy', move)
    np.save(SEG_PATH / 'grasp_pos.npy', grasp_pos)
    np.save(SEG_PATH / 'release_pos.npy', release_pos)
    np.save(SEG_PATH / 'pre_grasp_pos.npy', pre_grasp)
    np.save(SEG_PATH / 'post_grasp_pos.npy', post_grasp)
    np.save(SEG_PATH / 'pre_release_pos.npy', pre_release)
    np.save(SEG_PATH / 'post_release_pos.npy', post_release)

    phase_indices = {
        'reach_start': 0,
        'reach_end': grasp_idx - 1,
        'grasp_idx': grasp_idx,
        'move_start': grasp_idx + 1,
        'move_end': release_idx - 1,
        'release_idx': release_idx,
    }
    np.save(SEG_PATH / 'phase_indices.npy', phase_indices)
    with open(SEG_PATH / 'phase_indices.json', 'w', encoding='utf-8') as f:
        json.dump(phase_indices, f, indent=2)

    print('Skill phase extraction complete (Reach, Grasp, Move, Release).')
    print(f'Reach frames: {len(reach)}')
    print(f'Move frames: {len(move)}')
    print(f'Grasp index: {grasp_idx}')
    print(f'Release index: {release_idx}')


if __name__ == '__main__':
    main()
