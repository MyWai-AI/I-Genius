# Purpose: Segment object motion with GMM(K=3) and estimate release timestamp.
#
# Paper alignment (Algorithm 1, Section 2.4.3):
#   "The first timestamp in the [last/post-placing] state after placing
#    indicating the release of the hand."
#   - GMM(K=3) fitted on full object trajectory
#   - Last cluster = the cluster whose centroid has the highest position
#     similarity to finalPosition = mean(objTraj[end:])
#   - releaseIndex = first frame assigned to the last cluster
#
# Robustness additions (beyond paper):
#   1. Last-cluster identification via finalPosition proximity (more reliable
#      than tail-window voting when the object returns to a different location).
#   2. Minimum-move guard: release cannot be before grasp + MIN_MOVE_FRAMES.
#   3. Short stability window to skip isolated mis-classified frames.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import (
    GMM_COMPONENTS,
    OBJECTS_PATH,
    PLOTS_PATH,
    RELEASE_MIN_MOVE_FRAMES,
    RELEASE_STABLE_WINDOW,
    SEG_PATH,
    ensure_output_dirs,
)

# Fraction of frames at the END of the trajectory used to estimate final position.
_FINAL_FRAC = 0.10
_FINAL_MIN_FRAMES = 5


def _estimate_final_position(traj: np.ndarray) -> np.ndarray:
    """Return object's resting position after demonstration ends.

    Paper: finalPosition = mean(objTraj[end:])
    """
    n = len(traj)
    n_final = max(_FINAL_MIN_FRAMES, int(n * _FINAL_FRAC))
    n_final = min(n_final, n)
    segment = traj[max(0, n - n_final):]
    finite_mask = np.isfinite(segment).all(axis=1)
    if not np.any(finite_mask):
        return traj[-1]
    return segment[finite_mask].mean(axis=0)


def _identify_last_cluster(gmm: GaussianMixture, final_position: np.ndarray) -> int:
    """Identify which GMM cluster corresponds to the post-placing static state.

    Paper: last cluster = the state after placing (object at rest at final pos).
    We find the cluster whose mean is closest to the estimated final position.
    """
    means = gmm.means_  # (K, D)
    dists = np.linalg.norm(means - final_position, axis=1)
    return int(np.argmin(dists))


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    traj = np.load(OBJECTS_PATH / 'object_3d_reconstructed.npy')
    grasp_idx = int(np.load(SEG_PATH / 'grasp_idx.npy'))

    n = len(traj)

    # ── Step 1: estimate final (post-placing) position ──
    final_position = _estimate_final_position(traj)
    print(f'Object final position estimate: {final_position}')

    # ── Step 2: fit GMM(K=3) on full object trajectory ──
    # Paper: gmm.fit(objTraj), labels = gmm.predict(objTraj)
    gmm = GaussianMixture(
        n_components=GMM_COMPONENTS,
        covariance_type='full',
        random_state=0,
        n_init=5,          # multiple restarts for stable convergence
    )
    labels = gmm.fit_predict(traj)

    # ── Step 3: identify the "last cluster" (post-placing static state) ──
    # Primary: closest cluster mean to finalPosition (paper-aligned)
    last_cluster = _identify_last_cluster(gmm, final_position)

    # Sanity check: also compute tail-window vote as cross-reference
    tail_window = labels[max(0, n - 20):]
    tail_vote_cluster = int(np.bincount(tail_window).argmax())
    if last_cluster != tail_vote_cluster:
        print(
            f'Note: final-position cluster ({last_cluster}) differs from tail-vote '
            f'cluster ({tail_vote_cluster}). Using final-position cluster (paper-aligned).'
        )

    # ── Step 4: compute velocity for low-velocity guard ──
    vel = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    vel = np.concatenate([[vel[0] if len(vel) else 0.0], vel])
    low_vel_threshold = float(np.percentile(vel, 40))

    # ── Step 5: find release index ──
    # Require RELEASE_STABLE_WINDOW consecutive frames in last cluster AND
    # low velocity (object has stopped moving), starting no earlier than
    # grasp + RELEASE_MIN_MOVE_FRAMES.
    min_release_idx = min(n - 1, grasp_idx + RELEASE_MIN_MOVE_FRAMES)
    stable_window = max(2, RELEASE_STABLE_WINDOW)

    release_idx = None
    for i in range(min_release_idx, n - stable_window + 1):
        if (np.all(labels[i : i + stable_window] == last_cluster) and
                np.all(vel[i : i + stable_window] <= low_vel_threshold)):
            release_idx = i
            break

    if release_idx is None:
        # Relax: drop velocity requirement, cluster membership only
        for i in range(min_release_idx, n - stable_window + 1):
            if np.all(labels[i : i + stable_window] == last_cluster):
                release_idx = i
                break

    if release_idx is None:
        # Final fallback: first frame in last cluster after min guard
        candidates = np.where(labels == last_cluster)[0]
        candidates = candidates[candidates >= min_release_idx]
        release_idx = int(candidates[0]) if candidates.size else n - 1
        print(
            f'Warning: stable release window not found; '
            f'using first last-cluster frame after min guard: {release_idx}.'
        )

    # ── Save ──
    np.save(SEG_PATH / 'gmm_labels.npy', labels)
    np.save(SEG_PATH / 'gmm_means.npy', gmm.means_)
    np.save(SEG_PATH / 'gmm_covariances.npy', gmm.covariances_)
    np.save(SEG_PATH / 'gmm_weights.npy', gmm.weights_)
    np.save(SEG_PATH / 'release_idx.npy', np.array(release_idx, dtype=np.int32))
    np.save(SEG_PATH / 'last_cluster.npy', np.array(last_cluster, dtype=np.int32))
    np.save(SEG_PATH / 'object_velocity.npy', vel.astype(np.float32))

    # ── Plot: time-series ──
    plt.figure(figsize=(10, 5))
    for axis, axis_name in enumerate(['X', 'Y', 'Z']):
        plt.plot(traj[:, axis], alpha=0.35, label=axis_name)
    plt.scatter(
        np.arange(n), traj[:, 2],
        c=labels, s=9, cmap='viridis', label='GMM cluster'
    )
    plt.axvline(grasp_idx, color='black', linestyle='--', label=f'grasp ({grasp_idx})')
    plt.axvline(release_idx, color='red', linestyle='--', label=f'release ({release_idx})')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Object position [m]')
    plt.title(f'GMM(K={GMM_COMPONENTS}) Segmentation — last cluster: {last_cluster}')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'gmm_segmentation_time.png', dpi=300)
    plt.close()

    # ── Plot: 3D ──
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for k in range(GMM_COMPONENTS):
        idx = labels == k
        label = f'Cluster {k}' + (' (last/place)' if k == last_cluster else '')
        ax.scatter(traj[idx, 0], traj[idx, 1], traj[idx, 2], s=9, label=label)
    ax.scatter(*traj[grasp_idx], color='black', s=80, zorder=5, label='grasp')
    ax.scatter(*traj[release_idx], color='red', s=80, zorder=5, label='release')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Object Trajectory with GMM Segments')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'gmm_segmentation_3d.png', dpi=300)
    plt.close()

    print('GMM segmentation complete.')
    print(f'Grasp index: {grasp_idx}')
    print(f'Release index: {release_idx}')
    print(f'Last (post-placing) cluster: {last_cluster}')


if __name__ == '__main__':
    main()
