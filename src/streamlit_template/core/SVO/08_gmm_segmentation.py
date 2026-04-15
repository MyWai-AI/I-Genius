# Purpose: Segment object motion with GMM(K=3) and estimate release timestamp.
# Notes: Structured stage with stable defaults for repeatable runs.

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
    RELEASE_STABLE_WINDOW_MAX_FRAC,
    SEG_PATH,
    ensure_output_dirs,
)


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    traj = np.load(OBJECTS_PATH / 'object_3d_reconstructed.npy')
    grasp_idx = int(np.load(SEG_PATH / 'grasp_idx.npy'))

    gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type='full', random_state=0)
    labels = gmm.fit_predict(traj)

    tail_window = labels[max(0, len(labels) - 20) :]
    last_cluster = int(np.bincount(tail_window).argmax())

    vel = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    vel = np.concatenate([[vel[0] if len(vel) else 0.0], vel])
    low_vel_threshold = float(np.percentile(vel, 40))

    min_release_idx = min(len(traj) - 1, grasp_idx + RELEASE_MIN_MOVE_FRAMES)
    adaptive_window = int(round(len(traj) * RELEASE_STABLE_WINDOW_MAX_FRAC))
    stable_window = max(2, min(max(RELEASE_STABLE_WINDOW, adaptive_window), 20))

    release_idx = None
    for i in range(min_release_idx, len(traj) - stable_window + 1):
        in_last_cluster = np.all(labels[i : i + stable_window] == last_cluster)
        low_velocity = np.all(vel[i : i + stable_window] <= low_vel_threshold)
        if in_last_cluster and low_velocity:
            release_idx = i
            break

    if release_idx is None:
        candidate = np.where(labels == last_cluster)[0]
        candidate = candidate[candidate >= min_release_idx]
        release_idx = int(candidate[0]) if candidate.size else len(traj) - 1

    np.save(SEG_PATH / 'gmm_labels.npy', labels)
    np.save(SEG_PATH / 'gmm_means.npy', gmm.means_)
    np.save(SEG_PATH / 'gmm_covariances.npy', gmm.covariances_)
    np.save(SEG_PATH / 'gmm_weights.npy', gmm.weights_)
    np.save(SEG_PATH / 'release_idx.npy', np.array(release_idx, dtype=np.int32))
    np.save(SEG_PATH / 'object_velocity.npy', vel.astype(np.float32))

    plt.figure(figsize=(10, 5))
    for axis, axis_name in enumerate(['X', 'Y', 'Z']):
        plt.plot(traj[:, axis], alpha=0.35, label=axis_name)
    plt.scatter(np.arange(len(traj)), traj[:, 2], c=labels, s=9, cmap='viridis', label='GMM cluster')
    plt.axvline(grasp_idx, color='black', linestyle='--', label='grasp')
    plt.axvline(release_idx, color='red', linestyle='--', label='release')
    plt.xlabel('Timestamp [frame]')
    plt.ylabel('Object position [m]')
    plt.title('GMM(K=3) Segmentation for Release Detection')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'gmm_segmentation_time.png', dpi=300)
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for k in range(GMM_COMPONENTS):
        idx = labels == k
        ax.scatter(traj[idx, 0], traj[idx, 1], traj[idx, 2], s=9, label=f'Cluster {k}')
    ax.scatter(*traj[grasp_idx], color='black', s=80, label='grasp')
    ax.scatter(*traj[release_idx], color='red', s=80, label='release')
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
    print(f'Release stable window: {stable_window}')


if __name__ == '__main__':
    main()
