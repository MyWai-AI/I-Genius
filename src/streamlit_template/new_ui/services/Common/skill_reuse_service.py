"""
Skill Reuse Service — Compute offset-adjusted trajectories for new target objects.

Given:
  - The original skill_reuse_traj (from the pipeline DMP step)
  - The 3D position of the originally-tracked object (anchor)
  - The 3D position of a new target object (from detection + depth)
  - A user-specified release/goal position

Produces:
  - A new skill_reuse_<label>.csv with the trajectory shifted by the 3D offset
  - Joint trajectory via IK for push-to-robot
"""
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

EVENT_HOVER_MM = 60.0
EVENT_RELEASE_ABOVE_PICK_MM = 40.0

# If the demonstrated pick-to-release robot-Z delta is within this range we
# treat pick and place as co-planar (same table surface, delta = depth noise)
# and snap the release contact to the exact same robot Z as the pick contact.
# For genuine height differences (shelf, ramp, stack) the delta will be larger
# than this threshold and is preserved.  Units: millimetres.
FLAT_TABLE_SNAP_THRESHOLD_MM = 15.0


@dataclass
class DetectedObject:
    """A single object detection with 3D position."""
    label: str
    confidence: float
    bbox_xyxy: np.ndarray        # [x1, y1, x2, y2]
    center_uv: Tuple[int, int]   # pixel center
    xyz: Optional[np.ndarray]    # 3D camera-frame position (may be None if no depth)


def load_intrinsics_dict(path: str) -> dict:
    """Load intrinsics from .npy dict format used throughout the pipeline."""
    raw = np.load(path, allow_pickle=True)
    if hasattr(raw, "item"):
        try:
            return raw.item()
        except ValueError:
            return {k: raw[k] for k in raw.files}
    return dict(raw)


def _pixel_to_xyz(u: int, v: int, depth_m: np.ndarray, intr: dict) -> Optional[np.ndarray]:
    """Back-project pixel + depth to 3D camera coordinates."""
    h, w = depth_m.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    half = 2
    x0, x1 = max(0, u - half), min(w, u + half + 1)
    y0, y1 = max(0, v - half), min(h, v + half + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    z = float(np.median(valid))
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def pixel_to_xyz(u: int, v: int, depth_m: np.ndarray, intr: dict) -> Optional[np.ndarray]:
    """Public wrapper for converting image pixel coordinates into 3D camera coordinates."""
    return _pixel_to_xyz(u, v, depth_m, intr)


def _next_available_stem(output_dir: Path, base_stem: str) -> str:
    """Return a non-colliding stem by appending an incrementing suffix when needed."""
    if not (output_dir / f"{base_stem}.npy").exists():
        return base_stem

    idx = 1
    while True:
        candidate = f"{base_stem}_{idx}"
        if not (output_dir / f"{candidate}.npy").exists():
            return candidate
        idx += 1


def _rotation_between_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix that maps src direction onto dst direction."""
    src_norm = float(np.linalg.norm(src))
    dst_norm = float(np.linalg.norm(dst))
    if src_norm < 1e-9 or dst_norm < 1e-9:
        return np.eye(3, dtype=np.float64)

    a = np.asarray(src, dtype=np.float64) / src_norm
    b = np.asarray(dst, dtype=np.float64) / dst_norm
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if np.linalg.norm(cross) < 1e-9:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = axis - a * np.dot(axis, a)
        v /= max(float(np.linalg.norm(v)), 1e-9)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(v, v)

    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def load_cam2base_transform(search_dir: Path) -> Optional[np.ndarray]:
    """Load a camera-mm → robot-mm calibration matrix from a directory.

    Searches *search_dir* for a calibration JSON file, then falls back to the
    global calibration store at ``data/Common/calibration/``.

    The JSON file must contain one of:
      - ``"T_cam2base"``: a 3×4 or 4×4 homogeneous affine matrix
      - ``"affine_camera_to_robot"``: same format

    The matrix maps a point expressed in **camera millimetres** (X right,
    Y down, Z into the scene for a standard depth camera) into **robot-base
    millimetres** (robot manufacturer convention).

    To generate this file, perform a hand-eye calibration between your depth
    camera and robot base and export the result as JSON.  Place the file as
    ``cam2base_calibration.json`` in the session's skill-reuse output folder,
    or upload it once via the Local page AI Tools panel to use it globally
    (saved to ``data/Common/calibration/cam2base_calibration.json``).

    Returns:
        A (4, 4) float64 numpy array when found and valid, otherwise None.
        When None is returned all calibration-dependent steps (contact-Z
        correction, hover clearance derivation) fall back to camera-space
        heuristics.

    Example JSON format::

        {
            "T_cam2base": [
                [r00, r01, r02, tx],
                [r10, r11, r12, ty],
                [r20, r21, r22, tz],
                [0,   0,   0,   1 ]
            ]
        }
    """
    return _load_cam2base_transform(search_dir)


# Global calibration directory — users can upload once and it applies to all sessions.
_GLOBAL_CALIB_DIR = Path("data/Common/calibration")


def _load_cam2base_transform(search_dir: Path) -> Optional[np.ndarray]:
    """Load an optional camera-mm to robot-mm calibration matrix.

    Searches *search_dir* first, then the global calibration directory
    (``data/Common/calibration/``) as a fallback so a single upload covers
    all sessions.
    """
    search_dirs = [Path(search_dir), _GLOBAL_CALIB_DIR]
    for directory in search_dirs:
        for name in ("colleague_cam2base_calibration.json", "cam2base_calibration.json", "calibration.json"):
            path = directory / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                matrix = payload.get("T_cam2base") or payload.get("affine_camera_to_robot")
                if matrix is None:
                    continue
                arr = np.asarray(matrix, dtype=np.float64)
                if arr.shape == (3, 4):
                    arr = np.vstack([arr, np.array([0.0, 0.0, 0.0, 1.0])])
                if arr.shape == (4, 4):
                    logger.debug("Loaded cam2base calibration from %s", path)
                    return arr
            except Exception:
                logger.exception("Failed to load camera-to-base calibration from %s", path)
    return None


def _robot_z_mm(point_mm: np.ndarray, transform: Optional[np.ndarray]) -> Optional[float]:
    if transform is None:
        return None
    pt = np.array([float(point_mm[0]), float(point_mm[1]), float(point_mm[2]), 1.0], dtype=np.float64)
    return float((transform @ pt)[2])


def _point_with_robot_z_mm(point_mm: np.ndarray, desired_robot_z_mm: float, transform: Optional[np.ndarray]) -> np.ndarray:
    """Adjust only source camera Z so the transformed robot Z reaches a target height."""
    pt = np.asarray(point_mm, dtype=np.float64).copy()
    if transform is None or abs(float(transform[2, 2])) < 1e-9:
        # Fallback for the current camera convention: larger camera Z maps lower in robot Z.
        current = float(pt[2])
        pt[2] = current - float(desired_robot_z_mm)
        return pt
    pt[2] = (float(desired_robot_z_mm) - float(transform[2, 0]) * pt[0] - float(transform[2, 1]) * pt[1] - float(transform[2, 3])) / float(transform[2, 2])
    return pt


def _apply_robot_z_offset(
    cam_point_mm: np.ndarray,
    robot_z_offset_mm: float,
    transform: Optional[np.ndarray],
) -> np.ndarray:
    """Shift a camera-space point by *robot_z_offset_mm* along the robot Z axis.

    Uses the full inverse transform so robot X and Y stay at the detected object
    position — only robot Z moves.  This is the correct way to add a "contact
    approach depth" offset without introducing spurious X/Y errors.

    Without calibration the convention is: camera Z increases downward (overhead
    camera), so a negative robot-Z offset (go lower) corresponds to a positive
    camera-Z increment.

    Args:
        cam_point_mm     : Camera-space point in millimetres, shape (3,).
        robot_z_offset_mm: How much to shift in robot Z (negative = go lower).
        transform        : 4×4 homogeneous cam-to-robot transform, or None.

    Returns
    -------
    Camera-space point (mm) such that when the robot applies *transform*, it
    arrives exactly *robot_z_offset_mm* below the original robot position.
    """
    if robot_z_offset_mm == 0.0:
        return np.asarray(cam_point_mm, dtype=np.float64).copy()

    pt = np.asarray(cam_point_mm, dtype=np.float64).copy()

    if transform is None:
        # No calibration: camera +Z = physically lower (overhead convention).
        # Robot going lower (negative offset) → camera Z increases.
        pt[2] -= float(robot_z_offset_mm)
        return pt

    try:
        pt_h = np.array([float(pt[0]), float(pt[1]), float(pt[2]), 1.0], dtype=np.float64)
        robot_pt = (transform @ pt_h).copy()  # [rx, ry, rz, 1]
        robot_pt[2] += float(robot_z_offset_mm)
        inv = np.linalg.inv(transform)
        cam_shifted = inv @ robot_pt
        return cam_shifted[:3].copy()
    except np.linalg.LinAlgError:
        # Singular transform — fall back to camera-space sign convention
        pt[2] -= float(robot_z_offset_mm)
        return pt


def _derive_hover_clearances_mm(
    traj_mm: np.ndarray,
    grasp_idx: int,
    release_idx: int,
    transform: np.ndarray,
    default_hover_mm: float = EVENT_HOVER_MM,
    max_hover_mm: float = 500.0,
) -> tuple[float, float]:
    """Derive pick and place hover clearances from a demonstrated trajectory.

    Scans the approach window (frames 0 → grasp_idx) and the pre-release window
    (frames grasp_idx → release_idx) of the demonstrated trajectory to find the
    maximum robot-Z height above each contact point.  Those heights are returned
    as the pick and place hover clearances respectively.

    Requires a valid calibration transform.  Falls back to default_hover_mm if
    the windows are empty or a clearance cannot be computed.

    Args:
        traj_mm: Demonstrated trajectory in camera millimetres, shape (N, 3+).
        grasp_idx: Frame index of the grasp/pick contact.
        release_idx: Frame index of the release/place contact.
        transform: 4x4 camera-mm → robot-mm affine matrix.
        default_hover_mm: Fallback clearance when derivation fails.
        max_hover_mm: Upper cap to guard against outlier trajectory heights.

    Returns:
        (pick_hover_mm, place_hover_mm) in robot Z millimetres.
    """
    traj_mm = np.asarray(traj_mm, dtype=np.float64)
    n = len(traj_mm)
    grasp_idx = int(max(0, min(grasp_idx, n - 1)))
    release_idx = int(max(grasp_idx, min(release_idx, n - 1)))

    pick_contact_z = _robot_z_mm(traj_mm[grasp_idx, :3], transform)
    place_contact_z = _robot_z_mm(traj_mm[release_idx, :3], transform)
    if pick_contact_z is None or place_contact_z is None:
        return default_hover_mm, default_hover_mm

    # Approach window: frames before the grasp
    approach_zs = [_robot_z_mm(traj_mm[i, :3], transform) for i in range(grasp_idx)]
    approach_zs = [z for z in approach_zs if z is not None]
    if approach_zs:
        pick_hover = float(np.clip(max(approach_zs) - pick_contact_z, 0.0, max_hover_mm))
    else:
        pick_hover = default_hover_mm

    # Pre-release window: frames from grasp up to (but not including) release
    pre_release_zs = [_robot_z_mm(traj_mm[i, :3], transform) for i in range(grasp_idx, release_idx)]
    pre_release_zs = [z for z in pre_release_zs if z is not None]
    if pre_release_zs:
        place_hover = float(np.clip(max(pre_release_zs) - place_contact_z, 0.0, max_hover_mm))
    else:
        place_hover = default_hover_mm

    return pick_hover, place_hover


def build_skill_reuse_sparse_event_rows_mm(
    xyz_mm: np.ndarray,
    grasp_idx: int,
    release_idx: int,
    *,
    calibration_dir: Optional[Path] = None,
    hover_mm: float = EVENT_HOVER_MM,
    release_above_pick_mm: float = EVENT_RELEASE_ABOVE_PICK_MM,
    original_traj_mm: Optional[np.ndarray] = None,
    grasp_quaternion: Optional[Tuple[float, float, float, float]] = None,
    contact_z_offset_mm: float = 0.0,
) -> list[list]:
    """Build sparse robot-ready event waypoints from a dense trajectory in camera millimeters.

    The dense trajectory remains unchanged.  Sparse events are synthesized so the
    final pick/place approach uses the same XY at above/contact points.

    When a calibration matrix and the original demonstrated trajectory are available,
    hover clearances for pick_above and place_above are derived from the demonstrated
    motion (max robot-Z height above each contact in the approach / pre-release window).
    This makes the sparse events general across tasks rather than relying on hardcoded
    height thresholds.

    When calibration is unavailable the function falls back to a camera-space heuristic
    using the hover_mm and release_above_pick_mm constants.

    Args:
        xyz_mm: Reused trajectory in camera millimetres, shape (N, 3+).
        grasp_idx: Frame index of the grasp/pick contact.
        release_idx: Frame index of the release/place contact.
        calibration_dir: Directory that may contain a cam2base calibration file.
        hover_mm: Fallback hover clearance (mm) when derivation is not possible.
        release_above_pick_mm: Fallback minimum place-above-pick (mm) used only in
            the no-calibration camera-space path.
        original_traj_mm: Original demonstrated trajectory in camera millimetres,
            shape (N, 3+).  When provided together with a calibration matrix, hover
            clearances are derived from the demonstration instead of using hover_mm.
        grasp_quaternion: Optional (qx, qy, qz, qw) representing end-effector
            orientation at the pick contact (rotation around the approach axis so the
            finger gap is perpendicular to the object long axis).  When provided,
            pick_above and pick rows gain four extra columns; all other rows carry
            empty strings in those columns.  Intended for top-down parallel-jaw setups.
    """
    xyz_mm = np.asarray(xyz_mm, dtype=np.float64)
    if xyz_mm.ndim != 2 or xyz_mm.shape[1] < 3 or len(xyz_mm) == 0:
        return []

    n = len(xyz_mm)
    grasp_idx = int(max(0, min(grasp_idx, n - 1)))
    release_idx = int(max(grasp_idx, min(release_idx, n - 1)))
    transform = _load_cam2base_transform(calibration_dir) if calibration_dir else None

    start = xyz_mm[0, :3].copy()
    pick = xyz_mm[grasp_idx, :3].copy()
    release = xyz_mm[release_idx, :3].copy()
    end = xyz_mm[-1, :3].copy()

    # Contact approach offset — shifts pick and release along robot Z (negative = lower).
    # Uses the full inverse transform so robot X,Y stay at the detected object position
    # and only robot Z moves, preventing spurious X/Y errors when camera and robot axes
    # are not perfectly aligned.
    if contact_z_offset_mm != 0.0:
        pick    = _apply_robot_z_offset(pick,    contact_z_offset_mm, transform)
        release = _apply_robot_z_offset(release, contact_z_offset_mm, transform)

    pick_robot_z = _robot_z_mm(pick, transform)
    release_robot_z = _robot_z_mm(release, transform)
    if pick_robot_z is not None and release_robot_z is not None:
        # Derive hover clearances from the original demonstration when available;
        # fall back to the fixed hover_mm constant otherwise.
        if original_traj_mm is not None and transform is not None:
            pick_hover, place_hover = _derive_hover_clearances_mm(
                np.asarray(original_traj_mm, dtype=np.float64),
                grasp_idx,
                release_idx,
                transform,
                default_hover_mm=hover_mm,
            )
        else:
            pick_hover = place_hover = float(hover_mm)

        # Hover points: shift the contact point UP by pick_hover / place_hover in robot Z.
        # _apply_robot_z_offset uses the full inverse transform so robot X,Y stay exactly
        # at the detected object position — only robot Z changes.  This avoids the X,Y drift
        # that _point_with_robot_z_mm caused when the cam2base transform has off-diagonal terms
        # (changing camera-Z to hit a target robot-Z also shifts robot X and Y).
        pick_above = _apply_robot_z_offset(pick, +pick_hover, transform)
        release_above = _apply_robot_z_offset(release, +place_hover, transform)
    else:
        # Fallback: no calibration matrix available.
        # Camera-space convention: larger Z = further from camera = lower physically.
        #
        # Snap rule (20 mm noise threshold):
        #   |pick_z - release_z| < 20 mm  →  flat-table noise  →  snap release to pick Z
        #   |pick_z - release_z| ≥ 20 mm  →  real height difference  →  keep release as-is
        #
        # This replaces the old constant "release[2] = pick[2] - 40 mm" that fired
        # whenever release_z >= pick_z (including the flat-table case where they are
        # equal), creating a spurious 40 mm height difference in every trajectory
        # generated without a calibration file.
        _cam_delta_mm = abs(pick[2] - release[2])
        if _cam_delta_mm < 20.0:
            release[2] = pick[2]  # flat-table noise → snap
        # else: genuine height difference (≥ 20 mm) → keep release[2] unchanged
        pick_above = pick.copy()
        pick_above[2] -= float(hover_mm)
        release_above = release.copy()
        release_above[2] -= float(hover_mm)

    q_cols: Optional[Tuple[float, float, float, float]] = grasp_quaternion  # shorthand
    q_empty = ("", "", "", "") if q_cols is not None else ()

    def fmt(
        point: np.ndarray, event: str, source_idx: int, label: str,
        orientation: Optional[Tuple[float, float, float, float]] = None,
    ) -> list:
        row = [f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}", event, int(source_idx), label]
        if q_cols is not None:
            if orientation is not None:
                row += [f"{orientation[0]:.6f}", f"{orientation[1]:.6f}",
                        f"{orientation[2]:.6f}", f"{orientation[3]:.6f}"]
            else:
                row += list(q_empty)
        return row

    return [
        fmt(start,        "",      0,                              "start"),
        fmt(pick_above,   "",      max(0, grasp_idx - 1),         "pre_pick|pick_above",   q_cols),
        fmt(pick,         "close", grasp_idx,                     "pick|grasp",            q_cols),
        fmt(release_above,"",      max(grasp_idx, release_idx-1), "pre_release|place_above"),
        fmt(release,      "open",  release_idx,                   "place|release"),
        fmt(end,          "",      n - 1,                         "end"),
    ]


def estimate_grasp_orientation(
    det: "DetectedObject",
    depth_m: Optional[np.ndarray],
    intrinsics: dict,
    cam2base: Optional[np.ndarray] = None,
) -> dict:
    """Estimate end-effector orientation for grasping the detected object.

    Strategy
    --------
    1. Extract the 3-D point cloud of the object from depth + bbox.
    2. Run PCA on the point cloud to find the principal (long) axis.
    3. Project the axis to the robot horizontal plane and compute the wrist
       rotation angle so the finger gap is *perpendicular* to the object axis.

    Falls back to the 2-D bounding-box elongation axis when depth is absent.

    Returns
    -------
    dict with keys:
        method              : "pca_3d" | "bbox_2d"
        principal_axis_cam  : np.ndarray (3,) unit vector in camera frame
        principal_axis_robot: np.ndarray (3,) | None  (needs cam2base)
        angle_in_plane_deg  : float  – object axis angle in robot XY plane
        grip_angle_deg      : float  – recommended wrist angle (⊥ to object axis)
        quaternion_xyzw     : [qx, qy, qz, qw]  – rotation around approach Z
        axis_uv             : ((u1,v1),(u2,v2))  – axis line for visualization
        finger_gap_uv       : ((u1,v1),(u2,v2))  – perpendicular line for visualization
        center_uv           : (u, v)
    """
    bbox = getattr(det, "bbox_xyxy", None)
    if bbox is None:
        return {}

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cx_img = (x1 + x2) / 2.0
    cy_img = (y1 + y2) / 2.0
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)

    # ── 3-D PCA from depth ────────────────────────────────────────────────────
    principal_axis_cam: Optional[np.ndarray] = None
    method = "bbox_2d"

    if depth_m is not None and intrinsics:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        dh, dw = depth_m.shape
        ys = np.arange(max(0, y1), min(dh, y2))
        xs = np.arange(max(0, x1), min(dw, x2))
        if len(ys) > 2 and len(xs) > 2:
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            d = depth_m[yy, xx].astype(np.float64)
            valid = (d > 0.05) & (d < 5.0)
            if valid.sum() > 10:
                dv = d[valid]
                xv = (xx[valid].astype(np.float64) - cx) / fx * dv
                yv = (yy[valid].astype(np.float64) - cy) / fy * dv
                pts = np.stack([xv, yv, dv], axis=-1)
                centered = pts - pts.mean(axis=0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                ax = Vt[0].copy()
                # Canonical sign: point toward positive image-X
                if ax[0] < 0:
                    ax = -ax
                principal_axis_cam = ax
                method = "pca_3d"

    # ── Fallback: bbox elongation ─────────────────────────────────────────────
    if principal_axis_cam is None:
        principal_axis_cam = np.array([1.0, 0.0, 0.0]) if bbox_w >= bbox_h else np.array([0.0, 1.0, 0.0])

    # ── Project to robot frame ────────────────────────────────────────────────
    principal_axis_robot: Optional[np.ndarray] = None
    angle_in_plane_deg = 0.0

    if cam2base is not None:
        T = np.asarray(cam2base, dtype=np.float64)
        if T.shape[0] >= 3 and T.shape[1] >= 3:
            R = T[:3, :3]
            ar = R @ principal_axis_cam
            ar /= np.linalg.norm(ar) + 1e-12
            principal_axis_robot = ar
            axis_xy = ar[:2]
            norm_xy = np.linalg.norm(axis_xy)
            if norm_xy > 0.1:
                axis_xy = axis_xy / norm_xy
                angle_in_plane_deg = float(np.degrees(np.arctan2(axis_xy[1], axis_xy[0])))
    else:
        # Use 2-D image angle as proxy (reasonable for near-top-down cameras)
        angle_in_plane_deg = float(np.degrees(
            np.arctan2(principal_axis_cam[1], principal_axis_cam[0])
        ))

    # Fingers should be perpendicular to the object long axis
    grip_angle_deg = (angle_in_plane_deg + 90.0) % 180.0
    half = np.radians(grip_angle_deg) / 2.0
    qx, qy = 0.0, 0.0
    qz = float(np.sin(half))
    qw = float(np.cos(half))

    # ── Visualization vectors in image space ──────────────────────────────────
    # Project principal axis onto image (divide by Z for approximate 2-D direction)
    az = principal_axis_cam[2] if abs(principal_axis_cam[2]) > 1e-6 else 1.0
    fx_vis = float(intrinsics.get("fx", 500.0))
    fy_vis = float(intrinsics.get("fy", 500.0))
    u_dir = principal_axis_cam[0] / az * fx_vis
    v_dir = principal_axis_cam[1] / az * fy_vis
    uv_norm = (u_dir ** 2 + v_dir ** 2) ** 0.5 + 1e-9
    u_dir /= uv_norm
    v_dir /= uv_norm
    vis_len = max(bbox_w, bbox_h) * 0.55
    axis_uv = (
        (int(cx_img - vis_len * u_dir), int(cy_img - vis_len * v_dir)),
        (int(cx_img + vis_len * u_dir), int(cy_img + vis_len * v_dir)),
    )
    # Perpendicular = finger gap direction
    fg_len = vis_len * 0.35
    finger_gap_uv = (
        (int(cx_img + fg_len * v_dir), int(cy_img - fg_len * u_dir)),
        (int(cx_img - fg_len * v_dir), int(cy_img + fg_len * u_dir)),
    )

    return {
        "method": method,
        "principal_axis_cam": principal_axis_cam,
        "principal_axis_robot": principal_axis_robot,
        "angle_in_plane_deg": angle_in_plane_deg,
        "grip_angle_deg": grip_angle_deg,
        "quaternion_xyzw": [qx, qy, qz, qw],
        "axis_uv": axis_uv,
        "finger_gap_uv": finger_gap_uv,
        "center_uv": (int(cx_img), int(cy_img)),
    }


def detect_objects_on_frame(
    frame_path: str,
    depth_path: Optional[str],
    intrinsics: dict,
    model_path: str = "yolov8x.pt",
    conf: float = 0.05,
    max_area_pct: float = 15.0,
    min_area_pct: float = 0.0001,
) -> List[DetectedObject]:
    """Run YOLO on a single frame and produce DetectedObject list with 3D positions."""
    from ultralytics import YOLO

    bgr = cv2.imread(frame_path)
    if bgr is None:
        return []

    depth_m = None
    if depth_path and Path(depth_path).exists():
        depth_m = np.load(depth_path)

    model = YOLO(model_path)
    results = model.predict(bgr, verbose=False, conf=conf, imgsz=1280, max_det=100, agnostic_nms=True)
    if not results:
        return []

    fr_h, fr_w = bgr.shape[:2]
    r0 = results[0]
    names = r0.names or {}
    detections: List[DetectedObject] = []

    has_obb = r0.obb is not None and len(r0.obb) > 0
    has_boxes = r0.boxes is not None and len(r0.boxes) > 0

    raw_boxes = []  # list of (xyxy, conf, label)
    if has_obb:
        for obox in r0.obb:
            c = float(obox.conf[0])
            cls_id = int(obox.cls[0]) if obox.cls is not None else None
            label = names.get(cls_id) if cls_id is not None else None
            xywhr = obox.xywhr[0].cpu().numpy()
            rect = ((xywhr[0], xywhr[1]), (xywhr[2], xywhr[3]), float(np.degrees(xywhr[4])))
            pts = cv2.boxPoints(rect)
            x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
            x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
            raw_boxes.append((np.array([x1, y1, x2, y2], dtype=np.float32), c, label))
    elif has_boxes:
        for b, bc, bcls in zip(r0.boxes.xyxy, r0.boxes.conf, r0.boxes.cls):
            bn = b.cpu().numpy().astype(np.float32)
            label = names.get(int(bcls)) if names else None
            raw_boxes.append((bn, float(bc), label))

    for xyxy, conf_val, label in raw_boxes:
        bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        area = bw * bh
        if area > (fr_h * fr_w * (max_area_pct / 100.0)):
            continue
        if area < (fr_h * fr_w * (min_area_pct / 100.0)):
            continue

        cu, cv_coord = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
        xyz = None
        if depth_m is not None:
            xyz = _pixel_to_xyz(cu, cv_coord, depth_m, intrinsics)

        detections.append(DetectedObject(
            label=label or "unknown",
            confidence=conf_val,
            bbox_xyxy=xyxy,
            center_uv=(cu, cv_coord),
            xyz=xyz,
        ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def compute_skill_reuse_for_target(
    original_skill_reuse_path: str,
    anchor_xyz: np.ndarray,
    target_xyz: np.ndarray,
    release_xyz: Optional[np.ndarray],
    seg_dir: str,
    output_dir: str,
    target_label: str = "object",
    align_pick_z_to_release: bool = False,
) -> dict:
    """Compute a new skill_reuse trajectory shifted by the 3D offset from anchor to target.

    The anchor is the object that was tracked in the original pipeline.
    The target is the new object the user wants to apply the skill to.

    If release_xyz is provided, the release position in the trajectory is also
    adjusted to match the user's desired goal position.

    Returns dict with keys: csv_path, npy_path, trajectory, offset, grasp_idx, release_idx
    """
    seg = Path(seg_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    original_traj = np.load(original_skill_reuse_path)
    offset_3d = target_xyz - anchor_xyz

    # Load calibration transform once — used for release Z correction and hover derivation.
    _calib_transform = _load_cam2base_transform(out)
    _demonstrated_delta_mm: Optional[float] = None
    _release_z_corrected = False

    # Load segmentation info to know grasp/release indices
    reach_path = seg / "reach_traj.npy"
    move_path = seg / "move_traj.npy"
    reach_len = len(np.load(str(reach_path))) if reach_path.exists() else 0
    move_len = len(np.load(str(move_path))) if move_path.exists() else 0
    grasp_idx = reach_len
    release_idx = reach_len + 1 + move_len

    # Keep the demonstrated start/home position fixed.  The object offset should
    # move the grasp target, not the robot's initial approach pose.
    new_traj = original_traj.copy()

    original_start = np.asarray(original_traj[0], dtype=np.float64)
    original_grasp = np.asarray(original_traj[grasp_idx], dtype=np.float64)
    shifted_grasp = original_grasp + np.asarray(offset_3d, dtype=np.float64)
    new_traj[grasp_idx] = shifted_grasp

    if grasp_idx > 0:
        original_reach_vec = original_grasp - original_start
        new_reach_vec = shifted_grasp - original_start
        original_reach_len = float(np.linalg.norm(original_reach_vec))
        new_reach_len = float(np.linalg.norm(new_reach_vec))
        reach_residual_scale = new_reach_len / original_reach_len if original_reach_len > 1e-9 else 1.0

        for idx in range(1, grasp_idx):
            alpha = idx / grasp_idx
            original_line = (1.0 - alpha) * original_start + alpha * original_grasp
            demonstrated_residual = (original_traj[idx] - original_line) * reach_residual_scale
            new_line = (1.0 - alpha) * original_start + alpha * shifted_grasp
            new_traj[idx] = new_line + demonstrated_residual

    if release_xyz is None and grasp_idx + 1 < len(new_traj):
        new_traj[grasp_idx + 1:] = original_traj[grasp_idx + 1:] + offset_3d

    retarget_strategy = "fixed_start_offset_after_grasp"

    # If user specified a release position, retarget the move phase while
    # preserving the demonstrated dense motion.  The release endpoint is
    # corrected by moving the grasp-to-release baseline, but every intermediate
    # point keeps its original residual from that baseline.  This avoids turning
    # one-shot motions into a straight line and avoids rotating/scaling the
    # demonstrated shape when the new pick/release vector is very different.
    if release_xyz is not None and release_idx < len(new_traj):
        if align_pick_z_to_release and 0 <= grasp_idx < len(new_traj):
            # Preserve the original pick-vs-release height relation. Live object-center
            # depth can be noisy, while the clicked release/table point is often the
            # better reference for contact height.
            original_pick_release_dz = float(original_traj[grasp_idx, 2] - original_traj[release_idx, 2])
            desired_pick_z = float(release_xyz[2]) + original_pick_release_dz
            pick_z_delta = desired_pick_z - float(new_traj[grasp_idx, 2])

            if grasp_idx > 0:
                for i in range(grasp_idx + 1):
                    alpha = i / grasp_idx
                    new_traj[i, 2] += pick_z_delta * alpha
            else:
                new_traj[grasp_idx, 2] = desired_pick_z
            new_traj[grasp_idx, 2] = desired_pick_z

        original_release = np.asarray(original_traj[release_idx], dtype=np.float64)
        new_grasp = np.asarray(new_traj[grasp_idx], dtype=np.float64)
        new_release = np.asarray(release_xyz, dtype=np.float64)

        # Correct the release Z to preserve the demonstrated pick-to-release robot-space
        # height relationship.  User-selected release positions often have noisy depth,
        # which can place the release contact lower than the pick in robot space even
        # when the real surface is at the same or higher elevation.  We keep the user's
        # XY and adjust only camera Z so the transformed robot Z matches the demonstrated
        # pick-to-release height delta.  This correction is applied before the move-phase
        # and post-release retargeting loops so all phases use the corrected endpoint.
        if _calib_transform is not None:
            _orig_pick_rz = _robot_z_mm(
                np.asarray(original_traj[grasp_idx, :3], dtype=np.float64) * 1000.0,
                _calib_transform,
            )
            _orig_rel_rz = _robot_z_mm(
                np.asarray(original_traj[release_idx, :3], dtype=np.float64) * 1000.0,
                _calib_transform,
            )
            _new_pick_rz = _robot_z_mm(
                np.asarray(new_traj[grasp_idx, :3], dtype=np.float64) * 1000.0,
                _calib_transform,
            )
            if _orig_pick_rz is not None and _orig_rel_rz is not None and _new_pick_rz is not None:
                _demonstrated_delta_mm = _orig_rel_rz - _orig_pick_rz

                # Flat-table snap: if the demonstrated delta is within depth-noise
                # range, pick and place were on the same surface.  Zero the delta so
                # the noise in the original demonstration is not baked into every
                # retargeted trajectory.  For genuine height differences (shelf,
                # ramp, stacking) the delta will exceed the threshold and is kept.
                if abs(_demonstrated_delta_mm) <= FLAT_TABLE_SNAP_THRESHOLD_MM:
                    _effective_delta_mm = 0.0
                    logger.debug(
                        "Flat-table snap: demonstrated delta %.1f mm ≤ threshold %.1f mm → "
                        "release contact Z forced to match pick contact Z.",
                        _demonstrated_delta_mm, FLAT_TABLE_SNAP_THRESHOLD_MM,
                    )
                else:
                    _effective_delta_mm = _demonstrated_delta_mm
                    logger.debug(
                        "Height-difference preserved: demonstrated delta %.1f mm > threshold.",
                        _demonstrated_delta_mm,
                    )

                _desired_new_rel_rz = _new_pick_rz + _effective_delta_mm
                # Use _apply_robot_z_offset (full inverse transform) so robot X,Y stay at
                # the user-selected release position. _point_with_robot_z_mm only adjusted
                # camera-Z, which leaked into robot X,Y through the transform off-diagonals.
                _current_new_rel_rz = _robot_z_mm(
                    np.asarray(new_release[:3], dtype=np.float64) * 1000.0, _calib_transform
                )
                _rel_rz_delta = _desired_new_rel_rz - float(_current_new_rel_rz or 0.0)
                _corrected_rel_mm = _apply_robot_z_offset(
                    np.asarray(new_release[:3], dtype=np.float64) * 1000.0,
                    _rel_rz_delta,
                    _calib_transform,
                )
                new_release = new_release.copy()
                new_release[:3] = _corrected_rel_mm / 1000.0
                _release_z_corrected = True
        else:
            # No calibration available — apply flat-table snap in camera space.
            # Camera Z increases away from the lens (lower in the physical world),
            # so a demonstrated pick-to-release delta near zero means flat table.
            # Convert to mm for threshold comparison consistent with the calibrated path.
            _orig_pick_cam_z_mm = float(original_traj[grasp_idx, 2]) * 1000.0
            _orig_rel_cam_z_mm = float(original_traj[release_idx, 2]) * 1000.0
            _demonstrated_delta_mm = _orig_rel_cam_z_mm - _orig_pick_cam_z_mm

            if abs(_demonstrated_delta_mm) <= FLAT_TABLE_SNAP_THRESHOLD_MM:
                _effective_cam_delta_m = 0.0
                logger.debug(
                    "Camera-space flat-table snap (no calibration): demonstrated delta "
                    "%.1f mm ≤ threshold %.1f mm → release Z snapped to pick Z.",
                    _demonstrated_delta_mm, FLAT_TABLE_SNAP_THRESHOLD_MM,
                )
            else:
                _effective_cam_delta_m = _demonstrated_delta_mm / 1000.0
                logger.debug(
                    "Camera-space height-difference preserved (no calibration): "
                    "demonstrated delta %.1f mm > threshold.",
                    _demonstrated_delta_mm,
                )

            new_release = new_release.copy()
            new_release[2] = float(new_traj[grasp_idx, 2]) + _effective_cam_delta_m
            _release_z_corrected = True

        original_vec = original_release - original_grasp
        new_vec = new_release - new_grasp
        original_len = float(np.linalg.norm(original_vec))
        new_len = float(np.linalg.norm(new_vec))
        residual_scale = new_len / original_len if original_len > 1e-9 else 1.0

        span = max(1, release_idx - grasp_idx)
        for idx in range(grasp_idx + 1, release_idx):
            alpha = (idx - grasp_idx) / span
            original_line = (1.0 - alpha) * original_grasp + alpha * original_release
            demonstrated_residual = (original_traj[idx] - original_line) * residual_scale
            new_line = (1.0 - alpha) * new_grasp + alpha * new_release
            new_traj[idx] = new_line + demonstrated_residual

        if release_idx + 1 < len(new_traj):
            original_end = np.asarray(original_traj[-1], dtype=np.float64)
            post_span = max(1, (len(new_traj) - 1) - release_idx)
            for idx in range(release_idx + 1, len(new_traj)):
                beta = (idx - release_idx) / post_span
                original_line = (1.0 - beta) * original_release + beta * original_end
                demonstrated_residual = original_traj[idx] - original_line
                new_line = (1.0 - beta) * new_release + beta * original_end
                new_traj[idx] = new_line + demonstrated_residual

        new_traj[release_idx] = new_release
        retarget_strategy = "scaled_dense_residual_endpoint_retarget"

    # Save
    safe_label = "".join(c for c in target_label if c.isalnum() or c in ("_", "-")).lower()
    safe_label = safe_label or "object"
    base_stem = f"skill_reuse_{safe_label}"
    final_stem = _next_available_stem(out, base_stem)

    npy_path = out / f"{final_stem}.npy"
    csv_path = out / f"{final_stem}.csv"
    json_path = out / f"{final_stem}.json"

    np.save(str(npy_path), new_traj)
    np.savetxt(str(csv_path), new_traj, delimiter=",", header="x,y,z", comments="")
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(new_traj.tolist(), f)

    labels_by_idx: dict[int, list[str]] = {}

    def _add_label(idx: int, label: str) -> None:
        idx = int(max(0, min(idx, len(new_traj) - 1)))
        labels_by_idx.setdefault(idx, [])
        if label not in labels_by_idx[idx]:
            labels_by_idx[idx].append(label)

    if len(new_traj) > 0:
        _add_label(0, "start")
        _add_label(max(0, grasp_idx - 1), "pre_pick")
        _add_label(max(0, grasp_idx - 1), "pick_above")
        _add_label(grasp_idx, "pick")
        _add_label(grasp_idx, "grasp")
        _add_label(max(grasp_idx, release_idx - 1), "pre_release")
        _add_label(max(grasp_idx, release_idx - 1), "place_above")
        _add_label(release_idx, "place")
        _add_label(release_idx, "release")
        _add_label(len(new_traj) - 1, "end")

    xyz_mm = np.asarray(new_traj[:, :3], dtype=np.float64) * 1000.0
    raw_mm_path = out / f"{final_stem}_raw_mm.csv"
    all_labeled_mm_path = out / f"{final_stem}_all_points_labeled_mm.csv"
    events_mm_path = out / f"{final_stem}_events_mm.csv"

    def _event_for_label(label: str) -> str:
        if "pick|grasp" in label:
            return "close"
        if "place|release" in label:
            return "open"
        return ""

    with raw_mm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["x_mm", "y_mm", "z_mm"])
        for point in xyz_mm:
            writer.writerow([f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"])

    event_headers = ["x_mm", "y_mm", "z_mm", "event", "source_index", "label"]
    original_traj_mm = original_traj[:, :3] * 1000.0
    event_rows = build_skill_reuse_sparse_event_rows_mm(
        xyz_mm,
        grasp_idx,
        release_idx,
        calibration_dir=out,
        original_traj_mm=original_traj_mm,
    )

    # Compute hover clearances for metadata (re-derives using same logic as event builder)
    _transform = _load_cam2base_transform(out)
    if _transform is not None:
        _pick_hover, _place_hover = _derive_hover_clearances_mm(
            np.asarray(original_traj_mm, dtype=np.float64),
            grasp_idx,
            release_idx,
            _transform,
        )
        _hover_source = "demonstration"
    else:
        _pick_hover = _place_hover = EVENT_HOVER_MM
        _hover_source = "default_fallback_no_calibration"
    with all_labeled_mm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writerow(event_headers)
        for idx, point in enumerate(xyz_mm):
            label = "|".join(labels_by_idx.get(idx, []))
            event = _event_for_label(label)
            row = [f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}", event, idx, label]
            writer.writerow(row)

    with events_mm_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writerow(event_headers)
        writer.writerows(event_rows)

    # Save metadata
    meta = {
        "trajectory_name": final_stem,
        "target_label": target_label,
        "anchor_xyz": anchor_xyz.tolist(),
        "target_xyz": target_xyz.tolist(),
        "offset_3d": offset_3d.tolist(),
        "release_xyz": release_xyz.tolist() if release_xyz is not None else None,
        "align_pick_z_to_release": bool(align_pick_z_to_release),
        "retarget_strategy": retarget_strategy,
        "retarget_original_pick_release_len_m": float(original_len) if release_xyz is not None and release_idx < len(new_traj) else None,
        "retarget_new_pick_release_len_m": float(new_len) if release_xyz is not None and release_idx < len(new_traj) else None,
        "retarget_residual_scale": float(residual_scale) if release_xyz is not None and release_idx < len(new_traj) else None,
        "grasp_idx": int(grasp_idx),
        "release_idx": int(release_idx),
        "num_points": len(new_traj),
        "raw_mm_csv": str(raw_mm_path),
        "all_points_labeled_mm_csv": str(all_labeled_mm_path),
        "events_mm_csv": str(events_mm_path),
        "hover_source": _hover_source,
        "pick_hover_mm": float(_pick_hover),
        "place_hover_mm": float(_place_hover),
        "release_z_corrected": _release_z_corrected,
        "demonstrated_pick_release_delta_mm": float(_demonstrated_delta_mm) if _demonstrated_delta_mm is not None else None,
        "effective_pick_release_delta_mm": (
            0.0 if (
                _demonstrated_delta_mm is not None
                and abs(_demonstrated_delta_mm) <= FLAT_TABLE_SNAP_THRESHOLD_MM
            ) else float(_demonstrated_delta_mm) if _demonstrated_delta_mm is not None else None
        ),
        "flat_table_snap_applied": (
            abs(float(_demonstrated_delta_mm)) <= FLAT_TABLE_SNAP_THRESHOLD_MM
            if _demonstrated_delta_mm is not None else None
        ),
    }
    with open(str(out / f"{final_stem}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "csv_path": str(csv_path),
        "npy_path": str(npy_path),
        "trajectory": new_traj,
        "offset": offset_3d,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "meta": meta,
    }


def detect_colored_release_zone(
    frame_bgr: np.ndarray,
    color_hex: str,
    h_tolerance: int = 15,
    sv_tolerance: int = 60,
    min_area_px: int = 500,
) -> dict:
    """Detect a colored region (e.g. a box on the table) as the release zone.

    Uses HSV segmentation with hue wrap-around handling for red tones.

    Args:
        frame_bgr   : Camera frame in BGR.
        color_hex   : Target color as '#RRGGBB' from ``st.color_picker``.
        h_tolerance : Hue tolerance (0-89). Default 15.
        sv_tolerance: Saturation / value tolerance (0-127). Default 60.
        min_area_px : Minimum contour area (px²) to be considered valid. Default 500.

    Returns
    -------
    dict with keys:
        found       : bool
        center_uv   : (u, v) centroid of the largest matching region
        bbox_xyxy   : (x1, y1, x2, y2) bounding rect of that region
        area_px     : float area of contour
        contour     : raw OpenCV contour (for drawing)
        mask        : 8-bit binary mask (for visualisation)
    On failure returns ``{"found": False}``.
    """
    try:
        hex_clean = color_hex.lstrip("#")
        r, g, b = int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)
    except (ValueError, IndexError):
        return {"found": False}

    target_bgr = np.uint8([[[b, g, r]]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    hc, sc, vc = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])

    s_lo, s_hi = max(0, sc - sv_tolerance), min(255, sc + sv_tolerance)
    v_lo, v_hi = max(0, vc - sv_tolerance), min(255, vc + sv_tolerance)
    h_lo, h_hi = hc - h_tolerance, hc + h_tolerance

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if h_lo < 0:
        m1 = cv2.inRange(hsv, np.array([0,        s_lo, v_lo]), np.array([h_hi,       s_hi, v_hi]))
        m2 = cv2.inRange(hsv, np.array([h_lo+180, s_lo, v_lo]), np.array([179,        s_hi, v_hi]))
        mask = cv2.bitwise_or(m1, m2)
    elif h_hi > 179:
        m1 = cv2.inRange(hsv, np.array([h_lo,     s_lo, v_lo]), np.array([179,        s_hi, v_hi]))
        m2 = cv2.inRange(hsv, np.array([0,        s_lo, v_lo]), np.array([h_hi-180,   s_hi, v_hi]))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]), np.array([h_hi, s_hi, v_hi]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False}
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < min_area_px:
        return {"found": False}
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return {"found": False}
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, bw, bh = cv2.boundingRect(largest)
    return {
        "found": True,
        "center_uv": (cx, cy),
        "bbox_xyxy": (x, y, x + bw, y + bh),
        "area_px": area,
        "contour": largest,
        "mask": mask,
    }


def build_exec_waypoints_with_downsampling(
    traj_mm: np.ndarray,
    event_source_indices: List[int],
    step: int = 5,
    grasp_quaternion: Optional[Tuple[float, float, float, float]] = None,
) -> list:
    """Build an execution waypoint list: event points + downsampled intermediate points.

    Between each consecutive pair of event source indices, every ``step``-th dense
    trajectory frame is kept (endpoints always included).  The result can be written
    to a ``*_exec_mm.csv`` alongside the standard sparse events file.

    Args:
        traj_mm             : Dense trajectory in camera mm, shape (N, 3+).
        event_source_indices: Frame indices for each event row (events CSV column 4).
        step                : Subsample stride between events (1 = keep all points).
        grasp_quaternion    : Optional (qx, qy, qz, qw) applied to pick_above + pick rows.

    Returns
    -------
    list of rows ``[x_mm, y_mm, z_mm, is_event, qx, qy, qz, qw]`` where
    ``is_event`` is "1" for event points and "0" for intermediate waypoints.
    Quaternion columns are present only when ``grasp_quaternion`` is not None.
    """
    traj_mm = np.asarray(traj_mm, dtype=np.float64)
    n = len(traj_mm)
    sorted_events = sorted(set(max(0, min(int(i), n - 1)) for i in event_source_indices))
    event_set = set(sorted_events)

    indices: List[int] = []
    for si in range(len(sorted_events) - 1):
        s_idx = sorted_events[si]
        e_idx = sorted_events[si + 1]
        indices.append(s_idx)
        for k in range(s_idx + step, e_idx, step):
            indices.append(min(k, n - 1))
    if sorted_events:
        indices.append(sorted_events[-1])

    seen: set = set()
    unique: List[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)

    # pick_above = event index 1, pick = event index 2 (0-based in sorted_events)
    pick_above_src = sorted_events[1] if len(sorted_events) > 1 else -1
    pick_src       = sorted_events[2] if len(sorted_events) > 2 else -1
    q_cols: Optional[Tuple[str, str, str, str]] = None
    if grasp_quaternion is not None:
        q_cols = tuple(f"{v:.6f}" for v in grasp_quaternion)  # type: ignore[assignment]

    rows = []
    for i in unique:
        pt = traj_mm[i, :3]
        row: list = [f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}",
                     "1" if i in event_set else "0"]
        if q_cols is not None:
            row += list(q_cols) if i in (pick_above_src, pick_src) else ["", "", "", ""]
        rows.append(row)
    return rows


def build_exec_waypoints_by_count(
    traj_mm: np.ndarray,
    event_source_indices: List[int],
    target_count: int,
    grasp_quaternion: Optional[Tuple[float, float, float, float]] = None,
) -> list:
    """Build an execution waypoint list targeting a specific total point count.

    Like :func:`build_exec_waypoints_with_downsampling` but instead of a stride the
    caller specifies how many waypoints they want in total.  Event waypoints are always
    kept; the remaining budget is filled by uniformly sub-sampling the non-event frames
    across the whole trajectory.

    Args:
        traj_mm             : Dense trajectory in camera mm, shape (N, 3+).
        event_source_indices: Frame indices for each event row (events CSV column 4).
        target_count        : Desired total waypoints.  Clamped to
                              [len(event_source_indices), N].
        grasp_quaternion    : Optional (qx, qy, qz, qw) applied to pick_above + pick rows.

    Returns
    -------
    list of rows ``[x_mm, y_mm, z_mm, is_event[, qx, qy, qz, qw]]`` where ``is_event``
    is ``"1"`` for event points and ``"0"`` for intermediate waypoints.
    Quaternion columns are present only when *grasp_quaternion* is not None.
    """
    traj_mm = np.asarray(traj_mm, dtype=np.float64)
    n = len(traj_mm)
    sorted_events = sorted(set(max(0, min(int(i), n - 1)) for i in event_source_indices))
    event_set = set(sorted_events)

    # Clamp target to valid range
    target_count = max(len(sorted_events), min(int(target_count), n))
    intermediate_budget = target_count - len(sorted_events)

    # All non-event frame indices in order
    non_event_indices = [i for i in range(n) if i not in event_set]

    if intermediate_budget <= 0 or not non_event_indices:
        selected_intermediates: set = set()
    elif intermediate_budget >= len(non_event_indices):
        selected_intermediates = set(non_event_indices)
    else:
        # Uniform sub-sample: pick *intermediate_budget* frames from non_event_indices
        ratio = len(non_event_indices) / intermediate_budget
        selected_intermediates = set(
            non_event_indices[min(round(i * ratio), len(non_event_indices) - 1)]
            for i in range(intermediate_budget)
        )

    all_indices = sorted(event_set | selected_intermediates)

    # pick_above = event index 1, pick = event index 2 (0-based in sorted_events)
    pick_above_src = sorted_events[1] if len(sorted_events) > 1 else -1
    pick_src       = sorted_events[2] if len(sorted_events) > 2 else -1
    q_cols: Optional[Tuple[str, str, str, str]] = None
    if grasp_quaternion is not None:
        q_cols = tuple(f"{v:.6f}" for v in grasp_quaternion)  # type: ignore[assignment]

    rows = []
    for i in all_indices:
        pt = traj_mm[i, :3]
        row: list = [f"{pt[0]:.6f}", f"{pt[1]:.6f}", f"{pt[2]:.6f}",
                     "1" if i in event_set else "0"]
        if q_cols is not None:
            row += list(q_cols) if i in (pick_above_src, pick_src) else ["", "", "", ""]
        rows.append(row)
    return rows


def apply_orientation_to_events_csv(
    events_csv_path: "str | Path",
    grasp_quaternion: Tuple[float, float, float, float],
    *,
    also_place: bool = False,
) -> None:
    """Rewrite *events_csv_path* to include gripper-orientation columns.

    The function reads the existing events CSV, adds ``qx``, ``qy``, ``qz``,
    ``qw`` columns to **pick_above** and **pick|grasp** rows (and optionally
    **place_above** and **place|release** rows), then writes the file back
    in-place.

    Args:
        events_csv_path: Path to the ``*_events_mm.csv`` file.
        grasp_quaternion: (qx, qy, qz, qw) from ``estimate_grasp_orientation``.
        also_place: If True, apply the same quaternion to the place events too.
    """
    path = Path(events_csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    qx, qy, qz, qw = [f"{v:.6f}" for v in grasp_quaternion]
    pick_labels = {"pre_pick|pick_above", "pick|grasp"}
    place_labels = {"pre_release|place_above", "place|release"}

    rows_in: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows_in.append(row)

    if not rows_in:
        return

    header = rows_in[0]
    q_header = ["qx", "qy", "qz", "qw"]

    # Determine if orientation columns are already present
    if "qx" not in header:
        header = header + q_header
        data_rows = rows_in[1:]
        new_data: list[list[str]] = []
        for row in data_rows:
            label = row[5] if len(row) > 5 else ""
            apply_q = label in pick_labels or (also_place and label in place_labels)
            row = list(row) + ([qx, qy, qz, qw] if apply_q else ["", "", "", ""])
            new_data.append(row)
    else:
        # Update existing quaternion columns
        qi = header.index("qx")
        data_rows = rows_in[1:]
        new_data = []
        for row in data_rows:
            label = row[5] if len(row) > 5 else ""
            apply_q = label in pick_labels or (also_place and label in place_labels)
            row = list(row)
            while len(row) < qi + 4:
                row.append("")
            if apply_q:
                row[qi:qi + 4] = [qx, qy, qz, qw]
            new_data.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(new_data)

    logger.info("Orientation applied to %s (qx=%s qy=%s qz=%s qw=%s)", path.name, qx, qy, qz, qw)


def compute_ik_for_reuse_traj(
    skill_reuse_npy: str,
    urdf_path: str,
    robot_config: dict,
    grasp_idx: int,
    release_idx: int,
) -> dict:
    """Run the full robot IK pipeline on a reuse trajectory.

    Returns the same structure as handle_svo_robot step_results["robot"].
    """
    from src.streamlit_template.core.Common.robot_playback import (
        dmp_xyz_to_cartesian,
        compute_ik_trajectory,
    )

    traj = np.load(skill_reuse_npy)
    target_len = len(traj)

    default_offset = robot_config.get("dmp_offset", [0.4, 0.0, 0.2])
    default_scale = robot_config.get("dmp_scale", [0.5, 0.5, 0.5])
    dmp_rot_z = robot_config.get("dmp_rotation_z", 90.0)
    dmp_flip_z = robot_config.get("flip_z", False)
    dmp_arm_reach = robot_config.get("arm_reach", 0.0)

    cart = dmp_xyz_to_cartesian(
        dmp_npy=skill_reuse_npy,
        scale_xyz=tuple(default_scale),
        offset_xyz=tuple(default_offset),
        flip_y=False,
        flip_z=dmp_flip_z,
        rotate_z=dmp_rot_z,
        add_arch=False,
        target_frames=target_len,
        arm_reach=dmp_arm_reach,
    )
    cart_path = cart["cartesian_path"]

    ik = compute_ik_trajectory(urdf_path=urdf_path, cartesian_path=cart_path)
    q_traj = ik["q_traj"]

    video_fps = 15.0
    num_frames = len(q_traj)
    frame_timestamps = []
    sr_release = min(release_idx, num_frames - 1)
    n_pre = min(sr_release + 1, num_frames)
    n_post = num_frames - n_pre

    for i in range(num_frames):
        if i < n_pre:
            frame_timestamps.append(i / video_fps)
        else:
            post_i = i - n_pre
            frame_timestamps.append((n_pre + post_i * max(1, num_frames - n_pre) / max(n_post, 1)) / video_fps)

    return {
        "q_traj": q_traj,
        "cart_path": cart_path,
        "frame_timestamps": frame_timestamps,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "num_frames": num_frames,
    }
