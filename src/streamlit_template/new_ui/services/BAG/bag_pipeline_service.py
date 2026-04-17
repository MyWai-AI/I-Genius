"""
BAG Pipeline Service - Wraps BAG scripts 01-12 into 5 pipeline steps.
Script 00 (extraction) is handled by local_page on upload.

Data contract matches Generic pipeline_service for UI compatibility:
- handle_bag_hands    → {type: "hands",  paths: [annotated imgs]}
- handle_bag_objects   → {type: "objects", paths: [annotated imgs]}
- handle_bag_trajectory → {type: "trajectory2d", timestamps, x, y, z}
- handle_bag_dmp       → {type: "dmp3d", fig, timestamps}
- handle_bag_robot     → {type: "robot3d", q_traj, frame_timestamps}
"""
import json
import streamlit as st
from pathlib import Path
import numpy as np
import os
import cv2
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add core BAG scripts to path to import _paper_* modules
# Current file: src/streamlit_template/new_ui/services/BAG/bag_pipeline_service.py
# Core BAG: src/streamlit_template/core/BAG
CORE_BAG_PATH = Path(__file__).resolve().parents[3] / "core" / "BAG"
if str(CORE_BAG_PATH) not in sys.path:
    sys.path.append(str(CORE_BAG_PATH))

# Import constants and helpers from the new validated scripts
from _paper_config import (
    KALMAN_Q, KALMAN_R, KALMAN_AUTO_TUNE,
    GRASP_DISTANCE_THRESHOLD_M, GRASP_STABLE_WINDOW,
    RELEASE_MIN_MOVE_FRAMES, RELEASE_STABLE_WINDOW, RELEASE_STABLE_WINDOW_MAX_FRAC,
    GMM_COMPONENTS,
    DMP_N_BFS, DMP_ALPHA_Z, DMP_BETA_Z, DMP_ALPHA_S, DMP_REG_LAMBDA,
    DMP_LAMBDA_CANDIDATES, DMP_TUNE_SMOOTHNESS_WEIGHT,
    PREPOST_DELTA_P
)
from _paper_utils import (
    CameraIntrinsics,
    kalman_smooth_3d, auto_kalman_params, save_plot_xyz,
    stable_hand_bbox_center, median_valid_depth, to_camera_xyz,
)
from _paper_dmp import (
    learn_dmp, rollout_dmp, save_model, load_model
)


def _init_step_results():
    if "step_results" not in st.session_state:
        st.session_state.step_results = {}


def _objects_run_params_match(objects_dir: Path, params: dict) -> bool:
    """Return True if run_metadata.json exists in objects_dir and matches params."""
    meta_path = objects_dir / "run_metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(str(meta_path), "r", encoding="utf-8") as _f:
            saved = json.load(_f)
        for key, val in params.items():
            if saved.get(key) != val:
                return False
        return True
    except Exception:
        return False


def _save_objects_run_metadata(objects_dir: Path, params: dict, extra: dict = None):
    """Save run parameters + stats to run_metadata.json in objects_dir."""
    import datetime
    meta = dict(params)
    if extra:
        meta.update(extra)
    meta["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta_path = objects_dir / "run_metadata.json"
    try:
        with open(str(meta_path), "w", encoding="utf-8") as _f:
            json.dump(meta, _f, indent=2)
    except Exception:
        pass


def _resolve_versioned_objects_session(base_path: Path, base_sess: str, run_params: dict) -> str:
    """Return a versioned session id for the objects directory.
    Scans base_path/objects/<base_sess>_N/ for a matching run_metadata.json.
    Returns matched version id if found, otherwise allocates next version.
    """
    if not base_sess:
        return base_sess

    objects_root = base_path / "objects"
    best_match = None
    max_n = 0

    for candidate in sorted(objects_root.glob(f"{base_sess}_*")):
        if not candidate.is_dir():
            continue
        suffix = candidate.name[len(base_sess) + 1:]
        if not suffix.isdigit():
            continue
        n = int(suffix)
        max_n = max(max_n, n)
        if _objects_run_params_match(candidate, run_params):
            best_match = candidate.name

    if best_match:
        return best_match

    return f"{base_sess}_{max_n + 1}"


# ---------------------------------------------------------------------------
# Helper: generate annotated hand frames from RGB + hand_3d_raw.npy
# ---------------------------------------------------------------------------

def _annotate_hand_frames(rgb_dir: Path, hand_npy: Path, out_dir: Path):
    """Draw wrist crosshair on each RGB frame for the viewer grid."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = "data/Common/ai_model/hand/hand_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = vision.HandLandmarker.create_from_options(options)

    rgb_files = sorted(rgb_dir.glob("frame_*.png"))
    annotated_paths = []

    for fname in rgb_files:
        bgr = cv2.imread(str(fname))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hands.detect(mp_image)

        if result.hand_landmarks:
            h, w, _ = bgr.shape
            for hand_lm in result.hand_landmarks:
                for lm in hand_lm:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(bgr, (px, py), 3, (0, 255, 0), -1)
                # Draw wrist → middle finger line
                wrist = hand_lm[0]
                mid = hand_lm[9]
                cv2.line(
                    bgr,
                    (int(wrist.x * w), int(wrist.y * h)),
                    (int(mid.x * w), int(mid.y * h)),
                    (0, 255, 255), 2,
                )

        out_path = out_dir / f"hands_{fname.stem}.jpg"
        cv2.imwrite(str(out_path), bgr)
        annotated_paths.append(out_path)

    return annotated_paths


# ---------------------------------------------------------------------------
# Helper: generate annotated object frames from RGB + YOLO detections
# ---------------------------------------------------------------------------

def _annotate_object_frames(rgb_dir: Path, out_dir: Path, model_path: str = None):
    """Run YOLO on each RGB frame and save annotated images."""
    from ultralytics import YOLO

    if model_path is None:
        model_path = "data/Common/ai_model/object/blueball.pt"

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    rgb_files = sorted(rgb_dir.glob("frame_*.png"))
    annotated_paths = []
    total_detections = 0

    for fname in rgb_files:
        bgr = cv2.imread(str(fname))
        results = model(bgr, verbose=False)[0]

        if results.obb is not None and len(results.obb) > 0:
            total_detections += len(results.obb)
            # Draw OBB boxes (Bug 3)
            for box in results.obb:
                xywhr = box.xywhr[0].cpu().numpy()
                cx, cy, w_box, h_box, rotation = xywhr
                rect = ((cx, cy), (w_box, h_box), np.degrees(rotation))
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
                cv2.drawContours(bgr, [box_points], 0, (0, 0, 255), 2)
                
                conf = float(box.conf[0])
                cv2.putText(bgr, f"{conf:.2f}", (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_path = out_dir / f"objects_{fname.stem}.jpg"
        cv2.imwrite(str(out_path), bgr)
        annotated_paths.append(out_path)

    return annotated_paths, total_detections


# ========================================================================
# STEP 1: HANDS (Scripts 01 + 02)
# ========================================================================

def handle_bag_hands(base_path: Path, session_id: str = None):
    """Step 1: Extract 3D hand trajectory + generate annotated frames for UI."""
    _init_step_results()

    if session_id:
        rgb_dir = base_path / "frames" / session_id
        depth_dir = base_path / "depth_meters" / session_id
        hands_dir = base_path / "hands" / session_id
        plots_dir = base_path / "plots" / session_id
        intrinsics_path = base_path / "camera" / f"{session_id}.npy"
    else:
        # Legacy/Fallback
        rgb_dir = base_path / "frames"
        depth_dir = base_path / "depth_meters"
        hands_dir = base_path / "hands"
        plots_dir = base_path / "plots"
        if (base_path / "camera_intrinsics.npy").exists():
             intrinsics_path = base_path / "camera_intrinsics.npy"
        else:
             intrinsics_path = base_path.parent / "camera" / f"{base_path.name}.npy"

    hands_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not intrinsics_path.exists():
        # Try to infer session from base_path name as last resort for old style
        # But for now just error/warn
        pass

    if not intrinsics_path.exists():
        st.error(f"Camera intrinsics not found at {intrinsics_path}. Run BAG extraction first.")
        return None

    rgb_files = sorted(rgb_dir.glob("frame_*.png"))
    if len(rgb_files) == 0:
        st.error(f"No RGB frames found in {rgb_dir}. Run BAG extraction first.")
        return None

    with st.spinner("Extracting 3D hand trajectory..."):
        # --- Script 01 logic: extract hand trajectory ---
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        intr_raw = np.load(str(intrinsics_path), allow_pickle=True).item()
        intr = CameraIntrinsics(
            fx=float(intr_raw["fx"]), fy=float(intr_raw["fy"]),
            cx=float(intr_raw["cx"]), cy=float(intr_raw["cy"]),
            width=int(intr_raw.get("width", -1)), height=int(intr_raw.get("height", -1)),
        )

        model_path = "data/Common/ai_model/hand/hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands_detector = vision.HandLandmarker.create_from_options(options)

        # --- Single-pass: extract 3D trajectory AND generate annotations ---
        annotated_dir = hands_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)

        TIP_IDS = [4, 8, 12, 16, 20]
        trajectory_3d = []
        tips_3d = []
        annotated_paths = []

        for fname in rgb_files:
            depth_path = depth_dir / fname.name.replace(".png", ".npy")
            bgr = cv2.imread(str(fname))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = np.load(str(depth_path))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hands_detector.detect(mp_image)

            point_added = False
            if results.hand_landmarks:
                h, w, _ = bgr.shape
                lms = results.hand_landmarks[0]
                u, v = stable_hand_bbox_center(lms, width=w, height=h)
                z = median_valid_depth(depth, u=u, v=v, half=2)

                tip_xyz = np.full((len(TIP_IDS), 3), np.nan, dtype=np.float32)
                for j, tip_id in enumerate(TIP_IDS):
                    lm = lms[tip_id]
                    tu, tv = int(round(lm.x * w)), int(round(lm.y * h))
                    tz = median_valid_depth(depth, u=tu, v=tv, half=2)
                    if tz is not None:
                        tip_xyz[j] = to_camera_xyz(tu, tv, tz, intr)

                if z is None:
                    valid_tip = tip_xyz[np.isfinite(tip_xyz).all(axis=1)]
                    center_xyz = np.median(valid_tip, axis=0).astype(np.float32) if valid_tip.size else None
                else:
                    center_xyz = to_camera_xyz(u, v, z, intr)

                if center_xyz is not None:
                    trajectory_3d.append(center_xyz)
                    tips_3d.append(tip_xyz)
                    point_added = True

                # --- Draw annotations on the same frame (single-pass) ---
                for hand_lm in results.hand_landmarks:
                    for lm in hand_lm:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(bgr, (px, py), 3, (0, 255, 0), -1)
                    wrist = hand_lm[0]
                    mid = hand_lm[9]
                    cv2.line(
                        bgr,
                        (int(wrist.x * w), int(wrist.y * h)),
                        (int(mid.x * w), int(mid.y * h)),
                        (0, 255, 255), 2,
                    )

            if not point_added:
                if len(trajectory_3d) > 0:
                    trajectory_3d.append(trajectory_3d[-1].copy())
                    tips_3d.append(tips_3d[-1].copy())
                else:
                    trajectory_3d.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                    tips_3d.append(np.full((len(TIP_IDS), 3), np.nan, dtype=np.float32))

            # Save annotated frame
            out_path = annotated_dir / f"hands_{fname.stem}.jpg"
            cv2.imwrite(str(out_path), bgr)
            annotated_paths.append(out_path)

        trajectory_3d = np.asarray(trajectory_3d, dtype=np.float32)
        tips_np = np.asarray(tips_3d, dtype=np.float32)
        np.save(str(hands_dir / "hand_3d_raw.npy"), trajectory_3d)
        np.save(str(hands_dir / "hand_tips_3d_raw.npy"), tips_np)

    with st.spinner("Smoothing hand trajectory (Kalman)..."):
        # --- Script 02 logic: Kalman smoothing ---
        q, r = (KALMAN_Q, KALMAN_R)
        if KALMAN_AUTO_TUNE:
             q, r = auto_kalman_params(trajectory_3d, base_q=KALMAN_Q, base_r=KALMAN_R)

        traj_smooth = kalman_smooth_3d(trajectory_3d, q=q, r=r)
        np.save(str(hands_dir / "hand_3d_smooth.npy"), traj_smooth)

        # Smooth tips per-fingertip (matches standalone script 02 logic)
        tips_smooth = np.zeros_like(tips_np)
        for j in range(tips_np.shape[1]):   # 5 tips
            tips_smooth[:, j, :] = kalman_smooth_3d(tips_np[:, j, :], q=q, r=r)
        np.save(str(hands_dir / "hand_tips_3d_smooth.npy"), tips_smooth)

    st.session_state.step_results["hands"] = {"type": "hands", "paths": annotated_paths}
    return {"trajectory": trajectory_3d, "smooth": traj_smooth}


# ========================================================================
# STEP 2: OBJECTS (Scripts 03 + 04)
# ========================================================================

def handle_bag_objects(
    base_path: Path,
    session_id: str = None,
    tracking_bbox_xyxy=None,
    tracking_label: str = None,
    bbox_size_ratio: float = 4.0,
):
    """Step 2: Detect Objects (YOLO).

    If tracking_bbox_xyxy is provided (from Local page Step 2 selection),
    the tracker follows the object closest to that initial position each frame.
    Otherwise falls back to highest-confidence detection.
    """
    _init_step_results()
    
    status_ph = st.empty()
    status_ph.info("⏳ Detect Objects: Initializing...")

    if session_id:
        rgb_dir = base_path / "frames" / session_id
        depth_dir = base_path / "depth_meters" / session_id
        intrinsics_path = base_path / "camera" / f"{session_id}.npy"
    else:
        rgb_dir = base_path / "frames"
        depth_dir = base_path / "depth_meters"
        if (base_path / "camera_intrinsics.npy").exists():
             intrinsics_path = base_path / "camera_intrinsics.npy"
        else:
             intrinsics_path = base_path.parent / "camera" / f"{base_path.name}.npy"

    if not intrinsics_path.exists():
        st.error("Camera intrinsics not found.")
        return None

    # Build params fingerprint for cache matching
    _bag_model_path = st.session_state.get("selected_object_model_path", "data/Common/ai_model/object/blueball.pt") or "data/Common/ai_model/object/blueball.pt"
    _norm_bbox = None
    if isinstance(tracking_bbox_xyxy, (list, tuple)) and len(tracking_bbox_xyxy) == 4:
        try:
            _norm_bbox = [float(v) for v in tracking_bbox_xyxy]
        except Exception:
            _norm_bbox = None
    _run_params = {
        "model": _bag_model_path,
        "tracking_label": tracking_label,
        "bbox": _norm_bbox,
        "bbox_size_ratio": round(float(bbox_size_ratio), 2),
    }

    # Resolve versioned objects session id (e.g. a1b2c3d4_2)
    _obj_sess = _resolve_versioned_objects_session(base_path, session_id, _run_params)
    st.session_state["active_objects_session_id"] = _obj_sess

    if _obj_sess:
        objects_dir = base_path / "objects" / _obj_sess
        plots_dir = base_path / "plots" / _obj_sess
    else:
        objects_dir = base_path / "objects"
        plots_dir = base_path / "plots"

    objects_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Rerun guard: skip if run_metadata.json matches current params ---
    _obj_smooth_path = objects_dir / "object_3d_smooth.npy"
    _obj_raw_path = objects_dir / "object_3d_raw.npy"
    _obj_annotated_dir = objects_dir / "annotated"
    if _obj_smooth_path.exists() and _obj_raw_path.exists() and _obj_annotated_dir.exists():
        _cached_paths = sorted(_obj_annotated_dir.glob("*.jpg"))
        if _cached_paths and _objects_run_params_match(objects_dir, _run_params):
            _obj_raw = np.load(str(_obj_raw_path))
            _obj_smooth = np.load(str(_obj_smooth_path))
            _saved_meta = json.load(open(str(objects_dir / "run_metadata.json"), encoding="utf-8"))
            st.session_state.step_results["objects"] = {
                "type": "objects",
                "paths": [str(p) for p in _cached_paths],
                "total_detections": _saved_meta.get("total_detections", len(_cached_paths)),
                "model_used": _saved_meta.get("model"),
                "selection_mode": "local-selected" if _saved_meta.get("bbox") else "auto-best",
                "run_metadata": _saved_meta,
                "objects_session_id": _obj_sess,
            }
            status_ph.empty()
            return {"trajectory": _obj_raw, "smooth": _obj_smooth}

    rgb_files = sorted(rgb_dir.glob("frame_*.png"))

    with st.spinner("Extracting 3D object trajectory..."):
        # --- Script 03 logic (single-pass: extract + annotate) ---
        from ultralytics import YOLO

        intr_raw = np.load(str(intrinsics_path), allow_pickle=True).item()
        intr = CameraIntrinsics(
            fx=float(intr_raw["fx"]), fy=float(intr_raw["fy"]),
            cx=float(intr_raw["cx"]), cy=float(intr_raw["cy"]),
            width=int(intr_raw.get("width", -1)), height=int(intr_raw.get("height", -1)),
        )

        # Resolve tracking target from local-page selection
        _raw_bbox = tracking_bbox_xyxy
        if isinstance(_raw_bbox, (list, tuple)) and len(_raw_bbox) == 4:
            try:
                _raw_bbox = [float(v) for v in _raw_bbox]
            except Exception:
                _raw_bbox = None
        else:
            _raw_bbox = None
        _tracking_mode = _raw_bbox is not None
        _target_cx = ((_raw_bbox[0] + _raw_bbox[2]) / 2.0) if _tracking_mode else None
        _target_cy = ((_raw_bbox[1] + _raw_bbox[3]) / 2.0) if _tracking_mode else None
        _target_area = ((_raw_bbox[2] - _raw_bbox[0]) * (_raw_bbox[3] - _raw_bbox[1])) if _tracking_mode else None

        model_path = st.session_state.get("selected_object_model_path", "data/Common/ai_model/object/blueball.pt")
        if not model_path or not Path(model_path).exists():
            model_path = "data/Common/ai_model/object/blueball.pt"
        model = YOLO(model_path)
        trajectory_3d = []

        annotated_dir = objects_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_paths = []
        total_det = 0

        for fname in rgb_files:
            depth_path = depth_dir / fname.name.replace(".png", ".npy")
            bgr = cv2.imread(str(fname))
            depth = np.load(str(depth_path))
            results = model(bgr, verbose=False)[0]

            detected = False
            best_cx, best_cy = None, None

            if results.obb is not None and len(results.obb) > 0:
                total_det += len(results.obb)
                confs = results.obb.conf.detach().cpu().numpy()
                xywhr_all = results.obb.xywhr.detach().cpu().numpy()  # (N, 5)
                _obb_names = results.names or {}
                _cls_ids_all = results.obb.cls.detach().cpu().numpy().astype(int)

                if _tracking_mode and _target_cx is not None:
                    # Filter by label name (reject different object classes) and size,
                    # then pick closest to current target center.
                    _pool = []
                    for _pi in range(len(confs)):
                        _cls_name = _obb_names.get(_cls_ids_all[_pi], "")
                        if tracking_label and _cls_name and _cls_name != tracking_label:
                            continue
                        if _target_area is not None and _target_area > 1.0:
                            _det_area = float(xywhr_all[_pi][2]) * float(xywhr_all[_pi][3])
                            if not (1.0 / bbox_size_ratio <= _det_area / _target_area <= bbox_size_ratio):
                                continue
                        _pool.append(_pi)
                    if not _pool:
                        _pool = list(range(len(confs)))
                    best_idx = int(min(
                        _pool,
                        key=lambda i: (
                            (xywhr_all[i][0] - _target_cx) ** 2 +
                            (xywhr_all[i][1] - _target_cy) ** 2
                        ),
                    ))
                else:
                    best_idx = int(np.argmax(confs))

                cx, cy, w_box, h_box, rotation = xywhr_all[best_idx]

                # Draw only the selected box
                rect = ((float(cx), float(cy)), (float(w_box), float(h_box)), float(np.degrees(rotation)))
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
                cv2.drawContours(bgr, [box_points], 0, (0, 0, 255), 2)
                cv2.putText(bgr, f"{float(confs[best_idx]):.2f}", (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                best_cx, best_cy = float(cx), float(cy)
                u_i, v_i = int(round(best_cx)), int(round(best_cy))
                z = median_valid_depth(depth, u=u_i, v=v_i, half=2)
                if z is not None:
                    trajectory_3d.append(to_camera_xyz(u_i, v_i, z, intr))
                    detected = True
                    # Update target center for next frame
                    _target_cx, _target_cy = best_cx, best_cy

            if not detected:
                if len(trajectory_3d) > 0:
                    trajectory_3d.append(trajectory_3d[-1].copy())
                else:
                    trajectory_3d.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))

            # Save annotated frame
            out_path = annotated_dir / f"objects_{fname.stem}.jpg"
            cv2.imwrite(str(out_path), bgr)
            annotated_paths.append(out_path)

        trajectory_3d = np.asarray(trajectory_3d, dtype=np.float32)
        np.save(str(objects_dir / "object_3d_raw.npy"), trajectory_3d)

    with st.spinner("Smoothing object trajectory (Kalman)..."):
        # --- Script 04 logic: Kalman smoothing ---
        q, r = (KALMAN_Q, KALMAN_R)
        if KALMAN_AUTO_TUNE:
             q, r = auto_kalman_params(trajectory_3d, base_q=KALMAN_Q, base_r=KALMAN_R)
        
        smooth = kalman_smooth_3d(trajectory_3d, q=q, r=r)
        np.save(str(objects_dir / "object_3d_smooth.npy"), smooth)

    _save_objects_run_metadata(objects_dir, _run_params, {
        "total_detections": total_det,
        "frame_count": len(annotated_paths),
    })
    _saved_meta = json.load(open(str(objects_dir / "run_metadata.json"), encoding="utf-8")) if (objects_dir / "run_metadata.json").exists() else {}

    st.session_state.step_results["objects"] = {
        "type": "objects",
        "paths": annotated_paths,
        "total_detections": total_det,
        "model_used": _bag_model_path,
        "selection_mode": "local-selected" if _tracking_mode else "auto-best",
        "run_metadata": _saved_meta,
        "objects_session_id": _obj_sess,
    }
    return {"trajectory": trajectory_3d, "smooth": smooth}


# ========================================================================
# STEP 3: TRAJECTORY (Scripts 05 + 06 + 07 + 08)
# ========================================================================

def handle_bag_trajectory(base_path: Path, session_id: str = None):
    """Step 3: Trajectory Reconstruction."""
    _init_step_results()

    status_ph = st.empty()
    status_ph.info("⏳ Trajectory: Initializing...")

    _obj_sess = st.session_state.get("active_objects_session_id") or session_id
    if session_id:
        hands_dir = base_path / "hands" / session_id
        objects_dir = base_path / "objects" / _obj_sess if _obj_sess else base_path / "objects" / session_id
        seg_dir = base_path / "segmentation" / _obj_sess if _obj_sess else base_path / "segmentation" / session_id
        plots_dir = base_path / "plots" / _obj_sess if _obj_sess else base_path / "plots" / session_id
    else:
        hands_dir = base_path / "hands"
        objects_dir = base_path / "objects"
        seg_dir = base_path / "segmentation"
        plots_dir = base_path / "plots"

    seg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    hand_smooth = np.load(str(hands_dir / "hand_3d_smooth.npy"))
    obj_smooth = np.load(str(objects_dir / "object_3d_smooth.npy"))

    # --- Script 05: Grasp estimation ---
    with st.spinner("Estimating grasp point..."):
        tips_path = hands_dir / "hand_tips_3d_smooth.npy"
        if not tips_path.exists():
            st.error("hand_tips_3d_smooth.npy not found — re-run Step 1.")
            return None
        hand_tips = np.load(str(tips_path))  # (T, 5, 3)

        t_len = min(len(hand_tips), len(obj_smooth))
        hand_tips = hand_tips[:t_len]
        obj_smooth_clipped = obj_smooth[:t_len]

        # Paper Algorithm 1: initialPosition = mean(objTraj[:start])
        # Skip leading zero-padded frames (no detection yet).
        _nonzero_mask = np.any(obj_smooth_clipped != 0, axis=1) & np.isfinite(obj_smooth_clipped).all(axis=1)
        _first_real = int(np.argmax(_nonzero_mask)) if np.any(_nonzero_mask) else 0
        _INITIAL_FRAC, _INITIAL_MIN = 0.10, 5
        n_init = max(_INITIAL_MIN, int(t_len * _INITIAL_FRAC))
        n_init = min(n_init, t_len)
        init_seg = obj_smooth_clipped[_first_real : _first_real + n_init]
        finite_init = np.isfinite(init_seg).all(axis=1)
        initial_position = init_seg[finite_init].mean(axis=0) if np.any(finite_init) else obj_smooth_clipped[_first_real]

        # Distance from fingertips (or hand center) to fixed initial position
        distances = np.full(t_len, np.inf, dtype=np.float32)
        for i in range(t_len):
            tips_frame = hand_tips[i]
            valid = np.isfinite(tips_frame).all(axis=1)
            if np.any(valid):
                distances[i] = float(np.min(np.linalg.norm(tips_frame[valid] - initial_position, axis=1)))

        finite = np.isfinite(distances)
        if not np.any(finite):
            hand_center = hand_smooth[:t_len]
            center_finite = np.isfinite(hand_center).all(axis=1)
            if np.any(center_finite):
                distances = np.linalg.norm(hand_center - initial_position, axis=1).astype(np.float32)
                finite = np.isfinite(distances)

        if np.any(finite):
            distances[~np.isfinite(distances)] = float(np.nanmax(distances[finite]))

        np.save(str(seg_dir / "min_hand_initial_distance.npy"), distances)

        # Paper: graspIndex = argmin(smoothed distance to initialPosition)
        from scipy.ndimage import uniform_filter1d
        distances_smooth = uniform_filter1d(distances.astype(np.float64), size=5).astype(np.float32)
        local_min_idx = int(np.argmin(distances_smooth))
        if local_min_idx == 0:
            local_min_idx = int(np.argmin(distances_smooth[1:])) + 1

        grasp_idx = local_min_idx
        for i in range(max(0, grasp_idx - GRASP_STABLE_WINDOW), grasp_idx + 1):
            end = i + GRASP_STABLE_WINDOW
            if end <= len(distances_smooth) and np.all(distances_smooth[i:end] <= distances_smooth[grasp_idx] * 1.5):
                grasp_idx = i
                break

        grasp_idx = int(np.clip(grasp_idx, 0, t_len - 1))
        np.save(str(seg_dir / "grasp_idx.npy"), np.array(grasp_idx, dtype=np.int32))

    # --- Script 06: Reconstruct trajectory ---
    with st.spinner("Reconstructing object trajectory..."):
        hand_traj = hand_smooth
        obj_traj = obj_smooth
        T = len(obj_traj)
        offset = obj_traj[grasp_idx] - hand_traj[grasp_idx]
        reconstructed = obj_traj.copy()
        for i in range(grasp_idx, T):
            reconstructed[i] = hand_traj[i] + offset
        np.save(str(objects_dir / "object_3d_reconstructed.npy"), reconstructed)

    # --- Script 07: GMM segmentation ---
    with st.spinner("Running GMM segmentation (release detection)..."):
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=GMM_COMPONENTS, covariance_type="full", random_state=0, n_init=5
        )
        labels = gmm.fit_predict(reconstructed)

        np.save(str(seg_dir / "gmm_labels.npy"), labels)
        np.save(str(seg_dir / "gmm_means.npy"), gmm.means_)
        np.save(str(seg_dir / "gmm_covariances.npy"), gmm.covariances_)
        np.save(str(seg_dir / "gmm_weights.npy"), gmm.weights_)

        # Paper Algorithm 1: finalPosition = mean(objTraj[end:])
        # last_cluster = cluster whose mean is closest to finalPosition
        _FINAL_FRAC, _FINAL_MIN = 0.10, 5
        n_final = max(_FINAL_MIN, int(len(reconstructed) * _FINAL_FRAC))
        n_final = min(n_final, len(reconstructed))
        final_seg = reconstructed[max(0, len(reconstructed) - n_final):]
        finite_final = np.isfinite(final_seg).all(axis=1) & np.any(final_seg != 0, axis=1)
        if not np.any(finite_final):
            finite_final = np.isfinite(final_seg).all(axis=1)
        final_position = final_seg[finite_final].mean(axis=0) if np.any(finite_final) else reconstructed[-1]

        dists_to_final = np.linalg.norm(gmm.means_ - final_position, axis=1)
        last_cluster = int(np.argmin(dists_to_final))
        np.save(str(seg_dir / "last_cluster.npy"), np.array(last_cluster, dtype=np.int32))

        # Paper: releaseIndex = first stable window in last cluster
        min_release_idx = min(len(reconstructed) - 1, grasp_idx + RELEASE_MIN_MOVE_FRAMES)
        stable_window = max(2, RELEASE_STABLE_WINDOW)

        release_idx = None
        for i in range(min_release_idx, len(reconstructed) - stable_window + 1):
            if np.all(labels[i : i + stable_window] == last_cluster):
                release_idx = i
                break

        if release_idx is None:
            candidate = np.where(labels == last_cluster)[0]
            candidate = candidate[candidate >= min_release_idx]
            release_idx = int(candidate[0]) if candidate.size else len(reconstructed) - 1

        np.save(str(seg_dir / "release_idx.npy"), np.array(release_idx, dtype=np.int32))

    # --- Script 08: Extract skill phases ---
    with st.spinner("Extracting skill phases (Reach, Move)..."):
        # Logic from 08_extract_skill_phases.py
        t_len = len(reconstructed)
        grasp_idx = int(np.clip(grasp_idx, 1, t_len - 2))
        release_idx = int(np.clip(release_idx, grasp_idx + 1, t_len - 1))
        
        # Enforce minimum move duration
        min_release = min(t_len - 1, grasp_idx + RELEASE_MIN_MOVE_FRAMES)
        if release_idx < min_release:
            release_idx = min_release

        # Segments — use HAND trajectory (not reconstructed), matching core script 08
        reach = hand_smooth[:grasp_idx]                # [P(t1), ..., P(tg-1)]
        move = hand_smooth[grasp_idx + 1 : release_idx]  # [P(tg+1), ..., P(tr-1)]

        grasp_pos = hand_smooth[grasp_idx]
        release_pos = hand_smooth[release_idx]

        pre_grasp = grasp_pos + np.array([0.0, 0.0, PREPOST_DELTA_P], dtype=np.float32)
        post_grasp = pre_grasp.copy()
        pre_release = release_pos + np.array([0.0, 0.0, PREPOST_DELTA_P], dtype=np.float32)
        post_release = pre_release.copy()

        np.save(str(seg_dir / "reach_traj.npy"), reach)
        np.save(str(seg_dir / "move_traj.npy"), move)
        # Note: 'transport_traj.npy' and 'place_traj.npy' are removed in new logic.
        # But for compatibility with older viz code, we might need to handle their absence.
        
        np.save(str(seg_dir / "grasp_pos.npy"), grasp_pos)
        np.save(str(seg_dir / "release_pos.npy"), release_pos)
        np.save(str(seg_dir / "pre_grasp_pos.npy"), pre_grasp)
        np.save(str(seg_dir / "post_grasp_pos.npy"), post_grasp)
        np.save(str(seg_dir / "pre_release_pos.npy"), pre_release)
        np.save(str(seg_dir / "post_release_pos.npy"), post_release)

        phase_indices = {
            "reach_start": 0,
            "reach_end": grasp_idx - 1,
            "grasp_idx": grasp_idx,
            "move_start": grasp_idx + 1,
            "move_end": release_idx - 1,
            "release_idx": release_idx,
        }
        np.save(str(seg_dir / "phase_indices.npy"), phase_indices)
        with open(str(seg_dir / "phase_indices.json"), 'w', encoding='utf-8') as f:
            json.dump(phase_indices, f, indent=2)

    # --- Build trajectory2d data for UI ---
    timestamps = list(range(len(reconstructed)))
    x_vals = reconstructed[:, 0].tolist()
    y_vals = reconstructed[:, 1].tolist()
    z_vals = reconstructed[:, 2].tolist()
    T_total = len(reconstructed)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("X Position [m]", "Y Position [m]", "Z Position [m]"),
    )
    fig.add_trace(go.Scatter(x=timestamps, y=x_vals, mode="lines+markers", name="X", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=y_vals, mode="lines+markers", name="Y", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=z_vals, mode="lines+markers", name="Z", line=dict(color="red")), row=3, col=1)

    grasp_pos_idx = grasp_idx
    release_pos_idx = release_idx
    
    phase_colors = [
        (0, grasp_pos_idx, "rgba(66,133,244,0.12)"),          # Reach - blue
        (grasp_pos_idx, release_pos_idx, "rgba(52,168,83,0.12)"),  # Move - green
    ]
    # Optional: Post-release
    if release_pos_idx < T_total:
         phase_colors.append((release_pos_idx, T_total, "rgba(200,200,200,0.12)")) # Post - gray
         
    for x0, x1, color in phase_colors:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, layer="below", line_width=0)

    # Phase boundary lines + grasp/release
    markers = [
        (grasp_pos_idx, "dash", "black", 2, "Grasp"),
        (release_pos_idx, "dot", "#EA8600", 2, "Release"),
    ]
    for mx, dash, color, width, label in markers:
        fig.add_vline(x=mx, line_dash=dash, line_color=color, line_width=width,
                      annotation_text=label, annotation_position="top right", 
                      annotation_font=dict(size=12, color=color))

    fig.update_layout(height=500, showlegend=True, hovermode="x unified")
    fig.update_xaxes(title_text="Frame", row=3, col=1)

    st.session_state.step_results["trajectory"] = {
        "type": "trajectory2d",
        "fig": fig,
        "timestamps": timestamps,
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "paths": [str(p) for p in sorted((objects_dir / "annotated").glob("*.jpg"))],
    }
    return {"grasp_idx": grasp_idx, "release_idx": release_idx}


# ========================================================================
# STEP 4: DMP (Scripts 09 + 10 + 12)
# ========================================================================

def _generate_skill_reuse_dmp(dmp_dir: Path, plot_dir: Path, seg_dir: Path):
    """
    Generate the 'Skill Reuse' trajectory by concatenating adapted DMP
    rollouts with grasp/release events (mirrors core script 12 logic).

    Outputs:
        dmp_dir / skill_reuse_traj.npy
        dmp_dir / skill_reuse_traj.csv
        dmp_dir / skill_reuse_traj.json
        plot_dir / skill_reuse_final.png
    """
    # --- Try adapted trajectories first (produced by script 11 / adapt step) ---
    reach_adapted_path = dmp_dir / "reach_adapted.npy"
    move_adapted_path = dmp_dir / "move_adapted.npy"

    if reach_adapted_path.exists():
        reach = np.load(str(reach_adapted_path))
        move = np.load(str(move_adapted_path)) if move_adapted_path.exists() else np.empty((0, 3), dtype=np.float64)
    else:
        # Fallback: rollout from trained DMP models
        try:
            reach_model = load_model(str(dmp_dir / "reach_dmp.npz"))
            move_model = load_model(str(dmp_dir / "move_dmp.npz"))
        except Exception:
            return None

        T_reach = int(round(1.0 / reach_model.dt))
        T_move = int(round(1.0 / move_model.dt))

        r_res = rollout_dmp(reach_model, timesteps=T_reach)
        reach = r_res['y']
        m_res = rollout_dmp(move_model, timesteps=T_move, y0=reach[-1])
        move = m_res['y']

    # --- Load grasp / release event positions ---
    grasp_pos_path = seg_dir / "grasp_pos.npy"
    release_pos_path = seg_dir / "release_pos.npy"
    post_release_path = seg_dir / "post_release_pos.npy"

    grasp_pos = np.load(str(grasp_pos_path)) if grasp_pos_path.exists() else reach[-1]
    release_pos = np.load(str(release_pos_path)) if release_pos_path.exists() else (move[-1] if len(move) > 0 else reach[-1])
    post_release = np.load(str(post_release_path)) if post_release_path.exists() else release_pos

    # --- Concatenate into full skill trajectory (script 12 logic) ---
    # Hold post-release pose for multiple frames so gripper-open is visible
    POST_RELEASE_HOLD = 15
    post_release_hold = np.tile(post_release[None, :], (POST_RELEASE_HOLD, 1))
    parts = [reach, grasp_pos[None, :]]
    if len(move) > 0:
        parts.append(move)
    parts.extend([release_pos[None, :], post_release_hold])
    skill_traj = np.vstack(parts)

    # --- Save .npy, .csv, .json ---
    np.save(str(dmp_dir / "skill_reuse_traj.npy"), skill_traj)
    np.savetxt(
        str(dmp_dir / "skill_reuse_traj.csv"),
        skill_traj, delimiter=',', header='x,y,z', comments='',
    )
    with open(str(dmp_dir / "skill_reuse_traj.json"), 'w', encoding='utf-8') as f:
        json.dump(skill_traj.tolist(), f)

    # --- Plot (matches script 12) ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7.2, 5.2))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(skill_traj[:, 0], skill_traj[:, 1], skill_traj[:, 2],
                label='Skill Reuse Trajectory')
        ax.scatter(*skill_traj[0], marker='+', s=120, color='black', label='Start')
        ax.scatter(*np.atleast_1d(grasp_pos), marker='o', s=80, color='tab:orange', label='Grasp')
        ax.scatter(*np.atleast_1d(release_pos), marker='o', s=80, color='tab:red', label='Release')
        ax.scatter(*skill_traj[-1], marker='*', s=150, color='green', label='Final')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Full Skill Reuse from Reach/Move DMPs')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(plot_dir / 'skill_reuse_final.png'), dpi=300)
        plt.close()
    except Exception:
        pass  # Plotting is optional; don't fail the pipeline

    return skill_traj


def handle_bag_dmp(base_path: Path, session_id: str = None):
    """Step 4: DMP Learning."""
    _init_step_results()

    _obj_sess = st.session_state.get("active_objects_session_id") or session_id
    if session_id:
        seg_dir = base_path / "segmentation" / _obj_sess if _obj_sess else base_path / "segmentation" / session_id
        dmp_dir = base_path / "dmp" / _obj_sess if _obj_sess else base_path / "dmp" / session_id
        plots_dir = base_path / "plots" / _obj_sess if _obj_sess else base_path / "plots" / session_id
    else:
        seg_dir = base_path / "segmentation"
        dmp_dir = base_path / "dmp"
        plots_dir = base_path / "plots"

    seg_dir.mkdir(parents=True, exist_ok=True)
    dmp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    reach = np.load(str(seg_dir / "reach_traj.npy"))
    move_path = seg_dir / "move_traj.npy"
    move = np.load(str(move_path)) if move_path.exists() else np.empty((0, 3))
    
    # --- Script 09: Learn DMPs with Lambda Sweeping ---
    with st.spinner("Training DMPs (Reach, Move) with lambda tuning..."):
        
        # Helper to match script 09 logic
        def train_phase_pipeline(name: str, traj: np.ndarray):
            candidates = sorted(set([float(DMP_REG_LAMBDA), *[float(x) for x in DMP_LAMBDA_CANDIDATES]]))
            best = None
            best_pack = None
            
            for reg_lambda in candidates:
                # Use capped n_bfs logic if we want to enforce smoothness more than script 09?
                # Script 09 uses DMP_N_BFS (20). 
                # User config in _paper_config says DMP_N_BFS = 20.
                # Max 100 logic is GONE in new script. It uses fixed 20.
                # This inherently provides smoothing!
                model, diag = learn_dmp(
                    y_demo=traj,
                    n_bfs=DMP_N_BFS,
                    alpha_z=DMP_ALPHA_Z,
                    beta_z=DMP_BETA_Z,
                    alpha_s=DMP_ALPHA_S,
                    reg_lambda=reg_lambda,
                )
                rollout = rollout_dmp(model, timesteps=len(traj))
                y_rep = rollout['y']
                
                rmse = float(np.sqrt(np.mean((traj - y_rep) ** 2)))
                jerk = np.diff(y_rep, n=2, axis=0) if len(y_rep) > 2 else np.zeros((1, 3))
                smoothness = float(np.mean(np.linalg.norm(jerk, axis=1)))
                score = rmse + DMP_TUNE_SMOOTHNESS_WEIGHT * smoothness
                
                if best is None or score < best['score']:
                    best = {'score': score, 'rmse': rmse, 'smoothness': smoothness, 'lambda': reg_lambda}
                    best_pack = (model, diag, rollout)
            
            model, diag, rollout = best_pack
            y_rep = rollout['y']
            save_model(str(dmp_dir / f"{name}_dmp.npz"), model)
            np.save(str(dmp_dir / f"{name}_reproduction.npy"), y_rep)
            np.save(str(dmp_dir / f"{name}_forcing_target.npy"), diag['f_target'])
            np.save(str(dmp_dir / f"{name}_forcing_rollout.npy"), rollout['f'])
            np.save(str(dmp_dir / f"{name}_phase.npy"), diag['s'])
            
            return y_rep, best

        metrics = {}
        reach_recon = None
        if len(reach) >= 5:
            reach_recon, reach_metrics = train_phase_pipeline("reach", reach)
            metrics['reach'] = reach_metrics
            
        move_recon = None
        if len(move) >= 5:
            move_recon, move_metrics = train_phase_pipeline("move", move)
            metrics['move'] = move_metrics

        with open(str(dmp_dir / "metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    # --- Script 10: Test DMP reproduction ---
    with st.spinner("Validating DMP reproduction..."):
        report = {}
        if reach_recon is not None:
            reach_model = load_model(str(dmp_dir / "reach_dmp.npz"))
            reach_repro = rollout_dmp(reach_model, timesteps=len(reach))['y']
            err = reach - reach_repro
            rmse = np.sqrt(np.mean(err**2, axis=0))
            report['reach'] = {'rmse_x': float(rmse[0]), 'rmse_y': float(rmse[1]), 'rmse_z': float(rmse[2])}
        if move_recon is not None and len(move) >= 5:
            move_model_obj = load_model(str(dmp_dir / "move_dmp.npz"))
            move_repro = rollout_dmp(move_model_obj, timesteps=len(move))['y']
            err = move - move_repro
            rmse = np.sqrt(np.mean(err**2, axis=0))
            report['move'] = {'rmse_x': float(rmse[0]), 'rmse_y': float(rmse[1]), 'rmse_z': float(rmse[2])}
        else:
            report['move'] = {'skipped': True}
        with open(str(dmp_dir / "reproduction_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

    # --- Script 11: Adapt DMPs to new start/goal ---
    with st.spinner("Adapting DMPs to new start/goal poses..."):
        pre_grasp = np.load(str(seg_dir / "pre_grasp_pos.npy"))
        post_grasp = np.load(str(seg_dir / "post_grasp_pos.npy"))
        pre_release = np.load(str(seg_dir / "pre_release_pos.npy"))

        reach_new_start = reach[0] + np.array([0.03, -0.01, 0.00], dtype=np.float32)
        reach_new_goal = pre_grasp
        move_new_start = post_grasp
        move_new_goal = pre_release

        if reach_recon is not None:
            r_model = load_model(str(dmp_dir / "reach_dmp.npz"))
            r_repro = rollout_dmp(r_model, timesteps=len(reach))['y']
            r_adapted = rollout_dmp(r_model, timesteps=len(reach), y0=reach_new_start, goal=reach_new_goal)['y']
            np.save(str(dmp_dir / "reach_adapted.npy"), r_adapted)

        if move_recon is not None and len(move) >= 5:
            m_model = load_model(str(dmp_dir / "move_dmp.npz"))
            m_adapted = rollout_dmp(m_model, timesteps=len(move), y0=move_new_start, goal=move_new_goal)['y']
            np.save(str(dmp_dir / "move_adapted.npy"), m_adapted)
        else:
            np.save(str(dmp_dir / "move_adapted.npy"), np.empty((0, 3), dtype=np.float64))

        adapt_meta = {
            'reach_new_start': reach_new_start.tolist(),
            'reach_new_goal': reach_new_goal.tolist(),
            'move_new_start': move_new_start.tolist(),
            'move_new_goal': move_new_goal.tolist(),
        }
        with open(str(dmp_dir / "adaptation_config.json"), 'w', encoding='utf-8') as f:
            json.dump(adapt_meta, f, indent=2)

    # --- Script 12: Skill Reuse (uses adapted trajectories) ---
    with st.spinner("Generating Skill Reuse trajectory..."):
        reuse_traj = _generate_skill_reuse_dmp(dmp_dir, plots_dir, seg_dir)
    
    # --- Save DMP reconstructed trajectory (backward compat) ---
    recon_parts = []
    if reach_recon is not None:
        recon_parts.append(reach_recon)
    if move_recon is not None:
        recon_parts.append(move_recon)

    if recon_parts:
        full_dmp_traj = np.vstack(recon_parts)
    else:
        full_dmp_traj = np.vstack([p for p in [reach, move] if len(p) > 0]) if (len(reach)+len(move))>0 else np.empty((0,3))

    np.save(str(dmp_dir / "object_xyz_dmp.npy"), full_dmp_traj)

    # --- Use skill_reuse_traj for viewer (matches robot playback) ---
    if reuse_traj is not None and len(reuse_traj) > 0:
        viewer_traj = reuse_traj
        # Compute skill_reuse-space indices from adapted trajectory lengths
        ra_path = dmp_dir / "reach_adapted.npy"
        sr_reach_len = len(np.load(str(ra_path))) if ra_path.exists() else len(reach)
        ma_path = dmp_dir / "move_adapted.npy"
        if ma_path.exists():
            _ma = np.load(str(ma_path))
            sr_move_len = len(_ma) if len(_ma) > 0 else 0
        else:
            sr_move_len = len(move)
        sr_grasp_idx = sr_reach_len           # grasp_pos sits right after reach
        sr_release_idx = sr_reach_len + 1 + sr_move_len  # release_pos after move
    else:
        viewer_traj = full_dmp_traj
        reach_demo = reach_recon if reach_recon is not None else reach
        move_demo = move_recon if move_recon is not None else move
        sr_reach_len = len(reach_demo)
        sr_move_len = len(move_demo)
        sr_grasp_idx = sr_reach_len - 1
        sr_release_idx = sr_reach_len + sr_move_len - 1

    # Compute timestamps synced with video duration
    num_points = len(viewer_traj)
    if session_id:
        rgb_dir = base_path / "frames" / session_id
    else:
        rgb_dir = base_path / "frames"
    video_fps = 30.0
    num_video_frames = len(list(rgb_dir.glob("frame_*.png"))) if rgb_dir.exists() else num_points
    video_duration = num_video_frames / video_fps
    dmp_timestamps = []
    n_pre = min(sr_release_idx + 1, num_points)
    n_post = num_points - n_pre
    remaining_video_frames = max(1, num_video_frames - n_pre)

    for i in range(num_points):
        if i < n_pre:
            dmp_timestamps.append(i / video_fps)
        else:
            post_i = i - n_pre
            frame = n_pre + post_i * remaining_video_frames / max(n_post, 1)
            dmp_timestamps.append(frame / video_fps)

    # Removed debug print

    # --- Build 3D DMP plot with phase-colored segments ---
    fig = go.Figure()

    # Reach segment (blue): indices 0 .. sr_reach_len-1
    reach_slice = viewer_traj[:sr_reach_len]
    if len(reach_slice) > 0:
        fig.add_trace(go.Scatter3d(
            x=reach_slice[:, 0], y=reach_slice[:, 1], z=reach_slice[:, 2],
            mode="lines", line=dict(width=6, color="#4285F4"),
            name="Reach",
        ))
    # Move segment (green): indices grasp_idx+1 .. release_idx-1
    move_slice = viewer_traj[sr_grasp_idx + 1:sr_release_idx]
    if len(move_slice) > 0:
        fig.add_trace(go.Scatter3d(
            x=move_slice[:, 0], y=move_slice[:, 1], z=move_slice[:, 2],
            mode="lines", line=dict(width=6, color="#34A853"),
            name="Move",
        ))
    # Post-release hold (orange): indices release_idx .. end
    if sr_release_idx < len(viewer_traj):
        post_slice = viewer_traj[sr_release_idx:]
        fig.add_trace(go.Scatter3d(
            x=post_slice[:, 0], y=post_slice[:, 1], z=post_slice[:, 2],
            mode="lines", line=dict(width=6, color="#FF9800"),
            name="Post-Release",
        ))

    # Phase transition markers
    markers_x, markers_y, markers_z, markers_text, markers_color = [], [], [], [], []
    if len(viewer_traj) > 0:
        markers_x.append(viewer_traj[0, 0]); markers_y.append(viewer_traj[0, 1]); markers_z.append(viewer_traj[0, 2])
        markers_text.append("Start"); markers_color.append("#00cc96")
        if sr_grasp_idx < len(viewer_traj):
            markers_x.append(viewer_traj[sr_grasp_idx, 0]); markers_y.append(viewer_traj[sr_grasp_idx, 1]); markers_z.append(viewer_traj[sr_grasp_idx, 2])
            markers_text.append("Grasp"); markers_color.append("#111111")
        if sr_release_idx < len(viewer_traj):
            markers_x.append(viewer_traj[sr_release_idx, 0]); markers_y.append(viewer_traj[sr_release_idx, 1]); markers_z.append(viewer_traj[sr_release_idx, 2])
            markers_text.append("Release"); markers_color.append("#ef553b")
        markers_x.append(viewer_traj[-1, 0]); markers_y.append(viewer_traj[-1, 1]); markers_z.append(viewer_traj[-1, 2])
        markers_text.append("End"); markers_color.append("#9C27B0")

    fig.add_trace(go.Scatter3d(
        x=markers_x, y=markers_y, z=markers_z,
        mode="markers+text", marker=dict(size=8, color=markers_color),
        text=markers_text, textposition="top center", name="Phases",
    ))

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        height=550, title="Skill Reuse Trajectory (Reach \u2192 Grasp \u2192 Move \u2192 Release)",
        hovermode="closest",
    )

    st.session_state.step_results["dmp"] = {
        "type": "dmp3d",
        "fig": fig,
        "timestamps": dmp_timestamps,
        "reach_len": sr_reach_len,
        "move_len": sr_move_len,
        "grasp_idx": sr_grasp_idx,
        "release_idx": sr_release_idx,
    }
    return {"reach": reach, "move": move}


# ========================================================================
# STEP 5: ROBOT (Scripts 11 + 12)
# ========================================================================

def handle_bag_robot(
    base_path: Path,
    urdf_path: str = "data/Common/robot_models/openarm/openarm.urdf",
    robot_config: dict = None,
    session_id: str = None
):
    """Step 5: Robot Playback."""
    _init_step_results()

    if robot_config is None:
        robot_config = {}

    _obj_sess = st.session_state.get("active_objects_session_id") or session_id
    if session_id:
        dmp_dir = base_path / "dmp" / _obj_sess if _obj_sess else base_path / "dmp" / session_id
        seg_dir = base_path / "segmentation" / _obj_sess if _obj_sess else base_path / "segmentation" / session_id
    else:
        dmp_dir = base_path / "dmp"
        seg_dir = base_path / "segmentation"

    dmp_xyz_path = dmp_dir / "skill_reuse_traj.npy"
    if not dmp_xyz_path.exists():
         # Fallback to recon if reuse not found?
         dmp_xyz_path = dmp_dir / "object_xyz_dmp.npy"
    
    if not dmp_xyz_path.exists():
        st.error("DMP trajectory (skill_reuse_traj.npy) not found. Run DMP step first.")
        return None

    # --- Load grasp_idx from segmentation (skill_reuse_traj structure) ---
    grasp_idx_robot = None
    release_idx_robot = None
    try:
        reach_path = seg_dir / "reach_traj.npy"
        move_path = seg_dir / "move_traj.npy"
        if reach_path.exists():
            reach_len = len(np.load(str(reach_path)))
            grasp_idx_robot = reach_len  # grasp_pos sits right after reach
            if move_path.exists():
                move_len = len(np.load(str(move_path)))
                # In skill_reuse_traj: [reach, grasp_pos, move, release_pos, post_release]
                release_idx_robot = reach_len + 1 + move_len
    except Exception:
        pass

    from src.streamlit_template.core.Common.robot_playback import (
        dmp_xyz_to_cartesian,
        compute_ik_trajectory,
        cached_meshes_for_pose,
    )

    with st.spinner("Computing robot trajectory..."):
        default_offset = robot_config.get("dmp_offset", [0.4, 0.0, 0.2])
        default_scale = robot_config.get("dmp_scale", [0.5, 0.5, 0.5])
        dmp_rot_z = robot_config.get("dmp_rotation_z", 90.0)
        dmp_flip_z = robot_config.get("flip_z", False)
        dmp_arm_reach = robot_config.get("arm_reach", 0.0)

        all_traj = np.load(str(dmp_xyz_path))
        target_len = len(all_traj)

        cart = dmp_xyz_to_cartesian(
            dmp_npy=str(dmp_xyz_path),
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

    # Compute timestamps synced with video duration
    num_frames = len(q_traj)
    if session_id:
        rgb_dir = base_path / "frames" / session_id
    else:
        rgb_dir = base_path / "frames"
    video_fps = 30.0
    num_video_frames = len(list(rgb_dir.glob("frame_*.png"))) if rgb_dir.exists() else num_frames
    video_duration = num_video_frames / video_fps  # seconds
    
    # Map robot trajectory points synced with video up to release_idx
    frame_timestamps = []
    sr_release_idx = release_idx_robot if release_idx_robot is not None else num_frames - 1
    n_pre = min(sr_release_idx + 1, num_frames)
    n_post = num_frames - n_pre
    remaining_video_frames = max(1, num_video_frames - n_pre)

    for i in range(num_frames):
        if i < n_pre:
            frame_timestamps.append(i / video_fps)
        else:
            post_i = i - n_pre
            frame = n_pre + post_i * remaining_video_frames / max(n_post, 1)
            frame_timestamps.append(frame / video_fps)

    st.session_state.step_results["robot"] = {
        "type": "robot3d",
        "num_frames": num_frames,
        "frame_timestamps": frame_timestamps,
        "q_traj": q_traj,
        "cart_path": cart_path,
        "grasp_idx": grasp_idx_robot,
        "release_idx": release_idx_robot,
    }
    return {"cart_path": cart_path, "q_traj": q_traj, "frame_timestamps": frame_timestamps, "grasp_idx": grasp_idx_robot, "release_idx": release_idx_robot}
