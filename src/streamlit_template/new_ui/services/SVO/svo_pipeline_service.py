"""
SVO Pipeline Service - Wraps SVO scripts 01-13 into 5 pipeline steps.
Mirrors functionality of bag_pipeline_service using SVO-specific core modules.

Data contract matches Generic pipeline_service for UI compatibility:
- handle_svo_hands    → {type: "hands",  paths: [annotated imgs]}
- handle_svo_objects   → {type: "objects", paths: [annotated imgs]}
- handle_svo_trajectory → {type: "trajectory2d", timestamps, x, y, z}
- handle_svo_dmp       → {type: "dmp3d", fig, timestamps}
- handle_svo_robot     → {type: "robot3d", q_traj, frame_timestamps}
"""
import csv
import json
import streamlit as st
from pathlib import Path
import numpy as np
import os
import cv2
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.streamlit_template.new_ui.services.Common.svo_helpers import _annotate_hand_frames, _annotate_object_frames

# Add core SVO scripts to path to import _paper_* modules
# Current file: src/streamlit_template/new_ui/services/SVO/svo_pipeline_service.py
# Core SVO: src/streamlit_template/core/SVO
CORE_SVO_PATH = Path(__file__).resolve().parents[3] / "core" / "SVO"
if str(CORE_SVO_PATH) not in sys.path:
    sys.path.append(str(CORE_SVO_PATH))

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


def _resolve_detector_model(model_path):
    """Resolve detector model with safe fallbacks for official YOLO names."""
    candidate = (model_path or "").strip()
    if candidate:
        candidate_path = Path(candidate)
        if candidate_path.exists() or ("/" not in candidate and "\\" not in candidate):
            return candidate

    for fallback in ("yolov8x.pt", "yolov8n.pt"):
        fallback_path = Path(fallback)
        if fallback_path.exists() or ("/" not in fallback and "\\" not in fallback):
            return fallback

    return "yolov8n.pt"


def _normalize_bbox_xyxy(value):
    """Return [x1, y1, x2, y2] float list when input is valid, else None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except Exception:
        return None


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

    Scans base_path/objects/<base_sess>_1/, _2/, ... for a run_metadata.json
    that matches run_params exactly. Returns that version's id if found.
    Otherwise creates the next version (base_sess_N+1) and returns it.
    If base_sess is None, returns None (legacy/no-session path).
    """
    if not base_sess:
        return base_sess

    objects_root = base_path / "objects"
    best_match = None
    max_n = 0

    # Scan existing versioned dirs
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

    # No match — allocate next version
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

    rgb_files = sorted(
        [f for ext in ["*.png", "*.jpg", "*.jpeg"] for f in rgb_dir.glob(ext)],
        key=lambda p: p.name,
    )
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
        model_path = "data/Common/ai_model/object/best-obb.pt"

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    rgb_files = sorted(
        [f for ext in ["*.png", "*.jpg", "*.jpeg"] for f in rgb_dir.glob(ext)],
        key=lambda p: p.name,
    )
    annotated_paths = []
    total_detections = 0

    for fname in rgb_files:
        bgr = cv2.imread(str(fname))
        results = model(bgr, verbose=False)[0]

        if results.obb is not None and len(results.obb) > 0:
            for box in results.obb:
                cls_id = int(box.cls[0])
                if results.names[cls_id] != "shaft":
                    continue
                total_detections += 1
                xywhr = box.xywhr[0].cpu().numpy()
                cx, cy, w_box, h_box, rotation = xywhr
                rect = ((cx, cy), (w_box, h_box), np.degrees(rotation))
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
                cv2.drawContours(bgr, [box_points], 0, (0, 0, 255), 2)
                conf = float(box.conf[0])
                cv2.putText(bgr, f"shaft {conf:.2f}", (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_path = out_dir / f"objects_{fname.stem}.jpg"
        cv2.imwrite(str(out_path), bgr)
        annotated_paths.append(out_path)

    return annotated_paths, total_detections


# ========================================================================
# STEP 1: HANDS (Scripts 01 + 02)
# ========================================================================

def handle_svo_hands(base_path: Path, session_id: str = None):
    """Step 1: Extract 3D hand trajectory + generate annotated frames for UI."""
    _init_step_results()

    if session_id:
        rgb_dir = base_path / "frames" / session_id
        depth_dir = base_path / "depth_meters" / session_id
        hands_dir = base_path / "hands" / session_id
        plots_dir = base_path / "plots" / session_id
        # Try .npy first (has width/height), then .npz fallback
        intrinsics_path = base_path / "camera" / f"{session_id}.npy"
        if not intrinsics_path.exists():
            intrinsics_path = base_path / "camera" / f"{session_id}.npz"
    else:
        # Legacy/Fallback
        rgb_dir = base_path / "frames"
        depth_dir = base_path / "depth_meters"
        hands_dir = base_path / "hands"
        plots_dir = base_path / "plots"
        if (base_path / "camera_intrinsics.npy").exists():
             intrinsics_path = base_path / "camera_intrinsics.npy"
        elif (base_path / "camera_intrinsics.npz").exists():
             intrinsics_path = base_path / "camera_intrinsics.npz"
        else:
             intrinsics_path = base_path.parent / "camera" / f"{base_path.name}.npy"

    hands_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not intrinsics_path.exists():
        st.error(f"Camera intrinsics not found at {intrinsics_path}. Run SVO extraction first.")
        return None

    # --- Rerun guard: skip if already completed ---
    _smooth_path = hands_dir / "hand_3d_smooth.npy"
    _raw_path = hands_dir / "hand_3d_raw.npy"
    _tips_raw_path = hands_dir / "hand_tips_3d_raw.npy"
    _annotated_dir = hands_dir / "annotated"

    # Case 1: Fully completed — return cached results immediately
    if _smooth_path.exists() and _annotated_dir.exists():
        _cached_paths = sorted(_annotated_dir.glob("*.jpg"))
        if _cached_paths:
            st.session_state.step_results["hands"] = {"type": "hands", "paths": [str(p) for p in _cached_paths]}
            return {
                "trajectory": np.load(str(_raw_path)),
                "smooth": np.load(str(_smooth_path)),
            }

    # Case 2: Detection done but smoothing not done — skip detection, just smooth
    if _raw_path.exists() and _tips_raw_path.exists() and not _smooth_path.exists():
        trajectory_3d = np.load(str(_raw_path))
        tips_np = np.load(str(_tips_raw_path))

        annotated_paths = sorted([str(p) for p in _annotated_dir.glob("*.jpg")]) if _annotated_dir.exists() else []

        with st.spinner("Smoothing hand trajectory (Kalman)..."):
            q, r = (KALMAN_Q, KALMAN_R)
            if KALMAN_AUTO_TUNE:
                q, r = auto_kalman_params(trajectory_3d, base_q=KALMAN_Q, base_r=KALMAN_R)

            traj_smooth = kalman_smooth_3d(trajectory_3d, q=q, r=r)
            np.save(str(_smooth_path), traj_smooth)

            tips_smooth = np.zeros_like(tips_np)
            for j in range(tips_np.shape[1]):
                tips_smooth[:, j, :] = kalman_smooth_3d(tips_np[:, j, :], q=q, r=r)
            np.save(str(hands_dir / "hand_tips_3d_smooth.npy"), tips_smooth)

        st.session_state.step_results["hands"] = {"type": "hands", "paths": annotated_paths}
        return {"trajectory": trajectory_3d, "smooth": traj_smooth}

    if "pipeline_logs" not in st.session_state:
        st.session_state.pipeline_logs = []

    def log(msg, level="info"):
        st.session_state.pipeline_logs.append(f"[{level.upper()}] {msg}")
        if level == "error":
            st.error(msg)
        else:
            st.write(msg)

    # Collect all image files (png, jpg, jpeg)
    rgb_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        rgb_files.extend(list(rgb_dir.glob(ext)))
    rgb_files = sorted(rgb_files, key=lambda p: p.name)

    # log(f"Debug: Found {len(rgb_files)} RGB frames in {rgb_dir}")
    
    if len(rgb_files) == 0:
        log(f"No RGB frames found in {rgb_dir}. Run SVO extraction first.", "error")
        return None

    with st.spinner("Extracting 3D hand trajectory..."):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import re as _re

        # Load intrinsics
        if str(intrinsics_path).endswith('.npz'):
            intr_data = np.load(str(intrinsics_path), allow_pickle=True)
            if 'arr_0' in intr_data:
                intr_raw = intr_data['arr_0'].item()
            else:
                intr_raw = dict(intr_data)
        else:
            intr_raw = np.load(str(intrinsics_path), allow_pickle=True).item()

        # Build frame-number → depth file map
        _frame_num_re = _re.compile(r'frame_(\d+)')
        _depth_by_frame = {}
        for dp in depth_dir.glob('*.npy'):
            m = _frame_num_re.search(dp.name)
            if m:
                _depth_by_frame[int(m.group(1))] = dp

        intr = CameraIntrinsics(
            fx=float(intr_raw["fx"]), fy=float(intr_raw["fy"]),
            cx=float(intr_raw["cx"]), cy=float(intr_raw["cy"]),
            width=int(intr_raw.get("width", -1)), height=int(intr_raw.get("height", -1)),
        )
        # log(f"Debug: Intrinsics loaded. fx={intr.fx}, fy={intr.fy}")
        # log(f"Debug: Depth map has {len(_depth_by_frame)} entries")

        model_path = "data/Common/ai_model/hand/hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands_detector = vision.HandLandmarker.create_from_options(options)

        annotated_dir = hands_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)

        TIP_IDS = [4, 8, 12, 16, 20]
        trajectory_3d = []
        tips_3d = []
        valid_points = 0
        annotated_paths = []

        progress_bar = st.progress(0)

        for idx, fname in enumerate(rgb_files):
            frame_match = _frame_num_re.search(fname.name)
            frame_num = int(frame_match.group(1)) if frame_match else -1
            depth_path = _depth_by_frame.get(frame_num)

            if depth_path is None or not depth_path.exists():
                trajectory_3d.append(trajectory_3d[-1] if trajectory_3d else [0.0, 0.0, 0.0])
                tips_3d.append(tips_3d[-1].copy() if tips_3d else np.full((len(TIP_IDS), 3), np.nan, dtype=np.float32))
                bgr = cv2.imread(str(fname))
                if bgr is not None:
                    out_path = annotated_dir / f"hands_{fname.stem}.jpg"
                    cv2.imwrite(str(out_path), bgr)
                    annotated_paths.append(out_path)
                continue

            bgr = cv2.imread(str(fname))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                depth = np.load(str(depth_path))
            except Exception as e:
                log(f"Failed to load depth {depth_path.name}: {e}", "error")
                return None

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
                    valid_points += 1
                    point_added = True

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
                    trajectory_3d.append(trajectory_3d[-1])
                    tips_3d.append(tips_3d[-1].copy())
                else:
                    trajectory_3d.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                    tips_3d.append(np.full((len(TIP_IDS), 3), np.nan, dtype=np.float32))

            out_path = annotated_dir / f"hands_{fname.stem}.jpg"
            cv2.imwrite(str(out_path), bgr)
            annotated_paths.append(out_path)

            if idx % 10 == 0:
                progress_bar.progress((idx + 1) / len(rgb_files))

        progress_bar.empty()
        # log(f"Debug: Trajectory built. Total frames: {len(trajectory_3d)}. Valid detections: {valid_points}")

        trajectory_3d = np.array(trajectory_3d, dtype=np.float32)
        tips_np = np.asarray(tips_3d, dtype=np.float32)
        np.save(str(hands_dir / "hand_3d_raw.npy"), trajectory_3d)
        np.save(str(hands_dir / "hand_tips_3d_raw.npy"), tips_np)

        if valid_points == 0:
            log("No valid hand detections found.", "error")
            return None

    if len(trajectory_3d) == 0:
        st.error("No 3D hand trajectory found.")
        return None

    with st.spinner("Smoothing hand trajectory (Kalman)..."):
        # --- Script 02 logic ---
        q, r = (KALMAN_Q, KALMAN_R)
        if KALMAN_AUTO_TUNE:
             q, r = auto_kalman_params(trajectory_3d, base_q=KALMAN_Q, base_r=KALMAN_R)

        traj_smooth = kalman_smooth_3d(trajectory_3d, q=q, r=r)
        np.save(str(hands_dir / "hand_3d_smooth.npy"), traj_smooth)

        tips_smooth = np.zeros_like(tips_np)
        for j in range(tips_np.shape[1]):
            tips_smooth[:, j, :] = kalman_smooth_3d(tips_np[:, j, :], q=q, r=r)
        np.save(str(hands_dir / "hand_tips_3d_smooth.npy"), tips_smooth)

    st.session_state.step_results["hands"] = {"type": "hands", "paths": annotated_paths}
    return {"trajectory": trajectory_3d, "smooth": traj_smooth}


# ========================================================================
# STEP 2: OBJECTS (Scripts 03 + 07)
# ========================================================================

def handle_svo_objects(
    base_path: Path,
    session_id: str = None,
    model_path: str = None,
    tracking_bbox_xyxy=None,
    tracking_label: str = None,
    tracking_detection_index: int = None,
    confidence_threshold: float = 0.25,
    max_area_pct: float = 100.0,
    min_area_pct: float = 0.0,
    bbox_size_ratio: float = 4.0,
):
    """Step 2: Detect Objects (YOLO OBB with BoT-SORT tracking) and Smooth."""
    _init_step_results()

    status_ph = st.empty()
    status_ph.info("⏳ Detect Objects: Initializing...")

    if session_id:
        rgb_dir = base_path / "frames" / session_id
        depth_dir = base_path / "depth_meters" / session_id
        intrinsics_path = base_path / "camera" / f"{session_id}.npy"
        if not intrinsics_path.exists():
            intrinsics_path = base_path / "camera" / f"{session_id}.npz"
    else:
        rgb_dir = base_path / "frames"
        depth_dir = base_path / "depth_meters"
        if (base_path / "camera_intrinsics.npy").exists():
            intrinsics_path = base_path / "camera_intrinsics.npy"
        elif (base_path / "camera_intrinsics.npz").exists():
            intrinsics_path = base_path / "camera_intrinsics.npz"
        else:
            intrinsics_path = base_path.parent / "camera" / f"{base_path.name}.npy"

    if not intrinsics_path.exists():
        st.error("Camera intrinsics not found.")
        return None

    _selected_bbox = _normalize_bbox_xyxy(tracking_bbox_xyxy)
    _tracking_mode = _selected_bbox is not None
    try:
        _tracking_detection_index = int(tracking_detection_index) if tracking_detection_index is not None else None
    except Exception:
        _tracking_detection_index = None
    _custom_model_mode = bool((model_path or "").strip())

    # Build the params fingerprint used for cache matching
    _resolved_model_for_meta = model_path.strip() if (model_path or "").strip() else "best-obb.pt"
    _run_params = {
        "model": _resolved_model_for_meta,
        "tracking_label": tracking_label,
        "tracking_detection_index": _tracking_detection_index,
        "bbox": _selected_bbox,
        "confidence_threshold": round(float(confidence_threshold), 4),
        "max_area_pct": round(float(max_area_pct), 2),
        "min_area_pct": round(float(min_area_pct), 4),
        "bbox_size_ratio": round(float(bbox_size_ratio), 2),
        "tracking_strategy": "obb-track-id-motion-gated-v4" if _tracking_mode else "auto-best",
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
    _obj_annotated_dir = objects_dir / "annotated"
    _obj_raw_path = objects_dir / "object_3d_raw.npy"
    if _obj_smooth_path.exists() and _obj_annotated_dir.exists() and _obj_raw_path.exists():
        _cached_paths = sorted(_obj_annotated_dir.glob("*.jpg"))
        if _cached_paths and _objects_run_params_match(objects_dir, _run_params):
            _obj_raw = np.load(str(_obj_raw_path))
            _obj_smooth = np.load(str(_obj_smooth_path))
            _saved_meta = json.load(open(str(objects_dir / "run_metadata.json"), encoding="utf-8"))
            _cached_total_det = _saved_meta.get("total_detections", len(_cached_paths))
            st.session_state.step_results["objects"] = {
                "type": "objects",
                "paths": [str(p) for p in _cached_paths],
                "total_detections": _cached_total_det,
                "model_used": _saved_meta.get("model"),
                "selection_mode": "local-selected" if _saved_meta.get("bbox") else "auto-best",
                "run_metadata": _saved_meta,
                "objects_session_id": _obj_sess,
            }
            status_ph.empty()
            if _cached_total_det == 0:
                st.warning(
                    "⚠️ Cached run had 0 detections — trajectory is unreliable. "
                    "Change the model or settings to trigger a fresh run."
                )
                return None
            _cached_coverage = _saved_meta.get("detection_coverage_pct")
            _cached_longest_gap = _saved_meta.get("longest_missing_run")
            if _cached_coverage is not None and float(_cached_coverage) < 70.0:
                st.warning(
                    f"Selected object coverage is low ({float(_cached_coverage):.1f}%). "
                    f"Longest held/missing run: {_cached_longest_gap} frames. "
                    "Review selected_detection_debug.csv before trusting pick/release."
                )
            return {"trajectory": _obj_raw, "smooth": _obj_smooth}

    # Collect all image files (png, jpg, jpeg)
    rgb_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        rgb_files.extend(list(rgb_dir.glob(ext)))
    rgb_files = sorted(rgb_files, key=lambda p: p.name)

    if len(rgb_files) == 0:
        st.error(f"No RGB frames found in {rgb_dir}. Run SVO extraction first.")
        return None

    with st.spinner("Extracting 3D object trajectory..."):
        from ultralytics import YOLO
        import re as _re

        # Load intrinsics — handle both .npy and .npz formats
        if str(intrinsics_path).endswith('.npz'):
            intr_data = np.load(str(intrinsics_path), allow_pickle=True)
            if 'arr_0' in intr_data:
                intr_raw = intr_data['arr_0'].item()
            else:
                intr_raw = dict(intr_data)
        else:
            intr_raw = np.load(str(intrinsics_path), allow_pickle=True).item()

        intr = CameraIntrinsics(
            fx=float(intr_raw["fx"]), fy=float(intr_raw["fy"]),
            cx=float(intr_raw["cx"]), cy=float(intr_raw["cy"]),
            width=int(intr_raw.get("width", -1)), height=int(intr_raw.get("height", -1)),
        )

        # Build frame-number → depth file map
        _frame_num_re = _re.compile(r'frame_(\d+)')
        _depth_by_frame = {}
        for dp in depth_dir.glob('*.npy'):
            m = _frame_num_re.search(dp.name)
            if m:
                _depth_by_frame[int(m.group(1))] = dp

        _conf = max(0.01, min(0.95, float(confidence_threshold)))
        _max_area = max(0.1, min(100.0, float(max_area_pct)))
        _min_area = max(0.0, min(_max_area, float(min_area_pct)))

        # Choose model: respect explicit model path first; otherwise use best-obb.pt
        if _custom_model_mode:
            _model_in_use = _resolve_detector_model(model_path)
            model = YOLO(_model_in_use)
            try:
                _use_obb = model.task == "obb"
            except Exception:
                _use_obb = "obb" in Path(_model_in_use).stem.lower()
        else:
            _obb_model_path = Path("data/Common/ai_model/object/best-obb.pt")
            _use_obb = _obb_model_path.exists()
            if _use_obb:
                model = YOLO(str(_obb_model_path))
                _model_in_use = str(_obb_model_path)
            else:
                model = YOLO("yolov8n.pt")
                _model_in_use = "yolov8n.pt"

        _target_center = None
        _target_area = None
        _target_w = None
        _target_h = None
        _target_diag = None
        _last_detection_frame = None
        _matched_obb_track_id = None
        _obb_track_id_frames = 0
        _obb_rejected_jumps = 0
        _valid_detection_mask = []
        _selected_debug_rows = []
        if _tracking_mode:
            _target_center = [
                (_selected_bbox[0] + _selected_bbox[2]) / 2.0,
                (_selected_bbox[1] + _selected_bbox[3]) / 2.0,
            ]
            _target_w = max(1.0, _selected_bbox[2] - _selected_bbox[0])
            _target_h = max(1.0, _selected_bbox[3] - _selected_bbox[1])
            _target_area = _target_w * _target_h
            _target_diag = float(np.hypot(_target_w, _target_h))
        _max_reacquire_gap_frames = 12
        _short_gap_jump_px = max(35.0, 2.5 * float(_target_diag or 1.0))
        _long_gap_jump_px = max(45.0, 1.35 * float(_target_diag or 1.0))

        trajectory_3d = []

        annotated_dir = objects_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_paths = [annotated_dir / f"objects_{f.stem}.jpg" for f in rgb_files]
        total_det = 0

        # _draw_data: per-frame draw instructions for background rendering (Pass 2)
        _draw_data = []
        _progress_bar = st.progress(0)
        _total_frames = len(rgb_files)

        # --- Pass 1 (blocking): YOLO inference + XYZ extraction only ---
        for _fidx, fname in enumerate(rgb_files):
            status_ph.info(f"⏳ Detect Objects: frame {_fidx + 1} / {_total_frames} ({total_det} detections so far)...")
            _progress_bar.progress((_fidx + 1) / _total_frames)
            frame_match = _frame_num_re.search(fname.name)
            frame_num = int(frame_match.group(1)) if frame_match else -1
            depth_path = _depth_by_frame.get(frame_num)

            bgr_p1 = cv2.imread(str(fname))
            if bgr_p1 is None:
                # Frame file missing or unreadable (e.g. failed/partial download).
                # Hold the previous detection so the rest of the pipeline stays
                # aligned; avoids "NoneType has no attribute 'shape'" crash and
                # the "Not supported for the square ROI" error from YOLO preprocessing.
                logger.warning(f"handle_svo_objects: could not read frame {fname} — skipping (holding previous)")
                _valid_detection_mask.append(False)
                _selected_debug_rows.append({
                    "frame_idx": _fidx, "frame_num": frame_num, "image_file": fname.name,
                    "detected": 0, "held_previous": 1 if trajectory_3d else 0,
                    "reason": "unreadable_frame", "candidate_count": 0,
                    "label_candidate_count": 0, "selected_conf": "", "selected_label": "",
                    "selected_track_id": "", "cx": "", "cy": "", "w": "", "h": "",
                    "z_m": "", "gap_from_last_detection": "",
                })
                trajectory_3d.append(trajectory_3d[-1] if trajectory_3d else [0, 0, 0])
                _draw_data.append(None)
                continue
            if depth_path and depth_path.exists():
                depth = np.load(str(depth_path))
            else:
                depth = None

            detected = False
            _frame_draw = None
            _debug_row = {
                "frame_idx": _fidx,
                "frame_num": frame_num,
                "image_file": fname.name,
                "detected": 0,
                "held_previous": 0,
                "reason": "no_detection",
                "candidate_count": 0,
                "label_candidate_count": 0,
                "selected_conf": "",
                "selected_label": "",
                "selected_track_id": "",
                "cx": "",
                "cy": "",
                "w": "",
                "h": "",
                "z_m": "",
                "gap_from_last_detection": "",
            }

            if _use_obb:
                try:
                    if _tracking_mode:
                        _obb_prediction = model.track(
                            bgr_p1,
                            persist=True,
                            conf=_conf,
                            imgsz=1280,
                            max_det=100,
                            verbose=False,
                        )
                    else:
                        _obb_prediction = model(bgr_p1, verbose=False, conf=_conf, imgsz=1280, max_det=100)
                    results = _obb_prediction[0]
                except Exception:
                    results = model(bgr_p1, verbose=False, conf=_conf, imgsz=1280, max_det=100)[0]
                _obb_names = results.names or {}
                shaft_boxes = []
                _ranked_frame_detections = []
                if results.obb is not None and len(results.obb) > 0:
                    _all_confs = results.obb.conf.cpu().numpy()
                    _all_xywhr = results.obb.xywhr.cpu().numpy()
                    _frame_h_obb, _frame_w_obb = bgr_p1.shape[:2]
                    _frame_area_obb = float(_frame_h_obb * _frame_w_obb)
                    for i, cls in enumerate(results.obb.cls.cpu().numpy().astype(int)):
                        cls_name = _obb_names.get(cls, "")
                        _det_area_rank = float(_all_xywhr[i][2]) * float(_all_xywhr[i][3])
                        if _det_area_rank <= (_frame_area_obb * (_max_area / 100.0)) and _det_area_rank >= (_frame_area_obb * (_min_area / 100.0)):
                            _ranked_frame_detections.append((i, float(_all_confs[i]), cls_name))
                        if _tracking_mode:
                            if tracking_label and cls_name and cls_name != tracking_label:
                                continue
                            shaft_boxes.append(i)
                        elif cls_name == "shaft":
                            shaft_boxes.append(i)
                    _ranked_frame_detections.sort(key=lambda item: item[1], reverse=True)
                    _debug_row["candidate_count"] = len(_ranked_frame_detections)
                    _debug_row["label_candidate_count"] = len(shaft_boxes)

                if shaft_boxes:
                    confs = results.obb.conf.cpu().numpy()
                    _shaft_xywhr = results.obb.xywhr.cpu().numpy()
                    _obb_track_ids = None
                    try:
                        _obb_ids = getattr(results.obb, "id", None)
                        if _obb_ids is not None:
                            _obb_track_ids = _obb_ids.int().cpu().numpy()
                    except Exception:
                        _obb_track_ids = None

                    def _axis_bbox_from_obb(_xywhr):
                        _rect = (
                            (float(_xywhr[0]), float(_xywhr[1])),
                            (float(_xywhr[2]), float(_xywhr[3])),
                            float(np.degrees(_xywhr[4])),
                        )
                        _pts = cv2.boxPoints(_rect)
                        return [
                            float(np.min(_pts[:, 0])),
                            float(np.min(_pts[:, 1])),
                            float(np.max(_pts[:, 0])),
                            float(np.max(_pts[:, 1])),
                        ]

                    def _size_matches(_si):
                        if _target_area is None or _target_area <= 1.0:
                            return True
                        _det_area = float(_shaft_xywhr[_si][2]) * float(_shaft_xywhr[_si][3])
                        if _det_area <= 0:
                            return False
                        _ratio = _det_area / _target_area
                        return 1.0 / bbox_size_ratio <= _ratio <= bbox_size_ratio

                    def _movement_ok(_si):
                        if not (_tracking_mode and _target_center is not None):
                            return True
                        tx, ty = _target_center
                        _dist = float(np.hypot(_shaft_xywhr[_si][0] - tx, _shaft_xywhr[_si][1] - ty))
                        _frame_gap = (_fidx - _last_detection_frame) if _last_detection_frame is not None else 1
                        if _frame_gap > _max_reacquire_gap_frames:
                            _max_jump = _long_gap_jump_px
                        else:
                            _max_jump = _short_gap_jump_px * min(2.0, max(1.0, float(_frame_gap)))
                        return _dist <= _max_jump

                    best_idx = None
                    _anchor_idx = None
                    if _tracking_mode and _fidx == 0 and _tracking_detection_index is not None:
                        if 0 <= _tracking_detection_index < len(_ranked_frame_detections):
                            _candidate_anchor = int(_ranked_frame_detections[_tracking_detection_index][0])
                            _candidate_label = _ranked_frame_detections[_tracking_detection_index][2]
                            if _candidate_anchor in shaft_boxes and not (tracking_label and _candidate_label and _candidate_label != tracking_label):
                                _anchor_idx = _candidate_anchor

                    if _anchor_idx is not None:
                        best_idx = _anchor_idx
                        if _obb_track_ids is not None:
                            _matched_obb_track_id = int(_obb_track_ids[_anchor_idx])
                    elif _tracking_mode and _target_center is not None and _obb_track_ids is not None:
                        _candidate_pool = [_si for _si in shaft_boxes if _size_matches(_si)]
                        if _matched_obb_track_id is None and _candidate_pool:
                            sx1, sy1, sx2, sy2 = [float(v) for v in _selected_bbox]
                            _sel_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
                            _best_iou = 0.0
                            _best_match = None
                            for _si in _candidate_pool:
                                dx1, dy1, dx2, dy2 = _axis_bbox_from_obb(_shaft_xywhr[_si])
                                ix1 = max(sx1, dx1)
                                iy1 = max(sy1, dy1)
                                ix2 = min(sx2, dx2)
                                iy2 = min(sy2, dy2)
                                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                                det_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
                                union = _sel_area + det_area - inter
                                iou = inter / union if union > 0 else 0.0
                                if iou > _best_iou:
                                    _best_iou = iou
                                    _best_match = _si
                            if _best_match is None or _best_iou <= 0.01:
                                tx, ty = _target_center
                                _best_match = min(
                                    _candidate_pool,
                                    key=lambda _si: float(np.hypot(_shaft_xywhr[_si][0] - tx, _shaft_xywhr[_si][1] - ty)),
                                )
                            _matched_obb_track_id = int(_obb_track_ids[_best_match])

                        if _matched_obb_track_id is not None:
                            for _si in shaft_boxes:
                                if int(_obb_track_ids[_si]) == _matched_obb_track_id and _size_matches(_si) and _movement_ok(_si):
                                    best_idx = _si
                                    _obb_track_id_frames += 1
                                    break
                    elif _tracking_mode and _target_center is not None:
                        tx, ty = _target_center
                        _size_ok = [_si for _si in shaft_boxes if _size_matches(_si)]
                        if _size_ok:
                            _distances = [
                                float(np.hypot(_shaft_xywhr[_si][0] - tx, _shaft_xywhr[_si][1] - ty))
                                for _si in _size_ok
                            ]
                            _best_pos = int(np.argmin(_distances))
                            _best_dist = _distances[_best_pos]
                            _frame_gap = (_fidx - _last_detection_frame) if _last_detection_frame is not None else 1
                            if _frame_gap > _max_reacquire_gap_frames:
                                _max_jump = _long_gap_jump_px
                            else:
                                _max_jump = _short_gap_jump_px * min(2.0, max(1.0, float(_frame_gap)))
                            if _best_dist <= _max_jump:
                                best_idx = _size_ok[_best_pos]
                            else:
                                _obb_rejected_jumps += 1
                                _debug_row["reason"] = f"movement_gate_rejected_dist={_best_dist:.1f}_max={_max_jump:.1f}_gap={_frame_gap}"
                    else:
                        best_idx = shaft_boxes[int(np.argmax(confs[shaft_boxes]))]

                    if best_idx is not None:
                        xywhr = results.obb.xywhr[best_idx].cpu().numpy()
                        cx_b, cy_b = float(xywhr[0]), float(xywhr[1])
                        u_i, v_i = int(round(cx_b)), int(round(cy_b))
                        z = median_valid_depth(depth, u=u_i, v=v_i, half=2)
                        _debug_row.update({
                            "reason": "selected",
                            "selected_conf": f"{float(confs[best_idx]):.6f}",
                            "selected_label": tracking_label or "",
                            "cx": f"{cx_b:.3f}",
                            "cy": f"{cy_b:.3f}",
                            "w": f"{float(xywhr[2]):.3f}",
                            "h": f"{float(xywhr[3]):.3f}",
                            "gap_from_last_detection": (
                                str(_fidx - _last_detection_frame)
                                if _last_detection_frame is not None else "first"
                            ),
                        })
                        if _obb_track_ids is not None:
                            _debug_row["selected_track_id"] = str(int(_obb_track_ids[best_idx]))
                        if z is not None:
                            trajectory_3d.append(to_camera_xyz(u_i, v_i, z, intr))
                            detected = True
                            total_det += 1
                            _target_center = [cx_b, cy_b]
                            _last_detection_frame = _fidx
                            _debug_row["detected"] = 1
                            _debug_row["z_m"] = f"{float(z):.6f}"
                        else:
                            _debug_row["reason"] = "selected_but_missing_depth"
                        _frame_draw = {"type": "obb", "xywhr": xywhr.tolist(), "conf": float(confs[best_idx])}
                    elif _debug_row["reason"] == "no_detection":
                        _debug_row["reason"] = "no_selected_candidate_after_label_size_motion_gate"
                elif _debug_row["reason"] == "no_detection":
                    _debug_row["reason"] = "no_obb_candidates_after_label_area_filter"
            else:
                # Non-OBB model path
                results = model(bgr_p1, verbose=False, conf=_conf, imgsz=1280,
                                max_det=100, agnostic_nms=True)[0]
                _selected = None
                if results.boxes is not None and len(results.boxes) > 0:
                    _xyxy = results.boxes.xyxy.cpu().numpy()
                    _xywh = results.boxes.xywh.cpu().numpy()
                    _confs = results.boxes.conf.cpu().numpy()
                    _cls_ids_b = results.boxes.cls.cpu().numpy().astype(int) if results.boxes.cls is not None else np.zeros(len(_xyxy), dtype=int)
                    _det_names_b = results.names or {}
                    if isinstance(_det_names_b, list):
                        _det_names_b = {i: n for i, n in enumerate(_det_names_b)}
                    _frame_h, _frame_w = bgr_p1.shape[:2]
                    _frame_area = float(_frame_h * _frame_w)
                    _candidates = []
                    for _i in range(len(_xyxy)):
                        x1, y1, x2, y2 = [float(v) for v in _xyxy[_i].tolist()]
                        _w_box = max(0.0, x2 - x1)
                        _h_box = max(0.0, y2 - y1)
                        _area = _w_box * _h_box
                        if _area > (_frame_area * (_max_area / 100.0)):
                            continue
                        if _area < (_frame_area * (_min_area / 100.0)):
                            continue
                        _cname_b = _det_names_b.get(int(_cls_ids_b[_i]), "")
                        _candidates.append({
                            "xyxy": [x1, y1, x2, y2],
                            "xywh": _xywh[_i],
                            "conf": float(_confs[_i]),
                            "cx": float(_xywh[_i][0]),
                            "cy": float(_xywh[_i][1]),
                            "cls_name": _cname_b,
                            "area": _area,
                        })
                    _debug_row["candidate_count"] = len(_candidates)
                    if _candidates:
                        if _tracking_mode and _target_center is not None:
                            _label_ok = [c for c in _candidates if not (tracking_label and c["cls_name"] and c["cls_name"] != tracking_label)]
                            _debug_row["label_candidate_count"] = len(_label_ok)
                            _size_ok = [c for c in _label_ok if _target_area is None or _target_area <= 1.0 or (c["area"] > 0 and 1.0 / bbox_size_ratio <= c["area"] / _target_area <= bbox_size_ratio)]
                            _pool = _size_ok if _size_ok else (_label_ok if not (_target_area and _target_area > 1.0) else [])
                            if _pool:
                                tx, ty = _target_center
                                _selected = min(_pool, key=lambda c: (c["cx"] - tx) ** 2 + (c["cy"] - ty) ** 2)
                                _target_center = [_selected["cx"], _selected["cy"]]
                        else:
                            _selected = max(_candidates, key=lambda c: c["conf"])

                        if _selected is not None:
                            xywh = _selected["xywh"]
                            u_i, v_i = int(round(float(xywh[0]))), int(round(float(xywh[1])))
                            if depth is not None:
                                z = median_valid_depth(depth, u=u_i, v=v_i, half=2)
                                if z is not None:
                                    trajectory_3d.append(to_camera_xyz(u_i, v_i, z, intr))
                                    detected = True
                                    _debug_row["detected"] = 1
                                    _debug_row["z_m"] = f"{float(z):.6f}"
                                    _target_center = [float(xywh[0]), float(xywh[1])]
                                    _last_detection_frame = _fidx
                                else:
                                    _debug_row["reason"] = "selected_but_missing_depth"
                            else:
                                _debug_row["reason"] = "selected_but_no_depth_frame"
                            total_det += 1
                            _debug_row.update({
                                "reason": "selected" if detected else _debug_row["reason"],
                                "selected_conf": f"{float(_selected['conf']):.6f}",
                                "selected_label": str(_selected.get("cls_name") or tracking_label or ""),
                                "cx": f"{float(xywh[0]):.3f}",
                                "cy": f"{float(xywh[1]):.3f}",
                                "w": f"{float(xywh[2]):.3f}",
                                "h": f"{float(xywh[3]):.3f}",
                                "gap_from_last_detection": (
                                    str(_fidx - _last_detection_frame)
                                    if _last_detection_frame is not None else "first"
                                ),
                            })
                            _frame_draw = {"type": "box", "xyxy": _selected["xyxy"], "conf": _selected["conf"], "tracking": _tracking_mode}

            if not detected:
                _debug_row["held_previous"] = 1 if trajectory_3d else 0
                trajectory_3d.append(trajectory_3d[-1] if trajectory_3d else [0, 0, 0])

            _valid_detection_mask.append(bool(detected))
            _selected_debug_rows.append(_debug_row)
            _draw_data.append(_frame_draw)

        _progress_bar.empty()
        status_ph.info(f"✅ Object detection done — {total_det} detections across {_total_frames} frames. Generating bbox images in background...")

        trajectory_3d = np.array(trajectory_3d)
        np.save(str(objects_dir / "object_3d_raw.npy"), trajectory_3d)
        _valid_mask_arr = np.asarray(_valid_detection_mask, dtype=bool)
        np.save(str(objects_dir / "object_detection_valid_mask.npy"), _valid_mask_arr)
        if _selected_debug_rows:
            _debug_csv_path = objects_dir / "selected_detection_debug.csv"
            with _debug_csv_path.open("w", newline="", encoding="utf-8") as _fdbg:
                _writer = csv.DictWriter(_fdbg, fieldnames=list(_selected_debug_rows[0].keys()))
                _writer.writeheader()
                _writer.writerows(_selected_debug_rows)

        # --- Pass 2 (background daemon): draw stored detection geometry on frames ---
        def _write_annotated_frames(frame_paths, draw_data, out_paths):
            import cv2 as _cv2_bg
            import numpy as _np_bg
            for _fpath, _dd, _opath in zip(frame_paths, draw_data, out_paths):
                _bgr = _cv2_bg.imread(str(_fpath))
                if _bgr is None:
                    continue
                if _dd is not None:
                    if _dd["type"] == "obb":
                        _xywhr = _np_bg.array(_dd["xywhr"])
                        _cx, _cy, _w, _h, _rot = _xywhr
                        _rect = ((float(_cx), float(_cy)), (float(_w), float(_h)), float(_np_bg.degrees(_rot)))
                        _bp = _cv2_bg.boxPoints(_rect)
                        _bp = _np_bg.int0(_bp)
                        _cv2_bg.drawContours(_bgr, [_bp], 0, (0, 0, 255), 2)
                        _cv2_bg.putText(_bgr, f"{_dd['conf']:.2f}", (int(_cx), int(_cy)),
                                        _cv2_bg.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    elif _dd["type"] == "box":
                        _x1, _y1, _x2, _y2 = [int(round(v)) for v in _dd["xyxy"]]
                        _color = (0, 0, 255) if _dd["tracking"] else (0, 200, 0)
                        _cv2_bg.rectangle(_bgr, (_x1, _y1), (_x2, _y2), _color, 2)
                        _cv2_bg.putText(_bgr, f"obj {_dd['conf']:.2f}", (_x1, max(_y1 - 6, 10)),
                                        _cv2_bg.FONT_HERSHEY_SIMPLEX, 0.5, _color, 1)
                _cv2_bg.imwrite(str(_opath), _bgr)

        import threading as _threading
        _threading.Thread(
            target=_write_annotated_frames,
            args=(rgb_files, _draw_data, annotated_paths),
            daemon=True,
        ).start()

    if len(trajectory_3d) == 0:
        st.error("No 3D object trajectory extracted.")
        return None

    if total_det == 0:
        st.warning(
            "⚠️ Object detection produced 0 detections for this video. "
            "The trajectory is entirely interpolated from a zeroed starting point and will give unreliable grasp/release results. "
            "Try a different model, lower the confidence threshold, or check that the tracking label matches the model's class names."
        )
        return None

    with st.spinner("Smoothing object trajectory (Kalman)..."):
        q, r = (KALMAN_Q, KALMAN_R)
        if KALMAN_AUTO_TUNE:
            q, r = auto_kalman_params(trajectory_3d, base_q=KALMAN_Q, base_r=KALMAN_R)
        smooth = kalman_smooth_3d(trajectory_3d, q=q, r=r)
        np.save(str(objects_dir / "object_3d_smooth.npy"), smooth)

    _valid_mask_arr = np.asarray(_valid_detection_mask, dtype=bool)
    _longest_missing_run = 0
    _current_missing_run = 0
    for _is_valid in _valid_mask_arr.tolist():
        if _is_valid:
            _longest_missing_run = max(_longest_missing_run, _current_missing_run)
            _current_missing_run = 0
        else:
            _current_missing_run += 1
    _longest_missing_run = max(_longest_missing_run, _current_missing_run)
    _held_frames = int(len(_valid_mask_arr) - int(np.count_nonzero(_valid_mask_arr)))
    _coverage_pct = (100.0 * float(np.count_nonzero(_valid_mask_arr)) / float(len(_valid_mask_arr))) if len(_valid_mask_arr) else 0.0

    _save_objects_run_metadata(objects_dir, _run_params, {
        "total_detections": total_det,
        "frame_count": len(annotated_paths),
        "rejected_tracking_jumps": _obb_rejected_jumps,
        "valid_detection_frames": int(np.count_nonzero(_valid_mask_arr)),
        "held_previous_frames": _held_frames,
        "longest_missing_run": int(_longest_missing_run),
        "detection_coverage_pct": round(_coverage_pct, 2),
        "valid_mask": str((objects_dir / "object_detection_valid_mask.npy").resolve()),
        "debug_csv": str((objects_dir / "selected_detection_debug.csv").resolve()),
    })
    _saved_meta = json.load(open(str(objects_dir / "run_metadata.json"), encoding="utf-8")) if (objects_dir / "run_metadata.json").exists() else {}
    if _coverage_pct < 70.0:
        st.warning(
            f"Selected object coverage is low ({_coverage_pct:.1f}%). "
            f"Longest held/missing run: {_longest_missing_run} frames. "
            "The tracker did not switch to auto-best, but pick/release frames inside held ranges are unreliable."
        )

    st.session_state.step_results["objects"] = {
        "type": "objects",
        "paths": annotated_paths,
        "total_detections": total_det,
        "model_used": _model_in_use,
        "selection_mode": "local-selected" if _tracking_mode else "auto-best",
        "run_metadata": _saved_meta,
        "objects_session_id": _obj_sess,
    }
    return {"trajectory": trajectory_3d, "smooth": smooth}


# ========================================================================
# STEP 3: TRAJECTORY (Scripts 04 + 05 + 08 + 09)
# ========================================================================

def handle_svo_trajectory(base_path: Path, session_id: str = None):
    """Step 3: Trajectory Reconstruction & Skill Extraction."""
    _init_step_results()

    status_ph = st.empty()
    status_ph.info("⏳ Trajectory: Initializing...")

    # Route object/seg/dmp dirs through active versioned objects session
    _obj_sess = st.session_state.get("active_objects_session_id") or session_id
    if session_id:
        frames_dir = base_path / "frames" / session_id
        hands_dir = base_path / "hands" / session_id
        objects_dir = base_path / "objects" / _obj_sess if _obj_sess else base_path / "objects" / session_id
        seg_dir = base_path / "segmentation" / _obj_sess if _obj_sess else base_path / "segmentation" / session_id
        plots_dir = base_path / "plots" / _obj_sess if _obj_sess else base_path / "plots" / session_id
    else:
        frames_dir = base_path / "frames"
        hands_dir = base_path / "hands"
        objects_dir = base_path / "objects"
        seg_dir = base_path / "segmentation"
        plots_dir = base_path / "plots"

    seg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    hand_path = hands_dir / "hand_3d_smooth.npy"
    obj_path = objects_dir / "object_3d_smooth.npy"

    if not hand_path.exists():
        st.error(f"Missing hand trajectory file: {hand_path.name}")
        return None
    if not obj_path.exists():
        st.error(f"Missing object trajectory file: {obj_path.name}")
        return None

    hand_smooth = np.load(str(hand_path))
    obj_smooth = np.load(str(obj_path))
    obj_valid_mask = None
    valid_mask_path = objects_dir / "object_detection_valid_mask.npy"
    if valid_mask_path.exists():
        try:
            obj_valid_mask = np.load(str(valid_mask_path)).astype(bool)
        except Exception:
            obj_valid_mask = None

    # --- Script 04: Grasp estimation (paper Algorithm 1) ---
    with st.spinner("Estimating grasp point..."):
        from scipy.ndimage import uniform_filter1d as _uf1d

        tips_path = hands_dir / "hand_tips_3d_smooth.npy"
        if not tips_path.exists():
            st.error("hand_tips_3d_smooth.npy not found — re-run Step 1 (Hands).")
            return None
        hand_tips = np.load(str(tips_path))  # (T, 5, 3)

        t_len = min(len(hand_tips), len(obj_smooth))
        hand_tips = hand_tips[:t_len]
        obj_smooth_clipped = obj_smooth[:t_len]

        # Estimate object's initial (resting) position — skip leading zero/non-finite frames
        _nonzero_mask = np.any(obj_smooth_clipped != 0, axis=1) & np.isfinite(obj_smooth_clipped).all(axis=1)
        _first_real = int(np.argmax(_nonzero_mask)) if np.any(_nonzero_mask) else 0
        n_init = max(5, int(t_len * 0.10))
        init_seg = obj_smooth_clipped[_first_real : _first_real + n_init]
        finite_init = np.isfinite(init_seg).all(axis=1)
        initial_position = init_seg[finite_init].mean(axis=0) if np.any(finite_init) else obj_smooth_clipped[_first_real]

        distances = np.full(t_len, np.inf, dtype=np.float32)
        for i in range(t_len):
            tips_frame = hand_tips[i]
            valid = np.isfinite(tips_frame).all(axis=1)
            if np.any(valid):
                distances[i] = float(np.min(np.linalg.norm(tips_frame[valid] - initial_position, axis=1)))
            elif np.isfinite(hand_smooth[i]).all():
                distances[i] = float(np.linalg.norm(hand_smooth[i] - initial_position))

        finite = np.isfinite(distances)
        if not np.any(finite):
            st.error("No valid hand-to-object distances found.")
            return None
        distances[~finite] = float(np.nanmax(distances[finite]))

        distances_smooth = _uf1d(distances.astype(np.float64), size=5).astype(np.float32)
        _dist_thresh = max(0.03, float(np.percentile(distances_smooth[finite], 5)))

        grasp_idx = int(np.argmin(distances_smooth))
        for i in range(1, t_len - 1):
            if distances_smooth[i] <= _dist_thresh:
                if distances_smooth[i] <= distances_smooth[i - 1] and distances_smooth[i] <= distances_smooth[i + 1]:
                    grasp_idx = i
                    break
                if i >= 2 and distances_smooth[i] <= distances_smooth[i - 2] * 1.1:
                    grasp_idx = i
                    break

        if grasp_idx == 0:
            dist_current = np.full(t_len, np.inf, dtype=np.float32)
            for i in range(t_len):
                tips_frame = hand_tips[i]
                valid = np.isfinite(tips_frame).all(axis=1)
                if np.any(valid):
                    dist_current[i] = float(np.min(np.linalg.norm(tips_frame[valid] - obj_smooth_clipped[i], axis=1)))
            fc = np.isfinite(dist_current)
            if np.any(fc):
                dist_current[~fc] = float(np.nanmax(dist_current[fc]))
                grasp_idx = int(np.argmin(_uf1d(dist_current.astype(np.float64), size=5)))

        for i in range(max(0, grasp_idx - GRASP_STABLE_WINDOW), grasp_idx + 1):
            end = i + GRASP_STABLE_WINDOW
            if end <= len(distances_smooth) and np.all(distances_smooth[i:end] <= distances_smooth[grasp_idx] * 1.5):
                grasp_idx = i
                break

        grasp_idx = int(np.clip(grasp_idx, 0, t_len - 1))
        np.save(str(seg_dir / "min_hand_initial_distance.npy"), distances)
        np.save(str(seg_dir / "grasp_idx.npy"), np.array(grasp_idx, dtype=np.int32))

    # --- Script 05: Reconstruct trajectory (hand-driven from frame 0) ---
    with st.spinner("Reconstructing object trajectory..."):
        hand_traj = hand_smooth
        obj_traj = obj_smooth
        T = min(len(obj_traj), len(hand_traj))
        grasp_idx = int(np.clip(grasp_idx, 0, T - 1))
        reconstructed = hand_traj[:T].copy()
        segmentation_traj = obj_traj[:T].copy()
        reconstruction_source = np.full(T, 2, dtype=np.uint8)
        segmentation_source = np.zeros(T, dtype=np.uint8)

        for i in range(T):
            has_valid_object_detection = (
                obj_valid_mask is not None
                and i < len(obj_valid_mask)
                and bool(obj_valid_mask[i])
                and np.isfinite(obj_traj[i]).all()
                and np.any(obj_traj[i] != 0)
            )
            if has_valid_object_detection:
                segmentation_traj[i] = obj_traj[i]
                segmentation_source[i] = 1
            else:
                segmentation_traj[i] = reconstructed[i]
                segmentation_source[i] = 2

        np.save(str(objects_dir / "object_3d_reconstructed.npy"), reconstructed)
        np.save(str(objects_dir / "object_3d_segmentation.npy"), segmentation_traj)
        np.save(str(objects_dir / "object_3d_reconstruction_source.npy"), reconstruction_source)
        np.save(str(objects_dir / "object_3d_segmentation_source.npy"), segmentation_source)
        with open(str(objects_dir / "object_3d_segmentation_source.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "0": "unused",
                    "1": "selected_object_detection",
                    "2": "hand_fallback",
                    "object_detection_frames": int(np.count_nonzero(segmentation_source == 1)),
                    "hand_fallback_frames": int(np.count_nonzero(segmentation_source == 2)),
                    "grasp_idx": int(grasp_idx),
                },
                f, indent=2,
            )

    # --- Script 08: GMM segmentation on segmentation_traj (not reconstructed) ---
    with st.spinner("Running GMM segmentation (release detection)..."):
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="full", random_state=0, n_init=5)
        labels = gmm.fit_predict(segmentation_traj)

        np.save(str(seg_dir / "gmm_labels.npy"), labels)
        np.save(str(seg_dir / "gmm_means.npy"), gmm.means_)
        np.save(str(seg_dir / "gmm_covariances.npy"), gmm.covariances_)
        np.save(str(seg_dir / "gmm_weights.npy"), gmm.weights_)

        vel = np.linalg.norm(np.diff(segmentation_traj, axis=0), axis=1)
        vel = np.concatenate([[vel[0] if len(vel) else 0.0], vel])
        np.save(str(seg_dir / "object_velocity.npy"), vel.astype(np.float32))

        # Identify last cluster by proximity to final position (last 10% of frames)
        n_final = max(5, int(len(segmentation_traj) * 0.10))
        final_seg = segmentation_traj[max(0, len(segmentation_traj) - n_final):]
        finite_final = np.isfinite(final_seg).all(axis=1) & np.any(final_seg != 0, axis=1)
        if not np.any(finite_final):
            finite_final = np.isfinite(final_seg).all(axis=1)
        final_position = final_seg[finite_final].mean(axis=0) if np.any(finite_final) else segmentation_traj[-1]
        last_cluster = int(np.argmin(np.linalg.norm(gmm.means_ - final_position, axis=1)))

        low_vel_threshold = float(np.percentile(vel, 40))
        min_release_idx = min(len(segmentation_traj) - 1, grasp_idx + RELEASE_MIN_MOVE_FRAMES)
        stable_window = max(2, RELEASE_STABLE_WINDOW)
        release_idx = None
        release_detection_method = "gmm_final_cluster"

        # Primary: last stable displaced plateau (VILMA's best algorithm)
        disp_from_start = np.linalg.norm(segmentation_traj - initial_position, axis=1)
        post_grasp_disp = disp_from_start[min_release_idx:] if min_release_idx < len(disp_from_start) else disp_from_start
        max_post_grasp_disp = float(np.nanmax(post_grasp_disp)) if len(post_grasp_disp) else 0.0
        significant_disp = max(0.08, 0.60 * max_post_grasp_disp)
        stable_velocity_limit = max(low_vel_threshold * 2.0, low_vel_threshold + 1e-6)
        stable_place_candidates = []
        if max_post_grasp_disp >= 0.12:
            for i in range(min_release_idx, len(segmentation_traj) - stable_window + 1):
                if (
                    np.all(disp_from_start[i : i + stable_window] >= significant_disp)
                    and np.all(vel[i : i + stable_window] <= stable_velocity_limit)
                ):
                    stable_place_candidates.append(i)
        if stable_place_candidates:
            release_idx = int(min(len(segmentation_traj) - 1, stable_place_candidates[-1] + stable_window - 1))
            release_detection_method = "last_stable_displaced_plateau"

        # Fallback 1: stable window in last cluster + low velocity
        if release_idx is None:
            for i in range(min_release_idx, len(segmentation_traj) - stable_window + 1):
                if (np.all(labels[i : i + stable_window] == last_cluster) and
                        np.all(vel[i : i + stable_window] <= low_vel_threshold)):
                    release_idx = i
                    break

        # Fallback 2: relax velocity, cluster only
        if release_idx is None:
            for i in range(min_release_idx, len(segmentation_traj) - stable_window + 1):
                if np.all(labels[i : i + stable_window] == last_cluster):
                    release_idx = i
                    release_detection_method = "gmm_final_cluster_relaxed_velocity"
                    break

        # Fallback 3: first frame in last cluster
        if release_idx is None:
            candidate = np.where(labels == last_cluster)[0]
            candidate = candidate[candidate >= min_release_idx]
            release_idx = int(candidate[0]) if candidate.size else len(segmentation_traj) - 1
            release_detection_method = "gmm_final_cluster_first_candidate"

        np.save(str(seg_dir / "release_idx.npy"), np.array(release_idx, dtype=np.int32))
        np.save(str(seg_dir / "last_cluster.npy"), np.array(last_cluster, dtype=np.int32))
        with open(str(seg_dir / "release_detection_debug.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": release_detection_method,
                    "release_idx": int(release_idx),
                    "min_release_idx": int(min_release_idx),
                    "stable_window": int(stable_window),
                    "low_vel_threshold": float(low_vel_threshold),
                    "stable_velocity_limit": float(stable_velocity_limit),
                    "max_post_grasp_displacement_m": float(max_post_grasp_disp),
                    "significant_displacement_m": float(significant_disp),
                    "stable_place_candidate_count": int(len(stable_place_candidates)),
                    "first_stable_place_candidate": int(stable_place_candidates[0]) if stable_place_candidates else None,
                    "last_stable_place_candidate": int(stable_place_candidates[-1]) if stable_place_candidates else None,
                },
                f, indent=2,
            )

        if obj_valid_mask is not None and len(obj_valid_mask) > 0:
            _grasp_valid = bool(obj_valid_mask[min(grasp_idx, len(obj_valid_mask) - 1)])
            _release_valid = bool(obj_valid_mask[min(release_idx, len(obj_valid_mask) - 1)])
            with open(str(seg_dir / "object_detection_event_validity.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "grasp_idx": int(grasp_idx),
                        "grasp_has_object_detection": _grasp_valid,
                        "release_idx": int(release_idx),
                        "release_has_object_detection": _release_valid,
                    },
                    f, indent=2,
                )
    # --- Script 09: Extract skill phases (use HAND trajectory, not reconstructed) ---
    with st.spinner("Extracting skill phases (Reach, Move)..."):
        t_len = min(len(hand_smooth), len(reconstructed))
        grasp_idx = int(np.clip(grasp_idx, 1, t_len - 2))
        release_idx = int(np.clip(release_idx, grasp_idx + 1, t_len - 1))

        min_release = min(t_len - 1, grasp_idx + RELEASE_MIN_MOVE_FRAMES)
        if release_idx < min_release:
            release_idx = min_release

        reach = hand_smooth[:grasp_idx]
        move = hand_smooth[grasp_idx + 1 : release_idx]
        post_release_traj = hand_smooth[release_idx + 1 : t_len]

        grasp_pos = hand_smooth[grasp_idx]
        release_pos = hand_smooth[release_idx]

        pre_grasp = grasp_pos + np.array([0.0, 0.0, PREPOST_DELTA_P], dtype=np.float32)
        post_grasp = pre_grasp.copy()
        pre_release = release_pos + np.array([0.0, 0.0, PREPOST_DELTA_P], dtype=np.float32)
        post_release = post_release_traj[-1] if len(post_release_traj) > 0 else release_pos

        np.save(str(seg_dir / "reach_traj.npy"), reach)
        np.save(str(seg_dir / "move_traj.npy"), move)
        np.save(str(seg_dir / "post_release_traj.npy"), post_release_traj)

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
            "post_release_start": release_idx + 1 if release_idx + 1 < t_len else None,
            "post_release_end": t_len - 1 if release_idx + 1 < t_len else None,
            "release_detection_method": release_detection_method,
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

    phase_colors = [
        (0, grasp_idx, "rgba(66,133,244,0.12)"),
        (grasp_idx, release_idx, "rgba(52,168,83,0.12)"),
    ]
    if release_idx < T_total:
        phase_colors.append((release_idx, T_total, "rgba(200,200,200,0.12)"))

    for x0, x1, color in phase_colors:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, layer="below", line_width=0)

    for mx, dash, color, width, label in [
        (grasp_idx, "dash", "black", 2, "Grasp"),
        (release_idx, "dot", "#EA8600", 2, "Release"),
    ]:
        fig.add_vline(x=mx, line_dash=dash, line_color=color, line_width=width,
                      annotation_text=label, annotation_position="top right",
                      annotation_font=dict(size=12, color=color))

    fig.update_layout(height=500, showlegend=True, hovermode="x unified")
    fig.update_xaxes(title_text="Frame", row=3, col=1)

    # Build frame paths list for viewer (prefer annotated objects frames)
    rgb_files_sorted = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) + list(frames_dir.glob("*.png")),
        key=lambda p: p.name,
    )
    annotated_dir = objects_dir / "annotated"
    trajectory_paths = []
    for frame_path in rgb_files_sorted[:len(reconstructed)]:
        annotated_path = annotated_dir / f"objects_{frame_path.stem}.jpg"
        trajectory_paths.append(str(annotated_path if annotated_path.exists() else frame_path))

    st.session_state.step_results["trajectory"] = {
        "type": "trajectory2d",
        "fig": fig,
        "timestamps": timestamps,
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "grasp_idx": grasp_idx,
        "release_idx": release_idx,
        "release_detection_method": release_detection_method,
        "trajectory_source": "hand_trajectory",
        "segmentation_object_detection_frames": int(np.count_nonzero(segmentation_source == 1)),
        "segmentation_hand_fallback_frames": int(np.count_nonzero(segmentation_source == 2)),
        "paths": trajectory_paths,
    }
    return {"grasp_idx": grasp_idx, "release_idx": release_idx}


# ========================================================================
# STEP 4: DMP (Scripts 10 + 12)
# ========================================================================

def _generate_skill_reuse_dmp(dmp_dir: Path, plot_dir: Path, seg_dir: Path):
    """
    Generate the 'Skill Reuse' trajectory by concatenating adapted DMP
    rollouts with grasp/release events (mirrors core script 13 logic).

    Outputs:
        dmp_dir / skill_reuse_traj.npy
        dmp_dir / skill_reuse_traj.csv
        dmp_dir / skill_reuse_traj.json
        plot_dir / skill_reuse_final.png
    """
    # --- Try adapted trajectories first (produced by adapt step) ---
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
    post_release_traj_path = seg_dir / "post_release_traj.npy"
    post_release_pos_path = seg_dir / "post_release_pos.npy"

    grasp_pos = np.load(str(grasp_pos_path)) if grasp_pos_path.exists() else reach[-1]
    release_pos = np.load(str(release_pos_path)) if release_pos_path.exists() else (move[-1] if len(move) > 0 else reach[-1])

    # Use actual post-release hand trajectory if available, else hold 15 frames
    POST_RELEASE_HOLD = 15
    if post_release_traj_path.exists():
        _prt = np.load(str(post_release_traj_path))
        post_release_segment = _prt if len(_prt) > 0 else np.tile(release_pos[None, :], (POST_RELEASE_HOLD, 1))
    elif post_release_pos_path.exists():
        _prp = np.load(str(post_release_pos_path))
        post_release_segment = np.tile(_prp[None, :], (POST_RELEASE_HOLD, 1))
    else:
        post_release_segment = np.tile(release_pos[None, :], (POST_RELEASE_HOLD, 1))

    # --- Concatenate into full skill trajectory (script 13 logic) ---
    parts = [reach, grasp_pos[None, :]]
    if len(move) > 0:
        parts.append(move)
    parts.extend([release_pos[None, :], post_release_segment])
    skill_traj = np.vstack(parts)

    # --- Save .npy, .csv, .json ---
    np.save(str(dmp_dir / "skill_reuse_traj.npy"), skill_traj)
    np.savetxt(
        str(dmp_dir / "skill_reuse_traj.csv"),
        skill_traj, delimiter=',', header='x,y,z', comments='',
    )
    with open(str(dmp_dir / "skill_reuse_traj.json"), 'w', encoding='utf-8') as f:
        json.dump(skill_traj.tolist(), f)

    # --- Save segment-boundary metadata so viewer/robot step can read it directly ---
    _post_release_len = len(post_release_segment) if len(post_release_segment) > 0 else POST_RELEASE_HOLD
    _post_release_source = "hand_trajectory" if (post_release_traj_path.exists() and
                           len(np.load(str(post_release_traj_path))) > 0) else "hold"
    with open(str(dmp_dir / "skill_reuse_traj_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reach_len": int(len(reach)),
                "move_len": int(len(move)),
                "post_release_len": int(_post_release_len),
                "post_release_source": _post_release_source,
                "grasp_idx": int(len(reach)),
                "release_idx": int(len(reach) + 1 + len(move)),
            },
            f,
            indent=2,
        )

    # --- Plot (matches script 13) ---
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


def handle_svo_dmp(base_path: Path, session_id: str = None):
    """Step 4: DMP Learning."""
    _init_step_results()

    # Route seg/dmp/plots through versioned objects session when available
    _obj_sess = st.session_state.get("active_objects_session_id") or session_id
    if _obj_sess:
        seg_dir = base_path / "segmentation" / _obj_sess
        dmp_dir = base_path / "dmp" / _obj_sess
        plots_dir = base_path / "plots" / _obj_sess
    elif session_id:
        seg_dir = base_path / "segmentation" / session_id
        dmp_dir = base_path / "dmp" / session_id
        plots_dir = base_path / "plots" / session_id
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
    
    # --- Script 10: Learn DMPs with Lambda Sweeping ---
    with st.spinner("Training DMPs (Reach, Move) with lambda tuning..."):
        
        def train_phase_pipeline(name: str, traj: np.ndarray):
            candidates = sorted(set([float(DMP_REG_LAMBDA), *[float(x) for x in DMP_LAMBDA_CANDIDATES]]))
            best = None
            best_pack = None
            
            for reg_lambda in candidates:
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

    # --- Script 11: Test DMP reproduction ---
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

    # --- Script 12: Adapt DMPs to new start/goal ---
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

    # --- Script 13: Skill Reuse (uses adapted trajectories) ---
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

    # If skill reuse failed, promote object_xyz_dmp so the robot step can find it
    if (reuse_traj is None or len(reuse_traj) == 0) and len(full_dmp_traj) > 0:
        np.save(str(dmp_dir / "skill_reuse_traj.npy"), full_dmp_traj)

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

    # Compute timestamps synced with video frames
    # DMP points 0..release_idx map 1:1 to video frames 0..release_idx
    # DMP points release_idx+1..end spread across remaining video frames
    num_points = len(viewer_traj)
    if session_id:
        rgb_dir = base_path / "frames" / session_id
    else:
        rgb_dir = base_path / "frames"
    video_fps = 15.0
    num_video_frames = sum(len(list(rgb_dir.glob(ext))) for ext in ["*.png", "*.jpg", "*.jpeg"]) if rgb_dir.exists() else num_points
    video_duration = num_video_frames / video_fps

    # Build frame-aligned timestamps
    dmp_timestamps = []
    n_pre = min(sr_release_idx + 1, num_points)    # points that map 1:1 with video frames
    n_post = num_points - n_pre                      # post-release hold points
    remaining_video_frames = max(1, num_video_frames - n_pre)

    for i in range(num_points):
        if i < n_pre:
            # 1:1 mapping with video frames
            dmp_timestamps.append(i / video_fps)
        else:
            # Spread post-release points across remaining video time
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
# STEP 5: ROBOT (13?)
# ========================================================================

def handle_svo_robot(
    base_path: Path,
    urdf_path: str = "data/Common/robot_models/openarm/openarm.urdf",
    robot_config: dict = None,
    session_id: str = None
):
    """Step 5: Robot Playback."""
    _init_step_results()

    if robot_config is None:
        robot_config = {}

    # Route through versioned objects session so the correct detection run is used
    _obj_sess = st.session_state.get("active_objects_session_id") or session_id
    if _obj_sess:
        dmp_dir = base_path / "dmp" / _obj_sess
        seg_dir = base_path / "segmentation" / _obj_sess
    elif session_id:
        dmp_dir = base_path / "dmp" / session_id
        seg_dir = base_path / "segmentation" / session_id
    else:
        dmp_dir = base_path / "dmp"
        seg_dir = base_path / "segmentation"

    dmp_xyz_path = dmp_dir / "skill_reuse_traj.npy"
    if not dmp_xyz_path.exists():
         dmp_xyz_path = dmp_dir / "object_xyz_dmp.npy"
    
    if not dmp_xyz_path.exists():
        st.error("DMP trajectory (skill_reuse_traj.npy) not found. Run DMP step first.")
        return None

    from src.streamlit_template.core.Common.robot_playback import (
        dmp_xyz_to_cartesian,
        compute_ik_trajectory,
        cached_meshes_for_pose,
    )

    # --- Load segment boundaries (prefer metadata JSON, fall back to .npy files) ---
    grasp_idx_robot = None
    release_idx_robot = None
    post_release_len_robot = 0
    _meta_path = dmp_dir / "skill_reuse_traj_metadata.json"
    try:
        if _meta_path.exists():
            with open(str(_meta_path), encoding="utf-8") as _mf:
                _meta = json.load(_mf)
            grasp_idx_robot = int(_meta["grasp_idx"])
            release_idx_robot = int(_meta["release_idx"])
            post_release_len_robot = int(_meta.get("post_release_len", 0))
        else:
            reach_path = seg_dir / "reach_traj.npy"
            move_path = seg_dir / "move_traj.npy"
            if reach_path.exists():
                reach_len = len(np.load(str(reach_path)))
                grasp_idx_robot = reach_len
                if move_path.exists():
                    move_len = len(np.load(str(move_path)))
                    release_idx_robot = reach_len + 1 + move_len
    except Exception:
        pass

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

    num_frames = len(q_traj)
    if session_id:
        rgb_dir = base_path / "frames" / session_id
    else:
        rgb_dir = base_path / "frames"
    video_fps = 15.0
    num_video_frames = (
        sum(len(list(rgb_dir.glob(ext))) for ext in ["*.png", "*.jpg", "*.jpeg"])
        if rgb_dir.exists()
        else num_frames
    )

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
        "post_release_len": post_release_len_robot,
    }
    return {
        "cart_path": cart_path,
        "q_traj": q_traj,
        "frame_timestamps": frame_timestamps,
        "grasp_idx": grasp_idx_robot,
        "release_idx": release_idx_robot,
        "post_release_len": post_release_len_robot,
    }
