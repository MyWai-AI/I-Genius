"""
Pipeline Service - Core processing steps for VILMA.
Handles detection, trajectory, DMP, and robot playback.
Frame extraction is handled separately in frame_service.py.
"""
import streamlit as st
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

# Pipeline engines
from src.streamlit_template.core.Generic.realsense_bag_extract import extract_from_realsense_bag
from src.streamlit_template.core.Generic.hand_detection import detect_hands_on_frames
from src.streamlit_template.core.Generic.trajectory_extraction import extract_hand_object_trajectory
from src.streamlit_template.core.Generic.generate_dmp_xyz import generate_dmp_xyz
from src.streamlit_template.core.Common.robot_playback import (
    dmp_xyz_to_cartesian,
    compute_ik_trajectory,
    cached_meshes_for_pose,
    build_gripper_trajectory,
)


def _init_step_results():
    """Initialize step_results in session state if missing."""
    if "step_results" not in st.session_state:
        st.session_state.step_results = {}


def _resolve_detector_model(model_path: str) -> str:
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
    import json
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
    import json, datetime
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
    import json as _j
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


# --- BAG EXTRACTION ---

def handle_bag_extract(bag_path: Path, output_dir: Path):
    """Extract data from RealSense BAG file."""
    if not bag_path.exists():
        st.error("BAG file not found.")
        return None
    
    with st.spinner("Extracting RealSense BAG..."):
        res = extract_from_realsense_bag(
            bag_path=str(bag_path),
            out_dir=str(output_dir),
        )
    
    if "error" in res:
        st.error(f"BAG extraction failed: {res['error']}")
        return None
    
    return res


# --- HAND DETECTION ---

def handle_hands(frames_dir: Path, hands_dir: Path):
    """Detect hands on extracted frames."""
    _init_step_results()
    
    idx = frames_dir / "frames_index.csv"
    if not idx.exists():
        st.warning("Frame index not found. Run frame extraction first.")
        return None
    
    with st.spinner("Detecting Hands..."):
        res = detect_hands_on_frames(
            frames_dir=str(frames_dir),
            index_csv=str(idx),
            out_dir=str(hands_dir),
            max_hands=2,
            sample_every_n=1,
        )
    
    imgs = sorted(Path(res["annotated_dir"]).glob("hands_*.jpg"))
    st.session_state.step_results["hands"] = {"type": "hands", "paths": imgs}
    return res


# --- OBJECT DETECTION ---

def handle_objects(
    frames_dir: Path,
    objects_dir: Path,
    model_path: str = "yolov8x.pt",
    tracking_bbox_xyxy=None,
    tracking_label: str = None,
    confidence_threshold: float = 0.05,
    max_area_pct: float = 15.0,
    min_area_pct: float = 0.0001,
    bbox_size_ratio: float = 4.0,
):
    """Detect objects on extracted frames using Local Step 2 model/selection when available.

    Behavior:
    - If tracking_bbox_xyxy is provided, pick the closest candidate each frame to follow the selected object.
    - Otherwise, use best-confidence candidate per frame (SVO-like fallback behavior).
    """
    _init_step_results()

    idx = frames_dir / "frames_index.csv"
    if not idx.exists():
        st.warning("Frame index not found. Run frame extraction first.")
        return None

    import csv as _csv
    import cv2 as _cv2
    import json as _json
    from ultralytics import YOLO

    objects_dir = Path(objects_dir)
    objects_dir.mkdir(parents=True, exist_ok=True)

    selected_bbox_norm = _normalize_bbox_xyxy(tracking_bbox_xyxy)
    _run_params = {
        "model": _resolve_detector_model(model_path),
        "tracking_label": tracking_label,
        "bbox": selected_bbox_norm,
        "confidence_threshold": round(float(confidence_threshold), 4),
        "max_area_pct": round(float(max_area_pct), 2),
        "min_area_pct": round(float(min_area_pct), 4),
        "bbox_size_ratio": round(float(bbox_size_ratio), 2),
    }

    # --- Rerun guard: skip if run_metadata.json matches current params ---
    _obj_annotated_dir = objects_dir
    _cached_imgs = sorted(objects_dir.glob("objects_*.jpg"))
    if _cached_imgs and _objects_run_params_match(objects_dir, _run_params):
        _saved_meta = _json.load(open(str(objects_dir / "run_metadata.json"), encoding="utf-8"))
        imgs = _cached_imgs
        st.session_state.step_results["objects"] = {
            "type": "objects",
            "paths": imgs,
            "total_detections": _saved_meta.get("total_detections", len(imgs)),
            "detections_csv": str((objects_dir / "objects_detections.csv").resolve()),
            "model_used": _saved_meta.get("model"),
            "selection_mode": "local-selected" if _saved_meta.get("bbox") else "auto-best",
            "run_metadata": _saved_meta,
        }
        return {
            "processed_frames": _saved_meta.get("frame_count", len(imgs)),
            "total_detections": _saved_meta.get("total_detections", len(imgs)),
            "annotated_dir": str(objects_dir.resolve()),
            "detections_csv": str((objects_dir / "objects_detections.csv").resolve()),
            "model": _run_params["model"],
            "selection_mode": "local-selected" if selected_bbox_norm else "auto-best",
        }

    with open(str(idx), "r", newline="", encoding="utf-8") as _f:
        rows = list(_csv.DictReader(_f))

    if not rows:
        st.warning("No indexed frames available for object detection.")
        return None

    model_in_use = _resolve_detector_model(model_path)
    detector = YOLO(model_in_use)

    conf = float(confidence_threshold)
    conf = max(0.01, min(0.95, conf))
    max_area = float(max_area_pct)
    min_area = float(min_area_pct)
    max_area = max(0.1, min(100.0, max_area))
    min_area = max(0.0, min(max_area, min_area))

    selected_bbox = _normalize_bbox_xyxy(tracking_bbox_xyxy)
    tracking_mode = selected_bbox is not None

    # ── Track-ID state for YOLO built-in tracker (BoT-SORT) ──
    _matched_track_id = None  # Set on first frame via IoU match
    _target_center = None
    if selected_bbox is not None:
        _target_center = [
            (selected_bbox[0] + selected_bbox[2]) / 2.0,
            (selected_bbox[1] + selected_bbox[3]) / 2.0,
        ]

    det_csv_path = objects_dir / "objects_detections.csv"
    total_dets = 0

    with det_csv_path.open("w", newline="", encoding="utf-8") as _fout:
        _w = _csv.writer(_fout)
        _w.writerow([
            "frame_idx", "time_sec", "image_file",
            "det_id", "class_id", "class_name", "confidence",
            "xmin", "ymin", "xmax", "ymax",
        ])

        with st.spinner("Detecting Objects..."):
            for _row in rows:
                _img_path = frames_dir / _row["filename"]
                _frame_idx = int(float(_row["frame_idx"]))
                _time_sec = float(_row["time_sec"])

                if not _img_path.exists():
                    continue

                _bgr = _cv2.imread(str(_img_path))
                if _bgr is None:
                    continue

                _frame_h, _frame_w = _bgr.shape[:2]
                _frame_area = float(_frame_h * _frame_w)

                # Use model.track() with persist=True in tracking mode for BoT-SORT
                # identity tracking; use model.predict() otherwise.
                if tracking_mode:
                    _prediction = detector.track(
                        _bgr,
                        persist=True,
                        conf=conf,
                        iou=0.45,
                        max_det=100,
                        verbose=False,
                    )
                else:
                    _prediction = detector.predict(
                        _bgr,
                        conf=conf,
                        iou=0.45,
                        max_det=100,
                        agnostic_nms=True,
                        verbose=False,
                    )
                _res = _prediction[0] if len(_prediction) > 0 else None

                _selected = None
                if _res is not None and _res.boxes is not None and len(_res.boxes) > 0:
                    _xyxy = _res.boxes.xyxy.cpu().numpy()
                    _confs = _res.boxes.conf.cpu().numpy()
                    if _res.boxes.cls is not None:
                        _cls_ids = _res.boxes.cls.cpu().numpy().astype(int)
                    else:
                        _cls_ids = np.zeros(len(_xyxy), dtype=int)
                    _track_ids = None
                    if tracking_mode and _res.boxes.id is not None:
                        _track_ids = _res.boxes.id.int().cpu().numpy()

                    _det_names = _res.names if _res is not None else {}
                    if isinstance(_det_names, list):
                        _det_names = {i: n for i, n in enumerate(_det_names)}

                    if tracking_mode and _track_ids is not None:
                        # ── Track-ID based selection (BoT-SORT) ──
                        if _matched_track_id is None:
                            # First frame: lock onto the track whose detection best matches the
                            # user's selected bbox. Filter by label first so we don't accidentally
                            # lock onto "person" when the user selected e.g. "toothbrush" (shaft).
                            sx1, sy1, sx2, sy2 = [float(v) for v in selected_bbox]

                            _all_indices_g = list(range(len(_xyxy)))
                            _sel_area_g = (selected_bbox[2] - selected_bbox[0]) * (selected_bbox[3] - selected_bbox[1]) if selected_bbox is not None else None
                            _filtered_g = []
                            for _i in _all_indices_g:
                                _cname_g = _det_names.get(int(_cls_ids[_i]), "")
                                if tracking_label and _cname_g and _cname_g != tracking_label:
                                    continue
                                if _sel_area_g is not None and _sel_area_g > 1.0:
                                    _dx1, _dy1, _dx2, _dy2 = [float(v) for v in _xyxy[_i].tolist()]
                                    _det_area_g = max(0.0, _dx2 - _dx1) * max(0.0, _dy2 - _dy1)
                                    if _det_area_g > 0 and not (1.0 / bbox_size_ratio <= _det_area_g / _sel_area_g <= bbox_size_ratio):
                                        continue
                                _filtered_g.append(_i)
                            _candidate_indices_g = _filtered_g if _filtered_g else _all_indices_g

                            best_iou = 0.0
                            best_match_idx = -1
                            for _i in _candidate_indices_g:
                                dx1, dy1, dx2, dy2 = [float(v) for v in _xyxy[_i].tolist()]
                                ix1 = max(sx1, dx1); iy1 = max(sy1, dy1)
                                ix2 = min(sx2, dx2); iy2 = min(sy2, dy2)
                                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                                area_s = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
                                area_d = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
                                union = area_s + area_d - inter
                                iou = inter / union if union > 0 else 0.0
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match_idx = _i
                            if best_match_idx < 0 or best_iou <= 0.01:
                                # Centroid fallback among label-filtered candidates
                                tx, ty = _target_center
                                dists = [float(np.hypot(float(_xyxy[_i][0] + _xyxy[_i][2]) / 2.0 - tx,
                                                        float(_xyxy[_i][1] + _xyxy[_i][3]) / 2.0 - ty))
                                         for _i in _candidate_indices_g]
                                best_match_idx = _candidate_indices_g[int(np.argmin(dists))]
                            _matched_track_id = int(_track_ids[best_match_idx])

                        # Find the detection with our locked track ID.
                        # Reject if bbox grew too large (e.g. hand merged with object).
                        for _i in range(len(_xyxy)):
                            if int(_track_ids[_i]) == _matched_track_id:
                                x1, y1, x2, y2 = [float(v) for v in _xyxy[_i].tolist()]
                                _w_box = max(0.0, x2 - x1)
                                _h_box = max(0.0, y2 - y1)
                                _area = _w_box * _h_box
                                if _area > (_frame_area * (max_area / 100.0)):
                                    continue
                                if _area < (_frame_area * (min_area / 100.0)):
                                    continue
                                if _sel_area_g is not None and _sel_area_g > 1.0 and _area > 0:
                                    if not (1.0 / bbox_size_ratio <= _area / _sel_area_g <= bbox_size_ratio):
                                        break  # bbox too different — treat as missed detection
                                _cname = _det_names.get(int(_cls_ids[_i]), "")
                                _selected = {
                                    "xyxy": [x1, y1, x2, y2],
                                    "cx": x1 + (_w_box / 2.0),
                                    "cy": y1 + (_h_box / 2.0),
                                    "conf": float(_confs[_i]),
                                    "cls_id": int(_cls_ids[_i]),
                                    "cls_name": _cname,
                                }
                                break
                    else:
                        # Non-tracking mode: pick highest confidence candidate
                        _candidates = []
                        for _i in range(len(_xyxy)):
                            x1, y1, x2, y2 = [float(v) for v in _xyxy[_i].tolist()]
                            _w_box = max(0.0, x2 - x1)
                            _h_box = max(0.0, y2 - y1)
                            _area = _w_box * _h_box
                            if _area > (_frame_area * (max_area / 100.0)):
                                continue
                            if _area < (_frame_area * (min_area / 100.0)):
                                continue
                            _cname = _det_names.get(int(_cls_ids[_i]), "")
                            _candidates.append({
                                "xyxy": [x1, y1, x2, y2],
                                "cx": x1 + (_w_box / 2.0),
                                "cy": y1 + (_h_box / 2.0),
                                "conf": float(_confs[_i]),
                                "cls_id": int(_cls_ids[_i]),
                                "cls_name": _cname,
                            })
                        if _candidates:
                            _selected = max(_candidates, key=lambda _c: _c["conf"])

                if _selected is not None:
                    x1, y1, x2, y2 = [int(round(v)) for v in _selected["xyxy"]]
                    _class_names = _res.names if _res is not None else {}
                    if isinstance(_class_names, list):
                        _class_names = {i: n for i, n in enumerate(_class_names)}
                    _class_name = _class_names.get(_selected["cls_id"], "object")

                    _cv2.rectangle(_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    _cv2.putText(
                        _bgr,
                        f"obj {_selected['conf']:.2f}",
                        (x1, max(y1 - 6, 10)),
                        _cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    _w.writerow([
                        _frame_idx,
                        f"{_time_sec:.6f}",
                        _row["filename"],
                        0,
                        _selected["cls_id"],
                        _class_name,
                        f"{_selected['conf']:.4f}",
                        x1,
                        y1,
                        x2,
                        y2,
                    ])
                    total_dets += 1

                _out_name = objects_dir / f"objects_{_frame_idx:04d}.jpg"
                _cv2.imwrite(str(_out_name), _bgr)

    res = {
        "processed_frames": len(rows),
        "total_detections": total_dets,
        "annotated_dir": str(objects_dir.resolve()),
        "detections_csv": str(det_csv_path.resolve()),
        "model": model_in_use,
        "selection_mode": "local-selected" if tracking_mode else "auto-best",
    }

    _save_objects_run_metadata(objects_dir, _run_params, {
        "total_detections": total_dets,
        "frame_count": len(rows),
    })
    _saved_meta = _json.load(open(str(objects_dir / "run_metadata.json"), encoding="utf-8")) if (objects_dir / "run_metadata.json").exists() else {}

    imgs = sorted(Path(res["annotated_dir"]).glob("objects_*.jpg"))
    st.session_state.step_results["objects"] = {
        "type": "objects",
        "paths": imgs,
        "total_detections": res.get("total_detections", 0),
        "detections_csv": res.get("detections_csv"),
        "model_used": res.get("model"),
        "selection_mode": res.get("selection_mode"),
        "run_metadata": _saved_meta,
    }
    return res


# --- TRAJECTORY EXTRACTION ---

def handle_trajectory(frames_dir: Path, hands_dir: Path, objects_dir: Path, traj_dir: Path):
    """Extract hand-object trajectory from detection results."""
    _init_step_results()
    
    hands_csv = hands_dir / "hands_landmarks.csv"
    if not hands_csv.exists():
        st.warning("Run Hands detection first.")
        return None
    
    objects_csv = objects_dir / "objects_detections.csv"
    
    with st.spinner("Extracting Trajectory..."):
        res = extract_hand_object_trajectory(
            frames_dir=str(frames_dir),
            hands_csv=str(hands_csv),
            objects_csv=str(objects_csv) if objects_csv.exists() else None,
            out_csv=str(traj_dir / "hand_traj.csv"),
            out_plot=str(traj_dir / "trajectory.png"),
            fingertip_id=8,
            smooth_window=5,
        )
    
    # Load trajectory CSV to get timestamps for interactive 2D plots
    traj_csv_path = traj_dir / "hand_traj.csv"
    if traj_csv_path.exists():
        import csv
        from plotly.subplots import make_subplots
        
        timestamps = []
        x_vals, y_vals, z_vals = [], [], []
        
        with traj_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row["TIME"]))
                x_vals.append(float(row["X"]))
                y_vals.append(float(row["Y"]))
                z_vals.append(float(row["Z"]))
        
        # Create 2D subplots: X, Y, Z vs Time (3 rows)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("X Position [m]", "Y Position [m]", "Z Position [m]"),
        )
        
        # X vs Time
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=x_vals,
                mode="lines+markers",
                marker=dict(size=5, color="blue"),
                line=dict(width=2, color="blue"),
                name="X",
                hovertemplate="Time: %{x:.2f}s<br>X: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1
        )
        
        # Y vs Time
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=y_vals,
                mode="lines+markers",
                marker=dict(size=5, color="green"),
                line=dict(width=2, color="green"),
                name="Y",
                hovertemplate="Time: %{x:.2f}s<br>Y: %{y:.4f}<extra></extra>",
            ),
            row=2, col=1
        )
        
        # Z vs Time
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=z_vals,
                mode="lines+markers",
                marker=dict(size=5, color="red"),
                line=dict(width=2, color="red"),
                name="Z",
                hovertemplate="Time: %{x:.2f}s<br>Z: %{y:.4f}<extra></extra>",
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            title_text="Hand Trajectory (Click on plot to seek video)",
            showlegend=True,
            hovermode="x unified",
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="X [m]", row=1, col=1)
        fig.update_yaxes(title_text="Y [m]", row=2, col=1)
        fig.update_yaxes(title_text="Z [m]", row=3, col=1)
        
        st.session_state.step_results["trajectory"] = {
            "type": "trajectory2d",
            "fig": fig,
            "timestamps": timestamps,
            "x": x_vals,
            "y": y_vals,
            "z": z_vals,
        }
    else:
        # Fallback to static image
        out_img = Path(res.get("traj_plot", str(traj_dir / "trajectory.png")))
        if not out_img.exists():
            out_img = traj_dir / "object_xyz_dmp_plot.png"
        
        if out_img.exists():
            st.session_state.step_results["trajectory"] = {
                "type": "dmp_image",
                "image": str(out_img),
            }
        else:
            st.warning(f"Trajectory plot not found: {out_img.name}")
    
    return res


# --- DMP GENERATION ---

def handle_dmp(traj_dir: Path, dmp_dir: Path):
    """Generate 3D DMP from trajectory data with interactive click-to-seek."""
    _init_step_results()
    
    traj_csv_path = traj_dir / "hand_traj.csv"
    dmp_xyz_path = dmp_dir / "object_xyz_dmp.npy"
    
    if not traj_csv_path.exists():
        st.warning("Hand trajectory CSV not found (Run Trajectory step first).")
        return None
    
    # Load trajectory timestamps for time mapping
    timestamps = []
    import csv
    with traj_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["TIME"]))
    
    with st.spinner("Generating 3D DMP..."):
        res = generate_dmp_xyz(
            traj_csv=str(traj_csv_path),
            out_npy=str(dmp_xyz_path),
            smooth_window=7,
            polyorder=2,
        )
    
    if "error" in res:
        st.error(f"DMP Generation failed: {res['error']}")
        return None
    
    if not dmp_xyz_path.exists():
        st.warning("XYZ DMP file was not created.")
        return None
    
    # Load and visualize
    Yg = np.load(str(dmp_xyz_path))
    num_points = len(Yg)
    pick_pos = Yg[0]
    place_pos = Yg[-1]
    
    # Interpolate timestamps to match DMP points
    if timestamps:
        t_start = timestamps[0]
        t_end = timestamps[-1]
        dmp_timestamps = [t_start + (t_end - t_start) * i / (num_points - 1) for i in range(num_points)]
    else:
        dmp_timestamps = list(range(num_points))
    
    fig = go.Figure()
    
    # Main trajectory with time coloring and markers for click interaction
    # Main DMP trajectory (Old Style: Blue Line)
    fig.add_trace(go.Scatter3d(
        x=Yg[:, 0], y=Yg[:, 1], z=Yg[:, 2],
        mode="lines",
        line=dict(width=6, color="blue"),
        name="XYZ DMP Trajectory",
    ))
    
    # Pick point
    fig.add_trace(go.Scatter3d(
        x=[pick_pos[0]], y=[pick_pos[1]], z=[pick_pos[2]],
        mode="markers+text",
        marker=dict(size=10, color="green"),
        text=["Pick"], textposition="top center",
        name="Pick Position",
    ))
    
    # Place point
    fig.add_trace(go.Scatter3d(
        x=[place_pos[0]], y=[place_pos[1]], z=[place_pos[2]],
        mode="markers+text",
        marker=dict(size=10, color="red"),
        text=["Place"], textposition="top center",
        name="Place Position",
    ))
    
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        height=550,
        title="DMP Trajectory (Click point to seek video)",
        hovermode="closest",
    )
    
    st.session_state.step_results["dmp"] = {
        "type": "dmp3d",
        "fig": fig,
        "timestamps": dmp_timestamps,
    }
    return res


# --- ROBOT 3D PLAYBACK ---

def handle_robot(dmp_dir: Path, traj_dir: Path = None, urdf_path: str = "data/Common/robot_models/openarm/openarm.urdf", robot_config: dict = None):
    """Generate robot 3D animation from DMP with timestamp mapping for video sync."""
    _init_step_results()
    
    if robot_config is None:
        robot_config = {}

    dmp_xyz_path = dmp_dir / "object_xyz_dmp.npy"
    if not dmp_xyz_path.exists():
        st.error("XYZ DMP file not found. Run DMP generation first.")
        return None
    
    # Load trajectory timestamps if available
    timestamps = None
    if traj_dir:
        traj_csv_path = traj_dir / "hand_traj.csv"
        if traj_csv_path.exists():
            import csv
            timestamps = []
            with traj_csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamps.append(float(row["TIME"]))
    
    with st.spinner("Computing robot trajectory..."):
        # Trajectory transform params (load from config > safe defaults)
        # General Solution: Allow each robot to define its own reachable workspace in config.json
        
        # Default safe values (generic arm)
        default_offset = [0.4, 0.0, 0.2]
        default_scale = [0.5, 0.5, 0.5]
        
        dmp_offset = robot_config.get("dmp_offset", default_offset)
        dmp_scale = robot_config.get("dmp_scale", default_scale)
        dmp_rot_z = robot_config.get("dmp_rotation_z", 90.0)
        dmp_flip_z = robot_config.get("flip_z", False)
        dmp_arm_reach = robot_config.get("arm_reach", 0.0)
        
        # Ensure they are lists/floats
        if not isinstance(dmp_offset, list): dmp_offset = default_offset
        if not isinstance(dmp_scale, list): dmp_scale = default_scale
        
        # Convert offset/scale to values
        off_x, off_y, off_z = dmp_offset
        scale_x, scale_y, scale_z = dmp_scale
        
        # Sync robot frames to video frames
        target_len = len(timestamps) if timestamps else 100
        
        cart = dmp_xyz_to_cartesian(
            dmp_npy=str(dmp_xyz_path),
            scale_xyz=(scale_x, scale_y, scale_z),
            offset_xyz=(off_x, off_y, off_z),
            flip_y=False,
            flip_z=dmp_flip_z,
            rotate_z=dmp_rot_z,
            add_arch=False,
            arch_height=0.2,
            target_frames=target_len,
            arm_reach=dmp_arm_reach,
        )
        cart_path = cart["cartesian_path"]
        
        # Verify Cartesian bounds
        if len(cart_path) > 0:
            mins = np.min(cart_path, axis=0)
            maxs = np.max(cart_path, axis=0)
            deltas = maxs - mins
            
            # bounds msgs removed
        
        # Inverse kinematics
        ik = compute_ik_trajectory(urdf_path=urdf_path, cartesian_path=cart_path)
        q_traj = ik["q_traj"]
        
        # Check for static trajectory (IK failure -> constant output)
        q_deltas = np.ptp(q_traj, axis=0) # Peak-to-peak amplitude per joint
        max_delta = np.max(q_deltas[1:]) # Skip base if present
        if max_delta < 0.01: # Less than 0.01 rad movement
             st.error(f"Robot trajectory is static (Max Joint Delta: {max_delta:.4f} rad). Target likely out of reach.")
             st.write(f"Cartesian Bounds: X[{np.min(cart_path[:,0]):.2f}, {np.max(cart_path[:,0]):.2f}] Y[{np.min(cart_path[:,1]):.2f}, {np.max(cart_path[:,1]):.2f}] Z[{np.min(cart_path[:,2]):.2f}, {np.max(cart_path[:,2]):.2f}]")
    
    num_frames = len(q_traj)

    # Build gripper open/close trajectory (auto-detects grasp at lowest Z)
    gripper_traj = build_gripper_trajectory(cart_path, urdf_path)
    
    # Compute timestamp mapping for each animation frame
    frame_timestamps = None
    if timestamps and len(timestamps) > 0:
        # Linear interpolation: animation frames map to trajectory timestamps
        t_start = timestamps[0]
        t_end = timestamps[-1]
        frame_timestamps = [t_start + (t_end - t_start) * i / (num_frames - 1) for i in range(num_frames)]
    
    # Camera presets
    camera_front = dict(eye=dict(x=1.0, y=0.0, z=0.6), up=dict(x=0, y=0, z=1))
    camera_top = dict(eye=dict(x=0.0, y=0.0, z=1.5), up=dict(x=0, y=1, z=0))
    camera_left = dict(eye=dict(x=0.0, y=-1.5, z=0.6), up=dict(x=0, y=0, z=1))
    camera_right = dict(eye=dict(x=0.0, y=1.5, z=0.6), up=dict(x=0, y=0, z=1))
    camera_iso = dict(eye=dict(x=0.8, y=0.8, z=0.8), up=dict(x=0, y=0, z=1))
    camera_back = dict(eye=dict(x=-1.0, y=0.0, z=0.6), up=dict(x=0, y=0, z=1))
    
    # Build Plotly animation
    with st.spinner("Building 3D animation..."):
        fig = go.Figure()
        frames = []
        
        for i, q in enumerate(q_traj):
            # Merge gripper state for this frame
            grip_cfg = gripper_traj[i] if i < len(gripper_traj) else None
            grip_tuple = tuple(sorted(grip_cfg.items())) if grip_cfg else None
            traces = cached_meshes_for_pose(urdf_path, tuple(q.tolist()), extra_joint_cfg_tuple=grip_tuple)
            
            frame_data = []
            for t in traces:
                r, g, b, a = t.get("rgba", (0.8, 0.0, 0.0, 1.0))
                color_str = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
                
                frame_data.append(go.Mesh3d(
                    x=t["x"], y=t["y"], z=t["z"],
                    i=t["i"], j=t["j"], k=t["k"],
                    color=color_str, opacity=1.0, name=t.get("name", "")
                ))
            
            # End-effector path
            path_upto = cart_path[:i + 1]
            frame_data.append(go.Scatter3d(
                x=path_upto[:, 0], y=path_upto[:, 1], z=path_upto[:, 2],
                mode="lines", line=dict(width=5, color="blue"), name="EE Path",
            ))
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
        
        if frames:
            fig.add_traces(frames[0].data)
            fig.update(frames=frames)
        
        # Add slider for frame scrubbing
        sliders = [{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f"{i}" + (f" ({frame_timestamps[i]:.1f}s)" if frame_timestamps else ""),
                    "method": "animate"
                }
                for i in range(0, num_frames, max(1, num_frames // 20))  # Show ~20 steps
            ]
        }]
        
        # Layout with animation controls
        fig.update_layout(
            height=550,
            sliders=sliders,
            updatemenus=[
                # Play/Pause
                {
                    "type": "buttons", "direction": "right", "x": 0.0, "y": 1.25,
                    "buttons": [
                        {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]},
                        {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                    ],
                },
                # Camera views
                {
                    "type": "buttons", "direction": "down", "x": 0.0, "y": 1.13,
                    "buttons": [
                        {"label": "Front", "method": "relayout", "args": [{"scene.camera": camera_front}]},
                        {"label": "Top", "method": "relayout", "args": [{"scene.camera": camera_top}]},
                        {"label": "Left", "method": "relayout", "args": [{"scene.camera": camera_left}]},
                        {"label": "Right", "method": "relayout", "args": [{"scene.camera": camera_right}]},
                        {"label": "Iso", "method": "relayout", "args": [{"scene.camera": camera_iso}]},
                        {"label": "Back", "method": "relayout", "args": [{"scene.camera": camera_back}]},
                    ],
                },
            ],
            scene_camera=camera_front,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.5),
                xaxis=dict(range=[-2.5, 2.5], autorange=False),
                yaxis=dict(range=[-2.5, 2.5], autorange=False),
                zaxis=dict(range=[0.0, 2.0], autorange=False),
            ),
        )
    
    st.session_state.step_results["robot"] = {
        "type": "robot3d",
        "fig": fig,
        "num_frames": num_frames,
        "frame_timestamps": frame_timestamps,
        "q_traj": q_traj,
        "cart_path": cart_path,
    }
    return {"fig": fig, "cart_path": cart_path, "q_traj": q_traj, "frame_timestamps": frame_timestamps}
