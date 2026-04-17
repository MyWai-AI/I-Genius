"""
Skill Reuse Page — Apply a learned skill to different target objects.

Flow:
1. Shows the base frame with detections (same config as local page)
2. Highlights the pipeline-tracked object (anchor) with a distinct ROI color
3. Draws ROIs around all other detected objects
4. User selects a new target object from a dropdown
5. User optionally selects a release/goal position for the object
6. Computes the offset-shifted skill_reuse trajectory
7. Shows list of generated skill_reuse_<label>.csv files
8. Push to Robot button with DDS topic/domain config
"""
import json
import os
import io
import base64
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.streamlit_template.new_ui.services.Common.skill_reuse_service import (
    DetectedObject,
    compute_ik_for_reuse_traj,
    compute_skill_reuse_for_target,
    detect_objects_on_frame,
    load_intrinsics_dict,
)

try:
    from src.streamlit_template.new_ui.services.Common.skill_reuse_service import pixel_to_xyz
except Exception:
    def pixel_to_xyz(u: int, v: int, depth_m: np.ndarray, intr: dict) -> np.ndarray | None:
        """Fallback pixel-to-3D conversion when service symbol is unavailable."""
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

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None


def _resolve_active_robot():
    """Determine active robot URDF and Config (same as pipeline pages)."""
    def _load_animations(urdf_dir: Path, config: dict) -> dict:
        for name in ["animations.json", "animation.json"]:
            anim_path = urdf_dir / name
            if anim_path.exists():
                try:
                    with open(anim_path, "r") as f:
                        anims = json.load(f)
                    if "animations" not in config:
                        config["animations"] = {}
                    config["animations"].update(anims.get("animations", anims))
                except Exception:
                    pass
                break
        return config

    custom_dir = st.session_state.get("custom_robot_dir") or st.session_state.get("active_custom_robot_dir")
    if custom_dir and os.path.exists(custom_dir):
        urdfs = list(Path(custom_dir).rglob("*.urdf"))
        if urdfs:
            urdf_path = str(urdfs[0])
            urdf_dir = Path(urdfs[0]).parent
            config = {}
            config_path = urdf_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                except Exception:
                    pass
            config = _load_animations(urdf_dir, config)
            return urdf_path, config

    default_urdf = "data/Common/robot_models/openarm/openarm.urdf"
    urdf_dir = Path(default_urdf).parent
    config = {}
    config_path = urdf_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception:
            pass
    config = _load_animations(urdf_dir, config)
    return default_urdf, config


def _get_data_paths():
    """Resolve the data base path, session ID, frames dir, depth dir, intrinsics, and DMP dir."""
    # Determine which pipeline produced the results
    source_platform = st.session_state.get("skill_reuse_source_platform", "svo_pipeline")

    if source_platform == "bag_pipeline":
        session_id = st.session_state.get("bag_session_id", "")
        base = Path("data/BAG")
    elif source_platform == "svo_pipeline":
        base_path = st.session_state.get("svo_base_path", "data/SVO")
        base = Path(base_path)
        session_id = st.session_state.get("svo_session_id", "")
    else:
        # Generic pipeline
        base = Path("data/Generic")
        session_id = st.session_state.get("svo_session_id", "")

    obj_sess = st.session_state.get("active_objects_session_id") or session_id

    frames_dir = base / "frames" / session_id if session_id else base / "frames"
    depth_dir = base / "depth_meters" / session_id if session_id else base / "depth_meters"
    camera_dir = base / "camera"
    intrinsics_path = camera_dir / f"{session_id}.npy" if session_id else None
    dmp_dir = base / "dmp" / obj_sess if obj_sess else base / "dmp"
    seg_dir = base / "segmentation" / obj_sess if obj_sess else base / "segmentation"
    reuse_dir = base / "skill_reuse" / obj_sess if obj_sess else base / "skill_reuse"

    return {
        "base": base,
        "session_id": session_id,
        "obj_sess": obj_sess,
        "frames_dir": frames_dir,
        "depth_dir": depth_dir,
        "intrinsics_path": intrinsics_path,
        "dmp_dir": dmp_dir,
        "seg_dir": seg_dir,
        "reuse_dir": reuse_dir,
    }


def _xyz_to_uv(xyz: np.ndarray, intrinsics: dict, image_shape: tuple[int, int]) -> tuple[int, int] | None:
    """Project 3D camera coordinates to image pixel coordinates."""
    if xyz is None or intrinsics is None:
        return None
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if z <= 0:
        return None

    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    u = int(round((x * fx) / z + cx))
    v = int(round((y * fy) / z + cy))

    h, w = image_shape
    if 0 <= u < w and 0 <= v < h:
        return u, v
    return None


def _resolve_pipeline_release(skill_reuse_path: Path, seg_dir: Path) -> tuple[int | None, np.ndarray | None]:
    """Return (release_idx, release_xyz) from the pipeline-generated base trajectory."""
    try:
        traj = np.load(str(skill_reuse_path))
    except Exception:
        return None, None

    reach_path = seg_dir / "reach_traj.npy"
    move_path = seg_dir / "move_traj.npy"
    reach_len = len(np.load(str(reach_path))) if reach_path.exists() else 0
    move_len = len(np.load(str(move_path))) if move_path.exists() else 0
    release_idx = reach_len + 1 + move_len

    if 0 <= release_idx < len(traj):
        return int(release_idx), np.array(traj[release_idx], dtype=np.float32)
    return int(release_idx), None


def _trajectory_npy_to_csv_bytes(npy_path: Path) -> bytes:
    """Convert a trajectory .npy file to CSV bytes for direct download."""
    arr = np.asarray(np.load(str(npy_path)))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported trajectory shape: {arr.shape}")

    axis_names = ["x", "y", "z"]
    headers = [axis_names[i] if i < len(axis_names) else f"c{i + 1}" for i in range(arr.shape[1])]
    csv_buf = io.StringIO()
    np.savetxt(csv_buf, arr, delimiter=",", header=",".join(headers), comments="")
    return csv_buf.getvalue().encode("utf-8")


def _build_dmp_viewer_data(traj_xyz: np.ndarray, grasp_idx: int, release_idx: int) -> dict:
    """Build the same DMP viewer payload style used in pipeline Step 4."""
    total = int(len(traj_xyz))
    if total <= 0:
        raise ValueError("Trajectory is empty.")

    grasp_idx = max(0, min(int(grasp_idx), total - 1))
    release_idx = max(grasp_idx, min(int(release_idx), total - 1))

    segments = []
    if grasp_idx > 0:
        segments.append({"startIdx": 0, "endIdx": grasp_idx - 1, "color": "#4285F4", "label": "Reach"})
    if release_idx > grasp_idx + 1:
        segments.append({"startIdx": grasp_idx + 1, "endIdx": release_idx - 1, "color": "#34A853", "label": "Move"})
    if release_idx < total:
        segments.append({"startIdx": release_idx, "endIdx": total - 1, "color": "#FF9800", "label": "Post-Release"})

    markers = [
        {"label": "Start", "index": 0, "color": "#00cc96"},
        {"label": "Grasp", "index": grasp_idx, "color": "#111111"},
        {"label": "Release", "index": release_idx, "color": "#ef553b"},
        {"label": "End", "index": total - 1, "color": "#9C27B0"},
    ]

    return {
        "mode": "dmp",
        "dmpTrajectory": {
            "x": traj_xyz[:, 0].tolist(),
            "y": traj_xyz[:, 1].tolist(),
            "z": traj_xyz[:, 2].tolist(),
            "t": list(range(total)),
            "segments": segments,
            "markers": markers,
        },
        "models": [],
        "images": [],
        "timeseries": [],
    }


def _build_robot_viewer_data_for_reuse(
    skill_reuse_npy: Path,
    urdf_path: str,
    robot_config: dict,
    grasp_idx: int,
    release_idx: int,
) -> dict:
    """Build robot viewer payload for a skill reuse trajectory (Step 5 style, no video)."""
    from src.streamlit_template.new_ui.services.Common.robot_service import hydrate_urdf_with_meshes
    from src.streamlit_template.core.Common.robot_playback import (
        load_chain,
        get_configured_joint_names,
        augment_traj_points_with_gripper,
    )

    ik_result = compute_ik_for_reuse_traj(
        skill_reuse_npy=str(skill_reuse_npy),
        urdf_path=urdf_path,
        robot_config=robot_config,
        grasp_idx=int(grasp_idx),
        release_idx=int(release_idx),
    )

    q_traj = ik_result["q_traj"]
    timestamps = ik_result["frame_timestamps"]

    try:
        joint_names_ordered = get_configured_joint_names(urdf_path)
    except Exception:
        try:
            chain = load_chain(urdf_path)
            joint_names_ordered = [l.name for l in chain.links]
        except Exception:
            joint_names_ordered = [f"joint_{i}" for i in range(q_traj.shape[1])]

    traj_points = []
    for t, q in zip(timestamps, q_traj):
        if len(q) == len(joint_names_ordered):
            q_map = {name: float(val) for name, val in zip(joint_names_ordered, q)}
            traj_points.append({"time": float(t), "q": q_map})

    cart_path = ik_result.get("cart_path")
    if traj_points and cart_path is not None:
        augment_traj_points_with_gripper(
            traj_points,
            cart_path,
            urdf_path,
            grasp_idx=int(grasp_idx),
            release_idx=int(release_idx),
        )

    hydrated_urdf = hydrate_urdf_with_meshes(urdf_path, base_dir=Path(urdf_path).parent)
    if not hydrated_urdf:
        raise RuntimeError("Failed to hydrate URDF with mesh assets.")

    urdf_b64 = base64.b64encode(hydrated_urdf.encode("utf-8")).decode("utf-8")
    urdf_uri = f"data:text/xml;base64,{urdf_b64}"

    return {
        "models": [{
            "entityName": "Robot",
            "loadType": "URDF",
            "path": urdf_uri,
            "trajectory": traj_points,
            "rotation": robot_config.get("rotation", [-90, 0, 0]),
            "position": robot_config.get("position", [0, 0, 0]),
            "scale": robot_config.get("scale", [1, 1, 1]),
            "homePose": robot_config.get("home_pose", {}),
            "linkColors": robot_config.get("link_colors", {}),
            "animations": robot_config.get("animations", {}),
        }],
        "images": [],
        "timeseries": [],
    }


def _build_static_robot_viewer_data(urdf_path: str, robot_config: dict) -> dict:
    """Build a static robot viewer payload (no animation trajectory)."""
    from src.streamlit_template.new_ui.services.Common.robot_service import hydrate_urdf_with_meshes

    hydrated_urdf = hydrate_urdf_with_meshes(urdf_path, base_dir=Path(urdf_path).parent)
    if not hydrated_urdf:
        raise RuntimeError("Failed to hydrate URDF with mesh assets.")

    urdf_b64 = base64.b64encode(hydrated_urdf.encode("utf-8")).decode("utf-8")
    urdf_uri = f"data:text/xml;base64,{urdf_b64}"

    return {
        "models": [{
            "entityName": "Robot",
            "loadType": "URDF",
            "path": urdf_uri,
            "trajectory": [],
            "rotation": robot_config.get("rotation", [-90, 0, 0]),
            "position": robot_config.get("position", [0, 0, 0]),
            "scale": robot_config.get("scale", [1, 1, 1]),
            "homePose": robot_config.get("home_pose", {}),
            "linkColors": robot_config.get("link_colors", {}),
            "animations": robot_config.get("animations", {}),
        }],
        "images": [],
        "timeseries": [],
    }


def render_skill_reuse_page():
    """Render the Skill Reuse page."""
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Pipeline"):
            # Go back to the pipeline that produced the results
            source = st.session_state.get("skill_reuse_source_platform", "svo_pipeline")
            st.session_state["selected_platform"] = source
            st.rerun()

    st.markdown("### Skill Reuse — Apply to New Objects")

    paths = _get_data_paths()
    frames_dir = paths["frames_dir"]
    depth_dir = paths["depth_dir"]
    intrinsics_path = paths["intrinsics_path"]
    dmp_dir = paths["dmp_dir"]
    seg_dir = paths["seg_dir"]
    reuse_dir = paths["reuse_dir"]

    # --- Validate prerequisites ---
    skill_reuse_path = dmp_dir / "skill_reuse_traj.npy"
    if not skill_reuse_path.exists():
        skill_reuse_path = dmp_dir / "object_xyz_dmp.npy"
    if not skill_reuse_path.exists():
        st.error("No skill reuse trajectory found. Run the pipeline DMP step first.")
        return

    pipeline_release_idx, pipeline_release_xyz = _resolve_pipeline_release(skill_reuse_path, seg_dir)

    if not frames_dir.exists():
        st.error(f"Frames directory not found: {frames_dir}")
        return

    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_files:
        st.error("No frames found in frames directory.")
        return

    base_frame_path = frame_files[0]

    # Find matching depth frame
    depth_frame_path = None
    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("*.npy"))
        if depth_files:
            depth_frame_path = str(depth_files[0])

    depth_m = None
    if depth_frame_path and os.path.exists(depth_frame_path):
        try:
            depth_m = np.load(depth_frame_path)
        except Exception as e:
            st.warning(f"Could not load depth frame: {e}")

    # Load intrinsics
    intrinsics = None
    if intrinsics_path and intrinsics_path.exists():
        try:
            intrinsics = load_intrinsics_dict(str(intrinsics_path))
        except Exception as e:
            st.warning(f"Could not load intrinsics: {e}")

    if intrinsics is None:
        st.error("Camera intrinsics not found. Cannot compute 3D positions.")
        return

    if pipeline_release_xyz is not None:
        st.session_state.setdefault("sr_rel_x", float(pipeline_release_xyz[0]))
        st.session_state.setdefault("sr_rel_y", float(pipeline_release_xyz[1]))
        st.session_state.setdefault("sr_rel_z", float(pipeline_release_xyz[2]))

    # --- Recover detection config from the local page session state ---
    model_path = st.session_state.get("selected_object_model_path", "yolov8x.pt")
    conf_threshold = st.session_state.get("selected_object_conf_threshold", 0.05)
    max_area_pct = st.session_state.get("selected_object_max_area_pct", 15.0)
    min_area_pct = st.session_state.get("selected_object_min_area_pct", 0.0001)
    anchor_bbox = st.session_state.get("selected_tracking_bbox_xyxy")
    anchor_label = st.session_state.get("selected_tracking_label", "tracked_object")

    # ====================================================================
    # Section 1: Object Detection on Base Frame
    # ====================================================================
    st.markdown("---")
    st.markdown("#### Object Detection")

    _det_cache_key = "skill_reuse_detections"

    # Detection tuning (collapsed by default since we inherit from local page)
    with st.expander("Detection Settings", expanded=False):
        sr_col1, sr_col2 = st.columns(2)
        with sr_col1:
            sr_model = st.selectbox(
                "Detection Model", [model_path], index=0, key="sr_det_model",
                disabled=True, help="Inherited from local page settings",
            )
            sr_conf = st.slider(
                "Confidence", 0.01, 0.50, float(conf_threshold), 0.01, key="sr_det_conf",
            )
        with sr_col2:
            sr_max_a = st.slider(
                "Max Area (%)", 1.0, 100.0, float(max_area_pct), 1.0, key="sr_det_max_a",
            )
            sr_min_a = st.slider(
                "Min Area (%)", 0.0001, 1.0, float(min_area_pct), format="%.4f", key="sr_det_min_a",
            )

        if st.button("Re-detect", key="sr_redetect"):
            st.session_state.pop(_det_cache_key, None)

    # Run detection
    if _det_cache_key not in st.session_state:
        with st.spinner("Detecting objects on base frame..."):
            dets = detect_objects_on_frame(
                frame_path=str(base_frame_path),
                depth_path=depth_frame_path,
                intrinsics=intrinsics,
                model_path=model_path,
                conf=sr_conf if "sr_det_conf" in st.session_state else conf_threshold,
                max_area_pct=sr_max_a if "sr_det_max_a" in st.session_state else max_area_pct,
                min_area_pct=sr_min_a if "sr_det_min_a" in st.session_state else min_area_pct,
            )
            st.session_state[_det_cache_key] = dets

    detections: list = st.session_state.get(_det_cache_key, [])

    if not detections:
        st.warning("No objects detected on the base frame.")
        return

    # --- Identify the anchor (pipeline-tracked object) ---
    anchor_idx = None
    if anchor_bbox is not None:
        anchor_arr = np.array(anchor_bbox, dtype=np.float32)
        best_iou = 0.0
        for i, det in enumerate(detections):
            # Compute IoU between anchor bbox and detection
            x1 = max(anchor_arr[0], det.bbox_xyxy[0])
            y1 = max(anchor_arr[1], det.bbox_xyxy[1])
            x2 = min(anchor_arr[2], det.bbox_xyxy[2])
            y2 = min(anchor_arr[3], det.bbox_xyxy[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (anchor_arr[2] - anchor_arr[0]) * (anchor_arr[3] - anchor_arr[1])
            area_b = (det.bbox_xyxy[2] - det.bbox_xyxy[0]) * (det.bbox_xyxy[3] - det.bbox_xyxy[1])
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                anchor_idx = i

    if anchor_idx is None:
        st.warning("Could not match the pipeline-tracked object. First detection will be used as anchor.")
        anchor_idx = 0

    anchor_det = detections[anchor_idx]

    # --- Draw annotated frame ---
    ANCHOR_COLOR = (0, 255, 0)   # Green for anchor (BGR)
    TARGET_COLOR = (255, 0, 0)   # Blue for other objects (BGR)
    SELECTED_COLOR = (0, 165, 255)  # Orange for selected target (BGR)

    # Build target options (everything except the anchor)
    target_options = [(i, det) for i, det in enumerate(detections) if i != anchor_idx and det.xyz is not None]

    img_col, ctrl_col = st.columns([1, 1])

    if not target_options:
        st.warning("No other objects with 3D positions detected. Need at least one target object with depth.")
        _show_base_frame_only(base_frame_path, detections, anchor_idx, ANCHOR_COLOR, TARGET_COLOR, img_col)
        return

    if "sr_target_select" not in st.session_state or st.session_state["sr_target_select"] >= len(target_options):
        st.session_state["sr_target_select"] = 0

    selected_target_ui = int(st.session_state["sr_target_select"])
    selected_target_idx, selected_target_det = target_options[selected_target_ui]

    # --- Draw annotated image with target and release markers ---
    bgr = cv2.imread(str(base_frame_path))
    if bgr is None:
        st.error(f"Could not read frame: {base_frame_path}")
        return

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        if i == anchor_idx:
            color = ANCHOR_COLOR
            tag = f"ANCHOR #{i} {det.label}"
            thickness = 3
        elif i == selected_target_idx:
            color = SELECTED_COLOR
            tag = f"TARGET #{i} {det.label}"
            thickness = 3
        else:
            color = TARGET_COLOR
            tag = f"#{i} {det.label}"
            thickness = 2
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            bgr,
            f"{tag} {det.confidence:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    img_h, img_w = bgr.shape[:2]
    pipeline_release_uv = _xyz_to_uv(pipeline_release_xyz, intrinsics, (img_h, img_w))
    if pipeline_release_uv is not None:
        cv2.drawMarker(
            bgr,
            pipeline_release_uv,
            (255, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            bgr,
            "PIPELINE RELEASE",
            (pipeline_release_uv[0] + 8, max(14, pipeline_release_uv[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    custom_release_uv = st.session_state.get("sr_release_uv")
    if custom_release_uv is not None and len(custom_release_uv) == 2:
        try:
            custom_release_uv = (int(custom_release_uv[0]), int(custom_release_uv[1]))
        except Exception:
            custom_release_uv = None

    if custom_release_uv is not None:
        cv2.drawMarker(
            bgr,
            custom_release_uv,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.putText(
            bgr,
            "CUSTOM RELEASE",
            (custom_release_uv[0] + 8, max(14, custom_release_uv[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    rgb_show = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    release_mode_state = st.session_state.get("sr_release_mode", "Auto (offset-shifted)")
    with img_col:
        if release_mode_state.startswith("Custom") and streamlit_image_coordinates is not None:
            st.caption("Click on the frame to set custom release (uses depth + intrinsics).")
            click = streamlit_image_coordinates(
                rgb_show,
                key="sr_release_click",
                use_column_width="always",
            )
            if click and depth_m is not None:
                x_disp = float(click.get("x", -1))
                y_disp = float(click.get("y", -1))
                disp_w = int(click.get("width", img_w) or img_w)
                disp_h = int(click.get("height", img_h) or img_h)

                # streamlit-image-coordinates returns offsets in displayed-image pixels.
                # Map them back to original image pixels before depth back-projection.
                sx = img_w / max(1, disp_w)
                sy = img_h / max(1, disp_h)
                u = int(round(x_disp * sx))
                v = int(round(y_disp * sy))
                u = max(0, min(img_w - 1, u))
                v = max(0, min(img_h - 1, v))
                click_token = f"{u}:{v}"
                if st.session_state.get("sr_last_click_token") != click_token:
                    xyz = pixel_to_xyz(u, v, depth_m, intrinsics)
                    st.session_state["sr_last_click_token"] = click_token
                    st.session_state["sr_release_uv"] = [u, v]
                    if xyz is not None:
                        st.session_state["sr_rel_x"] = float(xyz[0])
                        st.session_state["sr_rel_y"] = float(xyz[1])
                        st.session_state["sr_rel_z"] = float(xyz[2])
                        st.rerun()
            elif release_mode_state.startswith("Custom") and depth_m is None:
                st.warning("Depth data is missing, so click-to-3D is unavailable. Enter XYZ manually.")
        else:
            st.image(
                rgb_show,
                caption="Green=Anchor | Orange=Target | Blue=Other | Magenta=Pipeline release | Red=Custom release",
                use_container_width=True,
            )
            if release_mode_state.startswith("Custom") and streamlit_image_coordinates is None:
                st.info("Install streamlit-image-coordinates to enable click-to-select release on frame.")

    with ctrl_col:
        target_labels = [f"#{i} {det.label} - conf {det.confidence:.2f}" for i, det in target_options]

        selected_target_ui = st.selectbox(
            "Select target object",
            options=list(range(len(target_options))),
            format_func=lambda idx: target_labels[idx],
            key="sr_target_select",
        )
        selected_target_idx, selected_target_det = target_options[selected_target_ui]

        st.caption(f"Anchor (tracked): #{anchor_idx} {anchor_det.label} - {anchor_det.xyz}")
        st.caption(f"Target: #{selected_target_idx} {selected_target_det.label} - {selected_target_det.xyz}")

        if anchor_det.xyz is not None and selected_target_det.xyz is not None:
            offset = selected_target_det.xyz - anchor_det.xyz
            st.caption(f"3D offset: dx={offset[0]:.4f} dy={offset[1]:.4f} dz={offset[2]:.4f} m")

        if pipeline_release_xyz is not None:
            st.caption(
                "Pipeline release: "
                f"x={float(pipeline_release_xyz[0]):.4f}, "
                f"y={float(pipeline_release_xyz[1]):.4f}, "
                f"z={float(pipeline_release_xyz[2]):.4f} m"
            )

        st.markdown("##### Release Position")
        release_mode = st.radio(
            "Release position",
            ["Auto (offset-shifted)", "Custom (click on frame or enter)"],
            key="sr_release_mode",
            horizontal=True,
        )

        custom_release_xyz = None
        if release_mode.startswith("Custom"):
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                rx = st.number_input("X (m)", value=float(st.session_state.get("sr_rel_x", 0.0)), format="%.4f", key="sr_rel_x")
            with rc2:
                ry = st.number_input("Y (m)", value=float(st.session_state.get("sr_rel_y", 0.0)), format="%.4f", key="sr_rel_y")
            with rc3:
                rz = st.number_input("Z (m)", value=float(st.session_state.get("sr_rel_z", 0.0)), format="%.4f", key="sr_rel_z")

            custom_release_xyz = np.array([rx, ry, rz], dtype=np.float32)

            if pipeline_release_xyz is not None and st.button("Use Pipeline Release", key="sr_use_pipeline_release"):
                st.session_state["sr_rel_x"] = float(pipeline_release_xyz[0])
                st.session_state["sr_rel_y"] = float(pipeline_release_xyz[1])
                st.session_state["sr_rel_z"] = float(pipeline_release_xyz[2])
                if pipeline_release_uv is not None:
                    st.session_state["sr_release_uv"] = [pipeline_release_uv[0], pipeline_release_uv[1]]
                st.rerun()

        can_compute = anchor_det.xyz is not None and selected_target_det.xyz is not None
        if st.button("Compute Skill Reuse Trajectory", key="sr_compute", type="primary", disabled=not can_compute):
            with st.spinner(f"Computing trajectory for {selected_target_det.label}..."):
                result = compute_skill_reuse_for_target(
                    original_skill_reuse_path=str(skill_reuse_path),
                    anchor_xyz=anchor_det.xyz,
                    target_xyz=selected_target_det.xyz,
                    release_xyz=custom_release_xyz if release_mode.startswith("Custom") else None,
                    seg_dir=str(seg_dir),
                    output_dir=str(reuse_dir),
                    target_label=selected_target_det.label,
                )
                st.session_state["sr_last_result"] = result
                st.success(f"Trajectory saved: {Path(result['npy_path']).name}")

    # ====================================================================
    # Section 2: Generated Skill Reuse Trajectories (table + actions)
    # ====================================================================
    st.markdown("---")
    st.markdown("#### Generated Skill Reuse Trajectories")

    reuse_dir.mkdir(parents=True, exist_ok=True)
    npy_files = sorted(
        (f for f in reuse_dir.glob("skill_reuse_*.npy") if "_meta" not in f.stem),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not npy_files:
        st.info("No skill reuse trajectories generated yet. Select a target and click Compute.")
        return

    from src.streamlit_template.new_ui.services.Common.robot_action_service import (
        TRANSPORT_CYCLONEDDS,
        get_default_robot_domain_id,
        get_default_robot_transport,
        get_default_vulcanexus_publish_settings,
        get_robot_transport_options,
        publish_cartesian_trajectory_vulcanexus,
        publish_trajectory_dds,
        save_animation,
    )
    from src.streamlit_template.components.sync_viewer import sync_viewer

    urdf_path, robot_config = _resolve_active_robot()
    _dialog_api = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

    def _sync_viewer_safe(
        viewer_data: dict,
        *,
        video_path,
        key: str,
        height: int | None = None,
        play_token: int | None = None,
    ):
        """Call sync_viewer with optional args, falling back for stale cached signatures."""
        kwargs = {
            "viewer_data": viewer_data,
            "video_path": video_path,
            "key": key,
        }
        if height is not None:
            kwargs["height"] = int(height)
        if play_token is not None:
            kwargs["play_token"] = int(play_token)

        try:
            return sync_viewer(**kwargs)
        except TypeError:
            return sync_viewer(viewer_data=viewer_data, video_path=video_path, key=key)

    if _dialog_api is not None:
        @_dialog_api("DMP Plot", width="large")
        def _show_sr_dmp_dialog(stem: str, npy_path: str, grasp_idx: int, release_idx: int):
            st.caption(f"Trajectory: {Path(npy_path).name}")
            try:
                traj_np = np.asarray(np.load(npy_path))
                dmp_viewer_data = _build_dmp_viewer_data(
                    traj_xyz=traj_np,
                    grasp_idx=int(grasp_idx),
                    release_idx=int(release_idx),
                )
                _sync_viewer_safe(
                    viewer_data=dmp_viewer_data,
                    video_path=None,
                    key=f"sr_dmp_dialog_preview_{stem}",
                    height=560,
                )
            except Exception as e:
                st.error(f"Failed to render DMP plot: {e}")

            if st.button("Close", key=f"sr_dmp_dialog_close_{stem}", use_container_width=True):
                st.session_state.pop("sr_open_dmp_dialog", None)
                st.rerun()

        @_dialog_api("Robot Playback", width="large")
        def _show_sr_robot_dialog(stem: str, npy_path: str, grasp_idx: int, release_idx: int):
            st.caption(f"Trajectory: {Path(npy_path).name}")

            _play_key = f"sr_robot_dialog_play_{stem}"
            _static_key = f"sr_robot_dialog_static_data_{stem}"
            _anim_key = f"sr_robot_dialog_anim_data_{stem}"
            _play_token_key = f"sr_robot_dialog_play_token_{stem}"

            if _play_token_key not in st.session_state:
                st.session_state[_play_token_key] = 0

            if _static_key not in st.session_state:
                with st.spinner("Preparing robot preview..."):
                    try:
                        st.session_state[_static_key] = _build_static_robot_viewer_data(
                            urdf_path=urdf_path,
                            robot_config=robot_config,
                        )
                    except Exception as e:
                        st.session_state[_static_key] = {"__error": str(e)}

            if st.button("Start Animation", key=f"sr_robot_dialog_start_{stem}", type="primary", use_container_width=True):
                if _anim_key not in st.session_state:
                    with st.spinner("Computing IK for robot playback..."):
                        try:
                            st.session_state[_anim_key] = _build_robot_viewer_data_for_reuse(
                                skill_reuse_npy=Path(npy_path),
                                urdf_path=urdf_path,
                                robot_config=robot_config,
                                grasp_idx=int(grasp_idx),
                                release_idx=int(release_idx),
                            )
                        except Exception as e:
                            st.session_state[_anim_key] = {"__error": str(e)}
                st.session_state[_play_key] = True
                st.session_state[_play_token_key] = int(st.session_state.get(_play_token_key, 0)) + 1

            show_anim = bool(st.session_state.get(_play_key, False))
            robot_viewer_data = st.session_state.get(_anim_key if show_anim else _static_key, {})
            if isinstance(robot_viewer_data, dict) and robot_viewer_data.get("__error"):
                st.error(f"Failed to render robot playback: {robot_viewer_data['__error']}")
            else:
                _sync_viewer_safe(
                    viewer_data=robot_viewer_data,
                    video_path=None,
                    key=f"sr_robot_dialog_preview_{stem}",
                    height=560,
                    play_token=(int(st.session_state.get(_play_token_key, 0)) if show_anim else None),
                )

            btn1, btn2 = st.columns(2)
            with btn1:
                if st.button("Reset Preview", key=f"sr_robot_dialog_reset_{stem}", use_container_width=True):
                    st.session_state[_play_key] = False
                    st.rerun()

            with btn2:
                if st.button("Close", key=f"sr_robot_dialog_close_{stem}", use_container_width=True):
                    st.session_state.pop("sr_open_robot_dialog", None)
                    st.session_state.pop(_play_key, None)
                    st.session_state.pop(_static_key, None)
                    st.session_state.pop(_anim_key, None)
                    st.session_state.pop(_play_token_key, None)
                    st.rerun()

    table_rows = []
    meta_by_stem = {}

    for npy_file in npy_files:
        stem = npy_file.stem
        meta_file = npy_file.with_name(f"{stem}_meta.json")
        meta = {}
        if meta_file.exists():
            try:
                with open(str(meta_file), "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        meta_by_stem[stem] = meta
        n_pts = int(meta.get("num_points", len(np.load(str(npy_file)))))
        offset = meta.get("offset_3d")
        release = meta.get("release_xyz")

        offset_str = "-"
        if isinstance(offset, list) and len(offset) == 3:
            offset_str = f"[{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]"

        release_str = "Auto"
        if isinstance(release, list) and len(release) == 3:
            release_str = f"[{release[0]:.3f}, {release[1]:.3f}, {release[2]:.3f}]"

        table_rows.append(
            {
                "__stem": stem,
                "__file": npy_file,
                "Trajectory": npy_file.name,
                "Target": meta.get("target_label", stem.replace("skill_reuse_", "")),
                "Points": n_pts,
                "Offset (m)": offset_str,
                "Release (m)": release_str,
                "Updated": datetime.fromtimestamp(npy_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    hdr_cols = st.columns([0.8, 2.3, 1.4, 0.8, 1.7, 1.7, 1.8, 2.2])
    hdr_cols[0].markdown("**Select**")
    hdr_cols[1].markdown("**Trajectory**")
    hdr_cols[2].markdown("**Target**")
    hdr_cols[3].markdown("**Points**")
    hdr_cols[4].markdown("**Offset (m)**")
    hdr_cols[5].markdown("**Release (m)**")
    hdr_cols[6].markdown("**Updated**")
    hdr_cols[7].markdown("**Action**")

    selected_indices = []

    for idx, row in enumerate(table_rows):
        stem = str(row["__stem"])
        npy_file = Path(row["__file"])
        meta = meta_by_stem.get(stem, {})
        sr_release_default = pipeline_release_idx if pipeline_release_idx is not None else 0
        grasp_idx = int(meta.get("grasp_idx", 0))
        release_idx = int(meta.get("release_idx", sr_release_default))

        row_cols = st.columns([0.8, 2.3, 1.4, 0.8, 1.7, 1.7, 1.8, 2.2])
        with row_cols[0]:
            sel_key = f"sr_sel_{stem}"
            if sel_key not in st.session_state:
                st.session_state[sel_key] = False
            checked = st.checkbox("Select", key=sel_key, label_visibility="collapsed")
            if checked:
                selected_indices.append(idx)

        row_cols[1].write(str(row["Trajectory"]))
        row_cols[2].write(str(row["Target"]))
        row_cols[3].write(str(row["Points"]))
        row_cols[4].write(str(row["Offset (m)"]))
        row_cols[5].write(str(row["Release (m)"]))
        row_cols[6].write(str(row["Updated"]))

        with row_cols[7]:
            a1, a2, a3 = st.columns(3)

            with a1:
                if st.button("👁", key=f"sr_dmp_icon_{stem}", help="Show DMP plot", use_container_width=True):
                    st.session_state["sr_open_dmp_dialog"] = {
                        "stem": stem,
                        "npy_path": str(npy_file),
                        "grasp_idx": int(grasp_idx),
                        "release_idx": int(release_idx),
                    }
                    st.session_state.pop("sr_open_robot_dialog", None)

            with a2:
                if st.button("🤖", key=f"sr_robot_icon_{stem}", help="Show robot animation", use_container_width=True):
                    st.session_state["sr_open_robot_dialog"] = {
                        "stem": stem,
                        "npy_path": str(npy_file),
                        "grasp_idx": int(grasp_idx),
                        "release_idx": int(release_idx),
                    }
                    st.session_state[f"sr_robot_dialog_play_{stem}"] = False
                    st.session_state.pop("sr_open_dmp_dialog", None)

            with a3:
                try:
                    csv_bytes = _trajectory_npy_to_csv_bytes(npy_file)
                    st.download_button(
                        "⬇",
                        data=csv_bytes,
                        file_name=f"{stem}.csv",
                        mime="text/csv",
                        key=f"sr_csv_download_{stem}",
                        help="Download CSV",
                        use_container_width=True,
                    )
                except Exception:
                    st.button(
                        "⬇",
                        key=f"sr_csv_download_disabled_{stem}",
                        disabled=True,
                        use_container_width=True,
                    )

        st.markdown("---")

    _dmp_req = st.session_state.get("sr_open_dmp_dialog")
    _robot_req = st.session_state.get("sr_open_robot_dialog")

    if _dialog_api is not None:
        if isinstance(_dmp_req, dict):
            _show_sr_dmp_dialog(
                stem=str(_dmp_req.get("stem", "dmp")),
                npy_path=str(_dmp_req.get("npy_path", "")),
                grasp_idx=int(_dmp_req.get("grasp_idx", 0)),
                release_idx=int(_dmp_req.get("release_idx", 0)),
            )
        elif isinstance(_robot_req, dict):
            _show_sr_robot_dialog(
                stem=str(_robot_req.get("stem", "robot")),
                npy_path=str(_robot_req.get("npy_path", "")),
                grasp_idx=int(_robot_req.get("grasp_idx", 0)),
                release_idx=int(_robot_req.get("release_idx", 0)),
            )
    else:
        if _dmp_req or _robot_req:
            st.info("Your Streamlit version does not support modal dialogs; update Streamlit to enable popup windows.")

    selected_file = npy_files[selected_indices[0]] if selected_indices else None
    selected_stem = selected_file.stem if selected_file is not None else None

    st.markdown("##### Execute Selected Trajectory")
    if len(selected_indices) > 1:
        st.warning("Multiple trajectories are selected. Actions will use the first selected row.")

    anim_default = selected_stem or "skill_reuse_animation"
    anim_name = st.text_input("Animation name", value=anim_default, key="sr_action_anim_name")

    transport_options = get_robot_transport_options()
    default_transport = get_default_robot_transport()
    vulcanexus_defaults = get_default_vulcanexus_publish_settings()
    transport = st.selectbox(
        "Transport",
        transport_options,
        index=transport_options.index(default_transport),
        key="sr_action_transport",
    )

    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col2:
        dds_domain = st.number_input(
            "Domain ID",
            value=get_default_robot_domain_id(),
            min_value=0,
            max_value=232,
            step=1,
            key="sr_action_dds_domain",
        )
    if transport == TRANSPORT_CYCLONEDDS:
        with cfg_col1:
            dds_topic = st.text_input("Topic", value="/joint_trajectory", key="sr_action_dds_topic")
        discovery_server = ""
        repeat_count = 1
        status_topic = ""
        status_wait_sec = 0.0
    else:
        with cfg_col1:
            dds_topic = st.text_input(
                "Topic",
                value=str(vulcanexus_defaults["topic"]),
                key="sr_action_vulcanexus_topic",
            )
        st.caption("Proven path: same-LAN Vulcanexus Discovery Server on `server1` plus `/learned_trajectory` PoseArray.")
        discovery_server = st.text_input(
            "Discovery Server",
            value=str(vulcanexus_defaults["discovery_server"]),
            key="sr_action_vulcanexus_discovery_server",
            help="Known-good LAN default is server1 on 192.168.0.10:14520. Override with env if needed.",
        )
        repeat_count = st.number_input(
            "Repeat Count",
            value=int(vulcanexus_defaults["repeat_count"]),
            min_value=1,
            max_value=100,
            step=1,
            key="sr_action_vulcanexus_repeat",
        )
        status_topic = st.text_input(
            "Status Topic",
            value=str(vulcanexus_defaults["status_topic"]),
            key="sr_action_vulcanexus_status_topic",
        )
        status_wait_sec = st.number_input(
            "Wait For Status (sec)",
            value=float(vulcanexus_defaults["status_wait_sec"]),
            min_value=0.0,
            max_value=60.0,
            step=1.0,
            key="sr_action_vulcanexus_status_wait",
            help="Set to 0 to skip waiting for the edge executor status.",
        )

    act_col1, act_col2 = st.columns(2)

    with act_col1:
        if st.button("Save Animation", key="sr_save_animation_table", use_container_width=True):
            if selected_file is None:
                st.warning("Select one trajectory in the first column before saving animation.")
            else:
                try:
                    traj_np = np.load(str(selected_file))
                    traj_points = [{"x": float(r[0]), "y": float(r[1]), "z": float(r[2])} for r in traj_np]
                    ok, msg = save_animation(urdf_path, traj_points, anim_name)
                    if ok:
                        st.success(f"Animation saved: {anim_name}")
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Failed to save animation: {e}")

    with act_col2:
        if st.button("Push to Robot", key="sr_push_robot_table", type="primary", use_container_width=True):
            if selected_file is None:
                st.warning("Select one trajectory in the first column before pushing to robot.")
            elif transport == TRANSPORT_CYCLONEDDS and not Path(urdf_path).exists():
                st.error(f"URDF not found: {urdf_path}")
            elif transport != TRANSPORT_CYCLONEDDS:
                with st.spinner("Publishing Cartesian trajectory via Vulcanexus LAN..."):
                    try:
                        cart_path = np.load(str(selected_file))
                        ok, msg = publish_cartesian_trajectory_vulcanexus(
                            cart_path=cart_path,
                            topic_name=dds_topic,
                            domain_id=int(dds_domain),
                            discovery_server=discovery_server.strip() or None,
                            status_topic=status_topic.strip() or None,
                            status_timeout_sec=float(status_wait_sec),
                            repeat=int(repeat_count),
                        )
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Failed: {e}")
            else:
                selected_meta = meta_by_stem.get(selected_file.stem, {})
                grasp_idx = int(selected_meta.get("grasp_idx", 0))
                release_idx = int(selected_meta.get("release_idx", pipeline_release_idx or 0))

                with st.spinner("Computing IK and publishing..."):
                    try:
                        ik_result = compute_ik_for_reuse_traj(
                            skill_reuse_npy=str(selected_file),
                            urdf_path=urdf_path,
                            robot_config=robot_config,
                            grasp_idx=grasp_idx,
                            release_idx=release_idx,
                        )
                        q_traj = ik_result["q_traj"]
                        timestamps = ik_result["frame_timestamps"]

                        from src.streamlit_template.core.Common.robot_playback import load_chain

                        chain = load_chain(urdf_path)
                        joint_names = [
                            link.name
                            for link in chain.links
                            if link.name != "base_link" and hasattr(link, "bounds") and link.bounds != (None, None)
                        ]
                        if not joint_names:
                            joint_names = [f"joint_{i}" for i in range(q_traj.shape[1])]

                        ok, msg = publish_trajectory_dds(
                            joint_names=joint_names,
                            q_traj=q_traj,
                            timestamps=timestamps,
                            topic_name=dds_topic,
                            domain_id=int(dds_domain),
                        )
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Failed: {e}")


def _show_base_frame_only(base_frame_path, detections, anchor_idx, anchor_color, other_color, img_col):
    """Draw base frame with anchor highlighted when no targets available."""
    bgr = cv2.imread(str(base_frame_path))
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = anchor_color if i == anchor_idx else other_color
        tag = f"ANCHOR #{i}" if i == anchor_idx else f"#{i}"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{tag} {det.label} {det.confidence:.2f}", (x1, max(y1 - 6, 12)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with img_col:
        st.image(rgb, caption="Green=Anchor (tracked) | Blue=Other", use_container_width=True)
