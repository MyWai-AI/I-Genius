"""
Pipeline Page - Dedicated page for running the VILMA processing pipeline.
Matches the original UI logic from src/streamlit_template/ui.
"""
import streamlit as st
import os
from pathlib import Path
import hashlib
from streamlit_clickable_images import clickable_images

from src.streamlit_template.new_ui.components.Common.stepper_bar import StepperBar, PIPELINE_STEPS
from src.streamlit_template.new_ui.services.Generic.pipeline_service import (
    handle_hands,
    handle_objects,
    handle_trajectory,
    handle_dmp,
    handle_robot,
)
from src.streamlit_template.ui.helpers import encode_images
from src.streamlit_template.new_ui.components.Common import frame_viewer as _frame_viewer


def _get_data_paths(video_path: str):
    """Get pipeline data paths based on video path."""
    vid_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
    base = Path("data") / "Generic"
    
    # Get frames dir from cached result if available
    video_key = f"frames_{video_path}"
    cached_result = st.session_state.get(video_key)
    if cached_result and cached_result.get("output_dir"):
        frames_dir = Path(cached_result["output_dir"])
    else:
        frames_dir = base / "frames" / vid_hash
    
    # Debug path resolution
    # st.info removed
    
    return {
        "frames": frames_dir,
        "hands": base / "hands" / vid_hash,
        "objects": base / "objects" / vid_hash,
        "trajectories": base / "dmp" / vid_hash,
        "dmp": base / "dmp" / vid_hash,
    }


def _resolve_active_robot():
    """Determine active robot URDF and Config."""
    # Check custom upload — prefer custom_robot_dir (set by mywai page), fallback to active_custom_robot_dir
    _custom = (
        st.session_state.get("custom_robot_dir")
        or st.session_state.get("active_custom_robot_dir")
    )
    if _custom and os.path.exists(_custom):
        # Sync both keys so BAG/SVO/Generic all see the same value
        st.session_state["custom_robot_dir"] = _custom
        st.session_state["active_custom_robot_dir"] = _custom
        model_dir = Path(_custom)
        urdfs = list(model_dir.rglob("*.urdf"))
        urdf_path = str(urdfs[0]) if urdfs else "data/Common/robot_models/openarm/openarm.urdf"
    else:
        model_dir = Path("data/Common/robot_models/openarm")
        # Generalization: Scan for any URDF in the default folder (ignoring generated sanitized ones)
        # This allows replacing the default robot file without changing code.
        candidates = [p for p in model_dir.glob("*.urdf") if "_sanitized" not in p.name]
        
        if candidates:
            # Pick the first valid URDF found (e.g. openarm.urdf)
            urdf_path = str(candidates[0])
        else:
            # Fallback if directory is empty (should not happen in standard setup)
            urdf_path = str(model_dir / "openarm.urdf")
    
    config_path = Path(urdf_path).parent / "config.json"
    robot_config = {}
    if config_path.exists():
        try:
             import json
             with open(config_path, "r") as f:
                 robot_config = json.load(f)
        except:
             pass

    # Check for separate animations.json or animation.json
    anim_candidates = ["animations.json", "animation.json"]
    for ac in anim_candidates:
        anim_path = Path(urdf_path).parent / ac
        if anim_path.exists():
            try:
                import json
                with open(anim_path, "r") as f:
                    anims = json.load(f)
                    # Merge into config
                    if "animations" not in robot_config:
                        robot_config["animations"] = {}
                    robot_config["animations"].update(anims.get("animations", anims))
                # removed debug printing
            except Exception:
                pass

    return urdf_path, robot_config


def render_pipeline_page():
    """Render the pipeline processing page."""
    # Force clear any stale elements from previous pages
    placeholder = st.empty()
    placeholder.empty()
    
    # Header removed
    # st.markdown...
    
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state["selected_platform"] = st.session_state.get("pipeline_source", "local")
            st.rerun()
    
    video_path = st.session_state.get("local_video_path")
    
    # Hide specific lingering mywai widgets from previous pages.
    st.markdown("""
        <style>
        /* Keep this scoped: do not hide all primary buttons on pipeline pages. */
        .st-key-mywai_logout,
        .st-key-mywai_back_sites,
        .st-key-mywai_back_areas,
        .st-key-mywai_back_eq,
        .st-key-mywai_back_tasks,
        .st-key-mywai_run_pipeline {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if not video_path or not os.path.exists(video_path):
        # Warning removed
        if st.button("Go to Local Upload"):
            st.session_state["selected_platform"] = "local"
            st.rerun()
        return
    
    # Video info removed
    
    paths = _get_data_paths(video_path)

    # Create directories
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    
    # Initialize session state
    if "step_results" not in st.session_state:
        st.session_state.step_results = {}
    if "pipeline_statuses" not in st.session_state:
        st.session_state.pipeline_statuses = ["pending"] * len(PIPELINE_STEPS)
    if "selected_step" not in st.session_state:
        st.session_state.selected_step = None
    if "clicked_index" not in st.session_state:
        st.session_state.clicked_index = -1
    if "selected_frame" not in st.session_state:
        st.session_state.selected_frame = None
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False

    statuses = st.session_state.pipeline_statuses
    is_running = st.session_state.pipeline_running
    
    # Stepper Bar - Clickable only if not running
    stepper = StepperBar(PIPELINE_STEPS)
    clicked_step = stepper.display(statuses=statuses, key_prefix="pipeline_", clickable=not is_running)
    
    st.markdown("---")

    # === STATE UPDATE HELPER ===
    def switch_step(step_index):
        """Handle context switching when a step is selected (auto or click)."""
        st.session_state.selected_step = step_index
        st.session_state.clicked_index = -1
        st.session_state.selected_frame = None
        
        # If switching to Hands/Objects, Auto-Select first image?
        # User requested: "in hands and object section show frames... viewer part updated when we change step"
        # "dmp and trajectory and robot the video should be shown"
        
        # Get results
        step_keys = ["hands", "objects", "trajectory", "dmp", "robot"]
        if step_index < len(step_keys):
            step_key = step_keys[step_index]
            res = st.session_state.step_results.get(step_key)
            
            if step_index in [0, 1] and res and res.get("paths"):
                # HANDS or OBJECTS: Select first frame
                st.session_state.clicked_index = 0
                st.session_state.selected_frame = str(res["paths"][0])
            else:
                # OTHERS: Clear frame to show video
                st.session_state.selected_frame = None

    
    # === AUTO-RUN LOGIC ===
    if is_running:
        # Find first non-completed step
        next_step = None
        for i, s in enumerate(statuses):
            if s == "pending":
                next_step = i
                break
            elif s == "running":
                next_step = i
                break
            elif s == "error":
                st.error("Pipeline stopped due to error.")
                st.session_state.pipeline_running = False
                # st.rerun() # Let user see error
                return
        
        if next_step is None:
            # All done
            # remove st.success
            st.session_state.pipeline_running = False
            # Auto-select first step (Hands/Step 1)
            switch_step(0) # Hands
            st.rerun()
            return
            
        # If status is pending, mark running and refresh
        if statuses[next_step] == "pending":
            statuses[next_step] = "running"
            st.session_state.pipeline_statuses = statuses
            st.rerun()
            return
            
        # If status is running, Execute
        if statuses[next_step] == "running":
            try:
                # Select step VISUAL only (no frame logic yet as results not ready)
                st.session_state.selected_step = next_step
                st.session_state.selected_frame = None  # Clear previous step's image
                
                if next_step == 0:
                    # Check if frames extracted. If not (e.g. from MYWAI), extract them now.
                    idx_path = paths["frames"] / "frames_index.csv"
                    if not idx_path.exists():
                        with st.spinner("Extracting frames (auto)..."):
                            # Use same extraction logic as Local page/Viewer
                            from src.streamlit_template.new_ui.services.Generic.frame_service import extract_and_cache_frames
                            video_p = st.session_state.get("local_video_path")
                            if video_p and os.path.exists(video_p):
                                extract_and_cache_frames(video_p)
                            else:
                                st.error(f"Cannot extract frames. Video path not found: {video_p}")
                                st.session_state.pipeline_running = False
                                return


                    handle_hands(paths["frames"], paths["hands"])
                elif next_step == 1:
                    _video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8] if video_path else ""
                    _local_src_hash = st.session_state.get("selected_tracking_source_hash")
                    _shared_bbox = st.session_state.get("selected_tracking_bbox_xyxy")
                    _legacy_bbox = st.session_state.get(f"tracking_bbox_{_video_hash}") if _video_hash else None

                    if _local_src_hash == _video_hash and isinstance(_shared_bbox, (list, tuple)) and len(_shared_bbox) == 4:
                        _selected_bbox = list(_shared_bbox)
                    else:
                        _selected_bbox = _legacy_bbox if isinstance(_legacy_bbox, (list, tuple)) and len(_legacy_bbox) == 4 else None

                    _selected_model = st.session_state.get("selected_object_model_path", "yolov8x.pt")
                    _conf = float(st.session_state.get("selected_object_conf_threshold", 0.05))
                    _max_area = float(st.session_state.get("selected_object_max_area_pct", 15.0))
                    _min_area = float(st.session_state.get("selected_object_min_area_pct", 0.0001))
                    _size_ratio = float(st.session_state.get("selected_object_bbox_size_ratio", 4.0))

                    _selected_label = st.session_state.get("selected_tracking_label") if _local_src_hash == _video_hash else None
                    handle_objects(
                        paths["frames"],
                        paths["objects"],
                        model_path=_selected_model,
                        tracking_bbox_xyxy=_selected_bbox,
                        tracking_label=_selected_label,
                        confidence_threshold=_conf,
                        max_area_pct=_max_area,
                        min_area_pct=_min_area,
                        bbox_size_ratio=_size_ratio,
                    )
                elif next_step == 2:
                    with st.spinner("Extracting trajectories..."):
                        handle_trajectory(paths["frames"], paths["hands"], paths["objects"], paths["trajectories"])
                elif next_step == 3:
                    with st.spinner("Generating DMP..."):
                        handle_dmp(paths["trajectories"], paths["dmp"])
                elif next_step == 4:
                    with st.spinner("Computing robot playback..."):
                        u_path, r_config = _resolve_active_robot()
                        handle_robot(
                            paths["dmp"], 
                            traj_dir=paths["trajectories"],
                            urdf_path=u_path,
                            robot_config=r_config
                        )
                
                statuses[next_step] = "completed"
                st.session_state.pipeline_statuses = statuses
                
                # Update viewer to show result of this step immediately?
                # Yes, "show where is the progress".
                switch_step(next_step)
                
                st.rerun()
                
            except Exception as e:
                import traceback
                st.error(f"Step failed details:\\n{traceback.format_exc()}")
                
                statuses[next_step] = "error"
                st.session_state.pipeline_statuses = statuses
                st.session_state.pipeline_running = False
    
    else:
        # === INTERACTIVE MODE ===
        if clicked_step is not None and clicked_step != st.session_state.selected_step:
            switch_step(clicked_step)
            st.rerun()
    
    # Viewer Section
    selected_step = st.session_state.get("selected_step")
    step_results = st.session_state.get("step_results", {})

    # Determine result type early to decide layout
    rtype = None
    if selected_step is not None:
        step_keys = ["hands", "objects", "trajectory", "dmp", "robot"]
        if selected_step < len(step_keys):
            step_key = step_keys[selected_step]
            res = step_results.get(step_key)
            if res:
                rtype = res.get("type")

    # SPECIAL FULL-WIDTH LAYOUT FOR ROBOT 3D
    if rtype == "robot3d":
        # Render Sync Viewer (Full Width)
        # Import moved here to keep localized
        from src.streamlit_template.components.sync_viewer import sync_viewer
        from src.streamlit_template.core.Common.robot_playback import load_chain, get_configured_joint_names, augment_traj_points_with_gripper
        import base64
        import json
        import numpy as np
        import xml.etree.ElementTree as ET

        urdf_path_local, robot_config = _resolve_active_robot()


        def hydrate_urdf_with_meshes(urdf_path, base_dir=None):
            """
            Reads a URDF file, finds all <mesh filename="..."> tags, 
            reads the referenced files (assuming local paths or package:// syntaz),
            encodes them as Base64, and returns the modified URDF string with Data URIs.
            """
            if base_dir is None:
                base_dir = Path(urdf_path).parent

            try:
                tree = ET.parse(urdf_path)
                root = tree.getroot()
                
                # Namespaces can be tricky in ElementTree, but usually URDF is simple.
                # Find all mesh tags
                for mesh in root.iter('mesh'):
                    filename = mesh.get('filename')
                    if filename:
                        # Resolve path
                        # Handle package:// syntax
                        real_path = None
                        if filename.startswith('package://'):
                            # localized assumption: package://comau_ns16hand_support/meshes/...
                            # maps to data/Common/robot_models/meshes/... or similar?
                            # Based on user context: data/Common/robot_models/urdf/comau_ns16hand_debug.urdf exists.
                            # And existing meshes are likely in data/Common/robot_models/meshes or relative.
                            
                            # Heuristic: Strip package prefix and look in data/Common/robot_models/
                            # e.g. package://comau_ns16hand_support/meshes/Link_1.STL
                            # -> data/Common/robot_models/comau_ns16hand_support/meshes/Link_1.STL ?
                            # Or just look relative to the URDF file known location?
                            
                            # Let's try finding the file recursively or relative to known root
                            subpath = filename.replace('package://', '') 
                            
                            # Standard assumption: package://pkg_name/meshes/file.stl
                            # If we just uploaded the robot, we might have dropped "pkg_name" folder 
                            # and just have meshes/file.stl relative to the URDF.
                            
                            candidates = [
                                Path("data/Common/robot_models") / subpath,
                                Path("data/Common/robot_models/urdf") / subpath,
                                Path(urdf_path).parent / subpath,
                            ]

                            # Heuristic: Try to strip the first component (the package name)
                            # e.g. jaco_description/meshes/joint_1.stl -> meshes/joint_1.stl
                            parts = subpath.split('/')
                            if len(parts) > 1:
                                # Try stripping just the first folder
                                suffix_no_pkg = "/".join(parts[1:])
                                candidates.append(Path(urdf_path).parent / suffix_no_pkg)
                                
                                # Try stripping everything up to "meshes" if it exists
                                if 'meshes' in parts:
                                    mesh_idx = parts.index('meshes')
                                    suffix_meshes = "/".join(parts[mesh_idx:])
                                    candidates.append(Path(urdf_path).parent / suffix_meshes)
                                    candidates.append(Path("data/Common/robot_models/urdf") / suffix_meshes)
                                    
                                # Try just the filename flat
                                candidates.append(Path(urdf_path).parent / parts[-1])

                            for c in candidates:
                                # print(f"DEBUG: Checking {c} (exists: {c.exists()})")
                                if c.exists():
                                    real_path = c
                                    # print removed
                                    break
                        else:
                            # Relative path (or absolute-looking path like /jaco_description/...)
                            # If it starts with slash, Windows Path joining might treat it as absolute root
                            # So we strip leading slash to force relative behavior from parent
                            clean_filename = filename.lstrip("/").lstrip("\\")
                            
                            # Candidates for relative search
                            candidates = [
                                (Path(urdf_path).parent / clean_filename).resolve(),
                                (Path(urdf_path).parent / filename).resolve(), # Try original just in case
                                # Try searching recursively for the filename only
                                Path(urdf_path).parent / Path(clean_filename).name 
                            ]
                            
                            # Also add recursive search in the parent directory
                            # This is expensive but robust for finding "joint_1.stl" anywhere in the zip structure
                            try:
                                found_files = list(Path(urdf_path).parent.rglob(Path(clean_filename).name))
                                if found_files:
                                    candidates.extend(found_files)
                            except:
                                pass

                            real_path = None
                            for c in candidates:
                                if c.exists():
                                    real_path = c
                                    break
                        
                        if real_path and real_path.exists():
                            # st.write(f"DEBUG: Hydrating {filename} from {real_path}")
                            with open(real_path, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                                # Determine mime type roughly
                                ext = real_path.suffix.lower()
                                mime = "model/stl" # Default for these robots
                                if ext == ".obj": mime = "model/obj"
                                elif ext == ".dae": mime = "model/vnd.collada+xml"
                                
                                # Replace filename with Data URI
                                mesh.set('filename', f"data:{mime};base64,{b64_data}")
                        else:
                            # print removed
                            st.warning(f"Warning: Could not find mesh file for {filename}")


                # Return string
                out_xml = ET.tostring(root, encoding='unicode')
                data_uri_count = out_xml.count("filename=\"data:")
                # debug prints removed
                return out_xml
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f"Failed to hydrate URDF: {e}")
                return None




        if os.path.exists(urdf_path_local):
            
            # Use the new hydration function
            urdf_content = hydrate_urdf_with_meshes(urdf_path_local)
            
            if urdf_content:
                urdf_b64 = base64.b64encode(urdf_content.encode('utf-8')).decode("utf-8")
                urdf_uri = f"data:text/xml;base64,{urdf_b64}"
            else:
                st.error("Could not process URDF file.")
                return

            # Prepare trajectory data
            q_traj = res.get("q_traj")
            timestamps = res.get("frame_timestamps")
            # 2. Dynamic Joint Name Mapping (Robust)
            try:
                joint_names_ordered = get_configured_joint_names(urdf_path_local)
                print(f"[DEBUG] Dynamic Joint Names from Chain: {joint_names_ordered}")
            except Exception as e:
                print(f"[ERROR] Helper failed: {e}. Attempting raw chain fallback.")
                try:
                    chain = load_chain(urdf_path_local)
                    joint_names_ordered = [l.name for l in chain.links]
                    print(f"[DEBUG] Fallback to Raw Chain Names: {joint_names_ordered}")
                except:
                    # Final fallback
                    joint_names_ordered = ["base_link", "Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6", "flange", "tool"]
            
            traj_points = []
            if q_traj is not None and timestamps is not None:
                    for t, q in zip(timestamps, q_traj):
                    # Ensure q is valid and has at least same length as names
                        if isinstance(q, (list, tuple, np.ndarray)) and len(q) == len(joint_names_ordered):
                             q_map = {name: float(val) for name, val in zip(joint_names_ordered, q)}
                             traj_points.append({"time": float(t), "q": q_map})
            
            # Augment trajectory with gripper open/close
            cart_path = res.get("cart_path")
            if len(traj_points) > 0 and cart_path is not None:
                augment_traj_points_with_gripper(traj_points, cart_path, urdf_path_local)
                print(f"DEBUG: Full-Width Robot Trajectory processed. Total frames: {len(traj_points)}")
                print(f"DEBUG: First frame mapping (Full-Width): {traj_points[0]['q']}")

            viewer_data = {
                "models": [
                    {
                        "entityName": "Robot",
                        "loadType": "URDF",
                        "path": urdf_uri,
                        "trajectory": traj_points,
                        "rotation": robot_config.get("rotation", [-90, 0, 0]),
                        "position": robot_config.get("position", [0, 0, 0]),
                        "scale": robot_config.get("scale", [1, 1, 1]),
                        "homePose": robot_config.get("home_pose", {}),
                        "linkColors": robot_config.get("link_colors", {}),
                        "animations": robot_config.get("animations", {})
                    }
                ],
                "images": [],
                "timeseries": []
            }
            
            # === ACTION BUTTONS ===
            if traj_points:
                with st.popover("Action", use_container_width=False):
                    st.markdown("#### Generate New Trajectory")
                    if st.button("Generate New Trajectory", key="gen_save_result", type="primary"):
                        st.session_state["skill_reuse_source_platform"] = "svo_pipeline"
                        st.session_state["selected_platform"] = "skill_reuse"
                        st.rerun()

            if video_path:
                    # Video Base64 encoding
                    video_src = str(video_path)
                    if os.path.exists(video_src):
                        with open(video_src, "rb") as vf:
                            vid_b64 = base64.b64encode(vf.read()).decode("utf-8")
                        video_src = f"data:video/mp4;base64,{vid_b64}"

                    sync_viewer(video_path=video_src, viewer_data=viewer_data, key=f"sync_v_robot", autoplay=False)
            else:
                st.warning("No video available for synchronization.")
        else:
                st.error(f"URDF file not found: {urdf_path_local}")

        return # Exit early, skip standard columns

    # SPECIAL FULL-WIDTH LAYOUT FOR DMP 3D
    if rtype == "dmp3d":
        from src.streamlit_template.components.sync_viewer import sync_viewer
        import base64
        import numpy as np

        timestamps = res.get("timestamps")
        dmp_dir = paths["dmp"]
        dmp_xyz_path = dmp_dir / "object_xyz_dmp.npy"
        
        traj_data = None
        if dmp_xyz_path.exists() and timestamps:
            try:
                yg = np.load(str(dmp_xyz_path))
                if len(timestamps) == len(yg):
                    traj_data = {
                        "x": yg[:, 0].tolist(),
                        "y": yg[:, 1].tolist(),
                        "z": yg[:, 2].tolist(),
                        "t": timestamps
                    }
                else:
                    st.warning(f"DMP Data Mismatch: {len(yg)} points vs {len(timestamps)} timestamps.")
            except Exception as e:
                st.error(f"Error loading DMP data: {e}")

        if traj_data and video_path:
            viewer_data = {
                "mode": "dmp",
                "dmpTrajectory": traj_data,
                "models": [],
                "images": [],
                "timeseries": []
            }
            
            video_src = str(video_path)
            if os.path.exists(video_src):
                with open(video_src, "rb") as vf:
                    vid_b64 = base64.b64encode(vf.read()).decode("utf-8")
                video_src = f"data:video/mp4;base64,{vid_b64}"
            
            sync_viewer(video_path=video_src, viewer_data=viewer_data, key=f"sync_v_dmp")
            return

    # STANDARD LAYOUT (Left: Viewer, Right: Results)
    left, right = st.columns([1, 1])
    
    # LEFT COLUMN — VIEWER
    with left:
        # Viewer header removed
        
        selected_frame = st.session_state.get("selected_frame")
        clicked_index = st.session_state.get("clicked_index", -1)
        
        if selected_frame:
            p = Path(selected_frame)
            if p.exists():
                st.image(str(p.resolve()), width="stretch")
            else:
                # info removed
                st.session_state.selected_frame = None
            
            # Navigation buttons for image steps
            if selected_step in [0, 1]:  # Hands, Objects
                step_key = "hands" if selected_step == 0 else "objects"
                res = step_results.get(step_key)
                if res and res.get("paths"):
                    img_paths = [str(p) for p in res["paths"]]
                    col_prev, _, col_next = st.columns([1, 6, 1])
                    
                    with col_prev:
                        if clicked_index > 0 and st.button("⬅ Prev", width="stretch"):
                            st.session_state.clicked_index -= 1
                            st.session_state.selected_frame = img_paths[st.session_state.clicked_index]
                            st.rerun()
                    
                    with col_next:
                        if clicked_index < len(img_paths) - 1 and st.button("Next ➡", width="stretch"):
                            st.session_state.clicked_index += 1
                            st.session_state.selected_frame = img_paths[st.session_state.clicked_index]
                            st.rerun()
        
        else:
            # Video player removed as per user request
            # st.info removed
            pass
    
    # RIGHT COLUMN — RESULTS
    with right:
        # Results header removed
        
        if selected_step is None:
            # Info removed
            return
        
        step_keys = ["hands", "objects", "trajectory", "dmp", "robot"]
        step_key = step_keys[selected_step] if selected_step < len(step_keys) else None
        
        res = step_results.get(step_key)
        if not res:
            if is_running and statuses[selected_step] == "running":
                pass
            else:
                pass # Info removed
            return
        
        rtype = res.get("type")
        
        # IMAGE GRIDS (Hands, Objects)
        if rtype in ["hands", "objects"]:
            img_paths = [str(p) for p in res["paths"]]

            if rtype == "objects":
                _mode = res.get("selection_mode")
                _model = res.get("model_used")
                _meta = res.get("run_metadata")
                if _mode or _model:
                    _model_name = Path(_model).name if _model else "unknown"
                    _caption = f"mode: {_mode or 'unknown'} | model: {_model_name}"
                    if _meta:
                        _label = _meta.get("tracking_label") or "—"
                        _ratio = _meta.get("bbox_size_ratio", "—")
                        _n_det = _meta.get("total_detections", "—")
                        _ts = _meta.get("timestamp", "")
                        _caption += f" | label: {_label} | size ratio: {_ratio}× | detections: {_n_det}"
                        if _ts:
                            _caption += f" | ran: {_ts}"
                    st.caption(_caption)

            ready = _frame_viewer.sync_timeline_touch_state(img_paths, key_prefix=f"pipeline_{rtype}") if hasattr(_frame_viewer, "sync_timeline_touch_state") else True
            if not ready:
                st.info("Adjust the timeline range to show frames.")
                if hasattr(_frame_viewer, "render_timeline_range_control"):
                    _frame_viewer.render_timeline_range_control(img_paths, key_prefix=f"pipeline_{rtype}")
                return

            display_paths = _frame_viewer.timeline_trim_paths(img_paths, key_prefix=f"pipeline_{rtype}")
            if not display_paths:
                return

            encoded = encode_images(tuple(display_paths))
            grid_key = f"viewer_grid_{selected_step}_{len(display_paths)}"

            clicked = clickable_images(
                encoded,
                titles=[f"Frame {i+1}" for i in range(len(encoded))],
                key=grid_key,
                div_style={
                    "display": "grid",
                    "grid-template-columns": "repeat(auto-fill, minmax(23%, 1fr))",
                    "gap": "10px",
                    "height": "600px",
                    "overflow-y": "auto",
                },
                img_style={
                    "width": "100%",
                    "border-radius": "8px",
                    "cursor": "pointer",
                },
            )

            if clicked > -1 and clicked != st.session_state.get("clicked_index"):
                st.session_state.selected_frame = display_paths[clicked]
                st.session_state.clicked_index = clicked
                st.rerun()

            if hasattr(_frame_viewer, "render_timeline_range_control"):
                _frame_viewer.render_timeline_range_control(img_paths, key_prefix=f"pipeline_{rtype}")
        
        # TRAJECTORY 2D SUBPLOTS with timestamp slider
        elif rtype == "trajectory2d":
            st.caption("Extracted trajectory coordinates (X, Y, Z) over time.")
            timestamps = res.get("timestamps", [])
            if timestamps:
                # Get current time from state or default to min
                t_min, t_max = min(timestamps), max(timestamps)
                
                # Retrieve value from session state if available, else min
                # Note: We use the widget key 'trajectory_time_slider' directly
                if "trajectory_time_slider" not in st.session_state:
                    st.session_state.trajectory_time_slider = t_min
                
                current_time = st.session_state.trajectory_time_slider
                
                # Update seek time if changed
                if current_time != st.session_state.get("last_traj_check"):
                    st.session_state["video_seek_time"] = current_time
                    st.session_state["last_traj_check"] = current_time
                    _show_frame_at_time(paths, current_time)
                    # No rerun needed here as we will render plot with new time immediately?
                    # Streamlit reruns on widget change anyway.
                
                # Create figure with vertical line at selected time
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go_local
                
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("X Position [m]", "Y Position [m]", "Z Position [m]"),
                )
                
                x_vals, y_vals, z_vals = res["x"], res["y"], res["z"]
                
                # Add traces
                fig.add_trace(go_local.Scatter(x=timestamps, y=x_vals, mode="lines+markers", name="X", line=dict(color="blue")), row=1, col=1)
                fig.add_trace(go_local.Scatter(x=timestamps, y=y_vals, mode="lines+markers", name="Y", line=dict(color="green")), row=2, col=1)
                fig.add_trace(go_local.Scatter(x=timestamps, y=z_vals, mode="lines+markers", name="Z", line=dict(color="red")), row=3, col=1)
                
                # Add vertical line at selected time
                for row in range(1, 4):
                    fig.add_vline(x=current_time, line_dash="dash", line_color="purple", line_width=2, row=row, col=1)
                
                # Fixed margins for alignment
                fig.update_layout(
                    height=450, 
                    showlegend=True, 
                    hovermode="x unified",
                    margin=dict(l=60, r=20, t=20, b=20)
                )
                # Force X-axis range to match slider exactly (remove auto-padding)
                fig.update_xaxes(range=[t_min, t_max])
                fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                fig.update_yaxes(title_text="X [m]", row=1, col=1)
                fig.update_yaxes(title_text="Y [m]", row=2, col=1)
                fig.update_yaxes(title_text="Z [m]", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Render Slider with CSS padding to match Plotly margin (l=60)
                st.markdown("""
                    <style>
                    div[data-testid="stSlider"] {
                        padding-left: 60px !important;
                        padding-right: 70px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                st.slider(
                        "Select Time (s)",
                        label_visibility="collapsed",
                        min_value=t_min,
                        max_value=t_max,
                        # value is controlled by key
                        step=0.1,
                        key="trajectory_time_slider",
                        format="%.2f s"
                    )

            else:
                st.plotly_chart(res["fig"], use_container_width=True)
        
        # TRAJECTORY IMAGE (fallback)
        elif rtype == "dmp_image":
            img_path = Path(res.get("image", ""))
            if img_path.exists():
                st.image(str(img_path.resolve()), use_container_width=True)
            else:
                st.warning("Trajectory image not found.")
        
        # DMP 3D INTERACTIVE PLOT with click-to-seek
        elif rtype == "dmp3d":
            # Fallback if standard layout is used (e.g. if sync_viewer failed in top-level block)
            st.plotly_chart(res["fig"], use_container_width=True)
        
        # ROBOT 3D ANIMATION
        elif rtype == "robot3d":
            from src.streamlit_template.components.sync_viewer import sync_viewer
            from src.streamlit_template.core.Common.robot_playback import load_chain, get_configured_joint_names, augment_traj_points_with_gripper
            import base64
            import json
            import numpy as np

            # 1. Resolve correct (Custom) URDF path
            urdf_path_local, _ = _resolve_active_robot()
            print(f"[DEBUG] Pipeline Page Viewer resolving robot: {urdf_path_local}")

            if os.path.exists(urdf_path_local):
                # Hydrate matches
                urdf_content = hydrate_urdf_with_meshes(urdf_path_local)
                if urdf_content:
                     urdf_b64 = base64.b64encode(urdf_content.encode('utf-8')).decode("utf-8")
                     urdf_uri = f"data:text/xml;base64,{urdf_b64}"
                else:
                     # Fallback to file read if hydration fails
                     with open(urdf_path_local, "rb") as f:
                        urdf_b64 = base64.b64encode(f.read()).decode("utf-8")
                     urdf_uri = f"data:text/xml;base64,{urdf_b64}"

                q_traj = res.get("q_traj")
                timestamps = res.get("frame_timestamps")
                
                # 2. Dynamic Joint Name Mapping
                # Load chain to get the ACTUAL joint names that correspond to q_traj indices
                try:
                    # Use helper function to get JOINT names, not Link names
                    joint_names_ordered = get_configured_joint_names(urdf_path_local)
                    print(f"[DEBUG] Dynamic Joint Names from Chain: {joint_names_ordered}")
                except Exception as e:
                    print(f"[ERROR] Could not load chain for mapping: {e}. Attempting raw fallback.")
                    try:
                        chain = load_chain(urdf_path_local)
                        joint_names_ordered = [l.name for l in chain.links]
                        print(f"[DEBUG] Fallback to Raw Chain Names: {joint_names_ordered}")
                    except Exception as e2:
                        print(f"[CRITICAL] Chain load failed completely: {e2}")
                        joint_names_ordered = ["base_link", "Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6", "flange", "tool"]

                
                traj_points = []
                if q_traj is not None and timestamps is not None:
                    # Ensure q_traj items are lists/arrays of floats
                    for t, q in zip(timestamps, q_traj):
                        if len(q) == len(joint_names_ordered):
                             q_map = {name: float(val) for name, val in zip(joint_names_ordered, q)}
                             traj_points.append({"time": float(t), "q": q_map})
                        else:
                             pass
                
                # Augment with gripper open/close
                cart_path = res.get("cart_path")
                if len(traj_points) > 0 and cart_path is not None:
                    augment_traj_points_with_gripper(traj_points, cart_path, urdf_path_local)
                    print(f"DEBUG: Trajectory processed. Total frames: {len(traj_points)}")
                    print(f"DEBUG: First frame mapping: {traj_points[0]['q']}")
                else:
                     print("DEBUG: No trajectory points generated.")

                viewer_data = {
                    "models": [
                        {
                            "entityName": "Robot",
                            "loadType": "URDF",
                        "path": urdf_uri,
                        "trajectory": traj_points,
                        "rotation": robot_config.get("rotation", [-90, 0, 0]),
                        "position": robot_config.get("position", [0, 0, 0]),
                        "scale": robot_config.get("scale", [1, 1, 1]),
                        "homePose": robot_config.get("home_pose", {}),
                        "linkColors": robot_config.get("link_colors", {}),
                        "animations": robot_config.get("animations", {})
                    }
                    ],
                    "images": [],
                    "timeseries": []
                }
                
                # Use sync_viewer instead of plotly
                # video_path is available in this scope (from line 59)
                if video_path:
                     # sync_viewer expects a path or URL. If local, Streamlit serves it?
                     # Streamlit components in IFrames cannot access local paths C:\...
                     # We must read video and pass as Base64 if not web-hosted 
                     # OR use Streamlit's media serving mechanism if possible (but tricky with Components)
                     # SAFEST: Base64 for now, or just try path if browser has access (unlikely)
                     
                     # 1. Try passing path directly (might fail in custom component if not served)
                     # 2. Convert video to base64 (Heavy!)
                     # 3. Use st.video and try to sync via hacks? No, we built a component.
                     
                     # Let's use Base64 for the video too to be safe, despite memory cost.
                     # Or check if video_path is a URL.
                     video_src = str(video_path)
                     if os.path.exists(video_src):
                         # Just use base64 for now to guarantee functionality
                         with open(video_src, "rb") as vf:
                             vid_b64 = base64.b64encode(vf.read()).decode("utf-8")
                         video_src = f"data:video/mp4;base64,{vid_b64}"
                     
                     sync_viewer(video_path=video_src, viewer_data=viewer_data, key=f"sync_v_{step_key}")
                else:
                    st.warning("No video available for synchronization.")
            else:
                 st.error(f"URDF file not found: {urdf_path_local}")

            
            # Add Streamlit slider for video sync
            frame_timestamps = res.get("frame_timestamps")
            num_frames = res.get("num_frames", 100)
            
            # Slider removed per user request


def _show_frame_at_time(paths: dict, target_time: float):
    """Find and display the frame closest to target_time."""
    import csv
    
    frames_index = paths["frames"] / "frames_index.csv"
    if not frames_index.exists():
        return
    
    # Find closest frame
    best_frame = None
    best_diff = float("inf")
    
    with frames_index.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_sec"])
            diff = abs(t - target_time)
            if diff < best_diff:
                best_diff = diff
                best_frame = row["filename"]
    
    if best_frame:
        frame_path = paths["frames"] / best_frame
        if frame_path.exists():
            st.session_state.selected_frame = str(frame_path)
            st.session_state["video_seek_time"] = target_time
