"""
SVO Pipeline Page - Dedicated page for running the SVO processing pipeline.
Same stepper UI as BAG pipeline, using SVO-specific service handlers.
"""
import streamlit as st
import os
from pathlib import Path
import numpy as np

try:
    from st_clickable_images import clickable_images
except ImportError:
    try:
        from streamlit_clickable_images import clickable_images
    except ImportError:
        clickable_images = None

from src.streamlit_template.new_ui.components.Common.stepper_bar import StepperBar, PIPELINE_STEPS
from src.streamlit_template.new_ui.services.SVO.svo_pipeline_service import (
    handle_svo_hands,
    handle_svo_objects,
    handle_svo_trajectory,
    handle_svo_dmp,
    handle_svo_robot,
)
from src.streamlit_template.ui.helpers import encode_images


def _get_svo_data_paths():
    """Get SVO pipeline data paths."""
    if "svo_base_path" in st.session_state and st.session_state["svo_base_path"]:
        base = Path(st.session_state["svo_base_path"])
    else:
        base = Path("data") / "SVO" # Root

    session_id = st.session_state.get("svo_session_id")
    
    return {
        "frames": base / "frames", # Legacy/Root access if needed
        "base": base,
        "session_id": session_id
    }


def _resolve_active_robot():
    """Determine active robot URDF and Config."""
    import json

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

    # Custom uploaded robot?
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
                except:
                    pass
            config = _load_animations(urdf_dir, config)
            return urdf_path, config

    # Default
    default_urdf = "data/Common/robot_models/openarm/openarm.urdf"
    urdf_dir = Path(default_urdf).parent
    config = {}
    config_path = urdf_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except:
            pass
    config = _load_animations(urdf_dir, config)
    return default_urdf, config


def render_svo_pipeline_page():
    """Render the SVO pipeline processing page."""
    placeholder = st.empty()
    placeholder.empty()

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state["selected_platform"] = st.session_state.get("pipeline_source", "local")
            st.rerun()

    # Hide any persistent buttons from previous pages (Streamlit ghost components bug)
    st.markdown("""
        <style>
        /* Hide old primary buttons generically */
        button[data-testid="stBaseButton-primary"] {
            display: none !important;
        }
        /* Hide specific lingering secondary buttons from external_page */
                                                .st-key-platform_run_pipeline {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Note: SVO selection logic might differ. Assuming user selects SVO session ID elsewhere or uploads.
    # For now, simplistic check: if no session ID, warn? Or just check data/SVO exists?
    # BAG page relies on 'bag_file_path'. Here we rely on 'svo_session_id' or 'svo_file_path'?
    # Let's assume 'svo_file_path' if relevant, or just proceed if data/SVO exists.
    
    paths = _get_svo_data_paths()
    base = paths["base"]

    if not base.exists():
       st.warning(f"SVO data directory not found at {base}. Please ensure data is available.")
       if st.button("Refresh"):
           st.rerun()
       return

    # Create directories
    for p in paths.values():
        if isinstance(p, Path):
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

    # Stepper Bar
    stepper = StepperBar(PIPELINE_STEPS)
    clicked_step = stepper.display(statuses=statuses, key_prefix="svo_pipeline_", clickable=not is_running)

    # === LOGS REMOVED ===

    st.markdown("---")
    
    # === STATE UPDATE HELPER ===
    def switch_step(step_index):
        st.session_state.selected_step = step_index
        st.session_state.clicked_index = -1
        st.session_state.selected_frame = None

        step_keys = ["hands", "objects", "trajectory", "dmp", "robot"]
        if step_index < len(step_keys):
            step_key = step_keys[step_index]
            res = st.session_state.step_results.get(step_key)

            if step_index in [0, 1] and res and res.get("paths"):
                st.session_state.clicked_index = 0
                st.session_state.selected_frame = str(res["paths"][0])
            else:
                st.session_state.selected_frame = None

    # === AUTO-RUN LOGIC ===
    if is_running:
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
                return

        if next_step is None:
            # st.success removed
            st.session_state.pipeline_running = False
            switch_step(0)
            st.rerun()
            return

        if statuses[next_step] == "pending":
            statuses[next_step] = "running"
            st.session_state.pipeline_statuses = statuses
            st.rerun()
            return

        if statuses[next_step] == "running":
            try:
                st.session_state.selected_step = next_step
                st.session_state.selected_frame = None  # Clear previous step's image
                sess = paths.get("session_id")

                if next_step == 0:
                    st.session_state.pipeline_logs = [] # Clear logs on new run
                    res = handle_svo_hands(base, session_id=sess)
                elif next_step == 1:
                    res = handle_svo_objects(base, session_id=sess)
                elif next_step == 2:
                    res = handle_svo_trajectory(base, session_id=sess)
                elif next_step == 3:
                    res = handle_svo_dmp(base, session_id=sess)
                elif next_step == 4:
                    u_path, r_config = _resolve_active_robot()
                    res = handle_svo_robot(base, session_id=sess, urdf_path=u_path, robot_config=r_config)

                if res is None:
                    # Step failed or returned None
                    statuses[next_step] = "error"
                    st.session_state.pipeline_statuses = statuses
                    st.session_state.pipeline_running = False
                else:
                    statuses[next_step] = "completed"
                    st.session_state.pipeline_statuses = statuses
                    switch_step(next_step)
                st.rerun()

            except Exception as e:
                import traceback
                st.error(f"Step failed details:\n{traceback.format_exc()}")
                statuses[next_step] = "error"
                st.session_state.pipeline_statuses = statuses
                st.session_state.pipeline_running = False

    else:
        # === INTERACTIVE MODE ===
        if clicked_step is not None and clicked_step != st.session_state.selected_step:
            switch_step(clicked_step)
            st.rerun()

    # ========================= VIEWER SECTION =========================
    selected_step = st.session_state.get("selected_step")
    step_results = st.session_state.get("step_results", {})

    rtype = None
    if selected_step is not None:
        step_keys = ["hands", "objects", "trajectory", "dmp", "robot"]
        if selected_step < len(step_keys):
            step_key = step_keys[selected_step]
            res = step_results.get(step_key)
            if res:
                rtype = res.get("type")

    # --- ROBOT 3D (full width) ---
    if rtype == "robot3d":
        from src.streamlit_template.components.sync_viewer import sync_viewer
        from src.streamlit_template.core.Common.robot_playback import load_chain, get_configured_joint_names, augment_traj_points_with_gripper
        import base64
        import json
        import xml.etree.ElementTree as ET

        urdf_path_local, robot_config = _resolve_active_robot()

        def hydrate_urdf_with_meshes(urdf_path, base_dir=None):
            if base_dir is None:
                base_dir = Path(urdf_path).parent
            try:
                tree = ET.parse(urdf_path)
                root = tree.getroot()
                for mesh in root.iter('mesh'):
                    filename = mesh.get('filename')
                    if filename:
                        real_path = None
                        if filename.startswith('package://'):
                            subpath = filename.replace('package://', '')
                            candidates = [
                                Path("data/Common/robot_models") / subpath,
                                Path("data/Common/robot_models/urdf") / subpath,
                                Path(urdf_path).parent / subpath,
                            ]
                            parts = subpath.split('/')
                            if len(parts) > 1:
                                candidates.append(Path(urdf_path).parent / "/".join(parts[1:]))
                                if 'meshes' in parts:
                                    mesh_idx = parts.index('meshes')
                                    candidates.append(Path(urdf_path).parent / "/".join(parts[mesh_idx:]))
                                candidates.append(Path(urdf_path).parent / parts[-1])
                            for c in candidates:
                                if c.exists():
                                    real_path = c
                                    break
                        else:
                            clean_filename = filename.lstrip("/").lstrip("\\")
                            candidates = [
                                (Path(urdf_path).parent / clean_filename).resolve(),
                                Path(urdf_path).parent / Path(clean_filename).name,
                            ]
                            try:
                                found = list(Path(urdf_path).parent.rglob(Path(clean_filename).name))
                                candidates.extend(found)
                            except:
                                pass
                            for c in candidates:
                                if c.exists():
                                    real_path = c
                                    break

                        if real_path and real_path.exists():
                            with open(real_path, "rb") as f:
                                b64_data = base64.b64encode(f.read()).decode("utf-8")
                                ext = real_path.suffix.lower()
                                mime = "model/stl"
                                if ext == ".obj": mime = "model/obj"
                                elif ext == ".dae": mime = "model/vnd.collada+xml"
                                mesh.set('filename', f"data:{mime};base64,{b64_data}")
                return ET.tostring(root, encoding='unicode')
            except Exception as e:
                st.error(f"Failed to hydrate URDF: {e}")
                return None

        if os.path.exists(urdf_path_local):
            urdf_content = hydrate_urdf_with_meshes(urdf_path_local)
            if urdf_content:
                urdf_b64 = base64.b64encode(urdf_content.encode('utf-8')).decode("utf-8")
                urdf_uri = f"data:text/xml;base64,{urdf_b64}"
            else:
                st.error("Could not process URDF file.")
                return

            q_traj = res.get("q_traj")
            timestamps = res.get("frame_timestamps")

            try:
                joint_names_ordered = get_configured_joint_names(urdf_path_local)
            except:
                try:
                    chain = load_chain(urdf_path_local)
                    joint_names_ordered = [l.name for l in chain.links]
                except:
                    joint_names_ordered = ["base_link", "Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6", "flange", "tool"]

            traj_points = []
            if q_traj is not None and timestamps is not None:
                for t, q in zip(timestamps, q_traj):
                    if len(q) == len(joint_names_ordered):
                        q_map = {name: float(val) for name, val in zip(joint_names_ordered, q)}
                        traj_points.append({"time": float(t), "q": q_map})

            # Augment with gripper open/close at grasp, re-open at release
            cart_path = res.get("cart_path")
            grasp_idx = res.get("grasp_idx")
            release_idx = res.get("release_idx")
            if len(traj_points) > 0 and cart_path is not None:
                augment_traj_points_with_gripper(traj_points, cart_path, urdf_path_local, grasp_idx=grasp_idx, release_idx=release_idx)

            viewer_data = {
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

            # Video handling for SVO? Usually SVO has video too.
            # Assuming video path is stored or not used yet.
            # If standard video path exists, we can use it.
            # For now, simplistic check or skip video.
            video_path = st.session_state.get("local_video_path")

            if video_path and os.path.exists(video_path):
                with open(video_path, "rb") as vf:
                    vid_b64 = base64.b64encode(vf.read()).decode("utf-8")
                video_src = f"data:video/mp4;base64,{vid_b64}"
                sync_viewer(video_path=video_src, viewer_data=viewer_data, key="sync_v_svo_robot")
            else:
                sync_viewer(viewer_data=viewer_data, video_path=None, key="sync_v_svo_robot_novid")

            # === ACTION BUTTONS ===
            if traj_points:
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
                with st.popover("Action", use_container_width=False):
                    st.markdown("#### Save Animation")
                    anim_name = st.text_input("Animation name", value="PipelineAction", key="svo_anim_name")
                    if st.button("Save Animation", key="svo_save_anim"):
                        ok, msg = save_animation(urdf_path_local, traj_points, anim_name)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

                    st.markdown("---")
                    st.markdown("#### Push to Robot")
                    transport_options = get_robot_transport_options()
                    default_transport = get_default_robot_transport()
                    vulcanexus_defaults = get_default_vulcanexus_publish_settings()
                    transport = st.selectbox(
                        "Transport",
                        transport_options,
                        index=transport_options.index(default_transport),
                        key="svo_robot_transport",
                    )
                    dds_domain = st.number_input(
                        "Domain ID",
                        value=get_default_robot_domain_id(),
                        min_value=0,
                        max_value=232,
                        step=1,
                        key="svo_dds_domain",
                    )
                    if transport == TRANSPORT_CYCLONEDDS:
                        dds_topic = st.text_input("Topic", value="/joint_trajectory", key="svo_dds_topic")
                    else:
                        st.caption("Proven path: same-LAN Vulcanexus Discovery Server on `server1` plus `/learned_trajectory` PoseArray.")
                        dds_topic = st.text_input(
                            "Topic",
                            value=str(vulcanexus_defaults["topic"]),
                            key="svo_vulcanexus_topic",
                        )
                        discovery_server = st.text_input(
                            "Discovery Server",
                            value=str(vulcanexus_defaults["discovery_server"]),
                            key="svo_vulcanexus_discovery_server",
                            help="Known-good LAN default is server1 on 192.168.0.10:14520. Override with env if needed.",
                        )
                        repeat_count = st.number_input(
                            "Repeat Count",
                            value=int(vulcanexus_defaults["repeat_count"]),
                            min_value=1,
                            max_value=100,
                            step=1,
                            key="svo_vulcanexus_repeat",
                        )
                        status_topic = st.text_input(
                            "Status Topic",
                            value=str(vulcanexus_defaults["status_topic"]),
                            key="svo_vulcanexus_status_topic",
                        )
                        status_wait_sec = st.number_input(
                            "Wait For Status (sec)",
                            value=float(vulcanexus_defaults["status_wait_sec"]),
                            min_value=0.0,
                            max_value=60.0,
                            step=1.0,
                            key="svo_vulcanexus_status_wait",
                            help="Set to 0 to skip waiting for the edge executor status.",
                        )
                    if st.button("🤖 Push to Robot", key="svo_push_robot"):
                        if transport == TRANSPORT_CYCLONEDDS:
                            ok, msg = publish_trajectory_dds(
                                joint_names=joint_names_ordered,
                                q_traj=q_traj,
                                timestamps=timestamps,
                                topic_name=dds_topic,
                                domain_id=int(dds_domain),
                            )
                        else:
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
        else:
            st.error(f"URDF file not found: {urdf_path_local}")
        return

    # --- DMP 3D (full width) ---
    if rtype == "dmp3d":
        from src.streamlit_template.components.sync_viewer import sync_viewer
        import base64

        timestamps = res.get("timestamps")
        sess = paths.get("session_id")
        dmp_dir = base / "dmp" / sess if sess else base / "dmp"
        # Prefer skill_reuse_traj (matches robot playback), fallback to object_xyz_dmp
        dmp_xyz_path = dmp_dir / "skill_reuse_traj.npy"
        if not dmp_xyz_path.exists():
            dmp_xyz_path = dmp_dir / "object_xyz_dmp.npy"

        traj_data = None
        if dmp_xyz_path.exists() and timestamps:
            try:
                yg = np.load(str(dmp_xyz_path))
                if len(timestamps) == len(yg):
                    # Build segment + marker data for the sync_viewer
                    reach_len = res.get("reach_len", 0)
                    move_len = res.get("move_len", 0)
                    total = len(yg)

                    grasp_idx = res.get("grasp_idx", reach_len)
                    release_idx = res.get("release_idx", reach_len + 1 + move_len)

                    segments = []
                    # Reach (Blue): 0 .. reach_len-1
                    if reach_len > 0:
                        segments.append({"startIdx": 0, "endIdx": reach_len - 1, "color": "#4285F4", "label": "Reach"})
                    # Move (Green): grasp_idx+1 .. release_idx-1
                    if move_len > 0:
                        segments.append({"startIdx": grasp_idx + 1, "endIdx": release_idx - 1, "color": "#34A853", "label": "Move"})
                    # Post-Release (Orange): release_idx .. end
                    if release_idx < total:
                        segments.append({"startIdx": release_idx, "endIdx": total - 1, "color": "#FF9800", "label": "Post-Release"})

                    markers = [
                        {"label": "Start", "index": 0, "color": "#00cc96"},
                        {"label": "Grasp", "index": grasp_idx, "color": "#111111"},
                        {"label": "Release", "index": release_idx, "color": "#ef553b"},
                        {"label": "End", "index": total - 1, "color": "#9C27B0"},
                    ]

                    traj_data = {
                        "x": yg[:, 0].tolist(),
                        "y": yg[:, 1].tolist(),
                        "z": yg[:, 2].tolist(),
                        "t": timestamps,
                        "segments": segments,
                        "markers": markers,
                    }
            except Exception as e:
                st.error(f"Error loading DMP data: {e}")

        if traj_data:
            viewer_data = {
                "mode": "dmp",
                "dmpTrajectory": traj_data,
                "models": [],
                "images": [],
                "timeseries": [],
            }

            video_path = st.session_state.get("local_video_path")
            video_src = None
            if video_path and os.path.exists(video_path):
                with open(video_path, "rb") as vf:
                    vid_b64 = base64.b64encode(vf.read()).decode("utf-8")
                video_src = f"data:video/mp4;base64,{vid_b64}"

            sync_viewer(video_path=video_src, viewer_data=viewer_data, key="sync_v_svo_dmp")
            return

    # --- STANDARD LAYOUT (Left: Viewer, Right: Results) ---
    left, right = st.columns([1, 1])

    with left:
        selected_frame = st.session_state.get("selected_frame")
        clicked_index = st.session_state.get("clicked_index", -1)

        # Trajectory step: auto-sync frame with timeline slider
        if selected_step == 2:
            res_traj = step_results.get("trajectory")
            if res_traj and res_traj.get("paths"):
                traj_paths = [str(p) for p in res_traj["paths"]]
                slider_val = st.session_state.get("trajectory_time_slider", 0)
                timestamps = res_traj.get("timestamps", [])
                # Find closest frame index to slider value
                if timestamps and traj_paths:
                    frame_idx = min(int(slider_val), len(traj_paths) - 1)
                    frame_idx = max(0, frame_idx)
                    synced_frame = traj_paths[frame_idx]
                    st.image(synced_frame, width="stretch", caption=f"Frame {frame_idx}")
        elif selected_frame:
            p = Path(selected_frame)
            if p.exists():
                st.image(str(p.resolve()), width="stretch")
            else:
                # removed st.info
                st.session_state.selected_frame = None

            if selected_step in [0, 1]:
                nav_keys = {0: "hands", 1: "objects"}
                step_key = nav_keys[selected_step]
                res_nav = step_results.get(step_key)
                if res_nav and res_nav.get("paths"):
                    img_paths = [str(p) for p in res_nav["paths"]]
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
            pass

    with right:
        if selected_step is None:
            return

        step_keys = ["hands", "objects", "trajectory", "dmp", "robot"]
        step_key = step_keys[selected_step] if selected_step < len(step_keys) else None
        res = step_results.get(step_key)

        if not res:
            pass

        rtype = res.get("type")

        # IMAGE GRIDS (Hands, Objects)
        if rtype in ["hands", "objects"]:
            if clickable_images is None:
                st.error(
                    "Missing dependency: install `st-clickable-images` (or `streamlit-clickable-images`) "
                    "to enable frame selection."
                )
                return
            img_paths = [str(p) for p in res["paths"]]
            encoded = encode_images(tuple(img_paths))
            grid_key = f"svo_viewer_grid_{selected_step}_{len(img_paths)}"

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
                st.session_state.selected_frame = img_paths[clicked]
                st.session_state.clicked_index = clicked
                st.rerun()

        # TRAJECTORY 2D SUBPLOTS
        elif rtype == "trajectory2d":
            st.caption("Reconstructed object trajectory (X, Y, Z) with grasp/release markers.")
            timestamps = res.get("timestamps", [])
            if timestamps:
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go_local

                t_min, t_max = min(timestamps), max(timestamps)

                if "trajectory_time_slider" not in st.session_state:
                    st.session_state.trajectory_time_slider = float(t_min)

                current_time = st.session_state.trajectory_time_slider

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("X Position [m]", "Y Position [m]", "Z Position [m]"),
                )

                x_vals, y_vals, z_vals = res["x"], res["y"], res["z"]
                grasp_idx, release_idx = res.get("grasp_idx"), res.get("release_idx")

                fig.add_trace(go_local.Scatter(x=timestamps, y=x_vals, mode="lines+markers", name="X", line=dict(color="blue")), row=1, col=1)
                fig.add_trace(go_local.Scatter(x=timestamps, y=y_vals, mode="lines+markers", name="Y", line=dict(color="green")), row=2, col=1)
                fig.add_trace(go_local.Scatter(x=timestamps, y=z_vals, mode="lines+markers", name="Z", line=dict(color="red")), row=3, col=1)

                # Current frame slider line
                for row in range(1, 4):
                    fig.add_vline(x=current_time, line_dash="dash", line_color="purple", line_width=2, row=row, col=1)

                # Grasp and Release markers
                if grasp_idx is not None and grasp_idx < len(timestamps):
                    fig.add_vline(x=timestamps[grasp_idx], line_dash="dash", line_color="black", line_width=2,
                                  annotation_text="Grasp", annotation_position="top right", annotation_font=dict(size=12, color="black"))
                if release_idx is not None and release_idx < len(timestamps):
                    fig.add_vline(x=timestamps[release_idx], line_dash="dot", line_color="#EA8600", line_width=2,
                                  annotation_text="Release", annotation_position="top right", annotation_font=dict(size=12, color="#EA8600"))

                fig.update_layout(height=450, showlegend=True, hovermode="x unified", margin=dict(l=60, r=20, t=20, b=20))
                fig.update_xaxes(range=[t_min, t_max])
                fig.update_xaxes(title_text="Frame", row=3, col=1)

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                    <style>
                    div[data-testid="stSlider"] {
                        padding-left: 60px !important;
                        padding-right: 70px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.slider(
                    "Select Frame",
                    label_visibility="collapsed",
                    min_value=float(t_min),
                    max_value=float(t_max),
                    step=1.0,
                    key="trajectory_time_slider",
                    format="%.0f",
                )
            else:
                st.plotly_chart(res["fig"], use_container_width=True)

        # DMP 3D (fallback)
        elif rtype == "dmp3d":
            st.plotly_chart(res["fig"], use_container_width=True)

        # ROBOT 3D (fallback)
        elif rtype == "robot3d":
            pass  # Robot visualization rendered above
