# viewer_ui.py — Bottom viewer (left preview + right results)
# FINAL VERSION — COMAU ROBOT ALWAYS VISIBLE

import streamlit as st
from streamlit_clickable_images import clickable_images
from pathlib import Path
import plotly.graph_objects as go

from .helpers import encode_images
from src.streamlit_template.core.Common.robot_playback import (
    cached_meshes_for_pose,
    compute_robot_workspace_bounds,
    get_robot_home_position,
)



# MAIN VIEWER

def render_viewer():
    left, right = st.columns([1, 1])

    selected_step = st.session_state.get("selected_step")
    step_results = st.session_state.get("step_results", {})

    # RESET IMAGE STATE WHEN STEP CHANGES

    if "last_selected_step" not in st.session_state:
        st.session_state.last_selected_step = None

    if st.session_state.last_selected_step != selected_step:
        st.session_state.selected_frame = None
        st.session_state.clicked_index = -1
        st.session_state.last_selected_step = selected_step

    # AUTO-SELECT FIRST IMAGE FRAME

    if selected_step in [0, 1, 2]:
        res = step_results.get(selected_step)
        paths = res.get("paths") if res else None
        if paths and st.session_state.get("selected_frame") is None:
            st.session_state.selected_frame = str(paths[0])
            st.session_state.clicked_index = 0


    # LEFT COLUMN — VIEWER

    with left:
        st.markdown("### Viewer")

        selected_frame = st.session_state.get("selected_frame")
        clicked_index = st.session_state.get("clicked_index", -1)

        if selected_frame:
            p = Path(selected_frame)
            if p.exists():
                st.image(str(p), use_container_width=True)
            else:
                st.info("Image no longer available. Please rerun pipeline.")
                st.session_state.selected_frame = None

            # Navigation buttons
            if selected_step in [0, 1, 2]:
                col_prev, _, col_next = st.columns([1, 6, 1])
                paths = [str(p) for p in step_results[selected_step]["paths"]]

                with col_prev:
                    if clicked_index > 0 and st.button("⬅ Prev", use_container_width=True):
                        st.session_state.clicked_index -= 1
                        st.session_state.selected_frame = paths[st.session_state.clicked_index]
                        st.rerun()

                with col_next:
                    if clicked_index < len(paths) - 1 and st.button("Next ➡", use_container_width=True):
                        st.session_state.clicked_index += 1
                        st.session_state.selected_frame = paths[st.session_state.clicked_index]
                        st.rerun()

        elif st.session_state.get("uploaded_video"):
            st.video(str(st.session_state.uploaded_video))
        else:
            st.info("Upload a video to begin.")

    # RIGHT COLUMN — RESULTS

    with right:
        st.markdown("### Results")

        if selected_step is None:
            st.info("Select a pipeline step.")
            return

        res = step_results.get(selected_step)
        if not res:
            st.info("No results available.")
            return

        rtype = res.get("type")

 
        # IMAGE GRIDS

        if rtype in ["frames", "hands", "objects"]:
            paths = [str(p) for p in res["paths"]]
            
            # Show detection count for objects
            if rtype == "objects" and "total_detections" in res:
                st.caption(f"📦 Total detections: {res['total_detections']} across {len(paths)} frames")
            
            encoded = encode_images(tuple(paths))

            # Dynamic key based on step and number of images to force refresh
            grid_key = f"viewer_grid_{selected_step}_{len(paths)}"

            clicked = clickable_images(
                encoded,
                titles=[f"Frame {i+1}" for i in range(len(encoded))],
                key=grid_key,
                div_style={
                    "display": "grid",
                    "grid-template-columns": "repeat(auto-fill, minmax(22%, 1fr))",
                    "gap": "10px",
                    "height": "300px",
                    "overflow-y": "auto",
                },
                img_style={
                    "width": "100%",
                    "border-radius": "8px",
                    "cursor": "pointer",
                },
            )

            if clicked > -1 and clicked != st.session_state.get("clicked_index"):
                st.session_state.selected_frame = paths[clicked]
                st.session_state.clicked_index = clicked
                st.rerun()


        # TRAJECTORY IMAGE

        
        # fOR THE GENERATED TRAJECTORY 
        # elif rtype == "trajectory":
        #     img_path = Path(res["paths"][0])
        #     if img_path.exists():
        #         st.image(str(img_path), use_container_width=True)
        #     else:
        #         st.warning("Trajectory image missing.")

        
        # FOR THE SVO TRAJECTORY 
        elif rtype == "dmp_image":
            img = res.get("image")
            if img and Path(img).exists():
                st.image(img, use_container_width=True)
            else:
                st.warning("DMP image not found.")
                

        # DMP 3D PLOT

        elif rtype == "dmp3d":
            st.plotly_chart(res["fig"], use_container_width=True)

        # ROBOT 3D VIEW — COMAU ONLY (FIXED)

        elif rtype == "robot3d":

            #  ALWAYS prefer robot_data (RViz-style)
            robot_data = st.session_state.get("robot_data")

            # 🔒 Fallback if Streamlit wiped state
            if not robot_data:
                st.plotly_chart(res["fig"], use_container_width=True)
                return

            urdf_path = robot_data["urdf_path"]
            q_traj = robot_data["q_traj"]

            if len(q_traj) == 0:
                st.error("Empty joint trajectory.")
                return
            
            st.markdown("#### 🎛 Robot Control Mode")

            st.session_state.robot_control_mode = st.radio(
                "Control Mode",
                ["Trajectory", "Manual Joints"],
                horizontal=True,
            )

            st.markdown("#### 🦾 Manual Joint Control")

            robot_data = st.session_state.get("robot_data")
            q_traj = robot_data["q_traj"]

            num_joints = q_traj.shape[1]

            if "robot_joints" not in st.session_state:
                # Initialize with home position instead of trajectory start
                from src.streamlit_template.core.Common.robot_playback import get_robot_home_position
                try:
                    st.session_state.robot_joints = get_robot_home_position(urdf_path)
                except:
                    st.session_state.robot_joints = q_traj[0].copy()

            cols = st.columns(2)

            for j in range(num_joints):
                with cols[j % 2]:
                    st.session_state.robot_joints[j] = st.slider(
                        label=f"Joint {j}",
                        min_value=-3.14,
                        max_value=3.14,
                        value=float(st.session_state.robot_joints[j]),
                        step=0.01,
                    )

            #  READ MANUAL JOINTS AFTER SLIDERS
            q_manual = st.session_state.robot_joints

            if st.session_state.robot_control_mode == "Manual Joints":
                q0 = q_manual
            else:
                q0 = q_traj[0]


            traces = cached_meshes_for_pose(
                urdf_path,
                tuple(q0.tolist())
            )

            if not traces:
                st.error("No robot meshes returned.")
                return

            # Compute dynamic workspace bounds for the robot
            try:
                bounds = compute_robot_workspace_bounds(urdf_path, num_samples=30)
                x_range = bounds["x_range"]
                y_range = bounds["y_range"]
                z_range = bounds["z_range"]
            except Exception:
                # Fallback to default bounds if computation fails
                x_range = [-1.5, 1.5]
                y_range = [-1.5, 1.5]
                z_range = [0.0, 2.0]

            fig = go.Figure()

            for t in traces:
                # Extract RGBA color from the trace
                r, g, b, a = t["rgba"]
                # Convert to RGB string for Plotly (0-255 scale)
                color_str = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                
                fig.add_trace(
                    go.Mesh3d(
                        x=t["x"],
                        y=t["y"],
                        z=t["z"],
                        i=t["i"],
                        j=t["j"],
                        k=t["k"],
                        color=color_str,
                        opacity=a,
                        flatshading=True,
                        name=t.get("name", "Link"),
                        showlegend=True,
                    )

                )

            fig.update_layout(
                scene=dict(
                    aspectmode="data",
                    xaxis=dict(range=x_range, title="X"),
                    yaxis=dict(range=y_range, title="Y"),
                    zaxis=dict(range=z_range, title="Z"),

                    camera=dict(
                        eye=dict(x=1.8, y=-2.2, z=1.6),     # Isometric 3/4 view for pick-place operations
                        center=dict(x=0.4, y=0.0, z=0.6),   # Focus on workspace area
                        up=dict(x=0, y=0, z=1),
                    ),

                ),
            )

            st.plotly_chart(fig, use_container_width=True)
