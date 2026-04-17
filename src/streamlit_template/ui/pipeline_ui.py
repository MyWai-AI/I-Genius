import streamlit as st
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from .helpers import run_step, get_csv_columns

# Pipeline engines
from src.streamlit_template.core.Generic.realsense_bag_extract import extract_from_realsense_bag
from src.streamlit_template.core.Generic.extract_frames import extract_frames_from_video
from src.streamlit_template.core.Generic.hand_detection import detect_hands_on_frames
from src.streamlit_template.core.Generic.object_detection import detect_objects_on_frames
from src.streamlit_template.core.Generic.trajectory_extraction import extract_hand_object_trajectory
from src.streamlit_template.core.Generic.generate_dmp import generate_dmp
from src.streamlit_template.core.Generic.generate_dmp_xyz import generate_dmp_xyz
from src.streamlit_template.core.Common.robot_playback import (
    dmp_xyz_to_cartesian,
    compute_ik_trajectory,
    cached_meshes_for_pose,
)

def _init_step_results():
    if "step_results" not in st.session_state:
        st.session_state.step_results = {}


# STEP: REALSENSE BAG EXTRACTION

def handle_bag_extract(bag_path: Path, BASE: Path):
    if not bag_path.exists():
        return st.error(" BAG file not found.")

    res = run_step(
        "Extracting RealSense BAG",
        extract_from_realsense_bag,
        bag_path=str(bag_path),
        out_dir=str(BASE),          # <- ALWAYS /data
    )

    if "error" in res:
        return st.error(f" BAG extraction failed:\n{res['error']}")


# STEP: FRAME EXTRACTION

def handle_frames(FRAMES: Path):
    _init_step_results()
    if not st.session_state.uploaded_video:
        return

    res = run_step(
        "Extracting Frames",
        extract_frames_from_video,
        video_path=str(st.session_state.uploaded_video),
        output_dir=str(FRAMES),
        every_n=10,
        start_sec=0,
        resize_width=640,
    )

    st.session_state.step_results[0] = {
        "type": "frames",
        "paths": res.get("frame_paths", []),
    }
    st.session_state.clicked_index = -1


# STEP: HAND DETECTION

def handle_hands(FRAMES: Path, HANDS: Path):
    _init_step_results()
    idx = FRAMES / "frames_index.csv"
    if not idx.exists():
        return

    res = run_step(
        "Detecting Hands",
        detect_hands_on_frames,
        frames_dir=str(FRAMES),
        index_csv=str(idx),
        out_dir=str(HANDS),
        max_hands=2,
        sample_every_n=1,
    )

    imgs = sorted(Path(res["annotated_dir"]).glob("hands_*.jpg"))
    st.session_state.step_results[1] = {"type": "hands", "paths": imgs}
    st.session_state.clicked_index = -1


# STEP: OBJECT DETECTION

def handle_objects(FRAMES: Path, OBJECTS: Path):
    _init_step_results()
    idx = FRAMES / "frames_index.csv"
    if not idx.exists():
        return

    res = run_step(
        "Detecting Objects",
        detect_objects_on_frames,
        frames_dir=str(FRAMES),
        index_csv=str(idx),
        out_dir=str(OBJECTS),
        model_path="data/Common/ai_model/object/best.pt",
        conf=0.6,
        iou=0.45,
        target_classes=["shaft"],
        sample_every_n=1,
    )

    # Get all annotated images
    imgs = sorted(Path(res["annotated_dir"]).glob("objects_*.jpg"))
    
    # Store results with detection info for filtering if needed
    st.session_state.step_results[2] = {
        "type": "objects", 
        "paths": imgs,
        "total_detections": res.get("total_detections", 0),
        "detections_csv": res.get("detections_csv")
    }
    st.session_state.clicked_index = -1

# STEP: TRAJECTORY EXTRACTION

def handle_trajectory(FRAMES: Path, HANDS: Path, OBJECTS: Path, TRAJ: Path):
    _init_step_results()
    hands_csv = HANDS / "hands_landmarks.csv"
    if not hands_csv.exists():
        return st.warning("Run Hands first.")

    res = run_step(
        "Extracting Trajectory",
        extract_hand_object_trajectory,
        frames_dir=str(FRAMES),
        hands_csv=str(hands_csv),
        objects_csv=str(OBJECTS / "objects_detections.csv")
                     if (OBJECTS / "objects_detections.csv").exists()
                     else None,
        out_csv=str(TRAJ / "hand_traj.csv"),
        out_plot=str(TRAJ / "trajectory.png"),
        fingertip_id=8,
        smooth_window=5,
    )

    # Prefer the output plot, otherwise check for the reference one
    out_img = Path(res.get("traj_plot", str(TRAJ / "trajectory.png")))
    
    if not out_img.exists():
        # Fallback
        out_img = TRAJ / "object_xyz_dmp_plot.png"

    if out_img.exists():
        st.session_state.step_results[3] = {
            "type": "dmp_image",
            "image": str(out_img),
        }
    else:
        st.warning(f"Trajectory plot not found: {out_img.name}")

    st.session_state.clicked_index = -1

# STEP: DMP GENERATION

def handle_dmp(TRAJ: Path, DMP: Path):
    _init_step_results()

    # Define input/output paths
    traj_csv_path = TRAJ / "hand_traj.csv"
    dmp_xyz_path = DMP / "object_xyz_dmp.npy"

    if not traj_csv_path.exists():
        return st.warning("Hand trajectory CSV not found (Run Trajectory step first).")

    # Run the DMP generation step
    res = run_step(
        "Generating 3D DMP",
        generate_dmp_xyz,
        traj_csv=str(traj_csv_path),
        out_npy=str(dmp_xyz_path),
        smooth_window=7,
        polyorder=2,
    )

    if "error" in res:
        return st.error(f"DMP Generation failed: {res['error']}")
        
    if not dmp_xyz_path.exists():
        return st.warning("XYZ DMP file was not created.")

    # Load XYZ DMP (N,3)
    Yg = np.load(str(dmp_xyz_path))

    # Pick & Place positions
    pick_pos = Yg[0]
    place_pos = Yg[-1]

    # Build 3D Plotly figure
    fig = go.Figure()

    # Main DMP trajectory
    fig.add_trace(
        go.Scatter3d(
            x=Yg[:, 0],
            y=Yg[:, 1],
            z=Yg[:, 2],
            mode="lines",
            line=dict(width=6, color="blue"),
            name="XYZ DMP Trajectory",
        )
    )

    # Pick point
    fig.add_trace(
        go.Scatter3d(
            x=[pick_pos[0]],
            y=[pick_pos[1]],
            z=[pick_pos[2]],
            mode="markers+text",
            marker=dict(size=8, color="green"),
            text=["Pick"],
            textposition="top center",
            name="Pick Position",
        )
    )

    # Place point
    fig.add_trace(
        go.Scatter3d(
            x=[place_pos[0]],
            y=[place_pos[1]],
            z=[place_pos[2]],
            mode="markers+text",
            marker=dict(size=8, color="red"),
            text=["Place"],
            textposition="top center",
            name="Place Position",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        height=550,
        title="Object XYZ DMP with Pick & Place",
    )

    st.session_state.step_results[4] = {
        "type": "dmp3d",
        "fig": fig,
    }
    st.session_state.clicked_index = -1


# STEP: ROBOT 3D PLAYBACK

def handle_robot(DMP: Path):
    _init_step_results()

    urdf_path = "data/Common/robot_models/openarm/openarm.urdf"

    # === XYZ DMP path (external learned trajectory) ===
    dmp_xyz_path = DMP / "object_xyz_dmp.npy"

    if not dmp_xyz_path.exists():
        return st.error("XYZ DMP file not found")

    # === Map XYZ DMP → Cartesian path ===
    
    # Defaults (Updated for SVO 3D Data - preserving metric scale)
    # We set scales to 1.0 to respect the real depth measurements.
    # Offsets are still needed to position the trajectory in the robot workspace (Camera Frame -> Robot Frame).
    # You may need to tune 'off_x/y/z' to place the motion on your virtual table.
    off_x, off_y, off_z = 0.5, 0.0, 0.2
    scale_x, scale_y, scale_z = 2.0, 2.0, 20.0
    
    # Synthetic Arch (Hardcoded as requested)
    arch_h = 0.2
    
    # Rotation hardcoded to 90 degrees as requested
    
    print(f"[DEBUG] Trajectory Transform (SVO / Metric Mode):")
    print(f"  Offset: ({off_x}, {off_y}, {off_z})")
    print(f"  Scale:  ({scale_x}, {scale_y}, {scale_z})")
    print(f"  Arch: Enabled (H={arch_h})")

    cart = dmp_xyz_to_cartesian(
        dmp_npy=str(dmp_xyz_path),
        scale_xyz=(scale_x, scale_y, scale_z),
        offset_xyz=(off_x, off_y, off_z),
        flip_y=False,
        rotate_z=90.0,
        add_arch=False,
        arch_height=arch_h,
    )

    cart_path = cart["cartesian_path"]

    # === Inverse kinematics ===
    ik = compute_ik_trajectory(
        urdf_path=urdf_path,
        cartesian_path=cart_path
    )
    q_traj = ik["q_traj"]

    
    # === Camera presets ===
    camera_front = dict(
        eye=dict(x=1.0, y=0.0, z=0.6),
        up=dict(x=0, y=0, z=1),
    )

    camera_top = dict(
        eye=dict(x=0.0, y=0.0, z=1.5),
        up=dict(x=0, y=1, z=0),
    )

    camera_left = dict(
        eye=dict(x=0.0, y=-1.5, z=0.6),
        up=dict(x=0, y=0, z=1),
    )

    camera_right = dict(
        eye=dict(x=0.0, y=1.5, z=0.6),
        up=dict(x=0, y=0, z=1),
    )
    
    camera_iso = dict(
        eye=dict(x=0.8, y=0.8, z=0.8),
        up=dict(x=0, y=0, z=1),
    )
    
    camera_back = dict(
        eye=dict(x=-1.0, y=0.0, z=0.6),
        up=dict(x=0, y=0, z=1),
    )

    # === Build Plotly animation ===
    fig = go.Figure()
    frames = []

    for i, q in enumerate(q_traj):
        traces = cached_meshes_for_pose(
            urdf_path,
            tuple(q.tolist())
        )

        frame_data = []
        for t in traces:
            r, g, b, a = t.get("rgba", (0.8, 0.0, 0.0, 1.0))
            color_str = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
            
            frame_data.append(
                go.Mesh3d(
                    x=t["x"],
                    y=t["y"],
                    z=t["z"],
                    i=t["i"],
                    j=t["j"],
                    k=t["k"],
                    color=color_str,
                    opacity=1.0,
                    name=t.get("name", "")
                )
            )

        # End-effector path
        path_upto = cart_path[: i + 1]
        frame_data.append(
            go.Scatter3d(
                x=path_upto[:, 0],
                y=path_upto[:, 1],
                z=path_upto[:, 2],
                mode="lines",
                line=dict(width=5, color="blue"),
                name="EE Path",
            )
        )

        frames.append(go.Frame(data=frame_data, name=str(i)))

    if frames:
        fig.add_traces(frames[0].data)
        fig.update(frames=frames)

    # === Layout + controls ===
    fig.update_layout(
        height=500,

        updatemenus=[
            # ▶ Animation controls
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.15,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                    },
                ],
            },

            # 📷 Camera views
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.05,
                "buttons": [
                    {
                        "label": "Front",
                        "method": "relayout",
                        "args": [{"scene.camera": camera_front}],
                    },
                    {
                        "label": "Top",
                        "method": "relayout",
                        "args": [{"scene.camera": camera_top}],
                    },
                    {
                        "label": "Left",
                        "method": "relayout",
                        "args": [{"scene.camera": camera_left}],
                    },
                    {
                        "label": "Right",
                        "method": "relayout",
                        "args": [{"scene.camera": camera_right}],
                    },
                    {
                        "label": "Iso",
                        "method": "relayout",
                        "args": [{"scene.camera": camera_iso}],
                    },
                    {
                        "label": "Back",
                        "method": "relayout",
                        "args": [{"scene.camera": camera_back}],
                    },
                ],
            },
        ],

        # Default camera
        scene_camera=camera_front,
        
        # Lock axes to prevent camera "breathing" / moving with robot
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5), # Adjust aspect ratio for wider view
            xaxis=dict(range=[-2.5, 2.5], autorange=False),
            yaxis=dict(range=[-2.5, 2.5], autorange=False),
            zaxis=dict(range=[0.0, 2.0], autorange=False),
        ),
    )

    st.session_state.step_results[5] = {
        "type": "robot3d",
        "fig": fig,
    }
    st.session_state.clicked_index = -1
