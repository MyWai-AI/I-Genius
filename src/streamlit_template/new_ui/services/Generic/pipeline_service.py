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
from src.streamlit_template.core.Generic.object_detection import detect_objects_on_frames
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

def handle_objects(frames_dir: Path, objects_dir: Path, model_path: str = "data/Common/ai_model/object/best.pt"):
    """Detect objects on extracted frames."""
    _init_step_results()
    
    idx = frames_dir / "frames_index.csv"
    if not idx.exists():
        st.warning("Frame index not found. Run frame extraction first.")
        return None
    
    with st.spinner("Detecting Objects..."):
        res = detect_objects_on_frames(
            frames_dir=str(frames_dir),
            index_csv=str(idx),
            out_dir=str(objects_dir),
            model_path=model_path,
            conf=0.6,
            iou=0.45,
            target_classes=["shaft"],
            sample_every_n=1,
        )
    
    imgs = sorted(Path(res["annotated_dir"]).glob("objects_*.jpg"))
    st.session_state.step_results["objects"] = {
        "type": "objects",
        "paths": imgs,
        "total_detections": res.get("total_detections", 0),
        "detections_csv": res.get("detections_csv"),
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
        cart_path = np.asarray(cart_path, dtype=float)
        finite_mask = np.all(np.isfinite(cart_path), axis=1)
        if not np.all(finite_mask):
            removed = int(len(cart_path) - int(np.sum(finite_mask)))
            st.warning(f"Removed {removed} non-finite trajectory points before IK.")
            cart_path = cart_path[finite_mask]
        if len(cart_path) == 0:
            st.error("Robot playback failed: no valid trajectory points after sanitization.")
            return None
        
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
