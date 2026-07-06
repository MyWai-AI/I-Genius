from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from services.receiver_launcher import start_receiver
from services.receiver_service import (
    receiver_running,
    stop_receiver,
    restart_receiver,
)
from services.ros2_service import ros2_connected
from services.zenoh_service import zenoh_running
from services.zenoh_launcher import (
    start_zenoh,
    stop_zenoh,
)
from services.robot_service import (
    robot_status,
    execute_trajectory,
)

st.set_page_config(
    page_title="Fairino Edge Operator",
    layout="wide"
)


def status_text(active, active_text, inactive_text):
    return active_text if active else inactive_text


def selected_trajectory_file(source):
    if source == "Received CSV":
        return (
            ROOT
            / "data"
            / "trajectories"
            / "received_trajectory.csv"
        )

    return (
        ROOT
        / "skill_reuse_live_screwdriver_1_events_mm.csv"
    )


if "execution_status" not in st.session_state:
    st.session_state.execution_status = "Idle"

source = st.session_state.get("trajectory_source", "Received CSV")
trajectory_file = selected_trajectory_file(source)
trajectory_exists = trajectory_file.exists()
df = None
load_error = None

if trajectory_exists:
    try:
        df = pd.read_csv(trajectory_file)
    except Exception as e:
        load_error = e

robot_info = robot_status()
zenoh_is_running = zenoh_running()
receiver_is_running = receiver_running()
ros2_is_connected = ros2_connected()
robot_is_connected = robot_info["connected"]

if load_error is not None:
    trajectory_status = "Error"
elif trajectory_exists:
    trajectory_status = "Ready to execute"
elif receiver_is_running and source == "Received CSV":
    trajectory_status = "Waiting for trajectory"
else:
    trajectory_status = "No trajectory file"

header_col, refresh_col = st.columns([4, 1])

with header_col:
    st.title("Fairino Edge Operator")

with refresh_col:
    st.write("")
    if st.button("Refresh Dashboard", use_container_width=True):
        st.rerun()

st.subheader("System Overview")

overview_cols = st.columns(4)

with overview_cols[0]:
    st.metric(
        "Zenoh",
        status_text(zenoh_is_running, "Running", "Stopped")
    )

with overview_cols[1]:
    st.metric(
        "Receiver",
        status_text(receiver_is_running, "Running", "Stopped")
    )

with overview_cols[2]:
    st.metric(
        "Robot",
        status_text(robot_is_connected, "Connected", "Disconnected")
    )

with overview_cols[3]:
    st.metric(
        "Trajectory",
        trajectory_status
    )

st.divider()

st.subheader("Connections")

zenoh_col, receiver_col, robot_col = st.columns([1, 1.5, 1])

with zenoh_col:
    st.caption("Zenoh")
    start_zenoh_col, stop_zenoh_col = st.columns(2)

    with start_zenoh_col:
        if st.button(
            "Start Zenoh",
            use_container_width=True
        ):
            start_zenoh()
            st.rerun()

    with stop_zenoh_col:
        if st.button(
            "Stop Zenoh",
            use_container_width=True
        ):
            stop_zenoh()
            st.rerun()

with receiver_col:
    st.caption("Receiver")
    start_receiver_col, restart_receiver_col, stop_receiver_col = st.columns(3)

    with start_receiver_col:
        if st.button(
            "Start Receiver",
            use_container_width=True
        ):
            if not receiver_running():
                start_receiver()

            st.rerun()

    with restart_receiver_col:
        if st.button(
            "Restart Receiver",
            use_container_width=True
        ):
            restart_receiver()
            st.rerun()

    with stop_receiver_col:
        if st.button(
            "Stop Receiver",
            use_container_width=True
        ):
            stop_receiver()
            st.rerun()

with robot_col:
    st.caption("Robot")
    if st.button(
        "Test Robot Connection",
        use_container_width=True
    ):
        robot_info = robot_status()

        if robot_info["connected"]:
            st.success(
                f"Connected to {robot_info['controller_ip']}"
            )
        else:
            st.error(
                "Robot not reachable"
            )

st.divider()

st.subheader("Trajectory")

trajectory_cols = st.columns([1.2, 1, 1, 1])

with trajectory_cols[0]:
    source = st.radio(
        "Select Source",
        [
            "Received CSV",
            "Test CSV"
        ],
        horizontal=True,
        key="trajectory_source"
    )
    trajectory_file = selected_trajectory_file(source)

with trajectory_cols[1]:
    st.metric(
        "Status",
        trajectory_status
    )

with trajectory_cols[2]:
    st.metric(
        "File",
        trajectory_file.name
    )

with trajectory_cols[3]:
    if trajectory_exists:
        modified_time = trajectory_file.stat().st_mtime
        st.metric(
            "Last Update",
            datetime.fromtimestamp(
                modified_time
            ).strftime("%H:%M:%S")
        )
    else:
        st.metric(
            "Last Update",
            "None"
        )

trajectory_action_cols = st.columns([1, 1, 2])

with trajectory_action_cols[0]:
    if df is not None:
        st.metric(
            "Points",
            len(df)
        )
    else:
        st.metric(
            "Points",
            0
        )

with trajectory_action_cols[1]:
    if source == "Received CSV":
        if st.button(
            "Clear Received Trajectory",
            use_container_width=True
        ):
            if trajectory_file.exists():
                trajectory_file.unlink()

            st.rerun()

with trajectory_action_cols[2]:
    if load_error is not None:
        st.error(
            f"Failed to load trajectory:\n{load_error}"
        )
    elif not trajectory_exists:
        st.warning(
            f"Trajectory file not found:\n{trajectory_file}"
        )
    else:
        st.success(
            "Trajectory received and ready."
        )

st.divider()

if not trajectory_exists:
    st.stop()

if load_error is not None:
    st.stop()

st.subheader("Execution")

execution_col, reload_col = st.columns([2, 1])

with execution_col:
    st.metric(
        "Execution Status",
        st.session_state.execution_status
    )

    if st.button(
        "Execute Trajectory",
        type="primary",
        use_container_width=True
    ):
        try:
            st.session_state.execution_status = "Running"

            execute_trajectory(
                str(trajectory_file)
            )

            st.session_state.execution_status = "Completed"

            st.success(
                "Trajectory execution completed."
            )

        except Exception as e:
            st.session_state.execution_status = "Failed"

            st.error(
                f"Execution failed:\n{e}"
            )

with reload_col:
    st.write("")
    st.write("")
    if st.button(
        "Reload Dashboard",
        use_container_width=True
    ):
        st.rerun()

st.divider()

st.subheader("Details")

with st.expander("Trajectory Statistics"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Points",
            len(df)
        )

    with col2:
        event_count = 0

        if "event" in df.columns:
            event_count = len(
                df["event"]
                .fillna("")
                .astype(str)
                .replace("nan", "")
                .loc[lambda x: x != ""]
            )

        st.metric(
            "Events",
            event_count
        )

    with col3:
        z_col = None

        if "z_mm" in df.columns:
            z_col = "z_mm"

        elif "z" in df.columns:
            z_col = "z"

        if z_col:
            st.metric(
                "Min Z",
                round(df[z_col].min(), 2)
            )

    with col4:
        if z_col:
            st.metric(
                "Max Z",
                round(df[z_col].max(), 2)
            )

with st.expander("Trajectory Preview"):
    st.dataframe(
        df,
        width="stretch",
        height=350
    )

with st.expander("3D Trajectory"):
    x_col = None
    y_col = None
    z_col = None

    if {"x_mm", "y_mm", "z_mm"}.issubset(df.columns):
        x_col = "x_mm"
        y_col = "y_mm"
        z_col = "z_mm"

    elif {"x", "y", "z"}.issubset(df.columns):
        x_col = "x"
        y_col = "y"
        z_col = "z"

    if x_col:
        fig = px.line_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )
    else:
        st.info(
            "No x/y/z columns available for 3D preview."
        )

with st.expander("Debug Information"):
    debug_cols = st.columns(4)

    with debug_cols[0]:
        st.metric(
            "ROS2",
            status_text(ros2_is_connected, "Connected", "Disconnected")
        )

    with debug_cols[1]:
        st.metric(
            "Receiver State",
            status_text(receiver_is_running, "Running", "Stopped")
        )

    with debug_cols[2]:
        st.metric(
            "Controller IP",
            robot_info["controller_ip"]
        )

    with debug_cols[3]:
        st.metric(
            "Robot State",
            status_text(robot_is_connected, "Connected", "Disconnected")
        )

st.info(
    "Workflow: Zenoh -> ROS2 Receiver -> Received Trajectory -> Trajectory Executor -> Fairino Robot"
)
