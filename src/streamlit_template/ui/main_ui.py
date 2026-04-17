# main_ui.py 

import streamlit as st
from pathlib import Path
from stqdm import stqdm
import time

#to see the join controllers in the sidebar
from src.streamlit_template.ui.robot_sidebar import robot_joint_sidebar

# Session helpers
from .helpers import ensure_session_state

# Upload UI
from .upload_ui import render_upload_panel

# Viewer (two-column post-run)
from .viewer_ui import render_viewer

# Stepper Bar
from .stepper_bar import StepperBar

# Pipeline functions
from .pipeline_ui import (
    #handle_bag_extract,
    handle_frames,
    handle_hands,
    handle_objects,
    handle_trajectory,
    handle_dmp,
    handle_robot,
)

# changing the upload mode if we stop
from .helpers import auto_clean_pipeline


# DATA DIRECTORIES

BASE = Path(__file__).resolve().parents[3] / "data"
UPLOADS = BASE / "Generic" / "uploads"
FRAMES = BASE / "Generic" / "frames"
HANDS = BASE / "Generic" / "hands"
OBJECTS = BASE / "Generic" / "objects"
TRAJ = BASE / "Generic" / "dmp"
DMP = BASE / "Generic" / "dmp"

for p in [UPLOADS, FRAMES, HANDS, OBJECTS, TRAJ, DMP]:
    p.mkdir(parents=True, exist_ok=True)


# PIPELINE RUNNER

def run_full_pipeline(step_functions, step_labels, stepper_placeholder):
    n_steps = len(step_labels)
    step_weight = 100.0 / n_steps

    st.session_state.pipeline_started = True
    st.session_state.pipeline_completed = False
    st.session_state.pipeline_error = None
    st.session_state.progress_value = 0.0
    st.session_state.selected_step = None

    st.session_state.step_status = ["pending"] * n_steps

    stepper = StepperBar(step_labels)

    for i, label in enumerate(stqdm(step_labels, desc="Running pipeline")):
        try:
            # ---- RUNNING ----
            st.session_state.step_status[i] = "running"

            with stepper_placeholder.container():
                stepper.display(
                    st.session_state.step_status,
                    key_prefix="run_",
                    clickable=False,
                )

            time.sleep(0.05)  # allow BLUE to render

            step_functions[i]()  # blocking step

            # ---- COMPLETED ----
            st.session_state.step_status[i] = "completed"
            st.session_state.progress_value += step_weight

            with stepper_placeholder.container():
                stepper.display(
                    st.session_state.step_status,
                    key_prefix="run_",
                    clickable=False,
                )

        except Exception as e:
            st.session_state.step_status[i] = "error"
            st.session_state.pipeline_error = str(e)

            with stepper_placeholder.container():
                stepper.display(
                    st.session_state.step_status,
                    key_prefix="run_",
                    clickable=False,
                )
            return

    st.session_state.pipeline_completed = True
    st.session_state.progress_value = 100.0

    # Auto-select Frames
    st.session_state.selected_step = 0



# MAIN APP

def run_vilma_ui():
    st.cache_resource.clear()   # <-- TEMPORARY
    # SESSION INIT (FIRST!)

    ensure_session_state()

    # Absolute safety
    st.session_state.setdefault("step_results", {})
    st.session_state.setdefault("pipeline_started", False)
    st.session_state.setdefault("pipeline_completed", False)
    st.session_state.setdefault("pipeline_error", None)
    st.session_state.setdefault("progress_value", 0.0)
    st.session_state.setdefault("selected_step", None)
    st.session_state.setdefault(
        "step_status",
        ["pending"] * 6
    )

    # SIDEBAR — Upload Mode
    st.sidebar.title("Upload Mode")

    st.sidebar.markdown("""
    <style>
        div[data-testid="stRadio"] {
            display: flex;
            justify-content: center;
        }
        div[data-testid="stRadio"] > div {
            margin-left: 12px;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        div[data-testid="stRadio"] label {
            font-size: 1.15rem;
            font-weight: 600;
            padding: 6px 0;
            margin-bottom: 10px !important;
        }
        div[data-testid="stRadio"] input[type="radio"] {
            transform: scale(1.15);
            accent-color: #e63946;
        }
    </style>
    """, unsafe_allow_html=True)

    upload_mode = st.sidebar.radio(
        "",
        ["Local", "MYWAI", "FIWARE"],
        index=0,
        key="upload_mode_selector"
    )

    # RESET PIPELINE BUTTON IN THE SIDEBAR
    
    st.sidebar.markdown("---")

    if st.sidebar.button("Reset Pipeline", use_container_width=True):
        auto_clean_pipeline(BASE)

        # Clear session state related to pipeline & upload
        for k in [
            "uploaded_video",
            "local_video_path",
            "show_pipeline",
            "pipeline_started",
            "pipeline_completed",
            "pipeline_error",
            "progress_value",
            "selected_step",
            "step_results",
            "step_status",
            "selected_frame",
            "clicked_index",
        ]:
            st.session_state.pop(k, None)

        st.rerun()

    # INITIALIZE LAST MODE AFTER upload_mode EXISTS
    if "last_upload_mode" not in st.session_state:
        st.session_state.last_upload_mode = upload_mode


    # ---- AUTO CLEAN ON UPLOAD MODE CHANGE ----
    if st.session_state.last_upload_mode != upload_mode:
        auto_clean_pipeline(BASE)

        st.session_state.last_upload_mode = upload_mode

        # HARD reset UI flow
        for k in [
            "uploaded_video",
            "local_video_path",
            "show_pipeline",
            "pipeline_started",
            "pipeline_completed",
            "pipeline_error",
            "progress_value",
            "selected_step",
            "step_results",
            "step_status",
        ]:
            st.session_state.pop(k, None)

        st.rerun()

    # ROBOT JOINT CONTROLS (only on Robot step)
    if st.session_state.get("pipeline_completed") and st.session_state.get("selected_step") == 5:
        mode, q = robot_joint_sidebar()
        

    # UPLOAD STAGE

    if not st.session_state.get("uploaded_video"):
        st.markdown("### Upload Video")
        render_upload_panel(UPLOADS, upload_mode)
        return

    # PIPELINE SETUP

    step_labels = [
        "Frames",
        "Hands",
        "Objects",
        "Trajectory",
        "DMP",
        "Robot",
    ]

    step_functions = {
        0: lambda: handle_frames(FRAMES),
        1: lambda: handle_hands(FRAMES, HANDS),
        2: lambda: handle_objects(FRAMES, OBJECTS),
        3: lambda: handle_trajectory(FRAMES, HANDS, OBJECTS, TRAJ),
        4: lambda: handle_dmp(TRAJ, DMP),
        5: lambda: handle_robot(DMP),
    }

 
    # STEPPER (SINGLE INSTANCE)

    stepper_placeholder = st.empty()
    stepper = StepperBar(step_labels)

    with stepper_placeholder.container():
        clicked = stepper.display(
            st.session_state.step_status,
            key_prefix="ui_",
            clickable=st.session_state.pipeline_completed,
        )

    # Enable navigation ONLY after completion
    if st.session_state.pipeline_completed and clicked is not None:
        st.session_state.selected_step = clicked
        st.rerun()
    

    # CENTERED VIEWER + BUTTONS

    _, center, _ = st.columns([1, 3, 1])

    with center:
        # VIEWER
        if not st.session_state.pipeline_completed:
            st.markdown("### Viewer")
            st.video(str(st.session_state.uploaded_video))

        # RUN PIPELINE BUTTON
        if not st.session_state.pipeline_started:
            if st.button("▶ RUN PIPELINE", use_container_width=True):
                run_full_pipeline(step_functions, step_labels, stepper_placeholder)
                st.rerun()

        

    # ERROR / SUCCESS
    if st.session_state.pipeline_error:
        st.error(st.session_state.pipeline_error)

    # POST-RUN: SIDE-BY-SIDE VIEW
    if st.session_state.pipeline_completed and st.session_state.selected_step is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        render_viewer()

    # PROGRESS BAR
    if st.session_state.pipeline_started:
        st.progress(
            int(st.session_state.progress_value),
            text=f"Pipeline progress: {int(st.session_state.progress_value)}%"
        )
    

    # ACTION BUTTONS
    if st.session_state.pipeline_completed:
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.button("Generate ROS Code", use_container_width=True)

        with col2:
            st.button("Push to Robot", use_container_width=True)
