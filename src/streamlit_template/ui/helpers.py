# helpers.py — Shared utility functions for VILMA UI
import csv, base64
from pathlib import Path
import streamlit as st
import shutil


def ensure_session_state():
    """Initialize Streamlit session_state keys if missing."""

    defaults = {
        # upload / viewer
        "uploaded_video": None,
        "latest_result": None,
        "clicked_index": -1,
        "selected_frame": None,

        # pipeline lifecycle
        "pipeline_started": False,
        "pipeline_completed": False,
        "pipeline_error": None,

        # pipeline execution
        "current_step": None,
        "progress_value": 0.0,

        # stepper / navigation
        "step_status": [
            "pending", "pending", "pending",
            "pending", "pending", "pending", "pending"
        ],
        "selected_step": None,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def run_step(label: str, func, **kwargs):
    """Run a long operation with spinner."""
    with st.spinner(f"{label} ..."):
        return func(**kwargs)
    

# CACHED HELPERS (VIEWER PERFORMANCE)

@st.cache_data(show_spinner=False)
def encode_images(paths: tuple):
    """Convert image file paths into Base64 encoded strings for UI thumbnails."""
    out = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                out.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode())
        except Exception:
            pass
    return out


@st.cache_data(show_spinner=False)
def get_csv_columns(path: Path, cols: tuple):
    """Load specific columns from a CSV into lists."""
    data = {c: [] for c in cols}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for c in cols:
                data[c].append(float(row[c]))
    return data

def auto_clean_pipeline(data_root: Path):
    """
    Deletes pipeline-generated data folders safely.
    Called automatically on new upload.
    """
    generic_root = data_root / "Generic"
    folders_to_clean = [
        "frames",
        "hands",
        "objects",
        "dmp",
    ]

    for name in folders_to_clean:
        p = generic_root / name
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)

    # Reset session state related to pipeline
    for k in [
        "step_results",
        "clicked_index",
        "selected_frame",
        "pipeline_started",
        "pipeline_completed",
        "pipeline_error",
        "current_step",
        "progress_value",
        "selected_step",
    ]:
        if k in st.session_state:
            del st.session_state[k]
