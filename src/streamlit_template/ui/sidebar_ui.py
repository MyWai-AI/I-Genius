# sidebar_ui.py — Sidebar logic for event + measure selection
import streamlit as st
from pathlib import Path

from src.streamlit_template.core.Common.mywai_api import (
    list_fact_measures,
    download_generic_video,
    find_video_file_by_measure,
    get_labelled_event_list,
)

def sidebar_labelled_events(token):
    st.sidebar.subheader("Labelled Events")

    ok, resp = get_labelled_event_list(
        token=token,
        equipment_id=1,
        start_date="2020-01-01T00:00:00Z",
        end_date="2035-01-01T00:00:00Z",
    )

    if not ok:
        st.sidebar.error(resp)
        return None

    items = resp.get("items", [])
    st.sidebar.success(f"Found {len(items)} labelled events")

    if not items:
        return None

    labels = [f"{i['serialNumber']} ({i['id'][:8]})" for i in items]
    chosen = st.sidebar.selectbox("Select Event", labels)

    return next((i for i in items if f"{i['serialNumber']} ({i['id'][:8]})" == chosen), None)

def sidebar_fact_measures(selected_fact, token):
    if not selected_fact:
        return None

    ok, measures = list_fact_measures(selected_fact["id"], token=token)
    if not ok:
        st.sidebar.error(measures)
        return None

    labels = [f"{m['measure_name']} ({m['sensor_name']})" for m in measures]
    chosen = st.sidebar.selectbox("Select Measure", labels)

    return next((m for m in measures if f"{m['measure_name']} ({m['sensor_name']})" == chosen), None)

def load_mywai_video(selected_measure, token, UPLOADS):
    """
    Download the video associated with the chosen measure.
    """
    remote_name = find_video_file_by_measure("video", selected_measure["measure_name"], token)
    if not remote_name:
        return st.sidebar.error("No matching video found.")

    out_path = UPLOADS / Path(remote_name).name

    ok, msg = download_generic_video(
        container="video",
        file_path=remote_name,
        output_path=str(out_path),
        token=token,
    )

    if not ok:
        return st.sidebar.error(msg)

    st.session_state.uploaded_video = out_path
    st.sidebar.success("Video loaded!")
    st.rerun()