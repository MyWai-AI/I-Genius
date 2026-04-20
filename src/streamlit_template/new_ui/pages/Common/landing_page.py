"""
Landing page for the public local-first I-Genius app.
"""

import logging
from pathlib import Path
import sys

import streamlit as st

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.streamlit_template.new_ui.Common.styles import COLORS, inject_custom_css, get_logo_base64
from src.streamlit_template.new_ui.pages.Common.local_page import render_local_page
from src.streamlit_template.new_ui.pages.Generic.pipeline_page import render_pipeline_page
from src.streamlit_template.new_ui.pages.BAG.bag_pipeline_page import render_bag_pipeline_page
from src.streamlit_template.new_ui.pages.SVO.svo_pipeline_page import render_svo_pipeline_page


OPTIONS = [
    {
        "id": "local",
        "title": "Local Workspace",
        "logo": "LOCAL.png",
        "description": "Process local video, BAG, SVO, or prepared ZIP inputs and generate reusable robot trajectories.",
        "color": COLORS["accent"],
    },
]


def render_landing_page() -> None:
    inject_custom_css()
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">I-Genius</div>
            <div class="page-subtitle">Local visual imitation learning for robot manipulation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    empty_col1, col1, empty_col2 = st.columns([2, 3, 2], gap="medium")
    with col1:
        opt = OPTIONS[0]
        logo_b64 = get_logo_base64(opt["logo"])
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="option-logo">' if logo_b64 else ""
        st.markdown(
            f"""
            <div class="card-wrapper">
            <div class="option-card">
            <div class="logo-container">{logo_html}</div>
            <div class="option-title">{opt['title']}</div>
            <div class="option-description">{opt['description']}</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Local Workspace", key="btn_local", width="stretch"):
            st.session_state["selected_platform"] = "local"
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="I-Genius",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "selected_platform" not in st.session_state:
        st.session_state["selected_platform"] = None

    selected_platform = st.session_state["selected_platform"]

    if selected_platform == "local":
        render_local_page()
    elif selected_platform == "pipeline":
        render_pipeline_page()
    elif selected_platform == "bag_pipeline":
        render_bag_pipeline_page()
    elif selected_platform == "svo_pipeline":
        render_svo_pipeline_page()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()
