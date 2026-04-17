"""Page components for the MyWai tool."""

import streamlit as st

from src.streamlit_template.auth.mywai_auth import get_mywai_payload, initialize_mywai_apis, enrich_user_data, logout
from src.streamlit_template.ui.components import render_user_info, render_logout_button
from src.streamlit_template.core.Common.config import Config


def create_home_page():
    """Main home page with MyWai authentication."""
    # Get MyWai payload (handles both debug and normal mode)
    mywai_payload = get_mywai_payload()
    if mywai_payload is None:
        return  # Authentication form is being shown
    
    # Initialize MyWai APIs
    initialize_mywai_apis(mywai_payload)
    
    # Enrich user data in debug mode
    enrich_user_data(mywai_payload)

    # Main page content
    st.header("MyWai Tool")

    # Logout button (only in debug mode)
    if render_logout_button():
        logout()

    # Display user information
    user_info = mywai_payload["user"]
    render_user_info(user_info)

    # Success message
    st.success("Successfully connected to MyWai platform!")
    
    # Debug info (only in debug mode)
    if Config.DEBUG_MODE:
        with st.expander("Debug Info"):
            st.json(mywai_payload)
    
    # Link to MYWAI Python API Client
    st.markdown("---")
    st.markdown(
        "### MYWAI Integration\n\n"
        "To start integrating functionality with MYWAI, check out the repository: "
        "[MYWAI Python API Client](https://dev.azure.com/zenatek-mywai/MYWAI/_git/PythonApiClient)"
    )