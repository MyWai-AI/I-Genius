"""UI components for the MyWai tool."""

import streamlit as st

from src.streamlit_template.core.Common.config import Config


def setup_page_config():
    """Setup basic Streamlit page configuration."""
    st.set_page_config(
        page_title="MyWai Tool", 
        layout="wide", 
        page_icon="🔧"
    )


def render_user_info(user_info: dict):
    """Render basic user information."""
    st.subheader("User Information")
    st.write(f"**Email:** {user_info.get('email', 'N/A')}")
    if user_info.get('name') or user_info.get('surname'):
        name_parts = [user_info.get('name', ''), user_info.get('surname', '')]
        full_name = ' '.join(filter(None, name_parts))
        if full_name:
            st.write(f"**Name:** {full_name}")


def render_logout_button():
    """Render logout button in debug mode."""
    if Config.DEBUG_MODE:
        return st.button("Logout", use_container_width=True)
    return False
