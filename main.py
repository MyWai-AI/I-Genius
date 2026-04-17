"""MyWai Tool - Main application entry point."""

import streamlit as st
from src.streamlit_template.ui.main_ui import run_vilma_ui
from src.streamlit_template.auth.mywai_auth import (
    get_mywai_payload,
    initialize_mywai_apis,
)

def main():

    st.set_page_config(
        page_title="VILMA – Visual Imitation Learning UI",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # MYWAI AUTH — RUN ONLY ONCE PER SESSION

    if "mywai_initialized" not in st.session_state:

        payload = get_mywai_payload()
        if payload is None:
            st.stop()

        initialize_mywai_apis(payload)

        st.session_state["auth_token"] = payload.get("user", {}).get("auth-token")
        st.session_state["mywai_initialized"] = True

    # UI

    run_vilma_ui()


if __name__ == "__main__":
    main()
