"""
Landing page for VILMA application.

Provides three options for the user to choose from:
- Local: Work with local files
- MYWAI: Connect to MYWAI platform  
- FIWARE: Connect to FIWARE platform

This module acts as the main router.
"""

import streamlit as st
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

# Add project root to path for imports when running directly
_project_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.streamlit_template.new_ui.Common.styles import COLORS, inject_custom_css, get_logo_base64
from src.streamlit_template.new_ui.pages.Common.fiware_page import render_fiware_page
from src.streamlit_template.new_ui.pages.Common.local_page import render_local_page
from src.streamlit_template.new_ui.pages.Common.mywai_page import render_mywai_page, MYWAI_API_AVAILABLE
from src.streamlit_template.new_ui.pages.Generic.pipeline_page import render_pipeline_page
from src.streamlit_template.new_ui.pages.BAG.bag_pipeline_page import render_bag_pipeline_page
from src.streamlit_template.new_ui.pages.SVO.svo_pipeline_page import render_svo_pipeline_page
from src.streamlit_template.new_ui.pages.Common.skill_reuse_page import render_skill_reuse_page

# Try to import MYWAI auth services
try:
    from src.streamlit_template.new_ui.services.Generic.mywai_service import login_mywai, DEFAULT_ENDPOINT
    from mywai_python_integration_kit.apis import initialize_apis as _init_apis
except ImportError:
    login_mywai = None
    DEFAULT_ENDPOINT = ""
    _init_apis = None

try:
    from streamlit_javascript import st_javascript
except ImportError:
    st_javascript = None


OPTIONS = [
    {
        "id": "local",
        "title": "Local",
        "logo": "LOCAL.png",
        "description": "Work with local video files and process them directly on your machine. Perfect for offline development and testing.",
        "color": COLORS["accent"],
    },
    {
        "id": "mywai", 
        "title": "MYWAI",
        "logo": "MYWAI.png",
        "description": "Connect to the MYWAI platform for collaborative robot programming and AI-powered motion learning.",
        "color": COLORS["primary"],
    },
    {
        "id": "fiware",
        "title": "FIWARE",
        "logo": "FIWAREiHubs.png",
        "description": "Publish execution state to Orion-LD so trajectory delivery becomes visible at the API and IT level.",
        "color": COLORS["primary_light"],
    },
]

def render_landing_page() -> None:
    """Render the initial landing page with platform selection."""
    inject_custom_css()
    
    # Header
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">VILMA</div>
        <div class="page-subtitle">Video-based Imitation Learning for Manipulation Automation</div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(OPTIONS), gap="medium")
    
    for idx, opt in enumerate(OPTIONS):
        with cols[idx]:
            logo_b64 = get_logo_base64(opt["logo"])
            logo_html = ""
            if logo_b64:
                logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="option-logo">'
            
            # Visual card
            # Note: We use a flat string to ensure no indentation triggers Markdown code blocks
            card_html = f"""
<div class="card-wrapper">
<div class="option-card">
<div class="logo-container">
{logo_html}
</div>
<div class="option-title">{opt["title"]}</div>
<div class="option-description">
{opt["description"]}
</div>
</div>
</div>
"""
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Action Button
            if st.button(f"Start {opt['title']}", key=f"btn_{opt['id']}", width="stretch"):
                 if opt["id"] == "mywai":
                     # Check auth before switching
                     if "auth_token" in st.session_state and st.session_state.get("mywai_auth", False):
                         st.session_state["selected_platform"] = "mywai"
                         st.rerun()
                     else:
                         mywai_login_dialog()
                 else:
                     st.session_state["selected_platform"] = opt["id"]
                     st.rerun()


@st.dialog("🔐 MYWAI Login")
def mywai_login_dialog():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    endpoint = st.text_input("Endpoint", value=DEFAULT_ENDPOINT)
    
    if st.button("Login", type="primary", width="stretch"):
        if not username or not password:
             st.warning("Please enter both username and password.")
        elif login_mywai:
             with st.spinner("Authenticating..."):
                success, token, _ = login_mywai(username, password, endpoint)
                if success:
                    st.session_state["auth_token"] = token
                    st.session_state["mywai_auth"] = True
                    st.session_state["username"] = username
                    st.session_state["mywai_endpoint"] = endpoint
                    st.success("Login successful!")
                    st.session_state["selected_platform"] = "mywai"
                    st.rerun()
                else:
                    st.error(f"Login failed: {token}") 
        else:
             st.error("Login service unavailable.")

    # Footer removed as requested


def _cleanup_mywai_ghost_state():
    """Remove MyWAI page session state keys that cause ghost components on pipeline pages."""
    ghost_keys = [
        "mywai_view_level",
        "mywai_logout",
        "mywai_back_sites",
        "mywai_back_areas",
        "mywai_back_eq",
        "mywai_back_tasks",
        "mywai_run_pipeline",
    ]
    for k in ghost_keys:
        st.session_state.pop(k, None)


def main():
    st.set_page_config(
        page_title="VILMA Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state for platform selection
    if "selected_platform" not in st.session_state:
        st.session_state["selected_platform"] = None
        
    selected_platform = st.session_state["selected_platform"]
    
    # Routing
    if selected_platform == "mywai":
        # Check login
        if "auth_token" in st.session_state and st.session_state.get("mywai_auth", False):
            # Re-initialize APIs with existing token (handles returning from other pages)
            if _init_apis:
                try:
                    _init_apis(
                        endpoint=st.session_state.get("mywai_endpoint", DEFAULT_ENDPOINT),
                        auth_token=st.session_state["auth_token"]
                    )
                except Exception:
                    pass  # Best-effort; API calls will re-init via service layer
            # Render MYWAI Page
            render_mywai_page(
                token=st.session_state["auth_token"],
                endpoint=st.session_state.get("mywai_endpoint", DEFAULT_ENDPOINT)
            )
        else:
            # Show Login Dialog
            if not MYWAI_API_AVAILABLE:
                st.error("MYWAI API not available.")
                if st.button("Back"):
                    st.session_state["selected_platform"] = None
                    st.rerun()
            else:
                # 1. First, check if token exists in browser sessionStorage
                browser_token = None
                if st_javascript:
                    # Access PARENT window session storage in case Streamlit is in an iframe
                    js_code = """
                    (function() {
                        try {
                            return window.parent.sessionStorage.getItem('currentToken') || window.sessionStorage.getItem('currentToken');
                        } catch(e) {
                            return window.sessionStorage.getItem('currentToken');
                        }
                    })();
                    """
                    browser_token = st_javascript(js_code)
                    
                    # st_javascript returns 0 on first render before JS evaluates
                    if browser_token == 0:
                        st.stop()
                
                # If we found a token in the browser, auto-login
                if browser_token and isinstance(browser_token, str):
                    import json
                    try:
                        # Depending on how the token was stored, it might be raw string or JSON
                        token_str = browser_token
                        if token_str.startswith("{"):
                             parsed = json.loads(token_str)
                             if "data" in parsed and "token" in parsed["data"]:
                                 token_str = parsed["data"]["token"]
                             elif "token" in parsed:
                                 token_str = parsed["token"]
                        
                        st.session_state["auth_token"] = token_str
                        st.session_state["mywai_auth"] = True
                        st.session_state["username"] = "Browser Session"
                        st.session_state["mywai_endpoint"] = DEFAULT_ENDPOINT
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Failed to parse browser token: {e}")
                        # Fallback to manual login below

                # 2. No valid browser token found, show Manual Login Logic
                # Use a dialog or just a form in the main area
                st.markdown("### 🔐 MYWAI Login")
                col_login, _ = st.columns([1, 2])
                with col_login:
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    endpoint = st.text_input("Endpoint", value=DEFAULT_ENDPOINT)
                    
                    c1, c2 = st.columns(2)
                    if c1.button("Login", type="primary"):
                        if login_mywai:
                             with st.spinner("Authenticating..."):
                                success, token = login_mywai(username, password, endpoint)
                                if success:
                                    st.session_state["auth_token"] = token
                                    st.session_state["mywai_auth"] = True
                                    st.session_state["username"] = username
                                    st.session_state["mywai_endpoint"] = endpoint
                                    st.success("Login successful!")
                                    st.rerun()
                                else:
                                    st.error(f"Login failed: {token}") # token holds error msg
                        else:
                             st.error("Login service unavailable.")
                             
                    if c2.button("Cancel"):
                        st.session_state["selected_platform"] = None
                        st.rerun()

    elif selected_platform == "local":
        render_local_page()

    elif selected_platform == "fiware":
        render_fiware_page()
    
    elif selected_platform == "pipeline":
        _cleanup_mywai_ghost_state()
        render_pipeline_page()
    
    elif selected_platform == "bag_pipeline":
        _cleanup_mywai_ghost_state()
        render_bag_pipeline_page()

    elif selected_platform == "svo_pipeline":
        _cleanup_mywai_ghost_state()
        render_svo_pipeline_page()

    elif selected_platform == "skill_reuse":
        _cleanup_mywai_ghost_state()
        render_skill_reuse_page()

    else:
        # No platform selected - show landing page
        render_landing_page()


if __name__ == "__main__":
    main()
