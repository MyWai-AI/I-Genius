import streamlit as st
import base64
from pathlib import Path
from typing import Optional

COLORS = {
    "primary": "#1A4B84",
    "primary_light": "#2E6B9E",
    "accent": "#008B8B",
    "theme": "#1A1A1A",
    "opacity": "#1A5555",
    "text": "#FFFFFF",
    "text_dark": "#E0E0E0",
    "border": "#808080",
    "border_dark": "#606060",
    "red": "#B22222",
    "red_light": "#E25555",
}

# Assuming sources are relative to project root or accessible
# In landing_page it was: Path(__file__).parent.parent / "sources"
# Here it is: src/streamlit_template/new_ui/styles.py
# Sources is in: src/streamlit_template/new_ui/sources
SOURCES_DIR = Path(__file__).parent / "sources"

def get_logo_base64(logo_filename: str) -> Optional[str]:
    """Load and encode logo image to base64."""
    # check if SOURCES_DIR is correct relative to this file
    # this file: new_ui/styles.py
    # parent: new_ui
    # parent.parent: streamlit_template
    # sources: streamlit_template/sources
    # So yes, .parent.parent / "sources" is correct if this file is in new_ui/
    
    # Wait, new_ui/styles.py ?
    # I am saving it to src/streamlit_template/new_ui/styles.py
    
    logo_path = SOURCES_DIR / logo_filename
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def inject_custom_css() -> None:
    """Inject custom CSS for the application styling."""
    st.markdown(f"""
    <style>
    .card-wrapper {{
        width: 280px;
        margin: 0 auto;
        background: linear-gradient(145deg, {COLORS["theme"]}, #2a2a3a);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 2px solid #555555; /* More visible border */
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        overflow: hidden;
        cursor: pointer;
    }}
    
    .card-wrapper:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 48px rgba(26, 75, 132, 0.4);
        border-color: {COLORS["primary_light"]};
    }}
    
    .option-card {{
        padding: 0; 
        height: 360px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    
    .logo-container {{
        width: 100%; 
        height: 140px;
        margin: 0;
        background: ghostwhite;
        border-bottom: 2px solid #555555;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .option-logo {{
        width: 80px;
        height: 80px;
        object-fit: contain;
    }}
    
    .option-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS["text"]};
        margin-top: 1.5rem;
        padding: 0 1rem;
        margin-bottom: 0.5rem;
    }}
    
    .option-description {{
        font-size: 0.9rem;
        color: {COLORS["text_dark"]};
        line-height: 1.6;
        flex-grow: 1;
        padding: 0 1.5rem 2rem 1.5rem; 
    }}
    
    .page-header {{
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 3rem;
    }}
    
    .page-title {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["accent"]} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }}
    
    .page-subtitle {{
        font-size: 1.1rem;
        color: {COLORS["border"]};
    }}
    
    /* Style buttons below cards */
    .stButton {{
        margin-top: 0.5rem !important;
        text-align: center !important;
        display: flex !important;
        justify-content: center !important;
    }}
    
    .stButton > button {{
        width: 280px !important; # Default style for landing page buttons
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(145deg, {COLORS["theme"]}, #2a2a3a) !important;
        color: {COLORS["text"]} !important;
        border: 2px solid #555555 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }}
    </style>
    """, unsafe_allow_html=True)
