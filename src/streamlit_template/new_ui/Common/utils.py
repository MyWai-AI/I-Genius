"""
Utility functions for VILMA UI.

Contains shared helper functions, caching utilities, and common operations.
"""

import csv
import base64
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import streamlit as st

from .config import UIConfig, DEFAULT_UI_CONFIG


# ============================================================================
# CACHING UTILITIES
# ============================================================================

@st.cache_data(show_spinner=False)
def encode_images_to_base64(paths: Tuple[str, ...]) -> List[str]:
    """
    Convert image file paths into Base64 encoded strings for UI thumbnails.
    
    Args:
        paths: Tuple of file paths to images
        
    Returns:
        List of base64 encoded image strings
    """
    encoded = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode()
                encoded.append(f"data:image/jpeg;base64,{data}")
        except (IOError, OSError):
            # Skip failed images
            pass
    return encoded


@st.cache_data(show_spinner=False)
def load_csv_columns(path: Path, columns: Tuple[str, ...]) -> Dict[str, List[float]]:
    """
    Load specific columns from a CSV into lists.
    
    Args:
        path: Path to CSV file
        columns: Column names to extract
        
    Returns:
        Dictionary mapping column names to lists of float values
    """
    data = {col: [] for col in columns}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in columns:
                data[col].append(float(row[col]))
    return data


# ============================================================================
# FILE/DIRECTORY UTILITIES  
# ============================================================================

def clean_pipeline_directories(config: UIConfig = DEFAULT_UI_CONFIG) -> None:
    """
    Delete and recreate pipeline-generated data folders.
    Called automatically on new video upload.
    
    Args:
        config: UI configuration with directory paths
    """
    folders_to_clean = [
        config.frames_path,
        config.hands_path,
        config.objects_path,
        config.trajectories_path,
        config.dmp_path,
    ]
    
    for folder in folders_to_clean:
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)


def get_sorted_files(
    directory: Path,
    pattern: str = "*",
    extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Get sorted list of files from a directory.
    
    Args:
        directory: Path to search
        pattern: Glob pattern to match
        extensions: Optional list of allowed extensions (e.g., [".jpg", ".png"])
        
    Returns:
        Sorted list of matching file paths
    """
    if not directory.exists():
        return []
    
    files = list(directory.glob(pattern))
    
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    
    return sorted(files)


# ============================================================================
# UI HELPERS
# ============================================================================

def run_with_spinner(label: str, func: callable, **kwargs) -> Any:
    """
    Run a function with a spinner indicator.
    
    Args:
        label: Text to display in spinner
        func: Function to execute
        **kwargs: Arguments to pass to function
        
    Returns:
        Result of the function call
    """
    with st.spinner(f"{label}..."):
        return func(**kwargs)


def create_download_link(
    data: bytes,
    filename: str,
    mime_type: str = "application/octet-stream"
) -> str:
    """
    Create an HTML download link for binary data.
    
    Args:
        data: Binary data to download
        filename: Suggested filename for download
        mime_type: MIME type of the data
        
    Returns:
        HTML anchor tag string for download
    """
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'


def display_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    help_text: Optional[str] = None
) -> None:
    """
    Display a metric in a styled card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta/change indicator
        help_text: Optional tooltip text
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text
    )
