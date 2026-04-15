import streamlit as st
import re
from pathlib import Path
try:
    from st_clickable_images import clickable_images
except ImportError:
    from streamlit_clickable_images import clickable_images
from src.streamlit_template.ui.helpers import encode_images
import logging

logger = logging.getLogger(__name__)

# Regex to extract frame index from filenames like "frame_000024" or "639076_frame_000024_OpenArm001"
_FRAME_IDX_RE = re.compile(r'frame_(\d+)')

def _extract_frame_index(filepath):
    """Extract frame index from a filename using regex. Returns None if not found."""
    stem = Path(filepath).stem
    m = _FRAME_IDX_RE.search(stem)
    if m:
        return int(m.group(1))
    # Fallback: try last numeric segment after underscore
    parts = stem.split('_')
    for part in reversed(parts):
        try:
            return int(part)
        except ValueError:
            continue
    return None


def render_frame_grid_viewer(paths, fps, metadata=None, video_seek_key="video_seek_time", key_prefix="default"):
    """
    Render a clickable grid of frames.
    
    Args:
        paths (list): List of file paths to images.
        fps (float): Frames per second.
        metadata (dict, optional): Mapping of filename -> timestamp (sec).
        video_seek_key (str): Session state key.
        key_prefix (str): Unique prefix to avoid key collisions across pages.
    """
    if not paths:
        st.warning("No frames to display.")
        return

    # Base64 encode images for the component
    current_batch = paths
    encoded = encode_images(tuple(current_batch))
    
    if not encoded:
        st.warning("Could not encode frames.")
        return
    
    # Titles 
    titles = []
    for i, p in enumerate(current_batch):
        fname = Path(p).name
        
        # Use metadata if available
        if metadata and fname in metadata:
             ts = metadata[fname]
             titles.append(f"Frame {i} ({ts:.2f}s)")
        else:
            f_idx = _extract_frame_index(p)
            if f_idx is not None:
                ts = f_idx / fps if fps > 0 else 0
            else:
                ts = i * (1/fps * 5) if fps else 0
            titles.append(f"Frame {i} (~{ts:.2f}s)")

    # Use a unique key per page context to avoid stale component state
    component_key = f"frame_grid_{key_prefix}_{len(paths)}"
    
    clicked = clickable_images(
        encoded,
        titles=titles,
        div_style={
            "display": "grid",
            "grid-template-columns": "repeat(auto-fill, minmax(120px, 1fr))",
            "gap": "10px",
            "padding": "10px",
            "height": "600px",
            "overflow-y": "auto",
        },
        img_style={
            "cursor": "pointer", 
            "border-radius": "5px", 
            "width": "100%",
            "transition": "transform 0.2s"
        },
        key=component_key
    )

    # Guard against the component replaying its last-clicked value on every rerun.
    # Only act when the clicked index is genuinely new (differs from the last one we handled).
    _last_key = f"_frame_grid_last_clicked_{component_key}"
    if clicked is not None and clicked > -1 and clicked != st.session_state.get(_last_key, -1):
        st.session_state[_last_key] = clicked

        selected_path = current_batch[clicked]
        fname = Path(selected_path).name
        
        # Calculate timestamp for seek
        timestamp = 0
        if metadata and fname in metadata:
             timestamp = metadata[fname]
        else:
            f_idx = _extract_frame_index(selected_path)
            if f_idx is not None:
                timestamp = f_idx / fps if fps > 0 else 0
            
        # Update session state
        st.session_state[video_seek_key] = timestamp
        logger.info(f"[frame_viewer] Seek to {timestamp:.2f}s (frame {clicked}, f_idx={f_idx})")

        st.rerun()  # Force re-render so the video player reads the updated seek time
