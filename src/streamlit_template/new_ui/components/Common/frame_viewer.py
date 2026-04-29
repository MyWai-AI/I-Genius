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


def _build_indexed_paths(paths):
    indexed_paths = []
    for i, p in enumerate(paths):
        fidx = _extract_frame_index(p)
        indexed_paths.append((i if fidx is None else fidx, p))
    return indexed_paths


def _normalize_range(min_idx, max_idx, selected):
    try:
        start, end = int(selected[0]), int(selected[1])
    except Exception:
        start, end = min_idx, max_idx
    start = max(min_idx, min(max_idx, start))
    end = max(min_idx, min(max_idx, end))
    if end < start:
        start, end = end, start
    return (start, end)


def sync_timeline_touch_state_bounds(min_idx, max_idx, key_prefix="default"):
    """Update touch-state from session values before rendering, returns whether user changed range."""
    if max_idx <= min_idx:
        return True

    range_key = f"{key_prefix}_timeline_range"
    prev_key = f"{key_prefix}_timeline_prev"
    touched_key = f"{key_prefix}_timeline_touched"

    selected = st.session_state.get(range_key, (min_idx, max_idx))
    current = _normalize_range(min_idx, max_idx, selected)

    if prev_key not in st.session_state:
        st.session_state[prev_key] = current
    if touched_key not in st.session_state:
        st.session_state[touched_key] = False

    if tuple(current) != tuple(st.session_state.get(prev_key, current)):
        st.session_state[touched_key] = True
    st.session_state[prev_key] = tuple(current)

    return bool(st.session_state.get(touched_key, False))


def sync_timeline_touch_state(paths, key_prefix="default"):
    """Path-based wrapper for sync_timeline_touch_state_bounds."""
    if not paths:
        return False
    indexed_paths = _build_indexed_paths(paths)
    min_idx = min(idx for idx, _ in indexed_paths)
    max_idx = max(idx for idx, _ in indexed_paths)
    return sync_timeline_touch_state_bounds(min_idx, max_idx, key_prefix=key_prefix)


def timeline_trim_paths(paths, key_prefix="default"):
    """Filter paths by the currently selected timeline range in session state."""
    if not paths:
        return []

    indexed_paths = _build_indexed_paths(paths)

    min_idx = min(idx for idx, _ in indexed_paths)
    max_idx = max(idx for idx, _ in indexed_paths)
    if min_idx == max_idx:
        return [indexed_paths[0][1]]

    range_key = f"{key_prefix}_timeline_range"
    selected = st.session_state.get(range_key, (min_idx, max_idx))

    try:
        selected_start, selected_end = int(selected[0]), int(selected[1])
    except Exception:
        selected_start, selected_end = min_idx, max_idx

    selected_start = max(min_idx, min(max_idx, selected_start))
    selected_end = max(min_idx, min(max_idx, selected_end))
    if selected_end < selected_start:
        selected_start, selected_end = selected_end, selected_start

    ranged = [p for idx, p in indexed_paths if selected_start <= idx <= selected_end]
    if not ranged:
        st.info("No frames in the selected timeline range.")
        return []

    return ranged


def render_timeline_range_control(paths, key_prefix="default"):
    """Render timeline slider for path-based frame indices."""
    if not paths:
        return False

    indexed_paths = _build_indexed_paths(paths)
    min_idx = min(idx for idx, _ in indexed_paths)
    max_idx = max(idx for idx, _ in indexed_paths)
    return render_timeline_range_control_bounds(min_idx, max_idx, key_prefix=key_prefix)


def render_timeline_range_control_bounds(min_idx, max_idx, key_prefix="default"):
    """Render timeline slider for explicit frame-index bounds."""
    if max_idx <= min_idx:
        return True

    range_key = f"{key_prefix}_timeline_range"

    if range_key not in st.session_state:
        st.session_state[range_key] = (min_idx, max_idx)

    selected = st.session_state.get(range_key, (min_idx, max_idx))
    selected_start, selected_end = _normalize_range(min_idx, max_idx, selected)

    st.slider(
        "Timeline frame range",
        min_value=min_idx,
        max_value=max_idx,
        value=(selected_start, selected_end),
        key=range_key,
    )
    return True


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
