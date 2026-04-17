import streamlit as st
import os
from typing import List, Dict, Callable, Optional, Union

def render_video_player(
    videos: Union[str, List[Dict]], 
    caption: str = None, 
    download_callback: Optional[Callable[[Dict], None]] = None,
    key_prefix: str = "vid_player"
) -> Dict:
    """
    Render video player with support for multiple layers/tabs and synchronization.
    
    Args:
        videos: Single video path (str) OR list of dicts.
                If list, each dict should have: 
                {"label": str, "file_path": str, "name": str, ...}
                The 'file_path' in the dict is the relative path from MYWAI (e.g. "videos/foo.mp4").
                You should pre-calculate the local path or we do it here?
                Better: Expect each dict to have "local_path" (absolute/relative local path) 
                and "exists" boolean, or we check existence here.
                
                Let's standardize input for list:
                [
                    {
                        "label": "RGB",
                        "local_path": "data/Generic/downloads/...",
                        "name": "rgb_video.mp4",
                        ... arbitrary extra data for callback ...
                    },
                    ...
                ]
        caption: Caption text (used if single video or as fallback).
        download_callback: Function to call if video missing. Receives the video dict.
        key_prefix: Unique key prefix for widgets.
        
    Returns:
        The selected video dictionary (or None).
    """
    
    # Normalize input to list
    if isinstance(videos, str):
        # Single video mode
        video_list = [{
            "label": "Video",
            "local_path": videos,
            "name": caption or "Video"
        }]
    elif isinstance(videos, list):
        video_list = videos
    else:
        st.error("Invalid video input")
        return None
        
    if not video_list:
        st.info("No videos available.")
        return None

    selected_video = video_list[0]

    # Render Tabs if > 1
    if len(video_list) > 1:
        # User requested using radio as tabs for functionality.
        # We previously had st.tabs() which was visual only. Removing it.
        
        # We need to know which tab is active to return the selected video
        # Streamlit tabs don't return state directly, we render content inside.
        # But we want the return value to reflect selection for the *parent* to use (e.g. for frames).
        # This is tricky in Streamlit. 
        # Best approach: Use st.radio for selection logic if we need return value, 
        # OR render the dependent content (frames) *inside* the tab here?
        # User said "include the diffrent layer ... as the tab at the top of the video player".
        # And usually "Right side ... refresh".
        # If I render frames in the Right Column based on this selection, I need the selection state.
        
        # If I use st.tabs, I can't easily get "which tab is selected" into a variable for the rest of the script 
        # without hacking or using a distinctive widget inside each tab that updates state.
        
        # Alternative: Use a pill/radio selector *visually* looking like tabs above the player?
        # Or stick to st.tabs and require the parent to pass the "frames renderer" callback?
        # That's a good pattern: `render_video_player(..., extra_content_callback=func)`
        # But the parent (mywai_page) puts frames in a completely different column (Right Col).
        
        # New visual design: 
        # [Tab RGB] [Tab Depth] -> Player
        
        # If the user wants the "Video Component" to handle this, and Frames are elsewhere, 
        # we face the Streamlit data flow limit.
        
        # WORKAROUND: Use st.radio with "horizontal=True" look like tabs, or 'st.pills' (if available).
        # Or custom HTML/CSS tabs that set a session state.
        
        # Let's use st.radio with horizontal=True and 'tabs' formatting for now as it's stateful.
        # It's robust.
        
        selected_label = st.radio(
            "Select Layer",
            options=[v["label"] for v in video_list],
            horizontal=True,
            label_visibility="collapsed",
            key=f"{key_prefix}_layer_select"
        )
        selected_video = next((v for v in video_list if v["label"] == selected_label), video_list[0])
    
    # Now render the player for the selected video
    path = selected_video.get("local_path")
    if path:
        path = os.path.abspath(str(path))
        
    exists = os.path.exists(path) if path else False
    
    if exists:
        # Seek logic
        seek_time = st.session_state.get("video_seek_time", 0)
        st.video(path, start_time=int(seek_time))
        
        # Display name/caption
        if len(video_list) == 1 and caption:
             st.caption(caption)
        else:
             st.caption(f"Playing: {selected_video.get('name', 'Unknown')}")
             
    else:
        # File missing
        st.info(f"File not found: {selected_video.get('name', 'Video')}")
        if download_callback:
            if st.button(f"⬇️ Download {selected_video.get('label')}", 
                         key=f"{key_prefix}_dl_{selected_video.get('label')}",
                         width="stretch"):
                download_callback(selected_video)

    return selected_video
