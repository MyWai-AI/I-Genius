import streamlit as st
import hashlib
import os
from src.streamlit_template.core.Generic.extract_frames import extract_frames_from_video

def extract_and_cache_frames(video_path: str):
    """
    Extract frames from video and cache results in session state.
    Returns the extraction result dict.
    """
    video_key = f"frames_{video_path}"
    
    if video_key not in st.session_state:
        # Create a unique folder for this video's frames to avoid collisions
        vid_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        output_dir = f"data/Generic/frames/{vid_hash}"
        
        with st.spinner("Extracting frames..."):
            try:
                # Extract every 5th frame for preview performance, resize to 640w for speed
                result = extract_frames_from_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    every_n=5, 
                    resize_width=640 
                )
                st.session_state[video_key] = result
                
                # Parse CSV for accurate timestamps
                csv_path = result.get("index_csv")
                if csv_path and os.path.exists(csv_path):
                    import csv
                    metadata = {}
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # filename -> time_sec
                            fname = row.get("filename")
                            t_sec = float(row.get("time_sec", 0.0))
                            metadata[fname] = t_sec
                    
                    st.session_state[video_key]["metadata"] = metadata
                    
            except Exception as e:
                st.error(f"Frame extraction failed: {e}")
                return None
                
    return st.session_state.get(video_key)
