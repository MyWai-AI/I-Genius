import streamlit as st
import hashlib
import os
import math
import cv2
from src.streamlit_template.core.Generic.extract_frames import extract_frames_from_video

def get_video_frame_info(video_path: str):
    """Return basic video metadata for timeline controls."""
    info_key = f"video_info_{video_path}"
    cached = st.session_state.get(info_key)
    if cached:
        return cached

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = (total_frames / fps) if fps > 0 else 0.0
        info = {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
        }
        st.session_state[info_key] = info
        return info
    finally:
        cap.release()


def extract_and_cache_frames(
    video_path: str,
    start_frame: int = None,
    end_frame: int = None,
    target_frames: int = 120,
):
    """
    Extract frames from video and cache results in session state.
    Returns the extraction result dict.
    """
    info = get_video_frame_info(video_path)
    if not info:
        st.error("Could not read video metadata.")
        return None

    total_frames = max(int(info.get("total_frames", 0)), 1)
    fps = float(info.get("fps", 0.0))

    start_frame = 0 if start_frame is None else max(0, min(int(start_frame), total_frames - 1))
    end_frame = (total_frames - 1) if end_frame is None else max(0, min(int(end_frame), total_frames - 1))
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame

    frame_span = (end_frame - start_frame + 1)
    target_frames = max(1, int(target_frames or 1))
    every_n = max(1, int(math.ceil(frame_span / target_frames)))

    video_key = f"frames_{video_path}_{start_frame}_{end_frame}_{every_n}"
    
    if video_key not in st.session_state:
        # Create a unique folder for this video's frame window to avoid collisions
        vid_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        output_dir = f"data/Generic/frames/{vid_hash}_{start_frame}_{end_frame}_{every_n}"
        
        with st.spinner("Extracting frames..."):
            try:
                # Extract only the selected timeline window and adapt downsampling for responsiveness.
                duration_sec = ((end_frame - start_frame + 1) / fps) if fps > 0 else None
                result = extract_frames_from_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    every_n=every_n,
                    start_sec=(start_frame / fps) if fps > 0 else 0.0,
                    duration_sec=duration_sec,
                    resize_width=640 
                )
                st.session_state[video_key] = result
                st.session_state[video_key]["frame_range"] = {
                    "start": start_frame,
                    "end": end_frame,
                    "every_n": every_n,
                    "requested": target_frames,
                }
                
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
