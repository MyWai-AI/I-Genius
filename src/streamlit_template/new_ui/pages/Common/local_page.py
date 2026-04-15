import streamlit as st
import os
import json
import numpy as np
import cv2
from pathlib import Path
from src.streamlit_template.new_ui.components.Common.frame_viewer import render_frame_grid_viewer
from src.streamlit_template.new_ui.services.Generic.frame_service import extract_and_cache_frames
from src.streamlit_template.core.Generic.extract_frames import extract_frames_from_video
from src.streamlit_template.new_ui.components.Common.video_player import render_video_player
from src.streamlit_template.new_ui.services.Common.bag_helpers import extract_bag_frames, reconstruct_video_from_frames
# Try to import SVO helpers (might fail if pyzed not installed in current env, handle gracefully)
try:
    from src.streamlit_template.new_ui.services.Common.svo_helpers import extract_svo_frames, reconstruct_svo_video
except ImportError:
    extract_svo_frames = None
    reconstruct_svo_video = None

from src.streamlit_template.new_ui.services.Common.zip_upload_handler import (
    validate_svo_zip,
    extract_svo_zip,
    get_zip_session_hash,
)

def render_local_page():
    # 1. Header
    # Header removed
    
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back to Home"):
            st.session_state["selected_platform"] = None
            st.rerun()

    # Layout
    def render_local_content():
        # Upload Section - Top & Horizontal
        with st.container():
            
            u_col1, u_col2, u_col3, u_col4 = st.columns([2, 2, 2, 1])
            
            with u_col1:
                uploaded_video = st.file_uploader(
                    "Select Video / BAG / SVO / Data ZIP",
                    type=['mp4', 'avi', 'mov', 'mkv', 'bag', 'svo', 'svo2', 'zip'],
                    key="local_vid_uploader",
                    help="Upload a video, BAG, SVO file, or a pre-extracted SVO data ZIP (camera + frames + depth)."
                )

            with u_col2:
                uploaded_meta = st.file_uploader(
                    "Select Metadata (Optional)",
                    type=['json'],
                    key="local_meta_uploader"
                )

            with u_col3:
                uploaded_robot = st.file_uploader(
                    "Select Robot Zip (Optional)",
                    type=['zip'],
                    key="local_robot_uploader"
                )
            
            with u_col4:
                # Vertical alignment spacer
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                has_video = st.session_state.get("persistent_video_path") is not None
                is_bag = st.session_state.get("is_bag_mode", False)
                is_svo = st.session_state.get("is_svo_mode", False)

                if st.button("Run Pipeline", disabled=(not has_video and not is_svo), type="primary", width="stretch", key="local_run_pipeline_inline"):
                    st.session_state["pipeline_source"] = "local"
                    if is_bag:
                        st.session_state["selected_platform"] = "bag_pipeline"
                    elif is_svo:
                        st.session_state["selected_platform"] = "svo_pipeline"
                        st.session_state["svo_base_path"] = "data/SVO"
                    else:
                        st.session_state["selected_platform"] = "pipeline"
                    st.session_state["pipeline_running"] = True
                    st.session_state["local_video_path"] = st.session_state.get("persistent_video_path")
                    st.session_state["custom_robot_dir"] = st.session_state.get("active_custom_robot_dir")
                    # Reset results/status for fresh run
                    st.session_state["pipeline_statuses"] = ["pending"] * 5
                    st.session_state["step_results"] = {}
                    st.rerun()

        if uploaded_video:
            is_bag = uploaded_video.name.lower().endswith(".bag")
            is_svo_zip = uploaded_video.name.lower().endswith(".zip")
            is_svo = uploaded_video.name.lower().endswith(('.svo', '.svo2')) or is_svo_zip
            st.session_state["is_bag_mode"] = is_bag
            st.session_state["is_svo_mode"] = is_svo

            if is_bag:
                # --- BAG file handling ---
                # 0. Save the uploaded file first
                upload_dir = "data/BAG/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_video.name)
                
                write_file = True
                if os.path.exists(file_path):
                    if os.path.getsize(file_path) == uploaded_video.size:
                        write_file = False

                if write_file:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())
                
                final_file_path = file_path

                # Generate session ID based on filename (deterministic)
                import hashlib
                _bag_name = uploaded_video.name.rsplit('.', 1)[0]
                _bag_name_clean = "".join(c for c in _bag_name if c.isalnum() or c in ('_', '-'))
                # Use first 8 chars of md5 hash to keep it short but unique
                _bag_hash = hashlib.md5(uploaded_video.name.encode()).hexdigest()[:8]
                _bag_derived_sid = f"{_bag_name_clean}_{_bag_hash}"
                if "bag_session_id" not in st.session_state or st.session_state.get("last_uploaded_bag") != uploaded_video.name:
                    st.session_state["bag_session_id"] = _bag_derived_sid
                    st.session_state["last_uploaded_bag"] = uploaded_video.name
                
                session_id = st.session_state["bag_session_id"]
                bag_root = Path("data/BAG")
                
                # --- New Paths (User Request) ---
                # Videos: data/BAG/videos/<session_id>/
                video_base = bag_root / "videos" / session_id
                video_base.mkdir(parents=True, exist_ok=True)

                # Camera: data/BAG/camera/<session_id>.npy
                camera_base = bag_root / "camera"
                camera_base.mkdir(parents=True, exist_ok=True)
                intrinsics_path = camera_base / f"{session_id}.npy"

                # Store root and session for pipeline
                st.session_state["bag_session_dir"] = str(bag_root) 
                st.session_state["bag_file_path"] = final_file_path

                # extraction outputs
                rgb_dir = bag_root / "frames" / session_id
                
                # Output videos in new video_base
                video_out = video_base / "reconstructed_video.mp4"
                depth_out = video_base / "reconstructed_depth.mp4"
                
                # Check if we need to run extraction (if new file or missing output)
                should_extract = not video_out.exists() or not depth_out.exists() or not intrinsics_path.exists() or write_file

                if should_extract:
                    with st.spinner(f"Extracting frames to {session_id}..."):
                        # Pass bag_root + session_id
                        extract_bag_frames(final_file_path, bag_root, session_id, intrinsics_out_path=intrinsics_path)
                    
                    # Reconstruct RGB Video
                    with st.spinner("Reconstructing RGB video..."):
                         reconstruct_video_from_frames(rgb_dir, str(video_out))

                    # Reconstruct Depth Video (from depth_color heatmaps)
                    with st.spinner("Reconstructing Depth video..."):
                         depth_color_dir = bag_root / "depth_color" / session_id
                         reconstruct_video_from_frames(depth_color_dir, str(depth_out))

                # Always set persistent_video_path after extraction so video shows immediately
                if not st.session_state.get("persistent_video_path") or not Path(st.session_state["persistent_video_path"]).exists():
                    if video_out.exists():
                        st.session_state["persistent_video_path"] = str(video_out)
                    elif depth_out.exists():
                        st.session_state["persistent_video_path"] = str(depth_out)

                # Also store for pipeline
                st.session_state["local_video_path"] = st.session_state.get("persistent_video_path", str(video_out))

            elif is_svo and not is_svo_zip:
                # --- SVO file handling ---
                upload_dir = "data/SVO/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_video.name)
                
                write_file = True
                if os.path.exists(file_path):
                    if os.path.getsize(file_path) == uploaded_video.size:
                        write_file = False

                if write_file:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())
                
                final_file_path = file_path

                # Use filename as session ID for manual data mapping (e.g. "myvideo.svo" -> "myvideo")
                session_id = uploaded_video.name.rsplit('.', 1)[0]
                session_id = "".join(c for c in session_id if c.isalnum() or c in ('_', '-')) # Sanitize
                
                st.session_state["svo_session_id"] = session_id
                st.session_state["last_uploaded_svo"] = uploaded_video.name
                
                svo_root = Path("data/SVO")
                
                # --- SVO Extraction Logic ---
                # Paths
                rgb_dir = svo_root / "frames" / session_id
                depth_meters_dir = svo_root / "depth_meters" / session_id
                depth_color_dir = svo_root / "depth_color" / session_id
                camera_dir = svo_root / "camera"
                intrinsics_path = camera_dir / f"{session_id}.npy"
                intrinsics_npz = camera_dir / f"{session_id}.npz"
                global_intrinsics = camera_dir / "camera_intrinsics.npz"

                camera_dir.mkdir(parents=True, exist_ok=True)
                
                # Check for manual/existing data (png, jpg, jpeg)
                frames_exist = False
                if rgb_dir.exists():
                    for ext in ["*.png", "*.jpg", "*.jpeg"]:
                        if len(list(rgb_dir.glob(ext))) > 0:
                            frames_exist = True
                            break
                
                # Handling NPY frames (User Request)
                if not frames_exist and rgb_dir.exists():
                    npy_frames = sorted(list(rgb_dir.glob("*.npy")))
                    if len(npy_frames) > 0:
                         progress_bar = st.progress(0)
                         for idx, npy_file in enumerate(npy_frames):
                             try:
                                 arr = np.load(str(npy_file))
                                 # Normalize if needed or just save
                                 # Assume (H, W, 3) BGR or RGB. cv2 writes BGR.
                                 # If float, scale to 255? Assume uint8 for frames usually.
                                 if arr.dtype != np.uint8:
                                     try:
                                         arr = (arr * 255).astype(np.uint8)
                                     except:
                                         pass
                                 
                                 # If RGB, convert to BGR for opencv write
                                 # Heuristic: Check structure? Or assume RGB likely.
                                 # SVO export is usually BGR if checking cv2. 
                                 # Let's write as is, user can check colours.
                                 
                                 out_name = npy_file.with_suffix(".png")
                                 cv2.imwrite(str(out_name), arr)
                             except Exception as e:
                                 print(f"Error converting {npy_file}: {e}")
                             
                             if idx % 10 == 0:
                                 progress_bar.progress((idx + 1) / len(npy_frames))
                         
                         progress_bar.empty()
                         frames_exist = True

                # Handle Intrinsics: Check .npy -> .npz -> global .npz
                # If valid .npz found, convert to expected .npy dict format
                valid_intrinsics = False
                
                if intrinsics_path.exists():
                     valid_intrinsics = True
                elif intrinsics_npz.exists() or global_intrinsics.exists():
                     src = intrinsics_npz if intrinsics_npz.exists() else global_intrinsics
                     try:
                         data = np.load(str(src))
                         # Check keys
                         if "width" in data and "fx" in data:
                             # Expected Dictionary format with all keys
                             intr_dict = {
                                 "fx": float(data["fx"]), "fy": float(data["fy"]),
                                 "cx": float(data["cx"]), "cy": float(data["cy"]),
                                 "width": int(data["width"]), "height": int(data["height"])
                             }
                         elif "fx" in data and "fy" in data and "cx" in data and "cy" in data:
                             # Individual scalar keys but missing width/height
                             fx, fy = float(data["fx"]), float(data["fy"])
                             cx, cy = float(data["cx"]), float(data["cy"])
                             w, h = 0, 0
                             
                             # Infer resolution from images
                             if rgb_dir.exists():
                                 frames = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
                                 if frames:
                                     try:
                                         img = cv2.imread(str(frames[0]))
                                         if img is not None:
                                             h, w, _ = img.shape
                                             pass
                                     except Exception:
                                         pass
                             
                             if w == 0 or h == 0:
                                 w, h = 1280, 720 # Default
                                 st.warning(f"Resolution missing in intrinsics and no frames found. Defaulting to {w}x{h}.")
                                 
                             intr_dict = {
                                 "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                                 "width": w, "height": h
                             }
                         else:
                             # Fallback: Check for 'arr_0' or valid array
                             keys = list(data.keys())
                             # st.warning(f"Intrinsics keys found: {keys}")
                             
                             # Try first array
                             arr = None
                             if "arr_0" in data:
                                 arr = data["arr_0"]
                             elif "intrinsic_matrix" in data:
                                 arr = data["intrinsic_matrix"]
                             elif len(keys) > 0:
                                 arr = data[keys[0]]
                             
                             if arr is not None:
                                 # Assume [fx, fy, cx, cy] or [fx, fy, cx, cy, w, h] or similar
                                 flat = arr.flatten()
                                 if flat.size >= 4:
                                     fx, fy, cx, cy = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
                                     w, h = 0, 0
                                     if flat.size >= 6:
                                         w, h = int(flat[4]), int(flat[5])
                                     
                                     # If W/H missing, try to infer from frames
                                     if w == 0 or h == 0:
                                         if rgb_dir.exists():
                                             frames = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")))
                                             if frames:
                                                 try:
                                                     img = cv2.imread(str(frames[0]))
                                                     if img is not None:
                                                         h, w, _ = img.shape
                                                         pass
                                                 except Exception:
                                                     pass
                                     
                                     # Default if still missing
                                     if w == 0 or h == 0:
                                         w, h = 1280, 720 # Default HD720
                                         st.warning(f"Resolution missing in intrinsics and no frames found. Defaulting to {w}x{h}.")

                                     intr_dict = {
                                         "fx": fx, "fy": fy,
                                         "cx": cx, "cy": cy,
                                         "width": w, "height": h
                                     }
                                 else:
                                      raise ValueError(f"Intrinsics array too small. Size: {flat.size}")
                             else:
                                 raise ValueError(f"Could not parse intrinsics. Keys: {keys}")

                         np.save(str(intrinsics_path), intr_dict)
                         # Removed info print
                         valid_intrinsics = True
                     except Exception as e:
                         st.warning(f"Failed to convert intrinsics {src.name}: {e}")

                # Extraction decision
                if frames_exist and valid_intrinsics:
                    pass
                else:
                    svo_extraction_successful = False
                    error_message = ""

                    # 1. Try SVO Extraction if function is available
                    if extract_svo_frames:
                         with st.spinner(f"Attempting to extract frames from SVO to {session_id}..."):
                             try:
                                 res = extract_svo_frames(
                                     svo_file_path=str(final_file_path),
                                     output_root=svo_root,
                                     session_id=session_id,
                                     intrinsics_out_path=intrinsics_path
                                 )
                                 # Start reconstruction if RGB frames found
                                 if res and res.get("rgb") and len(res["rgb"]) > 0:
                                     # Convert video (optional but good for playback)
                                     video_out = svo_root / "videos" / session_id / f"{session_id}.mp4"
                                     reconstruct_video_from_frames(res["rgb"], str(video_out), fps=res.get("fps", 15))
                                     st.session_state["persistent_video_path"] = str(video_out)
                                     svo_extraction_successful = True
                                     pass
                             except Exception as e: # Catch ImportErrors (pyzed) or RuntimeErrors
                                 error_message = str(e)
                                 svo_extraction_successful = False

                    # 2. Sidecar Fallback if SVO extraction failed or tool missing
                    if not svo_extraction_successful:
                        st.warning(f"SVO extraction skipped or failed ({error_message}). Checking for sidecar video...")
                        
                        # Try Sidecar Video (User Request: "extract frames from that video")
                        # Check data/SVO/videos/session_id/*.mp4
                        video_search_dir = svo_root / "videos" / session_id
                        sidecar_videos = []
                        if video_search_dir.exists():
                             sidecar_videos = list(video_search_dir.glob("*.mp4"))
                        
                        extracted_from_sidecar = False
                        if sidecar_videos:
                             sidecar_path = sidecar_videos[0]
                             # Removed info logger
                             with st.spinner(f"Extracting frames from {sidecar_path.name}..."):
                                 try:
                                     # extract_frames_from_video saves to output_dir
                                     # We want them in rgb_dir
                                     extract_frames_from_video(str(sidecar_path), str(rgb_dir), every_n=1, resize_width=None)
                                     # Removed success logger
                                     extracted_from_sidecar = True
                                     svo_extraction_successful = True # Treated as success for downstream
                                     
                                     # Update persistent path to this video
                                     if not st.session_state.get("persistent_video_path") or not Path(st.session_state["persistent_video_path"]).exists():
                                         st.session_state["persistent_video_path"] = str(sidecar_path)
                                         
                                 except Exception as e:
                                     st.error(f"Failed to extract frames from sidecar: {e}")
                        
                        if not extracted_from_sidecar:
                             st.error("SVO extraction tool failed (ZED SDK missing?) AND no sidecar video found. Please provide offline data (frames/intrinsics) or a sidecar video in data/SVO/videos/<id>/.")
                        
                        # Reconstruct Videos
                        video_base = svo_root / "videos" / session_id
                        video_out = video_base / "reconstructed_video.mp4"
                        depth_out = video_base / "reconstructed_depth.mp4"
                        
                        if reconstruct_svo_video and not video_out.exists():
                            with st.spinner("Reconstructing SVO videos..."):
                                reconstruct_svo_video(rgb_dir, str(video_out), fps=15)
                                reconstruct_svo_video(depth_color_dir, str(depth_out), fps=15)

                # Update paths
                video_base = svo_root / "videos" / session_id
                video_out = video_base / "reconstructed_video.mp4"
                depth_out = video_base / "reconstructed_depth.mp4"

                # Reconstruct videos from frames if not already done
                # (runs whether frames were just extracted OR already existed)
                if reconstruct_svo_video and not video_out.exists():
                    with st.spinner("Reconstructing video from frames..."):
                        video_base.mkdir(parents=True, exist_ok=True)
                        reconstruct_svo_video(rgb_dir, str(video_out), fps=15)
                        reconstruct_svo_video(depth_color_dir, str(depth_out), fps=15)

                # Sidecar fallback if reconstruction still produced nothing
                if not video_out.exists() and (not st.session_state.get("persistent_video_path") or not Path(st.session_state["persistent_video_path"]).exists()):
                    video_search_dir = svo_root / "videos" / session_id
                    if video_search_dir.exists():
                        sidecars = [v for v in video_search_dir.glob("*.mp4")
                                    if v.name not in ("reconstructed_video.mp4", "reconstructed_depth.mp4")]
                        if sidecars:
                            st.session_state["persistent_video_path"] = str(sidecars[0])

                if not st.session_state.get("persistent_video_path") or not Path(st.session_state["persistent_video_path"]).exists():
                    if video_out.exists():
                        st.session_state["persistent_video_path"] = str(video_out)
                    elif depth_out.exists():
                        st.session_state["persistent_video_path"] = str(depth_out)

                st.session_state["local_video_path"] = st.session_state.get("persistent_video_path", str(video_out))
                # st.info removed

            elif is_svo_zip:
                # --- SVO Data ZIP handling ---
                _zip_name = uploaded_video.name

                if st.session_state.get("last_uploaded_svo_zip") != _zip_name:
                    with st.spinner("Validating SVO data ZIP..."):
                        validation = validate_svo_zip(uploaded_video)

                    if not validation.is_valid:
                        st.error("**Invalid SVO Data ZIP**")
                        st.markdown(validation.summary)
                    else:
                        with st.expander("📦 ZIP Contents", expanded=False):
                            st.markdown(validation.summary)

                        uploaded_video.seek(0)  # Reset stream after validation read
                        with st.spinner(f"Extracting SVO data for session '{validation.session_id}'..."):
                            success, msg, session_id = extract_svo_zip(
                                uploaded_video,
                                target_root="data/SVO",
                                session_id=validation.session_id,
                            )

                        if success:
                            pass
                            st.session_state["last_uploaded_svo_zip"] = _zip_name
                            st.session_state["svo_session_id"] = session_id
                            st.session_state["svo_base_path"] = "data/SVO"

                            svo_root = Path("data/SVO")
                            rgb_dir = svo_root / "frames" / session_id
                            camera_dir = svo_root / "camera"
                            intrinsics_npy = camera_dir / f"{session_id}.npy"
                            intrinsics_npz = camera_dir / f"{session_id}.npz"
                            video_base = svo_root / "videos" / session_id
                            video_base.mkdir(parents=True, exist_ok=True)
                            video_out = video_base / f"{session_id}.mp4"

                            # Convert intrinsics .npz → .npy dict (pipeline requires .npy)
                            if not intrinsics_npy.exists() and intrinsics_npz.exists():
                                try:
                                    _d = np.load(str(intrinsics_npz))
                                    _keys = list(_d.keys())
                                    if "fx" in _keys and "fy" in _keys and "cx" in _keys and "cy" in _keys:
                                        _w = int(_d["width"]) if "width" in _keys else 0
                                        _h = int(_d["height"]) if "height" in _keys else 0
                                        # Infer resolution from frames if missing
                                        if _w == 0 or _h == 0:
                                            _sample_frames = sorted(
                                                list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))
                                            )
                                            if _sample_frames:
                                                _img = cv2.imread(str(_sample_frames[0]))
                                                if _img is not None:
                                                    _h, _w = _img.shape[:2]
                                        if _w == 0 or _h == 0:
                                            _w, _h = 1280, 720  # fallback
                                            st.warning("Could not infer resolution from frames, defaulting to 1280×720.")
                                        _intr_dict = {
                                            "fx": float(_d["fx"]), "fy": float(_d["fy"]),
                                            "cx": float(_d["cx"]), "cy": float(_d["cy"]),
                                            "width": _w, "height": _h,
                                        }
                                        np.save(str(intrinsics_npy), _intr_dict)
                                        # Removed info print
                                except Exception as _e:
                                    st.warning(f"Could not convert intrinsics: {_e}")

                            # Convert .npy frames to PNG if needed
                            _png_frames = list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))
                            _npy_frames = sorted(list(rgb_dir.glob("*.npy")))
                            if not _png_frames and _npy_frames:
                                # Removed info logger
                                for npy_file in _npy_frames:
                                    try:
                                        arr = np.load(str(npy_file))
                                        if arr.dtype != np.uint8:
                                            arr = (arr * 255).astype(np.uint8)
                                        cv2.imwrite(str(npy_file.with_suffix(".png")), arr)
                                    except Exception as e:
                                        print(f"Error converting {npy_file}: {e}")

                            # Reconstruct video — always rebuild on fresh ZIP extraction
                            # (removes any corrupt/partial file from a previous failed attempt)
                            _frame_list = sorted(
                                list(rgb_dir.glob("frame_*.png")) + list(rgb_dir.glob("frame_*.jpg"))
                                + list(rgb_dir.glob("frame_*.jpeg"))
                            )
                            if _frame_list:
                                if video_out.exists():
                                    try:
                                        video_out.unlink()
                                    except Exception:
                                        pass
                                with st.spinner(f"Reconstructing video from {len(_frame_list)} frames..."):
                                    reconstruct_video_from_frames(
                                        rgb_dir.resolve(), str(video_out.resolve()), fps=15
                                    )
                            else:
                                st.warning("No frames found — video reconstruction skipped.")

                            if video_out.exists() and video_out.stat().st_size > 0:
                                st.session_state["persistent_video_path"] = str(video_out.resolve())
                                st.session_state["local_video_path"] = str(video_out.resolve())

                            if not validation.has_depth_meters:
                                st.warning(
                                    "⚠️ Depth data missing from ZIP. "
                                    "3D hand tracking requires depth_meters/<session_id>/*.npy files."
                                )

                            st.rerun()
                        else:
                            st.error(f"Failed to extract ZIP: {msg}")
                else:
                    # Already processed — restore session state
                    _sid = st.session_state.get("svo_session_id", "")
                    if _sid:
                        # Ensure persistent_video_path is set (may be lost on page reload)
                        _vout = Path("data/SVO/videos") / _sid / f"{_sid}.mp4"
                        _needs_rebuild = not _vout.exists() or _vout.stat().st_size == 0
                        if _needs_rebuild:
                            _rgb_dir = Path("data/SVO/frames") / _sid
                            _fl = sorted(
                                list(_rgb_dir.glob("frame_*.png")) + list(_rgb_dir.glob("frame_*.jpg"))
                                + list(_rgb_dir.glob("frame_*.jpeg"))
                            )
                            if _fl:
                                with st.spinner(f"Reconstructing video from {len(_fl)} frames..."):
                                    _vout.parent.mkdir(parents=True, exist_ok=True)
                                    if _vout.exists():
                                        _vout.unlink()
                                    reconstruct_video_from_frames(
                                        _rgb_dir.resolve(), str(_vout.resolve()), fps=15
                                    )
                        if _vout.exists() and _vout.stat().st_size > 0 and not st.session_state.get("persistent_video_path"):
                            st.session_state["persistent_video_path"] = str(_vout.resolve())
                            st.session_state["local_video_path"] = str(_vout.resolve())

            else:
                # --- Normal video file ---
                upload_dir = "data/Generic/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_video.name)

                write_file = True
                if os.path.exists(file_path):
                    if os.path.getsize(file_path) == uploaded_video.size:
                        write_file = False

                if write_file:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_video.getbuffer())

                if st.session_state.get("persistent_video_path") != file_path:
                    st.session_state["persistent_video_path"] = file_path
                    st.rerun()
        # Note: We intentionally DO NOT clear persistent_video_path here if uploaded_video is None,
        # otherwise it will be cleared as soon as the user starts processing or switches tabs.

        # Handle Robot Upload
        if uploaded_robot:
            import zipfile
            import shutil
            
            # Save zip
            robot_upload_dir = "data/Common/robot_models/custom"
            os.makedirs(robot_upload_dir, exist_ok=True)
            
            # Create unique folder for this upload (simple hash or name)
            import hashlib
            file_hash = hashlib.md5(uploaded_robot.name.encode()).hexdigest()[:8]
            custom_model_dir = os.path.join(robot_upload_dir, file_hash)
            
            if not os.path.exists(custom_model_dir):
                with zipfile.ZipFile(uploaded_robot, 'r') as zip_ref:
                    zip_ref.extractall(custom_model_dir)
                
            st.session_state["active_custom_robot_dir"] = custom_model_dir
        
        # Note: We intentionally DO NOT clear active_custom_robot_dir here if uploaded_robot is None.
        # This keeps the last uploaded robot active even if the uploader widget's state is cleared by page switches.


        st.markdown("---")

        if uploaded_video:
            main_col1, main_col2 = st.columns([1, 1])
            
            # Left Column: Player
            with main_col1:
                # BAG files: show depth toggle checkbox aligned with tabs
                is_bag = uploaded_video.name.lower().endswith(".bag")
                if is_bag:
                    session_id = st.session_state.get("bag_session_id", "")
                    bag_root = Path("data/BAG")
                    video_out = bag_root / "videos" / session_id / "reconstructed_video.mp4"
                    depth_out = bag_root / "videos" / session_id / "reconstructed_depth.mp4"
                    
                    show_depth = st.checkbox("Depth Heatmap", key="bag_depth_toggle")
                    selected = str(depth_out) if show_depth else str(video_out)
                    
                    if Path(selected).exists() and st.session_state.get("persistent_video_path") != selected:
                        st.session_state["persistent_video_path"] = selected
                        st.session_state["local_video_path"] = selected
                        st.rerun()

                current_video_path = st.session_state.get("persistent_video_path")
                render_video_player(current_video_path, caption=f"Playing: {uploaded_video.name}")
                    
            # Right Column: Frames & Details & Robot
            with main_col2:
                tab1, tab2, tab3 = st.tabs(["🎞️ Frames", "ℹ️ Details", "🤖 Robot"])
                
                video_path = st.session_state.get("persistent_video_path")
                
                with tab1:
                    # Priority: Check for EXISTING extracted frames (SVO/BAG)
                    frames_idx = None
                    
                    if is_svo:
                         svo_id = st.session_state.get("svo_session_id")
                         if svo_id:
                             rgb_path = Path("data/SVO/frames") / svo_id
                             if rgb_path.exists():
                                 frames = sorted(list(rgb_path.glob("*.png")) + list(rgb_path.glob("*.jpg")))
                                 if frames:
                                     frames_idx = {
                                         "frame_paths": [str(p) for p in frames],
                                         "fps": 15.0, # SVO typical
                                         "metadata": None
                                     }
                    
                    elif is_bag:
                         bag_id = st.session_state.get("bag_session_id")
                         if bag_id:
                             # Switch frames directory based on depth toggle
                             _show_depth = st.session_state.get("bag_depth_toggle", False)
                             _frames_subdir = "depth_color" if _show_depth else "frames"
                             rgb_path = Path("data/BAG") / _frames_subdir / bag_id
                             if rgb_path.exists():
                                 frames = sorted(list(rgb_path.glob("*.png")) + list(rgb_path.glob("*.jpg")))
                                 if frames:
                                     frames_idx = {
                                         "frame_paths": [str(p) for p in frames],
                                         "fps": 30.0, # BAG typical
                                         "metadata": None
                                     }

                    # Fallback: Extract from video if not found above
                    if not frames_idx and video_path and os.path.exists(video_path):
                        frames_idx = extract_and_cache_frames(video_path)
                        
                    if frames_idx and frames_idx.get("frame_paths"):
                        paths = frames_idx["frame_paths"]
                        fps = frames_idx.get("fps", 30.0)
                        metadata = frames_idx.get("metadata")
                        render_frame_grid_viewer(paths, fps, metadata=metadata, key_prefix="local")
                    else:
                        if is_svo or is_bag:
                             pass
                        elif video_path:
                             st.warning("Could not extract frames.")
                        else:
                             pass
                        
                with tab2:
                    # Details
                    with st.expander("📹 Video Metadata", expanded=True):
                        st.markdown(f"**Filename:** `{uploaded_video.name}`")
                        st.markdown(f"**Size:** {uploaded_video.size / 1024 / 1024:.2f} MB")
                        st.markdown(f"**Type:** {uploaded_video.type}")
                    
                    with st.expander("📄 JSON Metadata", expanded=False):
                        if uploaded_meta:
                            try:
                                meta_content = json.load(uploaded_meta)
                                st.json(meta_content)
                            except Exception as e:
                                st.error(f"Invalid JSON: {e}")
                        else:
                            st.text("No metadata file")

                with tab3:
                    # Generic Robot Preview
                    from src.streamlit_template.components.sync_viewer import sync_viewer
                    import base64
                    import xml.etree.ElementTree as ET
                    
                    # Determine paths (Default vs Custom)
                    if "active_custom_robot_dir" in st.session_state:
                        model_dir = Path(st.session_state["active_custom_robot_dir"])
                        # Find URDF recursively
                        urdfs = list(model_dir.rglob("*.urdf"))
                        if urdfs:
                            urdf_path = urdfs[0]
                            # Update model_dir to the folder containing the URDF for relative asset resolution
                            model_dir = urdf_path.parent
                        else:
                            st.error("No .urdf file found in uploaded zip.")
                            urdf_path = None
                    else:
                        # Default
                        model_dir = Path("data/Common/robot_models/openarm")
                        urdf_path = model_dir / "openarm.urdf"

                    # Load Config
                    config_path = model_dir / "config.json"
                    robot_config = {}
                    if config_path.exists():
                        try:
                            with open(config_path, "r") as f:
                                robot_config = json.load(f)
                        except:
                            pass
                    
                    # Defaults from config or fallback
                    home_pose = robot_config.get("home_pose", {})
                    link_colors = robot_config.get("link_colors", {})
                    scale = robot_config.get("scale", [1, 1, 1])
                    rotation = robot_config.get("rotation", [-90, 0, 0])
                    
                    def hydrate_urdf_with_meshes(urdf_path, base_dir=None):
                        """
                        Reads a URDF file, finds all <mesh filename="..."> tags, 
                        resolves the paths robustly, and returns the modified string with Data URIs.
                        """
                        if base_dir is None:
                            base_dir = Path(urdf_path).parent

                        try:
                            tree = ET.parse(urdf_path)
                            root = tree.getroot()
                            for mesh in root.iter('mesh'):
                                filename = mesh.get('filename')
                                if filename:
                                    real_path = None
                                    if filename.startswith('package://'):
                                        subpath = filename.replace('package://', '') 
                                        candidates = [
                                            Path("data/Common/robot_models") / subpath,
                                            Path("data/Common/robot_models/urdf") / subpath,
                                            Path(urdf_path).parent / subpath,
                                        ]
                                        parts = subpath.split('/')
                                        if len(parts) > 1:
                                            candidates.append(Path(urdf_path).parent / "/".join(parts[1:]))
                                            if 'meshes' in parts:
                                                mesh_idx = parts.index('meshes')
                                                candidates.append(Path(urdf_path).parent / "/".join(parts[mesh_idx:]))
                                                candidates.append(Path("data/Common/robot_models/urdf") / "/".join(parts[mesh_idx:]))
                                            candidates.append(Path(urdf_path).parent / parts[-1])

                                        for c in candidates:
                                            if c.exists():
                                                real_path = c
                                                break
                                    else:
                                        clean_filename = filename.lstrip("/").lstrip("\\")
                                        candidates = [
                                            (Path(urdf_path).parent / clean_filename).resolve(),
                                            (Path(urdf_path).parent / filename).resolve(),
                                            Path(urdf_path).parent / Path(clean_filename).name 
                                        ]
                                        try:
                                            found_files = list(Path(urdf_path).parent.rglob(Path(clean_filename).name))
                                            if found_files:
                                                candidates.extend(found_files)
                                        except:
                                            pass

                                        for c in candidates:
                                            if c.exists():
                                                real_path = c
                                                break
                                    
                                    if real_path and real_path.exists():
                                        with open(real_path, "rb") as f:
                                            b64_data = base64.b64encode(f.read()).decode("utf-8")
                                            ext = real_path.suffix.lower()
                                            mime = "model/stl"
                                            if ext == ".obj": mime = "model/obj"
                                            elif ext == ".dae": mime = "model/vnd.collada+xml"
                                            mesh.set('filename', f"data:{mime};base64,{b64_data}")
                                    else:
                                        st.warning(f"Warning: Could not find mesh file for {filename}")

                            return ET.tostring(root, encoding='unicode')
                        except Exception as e:
                            st.error(f"URDF Hydration Error: {e}")
                            return None

                    if urdf_path and urdf_path.exists():
                        hydrated_urdf = hydrate_urdf_with_meshes(urdf_path, model_dir)
                        if hydrated_urdf:
                            urdf_b64 = base64.b64encode(hydrated_urdf.encode('utf-8')).decode("utf-8")
                            urdf_uri = f"data:text/xml;base64,{urdf_b64}"
                            
                            # Construct Viewer Data
                            viewer_data = {
                                "models": [
                                    {
                                        "entityName": "RobotPreview",
                                        "loadType": "URDF",
                                        "path": urdf_uri,
                                        "rotation": rotation,
                                        "position": [0,0,0],
                                        "scale": scale,
                                        "homePose": home_pose,  # Passing config
                                        "linkColors": link_colors, # Passing config
                                        "animations": {}, # Disable animations
                                        "trajectory": []  # Disable trajectory
                                    }
                                ]
                            }
                            
                            
                            
                            st.caption(f"Previewing: {urdf_path.name}")
                            sync_viewer(viewer_data=viewer_data, video_path=None, key="robot_preview_tab")
                    
                    else:
                        st.error(f"URDF file not found at {urdf_path}")

        else:
            st.info("Please upload a video to analyze.")

    render_local_content()
