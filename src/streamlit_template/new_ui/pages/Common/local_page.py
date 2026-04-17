import streamlit as st
import os
import json
import numpy as np
import cv2
from pathlib import Path
from src.streamlit_template.new_ui.components.Common.frame_viewer import render_frame_grid_viewer
from src.streamlit_template.new_ui.components.Common import frame_viewer as _frame_viewer
from src.streamlit_template.new_ui.services.Generic.frame_service import extract_and_cache_frames, get_video_frame_info
from src.streamlit_template.core.Generic.extract_frames import extract_frames_from_video
from src.streamlit_template.new_ui.components.Common.video_player import render_video_player
from src.streamlit_template.new_ui.services.Common.bag_helpers import extract_bag_frames, reconstruct_video_from_frames
from src.streamlit_template.core.Common.object_detector import ObjectDetector
from src.streamlit_template.core.Common.estimate_monocular_depth import (
    estimate_depth_for_frames,
    generate_default_intrinsics,
)
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
                        # If depth is ready for this generic video, route to SVO pipeline
                        # which has the full 3D trajectory → DMP → robot skill chain
                        import hashlib as _hl_btn
                        _vp = st.session_state.get("persistent_video_path", "")
                        _vh = _hl_btn.md5(_vp.encode()).hexdigest()[:8] if _vp else ""
                        _depth_ready = (Path("data/Generic/depth_meters") / _vh).exists() and \
                                       any((Path("data/Generic/depth_meters") / _vh).glob("*.npy")) if _vh else False
                        if _depth_ready:
                            st.session_state["selected_platform"] = "svo_pipeline"
                            st.session_state["svo_base_path"] = "data/Generic"
                            st.session_state["svo_session_id"] = _vh
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

        # ── Generic-video AI tools (depth estimation & object detection) ──
        _is_generic_video = (
            uploaded_video
            and not st.session_state.get("is_bag_mode", False)
            and not st.session_state.get("is_svo_mode", False)
        )
        if _is_generic_video:
            video_path_for_tools = st.session_state.get("persistent_video_path")
            if video_path_for_tools and Path(video_path_for_tools).exists():
                import hashlib as _hl
                _vid_hash = _hl.md5(video_path_for_tools.encode()).hexdigest()[:8]
                _generic_base = Path("data/Generic")
                _frames_dir = _generic_base / "frames" / _vid_hash
                _depth_meters_dir = _generic_base / "depth_meters" / _vid_hash
                _depth_color_dir = _generic_base / "depth_color" / _vid_hash
                _camera_dir = _generic_base / "camera"
                _intrinsics_path = _camera_dir / f"{_vid_hash}.npy"
                _objects_dir = _generic_base / "objects" / _vid_hash
                _plots_dir = _generic_base / "plots" / _vid_hash

                with st.expander("AI Tools (Generic Video)", expanded=True):

                    # ── Status flags ──
                    _frames_done = _frames_dir.exists() and (any(_frames_dir.glob("*.jpg")) or any(_frames_dir.glob("*.png")))
                    _depth_done = _depth_meters_dir.exists() and any(_depth_meters_dir.glob("*.npy"))
                    _det_done = _objects_dir.exists() and (_objects_dir / "object_3d_raw.npy").exists()

                    # ── Step 1: Frame + Depth Extraction ──
                    st.markdown("**Step 1 — Frame & Depth Extraction**")
                    tool_c1, tool_c2 = st.columns(2)
                    with tool_c1:
                        _depth_model_size = st.selectbox(
                            "Depth model size",
                            ["Small", "Base", "Large"],
                            index=0,
                            key="depth_model_size",
                        )
                        _max_depth = st.number_input(
                            "Max depth (m)",
                            min_value=0.5,
                            max_value=50.0,
                            value=5.0,
                            step=0.5,
                            key="depth_max_m",
                        )
                    with tool_c2:
                        if _frames_done:
                            _n_frames = len(list(_frames_dir.glob("*.jpg")) + list(_frames_dir.glob("*.png")))
                            st.success(f"Frames ready ({_n_frames})")
                        if _depth_done:
                            _n_depth = len(list(_depth_meters_dir.glob("*.npy")))
                            st.success(f"Depth ready ({_n_depth} frames)")

                    if st.button("Extract Frames + Depth", key="btn_estimate_depth", type="primary", disabled=_depth_done):
                        # Extract frames only if not already done
                        if not _frames_done:
                            with st.spinner("Extracting video frames..."):
                                extract_frames_from_video(
                                    video_path=video_path_for_tools,
                                    output_dir=str(_frames_dir),
                                    every_n=1,
                                )
                        else:
                            st.info("Frames already extracted, skipping.")

                        # Run depth estimation
                        with st.spinner(f"Running depth estimation ({_depth_model_size})..."):
                            _depth_result = estimate_depth_for_frames(
                                input_dir=str(_frames_dir),
                                output_dir=str(_depth_meters_dir),
                                depth_color_dir=str(_depth_color_dir),
                                model_size=_depth_model_size,
                                max_depth_m=_max_depth,
                            )

                        with st.spinner("Generating camera intrinsics..."):
                            generate_default_intrinsics(
                                frame_dir=str(_frames_dir),
                                output_path=str(_intrinsics_path),
                            )

                        st.success(
                            f"Done: {_depth_result['count']} depth frames "
                            f"(backend: {_depth_result['backend']})"
                        )
                        st.rerun()

                    if _depth_done and st.button("Re-run Depth", key="btn_rerun_depth"):
                        import shutil as _sh
                        _sh.rmtree(str(_depth_meters_dir), ignore_errors=True)
                        st.rerun()

                    st.markdown("---")

                    # ── Step 2: Select Tracking Target ──
                    st.markdown("**Step 2 — Select Object to Track**")
                    if not _frames_done and not _depth_done:
                        st.info("Run Step 1 first to extract frames.")
                    else:
                        # Load base frame (first extracted frame)
                        _base_frame_path = None
                        _frame_files = sorted(list(_frames_dir.glob("*.jpg")) + list(_frames_dir.glob("*.png")))
                        if _frame_files:
                            _base_frame_path = _frame_files[0]

                        if _base_frame_path:
                            # Run detector on base frame to get candidate boxes
                            _sel_key = f"selected_det_idx_{_vid_hash}"
                            _dets_key = f"base_frame_dets_{_vid_hash}"

                            _official_models = [
                                "yolov8n.pt",
                                "yolov8s.pt",
                                "yolov8m.pt",
                                "yolov8l.pt",
                                "yolov8x.pt",
                            ]
                            _cached_models = []
                            _model_cache_dir = Path("data/Common/ai_model/object")
                            if _model_cache_dir.exists():
                                _cached_models = [str(p).replace("\\", "/") for p in sorted(_model_cache_dir.glob("*.pt"))]

                            _model_options = []
                            for _mp in _official_models + _cached_models:
                                if _mp not in _model_options:
                                    _model_options.append(_mp)

                            _prev_model = st.session_state.get("selected_object_model_path", "yolov8x.pt")
                            if _prev_model and _prev_model not in _model_options:
                                _model_options.insert(0, _prev_model)

                            if "yolov8x.pt" in _model_options:
                                _default_model = _prev_model if _prev_model in _model_options else "yolov8x.pt"
                            else:
                                _default_model = _model_options[0]
                            _default_model_idx = _model_options.index(_default_model)

                            _img_side_col, _control_side_col = st.columns([1, 1])

                            # Allow user to tune thresholds; controls stay on the right side.
                            with _control_side_col:
                                st.write("###### Detection Tuning")
                                _selected_model = st.selectbox(
                                    "Detection Model",
                                    options=_model_options,
                                    index=_default_model_idx,
                                    key=f"scan_model_{_vid_hash}",
                                    format_func=lambda p: f"{Path(p).name} ({'cached' if ('/' in p or '\\\\' in p) else 'official'})",
                                )
                                _ui_conf = st.slider(
                                    "Confidence Threshold",
                                    min_value=0.01,
                                    max_value=0.50,
                                    value=0.05,
                                    step=0.01,
                                    key=f"scan_conf_{_vid_hash}",
                                )
                                _ui_max_a = st.slider(
                                    "Max Area (%)",
                                    min_value=1.0,
                                    max_value=100.0,
                                    value=15.0,
                                    step=1.0,
                                    key=f"scan_max_a_{_vid_hash}",
                                )
                                _ui_min_a = st.slider(
                                    "Min Area (%)",
                                    min_value=0.0001,
                                    max_value=1.0000,
                                    value=0.0001,
                                    format="%.4f",
                                    key=f"scan_min_a_{_vid_hash}",
                                )
                                _ui_size_ratio = st.slider(
                                    "BBox Size Tolerance (×)",
                                    min_value=1.5,
                                    max_value=20.0,
                                    value=4.0,
                                    step=0.5,
                                    help="Max allowed ratio between tracked bbox area and selected bbox area. Lower = stricter (rejects hands/large objects). Higher = more permissive.",
                                    key=f"scan_size_ratio_{_vid_hash}",
                                )
                                _run_scan_frame0 = st.button(
                                    "Detect on Frame 0",
                                    key="btn_scan_frame0_" + _vid_hash,
                                    use_container_width=True,
                                )

                            st.session_state["selected_object_model_path"] = _selected_model
                            st.session_state["selected_tracking_source_hash"] = _vid_hash
                            st.session_state["selected_object_conf_threshold"] = float(_ui_conf)
                            st.session_state["selected_object_max_area_pct"] = float(_ui_max_a)
                            st.session_state["selected_object_min_area_pct"] = float(_ui_min_a)
                            st.session_state["selected_object_bbox_size_ratio"] = float(_ui_size_ratio)

                            _model_sig_key = f"base_frame_model_{_vid_hash}"
                            if st.session_state.get(_model_sig_key) != _selected_model:
                                st.session_state.pop(_dets_key, None)
                                st.session_state[_model_sig_key] = _selected_model

                            if _dets_key not in st.session_state:
                                # Use YOLO directly so OBB models (best-obb.pt, best-obbb.pt) are handled
                                from ultralytics import YOLO as _YOLO_init
                                import cv2 as _cv2_sel
                                import numpy as _np_sel
                                from src.streamlit_template.core.Common.object_detector import Detection as _Det_init
                                _base_bgr = _cv2_sel.imread(str(_base_frame_path))
                                if _base_bgr is not None:
                                    _init_model = _YOLO_init(_selected_model)
                                    _init_res = _init_model.predict(_base_bgr, verbose=False, conf=0.05, imgsz=1280)
                                    _init_dets = []
                                    if _init_res:
                                        _ir0 = _init_res[0]
                                        _names = _ir0.names or {}
                                        if _ir0.obb is not None and len(_ir0.obb) > 0:
                                            for _ob in _ir0.obb:
                                                _oc = float(_ob.conf[0])
                                                _ocls = int(_ob.cls[0]) if _ob.cls is not None else None
                                                _olabel = _names.get(_ocls) if _ocls is not None else None
                                                _xywhr = _ob.xywhr[0].cpu().numpy()
                                                _rect = ((_xywhr[0], _xywhr[1]), (_xywhr[2], _xywhr[3]), float(_np_sel.degrees(_xywhr[4])))
                                                _pts = _cv2_sel.boxPoints(_rect)
                                                _ox1, _oy1 = float(_np_sel.min(_pts[:,0])), float(_np_sel.min(_pts[:,1]))
                                                _ox2, _oy2 = float(_np_sel.max(_pts[:,0])), float(_np_sel.max(_pts[:,1]))
                                                _ow2, _oh2 = _ox2 - _ox1, _oy2 - _oy1
                                                _init_dets.append(_Det_init(
                                                    bbox_xyxy=_np_sel.array([_ox1, _oy1, _ox2, _oy2], dtype=_np_sel.float32),
                                                    bbox_xywh=_np_sel.array([_ox1+_ow2/2, _oy1+_oh2/2, _ow2, _oh2], dtype=_np_sel.float32),
                                                    confidence=_oc,
                                                    label=_olabel,
                                                ))
                                        elif _ir0.boxes is not None and len(_ir0.boxes) > 0:
                                            for _bb, _bc, _bcls in zip(_ir0.boxes.xyxy, _ir0.boxes.conf, _ir0.boxes.cls):
                                                _bn = _bb.cpu().numpy()
                                                _bw2, _bh2 = _bn[2]-_bn[0], _bn[3]-_bn[1]
                                                _blabel = _names.get(int(_bcls)) if _names else None
                                                _init_dets.append(_Det_init(
                                                    bbox_xyxy=_bn.astype(_np_sel.float32),
                                                    bbox_xywh=_np_sel.array([_bn[0]+_bw2/2, _bn[1]+_bh2/2, _bw2, _bh2], dtype=_np_sel.float32),
                                                    confidence=float(_bc),
                                                    label=_blabel,
                                                ))
                                    st.session_state[_dets_key] = _init_dets

                            # Run detection on base frame only (frame 0).
                            if _run_scan_frame0:
                                with st.spinner("Running detector on frame 0..."):
                                    from ultralytics import YOLO
                                    import cv2 as _cv2_scan
                                    import numpy as _np_scan
                                    from src.streamlit_template.core.Common.object_detector import Detection

                                    _scan_model = YOLO(_selected_model)
                                    _new_dets = []

                                    _fr0_bgr = _cv2_scan.imread(str(_frame_files[0]))
                                    if _fr0_bgr is not None:
                                        _res = _scan_model.predict(
                                            _fr0_bgr,
                                            verbose=False,
                                            conf=_ui_conf,
                                            imgsz=1280,
                                            max_det=100,
                                            agnostic_nms=True,
                                        )
                                        _fr_h, _fr_w = _fr0_bgr.shape[:2]
                                        if len(_res) > 0:
                                            _r0 = _res[0]
                                            _scan_names = _r0.names or {}
                                            # OBB model (e.g. best-obb.pt, best-obbb.pt) — results in .obb not .boxes
                                            _has_obb = _r0.obb is not None and len(_r0.obb) > 0
                                            _has_boxes = _r0.boxes is not None and len(_r0.boxes) > 0

                                            if _has_obb:
                                                # Convert rotated OBB to axis-aligned bounding box for selection UI
                                                for _obox in _r0.obb:
                                                    _c = float(_obox.conf[0])
                                                    _ocls = int(_obox.cls[0]) if _obox.cls is not None else None
                                                    _olabel = _scan_names.get(_ocls) if _ocls is not None else None
                                                    _xywhr = _obox.xywhr[0].cpu().numpy()
                                                    _ocx, _ocy, _ow, _oh, _orot = _xywhr
                                                    _rect = ((_ocx, _ocy), (_ow, _oh), float(_np_scan.degrees(_orot)))
                                                    _pts = _cv2_scan.boxPoints(_rect)
                                                    _x1 = float(_np_scan.min(_pts[:, 0]))
                                                    _y1 = float(_np_scan.min(_pts[:, 1]))
                                                    _x2 = float(_np_scan.max(_pts[:, 0]))
                                                    _y2 = float(_np_scan.max(_pts[:, 1]))
                                                    _bw, _bh = _x2 - _x1, _y2 - _y1
                                                    _area = _bw * _bh
                                                    if _area > (_fr_h * _fr_w * (_ui_max_a / 100.0)):
                                                        continue
                                                    if _area < (_fr_h * _fr_w * (_ui_min_a / 100.0)):
                                                        continue
                                                    _bcx, _bcy = _x1 + _bw / 2, _y1 + _bh / 2
                                                    _new_dets.append(Detection(
                                                        bbox_xyxy=_np_scan.array([_x1, _y1, _x2, _y2], dtype=_np_scan.float32),
                                                        bbox_xywh=_np_scan.array([_bcx, _bcy, _bw, _bh], dtype=_np_scan.float32),
                                                        confidence=_c,
                                                        label=_olabel,
                                                    ))
                                            elif _has_boxes:
                                                _boxes = _r0.boxes
                                                for _b, _conf_tensor, _cls_tensor in zip(_boxes.xyxy, _boxes.conf, _boxes.cls):
                                                    _b_np = _b.cpu().numpy()
                                                    _c = float(_conf_tensor)
                                                    _blabel = _scan_names.get(int(_cls_tensor)) if _scan_names else None
                                                    _w, _h = _b_np[2] - _b_np[0], _b_np[3] - _b_np[1]
                                                    _area = _w * _h

                                                    if _area > (_fr_h * _fr_w * (_ui_max_a / 100.0)):
                                                        continue
                                                    if _area < (_fr_h * _fr_w * (_ui_min_a / 100.0)):
                                                        continue

                                                    _cx, _cy = _b_np[0] + _w / 2, _b_np[1] + _h / 2
                                                    _new_dets.append(Detection(
                                                        bbox_xyxy=_b_np.astype(_np_scan.float32),
                                                        bbox_xywh=_np_scan.array([_cx, _cy, _w, _h], dtype=_np_scan.float32),
                                                        confidence=_c,
                                                        label=_blabel,
                                                    ))

                                    _new_dets.sort(key=lambda d: d.confidence, reverse=True)
                                    st.session_state[_dets_key] = _new_dets

                            _base_dets = st.session_state.get(_dets_key, [])

                            if _base_dets:
                                # Draw all boxes on base frame for display
                                import cv2 as _cv2_draw
                                import numpy as _np_draw
                                _base_bgr_show = _cv2_draw.imread(str(_base_frame_path))
                                _box_labels = []
                                for _di, _det in enumerate(_base_dets):
                                    x1, y1, x2, y2 = [int(v) for v in _det.bbox_xyxy]
                                    _color = (0, 0, 255)  # Red in BGR
                                    _cv2_draw.rectangle(_base_bgr_show, (x1, y1), (x2, y2), _color, 2)
                                    _det_label = _det.label if getattr(_det, "label", None) else None
                                    _img_text = f"#{_di} {_det_label} {_det.confidence:.2f}" if _det_label else f"#{_di} {_det.confidence:.2f}"
                                    _box_text = f"#{_di} {_det_label} — conf {_det.confidence:.2f}" if _det_label else f"#{_di} — conf {_det.confidence:.2f}"
                                    _cv2_draw.putText(_base_bgr_show, _img_text, (x1, max(y1-6, 12)),
                                        _cv2_draw.FONT_HERSHEY_SIMPLEX, 0.6, _color, 2)
                                    _box_labels.append(_box_text)

                                _base_rgb_show = _cv2_draw.cvtColor(_base_bgr_show, _cv2_draw.COLOR_BGR2RGB)

                                with _img_side_col:
                                    st.image(_base_rgb_show, caption="Base frame â€” detected objects", use_container_width=True)

                                with _control_side_col:
                                    _sel_idx = st.selectbox(
                                        "Select target object",
                                        options=list(range(len(_base_dets))),
                                        format_func=lambda i: _box_labels[i],
                                        key=_sel_key,
                                    )
                                    # Save selected bbox and label to session state for pipeline use
                                    _sel_det = _base_dets[_sel_idx]
                                    st.session_state[f"tracking_bbox_{_vid_hash}"] = _sel_det.bbox_xyxy.tolist()
                                    st.session_state["selected_tracking_bbox_xyxy"] = _sel_det.bbox_xyxy.tolist()
                                    st.session_state["selected_tracking_label"] = getattr(_sel_det, "label", None)
                                    st.session_state["selected_tracking_source_hash"] = _vid_hash
                                    st.caption(f"Selected bbox: {[int(v) for v in _sel_det.bbox_xyxy]}")
                            else:
                                with _control_side_col:
                                    st.warning("No objects detected in the base frame. Click Detect on Frame 0.")
                                with _img_side_col:
                                    st.image(str(_base_frame_path), caption="Base frame", use_container_width=True)

        # ── SVO / BAG AI tools (object selection only — depth already extracted) ──
        _is_svo_or_bag = uploaded_video and (
            st.session_state.get("is_svo_mode", False) or st.session_state.get("is_bag_mode", False)
        )
        if _is_svo_or_bag:
            _is_bag_tool = st.session_state.get("is_bag_mode", False)
            if _is_bag_tool:
                _tool_session_id = st.session_state.get("bag_session_id", "")
                _tool_frames_dir = Path("data/BAG/frames") / _tool_session_id if _tool_session_id else None
                _tool_depth_dir = Path("data/BAG/depth_meters") / _tool_session_id if _tool_session_id else None
                _tool_label = "BAG"
            else:
                _tool_session_id = st.session_state.get("svo_session_id", "")
                _tool_frames_dir = Path("data/SVO/frames") / _tool_session_id if _tool_session_id else None
                _tool_depth_dir = Path("data/SVO/depth_meters") / _tool_session_id if _tool_session_id else None
                _tool_label = "SVO"

            _tool_frames_done = (
                _tool_frames_dir is not None
                and _tool_frames_dir.exists()
                and (any(_tool_frames_dir.glob("*.jpg")) or any(_tool_frames_dir.glob("*.png")))
            )

            if _tool_frames_done:
                # Derive a stable hash from the session_id for keying session state
                import hashlib as _hl_svo
                _tool_hash = _hl_svo.md5(_tool_session_id.encode()).hexdigest()[:8] if _tool_session_id else "svo"

                with st.expander(f"AI Tools ({_tool_label})", expanded=True):
                    st.markdown("**Select Object to Track**")

                    _tool_frame_files = sorted(
                        list(_tool_frames_dir.glob("*.jpg")) + list(_tool_frames_dir.glob("*.png"))
                    )
                    _tool_base_frame_path = _tool_frame_files[0] if _tool_frame_files else None

                    if _tool_base_frame_path:
                        _sel_key_t = f"selected_det_idx_{_tool_hash}"
                        _dets_key_t = f"base_frame_dets_{_tool_hash}"

                        _official_models_t = [
                            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                        ]
                        _cached_models_t = []
                        _model_cache_dir_t = Path("data/Common/ai_model/object")
                        if _model_cache_dir_t.exists():
                            _cached_models_t = [str(p).replace("\\", "/") for p in sorted(_model_cache_dir_t.glob("*.pt"))]
                        _model_options_t = []
                        for _mp_t in _official_models_t + _cached_models_t:
                            if _mp_t not in _model_options_t:
                                _model_options_t.append(_mp_t)
                        _prev_model_t = st.session_state.get("selected_object_model_path", "yolov8x.pt")
                        if _prev_model_t and _prev_model_t not in _model_options_t:
                            _model_options_t.insert(0, _prev_model_t)
                        _default_model_t = _prev_model_t if _prev_model_t in _model_options_t else _model_options_t[0]
                        _default_model_idx_t = _model_options_t.index(_default_model_t)

                        _img_col_t, _ctrl_col_t = st.columns([1, 1])

                        with _ctrl_col_t:
                            st.write("###### Detection Tuning")
                            _selected_model_t = st.selectbox(
                                "Detection Model",
                                options=_model_options_t,
                                index=_default_model_idx_t,
                                key=f"scan_model_{_tool_hash}",
                                format_func=lambda p: f"{Path(p).name} ({'cached' if ('/' in p or '\\\\' in p) else 'official'})",
                            )
                            _ui_conf_t = st.slider(
                                "Confidence Threshold",
                                min_value=0.01, max_value=0.50, value=0.05, step=0.01,
                                key=f"scan_conf_{_tool_hash}",
                            )
                            _ui_max_a_t = st.slider(
                                "Max Area (%)",
                                min_value=1.0, max_value=100.0, value=15.0, step=1.0,
                                key=f"scan_max_a_{_tool_hash}",
                            )
                            _ui_min_a_t = st.slider(
                                "Min Area (%)",
                                min_value=0.0001, max_value=1.0000, value=0.0001, format="%.4f",
                                key=f"scan_min_a_{_tool_hash}",
                            )
                            _ui_size_ratio_t = st.slider(
                                "BBox Size Tolerance (×)",
                                min_value=1.5, max_value=20.0, value=4.0, step=0.5,
                                help="Max allowed ratio between tracked bbox area and selected bbox area.",
                                key=f"scan_size_ratio_{_tool_hash}",
                            )
                            _run_scan_t = st.button(
                                "Detect on Frame 0",
                                key=f"btn_scan_frame0_{_tool_hash}",
                                use_container_width=True,
                            )

                        st.session_state["selected_object_model_path"] = _selected_model_t
                        st.session_state["selected_tracking_source_hash"] = _tool_hash
                        st.session_state["selected_object_conf_threshold"] = float(_ui_conf_t)
                        st.session_state["selected_object_max_area_pct"] = float(_ui_max_a_t)
                        st.session_state["selected_object_min_area_pct"] = float(_ui_min_a_t)
                        st.session_state["selected_object_bbox_size_ratio"] = float(_ui_size_ratio_t)

                        _model_sig_key_t = f"base_frame_model_{_tool_hash}"
                        if st.session_state.get(_model_sig_key_t) != _selected_model_t:
                            st.session_state.pop(_dets_key_t, None)
                            st.session_state[_model_sig_key_t] = _selected_model_t

                        if _dets_key_t not in st.session_state:
                            from ultralytics import YOLO as _YOLO_t
                            import cv2 as _cv2_t
                            import numpy as _np_t
                            from src.streamlit_template.core.Common.object_detector import Detection as _Det_t
                            _base_bgr_t = _cv2_t.imread(str(_tool_base_frame_path))
                            if _base_bgr_t is not None:
                                _init_model_t = _YOLO_t(_selected_model_t)
                                _init_res_t = _init_model_t.predict(_base_bgr_t, verbose=False, conf=0.05, imgsz=1280)
                                _init_dets_t = []
                                if _init_res_t:
                                    _ir0_t = _init_res_t[0]
                                    _names_t = _ir0_t.names or {}
                                    if _ir0_t.obb is not None and len(_ir0_t.obb) > 0:
                                        for _ob_t in _ir0_t.obb:
                                            _oc_t = float(_ob_t.conf[0])
                                            _ocls_t = int(_ob_t.cls[0]) if _ob_t.cls is not None else None
                                            _olabel_t = _names_t.get(_ocls_t) if _ocls_t is not None else None
                                            _xywhr_t = _ob_t.xywhr[0].cpu().numpy()
                                            _rect_t = ((_xywhr_t[0], _xywhr_t[1]), (_xywhr_t[2], _xywhr_t[3]), float(_np_t.degrees(_xywhr_t[4])))
                                            _pts_t = _cv2_t.boxPoints(_rect_t)
                                            _ox1_t, _oy1_t = float(_np_t.min(_pts_t[:,0])), float(_np_t.min(_pts_t[:,1]))
                                            _ox2_t, _oy2_t = float(_np_t.max(_pts_t[:,0])), float(_np_t.max(_pts_t[:,1]))
                                            _ow2_t, _oh2_t = _ox2_t - _ox1_t, _oy2_t - _oy1_t
                                            _init_dets_t.append(_Det_t(
                                                bbox_xyxy=_np_t.array([_ox1_t, _oy1_t, _ox2_t, _oy2_t], dtype=_np_t.float32),
                                                bbox_xywh=_np_t.array([_ox1_t+_ow2_t/2, _oy1_t+_oh2_t/2, _ow2_t, _oh2_t], dtype=_np_t.float32),
                                                confidence=_oc_t,
                                                label=_olabel_t,
                                            ))
                                    elif _ir0_t.boxes is not None and len(_ir0_t.boxes) > 0:
                                        for _bb_t, _bc_t, _bcls_t in zip(_ir0_t.boxes.xyxy, _ir0_t.boxes.conf, _ir0_t.boxes.cls):
                                            _bn_t = _bb_t.cpu().numpy()
                                            _bw2_t, _bh2_t = _bn_t[2]-_bn_t[0], _bn_t[3]-_bn_t[1]
                                            _blabel_t = _names_t.get(int(_bcls_t)) if _names_t else None
                                            _init_dets_t.append(_Det_t(
                                                bbox_xyxy=_bn_t.astype(_np_t.float32),
                                                bbox_xywh=_np_t.array([_bn_t[0]+_bw2_t/2, _bn_t[1]+_bh2_t/2, _bw2_t, _bh2_t], dtype=_np_t.float32),
                                                confidence=float(_bc_t),
                                                label=_blabel_t,
                                            ))
                                st.session_state[_dets_key_t] = _init_dets_t

                        if _run_scan_t:
                            with st.spinner("Running detector on frame 0..."):
                                from ultralytics import YOLO as _YOLO_scan_t
                                import cv2 as _cv2_scan_t
                                import numpy as _np_scan_t
                                from src.streamlit_template.core.Common.object_detector import Detection as _Det_scan_t

                                _scan_model_t = _YOLO_scan_t(_selected_model_t)
                                _new_dets_t = []
                                _fr0_bgr_t = _cv2_scan_t.imread(str(_tool_frame_files[0]))
                                if _fr0_bgr_t is not None:
                                    _res_t = _scan_model_t.predict(
                                        _fr0_bgr_t, verbose=False, conf=_ui_conf_t, imgsz=1280,
                                        max_det=100, agnostic_nms=True,
                                    )
                                    _fr_h_t, _fr_w_t = _fr0_bgr_t.shape[:2]
                                    if len(_res_t) > 0:
                                        _r0_t = _res_t[0]
                                        _scan_names_t = _r0_t.names or {}
                                        _has_obb_t = _r0_t.obb is not None and len(_r0_t.obb) > 0
                                        _has_boxes_t = _r0_t.boxes is not None and len(_r0_t.boxes) > 0

                                        if _has_obb_t:
                                            for _obox_t in _r0_t.obb:
                                                _c_t = float(_obox_t.conf[0])
                                                _ocls_t2 = int(_obox_t.cls[0]) if _obox_t.cls is not None else None
                                                _olabel_t2 = _scan_names_t.get(_ocls_t2) if _ocls_t2 is not None else None
                                                _xywhr_t2 = _obox_t.xywhr[0].cpu().numpy()
                                                _rect_t2 = ((_xywhr_t2[0], _xywhr_t2[1]), (_xywhr_t2[2], _xywhr_t2[3]), float(_np_scan_t.degrees(_xywhr_t2[4])))
                                                _pts_t2 = _cv2_scan_t.boxPoints(_rect_t2)
                                                _x1_t = float(_np_scan_t.min(_pts_t2[:, 0]))
                                                _y1_t = float(_np_scan_t.min(_pts_t2[:, 1]))
                                                _x2_t = float(_np_scan_t.max(_pts_t2[:, 0]))
                                                _y2_t = float(_np_scan_t.max(_pts_t2[:, 1]))
                                                _bw_t, _bh_t = _x2_t - _x1_t, _y2_t - _y1_t
                                                _area_t = _bw_t * _bh_t
                                                if _area_t > (_fr_h_t * _fr_w_t * (_ui_max_a_t / 100.0)):
                                                    continue
                                                if _area_t < (_fr_h_t * _fr_w_t * (_ui_min_a_t / 100.0)):
                                                    continue
                                                _bcx_t, _bcy_t = _x1_t + _bw_t / 2, _y1_t + _bh_t / 2
                                                _new_dets_t.append(_Det_scan_t(
                                                    bbox_xyxy=_np_scan_t.array([_x1_t, _y1_t, _x2_t, _y2_t], dtype=_np_scan_t.float32),
                                                    bbox_xywh=_np_scan_t.array([_bcx_t, _bcy_t, _bw_t, _bh_t], dtype=_np_scan_t.float32),
                                                    confidence=_c_t,
                                                    label=_olabel_t2,
                                                ))
                                        elif _has_boxes_t:
                                            for _b_t, _conf_t2, _cls_t2 in zip(_r0_t.boxes.xyxy, _r0_t.boxes.conf, _r0_t.boxes.cls):
                                                _b_np_t = _b_t.cpu().numpy()
                                                _c_t2 = float(_conf_t2)
                                                _blabel_t2 = _scan_names_t.get(int(_cls_t2)) if _scan_names_t else None
                                                _w_t, _h_t = _b_np_t[2] - _b_np_t[0], _b_np_t[3] - _b_np_t[1]
                                                _area_t2 = _w_t * _h_t
                                                if _area_t2 > (_fr_h_t * _fr_w_t * (_ui_max_a_t / 100.0)):
                                                    continue
                                                if _area_t2 < (_fr_h_t * _fr_w_t * (_ui_min_a_t / 100.0)):
                                                    continue
                                                _cx_t, _cy_t = _b_np_t[0] + _w_t / 2, _b_np_t[1] + _h_t / 2
                                                _new_dets_t.append(_Det_scan_t(
                                                    bbox_xyxy=_b_np_t.astype(_np_scan_t.float32),
                                                    bbox_xywh=_np_scan_t.array([_cx_t, _cy_t, _w_t, _h_t], dtype=_np_scan_t.float32),
                                                    confidence=_c_t2,
                                                    label=_blabel_t2,
                                                ))

                                _new_dets_t.sort(key=lambda d: d.confidence, reverse=True)
                                st.session_state[_dets_key_t] = _new_dets_t

                        _base_dets_t = st.session_state.get(_dets_key_t, [])

                        if _base_dets_t:
                            import cv2 as _cv2_draw_t
                            import numpy as _np_draw_t
                            _base_bgr_show_t = _cv2_draw_t.imread(str(_tool_base_frame_path))
                            _box_labels_t = []
                            for _di_t, _det_t in enumerate(_base_dets_t):
                                x1_t, y1_t, x2_t, y2_t = [int(v) for v in _det_t.bbox_xyxy]
                                _cv2_draw_t.rectangle(_base_bgr_show_t, (x1_t, y1_t), (x2_t, y2_t), (0, 0, 255), 2)
                                _det_label_t = _det_t.label if getattr(_det_t, "label", None) else None
                                _img_text_t = f"#{_di_t} {_det_label_t} {_det_t.confidence:.2f}" if _det_label_t else f"#{_di_t} {_det_t.confidence:.2f}"
                                _box_text_t = f"#{_di_t} {_det_label_t} — conf {_det_t.confidence:.2f}" if _det_label_t else f"#{_di_t} — conf {_det_t.confidence:.2f}"
                                _cv2_draw_t.putText(_base_bgr_show_t, _img_text_t, (x1_t, max(y1_t-6, 12)),
                                    _cv2_draw_t.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                _box_labels_t.append(_box_text_t)

                            _base_rgb_show_t = _cv2_draw_t.cvtColor(_base_bgr_show_t, _cv2_draw_t.COLOR_BGR2RGB)

                            with _img_col_t:
                                st.image(_base_rgb_show_t, caption="Base frame — detected objects", use_container_width=True)

                            with _ctrl_col_t:
                                _sel_idx_t = st.selectbox(
                                    "Select target object",
                                    options=list(range(len(_base_dets_t))),
                                    format_func=lambda i: _box_labels_t[i],
                                    key=_sel_key_t,
                                )
                                _sel_det_t = _base_dets_t[_sel_idx_t]
                                st.session_state[f"tracking_bbox_{_tool_hash}"] = _sel_det_t.bbox_xyxy.tolist()
                                st.session_state["selected_tracking_bbox_xyxy"] = _sel_det_t.bbox_xyxy.tolist()
                                st.session_state["selected_tracking_label"] = getattr(_sel_det_t, "label", None)
                                st.session_state["selected_tracking_source_hash"] = _tool_hash
                                st.caption(f"Selected bbox: {[int(v) for v in _sel_det_t.bbox_xyxy]}")
                        else:
                            with _ctrl_col_t:
                                st.warning("No objects detected in the base frame. Click Detect on Frame 0.")
                            with _img_col_t:
                                st.image(str(_tool_base_frame_path), caption="Base frame", use_container_width=True)

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
                    using_video_timeline = False
                    local_video_total_frames = 0
                    
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
                    
                    elif not is_svo and not is_bag:
                        # Check if frames were already extracted in Generic Step 1
                        if video_path:
                            import hashlib as _hl
                            _vid_hash = _hl.md5(video_path.encode()).hexdigest()[:8]
                            _generic_frames_dir = Path("data/Generic/frames") / _vid_hash
                            if _generic_frames_dir.exists():
                                frames = sorted(list(_generic_frames_dir.glob("*.jpg")) + list(_generic_frames_dir.glob("*.png")))
                                if frames:
                                    # Get fps from metadata if possible
                                    vinfo = get_video_frame_info(video_path)
                                    _fps = vinfo.get("fps", 30.0) if vinfo else 30.0
                                    frames_idx = {
                                        "frame_paths": [str(p) for p in frames],
                                        "fps": _fps,
                                        "metadata": None
                                    }

                    # Fallback: Extract from video if not found above
                    if not frames_idx and video_path and os.path.exists(video_path):
                        vinfo = get_video_frame_info(video_path)
                        if vinfo and vinfo.get("total_frames", 0) > 0:
                            total_frames = int(vinfo["total_frames"])
                            using_video_timeline = True
                            local_video_total_frames = total_frames
                            ready = _frame_viewer.sync_timeline_touch_state_bounds(0, total_frames - 1, key_prefix="local_video_frames")
                            # Always extract an initial set of frames if no touch state is found, or conditionally
                            selected = _frame_viewer.get_timeline_range_bounds(0, total_frames - 1, key_prefix="local_video_frames")
                            if selected:
                                start_f, end_f = selected
                                st.session_state["local_frame_query"] = (start_f, end_f, 120)
                                frames_idx = extract_and_cache_frames(
                                    video_path,
                                    start_frame=start_f,
                                    end_frame=end_f,
                                    target_frames=120,
                                )
                            if not ready:
                                st.info("Adjust the timeline range to narrow down and selectively load frames.")
                        else:
                            st.warning("Unable to inspect video frames for timeline loading.")
                        
                    if frames_idx and frames_idx.get("frame_paths"):
                        paths = frames_idx["frame_paths"]
                        fps = frames_idx.get("fps", 30.0)
                        metadata = frames_idx.get("metadata")
                        if using_video_timeline:
                            render_frame_grid_viewer(paths, fps, metadata=metadata, key_prefix="local")
                            if hasattr(_frame_viewer, "render_timeline_range_control_bounds"):
                                _frame_viewer.render_timeline_range_control_bounds(0, max(0, local_video_total_frames - 1), key_prefix="local_video_frames")
                        else:
                            ready = _frame_viewer.sync_timeline_touch_state(paths, key_prefix="local_frames") if hasattr(_frame_viewer, "sync_timeline_touch_state") else True
                            if ready:
                                display_paths = _frame_viewer.timeline_trim_paths(paths, key_prefix="local_frames")
                                if display_paths:
                                    render_frame_grid_viewer(display_paths, fps, metadata=metadata, key_prefix="local")
                            else:
                                st.info("Adjust the timeline range to show frames.")
                            if hasattr(_frame_viewer, "render_timeline_range_control"):
                                _frame_viewer.render_timeline_range_control(paths, key_prefix="local_frames")
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
                        default_model_dir = "data/Common/robot_models/openarm"
                        urdf_path = Path(f"{default_model_dir}/openarm.urdf")
                        model_dir = Path(default_model_dir)

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
