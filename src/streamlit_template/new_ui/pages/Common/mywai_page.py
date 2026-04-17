import streamlit as st
import os
import json
import base64
import numpy as np
from pathlib import Path
import logging

# Load search icon as base64 for buttons
_SEARCH_ICON_PATH = Path(__file__).resolve().parent.parent.parent / "Common" / "sources" / "search.png"
if _SEARCH_ICON_PATH.exists():
    with open(_SEARCH_ICON_PATH, "rb") as _f:
        _SEARCH_B64 = base64.b64encode(_f.read()).decode()
else:
    _SEARCH_B64 = None

try:
    from src.streamlit_template.new_ui.services.Generic.mywai_service import (
        login_mywai,
        get_equipments_list,
        get_facts_by_equipment,
        download_blob,
        extract_videos_from_facts,
        process_facts_for_display,
        is_kit_available,
        DEFAULT_ENDPOINT,
        MYWAI_KIT_AVAILABLE,
    )
    from src.streamlit_template.new_ui.services.Generic.mywai_service import download_equipment_model_folder
    MYWAI_API_AVAILABLE = True
except ImportError as e:
    MYWAI_API_AVAILABLE = False
    download_equipment_model_folder = None
    print(f"MYWAI Import Error: {e}")
    # Define dummy functions to prevent NameError if import fails but code execution continues (unlikely but safe)
    def process_facts_for_display(*args): return []


from src.streamlit_template.new_ui.services.Common import robot_service
from src.streamlit_template.new_ui.components.Common.video_player import render_video_player
from src.streamlit_template.new_ui.components.Common.frame_viewer import render_frame_grid_viewer
from src.streamlit_template.new_ui.components.Common import frame_viewer as _frame_viewer
from src.streamlit_template.new_ui.services.Generic.frame_service import extract_and_cache_frames, get_video_frame_info
from src.streamlit_template.new_ui.Common.styles import get_logo_base64
from src.streamlit_template.components.sync_viewer import sync_viewer
from src.streamlit_template.new_ui.services.Common.bag_helpers import extract_bag_frames, reconstruct_video_from_frames
from src.streamlit_template.core.Generic.extract_frames import extract_frames_from_video
from src.streamlit_template.new_ui.services.SVO.svo_helpers import reconstruct_rgb_video, reconstruct_rgbd_video

from src.streamlit_template.new_ui.services.Common.zip_upload_handler import (
    validate_svo_zip,
    extract_svo_zip,
    get_zip_session_hash,
)

logger = logging.getLogger(__name__)

# --- Helper Views ---

def _render_equipment_row(eq, on_select):
    """Render a single row for the Equipment Table."""
    c1, c2, c3, c5, c6 = st.columns([2.5, 2, 2.5, 2, 1])
    
    with c1:
        st.write(f"**{eq.name}**")
        if eq.description:
            st.caption(eq.description[:50] + "..." if len(eq.description) > 50 else eq.description)
            
    with c2:
        if eq.equipmentModel:
            st.success("✔ 3D Model")
        else:
            st.write("—")

    with c3:
        # Site / Area
        loc = []
        if eq.site: loc.append(eq.site.name if hasattr(eq.site, "name") else str(eq.site))
        if eq.area: loc.append(eq.area.name if hasattr(eq.area, "name") else str(eq.area))
        st.write(" / ".join(loc) if loc else "—")

    with c5:
        # Type/Sensors
        st.write("—")

    with c6:
        st.markdown(
            f'''
            <style>
            .st-key-btn_eq_{eq.id} {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .st-key-btn_eq_{eq.id} button {{
                width: 40px !important;
                height: 40px !important;
                padding: 0 !important;
                background-image: url("data:image/png;base64,{_SEARCH_B64}") !important;
                background-size: 20px 20px !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                color: transparent !important;
                border-radius: 4px;
            }}
            </style>
            ''',
            unsafe_allow_html=True,
        )
        if st.button(" ", key=f"btn_eq_{eq.id}", help="View Tasks"):
            on_select(eq)
    
    st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)

def _render_task_row(fact, equipment_name, on_select):
    """Render a single row for the Task Row."""
    c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 2, 2, 1.5, 1, 1, 1])
    
    has_video = bool(fact["videos"])
    has_rgbd = any("depth" in v["label"].lower() or "rgbd" in v["label"].lower() for v in fact["videos"])
    task_type_str = fact.get("task_type", "video").upper()
    
    with c1:
        st.write(f"**{fact['name']}**")
    
    with c2:
        st.write(f"{fact['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
    with c3:
        st.write(equipment_name)
        
    with c4:
        st.write(task_type_str)

    with c5:
        if has_video:
             st.success("✔ Video")
        else:
             st.write("—")

    with c6:
        if has_rgbd:
            st.success("✔ RGBD")
        else:
            st.write("—")
            
    with c7:
        st.markdown(
            f'''
            <style>
            .st-key-btn_fact_{fact['fact_id']} {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .st-key-btn_fact_{fact['fact_id']} button {{
                width: 40px !important;
                height: 40px !important;
                padding: 0 !important;
                background-image: url("data:image/png;base64,{_SEARCH_B64}") !important;
                background-size: 20px 20px !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                color: transparent !important;
                border-radius: 4px;
            }}
            </style>
            ''',
            unsafe_allow_html=True,
        )
        if st.button(" ", key=f"btn_fact_{fact['fact_id']}", help="View Playback"):
            on_select(fact)

    st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)


# --- Main Page ---

def render_mywai_page(token: str, endpoint: str) -> None:
    # Persist token + endpoint so other services (save_animation) can upload back
    st.session_state["mywai_token"] = token
    st.session_state["mywai_endpoint"] = endpoint
    # State Init
    if "mywai_view_level" not in st.session_state:
        st.session_state.mywai_view_level = "site_list" # site_list, area_list, equipment_list, tasks_list, playback
    if "mywai_selected_site" not in st.session_state:
        st.session_state.mywai_selected_site = None
    if "mywai_selected_area" not in st.session_state:
        st.session_state.mywai_selected_area = None
    if "mywai_selected_equipment" not in st.session_state:
        st.session_state.mywai_selected_equipment = None
    if "mywai_selected_fact" not in st.session_state:
        st.session_state.mywai_selected_fact = None

    # --- Page Header ---
    logo_b64 = get_logo_base64("MYWAI.png")
    if logo_b64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_b64}" style="width: 40px;">
                <h2 style="margin: 0; padding: 0;">MYWAI Platform</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.header("MYWAI Platform")

    # Header / Breadcrumbs
    header_col1, header_col_back, header_col_pipeline, header_col_logout = st.columns([7.2, 1, 1.1, 0.7])
    
    st.markdown(
        '''
        <style>
        .st-key-mywai_back_sites, .st-key-mywai_back_areas, .st-key-mywai_back_eq, .st-key-mywai_back_tasks,
        .st-key-mywai_logout, .st-key-mywai_run_pipeline {
            display: flex;
            justify-content: flex-end;
            margin-top: 0px !important;
        }
        .st-key-mywai_back_sites button, .st-key-mywai_back_areas button, .st-key-mywai_back_eq button, .st-key-mywai_back_tasks button,
        .st-key-mywai_logout button, .st-key-mywai_run_pipeline button {
            height: 50px;
        }
        </style>
        ''', unsafe_allow_html=True
    )
    
    with header_col_pipeline:
        pipeline_btn_placeholder = st.empty()

    with header_col1:
         # Dynamic Breadcrumb Header
        steps = []
        state_site = st.session_state.get("mywai_selected_site")
        state_area = st.session_state.get("mywai_selected_area")
        state_eq = st.session_state.get("mywai_selected_equipment")
        state_fact = st.session_state.get("mywai_selected_fact")

        if st.session_state.mywai_view_level == "site_list":
            steps.append(" Sites")
        elif st.session_state.mywai_view_level == "area_list":
            if state_site: steps.append(f" {state_site.name}")
            steps.append(" Areas")
        elif st.session_state.mywai_view_level == "equipment_list":
            if state_site: steps.append(f" {state_site.name}")
            if state_area: steps.append(f" {state_area.name}")
            steps.append(" Equipment")
        elif st.session_state.mywai_view_level == "tasks_list":
            if state_site: steps.append(f" {state_site.name}")
            if state_area: steps.append(f" {state_area.name}")
            if state_eq: steps.append(f" {state_eq.name}")
            steps.append(" Tasks")
        elif st.session_state.mywai_view_level == "playback":
            if state_site: steps.append(f" {state_site.name}")
            if state_area: steps.append(f" {state_area.name}")
            if state_eq: steps.append(f" {state_eq.name}")
            steps.append(" Tasks")
            if state_fact: steps.append(f" {state_fact['name']}")
        
        breadcrumb_html = ""
        for i, step in enumerate(steps):
             if i > 0:
                  breadcrumb_html += '<span style="opacity: 0.4; margin: 0 10px; font-size: 0.9em;">❯</span>'
             
             # Highlight the last step
             if i == len(steps) - 1:
                  breadcrumb_html += f'<span style="font-weight: 600; color: var(--primary-color);">{step}</span>'
             else:
                  breadcrumb_html += f'<span style="font-weight: 500; opacity: 0.7;">{step}</span>'
                  
        st.markdown(
            f'''
            <div style="
                background-color: rgba(128, 128, 128, 0.08); 
                padding: 0px 20px; 
                border-radius: 8px; 
                border: 1px solid rgba(128, 128, 128, 0.15);
                margin-bottom: 0px; 
                font-size: 1.15em;
                display: flex;
                align-items: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
                height: 50px;
            ">
                {breadcrumb_html}
            </div>
            ''',
            unsafe_allow_html=True
        )
        
    with header_col_back:

        if st.session_state.mywai_view_level == "area_list":
            if st.button("← Back to Sites", key="mywai_back_sites"):
                st.session_state.mywai_view_level = "site_list"
                st.session_state.mywai_selected_site = None
                st.rerun()
        elif st.session_state.mywai_view_level == "equipment_list":
            if st.button("← Back to Areas", key="mywai_back_areas"):
                st.session_state.mywai_view_level = "area_list"
                st.session_state.mywai_selected_area = None
                st.rerun()
        elif st.session_state.mywai_view_level == "tasks_list":
            if st.button("← Back to Equipment", key="mywai_back_eq"):
                st.session_state.mywai_view_level = "equipment_list"
                st.session_state.mywai_selected_equipment = None
                st.rerun()
        elif st.session_state.mywai_view_level == "playback":
            if st.button("← Back to Tasks", key="mywai_back_tasks"):
                st.session_state.mywai_view_level = "tasks_list"
                st.session_state.mywai_selected_fact = None
                st.session_state.selected_video_blob = None
                st.rerun()

    with header_col_logout:

        if st.button("Logout", key="mywai_logout"):
            if "mywai_auth" in st.session_state: del st.session_state["mywai_auth"]
            if "auth_token" in st.session_state: del st.session_state["auth_token"]
            st.session_state["selected_platform"] = None
            st.session_state.mywai_view_level = "site_list"
            st.session_state.mywai_selected_site = None
            st.session_state.mywai_selected_area = None
            st.session_state.mywai_selected_equipment = None
            st.session_state.mywai_selected_fact = None
            st.rerun()

    # Global line removed, moved down into specific views

    # --- VIEW 1: SITE LIST ---
    if st.session_state.mywai_view_level == "site_list":
        st.markdown("---")
        @st.cache_data(ttl=300, show_spinner=False)
        def _get_cached_equipments(token, endpoint):
            return get_equipments_list(token=token, endpoint=endpoint)

        with st.spinner("Loading sites..."):
             result = _get_cached_equipments(token, endpoint)

        if not isinstance(result, tuple) or not result[0]:
            st.error(f"Failed to load equipments: {result}")
            return

        equipments = result[1] or []
        
        # Extract unique sites
        sites = {}
        for eq in equipments:
            if eq.site and getattr(eq.site, "id", None):
                sites[eq.site.id] = eq.site
        
        if not sites:
            st.info("No sites found.")
            return

        # Table Header
        h1, h2 = st.columns([8, 2])
        h1.markdown("**Site Name**")
        h2.markdown("**Action**")
        st.markdown("---")

        def select_site(site):
            st.session_state.mywai_selected_site = site
            st.session_state.mywai_view_level = "area_list"
            st.rerun()

        for site_id, site in sites.items():
            c1, c2 = st.columns([8, 2])
            with c1:
                st.write(f"**{site.name}**")
            with c2:
                st.markdown(
                    f'''
                    <style>
                    .st-key-btn_site_{site.id} {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }}
                    .st-key-btn_site_{site.id} button {{
                        width: 40px !important;
                        height: 40px !important;
                        padding: 0 !important;
                        background-image: url("data:image/png;base64,{_SEARCH_B64}") !important;
                        background-size: 20px 20px !important;
                        background-position: center !important;
                        background-repeat: no-repeat !important;
                        color: transparent !important;
                        border-radius: 4px;
                    }}
                    </style>
                    ''',
                    unsafe_allow_html=True,
                )
                if st.button(" ", key=f"btn_site_{site.id}", help="View Areas"):
                    select_site(site)
            st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)
            
    # --- VIEW 1.5: AREA LIST ---
    elif st.session_state.mywai_view_level == "area_list":
        selected_site = st.session_state.mywai_selected_site
            
        st.markdown("---")
        @st.cache_data(ttl=300, show_spinner=False)
        def _get_cached_equipments(token, endpoint):
            return get_equipments_list(token=token, endpoint=endpoint)

        with st.spinner("Loading areas..."):
             result = _get_cached_equipments(token, endpoint)

        if not isinstance(result, tuple) or not result[0]:
            st.error(f"Failed to load equipments: {result}")
            return
            
        equipments = result[1] or []
        
        # Extract unique areas for the selected site
        areas = {}
        for eq in equipments:
            if eq.site and getattr(eq.site, "id", None) == selected_site.id:
                if eq.area and getattr(eq.area, "id", None):
                    areas[eq.area.id] = eq.area
                    
        if not areas:
            st.info("No areas found for this site.")
            return
                    
        # Table Header
        h1, h2 = st.columns([8, 2])
        h1.markdown("**Area Name**")
        h2.markdown("**Action**")
        st.markdown("---")

        def select_area(area):
            st.session_state.mywai_selected_area = area
            st.session_state.mywai_view_level = "equipment_list"
            st.rerun()

        for area_id, area in areas.items():
            c1, c2 = st.columns([8, 2])
            with c1:
                st.write(f"**{area.name}**")
            with c2:
                st.markdown(
                    f'''
                    <style>
                    .st-key-btn_area_{area.id} {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }}
                    .st-key-btn_area_{area.id} button {{
                        width: 40px !important;
                        height: 40px !important;
                        padding: 0 !important;
                        background-image: url("data:image/png;base64,{_SEARCH_B64}") !important;
                        background-size: 20px 20px !important;
                        background-position: center !important;
                        background-repeat: no-repeat !important;
                        color: transparent !important;
                        border-radius: 4px;
                    }}
                    </style>
                    ''',
                    unsafe_allow_html=True,
                )
                if st.button(" ", key=f"btn_area_{area.id}", help="View Equipment"):
                    select_area(area)
            st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)

    # --- VIEW 2: EQUIPMENT LIST ---
    elif st.session_state.mywai_view_level == "equipment_list":
        selected_site = st.session_state.mywai_selected_site
        selected_area = st.session_state.mywai_selected_area
            
        st.markdown("---")
        @st.cache_data(ttl=300, show_spinner=False)
        def _get_cached_equipments(token, endpoint):
            return get_equipments_list(token=token, endpoint=endpoint)

        with st.spinner("Loading equipments..."):
             result = _get_cached_equipments(token, endpoint)

        if not isinstance(result, tuple) or not result[0]:
            st.error(f"Failed to load equipments: {result}")
            return

        equipments = result[1] or []
        
        # Filter equipments by selected site and area
        filtered_equipments = [
            eq for eq in equipments 
            if eq.site and getattr(eq.site, "id", None) == selected_site.id and
               eq.area and getattr(eq.area, "id", None) == selected_area.id
        ]
        
        # Table Header
        h1, h2, h3, h5, h6 = st.columns([2.5, 2, 2.5, 2, 1])
        h1.markdown("**Name**")
        h2.markdown("**Custom 3D Model**")
        h3.markdown("**Site/Area**")
        h5.markdown("**Type**")
        h6.markdown("**Action**")
        st.markdown("---")

        def select_equipment(eq):
            st.session_state.mywai_selected_equipment = eq
            st.session_state.mywai_view_level = "tasks_list"
            st.rerun()

        for eq in filtered_equipments:
            _render_equipment_row(eq, select_equipment)


    # --- VIEW 2: TASKS LIST ---
    elif st.session_state.mywai_view_level == "tasks_list":
        equipment = st.session_state.mywai_selected_equipment
            
        st.markdown("---")
        @st.cache_data(ttl=300, show_spinner=False)
        def _get_cached_facts(eq_id, token, endpoint):
            return get_facts_by_equipment(equipment_id=eq_id, token=token, endpoint=endpoint)

        with st.spinner(f"Loading tasks for {equipment.name}..."):
            res = _get_cached_facts(equipment.id, token, endpoint)
            
        if not isinstance(res, tuple) or not res[0]:
            st.error(f"Failed to load facts: {res}")
            return
            
        grouped_facts = process_facts_for_display(res[1])
        
        # Table Header
        h1, h2, h3, h4, h5, h6, h7 = st.columns([2, 2, 2, 1.5, 1, 1, 1])
        h1.markdown("**Task Name**")
        h2.markdown("**Date**")
        h3.markdown("**Equipment**")
        h4.markdown("**Data Type**")
        h5.markdown("**Has Video**")
        h6.markdown("**Has RGBD**") 
        h7.markdown("**Action**")
        st.markdown("---")
        
        def select_fact(f):
            st.session_state.mywai_selected_fact = f
            st.session_state.mywai_view_level = "playback"
            st.rerun()
            
        if not grouped_facts:
            st.info("No tasks found for this equipment.")
        
        for fact in grouped_facts:
            _render_task_row(fact, equipment.name, select_fact)


    # --- VIEW 3: PLAYBACK VIEW ---
    elif st.session_state.mywai_view_level == "playback":
        fact = st.session_state.mywai_selected_fact
        equipment = st.session_state.mywai_selected_equipment
        
        # Top Row: Navigation and Actions
        top_c1, top_c2 = st.columns([5, 1])
        with top_c1:
            pass
                
        st.markdown("---")

        # Layout: Video (Left) | Tabs (Right)
        col_video, col_tabs = st.columns([1, 1])
        
        # --- VIDEO PLAYER / CONCURRENT DOWNLOADER (Left Column) ---
        with col_video:
            videos = fact["videos"]
            task_type = fact.get("task_type", "video")
            
            _is_bag = task_type == "bag"
            _is_svo = task_type == "svo"
            _playback_video_path = None
            
            # --- SVO SESSION HANDLING ---
            if _is_svo:
                session_id = fact["session_id"]
                svo_base_path = Path("data/SVO")
                rgb_dir = svo_base_path / "frames" / session_id
                depth_dir = svo_base_path / "depth_meters" / session_id
                cam_dir = svo_base_path / "camera"
                video_dir = svo_base_path / "videos" / session_id
                
                _video_out = video_dir / f"{session_id}.mp4"
                
                # Always scan for missing files and retry downloading them
                download_items = []
                for v in videos:
                    filename = v["name"]
                    if v.get("ext") == "jpg" or v.get("ext") == "png":
                        local_path = rgb_dir / filename
                    elif v.get("ext") == "npy":
                        if "camera_intrinsics" in filename.lower():
                            local_path = cam_dir / f"{session_id}.npy"
                        else:
                            local_path = depth_dir / filename
                    elif v.get("ext") == "npz":
                        local_path = cam_dir / f"{session_id}.npz"
                    else:
                        local_path = svo_base_path / "misc" / filename
                    
                    # Only queue files that don't exist yet
                    if not local_path.exists():
                        download_items.append({
                            "fact_id": v.get("fact_id", fact["fact_id"]),
                            "file_path": v["file_path"],
                            "local_path": str(local_path)
                        })
                
                if download_items:
                    already = len(videos) - len(download_items)

                    with st.spinner(f"Downloading {len(download_items)} files concurrently..."):
                        from src.streamlit_template.new_ui.services.Generic.mywai_service import download_blobs_concurrently
                        ok, msg = download_blobs_concurrently(download_items, token, endpoint)
                        
                    if ok:
                        st.success(msg)
                    else:
                        st.warning(msg)
                
                _is_downloaded = rgb_dir.exists() and any(rgb_dir.iterdir())
                
                # Convert .npz intrinsics → .npy dict (pipeline requires .npy with width/height)
                _intr_npz = cam_dir / f"{session_id}.npz"
                _intr_npy = cam_dir / f"{session_id}.npy"
                if _is_downloaded and _intr_npz.exists() and not _intr_npy.exists():
                    try:
                        import cv2 as _cv2
                        _d = np.load(str(_intr_npz))
                        _keys = list(_d.keys())
                        if "fx" in _keys and "fy" in _keys and "cx" in _keys and "cy" in _keys:
                            _w = int(_d["width"]) if "width" in _keys else 0
                            _h = int(_d["height"]) if "height" in _keys else 0
                            # Infer resolution from downloaded frames if missing
                            if _w == 0 or _h == 0:
                                _sample = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
                                if _sample:
                                    _img = _cv2.imread(str(_sample[0]))
                                    if _img is not None:
                                        _h, _w = _img.shape[:2]
                            if _w == 0 or _h == 0:
                                _w, _h = 1280, 720
                                st.warning("Could not infer resolution, defaulting to 1280×720.")
                            _intr_dict = {
                                "fx": float(_d["fx"]), "fy": float(_d["fy"]),
                                "cx": float(_d["cx"]), "cy": float(_d["cy"]),
                                "width": _w, "height": _h,
                            }
                            np.save(str(_intr_npy), _intr_dict)

                    except Exception as _e:
                        st.warning(f"Could not convert intrinsics: {_e}")

                # Once downloaded, prepare Video
                if _is_downloaded:
                    if not _video_out.exists():
                        with st.spinner("Reconstructing RGB Video..."):
                            reconstruct_rgb_video(rgb_dir, str(_video_out), fps=15)
                    
                    # Reconstruct depth video too
                    _depth_video_out = video_dir / f"{session_id}_depth.mp4"
                    if not _depth_video_out.exists() and depth_dir.exists() and any(depth_dir.glob("*.npy")):
                        with st.spinner("Reconstructing Depth Video..."):
                            reconstruct_rgbd_video(depth_dir, str(_depth_video_out), fps=15)
                    
                    if _video_out.exists():
                        st.session_state["svo_session_id"] = session_id
                        st.session_state["svo_base_path"] = "data/SVO"
                        
                        # Video mode selector (RGB vs Depth)
                        _has_depth_vid = _depth_video_out.exists() if '_depth_video_out' in dir() else False
                        if _has_depth_vid:
                            _vid_mode = st.radio(
                                "Video Mode",
                                ["RGB", "Depth"],
                                horizontal=True,
                                key="mywai_svo_vid_mode",
                            )
                            _playback_video_path = str(_depth_video_out) if _vid_mode == "Depth" else str(_video_out)
                        else:
                            _playback_video_path = str(_video_out)
                        
                        st.session_state["persistent_video_path"] = _playback_video_path
                        st.session_state["local_video_path"] = _playback_video_path
                        
                        # Store placeholder in session_state so we can empty it before leaving the page
                        if "mywai_video_container" not in st.session_state:
                            st.session_state["mywai_video_container"] = st.empty()
                        
                        seek_time = st.session_state.get("video_seek_time", 0)
                        with st.session_state["mywai_video_container"].container():
                            st.video(_playback_video_path, start_time=int(seek_time))
                        
                        # Set selected_video_data mock so pipeline button activates
                        selected_video_data = {"local_path": _playback_video_path}
                    else:
                        st.error("Frame reconstruction failed. Check downloaded files.")
                        selected_video_data = None
                else:
                    selected_video_data = None


            # --- ROBOT MODEL DOWNLOAD EXTRACTION ---
            # Extract models globally so they are ready for the Pipeline regardless of Tab clicks
            use_custom = False
            CONTAINER = "equipmentmodels"
            if equipment and equipment.equipmentModel:
                eq_id = getattr(equipment, "id", "unknown")
                local_dir_key = f"mywai_robot_dir_eq_{eq_id}"
                
                # Build target path precisely how pipeline expects it
                task_type = fact.get("task_type", "generic").upper()
                session_id = fact.get("session_id", "unknown")
                if task_type != "SVO" and "name" in fact:
                    session_id = fact["name"].replace(" ", "_")
                    
                target_local_dir = f"data/{task_type}/downloads/{session_id}/robot_models/eq_{eq_id}"
                
                local_robot_dir = st.session_state.get(local_dir_key)
                
                if not local_robot_dir or not os.path.exists(local_robot_dir):
                    with st.spinner("Downloading Robot URDF (+ config/meshes) in background..."):
                        from src.streamlit_template.new_ui.services.Generic.mywai_service import download_equipment_model_folder
                        dl_ok, dl_msg, local_robot_dir = download_equipment_model_folder(
                            equipment_id=eq_id,
                            model_blob_path=equipment.equipmentModel,
                            target_local_dir=target_local_dir,
                            container=CONTAINER,
                            token=token,
                            endpoint=endpoint,
                        )
                    if dl_ok:
                        st.session_state[local_dir_key] = local_robot_dir
                    else:
                        st.warning(f"Failed pulling custom robot model: {dl_msg}")
            
                if local_robot_dir and os.path.exists(local_robot_dir):
                    use_custom = True
                    # Set the magic keys expected by pipeline_page -> _resolve_active_robot()
                    st.session_state["custom_robot_dir"] = local_robot_dir
                    st.session_state["active_custom_robot_dir"] = local_robot_dir
                    
                    blob_dir = "/".join(equipment.equipmentModel.split("/")[:-1])
                    st.session_state["mywai_active_model_blob_base"] = blob_dir
                    st.session_state["mywai_active_model_container"] = CONTAINER
                else:
                    st.session_state.pop("custom_robot_dir", None)
                    st.session_state.pop("active_custom_robot_dir", None)
                    st.session_state.pop("mywai_active_model_blob_base", None)
                    st.session_state.pop("mywai_active_model_container", None)


            # --- BAG OR STANDARD VIDEO HANDLING ---
            else:
                # First, prepare basic paths for download
                video_objs = []
                for v in videos:
                    loc_path = f"data/Generic/downloads/{v['file_path'].replace('/', '_')}"
                    video_objs.append({
                        "label": v["label"],
                        "file_path": v["file_path"],
                        "local_path": loc_path,
                        "name": v["name"],
                        "container": v["container"]
                    })
                
                def on_download(video_item):
                    with st.spinner(f"Downloading {video_item['label']}..."):
                        success, msg = download_blob(
                            container=video_item["container"],
                            file_path=video_item["file_path"],
                            output_path=video_item["local_path"],
                            token=token,
                            endpoint=endpoint
                        )
                        if success:
                            st.success("Downloaded!")
                            st.rerun()
                        else:
                            st.error(f"Download failed: {msg}")
    
                selected_label = st.session_state.get(f"player_{fact['fact_id']}_layer_select", video_objs[0]["label"] if video_objs else None)
                selected_video_data = next((v for v in video_objs if v["label"] == selected_label), video_objs[0] if video_objs else None)
    
                if selected_video_data and os.path.exists(selected_video_data["local_path"]):
                    _dl_path = selected_video_data["local_path"]
                    _dl_name = selected_video_data.get("name", os.path.basename(_dl_path)).lower()
    
                    if _dl_name.endswith(".bag"):
                        _is_bag = True
                        import hashlib
                        # Generate session id based on filename (deterministic, avoids duplicates)
                        _bag_fname = os.path.basename(_dl_path)
                        _bag_name_clean = "".join(c for c in _bag_fname.rsplit('.', 1)[0] if c.isalnum() or c in ('_', '-'))
                        _bag_hash = hashlib.md5(_bag_fname.encode()).hexdigest()[:8]
                        _bag_sid = f"{_bag_name_clean}_{_bag_hash}"
                        _bag_root = Path("data/BAG")
        
                        _rgb_dir = _bag_root / "frames" / _bag_sid
                        _video_base = _bag_root / "videos" / _bag_sid
                        _video_base.mkdir(parents=True, exist_ok=True)
                        _cam_base = _bag_root / "camera"
                        _cam_base.mkdir(parents=True, exist_ok=True)
                        _intrinsics = _cam_base / f"{_bag_sid}.npy"
                        _video_out = _video_base / "reconstructed_video.mp4"
                        _depth_out = _video_base / "reconstructed_depth.mp4"
        
                        _need_extract = not _video_out.exists() or not _depth_out.exists() or not _intrinsics.exists()
                        if _need_extract:
                            with st.spinner(f"Extracting BAG frames ({_bag_sid})..."):
                                extract_bag_frames(os.path.abspath(_dl_path), _bag_root, _bag_sid, intrinsics_out_path=_intrinsics)
                            with st.spinner("Reconstructing RGB video..."):
                                reconstruct_video_from_frames(_rgb_dir, str(_video_out))
                            with st.spinner("Reconstructing Depth video..."):
                                _dc_dir = _bag_root / "depth_color" / _bag_sid
                                reconstruct_video_from_frames(_dc_dir, str(_depth_out))
        
                        # Store session state for BAG pipeline
                        st.session_state["bag_session_id"] = _bag_sid
                        st.session_state["bag_session_dir"] = str(_bag_root)
                        st.session_state["bag_file_path"] = os.path.abspath(_dl_path)
                        _playback_video_path = str(_video_out) if _video_out.exists() else None
                        if _playback_video_path:
                            st.session_state["persistent_video_path"] = _playback_video_path
                            st.session_state["local_video_path"] = _playback_video_path
                            
                    else:
                        _playback_video_path = _dl_path
    
                # Run standard video player component
                if selected_video_data and _playback_video_path and os.path.exists(_playback_video_path):
                    selected_video_data["local_path"] = _playback_video_path
    
                selected_video_data = render_video_player(
                    videos=video_objs,
                    download_callback=on_download,
                    key_prefix=f"player_{fact['fact_id']}"
                )

        # Run Pipeline Button (Top Right Header - dependent on video selection)
        with pipeline_btn_placeholder:

            if selected_video_data and os.path.exists(selected_video_data.get("local_path", "")):
                if st.button("Run Pipeline", type="primary", key="mywai_run_pipeline"):
                    # Fix ghost video: Empty the video container before leaving the page
                    if "mywai_video_container" in st.session_state:
                        st.session_state["mywai_video_container"].empty()

                    st.session_state["pipeline_source"] = "mywai"
                    st.session_state["pipeline_running"] = True

                    # Route to correct pipeline based on file type
                    if _is_bag:
                        st.session_state["selected_platform"] = "bag_pipeline"
                        # bag_file_path, bag_session_id already set above
                        st.session_state["local_video_path"] = st.session_state.get("persistent_video_path", os.path.abspath(selected_video_data["local_path"]))
                    elif _is_svo:
                        st.session_state["selected_platform"] = "svo_pipeline"
                        st.session_state["svo_base_path"] = "data/SVO"
                        st.session_state["local_video_path"] = st.session_state.get("persistent_video_path", os.path.abspath(selected_video_data["local_path"]))
                    else:
                        st.session_state["selected_platform"] = "pipeline"
                        st.session_state["local_video_path"] = os.path.abspath(selected_video_data["local_path"])
                    
                    # Reset pipeline state
                    from src.streamlit_template.new_ui.components.Common.stepper_bar import PIPELINE_STEPS
                    st.session_state.pipeline_statuses = ["pending"] * len(PIPELINE_STEPS)
                    st.session_state.step_results = {}
                    st.session_state.selected_step = None
                    
                    # Terminate MYWAI Page Lifecycle (keep auth token)
                    if "mywai_view_level" in st.session_state: del st.session_state["mywai_view_level"]
                    if "mywai_selected_site" in st.session_state: del st.session_state["mywai_selected_site"]
                    if "mywai_selected_area" in st.session_state: del st.session_state["mywai_selected_area"]
                    if "mywai_selected_equipment" in st.session_state: del st.session_state["mywai_selected_equipment"]
                    if "mywai_selected_fact" in st.session_state: del st.session_state["mywai_selected_fact"]
                    if "selected_video_blob" in st.session_state: del st.session_state["selected_video_blob"]
                    
                    st.rerun()

        # --- TABS CONTAINER (Right Column) ---
        with col_tabs:
            t_frames, t_details, t_robot = st.tabs(["🎞️ Frames", "ℹ️ Details", "🤖 Robot"])
            
            # TAB 1: Frames
            with t_frames:
                using_video_timeline = False
                mywai_video_total_frames = 0
                if _is_svo and rgb_dir.exists():
                    # For SVO sessions, show the actual downloaded frames directly
                    import re as _re
                    _frame_re = _re.compile(r'frame_(\d+)')
                    _frame_paths = sorted(
                        [str(p) for p in rgb_dir.glob("*frame_*.jpg")] + [str(p) for p in rgb_dir.glob("*frame_*.png")],
                        key=lambda x: int(_frame_re.search(x).group(1)) if _frame_re.search(x) else 0
                    )
                    if _frame_paths:
                        ready = _frame_viewer.sync_timeline_touch_state(_frame_paths, key_prefix="mywai_svo_frames") if hasattr(_frame_viewer, "sync_timeline_touch_state") else True
                        if ready:
                            trimmed_paths = _frame_viewer.timeline_trim_paths(_frame_paths, key_prefix="mywai_svo_frames")
                            if trimmed_paths:
                                render_frame_grid_viewer(trimmed_paths, 15.0, metadata=None, key_prefix="mywai_svo")
                        else:
                            st.info("Adjust the timeline range to show frames.")
                        if hasattr(_frame_viewer, "render_timeline_range_control"):
                            _frame_viewer.render_timeline_range_control(_frame_paths, key_prefix="mywai_svo_frames")
                    else:
                        st.info("No frames found in download directory.")
                elif selected_video_data and os.path.exists(selected_video_data["local_path"]):
                    _vpath = selected_video_data["local_path"]
                    f_idx = None
                    vinfo = get_video_frame_info(_vpath)
                    if vinfo and vinfo.get("total_frames", 0) > 0:
                        total_frames = int(vinfo["total_frames"])
                        using_video_timeline = True
                        mywai_video_total_frames = total_frames
                        key_prefix = f"mywai_video_frames_{fact.get('fact_id', 'default')}"
                        ready = _frame_viewer.sync_timeline_touch_state_bounds(0, total_frames - 1, key_prefix=key_prefix) if hasattr(_frame_viewer, "sync_timeline_touch_state_bounds") else True
                        if ready:
                            selected = _frame_viewer.get_timeline_range_bounds(0, total_frames - 1, key_prefix=key_prefix) if hasattr(_frame_viewer, "get_timeline_range_bounds") else (0, total_frames - 1)
                            start_f, end_f = selected
                            f_idx = extract_and_cache_frames(
                                _vpath,
                                start_frame=start_f,
                                end_frame=end_f,
                                target_frames=120,
                            )
                        else:
                            st.info("Adjust the timeline range to load and show frames.")
                    else:
                        st.warning("Unable to inspect video frames for timeline loading.")

                    if f_idx and f_idx.get("frame_paths"):
                        source_paths = f_idx["frame_paths"]
                        if using_video_timeline:
                            render_frame_grid_viewer(source_paths, f_idx.get("fps", 30.0), metadata=f_idx.get("metadata"), key_prefix="mywai")
                            if hasattr(_frame_viewer, "render_timeline_range_control_bounds"):
                                key_prefix = f"mywai_video_frames_{fact.get('fact_id', 'default')}"
                                _frame_viewer.render_timeline_range_control_bounds(0, max(0, mywai_video_total_frames - 1), key_prefix=key_prefix)
                        else:
                            ready = _frame_viewer.sync_timeline_touch_state(source_paths, key_prefix="mywai_frames") if hasattr(_frame_viewer, "sync_timeline_touch_state") else True
                            if ready:
                                trimmed_paths = _frame_viewer.timeline_trim_paths(source_paths, key_prefix="mywai_frames")
                                if trimmed_paths:
                                    render_frame_grid_viewer(trimmed_paths, f_idx.get("fps", 30.0), metadata=f_idx.get("metadata"), key_prefix="mywai")
                            else:
                                st.info("Adjust the timeline range to show frames.")
                            if hasattr(_frame_viewer, "render_timeline_range_control"):
                                _frame_viewer.render_timeline_range_control(source_paths, key_prefix="mywai_frames")
                    else:
                        st.info("No frames extracted.")
                else:
                    st.info("Select and download a video to view frames.")
            
            # TAB 2: Details
            with t_details:
                st.json(fact)

            # TAB 3: Robot Preview
            with t_robot:
                # Robot Viewer Logic
                viewer_data = None
                
                if st.session_state.get("custom_robot_dir"):
                    local_robot_dir = st.session_state.get("custom_robot_dir")
                    viewer_data = robot_service.prepare_robot_viewer_data(local_robot_dir, is_zip=False)
                    if not viewer_data:
                        st.warning("Failed to prepare custom model. Using default.")
                        use_custom = False
                
                if not st.session_state.get("custom_robot_dir") or not use_custom:
                    # Clear stale custom robot keys so pipeline falls back cleanly
                    st.session_state.pop("custom_robot_dir", None)
                    st.session_state.pop("active_custom_robot_dir", None)
                    st.session_state.pop("mywai_active_model_blob_base", None)
                    st.session_state.pop("mywai_active_model_container", None)
                    default_urdf = "data/Common/robot_models/openarm/openarm.urdf"
                    if os.path.exists(default_urdf):
                        viewer_data = robot_service.prepare_robot_viewer_data(default_urdf, is_zip=False)
                    else:
                        st.error("Default robot model not found!")
    
                if viewer_data:
                    # Give the viewer a new key on every render so it always
                    # reinitialises when the user navigates to this tab.
                    # Streamlit renders all tab blocks on every rerun; the new
                    # key is invisible while the tab is hidden and the viewer
                    # is already fresh by the time the user clicks through to it.
                    _mount_key = st.session_state.get("_mywai_robot_mount_key", 0) + 1
                    st.session_state["_mywai_robot_mount_key"] = _mount_key
                    preview_key = (
                        f"mywai_robot_preview_"
                        f"{fact.get('fact_id', 'default')}_{_mount_key}"
                    )
                    sync_viewer(viewer_data=viewer_data, video_path=None, key=preview_key)
                else:
                    st.write("No robot model available.")

