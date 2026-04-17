import os
import streamlit.components.v1 as components
import json

_RELEASE = False

if not _RELEASE:
    # During development, point to the local frontend directory
    _component_func = components.declare_component(
        "sync_viewer",
        path=os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend")),
    )
else:
    # For release, point to the build directory (same here for now)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend")
    _component_func = components.declare_component("sync_viewer", path=build_dir)

def sync_viewer(
    viewer_data: dict,
    video_path: str = None,
    key=None,
    height: int | None = None,
    play_token: int | None = None,
    autoplay: bool = True,
):
    """
    Component that renders a synced Video Player and 3D Robot Viewer.
    
    Args:
        video_path (str): URL or base64 data URI of the video.
        viewer_data (dict): Dictionary conforming to MYWAI Viewer Data schema.
    """
    # Ensure viewer_data is valid JSON serializable
    viewer_data_json = json.dumps(viewer_data)
    
    component_kwargs = {
        "video_path": video_path,
        "viewer_data": viewer_data_json,
        "key": key,
        "default": None,
        "autoplay": bool(autoplay),
    }
    if height is not None:
        component_kwargs["height"] = int(height)
    if play_token is not None:
        component_kwargs["play_token"] = int(play_token)

    return _component_func(**component_kwargs)
