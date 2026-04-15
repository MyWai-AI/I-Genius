"""
Common BAG helper service — BAG extraction and video reconstruction utilities.
Used by both local_page and external_page for .bag file handling.
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Read SUBSAMPLE from _paper_config so changing it there takes effect here.
# ---------------------------------------------------------------------------
def _load_paper_subsample(fallback: int = 3) -> int:
    """Import SUBSAMPLE from core/BAG/_paper_config.py with a safe fallback."""
    try:
        _core_bag = Path(__file__).resolve().parents[3] / "core" / "BAG"
        if str(_core_bag) not in sys.path:
            sys.path.insert(0, str(_core_bag))
        from _paper_config import SUBSAMPLE  # noqa: PLC0415
        return int(SUBSAMPLE)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Async I/O helpers – offload disk writes to a thread pool so the main
# extraction loop is never blocked waiting for cv2.imwrite / np.save.
# ---------------------------------------------------------------------------

def _save_rgb(path: str, bgr: np.ndarray):
    cv2.imwrite(path, bgr)

def _save_depth_raw(path: str, depth_image: np.ndarray):
    cv2.imwrite(path, depth_image)

def _save_depth_meters(path: str, depth_meters: np.ndarray):
    np.save(path, depth_meters)

def _save_depth_colormap(path: str, depth_meters: np.ndarray):
    depth_vis = np.where((depth_meters > 0) & (depth_meters < 5.0), depth_meters, 0)
    depth_vis_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_vis_norm, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_colormap)


def extract_bag_frames(
    bag_path: str,
    bag_root: Path,
    session_id: str,
    intrinsics_out_path: Path = None,
    subsample: int = 2,
) -> int:
    """
    Extract RGB + depth frames and camera intrinsics from a RealSense .bag file.
    Structure: bag_root / type / session_id / files...

    Disk writes are offloaded to a thread pool so the main decoding loop is
    never blocked waiting on I/O.

    Args:
        bag_path: Path to the .bag file.
        bag_root: Root directory (e.g. data/BAG).
        session_id: Unique session identifier.
        intrinsics_out_path: Optional path to save camera intrinsics.
        subsample: Extract every Nth frame. If None, reads SUBSAMPLE from
                   _paper_config.py automatically (default fallback = 3).

    Returns:
        Number of frames extracted.
    """
    if subsample is None:
        subsample = _load_paper_subsample()
    import pyrealsense2 as rs

    rgb_dir = bag_root / "frames" / session_id
    depth_raw_dir = bag_root / "depth_raw" / session_id
    depth_meters_dir = bag_root / "depth_meters" / session_id
    depth_color_dir = bag_root / "depth_color" / session_id

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_raw_dir.mkdir(parents=True, exist_ok=True)
    depth_meters_dir.mkdir(parents=True, exist_ok=True)
    depth_color_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    intrinsics_saved = False
    frame_idx = 0
    saved_idx = 0

    # Thread pool for async disk writes (4 workers ≈ 4 file types per frame)
    io_pool = ThreadPoolExecutor(max_workers=4)
    futures = []

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                break
            
            # Align first
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue
            
            # Subsampling check
            if frame_idx % subsample != 0:
                frame_idx += 1
                continue

            # Save intrinsics once
            if not intrinsics_saved:
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                intrinsics_dict = {
                    "fx": intr.fx, "fy": intr.fy,
                    "cx": intr.ppx, "cy": intr.ppy,
                    "width": intr.width, "height": intr.height,
                }
                
                # Save to custom path or default
                if intrinsics_out_path:
                    save_path = intrinsics_out_path
                else:
                    save_path = bag_root / "camera" / f"{session_id}.npy"
                
                save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                np.save(str(save_path), intrinsics_dict)
                intrinsics_saved = True

            # --- Copy frame data (numpy arrays are cheap to copy) ---
            color_image = np.asanyarray(color_frame.get_data())
            bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR).copy()

            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            depth_meters = depth_image.astype(np.float32) * depth_scale

            # --- Offload all 4 saves to the thread pool ---
            futures.append(io_pool.submit(
                _save_rgb,
                str(rgb_dir / f"frame_{saved_idx:06d}.png"), bgr,
            ))
            futures.append(io_pool.submit(
                _save_depth_raw,
                str(depth_raw_dir / f"frame_{saved_idx:06d}.png"), depth_image,
            ))
            futures.append(io_pool.submit(
                _save_depth_meters,
                str(depth_meters_dir / f"frame_{saved_idx:06d}.npy"), depth_meters,
            ))
            futures.append(io_pool.submit(
                _save_depth_colormap,
                str(depth_color_dir / f"frame_{saved_idx:06d}.png"), depth_meters.copy(),
            ))

            saved_idx += 1
            frame_idx += 1
    finally:
        pipeline.stop()
        # Wait for all pending writes to finish before returning
        for f in futures:
            f.result()
        io_pool.shutdown(wait=True)

    return saved_idx


def reconstruct_video_from_frames(frames_dir: Path, output_path: str, fps: float = 30.0):
    """
    Reconstruct an MP4 video from extracted RGB frames.
    Tries H.264 codec first (browser-compatible), falls back to mp4v.

    Args:
        frames_dir: Directory containing frame_XXXXX.png/jpg files.
        output_path: Output .mp4 file path.
        fps: Frames per second for the output video.
    """
    import cv2

    frames_dir = Path(frames_dir)
    # Support both PNG and JPG frames; sort by name for correct ordering
    frame_files = sorted(
        list(frames_dir.glob("frame_*.png")) + list(frames_dir.glob("frame_*.jpg"))
        + list(frames_dir.glob("frame_*.jpeg")),
        key=lambda p: p.name,
    )
    if not frame_files:
        return

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        return
    h, w = first_frame.shape[:2]

    # Try H.264 codecs (browser-compatible), fall back to mp4v
    writer = None
    for fourcc_str in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None

    if writer is None:
        return

    writer.write(first_frame)
    for f in frame_files[1:]:
        frame = cv2.imread(str(f))
        if frame is not None:
            writer.write(frame)
    writer.release()
