"""
SVO Helpers — Reconstruct RGB and RGBD videos from SVO flat frame files.

Handles both standard naming (frame_000000.jpg) and MYWAI timestamp-prefixed
naming (639076237154104597_frame_000000_OpenArm001.jpg).
"""
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _sort_frame_files(files: list[Path]) -> list[Path]:
    """
    Sort frame files by their embedded frame number (e.g. frame_000147).
    Handles both 'frame_000147.jpg' and '639076238549565273_frame_000147_OpenArm001.jpg'.
    """
    frame_num_pattern = re.compile(r'frame_(\d+)')

    def _extract_num(p: Path) -> int:
        m = frame_num_pattern.search(p.stem)
        return int(m.group(1)) if m else 0

    return sorted(files, key=_extract_num)


def _convert_to_h264_if_needed(output_path: str):
    """Fallback utility to convert video to H.264 using FFmpeg if OpenCV's encoder fails."""
    import subprocess
    import shutil
    try:
        if shutil.which("ffmpeg"):
            temp_output = output_path.replace('.mp4', '_temp.mp4')
            subprocess.run([
                'ffmpeg', '-y', '-i', output_path, 
                '-vcodec', 'libx264', '-crf', '23', temp_output
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.move(temp_output, output_path)
            logger.info(f"Successfully converted {output_path} to H.264 via FFmpeg CLI.")
    except Exception as e:
        logger.warning(f"FFmpeg fallback conversion failed for {output_path}: {e}")


def _collect_rgb_frames(frames_dir: Path) -> list[Path]:
    """Collect all RGB frame image files from a directory."""
    # Support both standard and timestamp-prefixed names
    files = (
        list(frames_dir.glob("*frame_*.png"))
        + list(frames_dir.glob("*frame_*.jpg"))
        + list(frames_dir.glob("*frame_*.jpeg"))
    )
    return _sort_frame_files(files)


def _collect_depth_files(depth_dir: Path) -> list[Path]:
    """Collect all depth .npy files from a directory."""
    files = list(depth_dir.glob("*frame_*.npy"))
    return _sort_frame_files(files)


def reconstruct_rgb_video(
    frames_dir: Path,
    output_path: str,
    fps: float = 15.0,
) -> Optional[str]:
    """
    Reconstruct an MP4 video from extracted RGB frames.

    Args:
        frames_dir: Directory containing frame images (jpg/png).
        output_path: Output .mp4 file path.
        fps: Frames per second for the output video.

    Returns:
        The output_path on success, None on failure.
    """
    import cv2

    frames_dir = Path(frames_dir)
    frame_files = _collect_rgb_frames(frames_dir)

    if not frame_files:
        logger.warning(f"No RGB frames found in {frames_dir}")
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        logger.error(f"Could not read first frame: {frame_files[0]}")
        return None
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
        logger.error("Could not open any video codec")
        return None

    writer.write(first_frame)
    for f in frame_files[1:]:
        frame = cv2.imread(str(f))
        if frame is not None:
            writer.write(frame)
    writer.release()

    _convert_to_h264_if_needed(output_path)

    logger.info(f"RGB video written: {output_path} ({len(frame_files)} frames @ {fps} fps)")
    return output_path


def reconstruct_rgbd_video(
    depth_dir: Path,
    output_path: str,
    fps: float = 15.0,
    colormap: int = None,
) -> Optional[str]:
    """
    Reconstruct a colorized depth video from .npy depth frames.

    Each .npy file is loaded, normalized to 0-255, and colorized with a
    colormap for visual inspection.

    Args:
        depth_dir: Directory containing depth .npy files.
        output_path: Output .mp4 file path.
        fps: Frames per second for the output video.
        colormap: OpenCV colormap ID. Defaults to cv2.COLORMAP_INFERNO.

    Returns:
        The output_path on success, None on failure.
    """
    import cv2
    import numpy as np

    if colormap is None:
        colormap = cv2.COLORMAP_INFERNO

    depth_dir = Path(depth_dir)
    depth_files = _collect_depth_files(depth_dir)

    if not depth_files:
        logger.warning(f"No depth .npy files found in {depth_dir}")
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Read first frame to get dimensions
    first_depth = np.load(str(depth_files[0]))
    if first_depth.ndim != 2:
        logger.error(f"Unexpected depth shape: {first_depth.shape}")
        return None
    h, w = first_depth.shape

    # Try H.264 codecs, fall back to mp4v
    writer = None
    for fourcc_str in ["avc1", "H264", "mp4v"]:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None

    if writer is None:
        logger.error("Could not open any video codec for depth")
        return None

    def _depth_to_color(depth_arr):
        """Normalize depth to 0-255 and apply colormap."""
        valid = depth_arr[np.isfinite(depth_arr)]
        if len(valid) == 0:
            norm = np.zeros_like(depth_arr, dtype=np.uint8)
        else:
            d_min, d_max = valid.min(), valid.max()
            if d_max - d_min < 1e-6:
                norm = np.zeros_like(depth_arr, dtype=np.uint8)
            else:
                clipped = np.clip(depth_arr, d_min, d_max)
                norm = ((clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, colormap)

    writer.write(_depth_to_color(first_depth))
    for f in depth_files[1:]:
        try:
            d = np.load(str(f))
            writer.write(_depth_to_color(d))
        except Exception as e:
            logger.warning(f"Skipping depth frame {f.name}: {e}")
    writer.release()

    _convert_to_h264_if_needed(output_path)

    logger.info(f"RGBD video written: {output_path} ({len(depth_files)} frames @ {fps} fps)")
    return output_path
