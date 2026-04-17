# import pyzed.sl as sl

import numpy as np
import cv2
import os
import shutil
from pathlib import Path
import imageio
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Async I/O helpers – offload disk writes to a thread pool so the main
# extraction loop is never blocked waiting for cv2.imwrite / np.save.
# ---------------------------------------------------------------------------

def _svo_save_rgb(path: str, bgr_np: np.ndarray):
    cv2.imwrite(path, bgr_np)

def _svo_save_depth_meters(path: str, depth_np: np.ndarray):
    np.save(path, depth_np)

def _svo_save_depth_color(path: str, depth_np: np.ndarray):
    depth_vis = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(path, depth_vis_color)

def extract_svo_frames(
    svo_file_path: str,
    output_root: Path,
    session_id: str,
    skip_frames: int = 1,
    fps: int = 15,
    intrinsics_out_path: Path = None
) -> dict:
    """
    Extracts RGB frames, depth frames (in meters), depth visualizations (in color), and camera intrinsics from an SVO/SVO2 file.

    Parameters:
    - svo_file_path: Path to the .svo or .svo2 file.
    - output_root: The root directory for SVO data (e.g., 'data/SVO').
    - session_id: A unique identifier for the extraction session.
    - skip_frames: Step size for frame extraction (1 = extract all frames).
    - fps: Frame rate for the extraction/playback.
    - intrinsics_out_path: Optional path to save camera intrinsics as a .npy file.

    Returns:
    - A dictionary containing lists of paths for extracted RGB frames, depth frames, and depth visualizations.
    """
    
    import pyzed.sl as sl

    # --- Setup Output Directories ---
    rgb_dir = output_root / "frames" / session_id
    depth_meters_dir = output_root / "depth_meters" / session_id
    depth_color_dir = output_root / "depth_color" / session_id
    camera_dir = output_root / "camera"

    # Ensure directories exist
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_meters_dir.mkdir(parents=True, exist_ok=True)
    depth_color_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    # --- ZED Camera Initialization ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_file_path))
    init_params.svo_real_time_mode = False  # Process as fast as possible
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Best quality depth
    init_params.coordinate_units = sl.UNIT.METER  # Depth in meters

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO file: {status}")
        return {"rgb": [], "depth_meters": [], "depth_color": []}

    # --- Intrinsics Extraction ---
    if intrinsics_out_path:
        cam_info = zed.get_camera_information()
        intr = cam_info.camera_configuration.calibration_parameters.left_cam
        
        # Save intrinsic dictionary [fx, fy, cx, cy, width, height] for compatibility
        intr_dict = {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
            "width": intr.image_size.width,
            "height": intr.image_size.height
        }
        np.save(intrinsics_out_path, intr_dict)
        print(f"Saved intrinsics to {intrinsics_out_path}")

    # --- Processing Loop ---
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    frame_id = 0
    saved_count = 0
    
    extracted_rgb_paths = []
    extracted_depth_meters_paths = []
    extracted_depth_color_paths = []

    print(f"Starting extraction from {svo_file_path}...")

    # Thread pool for async disk writes (3 workers ≈ 3 file types per frame)
    io_pool = ThreadPoolExecutor(max_workers=3)
    futures = []

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            break  # End of SVO file

        if frame_id % skip_frames == 0:
            # Retrieve Data
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            # --- RGB Processing ---
            # ZED provides BGRA (8-bit) by default for retrieve_image
            rgba_np = image.get_data()
            bgr_np = cv2.cvtColor(rgba_np, cv2.COLOR_BGRA2BGR).copy()
            
            # --- Depth Processing (Meters) ---
            depth_np = depth.get_data().copy()
            
            # Sanitization: Replace NaN/Inf with 0
            depth_np[np.isnan(depth_np)] = 0
            depth_np[np.isinf(depth_np)] = 0

            # Build paths
            curr_rgb_path = rgb_dir / f"frame_{saved_count:06d}.png"
            curr_depth_meters_path = depth_meters_dir / f"frame_{saved_count:06d}.npy"
            curr_depth_color_path = depth_color_dir / f"frame_{saved_count:06d}.png"

            # --- Offload all 3 saves to the thread pool ---
            futures.append(io_pool.submit(
                _svo_save_rgb, str(curr_rgb_path), bgr_np,
            ))
            futures.append(io_pool.submit(
                _svo_save_depth_meters, str(curr_depth_meters_path), depth_np,
            ))
            futures.append(io_pool.submit(
                _svo_save_depth_color, str(curr_depth_color_path), depth_np.copy(),
            ))

            extracted_rgb_paths.append(str(curr_rgb_path))
            extracted_depth_meters_paths.append(str(curr_depth_meters_path))
            extracted_depth_color_paths.append(str(curr_depth_color_path))

            saved_count += 1

        frame_id += 1

    zed.close()

    # Wait for all pending writes to finish before returning
    for f in futures:
        f.result()
    io_pool.shutdown(wait=True)

    print(f"Extraction complete. Saved {saved_count} frames.")
    
    return {
        "rgb": extracted_rgb_paths,
        "depth_meters": extracted_depth_meters_paths,
        "depth_color": extracted_depth_color_paths,
        "fps": fps  # Return intended FPS
    }


def reconstruct_svo_video(frames_dir: Path, output_path: str, fps: float = 15.0):
    """
    Reconstruct an MP4 video from extracted SVO frames.
    Uses imageio + ffmpeg backend for H.264 output (browser-compatible).

    Args:
        frames_dir: Directory containing frame_XXXXX.png files.
        output_path: Output .mp4 file path.
        fps: Frames per second for the output video.
    """
    
    frame_files = sorted(
        [f for ext in ["frame_*.png", "frame_*.jpg", "frame_*.jpeg"] for f in frames_dir.glob(ext)],
        key=lambda f: f.name,
    )
    if not frame_files:
        return

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Use imageio with ffmpeg backend -> produces H.264/MP4 that browsers can play
    # Note: 'macro_block_size' is often needed to be 16 or handled by ffmpeg automatically, 
    # but strictly setting purely typical params usually works:
    try:
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",  # Critical for browser compatibility
        )
        
        for f in frame_files:
            # Read image as RGB
            frame = imageio.imread(str(f))
            writer.append_data(frame)
        writer.close()
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"Failed to create video with imageio: {e}")
        # Fallback to subprocess ffmpeg if imageio fails (e.g. library missing, though unlikely in this env)
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.2",
                "-movflags", "+faststart",
                output_path
            ]
            import subprocess
            subprocess.run(cmd, check=True)
            print(f"Video saved to {output_path} (via ffmpeg subprocess)")
        except Exception as e2:
            print(f"Failed to create video with ffmpeg subprocess: {e2}")


def _annotate_hand_frames(rgb_dir: Path, hand_traj_3d_path: str, out_dir: Path):
    """
    Draw 3D hand trajectory projected onto 2D frames.
    Loads raw 3D points and projects them back using intrinsics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory
    try:
        traj_3d = np.load(hand_traj_3d_path)
    except:
        return []

    # Get intrinsics
    # Assuming standard location
    intr_path = rgb_dir.parent.parent / "camera" / f"{rgb_dir.parent.name}.npy"
    if not intr_path.exists():
         # Try global
         intr_path = rgb_dir.parent.parent / "camera/camera_intrinsics.npy"
    
    if not intr_path.exists():
        return []
        
    intr = np.load(str(intrinsics_path), allow_pickle=True).item()
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    
    # Collect all image files (png, jpg, jpeg)
    rgb_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        rgb_files.extend(list(rgb_dir.glob(ext)))
    rgb_files = sorted(rgb_files, key=lambda p: p.name)

    annotated_paths = []
    
    for i, fname in enumerate(rgb_files):
        if i >= len(traj_3d): break
        
        bgr = cv2.imread(str(fname))
        point_3d = traj_3d[i]
        
        # Project 3D -> 2D
        # X = (u - cx) * Z / fx  => u = (X * fx / Z) + cx
        # Y = (v - cy) * Z / fy  => v = (Y * fy / Z) + cy
        X, Y, Z = point_3d
        if Z > 0:
            u = int((X * fx / Z) + cx)
            v = int((Y * fy / Z) + cy)
            
            # Draw circle
            cv2.circle(bgr, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(bgr, f"Z: {Z:.2f}m", (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        out_path = out_dir / f"annotated_{fname.stem}.jpg"
        cv2.imwrite(str(out_path), bgr)
        annotated_paths.append(out_path)
        
    return annotated_paths


def _annotate_object_frames(rgb_dir: Path, out_dir: Path, model_path: str = None):
    """Run YOLO on each RGB frame and save annotated images."""
    from ultralytics import YOLO

    if model_path is None:
        model_path = "data/Common/ai_model/object/blueball.pt"

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    # Collect all image files (png, jpg, jpeg)
    rgb_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        rgb_files.extend(list(rgb_dir.glob(ext)))
    rgb_files = sorted(rgb_files, key=lambda p: p.name)

    annotated_paths = []
    total_detections = 0

    for fname in rgb_files:
        bgr = cv2.imread(str(fname))
        results = model(bgr, verbose=False)[0]

        if results.obb is not None and len(results.obb) > 0:
            total_detections += len(results.obb)
            # Draw OBB boxes
            for box in results.obb:
                xywhr = box.xywhr[0].cpu().numpy()
                cx, cy, w_box, h_box, rotation = xywhr
                rect = ((cx, cy), (w_box, h_box), np.degrees(rotation))
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
                cv2.drawContours(bgr, [box_points], 0, (0, 0, 255), 2)
                
                conf = float(box.conf[0])
                cv2.putText(bgr, f"{conf:.2f}", (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_path = out_dir / f"objects_{fname.stem}.jpg"
        cv2.imwrite(str(out_path), bgr)
        annotated_paths.append(out_path)

    return annotated_paths, total_detections
