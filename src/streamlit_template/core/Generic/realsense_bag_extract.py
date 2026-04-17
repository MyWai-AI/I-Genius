# realsense_bag_extract.py
import numpy as np
import cv2
import os
import shutil
from pathlib import Path
import csv


def extract_from_realsense_bag(bag_path: str, out_dir: str):
    """
    Extract synchronized RGB + Depth frames + intrinsics from a RealSense .bag file.
    Saves:
      out_dir/frames/frame_00000.jpg
      out_dir/frames/depth_00000.png
      out_dir/camera_intrinsics.npz
      out_dir/frames/frames_index.csv
    """

    import pyrealsense2 as rs

    bag_path = Path(bag_path)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Get intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()

    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_stream.get_intrinsics()

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    np.savez(
        out_dir / "camera_intrinsics.npz",
        fx=color_intr.fx,
        fy=color_intr.fy,
        cx=color_intr.ppx,
        cy=color_intr.ppy,
        depth_scale=depth_scale
    )

    # CSV index
    index_csv = frames_dir / "frames_index.csv"
    csv_file = index_csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_idx", "time_sec", "filename", "depth_file"])

    frame_idx = 0

    while True:
        try:
            fs = pipeline.wait_for_frames()
        except Exception:
            break

        color_frame = fs.get_color_frame()
        depth_frame = fs.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert frames
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())  # uint16

        rgb_name = f"frame_{frame_idx:05d}.jpg"
        depth_name = f"depth_{frame_idx:05d}.png"

        cv2.imwrite(str(frames_dir / rgb_name), color_img)
        cv2.imwrite(str(frames_dir / depth_name), depth_img)

        writer.writerow([
            frame_idx,
            frame_idx * 0.033,   # rough timestamp (30 FPS)
            rgb_name,
            depth_name
        ])

        frame_idx += 1

    csv_file.close()
    pipeline.stop()

    return {
        "msg": "Extraction complete",
        "frames": frame_idx,
        "frames_dir": str(frames_dir),
        "intrinsics": str(out_dir / "camera_intrinsics.npz")
    }
