# Purpose: Extract aligned RGB/depth frames from RealSense BAG and save camera intrinsics.
# Notes: Structured stage with stable defaults for repeatable runs.

import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import (
    BAG_FILE,
    DEPTH_M_PATH,
    DEPTH_RAW_PATH,
    INTRINSICS_PATH,
    RGB_PATH,
    SUBSAMPLE,
    ensure_output_dirs,
)


def main() -> None:
    """Main entry point for this stage."""
    ensure_output_dirs()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(BAG_FILE), repeat_playback=False)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsics = {
        'fx': intr.fx,
        'fy': intr.fy,
        'cx': intr.ppx,
        'cy': intr.ppy,
        'width': intr.width,
        'height': intr.height,
    }
    np.save(INTRINSICS_PATH, intrinsics)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f'Saved intrinsics to: {INTRINSICS_PATH}')
    print(f'Depth scale: {depth_scale}')

    frame_idx = 0
    saved_idx = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                break

            if frame_idx % SUBSAMPLE != 0:
                frame_idx += 1
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            cv2.imwrite(str(RGB_PATH / f'frame_{saved_idx:06d}.png'), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(DEPTH_RAW_PATH / f'frame_{saved_idx:06d}.png'), depth_raw)
            np.save(DEPTH_M_PATH / f'frame_{saved_idx:06d}.npy', depth_m)

            frame_idx += 1
            saved_idx += 1

    except RuntimeError:
        pass
    finally:
        pipeline.stop()

    print(f'Frame extraction complete. Saved {saved_idx} aligned RGB+Depth frames.')


if __name__ == '__main__':
    main()
