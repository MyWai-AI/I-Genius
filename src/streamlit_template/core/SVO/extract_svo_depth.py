import pyzed.sl as sl
import numpy as np
import cv2
import os

# ================= CONFIG =================
SVO_PATH = r"F:\zed_python\input\One Shot Demo.svo2"
OUT_DIR = r"F:\zed_python\output"
FRAME_STEP = 1        # set to >1 to skip frames
FPS = 15              # match SVO FPS
# =========================================

# Output folders
rgb_dir = os.path.join(OUT_DIR, "rgb")
depth_dir = os.path.join(OUT_DIR, "depth")
depth_vis_dir = os.path.join(OUT_DIR, "depth_viz")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(depth_vis_dir, exist_ok=True)

# Video paths
rgb_video_path = os.path.join(OUT_DIR, "rgb_video.mp4")
depth_video_path = os.path.join(OUT_DIR, "depth_video.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
rgb_writer = None
depth_writer = None

# ZED setup
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.set_from_svo_file(SVO_PATH)
init_params.svo_real_time_mode = False
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open SVO:", status)
    exit(1)

runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()

frame_id = 0
saved = 0

print("Starting extraction...")

while True:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        break

    if frame_id % FRAME_STEP == 0:

        # Retrieve data
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        # ---------- RGB ----------
        rgb = image.get_data()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)

        # ---------- DEPTH ----------
        depth_np = depth.get_data().copy()
        depth_np[np.isnan(depth_np)] = 0
        depth_np[np.isinf(depth_np)] = 0

        # ---------- INIT VIDEOS ----------
        if rgb_writer is None:
            h, w, _ = rgb.shape
            rgb_writer = cv2.VideoWriter(rgb_video_path, fourcc, FPS, (w, h))
            depth_writer = cv2.VideoWriter(depth_video_path, fourcc, FPS, (w, h))

        # ---------- SAVE FILES ----------
        cv2.imwrite(f"{rgb_dir}/frame_{frame_id:06d}.png", rgb)
        np.save(f"{depth_dir}/frame_{frame_id:06d}.npy", depth_np)

        # Depth visualization
        depth_vis = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(f"{depth_vis_dir}/frame_{frame_id:06d}.png", depth_vis)

        # ---------- WRITE VIDEOS ----------
        rgb_writer.write(rgb)
        depth_writer.write(depth_vis)

        saved += 1

    frame_id += 1

# Cleanup
zed.close()
if rgb_writer:
    rgb_writer.release()
if depth_writer:
    depth_writer.release()

print(f"Done. Saved {saved} frames.")
print("RGB video:", rgb_video_path)
print("Depth video:", depth_video_path)
