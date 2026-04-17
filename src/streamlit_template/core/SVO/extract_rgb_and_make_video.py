import pyzed.sl as sl
import numpy as np
import cv2
import os
import subprocess

# ================= CONFIG =================
SVO_PATH = r"F:\zed_python\input\One Shot Demo.svo2"
OUT_DIR = r"F:\zed_python\output"
FRAME_STEP = 1
FPS = 15
# =========================================

# Output dirs
rgb_dir = os.path.join(OUT_DIR, "rgb")
os.makedirs(rgb_dir, exist_ok=True)

final_video = os.path.join(OUT_DIR, "rgb_final.mp4")

# -------- ZED SETUP --------
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

frame_id = 0
saved = 0

print("Extracting RGB frames...")

# -------- FRAME EXTRACTION --------
while True:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        break

    if frame_id % FRAME_STEP == 0:
        zed.retrieve_image(image, sl.VIEW.LEFT)

        rgba = image.get_data()
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        cv2.imwrite(
            os.path.join(rgb_dir, f"frame_{saved:06d}.png"),
            rgb
        )

        saved += 1

    frame_id += 1

zed.close()

print(f"Saved {saved} RGB frames")

# -------- VIDEO CREATION (FFMPEG) --------
print("Creating upload-safe MP4...")

ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-framerate", str(FPS),
    "-i", os.path.join(rgb_dir, "frame_%06d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-profile:v", "high",
    "-level", "4.2",
    "-movflags", "+faststart",
    final_video
]

subprocess.run(ffmpeg_cmd, check=True)

print("✅ DONE")
print("Final video:", final_video)
