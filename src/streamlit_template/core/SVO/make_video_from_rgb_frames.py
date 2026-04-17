import subprocess
import os

# -------- CONFIG --------
RGB_DIR = r"C:\Users\VivekVijaykumar\Downloads\zed_python\output\rgb"
FPS = 15
OUT_VIDEO = r"C:\Users\VivekVijaykumar\Downloads\zed_python\output\rgb_final.mp4"
# ------------------------

cmd = [
    "ffmpeg",
    "-y",
    "-framerate", str(FPS),
    "-i", os.path.join(RGB_DIR, "frame_%06d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-profile:v", "high",
    "-level", "4.2",
    "-movflags", "+faststart",
    OUT_VIDEO
]

print("Creating playable MP4...")
subprocess.run(cmd, check=True)
print("✅ DONE")
print("Video saved to:", OUT_VIDEO)
