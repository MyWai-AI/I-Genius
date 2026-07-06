"""Check integrity of downloaded SVO files."""
from pathlib import Path
import os

# Check RGB frames
rgb_dir = Path("data/SVO/frames/OpenArm001")
rgb_files = sorted(rgb_dir.glob("*.jpg"))
print(f"Total RGB frames: {len(rgb_files)}")

jpeg_ok = 0
jpeg_bad = 0
for f in rgb_files:
    with open(f, "rb") as fh:
        header = fh.read(2)
    if header == b"\xff\xd8":
        jpeg_ok += 1
    else:
        jpeg_bad += 1
        print(f"  BAD JPEG: {f.name} (size={os.path.getsize(f)})")

print(f"Valid JPEGs: {jpeg_ok}, Bad: {jpeg_bad}")

# Check depth files
depth_dir = Path("data/SVO/depth_meters/OpenArm001")
depth_files = sorted(depth_dir.glob("*.npy"))
print(f"\nTotal depth .npy files: {len(depth_files)}")

npy_ok = 0
npy_bad = 0
for f in depth_files:
    with open(f, "rb") as fh:
        header = fh.read(6)
    if header == b"\x93NUMPY":
        npy_ok += 1
    else:
        npy_bad += 1
        print(f"  BAD NPY: {f.name} (size={os.path.getsize(f)}, header={header[:4]})")

print(f"Valid NPY: {npy_ok}, Bad: {npy_bad}")

# Check video
vid = Path("data/SVO/videos/OpenArm001/OpenArm001.mp4")
print(f"\nVideo exists: {vid.exists()}")

# Check camera
cam = Path("data/SVO/camera")
if cam.exists():
    print(f"\nCamera dir:")
    for f in cam.iterdir():
        sz = os.path.getsize(f) if f.is_file() else "dir"
        print(f"  {f.name}: {sz}")
