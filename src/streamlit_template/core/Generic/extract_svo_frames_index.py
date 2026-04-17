# extract_svo_frames_index.py
import csv
from pathlib import Path

def build_svo_frames_index(
    rgb_dir,
    depth_dir,
    out_csv,
    fps=30.0
):
    rgb_dir = Path(rgb_dir)
    depth_dir = Path(depth_dir)
    out_csv = Path(out_csv)

    rgb_files = sorted(rgb_dir.glob("*"))
    assert len(rgb_files) > 0, "No RGB images found"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx",
            "time_sec",
            "filename",
            "depth_file"
        ])

        for i, rgb in enumerate(rgb_files):
            depth_file = depth_dir / (rgb.stem + ".npy")

            w.writerow([
                i,
                i / fps,
                rgb.name,
                depth_file.name if depth_file.exists() else ""
            ])

    print(f"[OK] frames_index.csv written to {out_csv}")


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[3] / "data" / "SVO"

    build_svo_frames_index(
        rgb_dir=BASE / "rgb",
        depth_dir=BASE / "depth",
        out_csv=BASE / "frames_index.csv",
        fps=30.0
    )
