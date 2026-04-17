# vilma_core/extract_frames.py

from pathlib import Path
import cv2
import csv
from typing import List, Optional, Dict, Any


def extract_frames_from_video(
    video_path: str,
    output_dir: str = "data/Generic/frames",
    every_n: int = 1,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    resize_width: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract frames from a video and save them as JPEG images + an index CSV.

    Args:
        video_path: Path to input video (.mp4/.mov/.avi).
        output_dir: Folder to save frames.
        every_n: Save one frame every N frames (downsample).
        start_sec: Skip the first N seconds.
        duration_sec: If set, only extract this many seconds.
        resize_width: If set, resize (keep aspect) to this width.

    Returns:
        dict with: count, fps, frame_paths, index_csv, output_dir
    """
    video_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # OpenCV VideoCapture handles mp4/mov/avi if ffmpeg available (opencv-python wheels include it)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Seek to start time if possible
    if start_sec and fps > 0:
        start_idx = int(start_sec * fps)
        start_idx = min(max(start_idx, 0), max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    else:
        start_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Compute last frame if duration is specified
    last_idx = None
    if duration_sec is not None and fps > 0:
        last_idx = min(
            start_idx + int(duration_sec * fps),
            total_frames - 1 if total_frames > 0 else int(1e12)
        )

    index_csv_path = out_dir / "frames_index.csv"
    frame_paths: List[str] = []
    saved_count = 0
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    with index_csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame_idx", "time_sec", "filename"])

        try:
            while True:
                if last_idx is not None and frame_idx > last_idx:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_idx) % max(every_n, 1) == 0:
                    # Optional resize (keep aspect)
                    if resize_width and resize_width > 0:
                        h, w = frame.shape[:2]
                        if w > 0:
                            scale = resize_width / float(w)
                            frame = cv2.resize(
                                frame,
                                (resize_width, int(h * scale)),
                                interpolation=cv2.INTER_AREA
                            )

                    fname = out_dir / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(fname), frame)

                    t_sec = (frame_idx / fps) if fps > 0 else 0.0
                    writer.writerow([frame_idx, f"{t_sec:.6f}", fname.name])

                    frame_paths.append(str(fname))
                    saved_count += 1

                frame_idx += 1
        finally:
            cap.release()

    return {
        "count": saved_count,
        "fps": float(fps),
        "frame_paths": frame_paths,
        "index_csv": str(index_csv_path),
        "output_dir": str(out_dir),
    }


def _find_latest_upload(uploads_dir: str = "data/Generic/uploads") -> Optional[Path]:
    """Pick the most recently modified file in uploads_dir (or None)."""
    ud = Path(uploads_dir)
    if not ud.exists():
        return None
    files = [p for p in ud.iterdir() if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def run_default(
    uploads_dir: str = "data/Generic/uploads",
    output_dir: str = "data/Generic/frames",
    every_n: int = 1,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    resize_width: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience runner: finds latest upload in uploads_dir and extracts frames to output_dir.
    """
    latest = _find_latest_upload(uploads_dir)
    if latest is None:
        raise FileNotFoundError(f"No files found in {uploads_dir}. Please upload a video first.")
    return extract_frames_from_video(
        video_path=str(latest),
        output_dir=output_dir,
        every_n=every_n,
        start_sec=start_sec,
        duration_sec=duration_sec,
        resize_width=resize_width,
    )


if __name__ == "__main__":
    # CLI usage example (Windows):
    #   python -m vilma_core.extract_frames --video data\uploads\demo.mp4 --out data\frames --every_n 2 --resize_width 960
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames into data/Generic/frames with an index CSV.")
    parser.add_argument("--video", type=str, default="", help="Path to video. If omitted, use latest in data/Generic/uploads")
    parser.add_argument("--out", type=str, default="data/Generic/frames", help="Output directory")
    parser.add_argument("--every_n", type=int, default=1, help="Save one frame every N frames")
    parser.add_argument("--start_sec", type=float, default=0.0, help="Skip first N seconds")
    parser.add_argument("--duration_sec", type=float, default=None, help="Extract only this many seconds")
    parser.add_argument("--resize_width", type=int, default=None, help="Resize width (keep aspect)")

    args = parser.parse_args()

    if args.video:
        res = extract_frames_from_video(
            video_path=args.video,
            output_dir=args.out,
            every_n=args.every_n,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            resize_width=args.resize_width,
        )
    else:
        res = run_default(
            uploads_dir="data/Generic/uploads",
            output_dir=args.out,
            every_n=args.every_n,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            resize_width=args.resize_width,
        )

    print(
        f" Saved {res['count']} frames to {res['output_dir']} "
        f"(fps={res['fps']:.3f}). Index: {res['index_csv']}"
    )
