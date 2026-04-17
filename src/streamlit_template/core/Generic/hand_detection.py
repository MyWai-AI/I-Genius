# vilma_core/hand_detection.py
from pathlib import Path
from typing import Optional, Dict, Any, List
import csv
import cv2
import numpy as np
import urllib.request

def _download_hand_model(cache_dir: Path) -> Path:
    """Download MediaPipe hand landmarker model if not already cached."""
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    model_path = cache_dir / "hand_landmarker.task"
    
    if not model_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully.")
    
    return model_path

def detect_hands_on_frames(
    frames_dir: str = "data/Generic/frames",
    index_csv: Optional[str] = None,             # defaults to frames_dir/frames_index.csv
    out_dir: str = "data/Generic/hands",
    max_hands: int = 2,
    min_det_conf: float = 0.5,
    sample_every_n: int = 1,                      # process every Nth frame
    limit: Optional[int] = None                   # cap total frames processed
) -> Dict[str, Any]:
    """
    Detect hands in frames using MediaPipe (updated for v0.10+).
    """
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if index_csv is None:
        index_csv = str(frames_dir / "frames_index.csv")

    index_path = Path(index_csv)
    if not index_path.exists():
        raise FileNotFoundError(f"frames_index.csv not found at {index_path}. Run extract first.")

    # Load frame index
    rows = []
    with index_path.open("r", newline="", encoding="utf-8") as fcsv:
        reader = csv.DictReader(fcsv)
        for row in reader:
            rows.append(row)

    # Download and cache the model
    cache_dir = Path.home() / ".mediapipe" / "models"
    model_path = _download_hand_model(cache_dir)

    # Create hand landmarker with new API
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=max_hands,
        min_hand_detection_confidence=min_det_conf,
        min_hand_presence_confidence=min_det_conf,
        min_tracking_confidence=min_det_conf
    )
    
    detector = vision.HandLandmarker.create_from_options(options)

    lm_csv = out_dir / "hands_landmarks.csv"
    with lm_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        header = ["frame_idx", "time_sec", "image_file", "hand_id"]
        for i in range(21):
            header += [f"lm{i}_x", f"lm{i}_y"]
        writer.writerow(header)

        processed = 0
        saved_images: List[str] = []

        for k, row in enumerate(rows):
            if k % max(sample_every_n, 1) != 0:
                continue
            if limit is not None and processed >= limit:
                break

            img_name = row["filename"]
            frame_idx = int(float(row["frame_idx"]))
            time_sec = float(row["time_sec"])
            img_path = frames_dir / img_name
            # Debug path
            # print(f"DEBUG DETECT: frames_dir={frames_dir} img_name={img_name} path={img_path}")
            if not img_path.exists():
                print(f"MISSING: {img_path}")
                continue

            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Detect hands
            detection_result = detector.detect(mp_image)

            annotated = bgr.copy()
            if detection_result.hand_landmarks:
                for hand_id, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    row_out = [frame_idx, f"{time_sec:.6f}", img_name, hand_id]
                    for lm in hand_landmarks:
                        row_out += [f"{lm.x:.6f}", f"{lm.y:.6f}"]  # normalized [0..1]
                    writer.writerow(row_out)

                    # Draw landmarks manually since drawing_utils is not available
                    h, w = annotated.shape[:2]
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

            out_name = out_dir / f"hands_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_name), annotated)
            saved_images.append(str(out_name))
            processed += 1

    detector.close()

    return {
        "processed_frames": processed,
        "annotated_dir": str(out_dir.resolve()),
        "landmarks_csv": str(lm_csv.resolve()),
        "sample_every_n": sample_every_n,
        "limit": limit,
    }
