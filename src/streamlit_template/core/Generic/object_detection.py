# src/streamlit_template/core/object_detection.py
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import csv
import cv2
import numpy as np


def _to_class_id_set(
    target_classes: Optional[List[Union[str, int]]],
    names: Dict[int, str]
) -> Optional[set]:
    """Convert names/IDs to numeric class IDs."""
    if not target_classes:
        return None

    name_to_id = {v.lower().replace(" ", "_"): k for k, v in names.items()}
    out = set()

    for it in target_classes:
        s = str(it).strip()
        if s == "":
            continue
        if s.isdigit():
            out.add(int(s))
        else:
            cid = name_to_id.get(s.lower())
            if cid is not None:
                out.add(cid)

    return out if out else None


def detect_objects_on_frames(
    frames_dir: str = "data/Generic/frames",
    index_csv: Optional[str] = None,
    out_dir: str = "data/Generic/objects",
    model_path: str = "data/Common/ai_model/object/best.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    target_classes: Union[str, List[str], List[int]] = None,
    sample_every_n: int = 1,
    limit: Optional[int] = None
) -> Dict[str, Any]:

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("YOLO not installed. Run: pip install ultralytics") from e

    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if index_csv is None:
        index_csv = frames_dir / "frames_index.csv"
    index_csv = Path(index_csv)

    # Load frames index
    rows = []
    with index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Load YOLO model
    model = YOLO(model_path)
    class_names = model.names

    # Normalize target_classes
    if target_classes in (None, "None", [], ["None"]):
        target_classes = None
    elif isinstance(target_classes, (str, int)):
        target_classes = [target_classes]

    allow_ids = _to_class_id_set(target_classes, class_names)

    # Prepare CSV output
    det_csv = out_dir / "objects_detections.csv"
    with det_csv.open("w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow([
            "frame_idx", "time_sec", "image_file",
            "det_id", "class_id", "class_name", "confidence",
            "xmin", "ymin", "xmax", "ymax"
        ])

        processed = 0
        total = 0

        for k, row in enumerate(rows):
            if k % sample_every_n != 0:
                continue
            if limit is not None and processed >= limit:
                break

            img_path = frames_dir / row["filename"]
            frame_idx = int(float(row["frame_idx"]))
            time_sec = float(row["time_sec"])

            if not img_path.exists():
                continue

            # YOLO inference
            results = model.predict(str(img_path), conf=conf, iou=iou, verbose=False)
            res = results[0]

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            det_id = 0
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    cid = int(cls[i])

                    if allow_ids is not None and cid not in allow_ids:
                        continue

                    x1, y1, x2, y2 = xyxy[i].tolist()
                    confv = float(confs[i])
                    cname = class_names.get(cid, str(cid))

                    # Draw box
                    cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        bgr,
                        f"{cname} {confv:.2f}",
                        (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1
                    )

                    # Write CSV
                    w.writerow([
                        frame_idx, f"{time_sec:.6f}", row["filename"],
                        det_id, cid, cname, f"{confv:.4f}",
                        int(x1), int(y1), int(x2), int(y2)
                    ])

                    total += 1
                    det_id += 1

            out_name = out_dir / f"objects_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_name), bgr)
            processed += 1

    return {
        "processed_frames": processed,
        "total_detections": total,
        "annotated_dir": str(out_dir.resolve()),
        "detections_csv": str(det_csv.resolve()),
        "model": model_path,
    }