# Purpose: Class-agnostic object detection using YOLOv8 nano.
# Returns ONLY bounding boxes and confidence scores — no class labels.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

_DEFAULT_MODEL = "yolov8n.pt"


@dataclass
class Detection:
    """A single class-agnostic detection."""

    bbox_xyxy: np.ndarray  # (4,) — [x1, y1, x2, y2]
    bbox_xywh: np.ndarray  # (4,) — [cx, cy, w, h]
    confidence: float
    label: Optional[str] = None  # class name if available, None otherwise


class ObjectDetector:
    """Class-agnostic object detector backed by YOLOv8 nano.

    Parameters
    ----------
    model_path : str | Path
        Path or name of the YOLO model weights (default ``yolov8n.pt``).
    confidence_threshold : float
        Minimum confidence to keep a detection (default 0.25).
    device : str | None
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-select).
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.device = device

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Returns a list of :class:`Detection` instances (class-agnostic).
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            device=self.device,
        )[0]

        detections: List[Detection] = []
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()   # (N, 4)
        xywh = boxes.xywh.detach().cpu().numpy()   # (N, 4)
        confs = boxes.conf.detach().cpu().numpy()   # (N,)

        for i in range(len(confs)):
            detections.append(
                Detection(
                    bbox_xyxy=xyxy[i].astype(np.float32),
                    bbox_xywh=xywh[i].astype(np.float32),
                    confidence=float(confs[i]),
                )
            )
        return detections

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def detect_video(
        self,
        video_path: str | Path,
        frame_skip: int = 1,
    ) -> List[List[Detection]]:
        """Run detection on every *frame_skip*-th frame of a video file.

        Parameters
        ----------
        video_path : str | Path
            Path to the input video.
        frame_skip : int
            Process one out of every ``frame_skip`` frames (1 = all frames).

        Returns
        -------
        list[list[Detection]]
            Outer list has one entry per *processed* frame.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        all_detections: List[List[Detection]] = []
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % max(frame_skip, 1) == 0:
                    all_detections.append(self.detect_frame(frame))
                frame_idx += 1
        finally:
            cap.release()
        return all_detections

    def detect_frames_dir(
        self,
        frames_dir: str | Path,
        frame_skip: int = 1,
    ) -> List[List[Detection]]:
        """Run detection on pre-extracted image frames in a directory.

        Parameters
        ----------
        frames_dir : str | Path
            Directory containing frame images (png/jpg/jpeg).
        frame_skip : int
            Process one out of every ``frame_skip`` frames (1 = all).

        Returns
        -------
        list[list[Detection]]
            Outer list has one entry per *processed* frame.
        """
        d = Path(frames_dir)
        files = sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

        all_detections: List[List[Detection]] = []
        for idx, fpath in enumerate(files):
            if idx % max(frame_skip, 1) != 0:
                continue
            frame = cv2.imread(str(fpath))
            if frame is None:
                all_detections.append([])
                continue
            all_detections.append(self.detect_frame(frame))
        return all_detections

    # ------------------------------------------------------------------
    # Convenience: numpy export (for tracker consumption)
    # ------------------------------------------------------------------

    @staticmethod
    def to_numpy(detections: List[Detection]):
        """Convert a list of detections to numpy arrays.

        Returns
        -------
        boxes_xyxy : np.ndarray
            Shape ``(N, 4)`` with ``[x1, y1, x2, y2]``.
        boxes_xywh : np.ndarray
            Shape ``(N, 4)`` with ``[cx, cy, w, h]``.
        confidences : np.ndarray
            Shape ``(N,)`` confidence scores.
        """
        if not detections:
            empty4 = np.empty((0, 4), dtype=np.float32)
            return empty4, empty4.copy(), np.empty((0,), dtype=np.float32)
        xyxy = np.stack([d.bbox_xyxy for d in detections])
        xywh = np.stack([d.bbox_xywh for d in detections])
        confs = np.array([d.confidence for d in detections], dtype=np.float32)
        return xyxy, xywh, confs

    @staticmethod
    def to_dicts(detections: List[Detection]) -> List[dict]:
        """Convert detections to list-of-dicts format.

        Each dict has keys ``"bbox"`` (xyxy list) and ``"confidence"`` (float).
        """
        return [
            {
                "bbox": d.bbox_xyxy.tolist(),
                "confidence": d.confidence,
            }
            for d in detections
        ]
