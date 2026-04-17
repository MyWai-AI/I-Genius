# Purpose: Monocular depth estimation for raw video files (MP4/AVI/MOV).
#
# Uses Depth Anything V2 to produce depth maps from RGB frames. Output is
# plug-compatible with the SVO/BAG depth extractors:
#   - float32 .npy files in meters
#   - Filename convention: frame_XXXXXX.npy
#   - Colorized .png visualizations with JET colormap
#
# CLI usage:
#   python estimate_monocular_depth.py --input_dir data/Generic/frames/abc123 \
#                                      --output_dir data/Generic/depth_meters/abc123

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _load_depth_anything_v2(
    model_size: str = "Small",
    device: torch.device | None = None,
) -> tuple:
    """Load DepthAnythingV2 model.

    Falls back to MiDaS DPT-Large if DepthAnythingV2 is unavailable.

    Returns
    -------
    model : torch.nn.Module
        The loaded depth model.
    transform : callable
        Preprocessing transform for input images.
    backend : str
        ``"depth_anything_v2"`` or ``"midas"``.
    device : torch.device
        The device used.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Try Depth Anything V2 via transformers (HuggingFace) ---
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model = model.to(device).eval()
        return model, processor, "depth_anything_v2", device
    except Exception:
        pass

    # --- Fallback: MiDaS via torch.hub ---
    try:
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        model = model.to(device).eval()
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        transform = midas_transforms.dpt_transform
        return model, transform, "midas", device
    except Exception as e:
        raise RuntimeError(
            "Neither DepthAnythingV2 (transformers) nor MiDaS could be loaded. "
            "Install with: pip install transformers torch timm\n"
            f"Error: {e}"
        )


def _predict_depth(
    model,
    transform,
    backend: str,
    device: torch.device,
    rgb_bgr: np.ndarray,
) -> np.ndarray:
    """Run depth prediction on a single BGR frame.

    Returns
    -------
    depth : np.ndarray
        ``(H, W)`` float32 depth map. For monocular models the values are
        *relative* (not metric). A post-processing step can optionally
        rescale them.
    """
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    if backend == "depth_anything_v2":
        from PIL import Image

        pil_img = Image.fromarray(rgb)
        inputs = transform(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = depth.cpu().numpy().astype(np.float32)

    elif backend == "midas":
        input_batch = transform(rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        # MiDaS returns inverse depth — invert to get relative depth
        depth = prediction.cpu().numpy().astype(np.float32)
        depth = np.clip(depth, 1e-6, None)
        depth = 1.0 / depth
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return depth


def _relative_to_metric(
    depth: np.ndarray,
    scale: float = 1.0,
    shift: float = 0.0,
) -> np.ndarray:
    """Convert relative depth to pseudo-metric values.

    Without a known scale the best we can do is normalize and apply a
    user-provided scale factor. ``scale`` represents the approximate
    maximum scene depth in meters.
    """
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return depth
    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max - d_min < 1e-9:
        return np.full_like(depth, scale / 2.0)
    normalized = (depth - d_min) / (d_max - d_min)
    return (normalized * scale + shift).astype(np.float32)


def _save_depth_color(path: str, depth_np: np.ndarray) -> None:
    """Save a colorized depth visualization (JET colormap), matching SVO/BAG format."""
    valid = depth_np[np.isfinite(depth_np) & (depth_np > 0)]
    if valid.size == 0:
        cv2.imwrite(path, np.zeros(depth_np.shape[:2] + (3,), dtype=np.uint8))
        return
    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max - d_min < 1e-6:
        d_max = d_min + 1e-6
    norm = np.clip((depth_np - d_min) / (d_max - d_min), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite(path, color)


def estimate_depth_for_frames(
    input_dir: str | Path,
    output_dir: str | Path,
    depth_color_dir: Optional[str | Path] = None,
    model_size: str = "Small",
    max_depth_m: float = 5.0,
    device: Optional[str] = None,
) -> dict:
    """Run monocular depth estimation on a directory of RGB frame images.

    Parameters
    ----------
    input_dir : Path
        Directory containing RGB frames (png/jpg).
    output_dir : Path
        Directory to save depth ``.npy`` files (float32, meters).
    depth_color_dir : Path | None
        Optional directory for JET colorized depth PNGs.
    model_size : str
        DepthAnythingV2 model variant: ``"Small"``, ``"Base"``, or ``"Large"``.
    max_depth_m : float
        Assumed maximum scene depth in meters for normalization (default 5.0).
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto).

    Returns
    -------
    dict
        ``{"count": int, "output_dir": str, "backend": str}``
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if depth_color_dir:
        depth_color_dir = Path(depth_color_dir)
        depth_color_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device) if device else None
    model, transform, backend, torch_device = _load_depth_anything_v2(
        model_size=model_size, device=torch_device
    )
    print(f"Depth backend: {backend} on {torch_device}")

    frames = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found in {input_dir}")

    count = 0
    for idx, fpath in enumerate(tqdm(frames, desc="Depth estimation")):
        bgr = cv2.imread(str(fpath))
        if bgr is None:
            continue

        raw_depth = _predict_depth(model, transform, backend, torch_device, bgr)
        metric_depth = _relative_to_metric(raw_depth, scale=max_depth_m)

        # Replace invalid values with 0 (matching SVO/BAG convention)
        metric_depth[~np.isfinite(metric_depth)] = 0.0

        out_name = f"frame_{idx:06d}.npy"
        np.save(str(output_dir / out_name), metric_depth)

        if depth_color_dir:
            _save_depth_color(
                str(depth_color_dir / f"frame_{idx:06d}.png"),
                metric_depth,
            )

        count += 1

    print(f"Depth estimation complete: {count} frames processed ({backend}).")
    return {
        "count": count,
        "output_dir": str(output_dir),
        "backend": backend,
    }


def generate_default_intrinsics(
    frame_dir: str | Path,
    output_path: str | Path,
) -> None:
    """Generate approximate camera intrinsics from frame resolution.

    Uses a focal length heuristic of ``f = max(width, height)``, which is
    a common pinhole-camera assumption when real calibration is unavailable.
    """
    frame_dir = Path(frame_dir)
    frames = sorted(
        p for p in frame_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frames:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    img = cv2.imread(str(frames[0]))
    h, w = img.shape[:2]

    focal = float(max(w, h))
    intr_dict = {
        "fx": focal,
        "fy": focal,
        "cx": w / 2.0,
        "cy": h / 2.0,
        "width": w,
        "height": h,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), intr_dict)
    print(f"Saved default intrinsics: {intr_dict}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monocular depth estimation for extracted RGB frames."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing RGB frame images (png/jpg).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save depth .npy files (float32, meters).",
    )
    parser.add_argument(
        "--depth_color_dir",
        type=str,
        default=None,
        help="Optional directory for JET-colorized depth PNGs.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="Small",
        choices=["Small", "Base", "Large"],
        help="DepthAnythingV2 model variant (default: Small).",
    )
    parser.add_argument(
        "--max_depth_m",
        type=float,
        default=5.0,
        help="Assumed max scene depth in meters (default: 5.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'cpu', or None (auto).",
    )
    parser.add_argument(
        "--generate_intrinsics",
        type=str,
        default=None,
        help="If set, also generate default intrinsics .npy at this path.",
    )

    args = parser.parse_args()

    result = estimate_depth_for_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        depth_color_dir=args.depth_color_dir,
        model_size=args.model_size,
        max_depth_m=args.max_depth_m,
        device=args.device,
    )

    if args.generate_intrinsics:
        generate_default_intrinsics(args.input_dir, args.generate_intrinsics)

    print(f"Done. {result['count']} depth maps saved to {result['output_dir']}")
