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
#
# GPU usage:
#   Pass --device cuda  (or set CUDA_VISIBLE_DEVICES) for GPU acceleration.
#   Auto-selects model size: Small on CPU, Base on GPU.
#   Requires: torch (with CUDA), transformers, Pillow, tqdm.
#   When torch/transformers are unavailable the module gracefully falls back
#   to a Sobel-based pseudo-depth estimator so the pipeline still runs.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Any, Dict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Depth Anything V2 (local model)
# ---------------------------------------------------------------------------

def _load_depth_anything_v2(
    model_size: str | None = None,
    device=None,
) -> tuple:
    """Load DepthAnythingV2 model. Auto-selects size if not given: Small on CPU, Base on GPU.

    Returns (model, processor, "depth_anything_v2", device).
    Raises RuntimeError if the model cannot be loaded.
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_size is None:
        model_size = "Base" if device.type == "cuda" else "Small"

    model_size = str(model_size).strip().title()
    model_size_aliases = {
        "Tiny": "Small",
        "Medium": "Base",
    }
    model_size = model_size_aliases.get(model_size, model_size)
    valid_model_sizes = {"Small", "Base", "Large"}
    if model_size not in valid_model_sizes:
        raise ValueError(
            f"Unsupported Depth Anything V2 model size '{model_size}'. "
            f"Use one of: {', '.join(sorted(valid_model_sizes))}."
        )

    try:
        from transformers import AutoProcessor as _AutoProcessor
    except ImportError:
        from transformers import AutoImageProcessor as _AutoProcessor
    from transformers import AutoModelForDepthEstimation

    model_id = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
    processor = _AutoProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).eval()
    return model, processor, "depth_anything_v2", device


def _predict_depth(
    model,
    transform,
    device,
    rgb_bgr: np.ndarray,
) -> np.ndarray:
    """Run DepthAnythingV2 depth prediction on a single BGR frame.

    Returns (H, W) float32 relative depth map.
    """
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)
    inputs = transform(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    return depth.cpu().numpy().astype(np.float32)


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
    model_size: Optional[str] = None,
    max_depth_m: float = 5.0,
    device: Optional[str] = None,
    # legacy alias
    preview_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run DepthAnythingV2 depth estimation on a directory of RGB frames.

    Auto-selects model size when not specified: Small on CPU, Base on GPU.
    Falls back to Sobel-based pseudo-depth if torch/transformers are unavailable.

    Parameters
    ----------
    input_dir : Path
        Directory containing RGB frames (png/jpg).
    output_dir : Path
        Directory to save depth ``.npy`` files (float32, meters).
    depth_color_dir : Path | None
        Optional directory for JET colorized depth PNGs.
    model_size : str | None
        ``"Small"``, ``"Base"``, or ``"Large"``. None = auto.
    max_depth_m : float
        Assumed maximum scene depth in meters for normalization (default 5.0).
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto).
    preview_dir : Path | None
        Legacy alias for depth_color_dir.

    Returns
    -------
    dict
        ``{"count": int, "output_dir": str, "backend": str}``
    """
    # legacy alias
    if depth_color_dir is None and preview_dir is not None:
        depth_color_dir = preview_dir

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if depth_color_dir:
        depth_color_dir = Path(depth_color_dir)
        depth_color_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found in {input_dir}")

    # Try Depth Anything V2 first; fall back to Sobel pseudo-depth
    try:
        import torch
        from tqdm import tqdm as _tqdm

        torch_device = torch.device(device) if device else None
        model, transform, _, torch_device = _load_depth_anything_v2(
            model_size=model_size, device=torch_device
        )
        print(f"DepthAnythingV2 loaded on {torch_device}")
        backend = "depth_anything_v2"

        count = 0
        for idx, fpath in enumerate(_tqdm(frames, desc="Depth estimation")):
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                continue
            raw_depth = _predict_depth(model, transform, torch_device, bgr)
            metric_depth = _relative_to_metric(raw_depth, scale=max_depth_m)
            metric_depth[~np.isfinite(metric_depth)] = 0.0
            out_name = f"frame_{idx:06d}.npy"
            np.save(str(output_dir / out_name), metric_depth)
            if depth_color_dir:
                _save_depth_color(str(depth_color_dir / f"frame_{idx:06d}.png"), metric_depth)
            count += 1

    except Exception as _e:
        print(f"DepthAnythingV2 unavailable ({_e}); falling back to Sobel pseudo-depth.")
        backend = "sobel_pseudodepth"
        count = 0
        for idx, fpath in enumerate(frames):
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                continue
            metric_depth = estimate_depth_single_frame(bgr)
            metric_depth[~np.isfinite(metric_depth)] = 0.0
            out_name = f"frame_{idx:06d}.npy"
            np.save(str(output_dir / out_name), metric_depth)
            if depth_color_dir:
                _save_depth_color(str(depth_color_dir / f"frame_{idx:06d}.png"), metric_depth)
            count += 1

    print(f"Depth estimation complete: {count} frames processed (backend={backend}).")
    return {
        "count": count,
        "output_dir": str(output_dir),
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# Hugging Face Inference API (remote GPU — much faster)
# ---------------------------------------------------------------------------

def estimate_depth_hf_inference(
    input_dir: str | Path,
    output_dir: str | Path,
    depth_color_dir: Optional[str | Path] = None,
    max_depth_m: float = 5.0,
    hf_token: Optional[str] = None,
    model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
) -> Dict[str, Any]:
    """Run monocular depth estimation using Hugging Face Inference API (remote GPU).

    This is much faster than local computation because it offloads to HF's GPU servers.
    Requires HF_TOKEN environment variable or ``hf_token`` parameter.

    Parameters
    ----------
    input_dir : Path
        Directory containing RGB frames (png/jpg).
    output_dir : Path
        Directory to save depth ``.npy`` files (float32, meters).
    depth_color_dir : Path | None
        Optional directory for JET colorized depth PNGs.
    max_depth_m : float
        Assumed maximum scene depth in meters for normalization (default 5.0).
    hf_token : str | None
        Hugging Face API token. If None, reads from HF_TOKEN env var.
    model_id : str
        HF model ID (default: ``"depth-anything/Depth-Anything-V2-Large-hf"``).

    Returns
    -------
    dict
        ``{"count": int, "output_dir": str, "backend": str}``
    """
    from huggingface_hub import InferenceClient
    from PIL import Image
    import os

    from tqdm import tqdm as _tqdm

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if depth_color_dir:
        depth_color_dir = Path(depth_color_dir)
        depth_color_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Set HF_TOKEN environment variable or pass hf_token parameter."
        )

    client = InferenceClient(token=token)
    print(f"Using Hugging Face Inference API: {model_id}")

    frames = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found in {input_dir}")

    count = 0
    for idx, fpath in enumerate(_tqdm(frames, desc="Depth estimation (HF API)")):
        try:
            img = Image.open(str(fpath))
            depth_output = client.depth_estimation(image=img)
            raw_depth = np.array(depth_output["depth"], dtype=np.float32)
            metric_depth = _relative_to_metric(raw_depth, scale=max_depth_m)
            metric_depth[~np.isfinite(metric_depth)] = 0.0
            out_name = f"frame_{idx:06d}.npy"
            np.save(str(output_dir / out_name), metric_depth)
            if depth_color_dir:
                _save_depth_color(str(depth_color_dir / f"frame_{idx:06d}.png"), metric_depth)
            count += 1
        except Exception as e:
            print(f"Error processing {fpath.name}: {e}")

    print(f"Depth estimation complete: {count} frames processed (HF Inference API).")
    return {
        "count": count,
        "output_dir": str(output_dir),
        "backend": "hf_inference",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        "fx": focal, "fy": focal,
        "cx": w / 2.0, "cy": h / 2.0,
        "width": w, "height": h,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), intr_dict)
    print(f"Saved default intrinsics: {intr_dict}")


# ---------------------------------------------------------------------------
# Legacy shim — used by video_extraction_service.py for per-frame fallback
# ---------------------------------------------------------------------------

def estimate_depth_single_frame(frame: np.ndarray) -> np.ndarray:
    """Sobel-based pseudo depth for a single BGR frame (no GPU required).

    Used as an inline per-frame fallback during video extraction when
    Depth Anything V2 is not available. For batch processing prefer
    :func:`estimate_depth_for_frames` which will use DAv2 when possible.
    """
    if frame is None:
        raise ValueError("frame must not be None")
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame.copy()
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)
    depth = cv2.GaussianBlur(1.0 - normalized, (0, 0), sigmaX=3.0)
    return (0.25 + depth * 4.75).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monocular depth estimation for video frames (Depth Anything V2)."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing RGB frame images (png/jpg).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save depth .npy files (float32, meters).")
    parser.add_argument("--depth_color_dir", type=str, default=None,
                        help="Optional directory for JET-colorized depth PNGs.")
    parser.add_argument("--model_size", type=str, default=None,
                        choices=["Small", "Base", "Large"],
                        help="DepthAnythingV2 model variant (default: auto).")
    parser.add_argument("--max_depth_m", type=float, default=5.0,
                        help="Assumed max scene depth in meters (default: 5.0).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or None (auto).")
    parser.add_argument("--generate_intrinsics", type=str, default=None,
                        help="If set, also generate default intrinsics .npy at this path.")

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
