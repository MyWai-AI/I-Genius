"""Print container GPU visibility diagnostics for deployment debugging.

Run inside the container:
    python src/scripts/linux/gpu_diagnostics.py
"""

from __future__ import annotations

import os
import shutil
import subprocess


def _run(command: list[str]) -> str:
    if not shutil.which(command[0]):
        return f"{command[0]}: not found"
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        ).strip()
    except Exception as exc:
        return f"{command[0]} failed: {exc}"


def main() -> None:
    print("=== NVIDIA container visibility ===")
    for name in (
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    ):
        print(f"{name}={os.environ.get(name, '')}")

    print("\n=== nvidia-smi ===")
    print(_run(["nvidia-smi"]))

    print("\n=== PyTorch ===")
    try:
        import torch

        print(f"torch.__version__={torch.__version__}")
        print(f"torch.version.cuda={torch.version.cuda}")
        print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
        print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch import/check failed: {exc}")

    print("\n=== Ultralytics (YOLO) ===")
    try:
        import ultralytics

        print(f"ultralytics.__version__={ultralytics.__version__}")
        from ultralytics.utils import torch_utils
        print(f"YOLO device: {torch_utils.select_device('')}")
    except Exception as exc:
        print(f"ultralytics check failed: {exc}")

    print("\n=== OpenCV CUDA ===")
    try:
        import cv2

        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"cv2.__version__={cv2.__version__}")
        print(f"cv2.cuda.getCudaEnabledDeviceCount()={count}")
    except Exception as exc:
        print(f"cv2 CUDA check failed: {exc}")


if __name__ == "__main__":
    main()
