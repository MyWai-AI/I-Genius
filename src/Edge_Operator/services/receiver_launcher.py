import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def start_receiver():

    subprocess.Popen(
        [
            "python3",
            str(ROOT / "services" / "trajectory_receiver.py")
        ],
        cwd=ROOT
    )