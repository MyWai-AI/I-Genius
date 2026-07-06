from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def check_zenoh():
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=3,
        )

        return "zenoh_edge" in result.stdout

    except Exception:
        return False


def check_ros2():
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        return result.returncode == 0

    except Exception:
        return False


def receiver_running():
    try:
        result = subprocess.run(
            ["pgrep", "-f", "trajectory_receiver.py"],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0

    except Exception:
        return False