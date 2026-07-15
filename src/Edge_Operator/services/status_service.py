import subprocess


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
