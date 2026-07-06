import subprocess


def ros2_connected():

    try:

        result = subprocess.run(
            [
                "bash",
                "-c",
                "source /opt/ros/humble/setup.bash && ros2 topic list"
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        return result.returncode == 0

    except Exception:

        return False