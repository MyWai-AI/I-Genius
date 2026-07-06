from pathlib import Path
import sys
import subprocess

SDK_ROOT = (
    Path(__file__).resolve().parent.parent
    / "fairino-python-sdk-main"
    / "linux"
)

sys.path.insert(0, str(SDK_ROOT))

from fairino import Robot


ROBOT_IP = "192.168.57.2"


def connect_robot(ip=ROBOT_IP):
    return Robot.RPC(ip)


def robot_status(ip=ROBOT_IP):

    try:

        robot = connect_robot(ip)

        mode = robot.Mode()

        return {
            "connected": True,
            "controller_ip": ip,
            "mode": mode
        }

    except Exception:

        return {
            "connected": False,
            "controller_ip": "Unknown",
            "mode": None
        }


def execute_trajectory(csv_file):

    root = Path(__file__).resolve().parent.parent

    calibration = (
        root
        / "colleague_cam2base_calibration.json"
    )

    output_csv = (
        root
        / "data"
        / "generated_execution.csv"
    )

    command = [

        sys.executable,

        str(
            root
            / "services"
            / "trajectory_executor.py"
        ),

        "--csv",
        str(csv_file),

        "--calibration",
        str(calibration),

        "--out",
        str(output_csv),

        "--point-step",
        "20",

        "--z-offset",
        "-80",

        "--speed",
        "5",

        "--start-method",
        "staged",

        "--execute",
    ]

    result = subprocess.run(
        command,
        cwd=root,
        check=True,
    )

    return result.returncode == 0