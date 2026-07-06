import subprocess


def receiver_running():

    try:

        result = subprocess.run(
            ["pgrep", "-f", "trajectory_receiver.py"],
            capture_output=True,
            text=True
        )

        return result.returncode == 0

    except Exception:

        return False


def stop_receiver():

    try:

        subprocess.run(
            ["pkill", "-f", "trajectory_receiver.py"],
            check=False
        )

        return True

    except Exception:

        return False


def restart_receiver():

    stop_receiver()

    from services.receiver_launcher import start_receiver

    start_receiver()

    return True