import subprocess


def zenoh_running():

    try:

        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True
        )

        return "zenoh_edge" in result.stdout

    except Exception:

        return False