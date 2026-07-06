import subprocess


def start_zenoh():

    subprocess.run(
        [
            "docker",
            "start",
            "zenoh_edge"
        ],
        check=False
    )


def stop_zenoh():

    subprocess.run(
        [
            "docker",
            "stop",
            "zenoh_edge"
        ],
        check=False
    )