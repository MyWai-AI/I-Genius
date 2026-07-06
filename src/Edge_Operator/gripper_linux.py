from pathlib import Path
import sys
import time

SDK_ROOT = Path(__file__).resolve().parent / "fairino-python-sdk-main" / "linux"
sys.path.insert(0, str(SDK_ROOT))

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)

if stdout_reconfigure is not None:
    stdout_reconfigure(encoding="utf-8", errors="replace")

if stderr_reconfigure is not None:
    stderr_reconfigure(encoding="utf-8", errors="replace")

from fairino import Robot


ROBOT_IP = "192.168.57.2"

GRIPPER_INDEX = 1

OPEN_POS = 50
CLOSE_POS = 90

VEL = 30
FORCE = 40

MAX_TIME_MS = 30000
BLOCK = 0

# JODELL RG
COMPANY = 6
DEVICE = 0
SOFT_VERSION = 0

BUS = 0


def require_ok(name, result, fatal=True):

    code = result[0] if isinstance(result, tuple) else result

    print(f"{name}: {result}")

    if code != 0:

        if fatal:
            raise RuntimeError(
                f"{name} failed with code {code}"
            )

        return False

    return True


def connect():

    robot = Robot.RPC(ROBOT_IP)

    if not getattr(robot, "is_connect", False):

        raw_controller_ip = robot.robot.GetControllerIP()

        print(
            f"Raw XML-RPC GetControllerIP: "
            f"{raw_controller_ip}"
        )

        Robot.RPC.is_connect = True

        print(
            "Continuing with XML-RPC control. "
            "CNDE realtime-state unavailable."
        )

    return robot


def setup_gripper(robot, bus=BUS):

    print(
        f"\nConfiguring gripper:"
        f"\n  company={COMPANY}"
        f"\n  device={DEVICE}"
        f"\n  softversion={SOFT_VERSION}"
        f"\n  bus={bus}"
    )

    require_ok(
        "SetGripperConfig",
        robot.SetGripperConfig(
            company=COMPANY,
            device=DEVICE,
            softversion=SOFT_VERSION,
            bus=bus,
        ),
    )

    require_ok(
        "ActGripper reset",
        robot.ActGripper(
            index=GRIPPER_INDEX,
            action=0,
        ),
    )

    time.sleep(1)

    require_ok(
        "ActGripper enable",
        robot.ActGripper(
            index=GRIPPER_INDEX,
            action=1,
        ),
    )

    time.sleep(2)


def enable_auto(robot):

    require_ok(
        "RobotEnable",
        robot.RobotEnable(1),
        fatal=False,
    )

    require_ok(
        "Mode auto",
        robot.Mode(0),
        fatal=False,
    )


def manual_mode(robot):

    require_ok(
        "Mode manual",
        robot.Mode(1),
        fatal=False,
    )


def move_gripper(robot, pos, label):

    ok = require_ok(
        label,
        robot.MoveGripper(
            index=GRIPPER_INDEX,
            pos=pos,
            vel=VEL,
            force=FORCE,
            maxtime=MAX_TIME_MS,
            block=BLOCK,
            type=0,
            rotNum=0,
            rotVel=0,
            rotTorque=0,
        ),
        fatal=False,
    )

    if not ok:

        print(
            "\nMoveGripper failed."
            "\nIf code=73:"
            "\n - Check Auto mode"
            "\n - Check Start"
            "\n - Check gripper config"
            "\n - Try bus=1"
        )

    time.sleep(1)


def print_help():

    print(
        "\nCommands:"
        "\n  a        -> enable + auto"
        "\n  m        -> manual mode"
        "\n  r        -> reconfigure gripper"
        "\n  b 0      -> set bus 0"
        "\n  b 1      -> set bus 1"
        "\n  o        -> open"
        "\n  c        -> close"
        "\n  p 70     -> move to position 70"
        "\n  q        -> quit"
    )


def main():

    robot = connect()

    setup_gripper(robot)

    print_help()

    while True:

        command = input("\n> ").strip().lower()

        if command == "a":

            enable_auto(robot)

        elif command == "m":

            manual_mode(robot)

        elif command == "r":

            setup_gripper(robot)

        elif command.startswith("b "):

            try:
                bus = int(command.split()[1])

            except (IndexError, ValueError):

                print("Use: b 0 or b 1")

                continue

            setup_gripper(
                robot,
                bus=bus,
            )

        elif command == "o":

            move_gripper(
                robot,
                OPEN_POS,
                "open",
            )

        elif command == "c":

            move_gripper(
                robot,
                CLOSE_POS,
                "close",
            )

        elif command.startswith("p "):

            try:

                pos = int(command.split()[1])

            except (IndexError, ValueError):

                print("Use: p 70")

                continue

            if not 0 <= pos <= 100:

                print(
                    "Position must be between "
                    "0 and 100."
                )

                continue

            move_gripper(
                robot,
                pos,
                f"position {pos}",
            )

        elif command == "q":

            break

        else:

            print_help()


if __name__ == "__main__":
    main()