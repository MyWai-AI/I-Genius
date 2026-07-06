from pathlib import Path
import argparse
import csv
import json
import sys
import time

SDK_ROOT = Path(__file__).resolve().parent.parent / "fairino-python-sdk-main" / "linux"
sys.path.insert(0, str(SDK_ROOT))
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if stdout_reconfigure is not None:
    stdout_reconfigure(encoding="utf-8", errors="replace")
if stderr_reconfigure is not None:
    stderr_reconfigure(encoding="utf-8", errors="replace")

from fairino import Robot


ROBOT_IP = "192.168.57.2"
TRAJECTORY_CSV = Path(__file__).resolve().parent / "robot_traj_pipe_mm.csv"
CALIBRATION_JSON = Path(__file__).resolve().parent / "trajectory_calibration.json"
TRANSFORMED_CSV = Path(__file__).resolve().parent / "robot_traj_pipe_fairino_mm.csv"

TOOL = 2
USER = 0
SPEED = 10
POINT_STEP = 4

# Defaults are intentionally conservative. Use command-line flags to move.
Z_OFFSET_MM = 0.0

# Conservative workspace limits. Adjust only after checking your cell.
MIN_Z_MM = -300.0
MAX_Z_MM = 3000.0

GRIPPER_INDEX = 1
GRIPPER_OPEN_POS = 50
GRIPPER_CLOSE_POS = 90
GRIPPER_VEL = 30
GRIPPER_FORCE = 40
GRIPPER_MAX_TIME_MS = 30000

GRIPPER_COMPANY = 6
GRIPPER_DEVICE = 0
GRIPPER_SOFT_VERSION = 0
GRIPPER_BUS = 0

EVENT_CLOSE = {"close", "closed", "grasp", "grab", "pick"}
EVENT_OPEN = {"open", "release", "drop", "place"}


def require_ok(name, result):
    code = result[0] if isinstance(result, tuple) else result
    print(f"{name}: {result}")
    if code != 0:
        raise RuntimeError(f"{name} failed with code {code}")
    return result


def print_robot_state(robot):
    for name, call in [
        ("RobotErrorCode", robot.GetRobotErrorCode),
        ("SafetyStopState", robot.GetSafetyStopState),
        ("EmergencyStopState", robot.GetRobotEmergencyStopState),
        ("ActualTCPPose", robot.GetActualTCPPose),
    ]:
        try:
            print(f"{name}: {call()}")
        except Exception as exc:
            print(f"{name}: unavailable ({exc})")


def soft_refresh(robot, context):
    print(f"Soft refresh robot controller ({context})...")
    for name in ("StopMove", "ProgramStop", "ResetAllError", "DragTeachSwitch"):
        method = getattr(robot, name, None)
        if method is None:
            continue
        try:
            result = method(0) if name == "DragTeachSwitch" else method()
            print(f"{name}: {result}")
        except Exception as exc:
            print(f"{name}: unavailable ({exc})")
        time.sleep(0.2)


def prepare_auto(robot):
    print("Preparing robot for automatic motion...")
    soft_refresh(robot, "before run")
    require_ok("RobotEnable", robot.RobotEnable(1))
    mode = robot.Mode(0)
    print(f"Mode auto: {mode}")
    if mode != 0:
        print_robot_state(robot)
        raise RuntimeError(
            "Mode auto failed. On the pendant/UI, exit drag/manual mode, clear alarms, "
            "put the controller in Auto, and press Start/Play before running again."
        )


def connect():
    robot = Robot.RPC(ROBOT_IP)
    if not getattr(robot, "is_connect", getattr(robot, "is_conect", False)):
        rpc = getattr(robot, "robot", None)
        if rpc is None:
            raise RuntimeError("Robot XML-RPC object was not created")
        raw_controller_ip = rpc.GetControllerIP()
        print(f"Raw XML-RPC GetControllerIP: {raw_controller_ip}")
        Robot.RPC.is_connect = True
        print("Continuing with XML-RPC control. CNDE realtime-state may be unavailable.")
    return robot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform a ZED/camera trajectory into Fairino robot coordinates and optionally replay it."
    )
    parser.add_argument("--csv", type=Path, default=TRAJECTORY_CSV, help="Camera-space trajectory CSV.")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_JSON, help="Calibration JSON file.")
    parser.add_argument("--out", type=Path, default=TRANSFORMED_CSV, help="Transformed robot-space CSV.")
    parser.add_argument("--calibrate", action="store_true", help="Teach camera-to-robot calibration points.")
    parser.add_argument("--execute", action="store_true", help="Actually move the robot. Omit for preview only.")
    parser.add_argument("--preflight-only", action="store_true", help="Connect and check inverse kinematics, but do not move.")
    parser.add_argument("--use-gripper", action="store_true", help="Open/close the gripper at event or index points.")
    parser.add_argument("--setup-gripper", action="store_true", help="Configure/reactivate the gripper before replay.")
    parser.add_argument("--grasp-index", type=int, help="Close gripper after this trajectory index.")
    parser.add_argument("--release-index", type=int, help="Open gripper after this trajectory index.")
    parser.add_argument("--z-offset", type=float, default=Z_OFFSET_MM, help="Lift every robot-space point by this many mm.")
    parser.add_argument("--input-scale", type=float, default=1.0, help="Scale CSV XYZ before transform. Use 1000 if input is meters.")
    parser.add_argument("--rpy", help="Override TCP orientation as rx,ry,rz in degrees.")
    parser.add_argument("--start-index", type=int, default=0, help="First trajectory index to replay.")
    parser.add_argument("--end-index", type=int, help="Last trajectory index to replay, inclusive.")
    parser.add_argument("--point-step", type=int, default=POINT_STEP, help="Use every Nth trajectory point.")
    parser.add_argument("--speed", type=float, default=SPEED, help="MoveL velocity percentage.")
    parser.add_argument("--start-with-movej", action="store_true", help="MoveJ to the first replay point before MoveL trajectory following.")
    parser.add_argument(
        "--start-method",
        choices=["movel", "movej", "movecart", "staged"],
        default="movel",
        help="How to move from current robot pose to the first replay point.",
    )
    parser.add_argument("--safe-z", type=float, default=0.0, help="Safe Z height for --start-method staged. Use 0 for automatic nearby height.")
    parser.add_argument("--return-to-start", action="store_true", help="After replay, MoveL back to the first replay point so the next run starts cleanly.")
    parser.add_argument("--no-end-refresh", action="store_true", help="Do not stop/clear the controller after the trajectory finishes.")
    parser.add_argument("--min-z", type=float, default=MIN_Z_MM, help="Minimum allowed robot Z in mm.")
    parser.add_argument("--max-z", type=float, default=MAX_Z_MM, help="Maximum allowed robot Z in mm.")
    return parser.parse_args()


def normalize_event(value):
    event = (value or "").strip().lower()
    if event in EVENT_CLOSE:
        return "close"
    if event in EVENT_OPEN:
        return "open"
    return ""


def load_camera_samples(path):
    samples = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        required = {"x_mm", "y_mm", "z_mm"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            event = ""
            for key in ("event", "action", "gripper"):
                event = normalize_event(row.get(key))
                if event:
                    break
            samples.append({
                "camera_xyz": [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])],
                "rpy": parse_optional_rpy(row),
                "event": event,
            })
    if len(samples) < 4:
        raise ValueError("Need at least 4 trajectory points for affine calibration.")
    return samples


def camera_points(samples):
    return [sample["camera_xyz"] for sample in samples]


def parse_rpy(value):
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--rpy must look like: 150.0,17.0,23.0")
    return [float(part) for part in parts]


def parse_optional_rpy(row):
    names = [
        ("rx_deg", "ry_deg", "rz_deg"),
        ("rx", "ry", "rz"),
        ("r_mm", "p_mm", "y_mm"),
    ]
    for keys in names:
        if all(key in row and str(row.get(key, "")).strip() != "" for key in keys):
            return [float(row[key]) for key in keys]
    return None


def get_tcp_xyz(robot, label):
    input(f"\nJog/drag robot TCP to the real-world point for {label}, then press Enter...")
    error, pose = require_ok("GetActualTCPPose", robot.GetActualTCPPose())
    return pose[:3], pose[3:]


def solve_linear_system(a, b):
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-9:
            raise ValueError("Calibration points are degenerate; choose more separated points.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][c] - factor * aug[col][c] for c in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


def solve_affine(camera_points, robot_points):
    # Robot xyz = 3x4 matrix * [camera_x, camera_y, camera_z, 1]
    rows = []
    vals = []
    for cam, rob in zip(camera_points, robot_points):
        x, y, z = cam
        basis = [x, y, z, 1.0]
        for axis in range(3):
            row = [0.0] * 12
            start = axis * 4
            row[start:start + 4] = basis
            rows.append(row)
            vals.append(rob[axis])

    # Normal equations for least squares: (A^T A) p = A^T b
    ata = [[0.0] * 12 for _ in range(12)]
    atb = [0.0] * 12
    for row, val in zip(rows, vals):
        for i in range(12):
            atb[i] += row[i] * val
            for j in range(12):
                ata[i][j] += row[i] * row[j]

    params = solve_linear_system(ata, atb)
    return [params[0:4], params[4:8], params[8:12]]


def apply_affine(matrix, point):
    x, y, z = point
    v = [x, y, z, 1.0]
    return [sum(matrix[axis][i] * v[i] for i in range(4)) for axis in range(3)]


def choose_calibration_indices(n):
    return sorted(set([0, n // 3, (2 * n) // 3, n - 1]))


def calibrate(robot, points, calibration_path, trajectory_csv):
    indices = choose_calibration_indices(len(points))
    print("\nCalibration uses these camera trajectory points:")
    for i in indices:
        print(f"  index {i}: {points[i]}")
    print("\nFor each point, physically jog/drag the robot TCP to the matching point in the robot workspace.")

    robot_points = []
    last_rpy = None
    for i in indices:
        xyz, rpy = get_tcp_xyz(robot, f"camera index {i}: {points[i]}")
        robot_points.append(xyz)
        last_rpy = rpy
        print(f"Recorded robot XYZ for index {i}: {xyz}")

    matrix = solve_affine([points[i] for i in indices], robot_points)
    data = {
        "trajectory_csv": str(trajectory_csv),
        "indices": indices,
        "camera_points": [points[i] for i in indices],
        "robot_points": robot_points,
        "affine_camera_to_robot": matrix,
        "default_rpy": last_rpy,
    }
    calibration_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved calibration to {calibration_path}")
    return data


def load_calibration(path):
    if not path.exists():
        raise FileNotFoundError(f"No calibration found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def calibration_matrix(calibration):
    if "affine_camera_to_robot" in calibration:
        matrix = calibration["affine_camera_to_robot"]
    elif "T_cam2base" in calibration:
        matrix = calibration["T_cam2base"][:3]
    else:
        raise KeyError("Calibration must contain affine_camera_to_robot or T_cam2base.")
    if len(matrix) != 3 or any(len(row) != 4 for row in matrix):
        raise ValueError("Calibration transform must be a 3x4 matrix or a 4x4 T_cam2base matrix.")
    return matrix


def save_transformed(path, samples):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg", "event", "index"])
        for idx, sample in enumerate(samples):
            rpy = sample.get("rpy") or ["", "", ""]
            writer.writerow([*sample["robot_xyz"], *rpy, sample["event"], idx])
    print(f"Saved transformed trajectory to {path}")


def print_bounds(points, min_z=MIN_Z_MM):
    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]
    spans = [maxs[i] - mins[i] for i in range(3)]
    print(f"Point count: {len(points)}")
    print(f"Min XYZ: {mins}")
    print(f"Max XYZ: {maxs}")
    print(f"Span XYZ: {spans}")
    print("First 5 transformed points:")
    for p in points[:5]:
        print(f"  {p}")
    low_points = [(i, p[2]) for i, p in enumerate(points) if p[2] < min_z]
    if low_points:
        first_idx, first_z = low_points[0]
        print(f"Warning: first point below {min_z:.1f} mm Z floor is index {first_idx}, Z={first_z:.2f}.")


def validate_points(points, min_z, max_z, fatal):
    invalid = []
    for idx, p in enumerate(points):
        if not (min_z <= p[2] <= max_z):
            invalid.append((idx, p[2]))
    if not invalid:
        return
    first_idx, first_z = invalid[0]
    message = (
        f"{len(invalid)} point(s) are outside Z limits {min_z:.1f}..{max_z:.1f} mm; "
        f"first is index {first_idx}, Z={first_z:.2f}."
    )
    if fatal:
        raise RuntimeError(message)
    print(f"Warning: {message}")


def setup_gripper(robot):
    prepare_auto(robot)
    require_ok("SetGripperConfig", robot.SetGripperConfig(
        company=GRIPPER_COMPANY,
        device=GRIPPER_DEVICE,
        softversion=GRIPPER_SOFT_VERSION,
        bus=GRIPPER_BUS,
    ))
    require_ok("ActGripper reset", robot.ActGripper(index=GRIPPER_INDEX, action=0))
    time.sleep(1)
    require_ok("ActGripper enable", robot.ActGripper(index=GRIPPER_INDEX, action=1))
    time.sleep(2)


def move_gripper(robot, pos, label):
    require_ok(label, robot.MoveGripper(
        index=GRIPPER_INDEX,
        pos=pos,
        vel=GRIPPER_VEL,
        force=GRIPPER_FORCE,
        maxtime=GRIPPER_MAX_TIME_MS,
        block=0,
        type=0,
        rotNum=0,
        rotVel=0,
        rotTorque=0,
    ))


def add_index_events(samples, grasp_index, release_index):
    if grasp_index is not None:
        samples[grasp_index]["event"] = "close"
    if release_index is not None:
        samples[release_index]["event"] = "open"


def validate_event_index(samples, index, label):
    if index is None:
        return
    if not 0 <= index < len(samples):
        raise ValueError(f"{label} index {index} is outside trajectory range 0..{len(samples) - 1}.")


def validate_range(samples, start_index, end_index):
    if not 0 <= start_index < len(samples):
        raise ValueError(f"--start-index {start_index} is outside trajectory range 0..{len(samples) - 1}.")
    if end_index is not None and not start_index <= end_index < len(samples):
        raise ValueError(f"--end-index {end_index} must be in range {start_index}..{len(samples) - 1}.")


def print_events(samples):
    events = [(i, sample["event"]) for i, sample in enumerate(samples) if sample["event"]]
    if not events:
        print("No gripper events found. Add an event/action/gripper column, or pass --grasp-index and --release-index.")
        return
    print("Gripper events:")
    for idx, event in events:
        print(f"  index {idx}: {event}")


def replay_points(samples, args):
    end_index = args.end_index if args.end_index is not None else len(samples) - 1
    return [
        (i, sample)
        for i, sample in enumerate(samples)
        if args.start_index <= i <= end_index and (i % args.point_step == 0 or sample["event"])
    ]


def pose_for_sample(sample, default_rpy):
    return sample["robot_xyz"] + (sample.get("rpy") or default_rpy)


def check_inverse_kinematics(robot, replay, default_rpy):
    failures = []
    joint_result = robot.GetActualJointPosDegree()
    joint_ref = joint_result[1] if isinstance(joint_result, tuple) and joint_result[0] == 0 else None
    for idx, sample in replay:
        pose = pose_for_sample(sample, default_rpy)
        if joint_ref is not None:
            code, joints = robot.GetInverseKinRef(0, pose, joint_ref)
            if code != 0:
                code, joints = robot.GetInverseKin(0, pose, -1)
        else:
            code, joints = robot.GetInverseKin(0, pose, -1)
        if code != 0:
            failures.append((idx, code, pose))
            continue
        sample["joint_pos"] = joints
        joint_ref = joints
    if failures:
        print("\nInverse kinematics failed for these replay points:")
        for idx, code, pose in failures[:10]:
            print(f"  index {idx}: code {code}, pose {pose}")
        if len(failures) > 10:
            print(f"  ... {len(failures) - 10} more")
        raise RuntimeError(
            "At least one replay pose has no IK solution. "
            "Try a different --rpy, a shorter --start-index/--end-index segment, or verify the calibration/units."
        )
    print(f"IK preflight OK for {len(replay)} replay point(s).")


def follow(robot, samples, rpy, args):
    prepare_auto(robot)

    replay = replay_points(samples, args)
    check_inverse_kinematics(robot, replay, rpy)
    if args.preflight_only:
        print("Preflight only. No robot motion executed.")
        return

    start_offset = 0
    start_method = "movej" if args.start_with_movej else args.start_method
    if replay and start_method != "movel":
        start_idx, start_sample = replay[0]
        start_pose = pose_for_sample(start_sample, rpy)
        move_to_start(robot, start_idx, start_pose, start_sample, args, start_method)
        start_offset = 1
        if args.use_gripper and start_sample["event"] == "close":
            move_gripper(robot, GRIPPER_CLOSE_POS, f"close gripper at index {start_idx}")
        elif args.use_gripper and start_sample["event"] == "open":
            move_gripper(robot, GRIPPER_OPEN_POS, f"open gripper at index {start_idx}")

    print(f"Executing {len(replay)} MoveL points using every {args.point_step}th transformed point plus event points.")
    for move_i, (idx, sample) in enumerate(replay[start_offset:], start=start_offset):
        pose = pose_for_sample(sample, rpy)
        result = robot.MoveL(
            desc_pos=pose,
            tool=TOOL,
            user=USER,
            joint_pos=sample.get("joint_pos", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vel=args.speed,
        )
        if (result[0] if isinstance(result, tuple) else result) != 0:
            print_robot_state(robot)
        require_ok(f"MoveL {move_i} (trajectory index {idx})", result)
        if args.use_gripper and sample["event"] == "close":
            move_gripper(robot, GRIPPER_CLOSE_POS, f"close gripper at index {idx}")
        elif args.use_gripper and sample["event"] == "open":
            move_gripper(robot, GRIPPER_OPEN_POS, f"open gripper at index {idx}")
        time.sleep(0.02)

    if args.return_to_start and replay:
        start_idx, start_sample = replay[0]
        start_pose = pose_for_sample(start_sample, rpy)
        print(f"Return MoveL to trajectory start index {start_idx}: {start_pose}")
        result = robot.MoveL(
            desc_pos=start_pose,
            tool=TOOL,
            user=USER,
            joint_pos=start_sample.get("joint_pos", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vel=args.speed,
        )
        if (result[0] if isinstance(result, tuple) else result) != 0:
            print_robot_state(robot)
        require_ok(f"Return MoveL start (trajectory index {start_idx})", result)

    if not args.no_end_refresh:
        soft_refresh(robot, "after run")


def command_ok(result):
    return (result[0] if isinstance(result, tuple) else result) == 0


def move_to_start(robot, start_idx, start_pose, start_sample, args, method):
    print(f"Move to trajectory start index {start_idx} using {method}: {start_pose}")
    if method == "movej":
        result = robot.MoveJ(
            joint_pos=start_sample["joint_pos"],
            desc_pos=start_pose,
            tool=TOOL,
            user=USER,
            vel=args.speed,
        )
        if not command_ok(result):
            print_robot_state(robot)
        require_ok(f"MoveJ start (trajectory index {start_idx})", result)
        return

    if method == "movecart":
        result = robot.MoveCart(desc_pos=start_pose, tool=TOOL, user=USER, vel=args.speed)
        if not command_ok(result):
            print_robot_state(robot)
        require_ok(f"MoveCart start (trajectory index {start_idx})", result)
        return

    current_result = robot.GetActualTCPPose()
    if not isinstance(current_result, tuple) or current_result[0] != 0:
        raise RuntimeError(f"Could not read current TCP pose for staged start: {current_result}")
    current_pose = current_result[1]
    if args.safe_z > 0:
        safe_z = args.safe_z
    else:
        safe_z = max(current_pose[2], start_pose[2]) + 40.0
    staged_poses = [
        [current_pose[0], current_pose[1], safe_z, current_pose[3], current_pose[4], current_pose[5]],
        [start_pose[0], start_pose[1], safe_z, start_pose[3], start_pose[4], start_pose[5]],
        start_pose,
    ]
    for i, pose in enumerate(staged_poses):
        print(f"Staged start {i}: {pose}")
        result = robot.MoveCart(desc_pos=pose, tool=TOOL, user=USER, vel=args.speed)
        if not command_ok(result):
            print_robot_state(robot)
        require_ok(f"Staged MoveCart start {i}", result)


def main():
    args = parse_args()
    if args.point_step < 1:
        raise ValueError("--point-step must be 1 or greater.")
    rpy_override = parse_rpy(args.rpy)

    samples = load_camera_samples(args.csv)
    points = camera_points(samples)
    validate_range(samples, args.start_index, args.end_index)
    validate_event_index(samples, args.grasp_index, "--grasp-index")
    validate_event_index(samples, args.release_index, "--release-index")
    add_index_events(samples, args.grasp_index, args.release_index)

    robot = connect() if args.calibrate or args.execute or args.preflight_only else None

    if args.calibrate:
        calibration = calibrate(robot, points, args.calibration, args.csv)
    else:
        calibration = load_calibration(args.calibration)

    matrix = calibration_matrix(calibration)
    for sample in samples:
        point = [coord * args.input_scale for coord in sample["camera_xyz"]]
        robot_xyz = apply_affine(matrix, point)
        robot_xyz[2] += args.z_offset
        sample["robot_xyz"] = robot_xyz

    transformed = [sample["robot_xyz"] for sample in samples]
    save_transformed(args.out, samples)
    print_bounds(transformed, args.min_z)
    print_events(samples)
    validate_points(transformed, args.min_z, args.max_z, fatal=args.execute)

    if not args.execute and not args.preflight_only:
        print("\nPreview only. No robot motion executed.")
        print("For first robot motion, try a lifted path such as: python fairino_follow_csv_trajectory.py --z-offset 80 --execute")
        return

    rpy = rpy_override if rpy_override is not None else calibration["default_rpy"]
    if args.preflight_only:
        prepare_auto(robot)
        check_inverse_kinematics(robot, replay_points(samples, args), rpy)
        print("Preflight only. No robot motion executed.")
        return

    if args.setup_gripper:
        setup_gripper(robot)
    if args.use_gripper:
        move_gripper(robot, GRIPPER_OPEN_POS, "open gripper before replay")
    follow(robot, samples, rpy, args)


if __name__ == "__main__":
    main()
