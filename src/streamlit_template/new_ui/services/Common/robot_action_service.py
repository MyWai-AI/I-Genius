"""
Robot Action Service — Save Animation & Push to Robot (DDS/Vulcanexus).
"""
import csv
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TRANSPORT_VULCANEXUS = "Vulcanexus LAN (PoseArray)"
TRANSPORT_CYCLONEDDS = "CycloneDDS JointTrajectory"

DEFAULT_ROS_DOMAIN_ID = 42
DEFAULT_VULCANEXUS_TOPIC = "/learned_trajectory"
DEFAULT_VULCANEXUS_STATUS_TOPIC = "/trajectory_status"
DEFAULT_VULCANEXUS_FRAME_ID = "camera_frame"
DEFAULT_VULCANEXUS_DISCOVERY_SERVER = "192.168.0.10:14520"
DEFAULT_VULCANEXUS_REPEAT = 40
DEFAULT_VULCANEXUS_RATE_HZ = 2.0
DEFAULT_VULCANEXUS_WAIT_FOR_SUBSCRIBER_SEC = 15.0
DEFAULT_VULCANEXUS_STATUS_WAIT_SEC = 8.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def has_cyclonedds() -> bool:
    return importlib.util.find_spec("cyclonedds") is not None


def has_vulcanexus_publisher() -> bool:
    return (_repo_root() / "scripts" / "vulcanexus" / "docker_publish_traj.sh").is_file()


def has_vulcanexus_status_subscriber() -> bool:
    return (_repo_root() / "scripts" / "vulcanexus" / "docker_wait_for_status.sh").is_file()


def get_robot_transport_options() -> List[str]:
    options: List[str] = []
    if has_vulcanexus_publisher():
        options.append(TRANSPORT_VULCANEXUS)
    options.append(TRANSPORT_CYCLONEDDS)
    return options


def get_default_robot_transport() -> str:
    if has_vulcanexus_publisher():
        return TRANSPORT_VULCANEXUS
    return TRANSPORT_CYCLONEDDS


def get_default_robot_domain_id() -> int:
    return int(os.getenv("ROS_DOMAIN_ID", str(DEFAULT_ROS_DOMAIN_ID)))


def get_default_vulcanexus_discovery_server() -> str:
    return os.getenv(
        "VILMA_VULCANEXUS_DISCOVERY_SERVER",
        os.getenv("ROS_DISCOVERY_SERVER", DEFAULT_VULCANEXUS_DISCOVERY_SERVER),
    )


def get_default_vulcanexus_publish_settings() -> Dict[str, object]:
    return {
        "topic": DEFAULT_VULCANEXUS_TOPIC,
        "discovery_server": get_default_vulcanexus_discovery_server(),
        "repeat_count": int(os.getenv("VILMA_VULCANEXUS_REPEAT", str(DEFAULT_VULCANEXUS_REPEAT))),
        "status_topic": DEFAULT_VULCANEXUS_STATUS_TOPIC,
        "status_wait_sec": float(
            os.getenv("VILMA_VULCANEXUS_STATUS_WAIT_SEC", str(DEFAULT_VULCANEXUS_STATUS_WAIT_SEC))
        ),
    }


def _write_cartesian_csv(csv_path: Path, cart_path) -> np.ndarray:
    arr = np.asarray(cart_path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"cart_path must have shape (N, 3); got {arr.shape!r}")
    if arr.shape[0] == 0:
        raise ValueError("cart_path is empty.")
    finite_mask = np.all(np.isfinite(arr), axis=1)
    if not np.all(finite_mask):
        removed = int(arr.shape[0] - np.count_nonzero(finite_mask))
        logger.warning("Dropping %s non-finite Cartesian waypoint(s) before publish.", removed)
        arr = arr[finite_mask]
    if arr.shape[0] == 0:
        raise ValueError("cart_path has no finite waypoints after filtering.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z"])
        writer.writerows(arr.tolist())
    return arr


def wait_for_vulcanexus_status(
    topic_name: str = "/trajectory_status",
    domain_id: int = 42,
    discovery_server: Optional[str] = None,
    timeout_sec: float = 8.0,
    container_name: Optional[str] = None,
) -> Tuple[bool, str, Optional[dict]]:
    script_path = _repo_root() / "scripts" / "vulcanexus" / "docker_wait_for_status.sh"
    if not script_path.is_file():
        return False, f"Vulcanexus status subscriber script not found: {script_path}", None

    env = os.environ.copy()
    env["HOST_WORKSPACE"] = str(_repo_root())
    env["TOPIC"] = topic_name
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["RMW_IMPLEMENTATION"] = env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    env["ROS_LOCALHOST_ONLY"] = env.get("ROS_LOCALHOST_ONLY", "0")
    env["TIMEOUT_SEC"] = str(timeout_sec)
    if discovery_server:
        env["ROS_DISCOVERY_SERVER"] = discovery_server
    if container_name:
        env["VULCANEXUS_SUB_CONTAINER"] = container_name

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(_repo_root()),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in [exc.stdout, exc.stderr] if part).strip()
        return False, f"Waiting for edge status failed: {details or exc}", None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return False, "No status payload returned by Vulcanexus status subscriber.", None
    payload = lines[-1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    return True, payload, parsed


def _start_vulcanexus_status_waiter(
    topic_name: str = "/trajectory_status",
    domain_id: int = DEFAULT_ROS_DOMAIN_ID,
    discovery_server: Optional[str] = None,
    timeout_sec: float = DEFAULT_VULCANEXUS_STATUS_WAIT_SEC,
    container_name: Optional[str] = None,
) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    script_path = _repo_root() / "scripts" / "vulcanexus" / "docker_wait_for_status.sh"
    if not script_path.is_file():
        return None, f"Vulcanexus status subscriber script not found: {script_path}"

    env = os.environ.copy()
    env["HOST_WORKSPACE"] = str(_repo_root())
    env["TOPIC"] = topic_name
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["RMW_IMPLEMENTATION"] = env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    env["ROS_LOCALHOST_ONLY"] = env.get("ROS_LOCALHOST_ONLY", "0")
    env["TIMEOUT_SEC"] = str(timeout_sec)
    if discovery_server:
        env["ROS_DISCOVERY_SERVER"] = discovery_server
    if container_name:
        env["VULCANEXUS_SUB_CONTAINER"] = container_name

    process = subprocess.Popen(
        ["bash", str(script_path)],
        cwd=str(_repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return process, None


def _finish_vulcanexus_status_waiter(
    process: subprocess.Popen,
) -> Tuple[bool, str, Optional[dict]]:
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        details = "\n".join(part for part in [stdout, stderr] if part).strip()
        return False, f"Waiting for edge status failed: {details or process.returncode}", None

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return False, "No status payload returned by Vulcanexus status subscriber.", None
    payload = lines[-1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    return True, payload, parsed


def _smooth_and_upsample(
    traj_points: List[Dict],
    target_fps: float = 30.0,
    max_deg_per_frame: float = 4.0,
) -> List[Dict]:
    """Interpolate trajectory to target FPS and smooth to eliminate jerks.

    Uses cubic interpolation to resample the trajectory so that the saved
    animation is smooth enough for viewers that step through keyframes
    without their own interpolation.

    Args:
        traj_points: Original trajectory keyframes.
        target_fps: Desired output framerate.
        max_deg_per_frame: Maximum allowed joint delta (degrees) per output frame.

    Returns:
        Resampled and smoothed trajectory points.
    """
    if len(traj_points) < 4:
        return traj_points

    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter

    times = np.array([pt["time"] for pt in traj_points])
    joint_names = list(traj_points[0]["q"].keys())
    values = np.array([[pt["q"][j] for j in joint_names] for pt in traj_points])

    duration = times[-1] - times[0]
    if duration <= 0:
        return traj_points

    # Determine output sample count: at least target_fps, or more if needed
    n_out = max(int(np.ceil(duration * target_fps)), len(traj_points))
    t_out = np.linspace(times[0], times[-1], n_out)

    # Cubic-spline interpolation per joint
    resampled = np.zeros((n_out, len(joint_names)))
    for j_idx in range(len(joint_names)):
        cs = CubicSpline(times, values[:, j_idx])
        resampled[:, j_idx] = cs(t_out)

    # Light Savitzky-Golay smoothing to eliminate any residual high-freq noise
    win = min(11, n_out if n_out % 2 == 1 else n_out - 1)
    if n_out >= win >= 5:
        for j_idx in range(len(joint_names)):
            resampled[:, j_idx] = savgol_filter(resampled[:, j_idx], window_length=win, polyorder=3)

    # Velocity clamping: cap max change per frame
    max_rad = np.deg2rad(max_deg_per_frame)
    for _iteration in range(3):
        diffs = np.diff(resampled, axis=0)
        exceeded = np.abs(diffs) > max_rad
        if not np.any(exceeded):
            break
        for i in range(diffs.shape[0]):
            for j_idx in range(diffs.shape[1]):
                if abs(diffs[i, j_idx]) > max_rad:
                    sign = 1.0 if diffs[i, j_idx] > 0 else -1.0
                    resampled[i + 1, j_idx] = resampled[i, j_idx] + sign * max_rad

    # Build output
    result = []
    for i in range(n_out):
        q_map = {joint_names[j]: round(float(resampled[i, j]), 6) for j in range(len(joint_names))}
        result.append({"time": round(float(t_out[i]), 4), "q": q_map})
    return result


def save_animation(
    urdf_path: str,
    traj_points: List[Dict],
    animation_name: str = "PipelineAction",
) -> Tuple[bool, str]:
    """
    Save trajectory as a named animation in animations.json next to the URDF.

    The trajectory is resampled via cubic-spline interpolation and velocity-
    clamped so that playback is smooth even in viewers that do not interpolate
    between keyframes.

    Args:
        urdf_path: Path to the active URDF file.
        traj_points: List of {"time": float, "q": {"Joint_1": val, ...}}.
        animation_name: Key under which to store the animation.

    Returns:
        (success, message)
    """
    if not traj_points:
        return False, "No trajectory data to save."

    anim_dir = Path(urdf_path).parent
    anim_file = anim_dir / "animations.json"

    # Load existing or create new
    data: dict = {"animations": {}}
    if anim_file.exists():
        try:
            with open(anim_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "animations" not in data:
                data["animations"] = {}
        except Exception as e:
            logger.warning(f"Could not read existing animations.json: {e}. Creating new.")
            data = {"animations": {}}

    # Smooth, upsample, and velocity-clamp before saving
    try:
        clean_points = _smooth_and_upsample(traj_points, target_fps=30.0, max_deg_per_frame=4.0)
        logger.info(f"Animation upsampled: {len(traj_points)} → {len(clean_points)} frames")
    except Exception as e:
        logger.warning(f"Upsampling failed ({e}), saving raw trajectory.")
        clean_points = []
        for pt in traj_points:
            clean_q = {k: round(float(v), 4) for k, v in pt["q"].items()}
            clean_points.append({"time": round(float(pt["time"]), 4), "q": clean_q})

    data["animations"][animation_name] = clean_points

    try:
        anim_dir.mkdir(parents=True, exist_ok=True)
        with open(anim_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved animation '{animation_name}' ({len(clean_points)} frames) to {anim_file}")
        msg = f"Animation '{animation_name}' saved ({len(clean_points)} frames) → {anim_file}"
    except Exception as e:
        logger.exception("Failed to write animations.json")
        return False, f"Failed to save: {e}"

    # --- Push back to MYWAI server if this robot came from the platform ---
    try:
        import streamlit as st
        blob_base = st.session_state.get("mywai_active_model_blob_base")
        blob_container = st.session_state.get("mywai_active_model_container")
        mywai_token = st.session_state.get("mywai_token")
        mywai_endpoint = st.session_state.get("mywai_endpoint")

        if blob_base and blob_container:
            from src.streamlit_template.new_ui.services.Generic.mywai_service import upload_blob
            server_path = f"{blob_base}/animations.json"
            up_ok, up_msg = upload_blob(
                container=blob_container,
                file_path=server_path,
                local_path=str(anim_file),
                token=mywai_token,
                endpoint=mywai_endpoint,
            )
            if up_ok:
                msg += f" | Pushed to server: {blob_container}/{server_path}"
            else:
                logger.warning(f"Could not push animations.json to server: {up_msg}")
                msg += f" | Server upload failed: {up_msg}"
    except Exception as upload_err:
        logger.warning(f"Server upload skipped: {upload_err}")

    return True, msg


def publish_cartesian_trajectory_vulcanexus(
    cart_path,
    topic_name: str = DEFAULT_VULCANEXUS_TOPIC,
    domain_id: int = DEFAULT_ROS_DOMAIN_ID,
    discovery_server: Optional[str] = None,
    status_topic: Optional[str] = None,
    status_timeout_sec: float = 0.0,
    repeat: int = DEFAULT_VULCANEXUS_REPEAT,
    rate_hz: float = DEFAULT_VULCANEXUS_RATE_HZ,
    wait_for_subscriber_sec: float = DEFAULT_VULCANEXUS_WAIT_FOR_SUBSCRIBER_SEC,
    frame_id: str = DEFAULT_VULCANEXUS_FRAME_ID,
    container_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Publish a Cartesian trajectory through the existing Vulcanexus LAN helper.

    The current edge-side tooling already subscribes to geometry_msgs/PoseArray on
    /learned_trajectory, so this function converts the active Cartesian path into a
    temporary CSV and reuses scripts/vulcanexus/docker_publish_traj.sh.
    """
    script_path = _repo_root() / "scripts" / "vulcanexus" / "docker_publish_traj.sh"
    if not script_path.is_file():
        return False, f"Vulcanexus publisher script not found: {script_path}"

    runtime_dir = _repo_root() / "data" / "_runtime" / "vulcanexus"
    csv_path = runtime_dir / "last_cartesian_push.csv"

    try:
        arr = _write_cartesian_csv(csv_path, cart_path)
    except Exception as exc:
        return False, f"Could not prepare Cartesian trajectory for Vulcanexus publish: {exc}"

    env = os.environ.copy()
    env["HOST_WORKSPACE"] = str(_repo_root())
    env["CSV_RELATIVE_PATH"] = str(csv_path.relative_to(_repo_root()))
    env["TOPIC"] = topic_name
    env["FRAME_ID"] = frame_id
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["RMW_IMPLEMENTATION"] = env.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    env["ROS_LOCALHOST_ONLY"] = env.get("ROS_LOCALHOST_ONLY", "0")
    env["REPEAT"] = str(repeat)
    env["RATE_HZ"] = str(rate_hz)
    env["WAIT_FOR_SUBSCRIBER_SEC"] = str(wait_for_subscriber_sec)
    if discovery_server:
        env["ROS_DISCOVERY_SERVER"] = discovery_server
    if container_name:
        env["VULCANEXUS_CONTAINER"] = container_name

    status_waiter = None
    if status_topic and status_timeout_sec > 0 and has_vulcanexus_status_subscriber():
        status_waiter, waiter_error = _start_vulcanexus_status_waiter(
            topic_name=status_topic,
            domain_id=domain_id,
            discovery_server=discovery_server,
            timeout_sec=status_timeout_sec,
            container_name=container_name,
        )
        if waiter_error:
            status_waiter = None
            logger.warning(waiter_error)

    try:
        result = subprocess.run(
            [str(script_path)],
            cwd=str(_repo_root()),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in [exc.stdout, exc.stderr] if part).strip()
        return False, f"Vulcanexus publish failed: {details or exc}"

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else "Publish completed."
    logger.info(
        "Published Cartesian trajectory via Vulcanexus: %s points on %s (domain %s)",
        arr.shape[0],
        topic_name,
        domain_id,
    )
    msg = (
        f"Published {arr.shape[0]} Cartesian waypoints via Vulcanexus LAN on "
        f"'{topic_name}' (domain {domain_id}). {summary}"
    )
    if status_waiter is not None:
        ok, status_msg, status_payload = _finish_vulcanexus_status_waiter(status_waiter)
        if ok:
            state = status_payload.get("state") if isinstance(status_payload, dict) else None
            if state:
                msg += f" | Edge status: {state}"
            else:
                msg += f" | Edge status: {status_msg}"
        else:
            msg += f" | Edge status unavailable: {status_msg}"
    return True, msg


def publish_trajectory_dds(
    joint_names: List[str],
    q_traj,
    timestamps: List[float],
    topic_name: str = "/joint_trajectory",
    domain_id: int = 0,
) -> Tuple[bool, str]:
    """
    Publish a joint trajectory over DDS using cyclonedds.

    The message structure mirrors ROS2 trajectory_msgs/JointTrajectory:
      - joint_names: [str]
      - points: [{positions: [float], time_from_start: {sec, nanosec}}]

    Args:
        joint_names: Ordered list of joint names.
        q_traj: ndarray (N x num_joints) or list of lists.
        timestamps: List of float seconds per frame.
        topic_name: DDS topic name.
        domain_id: DDS domain ID.

    Returns:
        (success, message)
    """
    try:
        from cyclonedds.core import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic
        from cyclonedds.idl import IdlStruct
        from cyclonedds.idl.types import sequence, float64
        from dataclasses import dataclass
    except ImportError:
        return False, (
            "cyclonedds is not installed. "
            "Install it with: pip install cyclonedds"
        )

    # --- Define IDL-compatible types ---
    @dataclass
    class Duration(IdlStruct, typename="builtin_interfaces.msg.Duration"):
        sec: int = 0
        nanosec: int = 0

    @dataclass
    class JointTrajectoryPoint(IdlStruct, typename="trajectory_msgs.msg.JointTrajectoryPoint"):
        positions: sequence[float64]
        velocities: sequence[float64]
        accelerations: sequence[float64]
        effort: sequence[float64]
        time_from_start: Duration = Duration()

    @dataclass
    class JointTrajectory(IdlStruct, typename="trajectory_msgs.msg.JointTrajectory"):
        joint_names: sequence[str]
        points: sequence[JointTrajectoryPoint]

    try:
        arr = np.asarray(q_traj)  # (N, J)
        if arr.ndim != 2:
            return False, f"q_traj shape invalid: {arr.shape}"

        # Build points
        points = []
        t0 = timestamps[0] if timestamps else 0.0
        for t, q in zip(timestamps, arr):
            dt = t - t0
            sec = int(dt)
            nsec = int((dt - sec) * 1e9)
            points.append(
                JointTrajectoryPoint(
                    positions=q.tolist(),
                    velocities=[],
                    accelerations=[],
                    effort=[],
                    time_from_start=Duration(sec=sec, nanosec=nsec),
                )
            )

        msg = JointTrajectory(
            joint_names=list(joint_names),
            points=points,
        )

        # Publish
        dp = DomainParticipant(domain_id=domain_id)
        tp = Topic(dp, topic_name, JointTrajectory)
        writer = DataWriter(dp, tp)

        writer.write(msg)
        logger.info(
            f"Published JointTrajectory on '{topic_name}' (domain {domain_id}): "
            f"{len(points)} points, {len(joint_names)} joints"
        )

        return True, (
            f"Published {len(points)} trajectory points for "
            f"{len(joint_names)} joints on topic '{topic_name}' (domain {domain_id})"
        )

    except Exception as e:
        logger.exception("DDS publish failed")
        return False, f"Publish failed: {e}"
