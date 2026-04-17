#!/usr/bin/env python3

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe to a PoseArray trajectory, convert it into robot-frame "
            "waypoints, and publish execution status updates."
        )
    )
    parser.add_argument("--topic", default="/learned_trajectory")
    parser.add_argument("--status-topic", default="/trajectory_status")
    parser.add_argument("--node-name", default="traj_pose_array_executor")
    parser.add_argument(
        "--transform-path",
        default="",
        help="Optional 4x4 transform file (.npy, .npz, .json) from source frame to robot frame.",
    )
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1000.0,
        help="Scale factor after transform; use 1000.0 for robot millimeters.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where dry-run exports are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export converted waypoints instead of commanding a robot.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the first received trajectory and then exit.",
    )
    return parser.parse_args()


def load_transform(transform_path: str) -> np.ndarray:
    if not transform_path:
        return np.eye(4, dtype=float)

    path = Path(transform_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Transform file not found: {path}")

    if path.suffix == ".npy":
        matrix = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        for key in ("transform", "matrix", "T"):
            if key in archive:
                matrix = archive[key]
                break
        else:
            raise ValueError(f"No transform key found in {path}; expected transform/matrix/T.")
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrix = payload.get("transform") or payload.get("matrix") or payload.get("T")
        if matrix is None:
            raise ValueError(f"No transform key found in {path}; expected transform/matrix/T.")
    else:
        raise ValueError(f"Unsupported transform format: {path.suffix}")

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Transform must have shape (4, 4); got {matrix.shape!r}")
    return matrix


def pose_array_to_numpy(msg: PoseArray) -> np.ndarray:
    if not msg.poses:
        raise ValueError("Received PoseArray with no poses.")
    return np.asarray(
        [[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses],
        dtype=float,
    )


def transform_points(points: np.ndarray, transform: np.ndarray, unit_scale: float) -> np.ndarray:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=float)], axis=1)
    transformed = (transform @ hom.T).T[:, :3]
    return transformed * float(unit_scale)


def save_waypoints_csv(csv_path: Path, points: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points.tolist())


@dataclass
class ExecutorConfig:
    node_name: str
    source_topic: str
    status_topic: str
    transform: np.ndarray
    unit_scale: float
    output_dir: Path
    dry_run: bool
    once: bool
    transform_path: str


class TrajectoryExecutor(Node):
    def __init__(self, config: ExecutorConfig) -> None:
        super().__init__(config.node_name)
        self._config = config
        self._status_pub = self.create_publisher(String, config.status_topic, 10)
        self._subscription = self.create_subscription(
            PoseArray, config.source_topic, self._callback, 10
        )
        self._processed_messages = 0
        self.get_logger().info(
            f"Waiting for PoseArray messages on {config.source_topic}; "
            f"status topic is {config.status_topic}"
        )

    def _publish_status(self, state: str, **extra) -> None:
        payload = {
            "state": state,
            "source_topic": self._config.source_topic,
            "status_topic": self._config.status_topic,
            "dry_run": self._config.dry_run,
            "unit_scale": self._config.unit_scale,
        }
        payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self._status_pub.publish(msg)
        self.get_logger().info(f"STATUS {state}: {msg.data}")

    def _callback(self, msg: PoseArray) -> None:
        self._processed_messages += 1
        try:
            source_points = pose_array_to_numpy(msg)
            self._publish_status(
                "received",
                points=int(source_points.shape[0]),
                source_frame=msg.header.frame_id,
            )
            robot_points = transform_points(
                source_points, self._config.transform, self._config.unit_scale
            )
            output_csv = self._config.output_dir / "latest_robot_waypoints.csv"
            output_json = self._config.output_dir / "latest_robot_waypoints.json"
            output_meta = self._config.output_dir / "latest_robot_waypoints_meta.json"
            save_waypoints_csv(output_csv, robot_points)
            output_json.write_text(
                json.dumps(
                    {
                        "frame_id": msg.header.frame_id,
                        "unit_scale": self._config.unit_scale,
                        "points": robot_points.tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            output_meta.write_text(
                json.dumps(
                    {
                        "source_frame": msg.header.frame_id,
                        "transform_path": self._config.transform_path,
                        "source_point_count": int(source_points.shape[0]),
                        "robot_min": robot_points.min(axis=0).tolist(),
                        "robot_max": robot_points.max(axis=0).tolist(),
                        "dry_run": self._config.dry_run,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._publish_status(
                "prepared",
                output_csv=str(output_csv),
                first_point=robot_points[0].tolist(),
                last_point=robot_points[-1].tolist(),
            )

            if self._config.dry_run:
                self._publish_status(
                    "completed",
                    mode="dry_run",
                    output_csv=str(output_csv),
                    output_json=str(output_json),
                    output_meta=str(output_meta),
                )
            else:
                self._publish_status(
                    "failed",
                    reason="Robot command integration not implemented yet; rerun with --dry-run.",
                )

        except Exception as exc:
            self._publish_status("failed", reason=str(exc))

        if self._config.once and self._processed_messages >= 1:
            self.get_logger().info("Processed one trajectory; shutting down.")
            rclpy.shutdown()


def main() -> int:
    args = parse_args()
    transform = load_transform(args.transform_path)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = TrajectoryExecutor(
        ExecutorConfig(
            node_name=args.node_name,
            source_topic=args.topic,
            status_topic=args.status_topic,
            transform=transform,
            unit_scale=args.unit_scale,
            output_dir=output_dir,
            dry_run=args.dry_run,
            once=args.once,
            transform_path=args.transform_path,
        )
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
