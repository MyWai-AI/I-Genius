#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive a PoseArray trajectory on the edge, persist a normalized "
            "artifact, and optionally trigger a robot-specific backend."
        )
    )
    parser.add_argument("--topic", default="/learned_trajectory")
    parser.add_argument("--status-topic", default="/trajectory_status")
    parser.add_argument("--node-name", default="edge_receive_posearray")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the normalized trajectory artifacts are written.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="latest_received",
        help="Prefix used for the CSV/JSON artifact file names.",
    )
    parser.add_argument(
        "--units",
        default="m",
        help="Units label for the received trajectory. Defaults to meters.",
    )
    parser.add_argument(
        "--frame-id-override",
        default="",
        help="Optional frame_id to write into metadata when the incoming header is empty.",
    )
    parser.add_argument(
        "--backend-script",
        default="",
        help=(
            "Optional executable to invoke after persisting the artifact. "
            "The script receives artifact metadata via environment variables."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Persist the first received trajectory and then exit.",
    )
    return parser.parse_args()


def pose_array_to_rows(msg: PoseArray) -> list[list[float]]:
    if not msg.poses:
        raise ValueError("Received PoseArray with no poses.")
    return [
        [pose.position.x, pose.position.y, pose.position.z]
        for pose in msg.poses
    ]


def write_csv(csv_path: Path, rows: list[list[float]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z"])
        writer.writerows(rows)


class EdgeReceiver(Node):
    def __init__(
        self,
        topic: str,
        status_topic: str,
        output_dir: Path,
        artifact_prefix: str,
        units: str,
        frame_id_override: str,
        backend_script: str,
        once: bool,
        node_name: str,
    ) -> None:
        super().__init__(node_name)
        self._topic = topic
        self._status_topic = status_topic
        self._output_dir = output_dir
        self._artifact_prefix = artifact_prefix
        self._units = units
        self._frame_id_override = frame_id_override
        self._backend_script = backend_script
        self._once = once
        self._processed_messages = 0

        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._subscription = self.create_subscription(
            PoseArray, topic, self._callback, 10
        )
        self.get_logger().info(
            f"Waiting for PoseArray messages on {topic}; "
            f"artifacts will be written under {output_dir}"
        )

    def _publish_status(self, state: str, **extra) -> None:
        payload = {
            "state": state,
            "source_topic": self._topic,
            "status_topic": self._status_topic,
            "artifact_prefix": self._artifact_prefix,
            "units": self._units,
        }
        payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self._status_pub.publish(msg)
        self.get_logger().info(f"STATUS {state}: {msg.data}")

    def _run_backend(
        self,
        csv_path: Path,
        metadata_path: Path,
        frame_id: str,
        point_count: int,
    ) -> Optional[subprocess.CompletedProcess]:
        if not self._backend_script:
            return None

        env = dict(
            TRAJECTORY_CSV=str(csv_path),
            TRAJECTORY_METADATA_JSON=str(metadata_path),
            TRAJECTORY_FRAME_ID=frame_id,
            TRAJECTORY_UNITS=self._units,
            TRAJECTORY_POINT_COUNT=str(point_count),
            TRAJECTORY_ARTIFACT_PREFIX=self._artifact_prefix,
        )

        return subprocess.run(
            [self._backend_script],
            check=True,
            cwd=str(self._output_dir),
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )

    def _callback(self, msg: PoseArray) -> None:
        self._processed_messages += 1
        try:
            rows = pose_array_to_rows(msg)
            point_count = len(rows)
            frame_id = msg.header.frame_id or self._frame_id_override or "unknown_frame"
            self._publish_status(
                "received",
                frame_id=frame_id,
                points=point_count,
            )

            prefix = self._artifact_prefix
            csv_path = self._output_dir / f"{prefix}_trajectory.csv"
            metadata_path = self._output_dir / f"{prefix}_metadata.json"

            write_csv(csv_path, rows)

            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "frame_id": frame_id,
                "units": self._units,
                "point_count": point_count,
                "source_topic": self._topic,
                "status_topic": self._status_topic,
                "artifact_prefix": prefix,
                "first_point": rows[0],
                "last_point": rows[-1],
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            self._publish_status(
                "persisted",
                csv_path=str(csv_path),
                metadata_path=str(metadata_path),
                first_point=rows[0],
                last_point=rows[-1],
            )

            result = self._run_backend(csv_path, metadata_path, frame_id, point_count)
            if result is not None:
                self._publish_status(
                    "backend_completed",
                    backend_script=self._backend_script,
                    backend_returncode=result.returncode,
                    backend_stdout=result.stdout[-1000:],
                    backend_stderr=result.stderr[-1000:],
                )
            else:
                self._publish_status(
                    "completed",
                    mode="artifact_only",
                    csv_path=str(csv_path),
                    metadata_path=str(metadata_path),
                )

        except subprocess.CalledProcessError as exc:
            self._publish_status(
                "backend_failed",
                backend_script=self._backend_script,
                backend_returncode=exc.returncode,
                backend_stdout=(exc.stdout or "")[-1000:],
                backend_stderr=(exc.stderr or "")[-1000:],
            )
        except Exception as exc:
            self._publish_status("failed", reason=str(exc))

        if self._once and self._processed_messages >= 1:
            self.get_logger().info("Processed one trajectory; shutting down.")
            rclpy.shutdown()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = EdgeReceiver(
        topic=args.topic,
        status_topic=args.status_topic,
        output_dir=output_dir,
        artifact_prefix=args.artifact_prefix,
        units=args.units,
        frame_id_override=args.frame_id_override,
        backend_script=args.backend_script,
        once=args.once,
        node_name=args.node_name,
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
