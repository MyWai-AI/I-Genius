#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/edge_executor_output"

ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
TOPIC="${TOPIC:-/learned_trajectory}"
STATUS_TOPIC="${STATUS_TOPIC:-/trajectory_status}"
UNIT_SCALE="${UNIT_SCALE:-1000.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
TRANSFORM_PATH="${TRANSFORM_PATH:-}"
EXECUTOR_SCRIPT="${EXECUTOR_SCRIPT:-$SCRIPT_DIR/traj_pose_array_executor.py}"

echo "ROS_SETUP_FILE=$ROS_SETUP_FILE"
echo "TOPIC=$TOPIC"
echo "STATUS_TOPIC=$STATUS_TOPIC"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}"
if [ -n "${ROS_DISCOVERY_SERVER:-}" ]; then
  echo "ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
fi
if [ -n "$TRANSFORM_PATH" ]; then
  echo "TRANSFORM_PATH=$TRANSFORM_PATH"
fi

ARGS=(
  --topic "$TOPIC"
  --status-topic "$STATUS_TOPIC"
  --unit-scale "$UNIT_SCALE"
  --output-dir "$OUTPUT_DIR"
  --dry-run
)

if [ -n "$TRANSFORM_PATH" ]; then
  ARGS+=(--transform-path "$TRANSFORM_PATH")
fi

set +u
source "$ROS_SETUP_FILE"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

python3 "$EXECUTOR_SCRIPT" "${ARGS[@]}"
