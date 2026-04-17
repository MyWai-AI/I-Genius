#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/edge_receiver_output"

ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
TOPIC="${TOPIC:-/learned_trajectory}"
STATUS_TOPIC="${STATUS_TOPIC:-/trajectory_status}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-latest_received}"
UNITS="${UNITS:-m}"
FRAME_ID_OVERRIDE="${FRAME_ID_OVERRIDE:-}"
BACKEND_SCRIPT="${BACKEND_SCRIPT:-}"
RECEIVER_SCRIPT="${RECEIVER_SCRIPT:-$SCRIPT_DIR/edge_receive_posearray.py}"

echo "ROS_SETUP_FILE=$ROS_SETUP_FILE"
echo "TOPIC=$TOPIC"
echo "STATUS_TOPIC=$STATUS_TOPIC"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "ARTIFACT_PREFIX=$ARTIFACT_PREFIX"
echo "UNITS=$UNITS"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}"
if [ -n "${ROS_DISCOVERY_SERVER:-}" ]; then
  echo "ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
fi
if [ -n "$FRAME_ID_OVERRIDE" ]; then
  echo "FRAME_ID_OVERRIDE=$FRAME_ID_OVERRIDE"
fi
if [ -n "$BACKEND_SCRIPT" ]; then
  echo "BACKEND_SCRIPT=$BACKEND_SCRIPT"
fi

ARGS=(
  --topic "$TOPIC"
  --status-topic "$STATUS_TOPIC"
  --output-dir "$OUTPUT_DIR"
  --artifact-prefix "$ARTIFACT_PREFIX"
  --units "$UNITS"
)

if [ -n "$FRAME_ID_OVERRIDE" ]; then
  ARGS+=(--frame-id-override "$FRAME_ID_OVERRIDE")
fi

if [ -n "$BACKEND_SCRIPT" ]; then
  ARGS+=(--backend-script "$BACKEND_SCRIPT")
fi

set +u
source "$ROS_SETUP_FILE"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

python3 "$RECEIVER_SCRIPT" "${ARGS[@]}"
