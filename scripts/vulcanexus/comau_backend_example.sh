#!/usr/bin/env bash

set -euo pipefail

echo "Comau backend example invoked."
echo "TRAJECTORY_CSV=${TRAJECTORY_CSV:-}"
echo "TRAJECTORY_METADATA_JSON=${TRAJECTORY_METADATA_JSON:-}"
echo "TRAJECTORY_FRAME_ID=${TRAJECTORY_FRAME_ID:-}"
echo "TRAJECTORY_UNITS=${TRAJECTORY_UNITS:-}"
echo "TRAJECTORY_POINT_COUNT=${TRAJECTORY_POINT_COUNT:-}"

cat <<'EOF'
This example script is only a backend contract placeholder.

For the Ubuntu 20.04 + ROS1 Noetic + ROS2 Humble container setup, the intended
production pattern is:

1. ROS2 Humble edge receiver writes the trajectory artifact into a shared host folder.
2. A host-side Comau adapter validates/remaps that artifact if needed.
3. The existing Noetic Comau driver / replay path executes the validated result.

Do not put Comau-specific execution logic into vilma-agent.
Keep it in the robot-side backend.
EOF
