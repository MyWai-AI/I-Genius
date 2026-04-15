#!/usr/bin/env bash

set -euo pipefail

CONTAINER="${DDSROUTER_CONTAINER:-vulcanexus_humble}"
HOST_CONFIG_PATH="${HOST_CONFIG_PATH:-/tmp/ddsrouter_cloud.yaml}"
CONTAINER_CONFIG_PATH="${CONTAINER_CONFIG_PATH:-/workspace/vilma-agent/.tmp/ddsrouter.yaml}"
SETUP_FILE="${ROS_SETUP_FILE:-/opt/vulcanexus/humble/setup.bash}"

if [ ! -f "$HOST_CONFIG_PATH" ]; then
  echo "DDS Router config not found on host: $HOST_CONFIG_PATH" >&2
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  echo "Starting container $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi

CONTAINER_CONFIG_DIR="$(dirname "$CONTAINER_CONFIG_PATH")"

docker exec -i "$CONTAINER" mkdir -p "$CONTAINER_CONFIG_DIR"
docker cp "$HOST_CONFIG_PATH" "$CONTAINER:$CONTAINER_CONFIG_PATH"

echo "Container: $CONTAINER"
echo "DDS Router config: $CONTAINER_CONFIG_PATH"
echo "ROS setup: $SETUP_FILE"

docker exec -it "$CONTAINER" bash -lc "
source '$SETUP_FILE'
ddsrouter -c '$CONTAINER_CONFIG_PATH'
"
