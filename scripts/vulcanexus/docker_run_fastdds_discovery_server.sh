#!/usr/bin/env bash

set -euo pipefail

CONTAINER="${VULCANEXUS_CONTAINER:-vulcanexus_humble}"
SERVER_ID="${FASTDDS_SERVER_ID:-0}"
UDP_ADDRESS="${FASTDDS_UDP_ADDRESS:-}"
UDP_PORT="${FASTDDS_UDP_PORT:-}"
TCP_ADDRESS="${FASTDDS_TCP_ADDRESS:-}"
TCP_PORT="${FASTDDS_TCP_PORT:-}"
BACKUP="${FASTDDS_BACKUP:-0}"
FORCE_RESTART="${FASTDDS_FORCE_RESTART:-0}"

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  echo "Starting container $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi

EXISTING_SERVER="$(docker exec -i "$CONTAINER" bash -lc "ps -ef | grep '[f]astdds discovery' || true")"
if [ -n "$EXISTING_SERVER" ] && [ "$FORCE_RESTART" != "1" ]; then
  echo "Fast DDS Discovery Server is already running in $CONTAINER"
  echo "$EXISTING_SERVER"
  echo "Reuse the existing server, or set FASTDDS_FORCE_RESTART=1 to replace it."
  exit 0
fi

if [ -n "$EXISTING_SERVER" ] && [ "$FORCE_RESTART" = "1" ]; then
  echo "Stopping existing Fast DDS Discovery Server in $CONTAINER"
  docker exec -i "$CONTAINER" bash -lc "
PIDS=\$(ps -ef | awk '/[f]astdds discovery/ {print \$2}')
if [ -n \"\$PIDS\" ]; then
  kill \$PIDS
fi
" >/dev/null
fi

echo "Running Fast DDS Discovery Server in $CONTAINER"
echo "FASTDDS_SERVER_ID=$SERVER_ID"
if [ -n "$UDP_ADDRESS" ]; then
  echo "FASTDDS_UDP_ADDRESS=$UDP_ADDRESS"
fi
if [ -n "$UDP_PORT" ]; then
  echo "FASTDDS_UDP_PORT=$UDP_PORT"
else
  echo "FASTDDS_UDP_PORT=11811"
fi
if [ -n "$TCP_ADDRESS" ]; then
  echo "FASTDDS_TCP_ADDRESS=$TCP_ADDRESS"
fi
if [ -n "$TCP_PORT" ]; then
  echo "FASTDDS_TCP_PORT=$TCP_PORT"
fi

DISCOVERY_ARGS=(-i "$SERVER_ID")
if [ -n "$UDP_ADDRESS" ]; then
  DISCOVERY_ARGS+=(-l "$UDP_ADDRESS")
fi
if [ -n "$UDP_PORT" ]; then
  DISCOVERY_ARGS+=(-p "$UDP_PORT")
fi
if [ -n "$TCP_ADDRESS" ]; then
  DISCOVERY_ARGS+=(-t "$TCP_ADDRESS")
fi
if [ -n "$TCP_PORT" ]; then
  DISCOVERY_ARGS+=(-q "$TCP_PORT")
fi
if [ "$BACKUP" = "1" ]; then
  DISCOVERY_ARGS+=(-b)
fi

printf -v DISCOVERY_ARGS_STRING ' %q' "${DISCOVERY_ARGS[@]}"

echo "Point ROS_DISCOVERY_SERVER at <server-ip>:${UDP_PORT:-11811}"

docker exec -i "$CONTAINER" bash -lc "
source /opt/vulcanexus/humble/setup.bash
fastdds discovery${DISCOVERY_ARGS_STRING}
"
