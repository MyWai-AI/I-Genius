#!/usr/bin/env bash

set -u

run_or_note() {
  local label="$1"
  shift

  echo "$label"
  if "$@"; then
    :
  else
    echo "<not available in this environment>"
  fi
  echo
}

PRIMARY_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for (i = 1; i <= NF; ++i) if ($i == "src") {print $(i+1); exit}}' || true)"

run_or_note "Hostname:" hostname

echo "Primary IPv4 used for outbound traffic:"
if [ -n "$PRIMARY_IP" ]; then
  echo "$PRIMARY_IP"
else
  echo "<not found>"
fi
echo

run_or_note "All IPv4 addresses on UP interfaces:" ip -brief -4 addr show up
run_or_note "Default route:" sh -c "ip route | awk '\$1 == \"default\" {print}'"
run_or_note "hostname -I:" hostname -I
