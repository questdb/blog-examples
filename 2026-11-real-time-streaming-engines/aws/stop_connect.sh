#!/usr/bin/env bash
# QuestDB host: stop the Kafka Connect worker started by aws/start_connect.sh <table>.
#
# Usage: aws/stop_connect.sh <table>
# Settings (environment or .env): CONNECT_DIR (default /data/connect).
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
set -euo pipefail
CONNECT_DIR=${CONNECT_DIR:-/data/connect}
pidfile="$CONNECT_DIR/$1/connect.pid"
pid=$(cat "$pidfile")
kill "$pid"
for _ in $(seq 30); do
  if ! kill -0 "$pid" 2>/dev/null; then rm -f "$pidfile"; echo "stopped connect worker $pid"; exit 0; fi
  sleep 1
done
echo "connect worker $pid still running"; exit 1
