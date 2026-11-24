#!/usr/bin/env bash
# Recovery test, server side: wait, kill -9 QuestDB mid-stream, wait, restart it with the
# given config. Prints UTC timestamps of each step.
#
# Usage: aws/chaos.sh <wait_s> <down_s> [server.conf]
#   server.conf  passed to aws/start_qdb.sh (same QDB_* settings apply)
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
sleep "$1"
pid=$(pgrep -f io.questdb.ServerMain)
echo "kill -9 $pid at $(date -u +%H:%M:%S.%N)"
kill -9 "$pid"
sleep "$2"
echo "restart at $(date -u +%H:%M:%S.%N)"
bash "$ROOT/aws/start_qdb.sh" "${3:-}"
echo "up at $(date -u +%H:%M:%S.%N)"
