#!/usr/bin/env bash
# Start QuestDB on the server host and wait until it answers queries.
#
# Usage: aws/start_qdb.sh [server.conf]
#   server.conf  installed as QDB_ROOT/conf/server.conf before starting; a bare file name
#                is looked up in aws/ (for example pools_nap.conf). Without it, QuestDB
#                starts with whatever config QDB_ROOT already has (defaults on first start).
# Settings (environment or .env): QDB_HOME (default /opt/questdb), QDB_ROOT (default
#   /data/qdb), QDB_CPUS (taskset CPU list, for example 1-7; empty = no pinning).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
QDB_HOME=${QDB_HOME:-/opt/questdb}
QDB_ROOT=${QDB_ROOT:-/data/qdb}
QDB_CPUS=${QDB_CPUS:-}

conf=${1:-}
if [ -n "$conf" ] && [ ! -f "$conf" ]; then conf="$ROOT/aws/$conf"; fi
mkdir -p "$QDB_ROOT/conf"
if [ -n "$conf" ]; then cp "$conf" "$QDB_ROOT/conf/server.conf"; fi

ulimit -n 1048576
pin=()
if [ -n "$QDB_CPUS" ]; then pin=(taskset -c "$QDB_CPUS"); fi
${pin[@]+"${pin[@]}"} "$QDB_HOME/bin/questdb.sh" start -d "$QDB_ROOT"
curl -fsS --retry 60 --retry-delay 1 --retry-all-errors \
  "http://localhost:9000/exec?query=SELECT%20build%28%29"
echo
