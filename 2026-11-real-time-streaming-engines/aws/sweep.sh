#!/usr/bin/env bash
# Batch-size sweep from the producer host: a fresh base table per batch size, RUN_SECONDS
# each with 10 s idle between runs, then latency for every run.
#
# Usage: aws/sweep.sh <table_prefix> <batch> [<batch> ...]
# Settings (environment or .env): QDB_ADDR (required), RATE (default 50000),
#   RUN_SECONDS (default 30), DEDUP (default SKIP).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
: "${QDB_ADDR:?set QDB_ADDR to the QuestDB host:port}"
RATE=${RATE:-50000}
RUN_SECONDS=${RUN_SECONDS:-30}
DEDUP=${DEDUP:-SKIP}
cd "$ROOT"

prefix=$1; shift
for b in "$@"; do
  t="${prefix}_b${b}"
  "$PY" mktable.py "$t" --dedup "$DEDUP"
  "$PY" producer.py --addr "$QDB_ADDR" --batch "$b" --rate "$RATE" --table "$t" \
    --seconds "$RUN_SECONDS" --run-id "$t" --quiet
  sleep 10
done
for b in "$@"; do
  t="${prefix}_b${b}"
  last=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['last_event_id'])" "runs/$t.json")
  "$PY" latency.py "$t" 1 "$last"
done
