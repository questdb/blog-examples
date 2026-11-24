#!/usr/bin/env bash
# One benchmark run from the producer host: fresh table with the given view flags, the
# producer at RATE evt/s, optional query timings mid-run, then latency analysis.
#
# Usage: aws/run.sh <table> <batch> <seconds> <query_delay_s|0> [mktable flags...]
#   query_delay_s  start qtime.py this many seconds into the run (0 = no query timings)
#   mktable flags  any of --clean-mv, --clean-view, --s1-mv
# Settings (environment or .env): QDB_ADDR (required), RATE (default 50000),
#   DEDUP (SKIP or UPSERT, default SKIP), SF_DIR (enables store-and-forward when set).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
: "${QDB_ADDR:?set QDB_ADDR to the QuestDB host:port}"
RATE=${RATE:-50000}
DEDUP=${DEDUP:-SKIP}
cd "$ROOT"

t=$1; b=$2; s=$3; qd=$4; shift 4
sf=()
if [ -n "${SF_DIR:-}" ]; then sf=(--sf-dir "$SF_DIR"); fi

"$PY" mktable.py "$t" --dedup "$DEDUP" "$@"
mkdir -p runs
if [ "$qd" != "0" ]; then
  "$PY" qtime.py "$t" "now()" 5 30 "$qd" > "runs/${t}_qtime.txt" 2>&1 &
fi
"$PY" producer.py --addr "$QDB_ADDR" --batch "$b" --rate "$RATE" --table "$t" \
  --seconds "$s" --run-id "$t" --quiet ${sf[@]+"${sf[@]}"}
wait

last=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['last_event_id'])" "runs/$t.json")
"$PY" -c "from common import sql; import sys; t = sys.argv[1]; print('rows, distinct ids:', sql(f'SELECT count(), count_distinct(event_id) FROM {t}')['dataset'][0])" "$t"
"$PY" latency.py "$t" 1 "$last"
for flag in "$@"; do
  case "$flag" in
    --clean-mv) "$PY" latency.py "${t}_clean_mv" 1 "$last" ;;
    --s1-mv) "$PY" view_latency.py "$t" "${t}_s1_10s" ;;
  esac
done
if [ "$qd" != "0" ]; then cat "runs/${t}_qtime.txt"; fi
