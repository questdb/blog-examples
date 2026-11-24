#!/usr/bin/env bash
# Producer host: run the original benchmark's producer (through upstream_producer.py, which
# only re-encodes ts as integer microseconds) into <topic>, as
# PRODUCERS processes at RATE/PRODUCERS evt/s each (the article ran 8 pods at 6,250).
# Each process gets its own seed and a disjoint event_id range. Waits for all of them and
# writes each one's log to runs/<topic>_p<i>.log.
#
# Usage: aws/kafka_produce.sh <topic> <seconds> <dup_rate> [extra producer.py args...]
#   extra args go to every producer, e.g. --payments-topic <t> for S5 (then <topic> gets
#   the orders and RATE is orders/s)
# Settings (environment or .env): KAFKA_BOOTSTRAP (required), RATE (default 50000),
#   PRODUCERS (default 8), START_DELAY (seconds to wait first, default 0), SEED_OFFSET
#   (shifts seeds, id ranges and log names, so a second group of producers can join a
#   running one, default 0).
# Needs aws/fetch_upstream.sh first, and confluent-kafka in the venv.
# Order for a run: mktable.py <table>, mktopic.py <topic> (before Connect subscribes, or
# the broker auto-creates it with one partition), aws/start_connect.sh, then this script.
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
set -euo pipefail
: "${KAFKA_BOOTSTRAP:?set KAFKA_BOOTSTRAP to the broker host:port}"
RATE=${RATE:-50000}
PRODUCERS=${PRODUCERS:-8}
START_DELAY=${START_DELAY:-0}
SEED_OFFSET=${SEED_OFFSET:-0}
cd "$ROOT"
topic=$1; seconds=$2; dup=$3; shift 3
mkdir -p runs

sleep "$START_DELAY"
pids=()
for i in $(seq "$SEED_OFFSET" $((SEED_OFFSET + PRODUCERS - 1))); do
  "$PY" upstream_producer.py --bootstrap "$KAFKA_BOOTSTRAP" --topic "$topic" \
    --rate $((RATE / PRODUCERS)) --seconds "$seconds" --seed "$((42 + i))" --keys 1000000 \
    --dup-rate "$dup" --id-base "$((i * 1000000000000))" "$@" > "runs/${topic}_p$i.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for i in $(seq "$SEED_OFFSET" $((SEED_OFFSET + PRODUCERS - 1))); do
  grep -h DONE "runs/${topic}_p$i.log"
done
exit $status
