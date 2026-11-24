#!/usr/bin/env bash
# QuestDB host: start a standalone Kafka Connect worker running the QuestDB sink from
# <topic> into <table> (create the table first, e.g. mktable.py, so DEDUP is set), pinned
# to the same CPUs as QuestDB. Writes its config, offsets, log and PID under
# CONNECT_DIR/<table>/. Stop it with aws/stop_connect.sh <table>.
#
# Usage: aws/start_connect.sh <table> <topic>
#        aws/start_connect.sh <name> <topic>,<topic>... --per-topic
#   --per-topic  write each topic into the table of the same name (e.g. S5 orders and
#                payments); every one of those tables must exist; <name> names the worker
# Settings (environment or .env): KAFKA_BOOTSTRAP (required, host:9092), CONNECT_TASKS
#   (default 6, one per partition), CONNECT_BATCH (rows per QWP flush, default 2000),
#   CONNECT_CONF_EXTRA (extra client connect-string keys, default none),
#   CONNECT_DIR (default /data/connect), KAFKA_HOME, QDB_CPUS (as for start_qdb.sh).
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
set -euo pipefail
: "${KAFKA_BOOTSTRAP:?set KAFKA_BOOTSTRAP to the broker host:port}"
CONNECT_TASKS=${CONNECT_TASKS:-6}
CONNECT_BATCH=${CONNECT_BATCH:-2000}
CONNECT_CONF_EXTRA=${CONNECT_CONF_EXTRA:-}   # extra client keys, e.g. reconnect_max_backoff_millis=500;
CONNECT_DIR=${CONNECT_DIR:-/data/connect}
KAFKA_HOME=${KAFKA_HOME:-/opt/kafka}
QDB_CPUS=${QDB_CPUS:-}
table=$1; topic=$2
dir="$CONNECT_DIR/$table"
target=$table
required=$table
if [ "${3:-}" = "--per-topic" ]; then
  # shellcheck disable=SC2016  # literal: the connector expands ${topic} per record
  target='${topic}'   # one table per topic, named after it
  required=${topic//,/ }
fi

# The connector would create a missing table with defaults (no DEDUP, day partitions,
# inferred types); require the ones mktable.py / s5producer.py make so their settings are kept.
for t in $required; do
  exists=$(cd "$ROOT" && QDB_ADDR=localhost:9000 python3 -c \
    "import sys; from common import sql; print(sql(f\"SELECT count() FROM tables() WHERE table_name = '{sys.argv[1]}'\")['dataset'][0][0])" "$t")
  if [ "$exists" != "1" ]; then
    echo "table $t does not exist; create it first so its partitioning and DEDUP are used"
    exit 1
  fi
done
mkdir -p "$dir"

cat > "$dir/worker.properties" <<EOF
bootstrap.servers=$KAFKA_BOOTSTRAP
key.converter=org.apache.kafka.connect.storage.StringConverter
value.converter=org.apache.kafka.connect.converters.ByteArrayConverter
offset.storage.file.filename=$dir/connect.offsets
offset.flush.interval.ms=1000
EOF

cat > "$dir/questdb-sink.properties" <<EOF
name=questdb-$table
connector.class=io.questdb.kafka.QuestDBSinkConnector
tasks.max=$CONNECT_TASKS
topics=$topic
table=$target
client.conf.string=ws::addr=localhost:9000;auto_flush_rows=$CONNECT_BATCH;auto_flush_interval=1000;$CONNECT_CONF_EXTRA
key.converter=org.apache.kafka.connect.storage.StringConverter
value.converter=org.apache.kafka.connect.converters.ByteArrayConverter
value.format=json
include.key=false
symbols=event_type,country
doubles=amount
timestamp.field.name=ts
timestamp.units=micros
EOF

pin=()
if [ -n "$QDB_CPUS" ]; then pin=(taskset -c "$QDB_CPUS"); fi
nohup ${pin[@]+"${pin[@]}"} "$KAFKA_HOME/bin/connect-standalone.sh" \
  "$dir/worker.properties" "$dir/questdb-sink.properties" > "$dir/connect.log" 2>&1 &
echo $! > "$dir/connect.pid"
echo "connect worker pid $(cat "$dir/connect.pid"): $topic -> $table, $CONNECT_TASKS tasks, log $dir/connect.log"
