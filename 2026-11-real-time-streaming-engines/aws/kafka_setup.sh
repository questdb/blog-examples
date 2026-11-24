#!/usr/bin/env bash
# Kafka broker host (Ubuntu): install Java and Kafka, format a single-node KRaft broker
# (combined broker and controller), and start it.
#
# Usage: aws/kafka_setup.sh
# Settings (environment or .env): KAFKA_ADVERTISED_HOST (required: the address producers
#   and Kafka Connect use, e.g. the broker's private IP), KAFKA_VERSION (default 4.3.2),
#   KAFKA_HOME (default /opt/kafka), KAFKA_LOG_DIR (default /var/lib/kafka),
#   KAFKA_HEAP (default 4g).
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
set -euo pipefail
: "${KAFKA_ADVERTISED_HOST:?set KAFKA_ADVERTISED_HOST to the broker address clients use}"
KAFKA_VERSION=${KAFKA_VERSION:-4.3.2}
KAFKA_HOME=${KAFKA_HOME:-/opt/kafka}
KAFKA_LOG_DIR=${KAFKA_LOG_DIR:-/var/lib/kafka}
KAFKA_HEAP=${KAFKA_HEAP:-4g}
user=$(id -un)

bash "$ROOT/aws/kafka_install.sh"
# psutil is only for sampling the broker with cpusample.py; the broker runs without it.
sudo apt-get install -yq python3-psutil || echo "python3-psutil not installed; broker sampling unavailable"

sudo mkdir -p "$KAFKA_LOG_DIR"
sudo chown "$user:$user" "$KAFKA_LOG_DIR"
conf="$KAFKA_HOME/config/bench.properties"
sed -e "s|^log.dirs=.*|log.dirs=$KAFKA_LOG_DIR|" \
    -e "s|^advertised.listeners=.*|advertised.listeners=PLAINTEXT://$KAFKA_ADVERTISED_HOST:9092,CONTROLLER://localhost:9093|" \
    "$KAFKA_HOME/config/server.properties" > "$conf"
if [ ! -f "$KAFKA_LOG_DIR/meta.properties" ]; then
  "$KAFKA_HOME/bin/kafka-storage.sh" format --standalone \
    -t "$("$KAFKA_HOME/bin/kafka-storage.sh" random-uuid)" -c "$conf"
fi
KAFKA_HEAP_OPTS="-Xms$KAFKA_HEAP -Xmx$KAFKA_HEAP" "$KAFKA_HOME/bin/kafka-server-start.sh" -daemon "$conf"
for _ in $(seq 60); do
  if "$KAFKA_HOME/bin/kafka-broker-api-versions.sh" --bootstrap-server localhost:9092 >/dev/null 2>&1; then
    echo "kafka $KAFKA_VERSION up, advertised as $KAFKA_ADVERTISED_HOST:9092"; exit 0
  fi
  sleep 1
done
echo "kafka did not come up"; exit 1
