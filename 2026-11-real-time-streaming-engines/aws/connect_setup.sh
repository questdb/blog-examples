#!/usr/bin/env bash
# QuestDB host: install Kafka (for its Connect runtime) and the QuestDB Kafka connector.
#
# Usage: aws/connect_setup.sh
# Settings (environment or .env): CONNECTOR_VERSION (default 0.25), KAFKA_VERSION,
#   KAFKA_HOME (see kafka_install.sh).
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
set -euo pipefail
CONNECTOR_VERSION=${CONNECTOR_VERSION:-0.25}
KAFKA_HOME=${KAFKA_HOME:-/opt/kafka}

bash "$ROOT/aws/kafka_install.sh"
zip="kafka-questdb-connector-$CONNECTOR_VERSION-bin.zip"
curl -fsSL -o "/tmp/$zip" \
  "https://github.com/questdb/kafka-questdb-connector/releases/download/v$CONNECTOR_VERSION/$zip"
rm -rf /tmp/kafka-questdb-connector
python3 -m zipfile -e "/tmp/$zip" /tmp
cp /tmp/kafka-questdb-connector/*.jar "$KAFKA_HOME/libs/"
echo "QuestDB connector $CONNECTOR_VERSION installed in $KAFKA_HOME/libs"
