#!/usr/bin/env bash
# Install a JDK (Amazon Corretto tarball, no apt) and the Apache Kafka distribution into
# KAFKA_HOME (used by the broker host and, for Kafka Connect, by the QuestDB host).
#
# Usage: aws/kafka_install.sh
# Settings (environment or .env): KAFKA_VERSION (default 4.3.2), KAFKA_HOME (default
#   /opt/kafka), JAVA_DIR (default /opt/java), JAVA_URL (default: latest Corretto 21).
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
set -euo pipefail
KAFKA_VERSION=${KAFKA_VERSION:-4.3.2}
KAFKA_HOME=${KAFKA_HOME:-/opt/kafka}
JAVA_DIR=${JAVA_DIR:-/opt/java}
JAVA_URL=${JAVA_URL:-https://corretto.aws/downloads/latest/amazon-corretto-21-x64-linux-jdk.tar.gz}
user=$(id -un)

if [ ! -x "$JAVA_DIR/bin/java" ]; then
  curl -fsSL -o /tmp/jdk.tar.gz "$JAVA_URL"
  sudo mkdir -p "$JAVA_DIR"
  sudo tar -xzf /tmp/jdk.tar.gz -C "$JAVA_DIR" --strip-components=1
  sudo ln -sf "$JAVA_DIR/bin/java" /usr/local/bin/java
fi
if [ ! -d "$KAFKA_HOME" ]; then
  tarball="kafka_2.13-$KAFKA_VERSION.tgz"
  curl -fsSL -o "/tmp/$tarball" "https://downloads.apache.org/kafka/$KAFKA_VERSION/$tarball"
  sudo mkdir -p "$KAFKA_HOME"
  sudo tar -xzf "/tmp/$tarball" -C "$KAFKA_HOME" --strip-components=1
  sudo chown -R "$user:$user" "$KAFKA_HOME"
fi
java -version 2>&1 | head -1
