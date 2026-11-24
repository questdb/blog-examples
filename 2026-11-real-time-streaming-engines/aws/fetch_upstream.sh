#!/usr/bin/env bash
# Download the original benchmark's producer (fmorillo7694/spark-rtm-vs-flink) at a pinned
# commit into upstream/ (not part of this repo), for the Kafka runs.
#
# Usage: aws/fetch_upstream.sh
# Settings (environment or .env): UPSTREAM_SHA (default: the commit these results used).
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
set -euo pipefail
UPSTREAM_SHA=${UPSTREAM_SHA:-4166e7ba5d0bc59a2303d896f8e2e825ea158cf7}
mkdir -p "$ROOT/upstream"
for f in producer.py joindata.py; do
  curl -fsSL -o "$ROOT/upstream/$f" \
    "https://raw.githubusercontent.com/fmorillo7694/spark-rtm-vs-flink/$UPSTREAM_SHA/common/$f"
done
echo "upstream producer at $UPSTREAM_SHA in $ROOT/upstream"
