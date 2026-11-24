#!/usr/bin/env bash
# One-time setup of the producer host (Ubuntu): a Python venv in this folder with the
# QuestDB client, and chrony for clock sync.
#
# Usage: aws/producer_setup.sh
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
sudo apt-get update -q
sudo apt-get install -yq python3-venv chrony
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/python" -c "import questdb, psutil; print('client ok')"
chronyc sources
