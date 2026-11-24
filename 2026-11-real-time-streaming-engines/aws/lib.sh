# shellcheck shell=bash
# Sourced by the aws/*.sh scripts. Sets ROOT (this example's folder) and loads ROOT/.env:
# plain KEY=value lines; a variable already set in the environment wins, so a remote
# command can override any value (for example `env QDB_ADDR=10.0.0.5:9000 aws/run.sh ...`).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ROOT/.env" ]; then
  while IFS='=' read -r key value; do
    [[ $key =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [ -n "${!key+x}" ] || export "$key=$value"
  done < "$ROOT/.env"
fi
PY=${PY:-$ROOT/.venv/bin/python}
