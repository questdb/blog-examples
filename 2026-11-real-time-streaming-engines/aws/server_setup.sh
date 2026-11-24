#!/usr/bin/env bash
# One-time setup of the QuestDB server host (Ubuntu): format and mount the data volume,
# install a QuestDB release (tarball with bundled JRE), raise file and map limits, and
# install psutil for cpusample.py.
#
# Usage: aws/server_setup.sh
# Settings (environment or .env): QDB_VERSION (default 10.2.0), QDB_HOME (default
#   /opt/questdb), DATA_MOUNT (default /data), DATA_DEV_SIZE (size of the empty data
#   volume as lsblk prints it, default 200G; the disk with that size is formatted).
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "$0")/lib.sh"
QDB_VERSION=${QDB_VERSION:-10.2.0}
QDB_HOME=${QDB_HOME:-/opt/questdb}
DATA_MOUNT=${DATA_MOUNT:-/data}
DATA_DEV_SIZE=${DATA_DEV_SIZE:-200G}
user=$(id -un)

sudo apt-get update -q
sudo apt-get install -yq xfsprogs python3-psutil chrony

if ! mountpoint -q "$DATA_MOUNT"; then
  devs=$(lsblk -dpno NAME,SIZE,MOUNTPOINT | awk -v s="$DATA_DEV_SIZE" '$2 == s && $3 == "" {print $1}')
  if [ "$(echo "$devs" | grep -c .)" != "1" ]; then
    echo "expected exactly one unmounted $DATA_DEV_SIZE disk, found: ${devs:-none}"; exit 1
  fi
  sudo mkfs.xfs -f "$devs"
  sudo mkdir -p "$DATA_MOUNT"
  sudo mount -o noatime "$devs" "$DATA_MOUNT"
  sudo chown "$user:$user" "$DATA_MOUNT"
fi

if [ ! -d "$QDB_HOME" ]; then
  tarball="questdb-$QDB_VERSION-rt-linux-x86-64.tar.gz"
  curl -fsSL -o "/tmp/$tarball" \
    "https://github.com/questdb/questdb/releases/download/$QDB_VERSION/$tarball"
  sudo mkdir -p "$QDB_HOME"
  sudo tar -xzf "/tmp/$tarball" -C "$QDB_HOME" --strip-components=1
  sudo chown -R "$user:$user" "$QDB_HOME"
fi

# QuestDB recommends raising these for many open files and memory maps.
printf 'fs.file-max=1048576\nvm.max_map_count=1048576\n' | sudo tee /etc/sysctl.d/90-questdb.conf
sudo sysctl -q --system
printf '%s soft nofile 1048576\n%s hard nofile 1048576\n' "$user" "$user" \
  | sudo tee /etc/security/limits.d/questdb.conf

lsblk
df -h "$DATA_MOUNT"
chronyc sources
