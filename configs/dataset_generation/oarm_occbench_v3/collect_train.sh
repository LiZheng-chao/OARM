#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
SIM_CONFIG="$REPO_ROOT/Simulator/src/config/config.yaml"
BACKUP="${SIM_CONFIG}.oarm_backup"

cp "$SIM_CONFIG" "$BACKUP"
restore_config() {
  cp "$BACKUP" "$SIM_CONFIG"
}
trap restore_config EXIT

source "$REPO_ROOT/Simulator/devel/setup.bash"

for cfg in "$SCRIPT_DIR/train"/*.yaml; do
  echo "[OARM-OccBench] collecting train: $cfg"
  cp "$cfg" "$SIM_CONFIG"
  (cd "$REPO_ROOT/Simulator" && rosrun sensor_simulator dataset_generator)
done
