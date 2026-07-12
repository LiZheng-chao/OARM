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

for cfg in "$SCRIPT_DIR/test"/*.yaml; do
  echo "[OARM-OccBench] collecting test: $cfg"
  cp "$cfg" "$SIM_CONFIG"
  (cd "$REPO_ROOT/Simulator" && rosrun sensor_simulator dataset_generator)
  save_path="$(python3 - "$cfg" <<'PY'
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    value = line.split("#", 1)[0].strip()
    if value.startswith("save_path:"):
        value = value.split(":", 1)[1].strip()
        if len(value) >= 2 and value[0] in (chr(34), chr(39)) and value[-1] == value[0]:
            value = value[1:-1]
        print(value.rstrip("/"))
        break
else:
    raise SystemExit("save_path not found")
PY
)"
  if [[ "$save_path" == ../* ]]; then
    dataset_root="$(cd "$REPO_ROOT/Simulator" && realpath "$save_path")"
  elif [[ "$save_path" == /* ]]; then
    dataset_root="$save_path"
  else
    dataset_root="$REPO_ROOT/$save_path"
  fi
  (cd "$REPO_ROOT" && python3 -m OARM.tools.validate_raw_depth --dataset-root "$dataset_root")
done
