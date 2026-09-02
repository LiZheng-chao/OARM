"""Apply an OARM scenario preset to the YOPO Simulator config.

The scenario files intentionally use the same top-level keys as
Simulator/src/config/config.yaml so the change remains config-only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def parse_scalar_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_no}: expected 'key: value'")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"{path}:{line_no}: expected non-empty key and value")
        values[key] = value
    return values


def resolve_portable_paths(values: dict[str, str], project_root: Path) -> dict[str, str]:
    resolved = dict(values)
    raw_ply = resolved.get("ply_file")
    if raw_ply is None:
        return resolved

    unquoted = raw_ply.strip().strip('"').strip("'")
    workspace_prefix = "/workspace/YOPO/"
    if unquoted.startswith(workspace_prefix):
        ply_path = project_root / unquoted[len(workspace_prefix):]
    else:
        ply_path = Path(unquoted)
    if not ply_path.is_absolute():
        ply_path = project_root / ply_path
    ply_path = ply_path.resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"Scenario pointcloud not found: {ply_path}")
    resolved["ply_file"] = json.dumps(str(ply_path))
    return resolved


def patch_config(config_text: str, values: dict[str, str]) -> tuple[str, list[str], list[str]]:
    changed: list[str] = []
    missing: list[str] = []

    for key, value in values.items():
        pattern = re.compile(rf"^({re.escape(key)}\s*:\s*).*$", flags=re.MULTILINE)
        if not pattern.search(config_text):
            missing.append(key)
            continue

        def replace(match: re.Match[str]) -> str:
            return f"{match.group(1)}{value}"

        config_text, count = pattern.subn(replace, config_text, count=1)
        if count:
            changed.append(key)

    return config_text, changed, missing


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--config", default=Path("Simulator/src/config/config.yaml"), type=Path)
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    scenario = args.scenario
    config = args.config
    values = parse_scalar_yaml(scenario)
    values = resolve_portable_paths(values, Path(__file__).resolve().parents[2])
    original = config.read_text(encoding="utf-8")
    updated, changed, missing = patch_config(original, values)

    if missing:
        raise KeyError("Scenario keys not found in simulator config: " + ", ".join(missing))

    if args.dry_run:
        print(f"Dry run: would update {len(changed)} keys in {config}")
    else:
        if not args.no_backup:
            backup = config.with_suffix(config.suffix + ".oarm_bak")
            shutil.copy2(config, backup)
            print(f"Backup written: {backup}")
        config.write_text(updated, encoding="utf-8")
        print(f"Applied scenario: {scenario}")

    print("Changed keys: " + ", ".join(changed))


if __name__ == "__main__":
    main()

