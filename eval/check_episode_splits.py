import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


def _read_manifest(path: str, episode_key: str = "episode_id", map_key: str = "map_id") -> Tuple[Set[str], Set[str]]:
    episodes: Set[str] = set()
    maps: Set[str] = set()
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        rows = payload.get("episodes", payload.get("items", payload.get("rows", [])))
        if not rows and episode_key in payload:
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"unsupported manifest format: {path}")
    for row in rows:
        if isinstance(row, str):
            episodes.add(row)
            continue
        if not isinstance(row, dict):
            continue
        if row.get(episode_key) is not None:
            episodes.add(str(row[episode_key]))
        if row.get(map_key) is not None:
            maps.add(str(row[map_key]))
    return episodes, maps


def check_episode_splits(manifests: Dict[str, str], episode_key: str = "episode_id", map_key: str = "map_id", check_maps: bool = True) -> Dict:
    split_episodes: Dict[str, Set[str]] = {}
    split_maps: Dict[str, Set[str]] = {}
    for split, path in manifests.items():
        episodes, maps = _read_manifest(path, episode_key=episode_key, map_key=map_key)
        if not episodes:
            raise ValueError(f"{split} manifest has no episodes: {path}")
        split_episodes[split] = episodes
        split_maps[split] = maps

    overlaps = []
    splits = sorted(split_episodes.keys())
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            shared_eps = sorted(split_episodes[left] & split_episodes[right])
            shared_maps = sorted(split_maps[left] & split_maps[right]) if check_maps else []
            if shared_eps or shared_maps:
                overlaps.append({"left": left, "right": right, "episodes": shared_eps, "maps": shared_maps})
    if overlaps:
        raise ValueError(f"episode/map split leakage detected: {overlaps}")
    return {
        "splits": {split: {"episode_count": len(split_episodes[split]), "map_count": len(split_maps[split])} for split in splits},
        "check_maps": bool(check_maps),
        "ok": True,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check train/val/calibration/test episode and map manifest disjointness.")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--episode-key", default="episode_id")
    p.add_argument("--map-key", default="map_id")
    p.add_argument("--ignore-map-overlap", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    result = check_episode_splits(
        {"train": args.train, "val": args.val, "calibration": args.calibration, "test": args.test},
        episode_key=args.episode_key,
        map_key=args.map_key,
        check_maps=not args.ignore_map_overlap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
