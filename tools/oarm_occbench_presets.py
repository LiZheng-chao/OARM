import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class OccBenchPreset:
    name: str
    split: str
    bucket: str
    scenario: str
    bucket_weight: float
    preset_weight: float
    maze_type: int
    seed: int
    env_num: int
    image_num: int
    save_subdir: str
    x_length: int = 60
    y_length: int = 60
    z_length: int = 15
    x_range: int = 40
    y_range: int = 40
    z_min: float = 0.5
    z_max: float = 4.0
    safe_dist: float = 0.7
    max_depth_dist: float = 16.0
    fx: float = 80.0
    fy: float = 80.0
    roll_range: float = 20.0
    pitch_range: float = 20.0
    obstacle_number: int = 120
    tree_dist: float = 3.2
    room_number: int = 5
    max_windows: int = 2
    wall_number: int = 120
    wall_width_min: float = 1.5
    wall_width_max: float = 8.0
    road_width: float = 3.0
    complexity: float = 0.02
    fill: float = 0.1
    notes: str = ""

    @property
    def total_images(self) -> int:
        return self.env_num * self.image_num


TRAIN_BUCKETS: Tuple[Dict, ...] = (
    {
        "bucket": "wall_blind_corner",
        "bucket_weight": 0.35,
        "presets": [
            {
                "name": "wall_blind_corner_train",
                "scenario": "wall_blind_corner",
                "preset_weight": 0.35,
                "maze_type": 7,
                "seed": 3100,
                "max_depth_dist": 14.0,
                "safe_dist": 0.7,
                "wall_number": 140,
                "wall_width_min": 2.0,
                "wall_width_max": 9.0,
                "notes": "Random walls create blind-corner and side-occlusion samples.",
            }
        ],
    },
    {
        "bucket": "room_doorway_window",
        "bucket_weight": 0.25,
        "presets": [
            {
                "name": "room_doorway_window_train",
                "scenario": "room_doorway_window",
                "preset_weight": 0.25,
                "maze_type": 6,
                "seed": 3200,
                "max_depth_dist": 14.0,
                "safe_dist": 0.6,
                "room_number": 5,
                "max_windows": 3,
                "notes": "Room walls with windows/door-like openings emphasize threshold occlusion.",
            }
        ],
    },
    {
        "bucket": "maze_corridor_tjunction",
        "bucket_weight": 0.20,
        "presets": [
            {
                "name": "maze_corridor_tjunction_train",
                "scenario": "maze_corridor_tjunction",
                "preset_weight": 0.20,
                "maze_type": 3,
                "seed": 3300,
                "max_depth_dist": 12.0,
                "safe_dist": 0.6,
                "road_width": 3.0,
                "notes": "Recursive maze creates corridor, branch, and T-junction visibility changes.",
            }
        ],
    },
    {
        "bucket": "forest_clutter",
        "bucket_weight": 0.10,
        "presets": [
            {
                "name": "forest_clutter_train",
                "scenario": "forest_clutter",
                "preset_weight": 0.10,
                "maze_type": 5,
                "seed": 3400,
                "max_depth_dist": 16.0,
                "safe_dist": 0.7,
                "tree_dist": 3.0,
                "notes": "Sparse forest/clutter keeps natural partial occlusion coverage.",
            }
        ],
    },
    {
        "bucket": "perlin_random_generalization",
        "bucket_weight": 0.10,
        "presets": [
            {
                "name": "perlin_cave_generalization_train",
                "scenario": "perlin_cave_generalization",
                "preset_weight": 0.05,
                "maze_type": 1,
                "seed": 3500,
                "max_depth_dist": 18.0,
                "safe_dist": 0.7,
                "complexity": 0.025,
                "fill": 0.08,
                "notes": "Perlin/cave geometry provides shape-distribution generalization.",
            },
            {
                "name": "random_columns_generalization_train",
                "scenario": "random_columns_generalization",
                "preset_weight": 0.05,
                "maze_type": 2,
                "seed": 3600,
                "max_depth_dist": 20.0,
                "safe_dist": 0.6,
                "obstacle_number": 140,
                "notes": "Random columns preserve ordinary YOPO-style obstacle diversity.",
            },
        ],
    },
)


TEST_PRESETS: Tuple[Dict, ...] = (
    {"name": "wall_blind_corner_test", "bucket": "wall_blind_corner", "scenario": "wall_blind_corner", "maze_type": 7, "seed": 9100},
    {"name": "room_doorway_window_test", "bucket": "room_doorway_window", "scenario": "room_doorway_window", "maze_type": 6, "seed": 9200},
    {"name": "maze_corridor_tjunction_test", "bucket": "maze_corridor_tjunction", "scenario": "maze_corridor_tjunction", "maze_type": 3, "seed": 9300},
    {"name": "forest_clutter_test", "bucket": "forest_clutter", "scenario": "forest_clutter", "maze_type": 5, "seed": 9400},
    {"name": "perlin_cave_generalization_test", "bucket": "perlin_random_generalization", "scenario": "perlin_cave_generalization", "maze_type": 1, "seed": 9500},
    {"name": "random_columns_generalization_test", "bucket": "perlin_random_generalization", "scenario": "random_columns_generalization", "maze_type": 2, "seed": 9600},
)


def images_per_map(total_images: int, weight: float, env_num: int) -> int:
    return max(1, int(math.ceil(total_images * weight / max(env_num, 1))))


def build_train_presets(args: argparse.Namespace) -> List[OccBenchPreset]:
    presets: List[OccBenchPreset] = []
    for bucket in TRAIN_BUCKETS:
        for spec in bucket["presets"]:
            env_num = args.train_env_num
            if spec["maze_type"] in {1, 2, 5, 6}:
                env_num = max(2, int(round(args.train_env_num * args.secondary_env_scale)))
            image_num = images_per_map(args.train_total_images, spec["preset_weight"], env_num)
            values = dict(spec)
            presets.append(
                OccBenchPreset(
                    name=values.pop("name"),
                    split="train_raw",
                    bucket=bucket["bucket"],
                    scenario=values.pop("scenario"),
                    bucket_weight=bucket["bucket_weight"],
                    preset_weight=values.pop("preset_weight"),
                    maze_type=values.pop("maze_type"),
                    seed=values.pop("seed"),
                    env_num=env_num,
                    image_num=image_num,
                    save_subdir=f"{args.dataset_prefix}/{spec['name']}",
                    **values,
                )
            )
    return presets


def build_test_presets(args: argparse.Namespace) -> List[OccBenchPreset]:
    presets: List[OccBenchPreset] = []
    for spec in TEST_PRESETS:
        presets.append(
            OccBenchPreset(
                name=spec["name"],
                split="test_raw",
                bucket=spec["bucket"],
                scenario=spec["scenario"],
                bucket_weight=1.0 / len(TEST_PRESETS),
                preset_weight=1.0 / len(TEST_PRESETS),
                maze_type=spec["maze_type"],
                seed=spec["seed"],
                env_num=args.test_env_num,
                image_num=args.test_image_num,
                save_subdir=f"{args.dataset_prefix}_test/{spec['name']}",
                max_depth_dist=16.0,
                notes="Fixed-seed test split; do not train on this data.",
            )
        )
    return presets


def render_yaml(preset: OccBenchPreset) -> str:
    return f"""# Generated by OARM.tools.oarm_occbench_presets.
# YOPO Simulator config for OARM-OccBench raw data collection.

odom_topic: "/sim/odom"
depth_topic: "/depth_image"
lidar_topic: "/lidar_points"

render_depth: true
depth_fps: 33
render_lidar: true
lidar_fps: 10

random_map: true
resolution: 0.1
ply_file: "src/pointcloud/forest.ply"
expand_x_times: 0
expand_y_times: 0
occupy_threshold: 0

seed: {preset.seed}
x_length: {preset.x_length}
y_length: {preset.y_length}
z_length: {preset.z_length}
maze_type: {preset.maze_type}

complexity: {preset.complexity}
fill: {preset.fill}
fractal: 1
attenuation: 0.1

width_min: 0.6
width_max: 1.5
obstacle_number: {preset.obstacle_number}

road_width: {preset.road_width}
add_wall_x: 1
add_wall_y: 1

tree_file: "src/pointcloud/tree.ply"
tree_dist: {preset.tree_dist}

room_number: {preset.room_number}
max_windows: {preset.max_windows}
window_size_min: 1.4
window_size_max: 2.8
add_ceiling: 0

wall_width_min: {preset.wall_width_min}
wall_width_max: {preset.wall_width_max}
wall_thick: 0.5
wall_number: {preset.wall_number}
wall_ceiling: 1

camera:
  fx: {preset.fx}
  fy: {preset.fy}
  cx: 80.0
  cy: 45.0
  image_width: 160
  image_height: 90
  max_depth_dist: {preset.max_depth_dist}
  normalize_depth: false
  pitch: -0.0

lidar:
  vertical_lines: 16
  vertical_angle_start: -15.0
  vertical_angle_end: 15.0
  horizontal_num: 360
  horizontal_resolution: 1.0
  max_lidar_dist: 20.0

save_path: "../dataset/{preset.save_subdir}/"
env_num: {preset.env_num}
image_num: {preset.image_num}
roll_range: {preset.roll_range}
pitch_range: {preset.pitch_range}
x_range: {preset.x_range}
y_range: {preset.y_range}
z_range: [{preset.z_min}, {preset.z_max}]
safe_dist: {preset.safe_dist}
ply_res: 0.1
"""


def write_collect_script(path: Path, config_dir: str, label: str) -> None:
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="${{REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}}"
SIM_CONFIG="$REPO_ROOT/Simulator/src/config/config.yaml"
BACKUP="${{SIM_CONFIG}}.oarm_backup"

cp "$SIM_CONFIG" "$BACKUP"
restore_config() {{
  cp "$BACKUP" "$SIM_CONFIG"
}}
trap restore_config EXIT

source "$REPO_ROOT/Simulator/devel/setup.bash"

for cfg in "$SCRIPT_DIR/{config_dir}"/*.yaml; do
  echo "[OARM-OccBench] collecting {label}: $cfg"
  cp "$cfg" "$SIM_CONFIG"
  (cd "$REPO_ROOT/Simulator" && rosrun sensor_simulator dataset_generator)
done
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_runbook(output_dir: Path, train_presets: List[OccBenchPreset], test_presets: List[OccBenchPreset]) -> None:
    train_roots = [f"dataset/{preset.save_subdir}" for preset in train_presets]
    input_roots = " \\\n  ".join(f"--input-root {root}" for root in train_roots)
    lines = [
        "# OARM-OccBench Collection Plan",
        "",
        "This directory contains Simulator configs and scripts for raw OARM dataset collection.",
        "The raw data keeps the original YOPO format: `pointcloud-i.ply`, `i/img_k.png`, and `pose-i.csv`.",
        "",
        "## 1. Collect Raw Train Data",
        "",
        "```bash",
        f"bash {output_dir}/collect_train.sh",
        "```",
        "",
        "## 2. Curate OARM Train Data",
        "",
        "```bash",
        "OARM_GT_POINTCLOUD_CACHE_SIZE=16 python -m OARM.tools.curate_oarm_dataset \\",
        f"  {input_roots} \\",
        "  --output-root dataset/oarm_occbench_train \\",
        "  --target-images 60000 \\",
        "  --overwrite \\",
        "  --metrics-jsonl OARM/results/oarm_occbench_train_metrics.jsonl \\",
        "  --print-every 1000",
        "```",
        "",
        "## Train Scene Mix",
        "",
    ]
    for preset in train_presets:
        lines.append(
            f"- `{preset.name}`: bucket={preset.bucket}, scenario={preset.scenario}, "
            f"weight={preset.preset_weight:.2f}, maps={preset.env_num}, images/map={preset.image_num}, "
            f"total={preset.total_images}"
        )
    if test_presets:
        lines.extend(["", "## Test Collection", "", "```bash", f"bash {output_dir}/collect_test.sh", "```"])
    (output_dir / "RUNBOOK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate OARM-OccBench Simulator configs and collection scripts.")
    p.add_argument("--output-dir", default="OARM/configs/dataset_generation/oarm_occbench_v3")
    p.add_argument("--dataset-prefix", default="oarm_occbench_raw")
    p.add_argument("--train-total-images", type=int, default=160000)
    p.add_argument("--train-env-num", type=int, default=10)
    p.add_argument("--secondary-env-scale", type=float, default=0.6)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--test-env-num", type=int, default=3)
    p.add_argument("--test-image-num", type=int, default=1000)
    p.add_argument("--print-plan", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_presets = build_train_presets(args)
    test_presets = build_test_presets(args) if args.include_test else []

    for preset in train_presets:
        (train_dir / f"{preset.name}.yaml").write_text(render_yaml(preset), encoding="utf-8")
    if test_presets:
        test_dir.mkdir(parents=True, exist_ok=True)
        for preset in test_presets:
            (test_dir / f"{preset.name}.yaml").write_text(render_yaml(preset), encoding="utf-8")

    write_collect_script(output_dir / "collect_train.sh", "train", "train")
    if test_presets:
        write_collect_script(output_dir / "collect_test.sh", "test", "test")
    write_runbook(output_dir, train_presets, test_presets)

    manifest = {
        "name": "OARM-OccBench",
        "format": "YOPO-compatible raw collection configs",
        "train_total_images_requested": args.train_total_images,
        "train_total_images_configured": int(sum(preset.total_images for preset in train_presets)),
        "scene_mix": {
            "wall_blind_corner": 0.35,
            "room_doorway_window": 0.25,
            "maze_corridor_tjunction": 0.20,
            "forest_clutter": 0.10,
            "perlin_random_generalization": 0.10,
        },
        "train_presets": [asdict(preset) for preset in train_presets],
        "test_presets": [asdict(preset) for preset in test_presets],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.print_plan:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"Wrote {len(train_presets)} train presets to {train_dir}")
        if test_presets:
            print(f"Wrote {len(test_presets)} test presets to {test_dir}")
        print(f"Run: bash {output_dir / 'collect_train.sh'}")


if __name__ == "__main__":
    main()
