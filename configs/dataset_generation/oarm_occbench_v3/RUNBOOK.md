# OARM-OccBench Collection Plan

This directory contains Simulator configs and scripts for raw OARM dataset collection.
The raw data keeps the original YOPO format: `pointcloud-i.ply`, `i/img_k.png`, and `pose-i.csv`.

## 1. Collect Raw Train Data

```bash
bash OARM/configs/dataset_generation/oarm_occbench_v3/collect_train.sh
```

## 2. Curate OARM Train Data

```bash
OARM_GT_POINTCLOUD_CACHE_SIZE=16 python -m OARM.tools.curate_oarm_dataset \
  --input-root dataset/oarm_occbench_raw/wall_blind_corner_train \
  --input-root dataset/oarm_occbench_raw/room_doorway_window_train \
  --input-root dataset/oarm_occbench_raw/maze_corridor_tjunction_train \
  --input-root dataset/oarm_occbench_raw/forest_clutter_train \
  --input-root dataset/oarm_occbench_raw/perlin_cave_generalization_train \
  --input-root dataset/oarm_occbench_raw/random_columns_generalization_train \
  --output-root dataset/oarm_occbench_train \
  --target-images 60000 \
  --overwrite \
  --metrics-jsonl OARM/results/oarm_occbench_train_metrics.jsonl \
  --print-every 1000
```

## Train Scene Mix

- `wall_blind_corner_train`: bucket=wall_blind_corner, scenario=wall_blind_corner, weight=0.35, maps=10, images/map=5600, total=56000
- `room_doorway_window_train`: bucket=room_doorway_window, scenario=room_doorway_window, weight=0.25, maps=6, images/map=6667, total=40002
- `maze_corridor_tjunction_train`: bucket=maze_corridor_tjunction, scenario=maze_corridor_tjunction, weight=0.20, maps=10, images/map=3200, total=32000
- `forest_clutter_train`: bucket=forest_clutter, scenario=forest_clutter, weight=0.10, maps=6, images/map=2667, total=16002
- `perlin_cave_generalization_train`: bucket=perlin_random_generalization, scenario=perlin_cave_generalization, weight=0.05, maps=6, images/map=1334, total=8004
- `random_columns_generalization_train`: bucket=perlin_random_generalization, scenario=random_columns_generalization, weight=0.05, maps=6, images/map=1334, total=8004

## Test Collection

```bash
bash OARM/configs/dataset_generation/oarm_occbench_v3/collect_test.sh
```
