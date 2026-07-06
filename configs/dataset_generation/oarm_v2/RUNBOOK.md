# OARM Dataset V2 Presets

V2 fixes two V1 issues: preset weights now control actual image counts,
and limited depth range is separated from limited FOV.

After generating individual sub-datasets, merge train roots with:

```bash
python3 -m OARM.tools.merge_yopo_datasets \
  --input-root dataset/oarm_v2/wall_blind_corner_train \
  --input-root dataset/oarm_v2/room_doorway_train \
  --input-root dataset/oarm_v2/maze_t_junction_train \
  --input-root dataset/oarm_v2/occluded_forest_train \
  --input-root dataset/oarm_v2/random_columns_train \
  --output-root dataset/oarm_v2_merged_train
```

Generated presets:

- `wall_blind_corner_train.yaml`: split=oarm_targeted_train, scenario=wall_blind_corner, weight=0.30, maps=6, images/map=3400, total=20400
- `room_doorway_train.yaml`: split=oarm_targeted_train, scenario=room_doorway, weight=0.25, maps=3, images/map=5667, total=17001
- `maze_t_junction_train.yaml`: split=oarm_targeted_train, scenario=maze_t_junction, weight=0.20, maps=3, images/map=4533, total=13599
- `occluded_forest_train.yaml`: split=oarm_targeted_train, scenario=occluded_forest, weight=0.15, maps=3, images/map=3400, total=10200
- `random_columns_train.yaml`: split=oarm_targeted_train, scenario=random_columns, weight=0.10, maps=2, images/map=3400, total=6800
- `wall_blind_corner_test.yaml`: split=oarm_fixed_test, scenario=wall_blind_corner, weight=0.14, maps=2, images/map=500, total=1000
- `room_doorway_test.yaml`: split=oarm_fixed_test, scenario=doorway, weight=0.14, maps=2, images/map=500, total=1000
- `maze_t_junction_test.yaml`: split=oarm_fixed_test, scenario=t_junction, weight=0.14, maps=2, images/map=500, total=1000
- `occluded_forest_test.yaml`: split=oarm_fixed_test, scenario=occluded_forest, weight=0.14, maps=2, images/map=500, total=1000
- `limited_depth_range_test.yaml`: split=oarm_fixed_test, scenario=limited_depth_range, weight=0.14, maps=2, images/map=500, total=1000
- `limited_fov_wall_test.yaml`: split=oarm_fixed_test, scenario=limited_fov, weight=0.14, maps=2, images/map=500, total=1000
- `random_columns_test.yaml`: split=oarm_fixed_test, scenario=random_columns, weight=0.14, maps=2, images/map=500, total=1000
