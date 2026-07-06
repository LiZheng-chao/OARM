# OARM Dataset V1 Presets

These files are generated configs for the existing YOPO Simulator.
They do not modify Simulator source code by themselves.

The current `dataset_generator` binary reads `Simulator/src/config/config.yaml`
because that path is compiled in CMake. For each preset, copy the YAML to
that path only during generation, run the generator, then restore the original
config.

Example:

```bash
cp Simulator/src/config/config.yaml /tmp/oarm_sim_config_backup.yaml
cp OARM/configs/dataset_generation/oarm_v1/wall_blind_corner_train.yaml Simulator/src/config/config.yaml
cd Simulator
source devel/setup.bash
rosrun sensor_simulator dataset_generator
cp /tmp/oarm_sim_config_backup.yaml src/config/config.yaml
```

Generated presets:

- `wall_blind_corner_train.yaml`: split=oarm_targeted_train, scenario=wall_blind_corner, maze_type=7, seed=2000, maps=6, images/map=4000
- `room_doorway_train.yaml`: split=oarm_targeted_train, scenario=room_doorway, maze_type=6, seed=2200, maps=3, images/map=4000
- `maze_t_junction_train.yaml`: split=oarm_targeted_train, scenario=maze_t_junction, maze_type=3, seed=2400, maps=3, images/map=4000
- `occluded_forest_train.yaml`: split=oarm_targeted_train, scenario=occluded_forest, maze_type=5, seed=2600, maps=3, images/map=4000
- `random_columns_train.yaml`: split=oarm_targeted_train, scenario=random_columns, maze_type=2, seed=2800, maps=2, images/map=4000
- `wall_blind_corner_test.yaml`: split=oarm_fixed_test, scenario=wall_blind_corner, maze_type=7, seed=9000, maps=2, images/map=500
- `room_doorway_test.yaml`: split=oarm_fixed_test, scenario=doorway, maze_type=6, seed=9020, maps=2, images/map=500
- `maze_t_junction_test.yaml`: split=oarm_fixed_test, scenario=t_junction, maze_type=3, seed=9040, maps=2, images/map=500
- `occluded_forest_test.yaml`: split=oarm_fixed_test, scenario=occluded_forest, maze_type=5, seed=9060, maps=2, images/map=500
- `limited_fov_wall_test.yaml`: split=oarm_fixed_test, scenario=limited_fov, maze_type=7, seed=9080, maps=2, images/map=500
- `random_columns_test.yaml`: split=oarm_fixed_test, scenario=random_columns, maze_type=2, seed=9100, maps=2, images/map=500
