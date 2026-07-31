# OARM Simulator Scenarios

These presets apply OARM-friendly occlusion settings to the existing YOPO
Simulator config without changing YOPO planner code.

The first recommended scene is `oarm_wall_blind_corner_s0.yaml`. It uses the
Simulator's built-in `maze_type: 7` random wall generator with long walls, so it
creates partial observability, blind-corner choices, and late obstacle reveal.

Use `oarm_wall_sparse_s0.yaml` first if the blind-corner scene is too hard or
the vehicle cannot leave the start region.

```bash
cd /workspace/YOPO
python3 OARM/scenarios/apply_simulator_scenario.py \
  --scenario OARM/scenarios/oarm_wall_blind_corner_s0.yaml \
  --config Simulator/src/config/config.yaml
```

After restarting the Simulator, use a matched GT pointcloud generated from the
same config before trusting clearance, collision, or reaction-margin metrics.

