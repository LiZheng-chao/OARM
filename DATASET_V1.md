# OARM Dataset V1

This document defines the first OARM-targeted dataset plan while keeping the
original YOPO/Simulator source path untouched.

## Goals

- Preserve the YOPO dataset format:
  - `dataset/pointcloud-<map_id>.ply`
  - `dataset/<map_id>/img_<image_id>.png`
  - `dataset/pose-<map_id>.csv`
- Add OARM-targeted scene distributions through separate config presets.
- Keep fixed evaluation seeds separate from training seeds.
- Make the dataset auditable with a generated manifest.

## Splits

Use three families of data.

| Split | Purpose | Suggested seeds |
| --- | --- | --- |
| `yopo_default` | Fair baseline and non-regression training | `1000-1099` |
| `oarm_targeted_train` | OARM margin/risk training | `2000-2999` |
| `oarm_fixed_test` | Paper-only fixed scenarios | `9000-9099` |

## Scene Mix

The current Simulator already supports useful map types:

| Preset | Simulator `maze_type` | Purpose |
| --- | ---: | --- |
| `wall_blind_corner` | `7` | large occluders, blind-corner-like views |
| `room_doorway` | `6` | doorway/window threshold views |
| `maze_t_junction` | `3` | narrow corridors and T-junction structure |
| `occluded_forest` | `5` | natural occlusion behind trunks |
| `random_columns` | `2` | ordinary geometric avoidance coverage |

Recommended OARM-targeted training mix:

```text
wall_blind_corner: 30%
room_doorway:      25%
maze_t_junction:   20%
occluded_forest:   15%
random_columns:    10%
```

## First Implementation Step

Do not edit `Simulator/src/config/config.yaml` by hand. Generate standalone
YAML presets and a manifest instead:

```bash
python3 -m OARM.tools.oarm_dataset_presets \
  --output-dir OARM/configs/dataset_generation/oarm_v1
```

Each generated YAML file is compatible with the existing `dataset_generator`.
Because the compiled generator reads `Simulator/src/config/config.yaml`, use the
generated configs as controlled inputs: copy one preset to the Simulator config
only for the generation run, then restore the original config.

## Acceptance Checks

Before training full OARM again, run dataset-level checks:

- `hidden_risk_gt_coverage` should be non-trivial on OARM-targeted data.
- `selected_rmvr_gt_hidden` should separate good/bad candidates.
- `minimum_reaction_margin_gt_hidden` should include negative samples.
- Candidate type usage should include progress/probe/brake/yield in plausible
  proportions.

This dataset plan supports the paper story as occlusion-aware risk and reaction
margin learning, not as active yaw planning.
