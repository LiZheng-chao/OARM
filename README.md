# OARM-Planner modules

This directory adds an independent OARM code path while keeping `YOPO/` as a clean baseline.

Implemented scaffold:

- `policy/oarm_network.py`: 15D per-primitive OARM output head.
- `policy/oarm_state_transform.py`: YOPO-compatible residual decoding plus `T`, yaw, margin, risk and utility fields.
- `policy/oarm_poly_solver.py`: variable-time quintic trajectory sampling and yaw cubic utilities.
- `policy/oarm_candidate_generator.py`: progress/probe/brake anchor scaffold for the later OARM-V3 candidate rewrite.
- `loss/yaw_visibility_loss.py`: differentiable soft-FOV risk-point visibility guidance for the yaw head.
- `visibility/reaction_margin_labeler.py`: risk-point based reaction-margin label generation for privileged training.
- `visibility/risk_point_association.py`: candidate-aware closest-approach association from risk points to trajectories.
- `utils/risk_point_sampler.py`: fixed-size depth-frontier risk point proposal interface; later ESDF filtering can be added here.
- `utils/privileged_risk_filter.py`: optional YOPO ESDF-backed filter that turns proposals into privileged risk weights.
- `utils/visible_free_distance.py`: depth-based visible-free-distance estimate for stopping/yield feasibility labels.
- `loss/backup_feasibility_loss.py`: stopping/yield feasibility supervision from visible free distance. The file keeps its old name for checkpoint/log compatibility.
- `loss/collision_loss.py`: optional variable-time ESDF collision cost through YOPO's ESDF query backend.
- `eval/metrics_backup_feasibility.py`: stopping/yield feasibility, accuracy and reaction-margin violation metrics.
- `visibility/`: soft-FOV and first-visible-time utilities for reaction-margin labels.
- `eval/metrics_reaction_margin.py`: RMVR, selected-RMVR, margin prediction, risk calibration and pairwise ranking metrics.
- `eval/eval_dataset.py`: offline dataset evaluation for losses, selected candidate types, risk, margin and yield metrics.
- `eval/scenario_benchmark.py`: scenario-level metric aggregation for blind-corner, doorway, T-junction, occluded-forest and limited-FOV experiments.
- `eval/annotate_gt_reaction_margin.py`: offline GT-map post-processing for ROS JSONL logs; adds first-visible, arrival, reaction-margin and selected-RMVR labels.
- `eval/execution_monitor.py`: offline executed-path monitor for ROS JSONL logs; adds execution collision, success, clearance and metric-source fields.
- `eval/visualize_margin_labels.py`: sample-level depth/frontier/risk-point/candidate visualization for manual reaction-margin label checks.
- `loss/oarm_loss.py`: staged OARM loss with optional occlusion-risk, reaction-margin, pairwise ranking and yield labels.
- `utils/occlusion.py`: depth-frontier proxy labels for early OARM-V1 experiments.
- `dataset/oarm_dataset.py`: wrapper around `YOPODataset` that adds OARM label fields.
- `train_oarm.py`: minimal trainer entry point.
- `smoke_test.py`: random-tensor end-to-end check.

Suggested progression:

1. `python -m OARM.smoke_test`
2. `python -m OARM.tests.sanity_checks --device cpu --skip-dataset`
3. `python -m OARM.tests.sanity_checks --device cpu --one-batch-loss`
4. Train OARM-V0 with only variable time and utility supervision.
5. Add stronger privileged `occlusion_risk` labels from ESDF/risk points.
6. Feed `risk_points_w` / `risk_weight` labels to enable early yaw visibility cost and reaction-margin labels.
7. Add stopping/yield feasibility labels and yield candidates.
8. Replace YOPO lattice anchors with progress/probe/brake/yield candidates.

Runnable ablation presets:

```bash
python -m OARM.train_oarm --stage v0
python -m OARM.train_oarm --stage v1_occ
python -m OARM.train_oarm --stage v2_margin
python -m OARM.train_oarm --stage v3_yield
python -m OARM.train_oarm --stage full
python -m OARM.train_oarm --stage full --backbone-mode yopo_original
python -m OARM.train_oarm --stage v0 --candidate-mode yopo
python -m OARM.train_oarm --stage v0 --candidate-mode typed_frontier
```

Equivalent key-value config files live under `OARM/configs/`. They are intentionally small so experiment logs can say exactly which supervision terms were enabled.
`--train-yield-feasibility` is the preferred flag name for the stopping/yield prior; `--train-backup-feasibility` is still accepted as a compatibility alias.
`--train-margin-ranking` enables candidate-level pairwise ranking from reaction-margin labels; V2/V3/Full presets enable it by default.
`--candidate-mode yopo|typed_frontier` separates the YOPO-anchor and occlusion-conditioned typed-anchor ablations.
`--backbone-mode yopo_original` reuses YOPO's original ResNet18 depth backbone for the fair mainline comparison; keep checkpoints separated from `oarm_light` runs.

Evaluation commands:

```bash
python -m OARM.eval.eval_dataset --stage v2_margin --checkpoint OARM/saved/OARM_X/epochY.pth
python -m OARM.eval.eval_dataset --stage v3_yield --eval-yield-feasibility --checkpoint OARM/saved/OARM_X/epochY.pth
python -m OARM.eval.visualize_margin_labels --checkpoint OARM/saved/OARM_X/best_val.pth --sample 0 --count 16 --output-dir OARM/results/margin_label_viz
python -m OARM.eval.annotate_gt_reaction_margin --input OARM/results/runs/blind_corner.jsonl --output OARM/results/runs/blind_corner_gt.jsonl --dataset-dir dataset --map-id 0 --use-esdf-los --deployed-yaw-mode goal
python -m OARM.eval.execution_monitor --input OARM/results/runs/blind_corner_gt.jsonl --exec-input OARM/results/runs/blind_corner_odom.jsonl --output OARM/results/runs/blind_corner_exec.jsonl --dataset-dir dataset --map-id 0 --collision-clearance 0.25 --success-distance 1.0 --max-time 30 --clearance-sample-step 0.05
python -m OARM.eval.scenario_benchmark --print-manifest
python -m OARM.eval.scenario_benchmark OARM/results/runs/blind_corner_exec.jsonl OARM/results/runs/doorway_exec.jsonl --require-exec-metrics --output OARM/results/scenario_metrics.json --csv-output OARM/results/scenario_metrics.csv
```

Scenario logs should contain one row/object per control or planning sample with these fields:

```text
time, reaction_margin, first_visible_time, arrival_time_to_risk,
candidate_type, speed, inference_latency_ms, emergency_brake
```

After `annotate_gt_reaction_margin.py`, logs also contain:

```text
first_visible_time_gt, arrival_time_to_risk_gt, reaction_margin_gt, selected_rmvr_gt
selected_traj_min_clearance_gt, selected_traj_collision_gt, deployed_yaw_mode
```

After `execution_monitor.py`, logs also contain:

```text
collision_exec, success_exec, timeout_exec, min_clearance_exec, mean_clearance_exec,
path_time_exec, goal_distance_final, collision_exec_source, success_exec_source,
exec_rows_active, exec_position_count, exec_clearance_sample_count
```

`scenario_benchmark.py` uses `reaction_margin_gt` as the main RMVR source when it is present, while also reporting predicted-margin violation rate. It reports execution collision/success separately from selected-trajectory GT collision, so `selected_traj_collision_gt` is never treated as the executed collision rate.
For paper tables, report at least `success_exec_rate`, `collision_exec_rate`, `selected_rmvr_gt`, `mean_reaction_margin_gt`, `min_clearance_exec`, `mean_speed_exec`, `path_time_exec`, `emergency_brake_rate`, and `inference_latency_ms_mean`. Use `--require-exec-metrics` and `collision_metric_source` to verify the collision rate came from odometry (`executed_odom_to_gt_pointcloud`) rather than planner/reference starts.
By default, GT annotation uses `--deployed-yaw-mode goal` to match the current ROS controller. Use `predicted` only after deploying the OARM yaw reference in control.

`candidate_type` may be `progress`, `probe`, `brake`, `yield` or the legacy numeric ids `0..3`.

ROS paper runs:

```bash
python -m OARM.test_oarm_ros --checkpoint OARM/saved/OARM_X/epochY.pth --main-experiment
python -m OARM.test_oarm_ros --checkpoint OARM/saved/OARM_X/epochY.pth --main-experiment --method OARM-Full --scenario blind_corner --seed 0 --map-id 0 --candidate-mode typed_frontier --backbone-mode yopo_original --log-jsonl OARM/results/runs/blind_corner.jsonl --exec-log-jsonl OARM/results/runs/blind_corner_odom.jsonl
```

`--main-experiment` rejects `--fast-sim-mode` and nonzero `--progress-bonus-weight`, so paper runs use learned utility only.
`--method`, `--scenario`, `--seed`, checkpoint path, `candidate_mode`, and `backbone_mode` are written into both planner and execution logs so multi-scenario ablations remain traceable.
`--log-jsonl` records selected candidate type, predicted reaction margin, risk/yield probabilities, selected trajectory geometry, odometry snapshot, speed, latency, goal distance and online-input metadata for offline GT annotation and `eval/scenario_benchmark.py`.
`--exec-log-jsonl` records high-rate odometry rows from the odom callback after a goal is active; pass it to `execution_monitor.py --exec-input` so `collision_exec` and `min_clearance_exec` are based on true executed odometry. If no odom fields are available, the monitor marks the source as `reference_or_planner_start_to_gt_pointcloud` and that run should not be used as the primary collision result.
Run `annotate_gt_reaction_margin.py` first for selected-trajectory GT reaction-margin labels, then `execution_monitor.py --exec-input ...` for actual executed-path collision/success before sending logs to `scenario_benchmark.py`.

Current training behavior:

- The network head receives explicit anchor/type/frontier features, so progress/probe/brake/yield candidates are no longer implicit.
- Goal progress is measured from true start to decoded endpoint and normalized by candidate time.
- Braking cost is type-aware: brake/yield candidates are pushed to low terminal velocity more strongly than progress/probe candidates.
- Yaw cubic interpolation wraps terminal yaw through the shortest `[-pi, pi]` turn.
- `OARMDataset` now emits fixed-size `risk_points_w`, `risk_weight`, `yaw0`, and `yaw_rate0`; `OARMTrainer` forwards them only when the selected stage enables `train_risk_point_guidance`.
- `--use-privileged-risk-filter` enables GT ESDF filtering of depth-frontier risk point proposals for training/evaluation labels.
- Reaction-margin/yaw-visibility sampling includes `t=0`, and yaw visibility uses early-time weighting rather than only max visibility.
- Risk points are reweighted per candidate by closest approach distance before point-derived risk supervision; reaction-margin and yaw-visibility losses are stage-gated so V1 does not silently train V2 terms.
- `--use-occlusion-aware-visibility` upgrades first-visible-time from FOV-only to FOV plus GT-ESDF line-of-sight consistency.
- Yield supervision now uses a stopping-distance geometry label from visible free distance and terminal velocity, rather than the old frontier-threshold proxy.
- V2 and later stages can train an explicit margin-aware pairwise ranking loss so candidates with similar progress/base cost but larger reaction margin receive higher utility.
- `eval_dataset.py` now generates reaction-margin labels from risk points during evaluation when labels are not precomputed, so margin MAE, selected-RMVR and ranking accuracy are active metrics.
- `candidate_mode` can be set to `yopo` or `typed_frontier` to isolate typed-anchor gains from reaction-margin supervision.
- `backbone_mode` can be set to `oarm_light` or `yopo_original`; use `yopo_original` for the fair YOPO-backbone OARM-Full result.
- `train_oarm.py` writes `options.json`, copied config, git status/diff, `last.pth`, and `best_val.pth` into each run directory. Checkpoints include `candidate_mode`, `backbone_mode`, and `training_options`; evaluation and ROS loading reject mismatches unless `--allow-checkpoint-mismatch` is set.
- `eval_dataset.py --stage ...` records `online_inputs`, `privileged_training`, and `mapless_online_inference` in its JSON output to separate offline labels from online inference.

Narrative guardrail:

- OARM should be described as mapless one-stage reaction-margin guidance, not as YOPO plus FASTER backup.
- The current feasibility head is a learned stopping/yield prior, not a certified backup trajectory guarantee.
- Until the ROS node sends the predicted yaw reference into control, describe yaw as yaw-aware utility learning or training-time visibility guidance.

Baseline isolation rule:

- Do not edit files under `YOPO/` for OARM experiments.
- Put OARM-only training, inference, labels, metrics and configs under `OARM/`.
- If a YOPO helper is reused, import it through `OARM/utils/yopo_compat.py` or copy the idea into an OARM file.
