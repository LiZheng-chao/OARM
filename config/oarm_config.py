from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class OARMConfig:
    """Small OARM-specific config layer over YOPO's existing YAML config.

    The values here are deliberately conservative so OARM-V0 can run with the
    original YOPO dataset and candidate lattice before adding full occlusion
    labels and new candidate generation.
    """

    output_dim: int = 15
    candidate_mode: str = "typed_frontier"
    backbone_mode: str = "yopo_original"
    enable_yield_candidates: bool = False
    time_min_ratio: float = 0.45
    time_max_ratio: float = 1.2
    yaw_residual_limit_rad: float = 0.7853981633974483
    margin_scale: float = 4.0
    probe_time_ratio: float = 0.65
    stop_time_ratio: float = 0.95
    risk_depth_edge_threshold: float = 0.06
    risk_depth_far_threshold: float = 0.97
    risk_frontier_border_px: int = 3
    frontier_probe_threshold: float = 0.08
    frontier_brake_threshold: float = 0.20
    frontier_yield_threshold: float = 0.40
    typed_anchor_min_probe_frac: float = 0.20
    typed_anchor_min_brake_frac: float = 0.08
    typed_anchor_min_yield_frac: float = 0.04
    risk_point_count: int = 16
    risk_depth_max_m: float = 20.0
    risk_point_offsets_m: Tuple[float, ...] = (0.5, 1.0, 1.5)
    risk_assoc_sigma_m: float = 1.2
    risk_assoc_distance_m: float = 3.0
    risk_label_source: str = "proxy"
    gt_risk_point_count: int = 64
    gt_hidden_depth_margin_m: float = 0.6
    gt_min_forward_m: float = 0.5
    gt_max_forward_m: float = 10.0
    gt_horizon_fov_expand_deg: float = 90.0
    gt_vertical_fov_expand_deg: float = 20.0
    gt_depth_metric: str = "forward"
    gt_reachable_forward_center_m: float = 5.0
    gt_reachable_forward_sigma_m: float = 3.0
    gt_reachable_lateral_sigma_m: float = 3.0
    gt_reachable_vertical_sigma_m: float = 1.8
    gt_reachable_score_weight: float = 0.65
    gt_side_score_weight: float = 0.25
    gt_risk_nms_radius_m: float = 0.75
    gt_risk_voxel_size_m: float = 0.25
    use_privileged_risk_filter: bool = False
    cache_privileged_risk_labels: bool = True
    privileged_risk_cache_dir: str = "oarm_labels"
    privileged_risk_distance_m: float = 1.2
    privileged_risk_sigma_m: float = 0.25
    train_risk_from_points: bool = True
    reaction_time: float = 0.35
    risk_arrival_radius_m: float = 1.0
    no_arrival_margin_s: float = 0.50
    # Deprecated compatibility alias for older scripts; value is seconds, not meters.
    no_arrival_margin_m: float = 0.50
    margin_sigma: float = 0.25
    train_occlusion_risk: bool = False
    train_risk_point_guidance: bool = False
    train_reaction_margin: bool = False
    train_margin_ranking: bool = False
    train_yaw_visibility: bool = False
    deployed_yaw_mode: str = "goal"
    use_weak_margin_label: bool = False
    train_backup_feasibility: bool = False
    train_yield_feasibility: bool = False
    use_esdf_collision: bool = False
    collision_weight: float = 1.0
    use_occlusion_aware_visibility: bool = False
    visibility_ray_samples: int = 8
    visibility_ray_step_m: float = 0.10
    visibility_candidate_chunk: int = 16
    visibility_risk_point_chunk: int = 8
    visibility_endpoint_guard_m: float = 0.30
    visibility_occupancy_margin_m: float = 0.05
    visibility_clearance_m: float = 0.25
    occlusion_weight: float = 0.25
    margin_weight: float = 0.25
    risk_bce_weight: float = 0.2
    margin_reg_weight: float = 0.2
    yield_bce_weight: float = 0.2
    ranking_weight: float = 0.25
    ranking_progress_eps: float = 0.60
    ranking_base_cost_eps: float = 1.50
    ranking_margin_delta: float = 0.10
    ranking_speed_eps: float = 0.75
    ranking_time_eps: float = 0.35
    # A3 learned residual must stay conservative: YOPO base score remains the
    # main policy, and residual only nudges candidates when margin evidence is clear.
    yopo_preserve_utility_delta_scale: float = 0.35
    yopo_preserve_residual_reg_weight: float = 0.08
    yopo_preserve_unsafe_boost_weight: float = 0.45
    yopo_preserve_safe_suppression_weight: float = 0.05
    yopo_preserve_safe_margin_m: float = 0.20
    yaw_weight: float = 0.05
    yaw_early_time_tau: float = 0.6
    braking_weight: float = 0.05
    braking_nonstop_scale: float = 0.1
    yield_weight: float = 0.35
    no_yield_penalty: float = 1.0
    stop_overuse_weight: float = 1.0
    stop_hazard_discount: float = 0.35
    stop_hazard_margin_m: float = -0.45
    stop_hazard_risk: float = 0.45
    yield_risk_threshold: float = 0.45
    yield_stop_speed: float = 0.4
    backup_weight: float = 0.35
    no_backup_penalty: float = 1.0
    backup_risk_threshold: float = 0.45
    backup_stop_speed: float = 0.4
    visible_free_sector_px: int = 2
    yield_latency_s: float = 0.15
    yield_safe_distance_m: float = 0.6
    yield_acc_max_mps2: float = 6.0
    progress_weight: float = 1.05
    goal_distance_weight: float = 0.35
    lateral_weight: float = 0.20
    terminal_speed_weight: float = 0.08
    terminal_speed_target_mps: float = 3.0
    terminal_speed_max_mps: float = 6.0
    terminal_speed_gate_distance_m: float = 3.0
    terminal_speed_gate_width_m: float = 1.0
    agility_time_weight: float = 0.55
    stop_time_cost_scale: float = 0.5
    altitude_weight: float = 0.20
    altitude_band_weight: float = 0.15
    altitude_min_m: float = 1.0
    altitude_max_m: float = 3.0


oarm_cfg = OARMConfig()
