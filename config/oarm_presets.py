from dataclasses import dataclass
from typing import Dict

from .oarm_config import oarm_cfg


@dataclass(frozen=True)
class OARMTrainingPreset:
    candidate_mode: str = oarm_cfg.candidate_mode
    backbone_mode: str = oarm_cfg.backbone_mode
    enable_yield_candidates: bool = oarm_cfg.enable_yield_candidates
    train_occlusion_risk: bool = oarm_cfg.train_occlusion_risk
    train_risk_point_guidance: bool = oarm_cfg.train_risk_point_guidance
    train_reaction_margin: bool = oarm_cfg.train_reaction_margin
    train_margin_ranking: bool = oarm_cfg.train_margin_ranking
    train_yaw_visibility: bool = oarm_cfg.train_yaw_visibility
    deployed_yaw_mode: str = oarm_cfg.deployed_yaw_mode
    risk_label_source: str = oarm_cfg.risk_label_source
    gt_risk_point_count: int = oarm_cfg.gt_risk_point_count
    gt_hidden_depth_margin_m: float = oarm_cfg.gt_hidden_depth_margin_m
    gt_min_forward_m: float = oarm_cfg.gt_min_forward_m
    gt_max_forward_m: float = oarm_cfg.gt_max_forward_m
    gt_horizon_fov_expand_deg: float = oarm_cfg.gt_horizon_fov_expand_deg
    gt_vertical_fov_expand_deg: float = oarm_cfg.gt_vertical_fov_expand_deg
    gt_depth_metric: str = oarm_cfg.gt_depth_metric
    gt_reachable_forward_center_m: float = oarm_cfg.gt_reachable_forward_center_m
    gt_reachable_forward_sigma_m: float = oarm_cfg.gt_reachable_forward_sigma_m
    gt_reachable_lateral_sigma_m: float = oarm_cfg.gt_reachable_lateral_sigma_m
    gt_reachable_vertical_sigma_m: float = oarm_cfg.gt_reachable_vertical_sigma_m
    gt_reachable_score_weight: float = oarm_cfg.gt_reachable_score_weight
    gt_side_score_weight: float = oarm_cfg.gt_side_score_weight
    gt_risk_nms_radius_m: float = oarm_cfg.gt_risk_nms_radius_m
    gt_risk_voxel_size_m: float = oarm_cfg.gt_risk_voxel_size_m
    risk_assoc_distance_m: float = oarm_cfg.risk_assoc_distance_m
    risk_assoc_sigma_m: float = oarm_cfg.risk_assoc_sigma_m
    risk_arrival_radius_m: float = oarm_cfg.risk_arrival_radius_m
    use_weak_margin_label: bool = oarm_cfg.use_weak_margin_label
    train_backup_feasibility: bool = oarm_cfg.train_backup_feasibility
    train_yield_feasibility: bool = oarm_cfg.train_yield_feasibility
    use_esdf_collision: bool = oarm_cfg.use_esdf_collision
    use_occlusion_aware_visibility: bool = oarm_cfg.use_occlusion_aware_visibility
    use_privileged_risk_filter: bool = oarm_cfg.use_privileged_risk_filter
    ranking_weight: float = oarm_cfg.ranking_weight
    yopo_preserve_utility_delta_scale: float = oarm_cfg.yopo_preserve_utility_delta_scale
    yopo_preserve_residual_reg_weight: float = oarm_cfg.yopo_preserve_residual_reg_weight
    yopo_preserve_unsafe_boost_weight: float = oarm_cfg.yopo_preserve_unsafe_boost_weight
    yopo_preserve_safe_suppression_weight: float = oarm_cfg.yopo_preserve_safe_suppression_weight
    yopo_preserve_safe_margin_m: float = oarm_cfg.yopo_preserve_safe_margin_m
    yopo_preserve_safety_residual_weight: float = oarm_cfg.yopo_preserve_safety_residual_weight
    yopo_preserve_safe_clearance_residual_weight: float = oarm_cfg.yopo_preserve_safe_clearance_residual_weight
    yopo_preserve_safety_cost_threshold: float = oarm_cfg.yopo_preserve_safety_cost_threshold
    yopo_preserve_safe_cost_threshold: float = oarm_cfg.yopo_preserve_safe_cost_threshold
    yopo_preserve_safety_pairwise_weight: float = oarm_cfg.yopo_preserve_safety_pairwise_weight
    yopo_preserve_safety_pairwise_margin: float = oarm_cfg.yopo_preserve_safety_pairwise_margin
    yopo_preserve_unsafe_delta_target: float = oarm_cfg.yopo_preserve_unsafe_delta_target
    yopo_preserve_safe_delta_target: float = oarm_cfg.yopo_preserve_safe_delta_target
    yopo_preserve_freeze_margin_risk_head: bool = oarm_cfg.yopo_preserve_freeze_margin_risk_head
    yopo_preserve_oracle_ce_weight: float = oarm_cfg.yopo_preserve_oracle_ce_weight
    yopo_preserve_oracle_ce_temperature: float = oarm_cfg.yopo_preserve_oracle_ce_temperature
    yopo_preserve_oracle_min_margin: float = oarm_cfg.yopo_preserve_oracle_min_margin
    yopo_preserve_oracle_min_progress: float = oarm_cfg.yopo_preserve_oracle_min_progress

OARM_TRAINING_PRESETS: Dict[str, OARMTrainingPreset] = {
    "v0": OARMTrainingPreset(candidate_mode="yopo"),
    "v1_occ": OARMTrainingPreset(
        candidate_mode="typed_frontier",
        train_occlusion_risk=True,
        train_risk_point_guidance=True,
        use_privileged_risk_filter=True,
    ),
    "v2_margin": OARMTrainingPreset(
        candidate_mode="typed_frontier",
        train_occlusion_risk=True,
        train_risk_point_guidance=True,
        train_reaction_margin=True,
        train_margin_ranking=True,
        risk_label_source="gt_pointcloud",
        use_privileged_risk_filter=True,
        use_occlusion_aware_visibility=True,
    ),
    "v3_yield": OARMTrainingPreset(
        candidate_mode="typed_frontier",
        enable_yield_candidates=True,
        train_occlusion_risk=True,
        train_risk_point_guidance=True,
        train_reaction_margin=True,
        train_margin_ranking=True,
        risk_label_source="gt_pointcloud",
        train_backup_feasibility=True,
        train_yield_feasibility=True,
        use_privileged_risk_filter=True,
        use_occlusion_aware_visibility=True,
    ),
    "a3h": OARMTrainingPreset(
        candidate_mode="yopo_preserve_rerank",
        backbone_mode="yopo_original",
        train_occlusion_risk=True,
        train_risk_point_guidance=True,
        train_reaction_margin=True,
        train_margin_ranking=False,
        risk_label_source="gt_pointcloud",
        use_esdf_collision=True,
        use_privileged_risk_filter=True,
        use_occlusion_aware_visibility=True,
        deployed_yaw_mode="goal",
        yopo_preserve_utility_delta_scale=1.0,
        yopo_preserve_residual_reg_weight=0.05,
        yopo_preserve_unsafe_boost_weight=0.0,
        yopo_preserve_safe_suppression_weight=0.0,
        yopo_preserve_safety_residual_weight=0.5,
        yopo_preserve_safe_clearance_residual_weight=0.0,
        yopo_preserve_safety_pairwise_weight=0.0,
        yopo_preserve_freeze_margin_risk_head=True,
        yopo_preserve_oracle_ce_weight=2.0,
        yopo_preserve_oracle_ce_temperature=0.5,
        yopo_preserve_oracle_min_margin=0.0,
        yopo_preserve_oracle_min_progress=0.05,
    ),
    "full": OARMTrainingPreset(
        candidate_mode="typed_frontier",
        enable_yield_candidates=True,
        train_occlusion_risk=True,
        train_risk_point_guidance=True,
        train_reaction_margin=True,
        train_margin_ranking=True,
        risk_label_source="gt_pointcloud",
        train_backup_feasibility=True,
        train_yield_feasibility=True,
        use_esdf_collision=True,
        use_privileged_risk_filter=True,
        use_occlusion_aware_visibility=True,
    ),
}


def get_oarm_training_preset(stage: str) -> OARMTrainingPreset:
    try:
        return OARM_TRAINING_PRESETS[stage]
    except KeyError as exc:
        valid = ", ".join(sorted(OARM_TRAINING_PRESETS))
        raise ValueError(f"Unknown OARM training stage '{stage}'. Valid stages: {valid}") from exc
