from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from OARM.config import oarm_cfg
from OARM.loss.backup_feasibility_loss import stopping_yield_label, stopping_distance
from OARM.loss.yaw_visibility_loss import YawVisibilityLoss
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_poly_solver import quintic_coefficients, sample_polynomial, sample_yaw_cubic, yaw_cubic_coefficients
from OARM.utils.yopo_compat import ensure_yopo_path
from OARM.visibility.reaction_margin_labeler import ReactionMarginLabeler
from OARM.visibility.reaction_margin_targets import generate_reaction_margin_labels
from OARM.visibility.risk_point_association import associate_risk_points_to_trajectory

ensure_yopo_path()
from config.config import cfg


class OARMLoss(nn.Module):
    """Stage-gated OARM training objective.

    This module intentionally accepts optional labels so the same code can run
    in three stages:
    V0: variable-time trajectory utility without occlusion labels.
    V1: add occlusion risk supervision.
    V2: add reaction-margin supervision and penalty.
    V3: add stopping/yield feasibility supervision.
    """

    def __init__(
        self,
        smoothness_weight: float = 1.0,
        acceleration_weight: float = 0.1,
        goal_weight: float = 0.15,
        eval_points: int = 30,
        use_esdf_collision: bool = oarm_cfg.use_esdf_collision,
        use_occlusion_aware_visibility: bool = oarm_cfg.use_occlusion_aware_visibility,
        enable_occlusion_risk: bool = oarm_cfg.train_occlusion_risk,
        enable_risk_point_guidance: bool = oarm_cfg.train_risk_point_guidance,
        enable_reaction_margin: bool = oarm_cfg.train_reaction_margin,
        enable_margin_ranking: bool = oarm_cfg.train_margin_ranking,
        enable_yaw_visibility: bool = False,
        deployed_yaw_mode: str = oarm_cfg.deployed_yaw_mode,
        enable_yield_feasibility: Optional[bool] = None,
        risk_assoc_distance_m: float = oarm_cfg.risk_assoc_distance_m,
        risk_assoc_sigma_m: float = oarm_cfg.risk_assoc_sigma_m,
        risk_arrival_radius_m: float = oarm_cfg.risk_arrival_radius_m,
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.acceleration_weight = acceleration_weight
        self.goal_weight = goal_weight
        self.eval_points = eval_points
        self.enable_occlusion_risk = enable_occlusion_risk
        self.enable_risk_point_guidance = enable_risk_point_guidance
        self.enable_reaction_margin = enable_reaction_margin
        self.enable_margin_ranking = enable_margin_ranking
        self.enable_yaw_visibility = enable_yaw_visibility
        self.risk_assoc_distance_m = risk_assoc_distance_m
        self.risk_assoc_sigma_m = risk_assoc_sigma_m
        self.risk_arrival_radius_m = risk_arrival_radius_m
        if deployed_yaw_mode not in {"goal", "hold", "predicted"}:
            raise ValueError(f"Unknown deployed_yaw_mode: {deployed_yaw_mode}")
        self.deployed_yaw_mode = deployed_yaw_mode
        if enable_yield_feasibility is None:
            enable_yield_feasibility = oarm_cfg.train_yield_feasibility or oarm_cfg.train_backup_feasibility
        self.enable_yield_feasibility = enable_yield_feasibility
        if use_esdf_collision:
            from OARM.loss.collision_loss import ESDFCollisionLoss

            self.collision_loss = ESDFCollisionLoss()
        else:
            self.collision_loss = None
        self.yaw_visibility_loss = YawVisibilityLoss()
        self.margin_labeler = ReactionMarginLabeler(risk_arrival_radius_m=self.risk_arrival_radius_m)
        if use_occlusion_aware_visibility:
            from OARM.visibility.esdf_visibility import ESDFLineOfSight

            self.line_of_sight = ESDFLineOfSight()
        else:
            self.line_of_sight = None

    def forward(
        self,
        start_state_w: torch.Tensor,
        end_state_w: torch.Tensor,
        candidate_flat: Dict[str, torch.Tensor],
        goal_w: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        map_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        traj_time = candidate_flat["traj_time"]
        coeff = quintic_coefficients(start_state_w, end_state_w, traj_time)
        pos, vel, acc, jerk = sample_polynomial(coeff, traj_time, self.eval_points)
        margin_pos, margin_vel, _, _ = sample_polynomial(coeff, traj_time, self.eval_points, include_zero=True)
        sampled_time = self.sampled_time_grid(traj_time, self.eval_points, include_zero=True)

        smooth_cost = (jerk.square().sum(dim=-1).mean(dim=-1) * traj_time).clamp(max=1e6)
        accel_cost = (acc.square().sum(dim=-1).mean(dim=-1) * traj_time).clamp(max=1e6)
        goal_cost = self.goal_progress_cost(start_state_w, end_state_w, goal_w, traj_time)
        goal_distance_cost = self.goal_distance_cost(start_state_w, end_state_w, goal_w, traj_time)
        lateral_cost = self.lateral_goal_cost(start_state_w, end_state_w, goal_w, traj_time)
        altitude_cost = self.altitude_tracking_cost(start_state_w, end_state_w, goal_w)
        altitude_band_cost = self.altitude_band_cost(pos)
        braking_cost = self.braking_cost(end_state_w, candidate_flat.get("candidate_type"))
        candidate_type = candidate_flat.get("candidate_type")
        stop_type = self.stop_type_mask(candidate_type, goal_cost)
        non_stop_type = 1.0 - stop_type
        terminal_speed_cost = self.terminal_speed_cost(end_state_w, goal_w, non_stop_type)

        total_cost = (
            self.smoothness_weight * smooth_cost
            + self.acceleration_weight * accel_cost
            + self.goal_weight * goal_cost
            + oarm_cfg.goal_distance_weight * non_stop_type * goal_distance_cost
            + oarm_cfg.lateral_weight * non_stop_type * lateral_cost
            + oarm_cfg.terminal_speed_weight * terminal_speed_cost
            + oarm_cfg.altitude_weight * altitude_cost
            + oarm_cfg.altitude_band_weight * altitude_band_cost
            + oarm_cfg.braking_weight * braking_cost
        )
        progress_score = (-goal_cost).clamp(min=0.0)
        progress_bonus_cost = -oarm_cfg.progress_weight * non_stop_type * progress_score
        time_cost = traj_time / max(float(cfg["sgm_time"]), 1e-3)
        agility_time_cost = (non_stop_type + oarm_cfg.stop_time_cost_scale * stop_type) * time_cost
        total_cost = total_cost + progress_bonus_cost + oarm_cfg.agility_time_weight * agility_time_cost

        losses = {
            "smooth_cost": smooth_cost.mean(),
            "accel_cost": accel_cost.mean(),
            "goal_cost": goal_cost.mean(),
            "goal_distance_cost": goal_distance_cost.mean(),
            "lateral_cost": lateral_cost.mean(),
            "terminal_speed_cost": terminal_speed_cost.mean(),
            "altitude_cost": altitude_cost.mean(),
            "altitude_band_cost": altitude_band_cost.mean(),
            "braking_cost": braking_cost.mean(),
            "progress_bonus_cost": progress_bonus_cost.mean(),
            "agility_time_cost": agility_time_cost.mean(),
            "stop_type_rate": stop_type.mean(),
            "sampled_pos_w": pos.detach(),
        }

        if labels is not None and "safety_cost" in labels:
            safety_cost = labels["safety_cost"].reshape_as(total_cost)
            total_cost = total_cost + safety_cost
            losses["safety_cost"] = safety_cost.mean()
            losses["safety_cost_per_candidate"] = safety_cost.detach()
        elif self.collision_loss is not None and map_id is not None:
            safety_cost = self.collision_loss(pos, map_id.reshape(-1))
            total_cost = total_cost + oarm_cfg.collision_weight * safety_cost
            losses["safety_cost"] = safety_cost.mean()
            losses["safety_cost_per_candidate"] = safety_cost.detach()

        if self.enable_occlusion_risk and labels is not None and "occlusion_risk" in labels:
            risk_label = labels["occlusion_risk"].reshape_as(candidate_flat["risk_logit"]).float()
            risk_loss = F.binary_cross_entropy_with_logits(candidate_flat["risk_logit"], risk_label)
            risk_cost = risk_label.detach()
            total_cost = total_cost + oarm_cfg.occlusion_weight * risk_cost
            losses["risk_loss"] = risk_loss
            losses["risk_cost"] = risk_cost.mean()
        else:
            losses["risk_loss"] = torch.zeros((), device=total_cost.device)

        if self.enable_risk_point_guidance and labels is not None and "risk_points_w" in labels:
            risk_points_w = self.expand_candidate_label(labels["risk_points_w"], traj_time.shape[0], traj_time)
            risk_weight = labels.get("risk_weight")
            if risk_weight is not None:
                risk_weight = self.expand_candidate_label(risk_weight, traj_time.shape[0], traj_time)
            else:
                risk_weight = torch.ones(risk_points_w.shape[:-1], device=traj_time.device, dtype=traj_time.dtype)
            association = associate_risk_points_to_trajectory(
                margin_pos,
                sampled_time,
                risk_points_w,
                risk_weight,
                sigma_m=self.risk_assoc_sigma_m,
                max_distance_m=self.risk_assoc_distance_m,
            )
            risk_weight = association.associated_weight
            yaw0 = labels.get("yaw0")
            if yaw0 is None:
                yaw0 = torch.zeros_like(traj_time)
            else:
                yaw0 = self.expand_candidate_label(yaw0, traj_time.shape[0], traj_time)
            yaw_rate0 = labels.get("yaw_rate0")
            if yaw_rate0 is None:
                yaw_rate0 = torch.zeros_like(traj_time)
            else:
                yaw_rate0 = self.expand_candidate_label(yaw_rate0, traj_time.shape[0], traj_time)
            yaw_ref, yaw_rate = self.deployed_yaw_reference(
                yaw0,
                yaw_rate0,
                candidate_flat["yaw_terminal"],
                traj_time,
                margin_pos,
                margin_vel,
                sampled_time,
                goal_w,
            )
            visibility_mask = None
            if (self.enable_yaw_visibility or self.enable_reaction_margin) and self.line_of_sight is not None and map_id is not None:
                visibility_mask = self.line_of_sight(margin_pos, risk_points_w, map_id.reshape(-1))
                losses["occlusion_visible_rate"] = visibility_mask.float().mean()
            if self.enable_yaw_visibility:
                yaw_visibility_cost = self.yaw_visibility_loss(
                    margin_pos,
                    yaw_ref,
                    yaw_rate,
                    risk_points_w,
                    risk_weight,
                    sampled_time=sampled_time,
                    arrival_time=association.arrival_time,
                    visibility_mask=visibility_mask,
                )
                total_cost = total_cost + oarm_cfg.yaw_weight * yaw_visibility_cost
                losses["yaw_visibility_cost"] = yaw_visibility_cost.mean()
            else:
                losses["yaw_visibility_cost"] = torch.zeros((), device=total_cost.device)
            risk_weight_sum = risk_weight.sum(dim=-1)
            losses["risk_assoc_valid_rate"] = association.valid_mask.float().mean()
            losses["risk_assoc_weight_mean"] = risk_weight.mean()
            losses["risk_weight_sum_mean"] = risk_weight_sum.mean()
            losses["risk_weight_nonzero_rate"] = (risk_weight_sum > 1e-6).float().mean()
            losses["no_associated_risk_rate"] = (risk_weight_sum <= 1e-6).float().mean()
            losses["candidate_hidden_risk_gt_rate"] = (risk_weight_sum > 1e-6).float().mean()
            losses['candidate_risk_assoc_valid_rate'] = association.valid_mask.float().mean()
            losses['candidate_risk_assoc_weight_mean'] = association.association_weight.mean()
            if 'raw_gt_risk_point_valid_rate' in labels:
                losses['raw_gt_risk_point_valid_rate'] = labels['raw_gt_risk_point_valid_rate'].float().mean()
            if 'raw_gt_risk_point_weight_sum' in labels:
                losses['raw_gt_risk_point_weight_sum'] = labels['raw_gt_risk_point_weight_sum'].float().mean()
            if 'raw_gt_risk_point_weight_mean' in labels:
                losses['raw_gt_risk_point_weight_mean'] = labels['raw_gt_risk_point_weight_mean'].float().mean()
            if "uses_gt_reaction_margin" in labels:
                gt_source = self.expand_candidate_label(labels["uses_gt_reaction_margin"], traj_time.shape[0], traj_time)
                losses["reaction_margin_gt_source_rate"] = gt_source.float().mean()
            if "uses_proxy_reaction_margin" in labels:
                proxy_source = self.expand_candidate_label(labels["uses_proxy_reaction_margin"], traj_time.shape[0], traj_time)
                losses["reaction_margin_proxy_source_rate"] = proxy_source.float().mean()
            if "hidden_risk_gt" in labels:
                losses["hidden_risk_gt_rate"] = labels["hidden_risk_gt"].float().mean()
            losses["risk_min_distance"] = association.min_distance.mean()
            if oarm_cfg.train_risk_from_points and (labels is None or "occlusion_risk" not in labels):
                point_risk_label = 1.0 - torch.exp(-risk_weight.sum(dim=-1))
                risk_loss = F.binary_cross_entropy_with_logits(candidate_flat["risk_logit"], point_risk_label.detach())
                total_cost = total_cost + oarm_cfg.occlusion_weight * point_risk_label.detach()
                losses["risk_loss"] = risk_loss
                losses["risk_cost"] = point_risk_label.mean()
            if self.enable_reaction_margin and (labels is None or "reaction_margin" not in labels):
                labels = generate_reaction_margin_labels(
                    dict(labels or {}),
                    candidate_flat,
                    start_state_w,
                    end_state_w,
                    map_id if map_id is not None else torch.zeros_like(traj_time, dtype=torch.long),
                    goal_w,
                    enabled=True,
                    labeler=self.margin_labeler,
                    line_of_sight=self.line_of_sight if map_id is not None else None,
                    yaw_helper=self,
                    eval_points=self.eval_points,
                    include_diagnostics=True,
                    risk_weight_override=risk_weight,
                )
                valid = labels["reaction_margin_valid"].bool()
                zero = torch.zeros((), device=total_cost.device)
                losses["generated_margin_valid_rate"] = valid.float().mean()
                losses["generated_margin_censored_rate"] = labels["reaction_margin_censored"].float().mean()
                if bool(valid.any()):
                    valid_margin = labels["reaction_margin"][valid]
                    losses["generated_margin_mean"] = valid_margin.mean()
                    losses["generated_margin_min"] = labels["reaction_margin_min"][valid].mean()
                    losses["generated_margin_violation_rate"] = (valid_margin < 0.0).float().mean()
                    losses["generated_arrival_time_min"] = labels["reaction_margin_arrival_time_min"][valid].mean()
                else:
                    losses["generated_margin_mean"] = zero
                    losses["generated_margin_min"] = zero
                    losses["generated_margin_violation_rate"] = zero
                    losses["generated_arrival_time_min"] = zero
        else:
            losses["yaw_visibility_cost"] = torch.zeros((), device=total_cost.device)

        ranking_base_cost = self.ranking_base_cost_proxy(start_state_w, end_state_w, goal_w, traj_time).detach()

        margin_valid_mask = None
        if self.enable_reaction_margin and labels is not None and "reaction_margin" in labels:
            margin_label = labels["reaction_margin"].reshape_as(candidate_flat["margin_pred"]).float()
            label_valid = labels.get("reaction_margin_valid")
            if label_valid is None:
                label_valid = torch.ones_like(margin_label, dtype=torch.bool)
            else:
                label_valid = label_valid.to(device=margin_label.device).reshape_as(margin_label).bool()
            margin_valid_mask = (
                label_valid
                & torch.isfinite(margin_label)
                & torch.isfinite(candidate_flat["margin_pred"])
            )
            margin_violation = torch.zeros_like(margin_label)
            if bool(margin_valid_mask.any()):
                margin_label_reg = margin_label.clamp(-oarm_cfg.margin_scale, oarm_cfg.margin_scale)
                margin_loss = F.smooth_l1_loss(
                    candidate_flat["margin_pred"][margin_valid_mask],
                    margin_label_reg[margin_valid_mask],
                )
                margin_violation[margin_valid_mask] = F.softplus(
                    -margin_label[margin_valid_mask] / oarm_cfg.margin_sigma
                ).square()
            else:
                margin_loss = torch.zeros((), device=total_cost.device)
            total_cost = total_cost + oarm_cfg.margin_weight * margin_violation.detach()
            losses["margin_loss"] = margin_loss
            losses["margin_violation"] = margin_violation.mean()
            losses["margin_valid_rate"] = margin_valid_mask.float().mean()
            label_censored = labels.get("reaction_margin_censored")
            if label_censored is not None:
                losses["margin_censored_rate"] = label_censored.to(device=margin_label.device).reshape_as(margin_label).float().mean()
        else:
            margin_label = None
            losses["margin_loss"] = torch.zeros((), device=total_cost.device)
            losses["margin_valid_rate"] = torch.zeros((), device=total_cost.device)

        if "yield_logit" in candidate_flat:
            yield_logit = candidate_flat["yield_logit"]
        else:
            yield_logit = candidate_flat["backup_logit"]

        if self.enable_yield_feasibility and labels is not None and "visible_free_distance" in labels:
            visible_free_distance = labels["visible_free_distance"].reshape_as(candidate_flat["backup_logit"]).float()
            yield_label = stopping_yield_label(end_state_w[:, 1], visible_free_distance)
            stop_distance = stopping_distance(end_state_w[:, 1])
            losses["visible_free_distance"] = visible_free_distance.mean()
            losses["stop_distance"] = stop_distance.mean()
            losses["backup_feasible_rate"] = yield_label.mean()
            losses["yield_feasible_rate"] = yield_label.mean()
        elif self.enable_yield_feasibility and labels is not None and "yield_feasible" in labels:
            yield_label = labels["yield_feasible"].reshape_as(candidate_flat["backup_logit"]).float()
        elif self.enable_yield_feasibility and labels is not None and "backup_feasible" in labels:
            yield_label = labels["backup_feasible"].reshape_as(candidate_flat["backup_logit"]).float()
        else:
            yield_label = None

        if yield_label is not None:
            yield_loss = F.binary_cross_entropy_with_logits(yield_logit, yield_label)
            no_yield_cost = stop_type * (1.0 - yield_label).detach() * oarm_cfg.no_yield_penalty
            hazard = self.stop_hazard_signal(margin_label, candidate_flat.get("risk_logit"), total_cost)
            stop_overuse_cost = stop_type * oarm_cfg.stop_overuse_weight * (
                1.0 - oarm_cfg.stop_hazard_discount * hazard
            )
            total_cost = total_cost + oarm_cfg.yield_weight * no_yield_cost + stop_overuse_cost
            losses["backup_loss"] = yield_loss
            losses["yield_loss"] = yield_loss
            losses["no_backup_cost"] = no_yield_cost.mean()
            losses["no_yield_cost"] = no_yield_cost.mean()
            losses["stop_overuse_cost"] = stop_overuse_cost.mean()
            losses["stop_hazard_rate"] = hazard.mean()
        else:
            losses["backup_loss"] = torch.zeros((), device=total_cost.device)
            losses["yield_loss"] = torch.zeros((), device=total_cost.device)
            losses["stop_overuse_cost"] = torch.zeros((), device=total_cost.device)
            losses["stop_hazard_rate"] = torch.zeros((), device=total_cost.device)

        utility_label = -total_cost.detach()
        utility_loss = F.smooth_l1_loss(candidate_flat["utility_score"], utility_label)
        if self.enable_margin_ranking and margin_label is not None:
            ranking_loss, ranking_acc, ranking_pair_rate = self.margin_ranking_loss(
                candidate_flat["utility_score"],
                margin_label.detach(),
                progress=(-goal_cost).detach(),
                base_cost=ranking_base_cost,
                mean_speed=margin_vel.norm(dim=-1).mean(dim=-1).detach(),
                traj_time=traj_time.detach(),
                margin_valid=margin_valid_mask.detach() if margin_valid_mask is not None else None,
            )
        else:
            ranking_loss = torch.zeros((), device=total_cost.device)
            ranking_acc = torch.zeros((), device=total_cost.device)
            ranking_pair_rate = torch.zeros((), device=total_cost.device)
        trajectory_loss = total_cost.mean()
        total_loss = (
            trajectory_loss
            + utility_loss
            + oarm_cfg.risk_bce_weight * losses["risk_loss"]
            + oarm_cfg.margin_reg_weight * losses["margin_loss"]
            + oarm_cfg.yield_bce_weight * losses["backup_loss"]
            + oarm_cfg.ranking_weight * ranking_loss
        )

        losses.update(
            {
                "trajectory_loss": trajectory_loss,
                "utility_loss": utility_loss,
                "ranking_loss": ranking_loss,
                "pairwise_ranking_accuracy": ranking_acc,
                "ranking_pair_rate": ranking_pair_rate,
                "total_loss": total_loss,
                "mean_time": traj_time.mean(),
                "mean_utility": candidate_flat["utility_score"].mean(),
            }
        )
        return losses

    def deployed_yaw_reference(
        self,
        yaw0: torch.Tensor,
        yaw_rate0: torch.Tensor,
        yaw_terminal: torch.Tensor,
        traj_time: torch.Tensor,
        sampled_pos_w: torch.Tensor,
        sampled_vel_w: torch.Tensor,
        sampled_time: torch.Tensor,
        goal_w: torch.Tensor,
    ):
        if self.deployed_yaw_mode == "predicted":
            yaw_coeff = yaw_cubic_coefficients(yaw0, yaw_rate0, yaw_terminal, traj_time)
            return sample_yaw_cubic(yaw_coeff, traj_time, self.eval_points, include_zero=True)
        if self.deployed_yaw_mode == "hold":
            yaw_ref = yaw0[:, None].expand_as(sampled_time)
            return yaw_ref, torch.zeros_like(yaw_ref)
        return self.goal_oriented_yaw_like_ros(sampled_pos_w, sampled_vel_w, sampled_time, goal_w, yaw0)

    @staticmethod
    def goal_oriented_yaw_like_ros(
        sampled_pos_w: torch.Tensor,
        sampled_vel_w: torch.Tensor,
        sampled_time: torch.Tensor,
        goal_w: torch.Tensor,
        yaw0: torch.Tensor,
    ):
        goal_pos_w = sampled_pos_w[:, :1, :] + goal_w[:, None, :]
        last_yaw = yaw0.reshape(-1)
        last_time = sampled_time[:, 0]
        yaw_values = []
        yaw_rates = []
        for i in range(sampled_time.shape[1]):
            current_time = sampled_time[:, i]
            dt = (current_time - last_time).clamp(min=1e-3)
            vel_dir = sampled_vel_w[:, i]
            vel_dir = vel_dir / vel_dir.norm(dim=-1, keepdim=True).clamp(min=1e-5)
            goal_dir = goal_pos_w[:, 0] - sampled_pos_w[:, i]
            goal_dist = goal_dir.norm(dim=-1)
            goal_dir = goal_dir / goal_dist[:, None].clamp(min=1e-5)
            goal_yaw = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
            delta_yaw = OARMLoss.wrap_to_pi(goal_yaw - last_yaw)
            weight = 6.0 * delta_yaw.abs() / torch.pi
            dir_des = vel_dir + weight[:, None] * goal_dir
            yaw_desired = torch.atan2(dir_des[:, 1], dir_des[:, 0])
            yaw_desired = torch.where(goal_dist > 0.5, yaw_desired, last_yaw)
            yaw_diff = OARMLoss.wrap_to_pi(yaw_desired - last_yaw)
            max_change = 0.5 * torch.pi * dt
            yaw_change = torch.maximum(torch.minimum(yaw_diff, max_change), -max_change)
            last_yaw = OARMLoss.wrap_to_pi(last_yaw + yaw_change)
            last_time = current_time
            yaw_values.append(last_yaw)
            yaw_rates.append(yaw_change / dt)
        return torch.stack(yaw_values, dim=1), torch.stack(yaw_rates, dim=1)

    @staticmethod
    def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi

    @staticmethod
    def goal_progress_cost(
        start_state_w: torch.Tensor,
        end_state_w: torch.Tensor,
        goal_w: torch.Tensor,
        traj_time: torch.Tensor = None,
    ) -> torch.Tensor:
        goal_dir = goal_w / goal_w.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        progress = (end_state_w[:, 0] - start_state_w[:, 0]) * goal_dir
        progress = progress.sum(dim=-1)
        if traj_time is not None:
            progress = progress / traj_time.clamp(min=1e-3)
        return -progress

    @staticmethod
    def goal_distance_cost(
        start_state_w: torch.Tensor,
        end_state_w: torch.Tensor,
        goal_w: torch.Tensor,
        traj_time: torch.Tensor = None,
    ) -> torch.Tensor:
        goal_pos_w = start_state_w[:, 0] + goal_w
        start_distance = goal_w.norm(dim=-1)
        end_distance = (goal_pos_w - end_state_w[:, 0]).norm(dim=-1)
        distance_drop = start_distance - end_distance
        if traj_time is not None:
            distance_drop = distance_drop / traj_time.clamp(min=1e-3)
        return -distance_drop

    @staticmethod
    def lateral_goal_cost(
        start_state_w: torch.Tensor,
        end_state_w: torch.Tensor,
        goal_w: torch.Tensor,
        traj_time: torch.Tensor = None,
    ) -> torch.Tensor:
        goal_dir = goal_w / goal_w.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        offset = end_state_w[:, 0] - start_state_w[:, 0]
        progress = (offset * goal_dir).sum(dim=-1, keepdim=True)
        lateral = offset - progress * goal_dir
        lateral_norm = lateral.norm(dim=-1)
        if traj_time is not None:
            lateral_norm = lateral_norm / traj_time.clamp(min=1e-3)
        return lateral_norm

    @staticmethod
    def ranking_base_cost_proxy(
        start_state_w: torch.Tensor,
        end_state_w: torch.Tensor,
        goal_w: torch.Tensor,
        traj_time: torch.Tensor,
    ) -> torch.Tensor:
        goal_distance = OARMLoss.goal_distance_cost(start_state_w, end_state_w, goal_w, traj_time)
        lateral = OARMLoss.lateral_goal_cost(start_state_w, end_state_w, goal_w, traj_time)
        altitude = OARMLoss.altitude_tracking_cost(start_state_w, end_state_w, goal_w)
        time_cost = traj_time / max(float(cfg['sgm_time']), 1e-3)
        return goal_distance + lateral + altitude + time_cost

    @staticmethod
    def terminal_speed_cost(
        end_state_w: torch.Tensor,
        goal_w: torch.Tensor,
        non_stop_type: torch.Tensor,
    ) -> torch.Tensor:
        speed = end_state_w[:, 1].norm(dim=-1)
        under = F.relu(oarm_cfg.terminal_speed_target_mps - speed).square()
        over = F.relu(speed - oarm_cfg.terminal_speed_max_mps).square()
        goal_distance = goal_w.norm(dim=-1)
        gate_width = max(oarm_cfg.terminal_speed_gate_width_m, 1e-3)
        speed_gate = torch.sigmoid((goal_distance - oarm_cfg.terminal_speed_gate_distance_m) / gate_width)
        return speed_gate * non_stop_type * (under + over)

    @staticmethod
    def altitude_tracking_cost(
        start_state_w: torch.Tensor,
        end_state_w: torch.Tensor,
        goal_w: torch.Tensor,
    ) -> torch.Tensor:
        target_z = start_state_w[:, 0, 2] + goal_w[:, 2]
        return (end_state_w[:, 0, 2] - target_z).square()

    @staticmethod
    def altitude_band_cost(sampled_pos_w: torch.Tensor) -> torch.Tensor:
        z = sampled_pos_w[..., 2]
        below = F.relu(oarm_cfg.altitude_min_m - z).square()
        above = F.relu(z - oarm_cfg.altitude_max_m).square()
        return below.mean(dim=-1) + above.mean(dim=-1)

    @staticmethod
    def braking_cost(end_state_w: torch.Tensor, candidate_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        end_vel = end_state_w[:, 1]
        cost = end_vel.square().sum(dim=-1)
        if candidate_type is None:
            return cost
        stop_type = (candidate_type == OARMCandidateGenerator.BRAKE) | (
            candidate_type == OARMCandidateGenerator.YIELD
        )
        weight = torch.full_like(cost, oarm_cfg.braking_nonstop_scale)
        weight = torch.where(stop_type, torch.ones_like(weight), weight)
        return weight * cost

    @staticmethod
    def stop_type_mask(candidate_type: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if candidate_type is None:
            return torch.zeros_like(like)
        candidate_type = candidate_type.reshape_as(like)
        return (
            (candidate_type == OARMCandidateGenerator.BRAKE)
            | (candidate_type == OARMCandidateGenerator.YIELD)
        ).to(dtype=like.dtype, device=like.device)

    @staticmethod
    def stop_hazard_signal(
        margin_label: Optional[torch.Tensor],
        risk_logit: Optional[torch.Tensor],
        like: torch.Tensor,
    ) -> torch.Tensor:
        hazard = torch.zeros_like(like)
        if margin_label is not None:
            margin = margin_label.reshape_as(like)
            hazard = torch.maximum(hazard, (margin < oarm_cfg.stop_hazard_margin_m).to(dtype=like.dtype))
        if risk_logit is not None:
            risk = torch.sigmoid(risk_logit.reshape_as(like))
            hazard = torch.maximum(hazard, (risk > oarm_cfg.stop_hazard_risk).to(dtype=like.dtype))
        return hazard.detach()

    @staticmethod
    def margin_ranking_loss(
        utility_score: torch.Tensor,
        margin_label: torch.Tensor,
        progress: torch.Tensor,
        base_cost: torch.Tensor,
        mean_speed: Optional[torch.Tensor] = None,
        traj_time: Optional[torch.Tensor] = None,
        margin_valid: Optional[torch.Tensor] = None,
    ):
        traj_num = int(cfg["traj_num"])
        if traj_num <= 1 or utility_score.numel() % traj_num != 0:
            zero = torch.zeros((), device=utility_score.device)
            return zero, zero, zero

        batch_size = utility_score.numel() // traj_num
        utility = utility_score.reshape(batch_size, traj_num)
        margin = margin_label.reshape(batch_size, traj_num)
        progress = progress.reshape(batch_size, traj_num)
        base_cost = base_cost.reshape(batch_size, traj_num)
        valid = (
            torch.isfinite(utility)
            & torch.isfinite(margin)
            & torch.isfinite(progress)
            & torch.isfinite(base_cost)
        )
        if margin_valid is not None:
            valid = valid & margin_valid.reshape(batch_size, traj_num).bool()

        margin_delta = margin[:, :, None] - margin[:, None, :]
        comparable = (
            (progress[:, :, None] - progress[:, None, :]).abs() < oarm_cfg.ranking_progress_eps
        ) & ((base_cost[:, :, None] - base_cost[:, None, :]).abs() < oarm_cfg.ranking_base_cost_eps)
        if mean_speed is not None:
            mean_speed = mean_speed.reshape(batch_size, traj_num)
            valid = valid & torch.isfinite(mean_speed)
            comparable = comparable & (
                (mean_speed[:, :, None] - mean_speed[:, None, :]).abs() < oarm_cfg.ranking_speed_eps
            )
        if traj_time is not None:
            traj_time = traj_time.reshape(batch_size, traj_num)
            valid = valid & torch.isfinite(traj_time)
            comparable = comparable & (
                (traj_time[:, :, None] - traj_time[:, None, :]).abs() < oarm_cfg.ranking_time_eps
            )
        preference = margin_delta > oarm_cfg.ranking_margin_delta
        pair_mask = comparable & preference & valid[:, :, None] & valid[:, None, :]

        if not bool(pair_mask.any()):
            zero = torch.zeros((), device=utility_score.device)
            return zero, zero, zero

        utility_delta = utility[:, :, None] - utility[:, None, :]
        pair_loss = F.softplus(-utility_delta[pair_mask]).mean()
        pair_acc = (utility_delta[pair_mask] > 0.0).float().mean()
        pair_rate = pair_mask.float().mean()
        return pair_loss, pair_acc, pair_rate

    @staticmethod
    def sampled_time_grid(traj_time: torch.Tensor, eval_points: int, include_zero: bool = False) -> torch.Tensor:
        if include_zero:
            tau = torch.linspace(0.0, 1.0, eval_points, device=traj_time.device, dtype=traj_time.dtype)
        else:
            tau = torch.linspace(1.0 / eval_points, 1.0, eval_points, device=traj_time.device, dtype=traj_time.dtype)
        return traj_time[:, None] * tau[None, :]

    @staticmethod
    def expand_candidate_label(label: torch.Tensor, candidate_count: int, like: torch.Tensor) -> torch.Tensor:
        label = label.to(device=like.device, dtype=like.dtype)
        if label.shape[0] == candidate_count:
            return label
        if candidate_count % label.shape[0] != 0:
            raise ValueError(f"Cannot expand label with first dim {label.shape[0]} to {candidate_count} candidates")
        repeat = candidate_count // label.shape[0]
        return label.repeat_interleave(repeat, dim=0)
