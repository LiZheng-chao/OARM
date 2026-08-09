import atexit
import contextlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from OARM.dataset import OARMDataset
from OARM.config import oarm_cfg
from OARM.loss.reaction_margin_loss import weak_margin_label_from_risk
from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_state_transform import rotate_body2world, state_body2world
from OARM.utils.checkpoint import load_oarm_checkpoint, make_oarm_checkpoint, validate_checkpoint_metadata
from OARM.utils.visible_free_distance import visible_free_distance_from_depth
from OARM.utils.yopo_compat import ensure_yopo_path
from OARM.utils.yopo_dataset_context import yopo_dataset_cfg

ensure_yopo_path()
from config.config import cfg


@contextlib.contextmanager
def _null_record_function(*_args, **_kwargs):
    yield


class OARMTrainer:
    def __init__(
        self,
        learning_rate=1.5e-4,
        batch_size=16,
        tensorboard_path=None,
        checkpoint_path=None,
        yopo_checkpoint_path=None,
        save_on_exit=False,
        num_workers=4,
        max_train_batches=None,
        max_val_batches=None,
        dataset_root=None,
        candidate_mode=oarm_cfg.candidate_mode,
        backbone_mode=oarm_cfg.backbone_mode,
        enable_yield_candidates=oarm_cfg.enable_yield_candidates,
        train_occlusion_risk=oarm_cfg.train_occlusion_risk,
        train_risk_point_guidance=oarm_cfg.train_risk_point_guidance,
        train_reaction_margin=oarm_cfg.train_reaction_margin,
        train_margin_ranking=oarm_cfg.train_margin_ranking,
        train_yaw_visibility=oarm_cfg.train_yaw_visibility,
        deployed_yaw_mode=oarm_cfg.deployed_yaw_mode,
        risk_label_source=oarm_cfg.risk_label_source,
        gt_risk_point_count=oarm_cfg.gt_risk_point_count,
        gt_hidden_depth_margin_m=oarm_cfg.gt_hidden_depth_margin_m,
        gt_min_forward_m=oarm_cfg.gt_min_forward_m,
        gt_max_forward_m=oarm_cfg.gt_max_forward_m,
        gt_horizon_fov_expand_deg=oarm_cfg.gt_horizon_fov_expand_deg,
        gt_vertical_fov_expand_deg=oarm_cfg.gt_vertical_fov_expand_deg,
        gt_depth_metric=oarm_cfg.gt_depth_metric,
        gt_reachable_forward_center_m=oarm_cfg.gt_reachable_forward_center_m,
        gt_reachable_forward_sigma_m=oarm_cfg.gt_reachable_forward_sigma_m,
        gt_reachable_lateral_sigma_m=oarm_cfg.gt_reachable_lateral_sigma_m,
        gt_reachable_vertical_sigma_m=oarm_cfg.gt_reachable_vertical_sigma_m,
        gt_reachable_score_weight=oarm_cfg.gt_reachable_score_weight,
        gt_side_score_weight=oarm_cfg.gt_side_score_weight,
        gt_risk_nms_radius_m=oarm_cfg.gt_risk_nms_radius_m,
        gt_risk_voxel_size_m=oarm_cfg.gt_risk_voxel_size_m,
        risk_assoc_distance_m=oarm_cfg.risk_assoc_distance_m,
        risk_assoc_sigma_m=oarm_cfg.risk_assoc_sigma_m,
        risk_arrival_radius_m=oarm_cfg.risk_arrival_radius_m,
        use_weak_margin_label=oarm_cfg.use_weak_margin_label,
        train_backup_feasibility=oarm_cfg.train_backup_feasibility,
        train_yield_feasibility=oarm_cfg.train_yield_feasibility,
        use_esdf_collision=oarm_cfg.use_esdf_collision,
        use_occlusion_aware_visibility=oarm_cfg.use_occlusion_aware_visibility,
        use_privileged_risk_filter=oarm_cfg.use_privileged_risk_filter,
        yopo_preserve_utility_delta_scale=oarm_cfg.yopo_preserve_utility_delta_scale,
        yopo_preserve_safety_residual_weight=oarm_cfg.yopo_preserve_safety_residual_weight,
        yopo_preserve_safe_clearance_residual_weight=oarm_cfg.yopo_preserve_safe_clearance_residual_weight,
        yopo_preserve_safety_cost_threshold=oarm_cfg.yopo_preserve_safety_cost_threshold,
        yopo_preserve_safe_cost_threshold=oarm_cfg.yopo_preserve_safe_cost_threshold,
        yopo_preserve_safety_pairwise_weight=oarm_cfg.yopo_preserve_safety_pairwise_weight,
        yopo_preserve_safety_pairwise_margin=oarm_cfg.yopo_preserve_safety_pairwise_margin,
        yopo_preserve_unsafe_delta_target=oarm_cfg.yopo_preserve_unsafe_delta_target,
        yopo_preserve_safe_delta_target=oarm_cfg.yopo_preserve_safe_delta_target,
        experiment_options=None,
        config_path="",
        log_interval=50,
        allow_checkpoint_mismatch=False,
        grad_clip_norm=1.0,
        use_fused_adamw=False,
        train_yield_head_only=False,
    ):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if num_workers and num_workers > 0:
            torch.autograd.profiler.record_function = _null_record_function
        self.traj_num = cfg["traj_num"]
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.dataset_root = dataset_root
        self.log_interval = max(1, int(log_interval)) if log_interval else None
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else 0.0
        self.use_fused_adamw = bool(use_fused_adamw)
        self.train_yield_head_only = bool(train_yield_head_only)
        self._frozen_output_rows = None
        self.candidate_mode = candidate_mode
        self.backbone_mode = backbone_mode
        self.enable_yield_candidates = bool(enable_yield_candidates)
        self.train_occlusion_risk = train_occlusion_risk
        self.train_risk_point_guidance = train_risk_point_guidance
        self.train_reaction_margin = train_reaction_margin
        self.train_margin_ranking = bool(train_margin_ranking)
        self.train_yaw_visibility = bool(train_yaw_visibility)
        self.deployed_yaw_mode = deployed_yaw_mode
        self.risk_label_source = risk_label_source
        self.gt_sampler_options = {
            "point_count": gt_risk_point_count,
            "hidden_depth_margin_m": gt_hidden_depth_margin_m,
            "min_forward_m": gt_min_forward_m,
            "max_forward_m": gt_max_forward_m,
            "horizon_fov_expand_deg": gt_horizon_fov_expand_deg,
            "vertical_fov_expand_deg": gt_vertical_fov_expand_deg,
            "depth_metric": gt_depth_metric,
            "reachable_forward_center_m": gt_reachable_forward_center_m,
            "reachable_forward_sigma_m": gt_reachable_forward_sigma_m,
            "reachable_lateral_sigma_m": gt_reachable_lateral_sigma_m,
            "reachable_vertical_sigma_m": gt_reachable_vertical_sigma_m,
            "reachable_score_weight": gt_reachable_score_weight,
            "side_score_weight": gt_side_score_weight,
            "nms_radius_m": gt_risk_nms_radius_m,
            "voxel_size_m": gt_risk_voxel_size_m,
        }
        self.risk_assoc_distance_m = risk_assoc_distance_m
        self.risk_assoc_sigma_m = risk_assoc_sigma_m
        self.risk_arrival_radius_m = risk_arrival_radius_m
        self.use_weak_margin_label = use_weak_margin_label
        self.train_yield_feasibility = bool(train_backup_feasibility or train_yield_feasibility)
        self.train_backup_feasibility = self.train_yield_feasibility
        self.use_privileged_risk_filter = use_privileged_risk_filter
        self.yopo_preserve_utility_delta_scale = float(yopo_preserve_utility_delta_scale)
        self.yopo_preserve_safety_residual_weight = float(yopo_preserve_safety_residual_weight)
        self.yopo_preserve_safe_clearance_residual_weight = float(yopo_preserve_safe_clearance_residual_weight)
        self.yopo_preserve_safety_cost_threshold = float(yopo_preserve_safety_cost_threshold)
        self.yopo_preserve_safe_cost_threshold = float(yopo_preserve_safe_cost_threshold)
        self.yopo_preserve_safety_pairwise_weight = float(yopo_preserve_safety_pairwise_weight)
        self.yopo_preserve_safety_pairwise_margin = float(yopo_preserve_safety_pairwise_margin)
        self.yopo_preserve_unsafe_delta_target = float(yopo_preserve_unsafe_delta_target)
        self.yopo_preserve_safe_delta_target = float(yopo_preserve_safe_delta_target)
        if self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank"}:
            if self.train_backup_feasibility or self.train_yield_feasibility:
                raise ValueError("YOPO-preserve modes only support margin/risk/rerank auxiliary training; disable backup/yield feasibility")
            if self.candidate_mode == "yopo_preserve" and self.train_margin_ranking:
                raise ValueError("A1 yopo_preserve keeps YOPO selection fixed; use yopo_preserve_rerank for learned ranking")
            if self.train_yaw_visibility:
                raise ValueError("YOPO-preserve modes keep yaw policy fixed; disable yaw visibility training")
        self.experiment_options = dict(experiment_options or {})
        self.best_val_loss = float("inf")
        if save_on_exit:
            self._exit_func = atexit.register(self.save_model)

        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        self.write_experiment_artifacts(experiment_options or {}, config_path)

        self.policy = OARMNetwork(
            candidate_mode=self.candidate_mode,
            backbone_mode=self.backbone_mode,
            enable_yield_candidates=self.enable_yield_candidates,
            utility_delta_scale=self.yopo_preserve_utility_delta_scale,
        ).to(self.device)
        if checkpoint_path:
            state_dict, checkpoint_metadata = load_oarm_checkpoint(checkpoint_path, map_location=self.device)
            validate_checkpoint_metadata(
                checkpoint_metadata,
                self.candidate_mode,
                self.backbone_mode,
                allow_mismatch=allow_checkpoint_mismatch,
                enable_yield_candidates=self.enable_yield_candidates,
                deployed_yaw_mode=self.deployed_yaw_mode,
                risk_label_source=self.risk_label_source,
            )
            self.policy.load_state_dict(state_dict)
        elif self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank"}:
            if not yopo_checkpoint_path:
                raise ValueError("candidate_mode=yopo_preserve requires --yopo-checkpoint for YOPO base initialization")
            state_dict = torch.load(yopo_checkpoint_path, map_location=self.device, weights_only=True)
            self.policy.preserve_network.load_yopo_state_dict(state_dict, strict=True)
        self.configure_trainable_parameters()
        self.assert_trainable_parameter_contract()
        self.write_trainable_parameter_artifact()

        with yopo_dataset_cfg(self.dataset_root):
            self.oarm_loss = OARMLoss(
                use_esdf_collision=use_esdf_collision,
                use_occlusion_aware_visibility=use_occlusion_aware_visibility,
                enable_occlusion_risk=train_occlusion_risk,
                enable_risk_point_guidance=train_risk_point_guidance,
                enable_reaction_margin=train_reaction_margin,
                enable_margin_ranking=self.train_margin_ranking,
                enable_yaw_visibility=self.train_yaw_visibility,
                deployed_yaw_mode=self.deployed_yaw_mode,
                enable_yield_feasibility=self.train_yield_feasibility,
                risk_assoc_distance_m=self.risk_assoc_distance_m,
                risk_assoc_sigma_m=self.risk_assoc_sigma_m,
                risk_arrival_radius_m=self.risk_arrival_radius_m,
            )
        if self.use_fused_adamw:
            self.optimizer = torch.optim.AdamW(
                self.trainable_parameters(),
                lr=learning_rate,
                fused=(self.device.type == "cuda"),
            )
        else:
            self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=learning_rate)

        loader_kwargs = {}
        if num_workers and num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1

        self.train_dataloader = DataLoader(
            OARMDataset(mode="train", dataset_root=self.dataset_root, use_privileged_risk_filter=self.use_privileged_risk_filter, risk_label_source=self.risk_label_source, gt_sampler_options=self.gt_sampler_options),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            **loader_kwargs,
        )
        self.val_dataloader = DataLoader(
            OARMDataset(mode="valid", dataset_root=self.dataset_root, use_privileged_risk_filter=self.use_privileged_risk_filter, risk_label_source=self.risk_label_source, gt_sampler_options=self.gt_sampler_options),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            **loader_kwargs,
        )

    def train(self, epoch=50):
        train_start = time.time()
        train_steps = self._planned_steps(self.train_dataloader, self.max_train_batches)
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=max(1, epoch * train_steps))
            for self.epoch_i in range(epoch):
                self.policy.train()
                train_loss = self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                with torch.inference_mode():
                    val_loss = self.eval_one_epoch(self.epoch_i)
                self.progress_log.console.log(
                    f"Epoch: {self.epoch_i}, Train Loss: {train_loss:.4g}, "
                    f"Valid Loss: {val_loss:.4g}"
                )
                self.save_checkpoint("last.pth")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_val.pth")
            total_time = time.time() - train_start
            self.progress_log.console.log(f"Train OARM Finish! Total Time: {self._format_seconds(total_time)}")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch, total_progress=None):
        losses = []
        total_steps = self._planned_steps(self.train_dataloader, self.max_train_batches)
        epoch_start = time.time()
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=total_steps)
        for step, batch in enumerate(self.train_dataloader):
            if self.max_train_batches is not None and step >= self.max_train_batches:
                break
            if batch[0].shape[0] != self.batch_size:
                continue
            global_step = epoch * total_steps + step
            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self.forward_and_compute_loss(batch)
            total_loss = loss_dict["total_loss"]
            if not torch.isfinite(total_loss):
                self.tensorboard_log.add_scalar("Train/nonfinite_skip", 1.0, global_step)
                self.progress_log.console.log(f"Train: skipped non-finite loss at epoch {epoch}, step {step}")
                self.progress_log.update(one_epoch_progress, advance=1)
                if total_progress is not None:
                    self.progress_log.update(total_progress, advance=1)
                continue
            total_loss.backward()
            grad_norm = self.clip_or_measure_grad_norm()
            self.tensorboard_log.add_scalar("Train/grad_norm", float(grad_norm.detach().cpu()), global_step)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                self.tensorboard_log.add_scalar("Train/nonfinite_skip", 1.0, global_step)
                self.progress_log.console.log(f"Train: skipped non-finite gradient at epoch {epoch}, step {step}")
                self.progress_log.update(one_epoch_progress, advance=1)
                if total_progress is not None:
                    self.progress_log.update(total_progress, advance=1)
                continue
            self.optimizer.step()
            self.restore_frozen_output_rows()
            if not self.parameters_are_finite():
                raise RuntimeError(f"Non-finite model parameters after optimizer step at epoch {epoch}, step {step}")
            losses.append(total_loss.item())
            self.log_losses("Train", loss_dict, epoch, step)
            self.log_progress("Train", epoch, step, total_steps, losses, epoch_start)
            self.progress_log.update(one_epoch_progress, advance=1)
            if total_progress is not None:
                self.progress_log.update(total_progress, advance=1)
        self.progress_log.remove_task(one_epoch_progress)
        return float(np.mean(losses)) if losses else 0.0

    def eval_one_epoch(self, epoch):
        losses = []
        total_steps = self._planned_steps(self.val_dataloader, self.max_val_batches)
        epoch_start = time.time()
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=total_steps)
        for step, batch in enumerate(self.val_dataloader):
            if self.max_val_batches is not None and step >= self.max_val_batches:
                break
            if batch[0].shape[0] != self.batch_size:
                continue
            loss_dict = self.forward_and_compute_loss(batch)
            total_loss = loss_dict["total_loss"]
            if not torch.isfinite(total_loss):
                self.tensorboard_log.add_scalar("Eval/nonfinite_skip", 1.0, epoch * total_steps + step)
                self.progress_log.console.log(f"Eval: skipped non-finite loss at epoch {epoch}, step {step}")
                self.progress_log.update(one_epoch_progress, advance=1)
                continue
            losses.append(total_loss.item())
            self.log_losses("Eval", loss_dict, epoch, step)
            self.log_progress("Eval", epoch, step, total_steps, losses, epoch_start)
            self.progress_log.update(one_epoch_progress, advance=1)
        self.progress_log.remove_task(one_epoch_progress)
        return float(np.mean(losses)) if losses else 0.0

    def forward_and_compute_loss(self, batch):
        depth, pos, rot, obs_b, map_id, labels = batch
        depth = depth.to(self.device)
        pos = pos.to(self.device)
        rot = rot.to(self.device)
        obs_b = obs_b.to(self.device)

        goal_w = rotate_body2world(rot, obs_b[:, 6:9])
        start_vel_w = rotate_body2world(rot, obs_b[:, 0:3])
        start_acc_w = rotate_body2world(rot, obs_b[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        candidate = self.policy.inference(depth, obs_b)
        flat = candidate.flatten()

        endstate_flat = flat["end_state_b"]
        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)
        start_state_w = start_state_w.repeat_interleave(self.traj_num, dim=0)
        goal_w = goal_w.repeat_interleave(self.traj_num, dim=0)

        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded,
            rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        flat_labels = {}
        if self.train_occlusion_risk and "occlusion_risk" in labels:
            flat_labels["occlusion_risk"] = labels["occlusion_risk"].to(self.device).reshape(-1)

        if self.train_risk_point_guidance and "risk_points_w" in labels:
            flat_labels["risk_points_w"] = labels["risk_points_w"].to(self.device)
            if "risk_weight" in labels:
                flat_labels["risk_weight"] = labels["risk_weight"].to(self.device)
            if "yaw0" in labels:
                flat_labels["yaw0"] = labels["yaw0"].to(self.device)
            if "yaw_rate0" in labels:
                flat_labels["yaw_rate0"] = labels["yaw_rate0"].to(self.device)
            for source_key in (
                "uses_gt_reaction_margin",
                "uses_proxy_reaction_margin",
                "reaction_margin_label_source_id",
                "hidden_risk_gt",
                "raw_gt_risk_point_valid_rate",
                "raw_gt_risk_point_weight_sum",
                "raw_gt_risk_point_weight_mean",
            ):
                if source_key in labels:
                    flat_labels[source_key] = labels[source_key].to(self.device)

        if self.train_reaction_margin and "reaction_margin" in labels:
            flat_labels["reaction_margin"] = labels["reaction_margin"].to(self.device).reshape(-1)
            if "reaction_margin_valid" in labels:
                flat_labels["reaction_margin_valid"] = labels["reaction_margin_valid"].to(self.device).reshape(-1)
        elif self.train_reaction_margin and self.use_weak_margin_label and "occlusion_risk" in flat_labels:
            flat_labels["reaction_margin"] = weak_margin_label_from_risk(flat["traj_time"], flat_labels["occlusion_risk"])

        if self.train_backup_feasibility and "yield_feasible" in labels:
            flat_labels["yield_feasible"] = labels["yield_feasible"].to(self.device).reshape(-1)
        elif self.train_backup_feasibility and "backup_feasible" in labels:
            flat_labels["yield_feasible"] = labels["backup_feasible"].to(self.device).reshape(-1)
        if self.train_backup_feasibility:
            flat_labels["visible_free_distance"] = visible_free_distance_from_depth(
                depth,
                endstate_flat[:, 0:3],
            )

        map_id_expanded = map_id.to(self.device).repeat_interleave(self.traj_num, dim=0)
        loss_dict = self.oarm_loss(start_state_w, end_state_w, flat, goal_w, flat_labels, map_id_expanded)
        if self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank"}:
            aux_loss = torch.zeros((), device=self.device)
            if self.train_occlusion_risk:
                aux_loss = aux_loss + oarm_cfg.risk_bce_weight * loss_dict["risk_loss"]
            if self.train_reaction_margin:
                aux_loss = aux_loss + oarm_cfg.margin_reg_weight * loss_dict["margin_loss"]

            rerank_loss = torch.zeros((), device=self.device)
            residual_reg = torch.zeros((), device=self.device)
            unsafe_boost_loss = torch.zeros((), device=self.device)
            safe_suppression_loss = torch.zeros((), device=self.device)
            safety_residual_loss = torch.zeros((), device=self.device)
            safe_clearance_residual_loss = torch.zeros((), device=self.device)
            safety_pairwise_loss = torch.zeros((), device=self.device)
            unsafe_residual_positive_rate = torch.zeros((), device=self.device)
            safe_residual_negative_rate = torch.zeros((), device=self.device)
            safety_residual_positive_rate = torch.zeros((), device=self.device)
            safe_clearance_residual_negative_rate = torch.zeros((), device=self.device)
            safety_candidate_rate = torch.zeros((), device=self.device)
            safe_clearance_candidate_rate = torch.zeros((), device=self.device)
            safety_pairwise_pair_rate = torch.zeros((), device=self.device)
            margin_pairwise_unsafe = None
            margin_pairwise_safe = None
            if self.candidate_mode == "yopo_preserve_rerank":
                rerank_loss = oarm_cfg.ranking_weight * loss_dict["ranking_loss"]
                delta = flat.get("utility_delta")
                if delta is not None:
                    residual_reg = oarm_cfg.yopo_preserve_residual_reg_weight * delta.square().mean()
                    margin_label = flat_labels.get("reaction_margin")
                    if margin_label is not None:
                        margin_label = margin_label.to(self.device).reshape_as(delta).float()
                        margin_valid = flat_labels.get("reaction_margin_valid")
                        if margin_valid is None:
                            margin_valid = torch.ones_like(delta, dtype=torch.bool)
                        else:
                            margin_valid = margin_valid.to(self.device).reshape_as(delta).bool()
                        margin_valid = margin_valid & torch.isfinite(margin_label) & torch.isfinite(delta)
                        unsafe = margin_valid & (margin_label < 0.0)
                        margin_pairwise_unsafe = unsafe
                        safe = margin_valid & (margin_label > oarm_cfg.yopo_preserve_safe_margin_m)
                        margin_pairwise_safe = safe
                        if bool(unsafe.any()):
                            unsafe_positive = torch.relu(delta[unsafe])
                            unsafe_boost_loss = (
                                oarm_cfg.yopo_preserve_unsafe_boost_weight * unsafe_positive.square().mean()
                            )
                            unsafe_residual_positive_rate = (delta[unsafe] > 0.0).float().mean()
                        if bool(safe.any()):
                            safe_negative = torch.relu(-delta[safe])
                            safe_suppression_loss = (
                                oarm_cfg.yopo_preserve_safe_suppression_weight * safe_negative.square().mean()
                            )
                            safe_residual_negative_rate = (delta[safe] < 0.0).float().mean()
                    safety_cost = loss_dict.get("safety_cost_per_candidate")
                    if safety_cost is not None:
                        safety_cost = safety_cost.to(self.device).reshape_as(delta).float()
                        safety_valid = torch.isfinite(safety_cost) & torch.isfinite(delta)
                        unsafe_safety = safety_valid & (safety_cost > self.yopo_preserve_safety_cost_threshold)
                        safe_clearance = safety_valid & (safety_cost <= self.yopo_preserve_safe_cost_threshold)
                        pairwise_unsafe = unsafe_safety
                        pairwise_safe = safe_clearance
                        if margin_pairwise_unsafe is not None:
                            pairwise_unsafe = pairwise_unsafe | margin_pairwise_unsafe
                        if margin_pairwise_safe is not None:
                            pairwise_safe = pairwise_safe | margin_pairwise_safe
                        safety_candidate_rate = unsafe_safety.float().mean()
                        safe_clearance_candidate_rate = safe_clearance.float().mean()
                        if bool(unsafe_safety.any()) and self.yopo_preserve_safety_residual_weight > 0.0:
                            safety_positive = torch.relu(delta[unsafe_safety] + self.yopo_preserve_unsafe_delta_target)
                            safety_residual_loss = (
                                self.yopo_preserve_safety_residual_weight * safety_positive.square().mean()
                            )
                            safety_residual_positive_rate = (delta[unsafe_safety] > 0.0).float().mean()
                        if bool(safe_clearance.any()) and self.yopo_preserve_safe_clearance_residual_weight > 0.0:
                            safe_clearance_negative = torch.relu(self.yopo_preserve_safe_delta_target - delta[safe_clearance])
                            safe_clearance_residual_loss = (
                                self.yopo_preserve_safe_clearance_residual_weight * safe_clearance_negative.square().mean()
                            )
                            safe_clearance_residual_negative_rate = (delta[safe_clearance] < 0.0).float().mean()
                        if self.yopo_preserve_safety_pairwise_weight > 0.0 and delta.numel() % self.traj_num == 0:
                            pairwise_score = flat.get("utility_score", delta).reshape_as(delta)
                            score_group = pairwise_score.reshape(-1, self.traj_num)
                            unsafe_group = pairwise_unsafe.reshape(-1, self.traj_num)
                            safe_group = pairwise_safe.reshape(-1, self.traj_num)
                            has_safety_pair = unsafe_group.any(dim=1) & safe_group.any(dim=1)
                            safety_pairwise_pair_rate = has_safety_pair.float().mean()
                            if bool(has_safety_pair.any()):
                                neg_inf = torch.full_like(score_group, -float("inf"))
                                safe_best_score = torch.where(safe_group, score_group, neg_inf).max(dim=1).values
                                unsafe_best_score = torch.where(unsafe_group, score_group, neg_inf).max(dim=1).values
                                safety_pairwise_gap = torch.relu(
                                    unsafe_best_score[has_safety_pair]
                                    + self.yopo_preserve_safety_pairwise_margin
                                    - safe_best_score[has_safety_pair]
                                )
                                safety_pairwise_loss = (
                                    self.yopo_preserve_safety_pairwise_weight * safety_pairwise_gap.square().mean()
                                )
            loss_dict["aux_only_loss"] = aux_loss
            loss_dict["rerank_only_loss"] = rerank_loss
            loss_dict["utility_delta_reg_loss"] = residual_reg
            loss_dict["unsafe_boost_loss"] = unsafe_boost_loss
            loss_dict["safe_suppression_loss"] = safe_suppression_loss
            loss_dict["safety_residual_loss"] = safety_residual_loss
            loss_dict["safe_clearance_residual_loss"] = safe_clearance_residual_loss
            loss_dict["safety_pairwise_loss"] = safety_pairwise_loss

            loss_dict["unsafe_residual_positive_rate"] = unsafe_residual_positive_rate
            loss_dict["safe_residual_negative_rate"] = safe_residual_negative_rate
            loss_dict["safety_residual_positive_rate"] = safety_residual_positive_rate
            loss_dict["safe_clearance_residual_negative_rate"] = safe_clearance_residual_negative_rate
            loss_dict["safety_candidate_rate"] = safety_candidate_rate
            loss_dict["safe_clearance_candidate_rate"] = safe_clearance_candidate_rate
            loss_dict["safety_pairwise_pair_rate"] = safety_pairwise_pair_rate

            loss_dict["total_loss_full_objective_detached"] = loss_dict["total_loss"].detach()
            loss_dict["total_loss"] = (
                aux_loss
                + rerank_loss
                + residual_reg
                + unsafe_boost_loss
                + safe_suppression_loss
                + safety_residual_loss
                + safe_clearance_residual_loss
                + safety_pairwise_loss
            )
        return loss_dict

    def log_losses(self, prefix, loss_dict, epoch, step):
        global_step = epoch * max(1, len(self.train_dataloader)) + step
        for key, value in loss_dict.items():
            if torch.is_tensor(value) and value.dim() == 0:
                self.tensorboard_log.add_scalar(f"{prefix}/{key}", value.item(), global_step)

    def log_progress(self, prefix, epoch, step, total_steps, losses, epoch_start):
        if not self.log_interval:
            return
        current_step = step + 1
        if current_step != 1 and current_step % self.log_interval != 0 and current_step < total_steps:
            return
        elapsed = time.time() - epoch_start
        avg_step_s = elapsed / max(current_step, 1)
        eta_s = avg_step_s * max(total_steps - current_step, 0)
        batch_fps = current_step / max(elapsed, 1e-6)
        self.progress_log.console.log(
            f"{prefix}: {epoch}, Step: {current_step}/{total_steps}, "
            f"Total Loss: {np.mean(losses):.3g}, Batch FPS: {batch_fps:.3g}, "
            f"ETA: {self._format_seconds(eta_s)}"
        )

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = os.path.join(self.tensorboard_path, f"epoch{self.epoch_i + 1}.pth")
            self.save_checkpoint(os.path.basename(policy_path))
            if hasattr(self, "_exit_func"):
                atexit.unregister(self._exit_func)

    def save_checkpoint(self, filename):
        if not self.parameters_are_finite():
            self.progress_log.console.log('Skip checkpoint {filename}: model has non-finite parameters'.format(filename=filename))
            return
        policy_path = os.path.join(self.tensorboard_path, filename)
        torch.save(
            make_oarm_checkpoint(
                self.policy.state_dict(),
                self.candidate_mode,
                self.backbone_mode,
                self.experiment_options,
                enable_yield_candidates=self.enable_yield_candidates,
                deployed_yaw_mode=self.deployed_yaw_mode,
                risk_label_source=self.risk_label_source,
            ),
            policy_path,
        )

    def clip_or_measure_grad_norm(self):
        parameters = [p for p in self.trainable_parameters() if p.grad is not None]
        if not parameters:
            return torch.zeros((), device=self.device)
        if self.grad_clip_norm and self.grad_clip_norm > 0.0:
            return torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)
        norms = [p.grad.detach().norm(2) for p in parameters]
        return torch.norm(torch.stack(norms), 2)

    def parameters_are_finite(self):
        return all(torch.isfinite(p).all().item() for p in self.policy.parameters())

    def trainable_parameters(self):
        return [p for p in self.policy.parameters() if p.requires_grad]

    def trainable_parameter_names(self):
        return [name for name, param in self.policy.named_parameters() if param.requires_grad]

    def configure_trainable_parameters(self):
        if self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank"}:
            if self.train_yield_head_only:
                raise ValueError(f"--train-yield-head-only is not compatible with candidate_mode={self.candidate_mode}")
            for name, param in self.policy.named_parameters():
                param.requires_grad_(name.startswith("preserve_network.aux_head."))
            return
        if not self.train_yield_head_only:
            return
        for param in self.policy.parameters():
            param.requires_grad_(False)

        final_layer = self.policy.oarm_head.model[-1]
        final_layer.weight.requires_grad_(True)
        final_layer.bias.requires_grad_(True)

        row_mask = torch.zeros(final_layer.out_channels, device=self.device, dtype=final_layer.weight.dtype)
        row_mask[13] = 1.0
        weight_mask = row_mask.view(-1, 1, 1, 1)
        bias_mask = row_mask
        final_layer.weight.register_hook(lambda grad: grad * weight_mask)
        final_layer.bias.register_hook(lambda grad: grad * bias_mask)
        self._frozen_output_rows = {
            'weight': final_layer.weight.detach().clone(),
            'bias': final_layer.bias.detach().clone(),
            'row_mask_bool': row_mask.bool(),
        }

    def restore_frozen_output_rows(self):
        if not self._frozen_output_rows:
            return
        final_layer = self.policy.oarm_head.model[-1]
        row_mask = self._frozen_output_rows['row_mask_bool']
        with torch.no_grad():
            final_layer.weight[~row_mask].copy_(self._frozen_output_rows['weight'][~row_mask])
            final_layer.bias[~row_mask].copy_(self._frozen_output_rows['bias'][~row_mask])

    def assert_trainable_parameter_contract(self):
        trainable = self.trainable_parameter_names()
        if not trainable:
            raise RuntimeError("No trainable OARM parameters were configured")
        if self.candidate_mode not in {"yopo_preserve", "yopo_preserve_rerank"}:
            return
        bad = [name for name in trainable if not name.startswith("preserve_network.aux_head.")]
        if bad:
            raise RuntimeError(
                f"{self.candidate_mode} must train only preserve auxiliary heads; unexpected trainable parameters: "
                + ", ".join(bad)
            )
        aux_names = [name for name, _param in self.policy.named_parameters() if name.startswith("preserve_network.aux_head.")]
        missing = sorted(set(aux_names) - set(trainable))
        if missing:
            raise RuntimeError("Some preserve auxiliary head parameters are frozen: " + ", ".join(missing))

    def write_trainable_parameter_artifact(self):
        trainable = self.trainable_parameter_names()
        frozen = [name for name, param in self.policy.named_parameters() if not param.requires_grad]
        path = os.path.join(self.tensorboard_path, "trainable_parameters.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Trainable parameters\n")
            for name in trainable:
                f.write(f"{name}\n")
            f.write("\n# All parameters\n")
            for name, param in self.policy.named_parameters():
                f.write(f"{name}\trequires_grad={bool(param.requires_grad)}\tshape={tuple(param.shape)}\n")
        json_path = os.path.join(self.tensorboard_path, "trainable_parameters.json")
        preserve = getattr(self.policy, "preserve_network", None)
        yopo_base_modules = {}
        if preserve is not None:
            yopo_base_modules = {
                "image_backbone_training": bool(preserve.image_backbone.training),
                "state_backbone_training": bool(preserve.state_backbone.training),
                "yopo_head_training": bool(preserve.yopo_head.training),
                "aux_head_training": bool(preserve.aux_head.training),
                "enable_utility_delta": bool(getattr(preserve, "enable_utility_delta", False)),
            }
        artifact = {
            "candidate_mode": self.candidate_mode,
            "backbone_mode": self.backbone_mode,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_count": len(trainable),
            "frozen_count": len(frozen),
            "yopo_base_modules": yopo_base_modules,
            "a1_auxiliary_only": bool(self.candidate_mode == "yopo_preserve"),
            "preserve_rerank": bool(self.candidate_mode == "yopo_preserve_rerank"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, sort_keys=True)

    def write_experiment_artifacts(self, experiment_options, config_path=""):
        metadata = {
            "online_inputs": ["depth", "state", "goal"],
            "uses_privileged_training": bool(
                experiment_options.get("use_privileged_risk_filter")
                or experiment_options.get("use_occlusion_aware_visibility")
                or experiment_options.get("use_esdf_collision")
                or experiment_options.get("train_reaction_margin")
                or experiment_options.get("train_risk_point_guidance")
            ),
            "uses_privileged_online": False,
            "mapless_online_inference": True,
            "backbone_policy": experiment_options.get("backbone_mode", self.backbone_mode),
            "training_options": experiment_options,
        }
        options_path = os.path.join(self.tensorboard_path, "options.json")
        with open(options_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        if config_path:
            abs_config = os.path.abspath(config_path)
            if os.path.isfile(abs_config):
                shutil.copy2(abs_config, os.path.join(self.tensorboard_path, "config.yaml"))

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.write_git_artifact(repo_root, ["rev-parse", "HEAD"], "git_commit.txt")
        self.write_git_artifact(repo_root, ["status", "--short"], "git_status.txt")
        self.write_git_artifact(repo_root, ["diff", "--", "OARM"], "git_diff_oarm.patch")

    def write_git_artifact(self, repo_root, git_args, filename):
        try:
            result = subprocess.run(
                ["git", *git_args],
                cwd=repo_root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            content = result.stdout
        except Exception as exc:
            content = f"git unavailable: {exc}\n"
        with open(os.path.join(self.tensorboard_path, filename), "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def get_next_log_path(base_path):
        base_path = base_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved")
        os.makedirs(base_path, exist_ok=True)
        nums = [
            int(name.split("_")[1])
            for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name)) and name.startswith("OARM_") and name.split("_")[1].isdigit()
        ]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"OARM_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard log to ", next_path)
        return next_path

    @staticmethod
    def _planned_steps(dataloader, max_batches):
        total = len(dataloader)
        if max_batches is not None:
            total = min(total, max_batches)
        return max(1, total)

    @staticmethod
    def _format_seconds(seconds):
        seconds = int(max(0, seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m"
        if minutes:
            return f"{minutes:d}m{secs:02d}s"
        return f"{secs:d}s"
