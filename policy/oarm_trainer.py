import atexit
import contextlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter

from OARM.dataset import OARMDataset
from OARM.config import oarm_cfg
from OARM.loss.reaction_margin_loss import weak_margin_label_from_risk
from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_state_transform import rotate_body2world, state_body2world
from OARM.utils.checkpoint import load_oarm_checkpoint, make_oarm_checkpoint, validate_checkpoint_metadata
from OARM.utils.gt_clearance import candidate_min_clearance_gt
from OARM.utils.visible_free_distance import visible_free_distance_from_depth
from OARM.utils.yopo_compat import ensure_yopo_path
from OARM.utils.yopo_dataset_context import resolve_dataset_dir, yopo_dataset_cfg
from OARM.visibility.reaction_margin_targets import generate_reaction_margin_labels

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
        init_from_a1_checkpoint_path=None,
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
        train_probabilistic_rm_critic=oarm_cfg.train_probabilistic_rm_critic,
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
        ranking_weight=oarm_cfg.ranking_weight,
        yopo_preserve_utility_delta_scale=oarm_cfg.yopo_preserve_utility_delta_scale,
        yopo_preserve_residual_reg_weight=oarm_cfg.yopo_preserve_residual_reg_weight,
        yopo_preserve_unsafe_boost_weight=oarm_cfg.yopo_preserve_unsafe_boost_weight,
        yopo_preserve_safe_suppression_weight=oarm_cfg.yopo_preserve_safe_suppression_weight,
        yopo_preserve_safe_margin_m=oarm_cfg.yopo_preserve_safe_margin_m,
        yopo_preserve_safety_residual_weight=oarm_cfg.yopo_preserve_safety_residual_weight,
        yopo_preserve_safe_clearance_residual_weight=oarm_cfg.yopo_preserve_safe_clearance_residual_weight,
        yopo_preserve_safety_cost_threshold=oarm_cfg.yopo_preserve_safety_cost_threshold,
        yopo_preserve_safe_cost_threshold=oarm_cfg.yopo_preserve_safe_cost_threshold,
        yopo_preserve_geometry_oracle_source=oarm_cfg.yopo_preserve_geometry_oracle_source,
        yopo_preserve_unsafe_clearance_m=oarm_cfg.yopo_preserve_unsafe_clearance_m,
        yopo_preserve_safe_clearance_m=oarm_cfg.yopo_preserve_safe_clearance_m,
        yopo_preserve_safety_pairwise_weight=oarm_cfg.yopo_preserve_safety_pairwise_weight,
        yopo_preserve_safety_pairwise_margin=oarm_cfg.yopo_preserve_safety_pairwise_margin,
        yopo_preserve_unsafe_delta_target=oarm_cfg.yopo_preserve_unsafe_delta_target,
        yopo_preserve_safe_delta_target=oarm_cfg.yopo_preserve_safe_delta_target,
        yopo_preserve_freeze_margin_risk_head=oarm_cfg.yopo_preserve_freeze_margin_risk_head,
        yopo_preserve_oracle_ce_weight=oarm_cfg.yopo_preserve_oracle_ce_weight,
        yopo_preserve_oracle_ce_temperature=oarm_cfg.yopo_preserve_oracle_ce_temperature,
        yopo_preserve_oracle_min_margin=oarm_cfg.yopo_preserve_oracle_min_margin,
        yopo_preserve_oracle_min_progress=oarm_cfg.yopo_preserve_oracle_min_progress,
        yopo_preserve_geometry_ce_weight=oarm_cfg.yopo_preserve_geometry_ce_weight,
        yopo_preserve_geometry_ce_temperature=oarm_cfg.yopo_preserve_geometry_ce_temperature,
        experiment_options=None,
        config_path="",
        log_interval=50,
        allow_checkpoint_mismatch=False,
        grad_clip_norm=1.0,
        use_fused_adamw=False,
        train_yield_head_only=False,
        progress_bar=True,
        sample_weights_path=None,
    ):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if num_workers and num_workers > 0:
            torch.autograd.profiler.record_function = _null_record_function
        self.traj_num = cfg["traj_num"]
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches
        self.dataset_root = resolve_dataset_dir(dataset_root)
        self.log_interval = max(1, int(log_interval)) if log_interval else None
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else 0.0
        self.use_fused_adamw = bool(use_fused_adamw)
        self.train_yield_head_only = bool(train_yield_head_only)
        self.progress_bar = bool(progress_bar)
        self.sample_weights_path = sample_weights_path
        self.oracle_ce_sanity_window = 50
        self._oracle_ce_sanity_seen = 0
        self._oracle_ce_sanity_pair_rate_sum = 0.0
        self._oracle_ce_sanity_loss_sum = 0.0
        self._oracle_ce_sanity_reported = False
        self._frozen_output_rows = None
        self.a1_initialization_report = {}
        self.candidate_mode = candidate_mode
        self.backbone_mode = backbone_mode
        self.enable_yield_candidates = bool(enable_yield_candidates)
        self.train_occlusion_risk = train_occlusion_risk
        self.train_risk_point_guidance = train_risk_point_guidance
        self.train_reaction_margin = train_reaction_margin
        self.train_margin_ranking = bool(train_margin_ranking)
        self.train_probabilistic_rm_critic = bool(train_probabilistic_rm_critic)
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
        self.ranking_weight = float(ranking_weight)
        self.yopo_preserve_utility_delta_scale = float(yopo_preserve_utility_delta_scale)
        self.yopo_preserve_residual_reg_weight = float(yopo_preserve_residual_reg_weight)
        self.yopo_preserve_unsafe_boost_weight = float(yopo_preserve_unsafe_boost_weight)
        self.yopo_preserve_safe_suppression_weight = float(yopo_preserve_safe_suppression_weight)
        self.yopo_preserve_safe_margin_m = float(yopo_preserve_safe_margin_m)
        self.yopo_preserve_safety_residual_weight = float(yopo_preserve_safety_residual_weight)
        self.yopo_preserve_safe_clearance_residual_weight = float(yopo_preserve_safe_clearance_residual_weight)
        self.yopo_preserve_safety_cost_threshold = float(yopo_preserve_safety_cost_threshold)
        self.yopo_preserve_safe_cost_threshold = float(yopo_preserve_safe_cost_threshold)
        self.yopo_preserve_geometry_oracle_source = str(yopo_preserve_geometry_oracle_source)
        self.yopo_preserve_unsafe_clearance_m = float(yopo_preserve_unsafe_clearance_m)
        self.yopo_preserve_safe_clearance_m = float(yopo_preserve_safe_clearance_m)
        if self.yopo_preserve_geometry_oracle_source not in {"esdf_cost", "gt_clearance"}:
            raise ValueError(f"Unknown yopo_preserve_geometry_oracle_source: {self.yopo_preserve_geometry_oracle_source}")
        self.yopo_preserve_safety_pairwise_weight = float(yopo_preserve_safety_pairwise_weight)
        self.yopo_preserve_safety_pairwise_margin = float(yopo_preserve_safety_pairwise_margin)
        self.yopo_preserve_unsafe_delta_target = float(yopo_preserve_unsafe_delta_target)
        self.yopo_preserve_safe_delta_target = float(yopo_preserve_safe_delta_target)
        self.yopo_preserve_freeze_margin_risk_head = bool(yopo_preserve_freeze_margin_risk_head)
        self.yopo_preserve_oracle_ce_weight = float(yopo_preserve_oracle_ce_weight)
        self.yopo_preserve_oracle_ce_temperature = max(float(yopo_preserve_oracle_ce_temperature), 1e-3)
        self.yopo_preserve_oracle_min_margin = float(yopo_preserve_oracle_min_margin)
        self.yopo_preserve_oracle_min_progress = float(yopo_preserve_oracle_min_progress)
        self.yopo_preserve_geometry_ce_weight = float(yopo_preserve_geometry_ce_weight)
        self.yopo_preserve_geometry_ce_temperature = max(float(yopo_preserve_geometry_ce_temperature), 1e-3)
        if self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}:
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

        self.console = Console(force_terminal=self.progress_bar)
        self.progress_log = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=2,
            disable=not self.progress_bar,
        )
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        self.write_experiment_artifacts(experiment_options or {}, config_path)

        self.policy = OARMNetwork(
            candidate_mode=self.candidate_mode,
            backbone_mode=self.backbone_mode,
            enable_yield_candidates=self.enable_yield_candidates,
            utility_delta_scale=self.yopo_preserve_utility_delta_scale,
            enable_rm_critic=self.train_probabilistic_rm_critic,
        ).to(self.device)
        if checkpoint_path and init_from_a1_checkpoint_path:
            raise ValueError("Use either --checkpoint for same-structure resume or --init-from-a1-checkpoint for A1->A3h initialization, not both")
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
                yopo_preserve_utility_delta_scale=self.yopo_preserve_utility_delta_scale,
            )
            self.policy.load_state_dict(state_dict)
        elif init_from_a1_checkpoint_path:
            self.load_a1_initialization_checkpoint(init_from_a1_checkpoint_path, allow_mismatch=allow_checkpoint_mismatch)
        elif self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}:
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
                enable_probabilistic_rm_critic=self.train_probabilistic_rm_critic,
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

        train_dataset = OARMDataset(
            mode="train",
            dataset_root=self.dataset_root,
            use_privileged_risk_filter=self.use_privileged_risk_filter,
            risk_label_source=self.risk_label_source,
            gt_sampler_options=self.gt_sampler_options,
        )
        train_sampler = self.build_train_sampler(train_dataset)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
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


    def build_train_sampler(self, train_dataset):
        if not self.sample_weights_path:
            return None
        if not os.path.isfile(self.sample_weights_path):
            raise FileNotFoundError(f"A3 critical sample weights not found: {self.sample_weights_path}")
        try:
            payload = torch.load(self.sample_weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(self.sample_weights_path, map_location="cpu")
        weights = payload.get("weights") if isinstance(payload, dict) else payload
        if isinstance(payload, dict):
            summary = payload.get("summary") or {}
            identity = summary.get("dataset_identity") or {}
            mode = identity.get("mode")
            if mode not in {None, "train"}:
                raise ValueError(f"Sample weights were generated for mode={mode}, but training requires mode=train")
            weight_root = identity.get("dataset_root")
            if weight_root:
                current_root = os.path.realpath(self.dataset_root)
                stored_root = os.path.realpath(weight_root)
                roots_match = current_root == stored_root
                if not roots_match and os.path.exists(current_root) and os.path.exists(stored_root):
                    try:
                        roots_match = os.path.samefile(current_root, stored_root)
                    except OSError:
                        roots_match = False
                if not roots_match:
                    raise ValueError(
                        "Sample weights dataset root does not match current training dataset. "
                        f"weights={weight_root}, current={self.dataset_root}. Regenerate weights for this dataset."
                    )
        weights = torch.as_tensor(weights, dtype=torch.double).reshape(-1)
        if weights.numel() != len(train_dataset):
            raise ValueError(
                f"Sample weights length {weights.numel()} does not match train dataset length {len(train_dataset)}. "
                "Regenerate weights with OARM.tools.build_a3_critical_sampler --mode train for the same dataset root."
            )
        finite = torch.isfinite(weights) & (weights > 0.0)
        if not bool(finite.all()):
            bad = int((~finite).sum().item())
            raise ValueError(f"Sample weights must be finite and positive; invalid entries: {bad}")
        self.progress_log.console.log(
            f"Using weighted A3 critical-frame sampler from {self.sample_weights_path}; "
            f"mean={weights.mean().item():.3g}, max={weights.max().item():.3g}"
        )
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


    def load_a1_initialization_checkpoint(self, checkpoint_path, allow_mismatch=False):
        if self.candidate_mode != "yopo_preserve_rerank":
            raise ValueError("--init-from-a1-checkpoint is only valid for candidate_mode=yopo_preserve_rerank")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"A1 initialization checkpoint not found: {checkpoint_path}")
        state_dict, metadata = load_oarm_checkpoint(checkpoint_path, map_location=self.device)
        training_options = metadata.get("training_options") or {}
        stored_candidate = metadata.get("candidate_mode") or training_options.get("candidate_mode")
        stored_backbone = metadata.get("backbone_mode") or training_options.get("backbone_mode")
        if stored_candidate not in {None, "yopo_preserve"} and not allow_mismatch:
            raise ValueError(
                "--init-from-a1-checkpoint expects an A1 candidate_mode=yopo_preserve checkpoint; "
                f"got candidate_mode={stored_candidate}. Pass --allow-checkpoint-mismatch only if intentional."
            )
        if stored_backbone not in {None, "yopo_original"} and not allow_mismatch:
            raise ValueError(
                "--init-from-a1-checkpoint expects backbone_mode=yopo_original; "
                f"got backbone_mode={stored_backbone}."
            )
        has_preserve = any(key.startswith("preserve_network.") for key in state_dict)
        if not has_preserve:
            raise ValueError("A1 initialization checkpoint does not contain preserve_network.* weights")
        missing, unexpected = self.policy.load_state_dict(state_dict, strict=True)
        self.policy.preserve_network.reset_rerank_output()
        self.a1_initialization_report = {
            "path": checkpoint_path,
            "stored_candidate_mode": stored_candidate,
            "stored_backbone_mode": stored_backbone,
            "loaded_margin_risk_head": True,
            "reset_rerank_output": True,
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
        self.progress_log.console.log(
            "Loaded A1 margin/risk weights for A3h warm-start; reset rerank output to zero."
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
                self.progress_log.update(one_epoch_progress, advance=1)
                if total_progress is not None:
                    self.progress_log.update(total_progress, advance=1)
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
            self.update_oracle_ce_sanity(loss_dict, epoch, step)
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
                self.progress_log.update(one_epoch_progress, advance=1)
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
        traj_num = int(flat["traj_time"].numel() // max(depth.shape[0], 1))
        pos_expanded = pos.repeat_interleave(traj_num, dim=0)
        rot_expanded = rot.repeat_interleave(traj_num, dim=0)
        start_state_w = start_state_w.repeat_interleave(traj_num, dim=0)
        goal_w = goal_w.repeat_interleave(traj_num, dim=0)

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

        map_id_expanded = map_id.to(self.device).repeat_interleave(traj_num, dim=0)
        if self.train_reaction_margin:
            flat_labels = generate_reaction_margin_labels(
                flat_labels,
                flat,
                start_state_w,
                end_state_w,
                map_id_expanded,
                goal_w,
                enabled=True,
                labeler=self.oarm_loss.margin_labeler,
                line_of_sight=self.oarm_loss.line_of_sight,
                yaw_helper=self.oarm_loss,
            )
        loss_dict = self.oarm_loss(start_state_w, end_state_w, flat, goal_w, flat_labels, map_id_expanded)
        if self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}:
            aux_loss = torch.zeros((), device=self.device)
            if self.candidate_mode != "a4_preserve_brake":
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
            oracle_ce_loss = torch.zeros((), device=self.device)
            geometry_ce_loss = torch.zeros((), device=self.device)
            brake_gate_loss = torch.zeros((), device=self.device)
            rm_critic_loss = loss_dict.get("rm_critic_loss", torch.zeros((), device=self.device))
            oracle_ce_pair_rate = torch.zeros((), device=self.device)
            oracle_ce_top1_acc = torch.zeros((), device=self.device)
            oracle_ce_target_margin_mean = torch.zeros((), device=self.device)
            geometry_ce_pair_rate = torch.zeros((), device=self.device)
            geometry_ce_top1_acc = torch.zeros((), device=self.device)
            geometry_ce_target_clearance_mean = torch.zeros((), device=self.device)
            geometry_ce_suppressed_by_oracle_rate = torch.zeros((), device=self.device)
            geom_unsafe_mask_rate = torch.zeros((), device=self.device)
            geom_safe_mask_rate = torch.zeros((), device=self.device)
            margin_unsafe_mask_rate = torch.zeros((), device=self.device)
            margin_safe_mask_rate = torch.zeros((), device=self.device)
            hard_unsafe_mask_rate = torch.zeros((), device=self.device)
            hard_safe_mask_rate = torch.zeros((), device=self.device)
            final_unsafe_mask_rate = torch.zeros((), device=self.device)
            final_safe_mask_rate = torch.zeros((), device=self.device)
            safe_and_unsafe_overlap_rate = torch.zeros((), device=self.device)
            geom_unsafe_and_margin_safe_rate = torch.zeros((), device=self.device)
            geom_safe_and_margin_unsafe_rate = torch.zeros((), device=self.device)
            gt_min_clearance_mean = torch.zeros((), device=self.device)
            gt_min_clearance_selected_mean = torch.zeros((), device=self.device)
            gt_clearance_valid_rate = torch.zeros((), device=self.device)
            gt_clearance_unsafe_candidate_rate = torch.zeros((), device=self.device)
            gt_clearance_safe_candidate_rate = torch.zeros((), device=self.device)
            oracle_ce_primary_rate = torch.zeros((), device=self.device)
            oracle_ce_fallback_rate = torch.zeros((), device=self.device)
            if self.candidate_mode == "a4_preserve_brake":
                a4a_loss = self.compute_a4a_binary_brake_loss(
                    flat,
                    flat_labels,
                    loss_dict,
                    map_id_expanded,
                    traj_num,
                )
                brake_gate_loss = a4a_loss["brake_gate_loss"]
                residual_reg = a4a_loss["brake_gate_l2_loss"]
                for key, value in a4a_loss.items():
                    loss_dict[key] = value
            elif self.candidate_mode == "yopo_preserve_rerank":
                rerank_loss = self.ranking_weight * loss_dict["ranking_loss"]
                delta = flat.get("utility_delta")
                if delta is not None:
                    residual_reg = self.yopo_preserve_residual_reg_weight * delta.square().mean()
                    finite_delta = torch.isfinite(delta)
                    selector_unsafe = None
                    selector_safe = None
                    hard_unsafe = None
                    hard_safe = None
                    progress_ok = None
                    geom_unsafe = None
                    geom_safe = None
                    margin_unsafe = None
                    margin_safe = None
                    margin_valid = None

                    safety_cost = loss_dict.get("safety_cost_per_candidate")
                    if self.yopo_preserve_geometry_oracle_source == "gt_clearance":
                        sampled_pos_w = loss_dict.get("sampled_pos_w")
                        if sampled_pos_w is None:
                            raise RuntimeError("GT clearance oracle requires sampled_pos_w from OARMLoss")
                        min_clearance = candidate_min_clearance_gt(sampled_pos_w, map_id_expanded, self.dataset_root).reshape_as(delta)
                        clearance_valid = torch.isfinite(min_clearance) & finite_delta
                        geom_unsafe = clearance_valid & (min_clearance < self.yopo_preserve_unsafe_clearance_m)
                        geom_safe = clearance_valid & (min_clearance > self.yopo_preserve_safe_clearance_m)
                        gt_clearance_valid_rate = clearance_valid.float().mean()
                        if bool(clearance_valid.any()):
                            gt_min_clearance_mean = min_clearance[clearance_valid].mean()
                        selected_id = flat.get("utility_score", delta).reshape(-1, traj_num).argmax(dim=1)
                        selected_clearance = min_clearance.reshape(-1, traj_num).gather(1, selected_id[:, None]).squeeze(1)
                        selected_valid = torch.isfinite(selected_clearance)
                        if bool(selected_valid.any()):
                            gt_min_clearance_selected_mean = selected_clearance[selected_valid].mean()
                        gt_clearance_unsafe_candidate_rate = geom_unsafe.float().mean()
                        gt_clearance_safe_candidate_rate = geom_safe.float().mean()
                        safety_candidate_rate = gt_clearance_unsafe_candidate_rate
                        safe_clearance_candidate_rate = gt_clearance_safe_candidate_rate
                        geom_unsafe_mask_rate = safety_candidate_rate
                        geom_safe_mask_rate = safe_clearance_candidate_rate
                    elif safety_cost is not None:
                        safety_cost = safety_cost.to(self.device).reshape_as(delta).float()
                        safety_valid = torch.isfinite(safety_cost) & finite_delta
                        geom_unsafe = safety_valid & (safety_cost > self.yopo_preserve_safety_cost_threshold)
                        geom_safe = safety_valid & (safety_cost <= self.yopo_preserve_safe_cost_threshold)
                        safety_candidate_rate = geom_unsafe.float().mean()
                        safe_clearance_candidate_rate = geom_safe.float().mean()
                        geom_unsafe_mask_rate = safety_candidate_rate
                        geom_safe_mask_rate = safe_clearance_candidate_rate

                    margin_label = flat_labels.get("reaction_margin")
                    if margin_label is not None:
                        margin_label = margin_label.to(self.device).reshape_as(delta).float()
                        margin_valid = flat_labels.get("reaction_margin_valid")
                        if margin_valid is None:
                            margin_valid = torch.ones_like(delta, dtype=torch.bool)
                        else:
                            margin_valid = margin_valid.to(self.device).reshape_as(delta).bool()
                        margin_valid = margin_valid & torch.isfinite(margin_label) & finite_delta
                        margin_unsafe = margin_valid & (margin_label < 0.0)
                        margin_safe = margin_valid & (margin_label > self.yopo_preserve_safe_margin_m)
                        margin_unsafe_mask_rate = margin_unsafe.float().mean()
                        margin_safe_mask_rate = margin_safe.float().mean()

                    if geom_unsafe is not None and margin_safe is not None:
                        geom_unsafe_and_margin_safe_rate = (geom_unsafe & margin_safe).float().mean()
                    if geom_safe is not None and margin_unsafe is not None:
                        geom_safe_and_margin_unsafe_rate = (geom_safe & margin_unsafe).float().mean()

                    if self.yopo_preserve_oracle_min_progress > 0.0:
                        progress_score = -OARMLoss.goal_progress_cost(
                            start_state_w, end_state_w, goal_w, flat["traj_time"]
                        ).detach()
                        progress_ok = finite_delta & torch.isfinite(progress_score) & (progress_score > self.yopo_preserve_oracle_min_progress)

                    if geom_unsafe is not None:
                        hard_unsafe = geom_unsafe
                        hard_safe = geom_safe & ~geom_unsafe
                        if progress_ok is not None:
                            hard_safe = hard_safe & progress_ok
                        hard_unsafe_mask_rate = hard_unsafe.float().mean()
                        hard_safe_mask_rate = hard_safe.float().mean()

                    if geom_unsafe is not None:
                        if margin_unsafe is not None:
                            selector_unsafe = geom_unsafe | margin_unsafe
                            selector_safe = geom_safe & margin_safe & ~selector_unsafe
                        else:
                            selector_unsafe = geom_unsafe
                            selector_safe = geom_safe & ~selector_unsafe
                    elif margin_unsafe is not None:
                        selector_unsafe = margin_unsafe
                        selector_safe = margin_safe & ~selector_unsafe

                    if selector_safe is not None and progress_ok is not None:
                        selector_safe = selector_safe & progress_ok

                    if selector_unsafe is not None and selector_safe is not None:
                        overlap = selector_unsafe & selector_safe
                        final_unsafe_mask_rate = selector_unsafe.float().mean()
                        final_safe_mask_rate = selector_safe.float().mean()
                        safe_and_unsafe_overlap_rate = overlap.float().mean()
                        if bool(overlap.any()):
                            raise RuntimeError("A3 safety masks must be mutually exclusive, but safe&unsafe overlap was nonzero")
                        if hard_unsafe is not None and bool(hard_unsafe.any()):
                            unsafe_positive = torch.relu(delta[hard_unsafe])
                            unsafe_boost_loss = (
                                self.yopo_preserve_unsafe_boost_weight * unsafe_positive.square().mean()
                            )
                            unsafe_residual_positive_rate = (delta[hard_unsafe] > 0.0).float().mean()
                        if bool(selector_safe.any()):
                            safe_negative = torch.relu(-delta[selector_safe])
                            safe_suppression_loss = (
                                self.yopo_preserve_safe_suppression_weight * safe_negative.square().mean()
                            )
                            safe_residual_negative_rate = (delta[selector_safe] < 0.0).float().mean()

                    residual_unsafe = hard_unsafe
                    residual_safe = hard_safe
                    if residual_unsafe is not None and bool(residual_unsafe.any()) and self.yopo_preserve_safety_residual_weight > 0.0:
                        safety_positive = torch.relu(delta[residual_unsafe] + self.yopo_preserve_unsafe_delta_target)
                        safety_residual_loss = (
                            self.yopo_preserve_safety_residual_weight * safety_positive.square().mean()
                        )
                        safety_residual_positive_rate = (delta[residual_unsafe] > 0.0).float().mean()
                    if residual_safe is not None and bool(residual_safe.any()) and self.yopo_preserve_safe_clearance_residual_weight > 0.0:
                        safe_clearance_negative = torch.relu(self.yopo_preserve_safe_delta_target - delta[residual_safe])
                        safe_clearance_residual_loss = (
                            self.yopo_preserve_safe_clearance_residual_weight * safe_clearance_negative.square().mean()
                        )
                        safe_clearance_residual_negative_rate = (delta[residual_safe] < 0.0).float().mean()

                    if (
                        selector_unsafe is not None
                        and selector_safe is not None
                        and delta.numel() % traj_num == 0
                    ):
                        pairwise_score = flat.get("utility_score", delta).reshape_as(delta)
                        score_group = pairwise_score.reshape(-1, traj_num)
                        unsafe_group = selector_unsafe.reshape(-1, traj_num)
                        safe_group = selector_safe.reshape(-1, traj_num)
                        geom_safe_group = geom_safe.reshape(-1, traj_num) if geom_safe is not None else safe_group
                        if progress_ok is not None:
                            geom_safe_group = geom_safe_group & progress_ok.reshape(-1, traj_num)
                        margin_group = None
                        if margin_label is not None:
                            margin_group = margin_label.reshape(-1, traj_num)
                        has_oracle = torch.zeros(score_group.shape[0], dtype=torch.bool, device=self.device)
                        oracle_group = safe_group
                        primary_available = safe_group.any(dim=1)
                        fallback_available = torch.zeros_like(primary_available)
                        if margin_group is not None:
                            if geom_safe is not None and margin_valid is not None:
                                fallback = geom_safe & margin_valid
                                if progress_ok is not None:
                                    fallback = fallback & progress_ok
                                fallback_group = fallback.reshape(-1, traj_num)
                                fallback_available = fallback_group.any(dim=1) & ~primary_available
                                oracle_group = torch.where(primary_available[:, None], safe_group, fallback_group)
                            neg_inf = torch.full_like(margin_group, -float("inf"))
                            oracle_source = torch.where(oracle_group, margin_group, neg_inf)
                            oracle_margin, oracle_id = oracle_source.max(dim=1)
                            has_oracle = torch.isfinite(oracle_margin) & oracle_group.any(dim=1)
                        has_safety_pair = unsafe_group.any(dim=1) & safe_group.any(dim=1)
                        safety_pairwise_pair_rate = has_safety_pair.float().mean()
                        if self.yopo_preserve_safety_pairwise_weight > 0.0 and bool(has_safety_pair.any()):
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
                        if self.yopo_preserve_geometry_ce_weight > 0.0 and geom_safe_group is not None:
                            base_score = flat.get("utility_base", pairwise_score).reshape_as(delta).detach().reshape(-1, traj_num)
                            geometry_available_raw = geom_safe_group.any(dim=1)
                            geometry_available = geometry_available_raw & ~has_oracle
                            geometry_ce_pair_rate = geometry_available.float().mean()
                            geometry_ce_suppressed_by_oracle_rate = (geometry_available_raw & has_oracle).float().mean()
                            if bool(geometry_available.any()):
                                neg_inf = torch.full_like(base_score, -float("inf"))
                                geometry_source = torch.where(geom_safe_group, base_score, neg_inf)
                                geometry_target_id = geometry_source.max(dim=1).indices
                                geometry_logits = score_group[geometry_available] / self.yopo_preserve_geometry_ce_temperature
                                geometry_ce_loss = self.yopo_preserve_geometry_ce_weight * F.cross_entropy(
                                    geometry_logits, geometry_target_id[geometry_available].long()
                                )
                                geometry_ce_top1_acc = (geometry_logits.argmax(dim=1) == geometry_target_id[geometry_available]).float().mean()
                                if 'min_clearance' in locals():
                                    clearance_group = min_clearance.reshape(-1, traj_num)
                                    target_clearance = clearance_group.gather(1, geometry_target_id[:, None]).squeeze(1)
                                    target_valid = geometry_available & torch.isfinite(target_clearance)
                                    if bool(target_valid.any()):
                                        geometry_ce_target_clearance_mean = target_clearance[target_valid].mean()
                        if self.yopo_preserve_oracle_ce_weight > 0.0 and margin_group is not None:
                            oracle_ce_pair_rate = has_oracle.float().mean()
                            oracle_ce_primary_rate = (primary_available & has_oracle).float().mean()
                            oracle_ce_fallback_rate = (fallback_available & has_oracle).float().mean()
                            if bool(has_oracle.any()):
                                logits = score_group[has_oracle] / self.yopo_preserve_oracle_ce_temperature
                                oracle_ce_loss = self.yopo_preserve_oracle_ce_weight * F.cross_entropy(
                                    logits, oracle_id[has_oracle].long()
                                )
                                oracle_ce_top1_acc = (logits.argmax(dim=1) == oracle_id[has_oracle]).float().mean()
                                oracle_ce_target_margin_mean = oracle_margin[has_oracle].mean()
            loss_dict["aux_only_loss"] = aux_loss
            loss_dict["rerank_only_loss"] = rerank_loss
            loss_dict["utility_delta_reg_loss"] = residual_reg
            loss_dict["unsafe_boost_loss"] = unsafe_boost_loss
            loss_dict["safe_suppression_loss"] = safe_suppression_loss
            loss_dict["safety_residual_loss"] = safety_residual_loss
            loss_dict["safe_clearance_residual_loss"] = safe_clearance_residual_loss
            loss_dict["safety_pairwise_loss"] = safety_pairwise_loss
            loss_dict["oracle_ce_loss"] = oracle_ce_loss
            loss_dict["geometry_ce_loss"] = geometry_ce_loss
            loss_dict["brake_gate_loss"] = brake_gate_loss

            loss_dict["unsafe_residual_positive_rate"] = unsafe_residual_positive_rate
            loss_dict["safe_residual_negative_rate"] = safe_residual_negative_rate
            loss_dict["safety_residual_positive_rate"] = safety_residual_positive_rate
            loss_dict["safe_clearance_residual_negative_rate"] = safe_clearance_residual_negative_rate
            loss_dict["safety_candidate_rate"] = safety_candidate_rate
            loss_dict["safe_clearance_candidate_rate"] = safe_clearance_candidate_rate
            loss_dict["safety_pairwise_pair_rate"] = safety_pairwise_pair_rate
            loss_dict["oracle_ce_pair_rate"] = oracle_ce_pair_rate
            loss_dict["oracle_ce_top1_acc"] = oracle_ce_top1_acc
            loss_dict["oracle_ce_target_margin_mean"] = oracle_ce_target_margin_mean
            loss_dict["geometry_ce_pair_rate"] = geometry_ce_pair_rate
            loss_dict["geometry_ce_top1_acc"] = geometry_ce_top1_acc
            loss_dict["geometry_ce_target_clearance_mean"] = geometry_ce_target_clearance_mean
            loss_dict["oracle_ce_primary_rate"] = oracle_ce_primary_rate
            loss_dict["oracle_ce_fallback_rate"] = oracle_ce_fallback_rate
            loss_dict["geom_unsafe_mask_rate"] = geom_unsafe_mask_rate
            loss_dict["geom_safe_mask_rate"] = geom_safe_mask_rate
            loss_dict["margin_unsafe_mask_rate"] = margin_unsafe_mask_rate
            loss_dict["margin_safe_mask_rate"] = margin_safe_mask_rate
            loss_dict["hard_unsafe_mask_rate"] = hard_unsafe_mask_rate
            loss_dict["hard_safe_mask_rate"] = hard_safe_mask_rate
            loss_dict["final_unsafe_mask_rate"] = final_unsafe_mask_rate
            loss_dict["final_safe_mask_rate"] = final_safe_mask_rate
            loss_dict["safe_and_unsafe_overlap_rate"] = safe_and_unsafe_overlap_rate
            loss_dict["geom_unsafe_and_margin_safe_rate"] = geom_unsafe_and_margin_safe_rate
            loss_dict["geom_safe_and_margin_unsafe_rate"] = geom_safe_and_margin_unsafe_rate
            loss_dict["gt_min_clearance_mean"] = gt_min_clearance_mean
            loss_dict["gt_min_clearance_selected_mean"] = gt_min_clearance_selected_mean
            loss_dict["gt_clearance_valid_rate"] = gt_clearance_valid_rate
            loss_dict["gt_clearance_unsafe_candidate_rate"] = gt_clearance_unsafe_candidate_rate
            loss_dict["gt_clearance_safe_candidate_rate"] = gt_clearance_safe_candidate_rate
            loss_dict["geometry_ce_suppressed_by_oracle_rate"] = geometry_ce_suppressed_by_oracle_rate

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
                + oracle_ce_loss
                + geometry_ce_loss
                + brake_gate_loss
                + rm_critic_loss
            )
        return loss_dict

    def compute_a4a_binary_brake_loss(self, flat, flat_labels, loss_dict, map_id_expanded, traj_num):
        if traj_num < 2:
            raise RuntimeError("A4a binary brake oracle requires YOPO candidates plus one brake candidate")
        delta = flat.get("utility_delta")
        utility_score = flat.get("utility_score")
        if delta is None or utility_score is None:
            raise RuntimeError("A4a binary brake oracle requires utility_delta and utility_score")

        delta_group = delta.to(self.device).reshape(-1, traj_num).float()
        score_group = utility_score.to(self.device).reshape(-1, traj_num).float()
        base_score = flat.get("utility_base")
        if base_score is None:
            base_group = score_group.detach()
        else:
            base_group = base_score.to(self.device).reshape(-1, traj_num).float().detach()

        progress_n = traj_num - 1
        yopo_scores = base_group[:, :progress_n]
        yopo_id = yopo_scores.argmax(dim=1)
        batch_id = torch.arange(yopo_scores.shape[0], device=self.device)
        yopo_score = yopo_scores[batch_id, yopo_id]
        brake_logit = delta_group[:, -1]
        finite_gate = torch.isfinite(brake_logit)
        finite_yopo = torch.isfinite(yopo_score)

        geom_unsafe = None
        geom_safe = None
        if self.yopo_preserve_geometry_oracle_source == "gt_clearance":
            sampled_pos_w = loss_dict.get("sampled_pos_w")
            if sampled_pos_w is None:
                raise RuntimeError("A4a GT clearance oracle requires sampled_pos_w from OARMLoss")
            min_clearance = candidate_min_clearance_gt(sampled_pos_w, map_id_expanded, self.dataset_root).reshape(-1, traj_num)
            clearance_valid = torch.isfinite(min_clearance)
            geom_unsafe = clearance_valid & (min_clearance < self.yopo_preserve_unsafe_clearance_m)
            geom_safe = clearance_valid & (min_clearance > self.yopo_preserve_safe_clearance_m)
        else:
            safety_cost = loss_dict.get("safety_cost_per_candidate")
            if safety_cost is not None:
                safety_cost = safety_cost.to(self.device).reshape(-1, traj_num).float()
                safety_valid = torch.isfinite(safety_cost)
                geom_unsafe = safety_valid & (safety_cost > self.yopo_preserve_safety_cost_threshold)
                geom_safe = safety_valid & (safety_cost <= self.yopo_preserve_safe_cost_threshold)

        yopo_geom_unsafe = torch.zeros_like(finite_gate, dtype=torch.bool)
        yopo_geom_safe = torch.zeros_like(finite_gate, dtype=torch.bool)
        brake_feasible = torch.zeros_like(finite_gate, dtype=torch.bool)
        if geom_unsafe is not None and geom_safe is not None:
            yopo_geom_unsafe = geom_unsafe[:, :progress_n][batch_id, yopo_id]
            yopo_geom_safe = geom_safe[:, :progress_n][batch_id, yopo_id]
            brake_feasible = geom_safe[:, -1]

        margin_label = flat_labels.get("reaction_margin")
        margin_valid_label = flat_labels.get("reaction_margin_valid")
        yopo_margin_unsafe = torch.zeros_like(finite_gate, dtype=torch.bool)
        yopo_margin_valid = torch.zeros_like(finite_gate, dtype=torch.bool)
        yopo_margin = torch.full_like(brake_logit, float("nan"))
        if margin_label is not None:
            margin_group = margin_label.to(self.device).reshape(-1, traj_num).float()
            if margin_valid_label is None:
                margin_valid_group = torch.ones_like(margin_group, dtype=torch.bool)
            else:
                margin_valid_group = margin_valid_label.to(self.device).reshape(-1, traj_num).bool()
            yopo_margin = margin_group[:, :progress_n][batch_id, yopo_id]
            yopo_margin_valid = margin_valid_group[:, :progress_n][batch_id, yopo_id] & torch.isfinite(yopo_margin)
            yopo_margin_unsafe = yopo_margin_valid & (yopo_margin < 0.0)

        yopo_bad = yopo_geom_unsafe | yopo_margin_unsafe
        yopo_keep_safe = yopo_geom_safe & (~yopo_margin_valid | (yopo_margin >= 0.0))
        target_brake = yopo_bad & brake_feasible
        valid = finite_gate & finite_yopo & (yopo_keep_safe | target_brake)

        zero = torch.zeros((), device=self.device)
        brake_gate_loss = zero
        if bool(valid.any()):
            target = target_brake[valid].float()
            brake_gate_loss = self.yopo_preserve_oracle_ce_weight * F.binary_cross_entropy_with_logits(
                brake_logit[valid], target
            )
        brake_gate_l2_loss = zero
        finite_brake = torch.isfinite(brake_logit)
        if self.yopo_preserve_residual_reg_weight > 0.0 and bool(finite_brake.any()):
            brake_gate_l2_loss = self.yopo_preserve_residual_reg_weight * brake_logit[finite_brake].square().mean()

        pred_brake = brake_logit > 0.0
        metrics = {
            "brake_gate_loss": brake_gate_loss,
            "brake_gate_l2_loss": brake_gate_l2_loss,
            "a4a_brake_gate_valid_rate": valid.float().mean(),
            "a4a_yopo_top1_bad_rate": yopo_bad.float().mean(),
            "a4a_yopo_top1_geom_unsafe_rate": yopo_geom_unsafe.float().mean(),
            "a4a_yopo_top1_geom_safe_rate": yopo_geom_safe.float().mean(),
            "a4a_yopo_top1_keep_safe_rate": yopo_keep_safe.float().mean(),
            "a4a_yopo_top1_margin_unsafe_rate": yopo_margin_unsafe.float().mean(),
            "a4a_yopo_top1_margin_valid_rate": yopo_margin_valid.float().mean(),
            "a4a_brake_feasible_rate": brake_feasible.float().mean(),
            "a4a_ignored_bad_and_no_brake_rate": (yopo_bad & ~brake_feasible & finite_gate).float().mean(),
            "a4a_ambiguous_ignore_rate": (finite_gate & finite_yopo & ~valid).float().mean(),
        }
        if bool(valid.any()):
            target_valid = target_brake[valid]
            pred_valid = pred_brake[valid]
            positive = target_valid
            negative = ~target_valid
            predicted_positive = pred_valid
            metrics.update(
                {
                    "a4a_brake_target_rate": target_valid.float().mean(),
                    "a4a_brake_pred_rate": pred_valid.float().mean(),
                    "a4a_brake_gate_acc": (pred_valid == target_valid).float().mean(),
                    "a4a_brake_recall": pred_valid[positive].float().mean() if bool(positive.any()) else zero,
                    "a4a_false_brake_rate": pred_valid[negative].float().mean() if bool(negative.any()) else zero,
                    "a4a_keep_recall": (~pred_valid[negative]).float().mean() if bool(negative.any()) else zero,
                    "a4a_brake_precision": target_valid[predicted_positive].float().mean()
                    if bool(predicted_positive.any())
                    else zero,
                    "a4a_brake_positive_valid_count": positive.float().sum(),
                    "a4a_keep_negative_valid_count": negative.float().sum(),
                    "a4a_yopo_top1_margin_mean": yopo_margin[valid & yopo_margin_valid].mean()
                    if bool((valid & yopo_margin_valid).any())
                    else zero,
                }
            )
        else:
            metrics.update(
                {
                    "a4a_brake_target_rate": zero,
                    "a4a_brake_pred_rate": zero,
                    "a4a_brake_gate_acc": zero,
                    "a4a_brake_recall": zero,
                    "a4a_false_brake_rate": zero,
                    "a4a_keep_recall": zero,
                    "a4a_brake_precision": zero,
                    "a4a_brake_positive_valid_count": zero,
                    "a4a_keep_negative_valid_count": zero,
                    "a4a_yopo_top1_margin_mean": zero,
                }
            )
        return metrics

    def update_oracle_ce_sanity(self, loss_dict, epoch, step):
        if self._oracle_ce_sanity_reported:
            return
        if self.candidate_mode != "yopo_preserve_rerank":
            return
        if not self.train_reaction_margin or self.yopo_preserve_oracle_ce_weight <= 0.0:
            return
        pair_rate = loss_dict.get("oracle_ce_pair_rate")
        oracle_loss = loss_dict.get("oracle_ce_loss")
        if pair_rate is None or oracle_loss is None:
            return
        pair_rate_value = self._scalar_value(pair_rate)
        oracle_loss_value = self._scalar_value(oracle_loss)
        primary_rate = self._scalar_value(loss_dict.get("oracle_ce_primary_rate", 0.0))
        fallback_rate = self._scalar_value(loss_dict.get("oracle_ce_fallback_rate", 0.0))
        suppressed_rate = self._scalar_value(loss_dict.get("geometry_ce_suppressed_by_oracle_rate", 0.0))
        self._oracle_ce_sanity_seen += 1
        self._oracle_ce_sanity_pair_rate_sum += pair_rate_value
        self._oracle_ce_sanity_loss_sum += oracle_loss_value
        if pair_rate_value > 1e-8:
            self.progress_log.console.log(
                "A3 oracle CE sanity OK: "
                f"epoch={epoch}, step={step + 1}, pair_rate={pair_rate_value:.4g}, "
                f"loss={oracle_loss_value:.4g}, primary={primary_rate:.4g}, "
                f"fallback={fallback_rate:.4g}, geom_suppressed={suppressed_rate:.4g}"
            )
            self._oracle_ce_sanity_reported = True
            return
        if self._oracle_ce_sanity_seen >= self.oracle_ce_sanity_window:
            avg_pair = self._oracle_ce_sanity_pair_rate_sum / max(self._oracle_ce_sanity_seen, 1)
            avg_loss = self._oracle_ce_sanity_loss_sum / max(self._oracle_ce_sanity_seen, 1)
            self.progress_log.console.log(
                "A3 oracle CE sanity WARNING: "
                f"first {self._oracle_ce_sanity_seen} train batches had avg_pair_rate={avg_pair:.4g}, "
                f"avg_loss={avg_loss:.4g}. Check reaction-margin labels, risk_points_w, and sampler coverage."
            )
            global_step = epoch * max(1, len(self.train_dataloader)) + step
            self.tensorboard_log.add_scalar("Train/oracle_ce_sanity_warning", 1.0, global_step)
            self._oracle_ce_sanity_reported = True

    @staticmethod
    def _scalar_value(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

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
        if self.candidate_mode in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}:
            if self.train_yield_head_only:
                raise ValueError(f"--train-yield-head-only is not compatible with candidate_mode={self.candidate_mode}")
            if self.train_probabilistic_rm_critic:
                train_prefixes = ["preserve_network.rm_critic."]
            elif self.candidate_mode == "a4_preserve_brake":
                train_prefixes = ["preserve_network.brake_gate_head."]
            else:
                train_prefixes = ["preserve_network.margin_risk_head."]
                if self.candidate_mode == "yopo_preserve_rerank":
                    if self.yopo_preserve_freeze_margin_risk_head:
                        train_prefixes = ["preserve_network.rerank_head."]
                    else:
                        train_prefixes.append("preserve_network.rerank_head.")
            for name, param in self.policy.named_parameters():
                param.requires_grad_(any(name.startswith(prefix) for prefix in train_prefixes))
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
        if self.candidate_mode not in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}:
            return
        allowed_prefixes = ("preserve_network.margin_risk_head.", "preserve_network.rerank_head.", "preserve_network.brake_gate_head.", "preserve_network.rm_critic.")
        bad = [name for name in trainable if not name.startswith(allowed_prefixes)]
        if bad:
            raise RuntimeError(
                f"{self.candidate_mode} must train only preserve auxiliary heads; unexpected trainable parameters: "
                + ", ".join(bad)
            )
        if self.train_probabilistic_rm_critic:
            required_prefixes = ("preserve_network.rm_critic.",)
        elif self.candidate_mode == "yopo_preserve":
            required_prefixes = ("preserve_network.margin_risk_head.",)
        elif self.candidate_mode == "a4_preserve_brake":
            required_prefixes = ("preserve_network.brake_gate_head.",)
        elif self.yopo_preserve_freeze_margin_risk_head:
            required_prefixes = ("preserve_network.rerank_head.",)
        else:
            required_prefixes = allowed_prefixes
        aux_names = [
            name for name, _param in self.policy.named_parameters()
            if name.startswith(required_prefixes)
        ]
        missing = sorted(set(aux_names) - set(trainable))
        if missing:
            raise RuntimeError("Some required preserve auxiliary head parameters are frozen: " + ", ".join(missing))

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
                "margin_risk_head_training": bool(preserve.margin_risk_head.training),
                "rerank_head_training": bool(preserve.rerank_head.training),
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
            "a1_initialization": self.a1_initialization_report,
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
