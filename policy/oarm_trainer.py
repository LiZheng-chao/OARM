import atexit
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


class OARMTrainer:
    def __init__(
        self,
        learning_rate=1.5e-4,
        batch_size=16,
        tensorboard_path=None,
        checkpoint_path=None,
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
        use_weak_margin_label=oarm_cfg.use_weak_margin_label,
        train_backup_feasibility=oarm_cfg.train_backup_feasibility,
        train_yield_feasibility=oarm_cfg.train_yield_feasibility,
        use_esdf_collision=oarm_cfg.use_esdf_collision,
        use_occlusion_aware_visibility=oarm_cfg.use_occlusion_aware_visibility,
        use_privileged_risk_filter=oarm_cfg.use_privileged_risk_filter,
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
        self.use_weak_margin_label = use_weak_margin_label
        self.train_yield_feasibility = bool(train_backup_feasibility or train_yield_feasibility)
        self.train_backup_feasibility = self.train_yield_feasibility
        self.use_privileged_risk_filter = use_privileged_risk_filter
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
            )
            self.policy.load_state_dict(state_dict)
        self.configure_trainable_parameters()

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
            )
        if self.use_fused_adamw:
            self.optimizer = torch.optim.AdamW(
                self.trainable_parameters(),
                lr=learning_rate,
                fused=(self.device.type == "cuda"),
            )
        else:
            self.optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=learning_rate)

        self.train_dataloader = DataLoader(
            OARMDataset(mode="train", dataset_root=self.dataset_root, use_privileged_risk_filter=self.use_privileged_risk_filter),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            OARMDataset(mode="valid", dataset_root=self.dataset_root, use_privileged_risk_filter=self.use_privileged_risk_filter),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
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
        return self.oarm_loss(start_state_w, end_state_w, flat, goal_w, flat_labels, map_id_expanded)

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

    def configure_trainable_parameters(self):
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
