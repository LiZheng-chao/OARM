import torch
from torch import nn

from OARM.config import oarm_cfg
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_poly_solver import quintic_coefficients, sample_polynomial
from OARM.policy.oarm_rm_critic import CandidateRMCritic, risk_logit_from_window, two_stage_risk_logit
from OARM.policy.oarm_types import OARMCandidate
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.primitive import LatticePrimitive
from policy.state_transform import StateTransform


class YOPOPreserveOARMNetwork(nn.Module):
    """YOPO-preserving policy with optional additive safety candidates.

    The first 15 candidates are the official YOPO decoded primitives: endpoint
    PVA, fixed segment time, yaw, and YOPO score are preserved exactly. A4a can
    append one deterministic braking primitive after those 15 candidates; only
    OARM auxiliary/gate heads are trainable.
    """

    def __init__(
        self,
        observation_dim=9,
        hidden_state=64,
        freeze_yopo_base=True,
        margin_scale=oarm_cfg.margin_scale,
        enable_utility_delta=False,
        utility_delta_scale=oarm_cfg.yopo_preserve_utility_delta_scale,
        append_brake_candidate=False,
        enable_rm_critic=oarm_cfg.train_probabilistic_rm_critic,
        rm_critic_hidden_dim=oarm_cfg.rm_critic_hidden_dim,
        rm_critic_hazard_bins=oarm_cfg.rm_critic_hazard_bins,
        rm_critic_hazard_max_time_s=oarm_cfg.rm_critic_hazard_max_time_s,
    ):
        super().__init__()
        self.append_brake_candidate = bool(append_brake_candidate)
        if self.append_brake_candidate:
            self.candidate_mode = "a4_preserve_brake"
            enable_utility_delta = False
        else:
            self.candidate_mode = "yopo_preserve_rerank" if enable_utility_delta else "yopo_preserve"
        self.backbone_mode = "yopo_original"
        self.margin_scale = float(margin_scale)
        self.enable_utility_delta = bool(enable_utility_delta)
        self.utility_delta_scale = float(utility_delta_scale)
        self.enable_rm_critic = bool(enable_rm_critic)
        self.rm_critic_hazard_max_time_s = max(float(rm_critic_hazard_max_time_s), 1e-3)
        self.rm_critic_trajectory_samples = 8
        self.rm_critic_geometry_dim = 9 + 1 + 3 + 3 * self.rm_critic_trajectory_samples
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        segment_time = float(self.lattice_primitive.segment_time)
        self.candidate_generator = OARMCandidateGenerator(
            fast_time=segment_time,
            normal_time=segment_time,
            brake_time=segment_time,
            enable_yield=False,
        )

        self.image_backbone = YopoBackbone(hidden_state)
        self.state_backbone = nn.Sequential()
        self.yopo_head = YopoHead(hidden_state + observation_dim, 10)
        self.margin_risk_head = nn.Sequential(
            nn.Conv2d(hidden_state + observation_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1),
        )
        self.rerank_head = nn.Sequential(
            nn.Conv2d(hidden_state + observation_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.brake_gate_head = nn.Sequential(
            nn.Conv2d(hidden_state + observation_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.rm_critic = None
        if self.enable_rm_critic:
            self.rm_critic = CandidateRMCritic(
                candidate_feature_dim=hidden_state + observation_dim,
                geometry_feature_dim=self.rm_critic_geometry_dim,
                hidden_dim=rm_critic_hidden_dim,
                hazard_bins=rm_critic_hazard_bins,
            )
        self.reset_aux_heads()
        if freeze_yopo_base:
            self.freeze_yopo_base()

    def reset_aux_heads(self):
        for head in (self.margin_risk_head, self.rerank_head):
            final = head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        final = self.brake_gate_head[-1]
        nn.init.zeros_(final.weight)
        # Inactive by default so A4a starts as exact YOPO unless trained to brake.
        nn.init.constant_(final.bias, -5.0)

    def reset_rerank_output(self):
        final = self.rerank_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def freeze_yopo_base(self):
        for module in (self.image_backbone, self.state_backbone, self.yopo_head):
            for param in module.parameters():
                param.requires_grad_(False)
        self.freeze_yopo_base_state()

    def freeze_yopo_base_state(self):
        for module in (self.image_backbone, self.state_backbone, self.yopo_head):
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.freeze_yopo_base_state()
        self.margin_risk_head.train(mode)
        self.rerank_head.train(mode)
        self.brake_gate_head.train(mode)
        if self.rm_critic is not None:
            self.rm_critic.train(mode)
        return self

    def adapt_legacy_aux_state_dict(self, state_dict):
        adapted = dict(state_dict)
        prefix = ""
        if any(key.startswith("preserve_network.") for key in adapted):
            prefix = "preserve_network."

        critic_input_weight = prefix + "rm_critic.mlp.0.weight"
        critic_final_weight = prefix + "rm_critic.mlp.4.weight"
        critic_final_bias = prefix + "rm_critic.mlp.4.bias"
        own_state = self.state_dict()
        if critic_input_weight in adapted and critic_input_weight in own_state:
            src_w = adapted[critic_input_weight]
            dst_w = own_state[critic_input_weight]
            if src_w.shape != dst_w.shape and src_w.ndim == dst_w.ndim == 2 and src_w.shape[0] == dst_w.shape[0] and src_w.shape[1] <= dst_w.shape[1]:
                new_w = torch.zeros_like(dst_w)
                new_w[:, : src_w.shape[1]] = src_w.to(device=new_w.device, dtype=new_w.dtype)
                adapted[critic_input_weight] = new_w
        if critic_final_weight in adapted and critic_final_weight in own_state:
            src_w = adapted[critic_final_weight]
            dst_w = own_state[critic_final_weight]
            src_b = adapted.get(critic_final_bias)
            dst_b = own_state.get(critic_final_bias)
            if src_w.shape != dst_w.shape and src_w.ndim == dst_w.ndim == 2 and src_w.shape[1] == dst_w.shape[1]:
                rows = min(src_w.shape[0], dst_w.shape[0])
                new_w = dst_w.detach().clone()
                new_w[:rows] = src_w[:rows].to(device=new_w.device, dtype=new_w.dtype)
                adapted[critic_final_weight] = new_w
                if src_b is not None and dst_b is not None:
                    new_b = dst_b.detach().clone()
                    new_b[:rows] = src_b[:rows].to(device=new_b.device, dtype=new_b.dtype)
                    adapted[critic_final_bias] = new_b

        old_prefix = prefix + "aux_head."
        margin_prefix = prefix + "margin_risk_head."
        rerank_prefix = prefix + "rerank_head."
        if old_prefix + "0.weight" not in adapted:
            return adapted

        shared_weight = adapted.pop(old_prefix + "0.weight")
        shared_bias = adapted.pop(old_prefix + "0.bias")
        adapted[margin_prefix + "0.weight"] = shared_weight
        adapted[margin_prefix + "0.bias"] = shared_bias
        adapted[rerank_prefix + "0.weight"] = shared_weight.clone()
        adapted[rerank_prefix + "0.bias"] = shared_bias.clone()
        final_weight = adapted.pop(old_prefix + "2.weight")
        final_bias = adapted.pop(old_prefix + "2.bias")
        adapted[margin_prefix + "2.weight"] = final_weight[:2].clone()
        adapted[margin_prefix + "2.bias"] = final_bias[:2].clone()
        if final_weight.shape[0] >= 3:
            adapted[rerank_prefix + "2.weight"] = final_weight[2:3].clone()
            adapted[rerank_prefix + "2.bias"] = final_bias[2:3].clone()
        return adapted

    def load_yopo_state_dict(self, state_dict, strict=True):
        state_dict = self.adapt_legacy_aux_state_dict(state_dict)
        own_state = self.state_dict()
        adapted = {}
        for key, value in state_dict.items():
            if key in own_state:
                adapted[key] = value
            elif key.startswith("module.") and key[len("module.") :] in own_state:
                adapted[key[len("module.") :]] = value
        missing, unexpected = self.load_state_dict(adapted, strict=False)
        ignored_prefixes = ("margin_risk_head.", "rerank_head.", "brake_gate_head.", "rm_critic.")
        missing = [key for key in missing if not key.startswith(ignored_prefixes)]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "YOPO checkpoint did not match YOPO-preserve network: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def forward(self, depth: torch.Tensor, obs_prepared: torch.Tensor, return_critic: bool = False):
        with torch.no_grad():
            depth_feature = self.image_backbone(depth)
            obs_feature = self.state_backbone(obs_prepared)
            features = torch.cat((obs_feature, depth_feature), dim=1)
            output = self.yopo_head(features)
            endstate_pred = torch.tanh(output[:, :9])
            score_pred = torch.nn.functional.softplus(output[:, 9])
        detached_features = features.detach()
        margin_risk = self.margin_risk_head(detached_features)
        if self.enable_utility_delta:
            utility_delta_raw = self.rerank_head(detached_features)
            aux = torch.cat((margin_risk, utility_delta_raw), dim=1)
        else:
            aux = margin_risk
        if self.append_brake_candidate:
            gate_map = self.brake_gate_head(detached_features)
            utility_base = -score_pred
            top1_id = utility_base.flatten(1).argmax(dim=1)
            brake_gate_raw = gate_map.flatten(1).gather(1, top1_id[:, None]).reshape(-1, 1, 1, 1)
            brake_gate_raw = brake_gate_raw.expand(-1, -1, margin_risk.shape[2], margin_risk.shape[3])
            aux = torch.cat((aux, brake_gate_raw), dim=1)
        if return_critic:
            return endstate_pred, score_pred, aux, detached_features
        return endstate_pred, score_pred, aux

    def rm_critic_geometry_features(self, end_state_b: torch.Tensor, traj_time: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        b, _, v, h = end_state_b.shape
        n = v * h
        device = end_state_b.device
        dtype = end_state_b.dtype
        end_flat = end_state_b.permute(0, 2, 3, 1).reshape(b * n, 9).detach()
        time_flat = traj_time.reshape(b * n).to(device=device, dtype=dtype).detach().clamp_min(1e-3)
        vel0 = obs[:, 0:3].to(device=device, dtype=dtype).detach()[:, None, :].expand(b, n, 3).reshape(b * n, 3)
        acc0 = obs[:, 3:6].to(device=device, dtype=dtype).detach()[:, None, :].expand(b, n, 3).reshape(b * n, 3)
        pos0 = torch.zeros_like(vel0)
        start_state = torch.stack((pos0, vel0, acc0), dim=1)
        end_state = torch.stack((end_flat[:, 0:3], end_flat[:, 3:6], end_flat[:, 6:9]), dim=1)
        coeff = quintic_coefficients(start_state, end_state, time_flat)
        sampled_pos, _, _, _ = sample_polynomial(
            coeff,
            time_flat,
            eval_points=self.rm_critic_trajectory_samples,
            include_zero=False,
        )
        geometry = torch.cat(
            (
                end_flat,
                time_flat[:, None],
                vel0,
                sampled_pos.reshape(b * n, -1),
            ),
            dim=-1,
        )
        return geometry.reshape(b, n, self.rm_critic_geometry_dim)

    def deterministic_brake_candidate(self, obs: torch.Tensor, dtype: torch.dtype, device: torch.device):
        vel_b = obs[:, 0:3].to(device=device, dtype=dtype)
        speed = vel_b.norm(dim=1, keepdim=True)
        acc_max = max(float(self.lattice_primitive.acc_max), 1e-3)
        direction = vel_b / speed.clamp_min(1e-3)
        brake_time_scalar = torch.clamp(1.5 * speed / acc_max, min=0.6, max=2.5)
        stopping_distance = torch.clamp(0.5 * speed * brake_time_scalar, min=0.0, max=6.0)
        end_pos = torch.where(speed > 1e-3, direction * stopping_distance, torch.zeros_like(vel_b))
        end_vel = torch.zeros_like(end_pos)
        end_acc = torch.zeros_like(end_pos)
        end_state = torch.cat((end_pos, end_vel, end_acc), dim=1).reshape(obs.shape[0], 9, 1, 1)
        brake_time = brake_time_scalar.reshape(obs.shape[0], 1, 1, 1)
        return end_state, brake_time

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> OARMCandidate:
        obs_norm = self.state_transform.normalize_obs(obs.clone())
        obs_prepared = self.state_transform.prepare_input(obs_norm)
        endstate_pred, score_pred, aux, critic_features = self.forward(depth, obs_prepared, return_critic=True)
        end_state_b = self.state_transform.pred_to_endstate(endstate_pred)

        b, _, v, h = end_state_b.shape
        traj_time = torch.full(
            (b, 1, v, h),
            float(self.lattice_primitive.segment_time),
            device=depth.device,
            dtype=depth.dtype,
        )
        yaw_terminal = torch.zeros((b, 1, v, h), device=depth.device, dtype=depth.dtype)
        margin_pred = self.margin_scale * torch.tanh(aux[:, 0:1])
        risk_logit = aux[:, 1:2]
        reaction_window_mean = None
        reaction_window_logvar = None
        validity_logit = None
        rm_insufficient_logit = None
        zero_window_logit = None
        hazard_logits = None
        critic = None
        if self.rm_critic is not None:
            _, c_feat, _, _ = critic_features.shape
            candidate_feature = critic_features.permute(0, 2, 3, 1).reshape(b, v * h, c_feat)
            candidate_geometry = self.rm_critic_geometry_features(end_state_b, traj_time, obs)
            critic = self.rm_critic(
                candidate_feature,
                yopo_cost=score_pred.reshape(b, v * h),
                candidate_geometry=candidate_geometry,
            )
        if critic is not None:
            reaction_window_mean = critic["reaction_window_mean"].reshape(b, 1, v, h)
            reaction_window_logvar = critic["reaction_window_logvar"].reshape(b, 1, v, h)
            validity_logit = critic["validity_logit"].reshape(b, 1, v, h)
            rm_insufficient_logit = critic["insufficient_margin_logit"].reshape(b, 1, v, h)
            zero_window_logit = critic.get("zero_window_logit", critic["insufficient_margin_logit"]).reshape(b, 1, v, h)
            margin_pred = reaction_window_mean - float(oarm_cfg.reaction_time)
            if "hazard_logits" in critic:
                hazard_count = int(critic["hazard_logits"].shape[-1])
                hazard_last = critic["hazard_logits"].reshape(b, v, h, hazard_count)
                hazard_logits = hazard_last.permute(0, 3, 1, 2).contiguous()
                risk_logit = two_stage_risk_logit(
                    validity_logit.reshape(b, v, h),
                    zero_window_logit.reshape(b, v, h),
                    hazard_last,
                    reaction_budget_s=float(oarm_cfg.reaction_time),
                    hazard_max_time_s=self.rm_critic_hazard_max_time_s,
                ).reshape(b, 1, v, h)
            else:
                risk_logit = risk_logit_from_window(
                    reaction_window_mean,
                    reaction_window_logvar,
                    reaction_budget_s=float(oarm_cfg.reaction_time),
                )
        backup_logit = torch.zeros_like(risk_logit)
        utility_base = -score_pred
        if self.enable_utility_delta:
            utility_delta = self.utility_delta_scale * torch.tanh(aux[:, 2:3])
        else:
            utility_delta = torch.zeros_like(risk_logit)
        utility_score = utility_base + utility_delta.reshape(b, v, h)
        anchors = self.candidate_generator.yopo_anchors(b, device=depth.device)
        candidate_type = torch.full((b, v, h), self.candidate_generator.PROGRESS, device=depth.device, dtype=torch.long)
        candidate = OARMCandidate(
            end_state_b=end_state_b,
            traj_time=traj_time,
            yaw_terminal=yaw_terminal,
            margin_pred=margin_pred,
            risk_logit=risk_logit,
            backup_logit=backup_logit,
            utility_score=utility_score,
            candidate_type=candidate_type,
            frontier_score=torch.zeros((b, v, h), device=depth.device, dtype=depth.dtype),
            time_anchor=anchors.time_anchor.reshape(b, v, h),
            yaw_anchor=anchors.yaw_anchor.reshape(b, v, h),
            utility_base=utility_base,
            utility_delta=utility_delta.reshape(b, v, h),
            reaction_window_mean=reaction_window_mean,
            reaction_window_logvar=reaction_window_logvar,
            validity_logit=validity_logit,
            rm_insufficient_logit=rm_insufficient_logit,
            zero_window_logit=zero_window_logit,
            hazard_logits=hazard_logits,
        )
        if not self.append_brake_candidate:
            return candidate

        yopo_n = v * h
        end_flat = end_state_b.reshape(b, 9, 1, yopo_n)
        time_flat = traj_time.reshape(b, 1, 1, yopo_n)
        yaw_flat = yaw_terminal.reshape(b, 1, 1, yopo_n)
        margin_flat = margin_pred.reshape(b, 1, 1, yopo_n)
        risk_flat = risk_logit.reshape(b, 1, 1, yopo_n)
        backup_flat = backup_logit.reshape(b, 1, 1, yopo_n)
        type_flat = candidate_type.reshape(b, 1, yopo_n)
        frontier_flat = torch.zeros((b, 1, yopo_n), device=depth.device, dtype=depth.dtype)
        time_anchor_flat = anchors.time_anchor.reshape(b, 1, yopo_n)
        yaw_anchor_flat = anchors.yaw_anchor.reshape(b, 1, yopo_n)
        utility_base_flat = utility_base.reshape(b, 1, yopo_n)
        utility_delta_flat = torch.zeros_like(utility_base_flat)

        brake_end, brake_time = self.deterministic_brake_candidate(obs, depth.dtype, depth.device)
        brake_yaw = torch.zeros((b, 1, 1, 1), device=depth.device, dtype=depth.dtype)
        brake_margin = torch.full_like(margin_flat[:, :, :, :1], -float(oarm_cfg.reaction_time))
        brake_risk = torch.zeros_like(risk_flat[:, :, :, :1])
        brake_backup = torch.ones_like(brake_risk) * 5.0
        brake_type = torch.full((b, 1, 1), self.candidate_generator.BRAKE, device=depth.device, dtype=torch.long)
        brake_frontier = torch.zeros((b, 1, 1), device=depth.device, dtype=depth.dtype)
        brake_gate = aux[:, -1:, :1, :1].reshape(b, 1, 1)
        brake_base = utility_base_flat.max(dim=2, keepdim=True).values
        brake_utility = brake_base + brake_gate
        if reaction_window_mean is not None:
            window_flat = reaction_window_mean.reshape(b, 1, 1, yopo_n)
            logvar_flat = reaction_window_logvar.reshape(b, 1, 1, yopo_n)
            validity_flat = validity_logit.reshape(b, 1, 1, yopo_n)
            insufficient_flat = rm_insufficient_logit.reshape(b, 1, 1, yopo_n)
            zero_flat = zero_window_logit.reshape(b, 1, 1, yopo_n) if zero_window_logit is not None else insufficient_flat
            if hazard_logits is not None:
                hazard_flat = hazard_logits.reshape(b, hazard_logits.shape[1], 1, yopo_n)
                brake_hazard = torch.full_like(hazard_flat[:, :, :, :1], -10.0)
            else:
                hazard_flat = brake_hazard = None
            brake_window = torch.zeros_like(window_flat[:, :, :, :1])
            brake_logvar = torch.full_like(brake_window, 8.0)
            brake_validity = torch.full_like(brake_window, -10.0)
            brake_insufficient = torch.zeros_like(brake_window)
            brake_zero = torch.full_like(brake_window, -10.0)
        else:
            window_flat = logvar_flat = validity_flat = insufficient_flat = zero_flat = None
            hazard_flat = None
            brake_window = brake_logvar = brake_validity = brake_insufficient = brake_zero = None
            brake_hazard = None

        return OARMCandidate(
            end_state_b=torch.cat((end_flat, brake_end), dim=3),
            traj_time=torch.cat((time_flat, brake_time), dim=3),
            yaw_terminal=torch.cat((yaw_flat, brake_yaw), dim=3),
            margin_pred=torch.cat((margin_flat, brake_margin), dim=3),
            risk_logit=torch.cat((risk_flat, brake_risk), dim=3),
            backup_logit=torch.cat((backup_flat, brake_backup), dim=3),
            utility_score=torch.cat((utility_base_flat, brake_utility), dim=2),
            candidate_type=torch.cat((type_flat, brake_type), dim=2),
            frontier_score=torch.cat((frontier_flat, brake_frontier), dim=2),
            time_anchor=torch.cat((time_anchor_flat, brake_time.reshape(b, 1, 1)), dim=2),
            yaw_anchor=torch.cat((yaw_anchor_flat, torch.zeros((b, 1, 1), device=depth.device, dtype=depth.dtype)), dim=2),
            utility_base=torch.cat((utility_base_flat, brake_base), dim=2),
            utility_delta=torch.cat((utility_delta_flat, brake_gate), dim=2),
            reaction_window_mean=torch.cat((window_flat, brake_window), dim=3) if window_flat is not None else None,
            reaction_window_logvar=torch.cat((logvar_flat, brake_logvar), dim=3) if logvar_flat is not None else None,
            validity_logit=torch.cat((validity_flat, brake_validity), dim=3) if validity_flat is not None else None,
            rm_insufficient_logit=torch.cat((insufficient_flat, brake_insufficient), dim=3) if insufficient_flat is not None else None,
            zero_window_logit=torch.cat((zero_flat, brake_zero), dim=3) if zero_flat is not None else None,
            hazard_logits=torch.cat((hazard_flat, brake_hazard), dim=3) if hazard_flat is not None else None,
        )
