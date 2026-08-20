import torch
from torch import nn

from OARM.config import oarm_cfg
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
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
        return self

    def adapt_legacy_aux_state_dict(self, state_dict):
        adapted = dict(state_dict)
        prefix = ""
        if any(key.startswith("preserve_network.") for key in adapted):
            prefix = "preserve_network."

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
        ignored_prefixes = ("margin_risk_head.", "rerank_head.", "brake_gate_head.")
        missing = [key for key in missing if not key.startswith(ignored_prefixes)]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "YOPO checkpoint did not match YOPO-preserve network: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def forward(self, depth: torch.Tensor, obs_prepared: torch.Tensor):
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
        return endstate_pred, score_pred, aux

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
        endstate_pred, score_pred, aux = self.forward(depth, obs_prepared)
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
        brake_margin = margin_flat.mean(dim=3, keepdim=True)
        brake_risk = risk_flat.mean(dim=3, keepdim=True)
        brake_backup = torch.ones_like(brake_risk) * 5.0
        brake_type = torch.full((b, 1, 1), self.candidate_generator.BRAKE, device=depth.device, dtype=torch.long)
        brake_frontier = torch.zeros((b, 1, 1), device=depth.device, dtype=depth.dtype)
        brake_gate = aux[:, -1:, :1, :1].reshape(b, 1, 1)
        brake_base = utility_base_flat.max(dim=2, keepdim=True).values
        brake_utility = brake_base + brake_gate

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
        )
