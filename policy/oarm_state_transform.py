import torch

from OARM.config import oarm_cfg
from OARM.policy.oarm_candidate_generator import OARMAnchorSet
from OARM.policy.oarm_types import OARMCandidate, OARMRawPrediction
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg
from policy.primitive import LatticePrimitive


class OARMStateTransform:
    """Decode OARM network outputs while preserving YOPO's primitive contract."""

    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.goal_length = cfg["goal_length"]
        self.segment_time = cfg["sgm_time"]
        self.time_min = oarm_cfg.time_min_ratio * self.segment_time
        self.time_max = oarm_cfg.time_max_ratio * self.segment_time

    def pred_to_candidate(self, raw: OARMRawPrediction, anchors: OARMAnchorSet = None) -> OARMCandidate:
        end_state_b = self.pred_to_endstate(raw.endstate_residual, anchors)
        traj_time = self.decode_time(raw.time_raw, anchors)
        yaw_terminal = self.decode_yaw(raw.yaw_raw, anchors)
        margin_pred = oarm_cfg.margin_scale * torch.tanh(raw.margin_raw)
        return OARMCandidate(
            end_state_b=end_state_b,
            traj_time=traj_time,
            yaw_terminal=yaw_terminal,
            margin_pred=margin_pred,
            risk_logit=raw.risk_logit,
            backup_logit=raw.backup_logit,
            utility_score=raw.utility_score,
            candidate_type=self.reshape_anchor_field(anchors.candidate_type, raw.utility_score) if anchors is not None else None,
            frontier_score=self.reshape_anchor_field(anchors.frontier_score, raw.utility_score) if anchors is not None and anchors.frontier_score is not None else None,
            time_anchor=self.reshape_anchor_field(anchors.time_anchor, raw.utility_score) if anchors is not None else None,
            yaw_anchor=self.reshape_anchor_field(anchors.yaw_anchor, raw.utility_score) if anchors is not None else None,
        )

    @staticmethod
    def reshape_anchor_field(anchor_field: torch.Tensor, lattice_like: torch.Tensor) -> torch.Tensor:
        b, v, h = lattice_like.shape
        return anchor_field.reshape(b, v, h)

    def pred_to_endstate(self, endstate_pred: torch.Tensor, anchors: OARMAnchorSet = None) -> torch.Tensor:
        """Decode endpoint residuals into body-frame PVA using OARM anchors."""

        b, v, h = endstate_pred.shape[0], endstate_pred.shape[2], endstate_pred.shape[3]
        pred = endstate_pred.permute(0, 2, 3, 1).reshape(b, v * h, 9)

        if anchors is None:
            yaw, pitch = self.lattice_primitive.getAngleLattice()
            yaw = yaw.flip(0).to(device=endstate_pred.device, dtype=endstate_pred.dtype)[None, :].expand(b, -1)
            pitch = pitch.flip(0).to(device=endstate_pred.device, dtype=endstate_pred.dtype)[None, :].expand(b, -1)
            radius_anchor = (2.0 * self.lattice_primitive.radio_range) * torch.ones_like(yaw)
        else:
            yaw = anchors.yaw.to(device=endstate_pred.device, dtype=endstate_pred.dtype)
            pitch = anchors.pitch.to(device=endstate_pred.device, dtype=endstate_pred.dtype)
            radius_anchor = anchors.radius.to(device=endstate_pred.device, dtype=endstate_pred.dtype)

        r_bp = self.lattice_primitive.getRotation().flip(0).to(device=endstate_pred.device, dtype=endstate_pred.dtype)
        r_bp = r_bp[None, :, :, :].expand(b, -1, -1, -1)

        delta_yaw = pred[:, :, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = pred[:, :, 1] * self.lattice_primitive.pitch_diff
        radius_delta = pred[:, :, 2] * 0.35 * self.lattice_primitive.radio_range
        radius = (radius_anchor + radius_delta).clamp(min=0.2, max=2.0 * self.lattice_primitive.radio_range)

        cos_pitch = torch.cos(pitch + delta_pitch)
        end_x = cos_pitch * torch.cos(yaw + delta_yaw) * radius
        end_y = cos_pitch * torch.sin(yaw + delta_yaw) * radius
        end_z = torch.sin(pitch + delta_pitch) * radius
        end_p = torch.stack([end_x, end_y, end_z], dim=-1)

        end_vp = pred[:, :, 3:6] * self.lattice_primitive.vel_max
        end_ap = pred[:, :, 6:9] * self.lattice_primitive.acc_max
        end_vb = torch.matmul(r_bp, end_vp.unsqueeze(-1)).squeeze(-1)
        end_ab = torch.matmul(r_bp, end_ap.unsqueeze(-1)).squeeze(-1)

        end_state = torch.cat([end_p, end_vb, end_ab], dim=-1)
        return end_state.permute(0, 2, 1).reshape(b, 9, v, h)

    def decode_time(self, time_raw: torch.Tensor, anchors: OARMAnchorSet = None) -> torch.Tensor:
        if anchors is None:
            return self.time_min + (self.time_max - self.time_min) * torch.sigmoid(time_raw)

        b, _, v, h = time_raw.shape
        time_anchor = anchors.time_anchor.to(device=time_raw.device, dtype=time_raw.dtype).reshape(b, 1, v, h)
        scale = 0.75 + 0.5 * torch.sigmoid(time_raw)
        return (time_anchor * scale).clamp(min=self.time_min, max=self.time_max)

    def decode_yaw(self, yaw_raw: torch.Tensor, anchors: OARMAnchorSet = None) -> torch.Tensor:
        b, _, v, h = yaw_raw.shape
        if anchors is None:
            yaw_anchor, _ = self.lattice_primitive.getAngleLattice()
            yaw_anchor = yaw_anchor.flip(0).to(device=yaw_raw.device, dtype=yaw_raw.dtype).reshape(1, 1, v, h).expand(b, -1, -1, -1)
        else:
            yaw_anchor = anchors.yaw_anchor.to(device=yaw_raw.device, dtype=yaw_raw.dtype).reshape(b, 1, v, h)
        return yaw_anchor + oarm_cfg.yaw_residual_limit_rad * torch.tanh(yaw_raw)

    def prepare_input(self, obs: torch.Tensor) -> torch.Tensor:
        b, n = obs.shape[0], self.lattice_primitive.traj_num
        r_bp_all = self.lattice_primitive.getRotation().flip(0).to(device=obs.device, dtype=obs.dtype)
        obs = obs.view(b, 3, 3)
        obs_exp = obs[:, None, :, :].expand(b, n, 3, 3)
        r_bp_exp = r_bp_all[None, :, :, :].expand(b, n, 3, 3)
        transformed = torch.matmul(obs_exp, r_bp_exp)
        out = transformed.view(b, n, 9).permute(0, 2, 1).contiguous()
        return out.view(b, 9, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)

    def normalize_obs(self, vel_acc_goal: torch.Tensor) -> torch.Tensor:
        vel_acc_goal = vel_acc_goal.clone()
        vel_acc_goal[:, 0:3] = vel_acc_goal[:, 0:3] / self.lattice_primitive.vel_max
        vel_acc_goal[:, 3:6] = vel_acc_goal[:, 3:6] / self.lattice_primitive.acc_max
        goal_norm = vel_acc_goal[:, 6:9].norm(dim=1, keepdim=True)
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] / goal_norm.clamp(min=self.goal_length)
        return vel_acc_goal


def rotate_body2world(rot_wb, pos_b):
    return torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)


def transform_body2world(rot_wb, t_w, pos_b):
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b
