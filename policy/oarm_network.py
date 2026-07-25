import torch
from torch import nn

from OARM.config import oarm_cfg
from OARM.policy.oarm_backbone import OARMBackbone, YOPOOriginalBackbone
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_head import OARMHead
from OARM.policy.oarm_state_transform import OARMStateTransform
from OARM.policy.oarm_types import OARMRawPrediction
from OARM.policy.yopo_preserve_network import YOPOPreserveOARMNetwork
from OARM.utils.occlusion import DepthFrontierExtractor, candidate_frontier_overlap
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


class OARMNetwork(nn.Module):
    """Occlusion-aware reaction-margin planner head.

    Per primitive output layout for legacy OARM modes:
    [dtheta, dphi, dr, vT(3), aT(3), T, yaw_T, margin, risk_logit, backup_logit, utility]
    """

    def __init__(
        self,
        observation_dim=9,
        output_dim=oarm_cfg.output_dim,
        hidden_state=64,
        candidate_mode=oarm_cfg.candidate_mode,
        backbone_mode=oarm_cfg.backbone_mode,
        enable_yield_candidates=oarm_cfg.enable_yield_candidates,
    ):
        super().__init__()
        self.preserve_network = None
        if candidate_mode == "yopo_preserve":
            if backbone_mode != "yopo_original":
                raise ValueError("candidate_mode=yopo_preserve requires backbone_mode=yopo_original")
            self.candidate_mode = candidate_mode
            self.backbone_mode = backbone_mode
            self.preserve_network = YOPOPreserveOARMNetwork(
                observation_dim=observation_dim,
                hidden_state=hidden_state,
                freeze_yopo_base=True,
            )
            return
        if output_dim != 15:
            raise ValueError("OARMNetwork currently expects output_dim=15")
        if candidate_mode not in {"yopo", "typed_frontier"}:
            raise ValueError(f"Unknown OARM candidate_mode: {candidate_mode}")
        if backbone_mode not in {"oarm_light", "yopo_original"}:
            raise ValueError(f"Unknown OARM backbone_mode: {backbone_mode}")
        self.candidate_mode = candidate_mode
        self.backbone_mode = backbone_mode
        self.anchor_feature_dim = 11
        self.state_transform = OARMStateTransform()
        self.frontier_extractor = DepthFrontierExtractor()
        self.candidate_generator = OARMCandidateGenerator(
            fast_time=self.state_transform.time_min,
            normal_time=oarm_cfg.probe_time_ratio * self.state_transform.segment_time,
            brake_time=oarm_cfg.stop_time_ratio * self.state_transform.segment_time,
            enable_yield=enable_yield_candidates,
        )
        if backbone_mode == "yopo_original":
            self.image_backbone = YOPOOriginalBackbone(hidden_state)
        else:
            self.image_backbone = OARMBackbone(hidden_state, cfg["vertical_num"], cfg["horizon_num"])
        self.state_backbone = nn.Sequential()
        self.oarm_head = OARMHead(hidden_state + observation_dim + self.anchor_feature_dim, output_dim)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor, anchors=None):
        if self.preserve_network is not None:
            return self.preserve_network(depth, obs)
        if anchors is None:
            anchors = self.candidate_generator.yopo_anchors(depth.shape[0], device=depth.device)
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)
        anchor_feature = self.anchor_features(anchors, depth_feature)
        input_tensor = torch.cat((obs_feature, depth_feature, anchor_feature), dim=1)
        output = self.oarm_head(input_tensor)
        return OARMRawPrediction(
            endstate_residual=torch.tanh(output[:, 0:9]),
            time_raw=output[:, 9:10],
            yaw_raw=output[:, 10:11],
            margin_raw=output[:, 11:12],
            risk_logit=output[:, 12:13],
            backup_logit=output[:, 13:14],
            utility_score=output[:, 14],
        )

    def anchor_features(self, anchors, lattice_like: torch.Tensor) -> torch.Tensor:
        b, _, v, h = lattice_like.shape
        dtype = lattice_like.dtype
        device = lattice_like.device

        def grid(field):
            return field.to(device=device, dtype=dtype).reshape(b, 1, v, h)

        yaw = grid(anchors.yaw) / torch.pi
        pitch = grid(anchors.pitch) / (0.5 * torch.pi)
        radius = grid(anchors.radius) / max(float(self.candidate_generator.lattice.radio_range), 1e-3)
        time_anchor = (grid(anchors.time_anchor) - self.state_transform.segment_time) / max(
            self.state_transform.segment_time, 1e-3
        )
        yaw_anchor = grid(anchors.yaw_anchor)
        if anchors.frontier_score is None:
            frontier_score = torch.zeros((b, 1, v, h), device=device, dtype=dtype)
        else:
            frontier_score = grid(anchors.frontier_score)

        candidate_type = anchors.candidate_type.to(device=device).reshape(b, v, h)
        type_channels = [
            (candidate_type == self.candidate_generator.PROGRESS).to(dtype).unsqueeze(1),
            (candidate_type == self.candidate_generator.PROBE).to(dtype).unsqueeze(1),
            (candidate_type == self.candidate_generator.BRAKE).to(dtype).unsqueeze(1),
            (candidate_type == self.candidate_generator.YIELD).to(dtype).unsqueeze(1),
        ]
        return torch.cat(
            [
                yaw,
                pitch,
                radius,
                time_anchor,
                torch.sin(yaw_anchor),
                torch.cos(yaw_anchor),
                frontier_score,
                *type_channels,
            ],
            dim=1,
        )

    def inference(self, depth: torch.Tensor, obs: torch.Tensor):
        if self.preserve_network is not None:
            return self.preserve_network.inference(depth, obs)
        if self.candidate_mode == "yopo":
            anchors = self.candidate_generator.yopo_anchors(depth.shape[0], device=depth.device)
        else:
            frontier_map = self.frontier_extractor(depth)
            frontier_lattice = candidate_frontier_overlap(frontier_map, cfg["vertical_num"], cfg["horizon_num"])
            anchors = self.candidate_generator(depth.shape[0], frontier_lattice)
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        raw = self.forward(depth, obs, anchors)
        return self.state_transform.pred_to_candidate(raw, anchors)