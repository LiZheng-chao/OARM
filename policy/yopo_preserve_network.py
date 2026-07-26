import torch
from torch import nn

from OARM.config import oarm_cfg
from OARM.policy.oarm_candidate_generator import OARMCandidateGenerator
from OARM.policy.oarm_types import OARMCandidate
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.primitive import LatticePrimitive
from policy.state_transform import StateTransform


class YOPOPreserveOARMNetwork(nn.Module):
    """YOPO-preserving A0/A1 policy.

    The trajectory branch is the official YOPO backbone/head/decode path:
    endpoint PVA comes from StateTransform.pred_to_endstate(), time is the
    fixed lattice segment time, and utility is -YOPO score so OARM's argmax
    selector is equivalent to YOPO's argmin(score). OARM auxiliary heads can
    learn margin/risk labels, but they do not affect selection.
    """

    def __init__(
        self,
        observation_dim=9,
        hidden_state=64,
        freeze_yopo_base=True,
        margin_scale=oarm_cfg.margin_scale,
    ):
        super().__init__()
        self.candidate_mode = "yopo_preserve"
        self.backbone_mode = "yopo_original"
        self.margin_scale = float(margin_scale)
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
        self.aux_head = nn.Sequential(
            nn.Conv2d(hidden_state + observation_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1),
        )
        self.reset_aux_head()
        if freeze_yopo_base:
            self.freeze_yopo_base()

    def reset_aux_head(self):
        final = self.aux_head[-1]
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
        self.aux_head.train(mode)
        return self

    def load_yopo_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        adapted = {}
        for key, value in state_dict.items():
            if key in own_state:
                adapted[key] = value
            elif key.startswith("module.") and key[len("module.") :] in own_state:
                adapted[key[len("module.") :]] = value
        missing, unexpected = self.load_state_dict(adapted, strict=False)
        missing = [key for key in missing if not key.startswith("aux_head.")]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "YOPO checkpoint did not match YOPO-preserve network: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return missing, unexpected

    def forward(self, depth: torch.Tensor, obs_prepared: torch.Tensor):
        # Keep the official YOPO planning path mathematically isolated during A1.
        # Only aux_head receives gradients; YOPO features, endpoint, and score are frozen references.
        with torch.no_grad():
            depth_feature = self.image_backbone(depth)
            obs_feature = self.state_backbone(obs_prepared)
            features = torch.cat((obs_feature, depth_feature), dim=1)
            output = self.yopo_head(features)
            endstate_pred = torch.tanh(output[:, :9])
            score_pred = torch.nn.functional.softplus(output[:, 9])
        aux = self.aux_head(features.detach())
        return endstate_pred, score_pred, aux

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> OARMCandidate:
        obs_norm = self.state_transform.normalize_obs(obs)
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
        utility_score = -score_pred
        anchors = self.candidate_generator.yopo_anchors(b, device=depth.device)
        candidate_type = torch.full((b, v, h), self.candidate_generator.PROGRESS, device=depth.device, dtype=torch.long)
        return OARMCandidate(
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
        )
