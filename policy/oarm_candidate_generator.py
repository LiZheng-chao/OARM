from dataclasses import dataclass

import torch

from OARM.config import oarm_cfg
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from policy.primitive import LatticePrimitive


@dataclass
class OARMAnchorSet:
    yaw: torch.Tensor
    pitch: torch.Tensor
    radius: torch.Tensor
    time_anchor: torch.Tensor
    yaw_anchor: torch.Tensor
    candidate_type: torch.Tensor
    frontier_score: torch.Tensor = None


class OARMCandidateGenerator:
    """OARM-V3 anchor generator scaffold.

    V0-V2 still use YOPO's lattice anchors. This class adds explicit candidate
    types so later experiments can split progress/probe/brake anchors without
    changing the network output contract.
    """

    PROGRESS = 0
    PROBE = 1
    BRAKE = 2
    YIELD = 3
    BACKUP = YIELD  # compatibility alias for older checkpoints/scripts.

    def __init__(self, fast_time: float, normal_time: float, brake_time: float, enable_yield: bool = False):
        self.lattice = LatticePrimitive.get_instance()
        self.fast_time = fast_time
        self.normal_time = normal_time
        self.brake_time = brake_time
        self.enable_yield = bool(enable_yield)

    def __call__(self, batch_size: int, frontier_score: torch.Tensor = None) -> OARMAnchorSet:
        device = frontier_score.device if frontier_score is not None else None
        return self.typed_anchors_from_frontier(batch_size, frontier_score, device=device)

    def yopo_anchors(self, batch_size: int, device=None) -> OARMAnchorSet:
        yaw, pitch = self.lattice.getAngleLattice()
        radius = self.lattice.getStateLattice().norm(dim=-1)
        device = device or yaw.device
        yaw = yaw.flip(0).to(device)
        pitch = pitch.flip(0).to(device)
        radius = radius.flip(0).to(device)
        n = yaw.numel()
        return OARMAnchorSet(
            yaw=yaw[None, :].expand(batch_size, -1),
            pitch=pitch[None, :].expand(batch_size, -1),
            radius=radius[None, :].expand(batch_size, -1),
            time_anchor=torch.full((batch_size, n), self.normal_time, device=device),
            yaw_anchor=yaw[None, :].expand(batch_size, -1),
            candidate_type=torch.full((batch_size, n), self.PROGRESS, device=device, dtype=torch.long),
            frontier_score=None,
        )

    def typed_anchors_from_frontier(self, batch_size: int, frontier_score: torch.Tensor = None, device=None) -> OARMAnchorSet:
        """Generate typed anchors from depth-frontier evidence.

        Low-frontier cells keep fast progress anchors. Medium frontier cells
        become probe anchors. Higher-risk cells become brake or yield anchors.
        This is still a lightweight proxy, but the decoding path now actually
        depends on OARM anchors instead of hard-coded YOPO lattice anchors.
        """

        anchors = self.yopo_anchors(batch_size, device=device)
        n = anchors.yaw.shape[1]
        candidate_type = anchors.candidate_type.clone()
        time_anchor = anchors.time_anchor.clone()
        radius = anchors.radius.clone()
        yaw_anchor = anchors.yaw_anchor.clone()
        score = None

        if frontier_score is None:
            if self.enable_yield:
                quarter = max(1, n // 4)
                candidate_type[:, quarter : 2 * quarter] = self.PROBE
                candidate_type[:, 2 * quarter : 3 * quarter] = self.BRAKE
                candidate_type[:, 3 * quarter :] = self.YIELD
            else:
                third = max(1, n // 3)
                candidate_type[:, third : 2 * third] = self.PROBE
                candidate_type[:, 2 * third :] = self.BRAKE
        else:
            score = frontier_score.reshape(batch_size, -1).to(device=anchors.yaw.device, dtype=anchors.yaw.dtype)
            candidate_type = torch.full_like(candidate_type, self.PROGRESS)
            candidate_type = torch.where(
                score > oarm_cfg.frontier_probe_threshold,
                torch.full_like(candidate_type, self.PROBE),
                candidate_type,
            )
            candidate_type = torch.where(
                score > oarm_cfg.frontier_brake_threshold,
                torch.full_like(candidate_type, self.BRAKE),
                candidate_type,
            )
            if self.enable_yield:
                candidate_type = torch.where(
                    score > oarm_cfg.frontier_yield_threshold,
                    torch.full_like(candidate_type, self.YIELD),
                    candidate_type,
                )
            candidate_type = self.ensure_typed_anchor_coverage(candidate_type, score)

        progress_mask = candidate_type == self.PROGRESS
        probe_mask = candidate_type == self.PROBE
        brake_mask = candidate_type == self.BRAKE
        yield_mask = candidate_type == self.YIELD

        time_anchor = torch.where(progress_mask, torch.full_like(time_anchor, self.fast_time), time_anchor)
        time_anchor = torch.where(probe_mask, torch.full_like(time_anchor, self.normal_time), time_anchor)
        time_anchor = torch.where(brake_mask | yield_mask, torch.full_like(time_anchor, self.brake_time), time_anchor)

        radius = torch.where(probe_mask, 0.9 * radius, radius)
        radius = torch.where(brake_mask, radius.clamp(max=2.5), radius)
        radius = torch.where(yield_mask, radius.clamp(max=0.9), radius)

        yaw_diff = torch.as_tensor(self.lattice.yaw_diff, device=anchors.yaw.device, dtype=anchors.yaw.dtype)
        probe_yaw_shift = 0.5 * yaw_diff * torch.sign(anchors.yaw.clamp(min=-1e-6))
        yaw_anchor = torch.where(probe_mask, yaw_anchor + probe_yaw_shift, yaw_anchor)
        return OARMAnchorSet(
            yaw=anchors.yaw,
            pitch=anchors.pitch,
            radius=radius,
            time_anchor=time_anchor,
            yaw_anchor=yaw_anchor,
            candidate_type=candidate_type,
            frontier_score=score,
        )

    def ensure_typed_anchor_coverage(self, candidate_type: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Reserve a few high-frontier lattice cells for brake/yield behavior.

        Depth-frontier scores are often sparse after adaptive pooling. Without
        this coverage floor, the Full policy can enable yield losses while
        never receiving yield candidates at inference time.
        """

        b, n = candidate_type.shape
        probe_count = max(1, int(round(oarm_cfg.typed_anchor_min_probe_frac * n)))
        brake_count = max(1, int(round(oarm_cfg.typed_anchor_min_brake_frac * n)))
        yield_count = max(1, int(round(oarm_cfg.typed_anchor_min_yield_frac * n))) if self.enable_yield else 0
        order = torch.argsort(score, dim=1, descending=True)
        out = candidate_type.clone()
        for batch_id in range(b):
            cursor = 0
            max_score = score[batch_id].amax()
            active_yield_count = yield_count if self.enable_yield and max_score > oarm_cfg.frontier_yield_threshold else 0
            active_brake_count = brake_count if max_score > oarm_cfg.frontier_brake_threshold else 0
            active_probe_count = probe_count if max_score > oarm_cfg.frontier_probe_threshold else 0
            yield_idx = order[batch_id, cursor : cursor + active_yield_count]
            out[batch_id, yield_idx] = self.YIELD
            cursor += active_yield_count
            brake_idx = order[batch_id, cursor : cursor + active_brake_count]
            out[batch_id, brake_idx] = self.BRAKE
            cursor += active_brake_count
            probe_idx = order[batch_id, cursor : cursor + active_probe_count]
            out[batch_id, probe_idx] = torch.maximum(
                out[batch_id, probe_idx],
                torch.full_like(out[batch_id, probe_idx], self.PROBE),
            )
        return out

    def generate_yield_candidates(self, batch_size: int, device=None) -> OARMAnchorSet:
        """Short-horizon stop/yield anchors.

        The method name is kept for compatibility with earlier experiments.
        """

        anchors = self.yopo_anchors(batch_size, device=device)
        radius = anchors.radius.clamp(max=1.2)
        time_anchor = torch.full_like(anchors.time_anchor, self.brake_time)
        candidate_type = torch.full_like(anchors.candidate_type, self.YIELD)
        return OARMAnchorSet(
            yaw=anchors.yaw,
            pitch=anchors.pitch,
            radius=radius,
            time_anchor=time_anchor,
            yaw_anchor=anchors.yaw_anchor,
            candidate_type=candidate_type,
            frontier_score=anchors.frontier_score,
        )

    def generate_backup_candidates(self, batch_size: int, device=None) -> OARMAnchorSet:
        return self.generate_yield_candidates(batch_size, device=device)
