import argparse

import torch

from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_state_transform import rotate_body2world, state_body2world
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-mode", choices=["yopo", "typed_frontier"], default="typed_frontier")
    p.add_argument("--backbone-mode", choices=["oarm_light", "yopo_original"], default="oarm_light")
    return p


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 2
    depth = torch.rand(batch, 1, cfg["image_height"], cfg["image_width"], device=device)
    obs_b = torch.randn(batch, 9, device=device)
    obs_b[:, 6:9] = torch.tensor([cfg["goal_length"], 0.0, 0.0], device=device)
    pos = torch.zeros(batch, 3, device=device)
    rot = torch.eye(3, device=device).unsqueeze(0).expand(batch, -1, -1)

    policy = OARMNetwork(candidate_mode=args.candidate_mode, backbone_mode=args.backbone_mode).to(device)
    candidate = policy.inference(depth, obs_b)
    flat = candidate.flatten()

    goal_w = rotate_body2world(rot, obs_b[:, 6:9])
    start_vel_w = rotate_body2world(rot, obs_b[:, 0:3])
    start_acc_w = rotate_body2world(rot, obs_b[:, 3:6])
    start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1).repeat_interleave(cfg["traj_num"], dim=0)
    goal_w = goal_w.repeat_interleave(cfg["traj_num"], dim=0)
    pos_expanded = pos.repeat_interleave(cfg["traj_num"], dim=0)
    rot_expanded = rot.repeat_interleave(cfg["traj_num"], dim=0)
    end = flat["end_state_b"]
    end_pos_w, end_vel_w, end_acc_w = state_body2world(
        pos_expanded, rot_expanded, end[:, 0:3], end[:, 3:6], end[:, 6:9]
    )
    end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

    labels = {
        "occlusion_risk": torch.rand(batch * cfg["traj_num"], device=device),
        "backup_feasible": torch.randint(0, 2, (batch * cfg["traj_num"],), device=device).float(),
    }
    loss = OARMLoss()(start_state_w, end_state_w, flat, goal_w, labels)
    print("OARM smoke test ok")
    print("candidate_mode:", args.candidate_mode)
    print("backbone_mode:", args.backbone_mode)
    print("candidate end_state_b:", tuple(candidate.end_state_b.shape))
    print("candidate traj_time:", tuple(candidate.traj_time.shape))
    print("candidate yield/stopping logit:", tuple(candidate.backup_logit.shape))
    print("total_loss:", float(loss["total_loss"].detach().cpu()))


if __name__ == "__main__":
    main(parser().parse_args())
