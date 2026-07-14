import argparse
import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from OARM.config import oarm_cfg
from OARM.dataset import OARMDataset
from OARM.eval.eval_dataset import maybe_generate_reaction_margin_labels
from OARM.loss import OARMLoss
from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_state_transform import rotate_body2world, state_body2world
from OARM.utils.occlusion import candidate_frontier_overlap
from OARM.utils.visible_free_distance import visible_free_distance_from_depth
from OARM.visibility.first_visible_time import first_entry_time_to_points, first_visible_time
from OARM.visibility import soft_fov_score
from OARM.visibility.esdf_visibility import ESDFLineOfSight
from OARM.visibility.reaction_margin_labeler import ReactionMarginLabeler
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg
from policy.poly_solver import calculate_yaw


def check_soft_fov_shapes(device):
    n, t, q = 4, 30, 16
    observer_pos = torch.randn(n, t, 3, device=device)
    yaw_ref = torch.randn(n, t, device=device)
    points = torch.randn(n, q, 3, device=device)
    score = soft_fov_score(
        observer_pos,
        yaw_ref,
        points,
        horizon_fov_rad=math.radians(cfg["horizon_camera_fov"]),
        vertical_fov_rad=math.radians(cfg["vertical_camera_fov"]),
    )
    assert score.shape == (n, t, q), score.shape
    assert torch.isfinite(score).all()
    print("soft_fov_shapes ok", tuple(score.shape))


def check_occlusion_masked_first_visible_time(device):
    sampled_pos = torch.zeros(1, 4, 3, device=device)
    sampled_time = torch.tensor([[0.0, 0.5, 1.0, 1.5]], device=device)
    yaw_ref = torch.zeros(1, 4, device=device)
    risk_points = torch.tensor([[[2.0, 0.0, 0.0]]], device=device)
    fov_first = first_visible_time(sampled_pos, sampled_time, yaw_ref, risk_points, 1.57, 1.0)
    visibility_mask = torch.tensor([[[False], [False], [True], [True]]], device=device)
    masked_first = first_visible_time(
        sampled_pos,
        sampled_time,
        yaw_ref,
        risk_points,
        1.57,
        1.0,
        visibility_mask=visibility_mask,
    )
    assert torch.isclose(fov_first[0, 0], torch.tensor(0.0, device=device))
    assert torch.isclose(masked_first[0, 0], torch.tensor(1.0, device=device))
    print("occlusion_masked_first_visible_time ok")


def check_first_entry_precedes_closest_approach(device):
    sampled_time = torch.tensor([[0.0, 0.4, 0.8, 1.2, 1.6, 2.0]], device=device)
    sampled_pos = torch.stack([sampled_time, torch.zeros_like(sampled_time), torch.zeros_like(sampled_time)], dim=-1)
    risk_points = torch.tensor([[[1.8, 0.0, 0.0]]], device=device)
    entry_time, entry_valid = first_entry_time_to_points(sampled_pos, sampled_time, risk_points, radius_m=1.0)
    dist = (sampled_pos[:, :, None, :] - risk_points[:, None, :, :]).norm(dim=-1)
    closest_idx = dist.argmin(dim=1)
    closest_time = sampled_time.gather(1, closest_idx)
    assert entry_valid[0, 0]
    assert torch.isclose(entry_time[0, 0], torch.tensor(0.8, device=device), atol=1e-5), entry_time
    assert entry_time[0, 0] < closest_time[0, 0], (entry_time, closest_time)
    print('first_entry_precedes_closest_approach ok', float(entry_time.detach().cpu()))


class _ToyLOSBackend:
    def get_distance_cost(self, query_points, map_id):
        x = query_points[..., 0]
        y = query_points[..., 1]
        mid_block = (x > 0.45) & (x < 0.55) & (y > 0.45) & (y < 0.55)
        endpoint_surface = (x > 0.85) & (x < 1.05) & (y > -0.05) & (y < 0.05)
        blocked = mid_block | endpoint_surface
        dist = torch.where(blocked, torch.zeros_like(x), torch.ones_like(x))
        return torch.zeros_like(dist), dist


def check_los_short_ray_covers_full_ray(device):
    los = ESDFLineOfSight.__new__(ESDFLineOfSight)
    los.device = device
    los.ray_samples = 0
    los.ray_step_m = 0.1
    los.clearance_m = 0.25
    los.candidate_chunk = 0
    los.endpoint_guard_m = 0.0
    los.query_backend = _ToyLOSBackend()
    observer = torch.zeros(1, 1, 3, device=device)
    risk_points = torch.tensor([[[10.0, 2.0, 0.0], [1.0, 1.0, 0.0]]], device=device)
    map_id = torch.zeros(1, dtype=torch.long, device=device)
    visible = los(observer, risk_points, map_id)
    assert bool(visible[0, 0, 0])
    assert not bool(visible[0, 0, 1]), visible
    print('los_short_ray_covers_full_ray ok', visible.detach().cpu().tolist())


def check_los_endpoint_guard_avoids_surface_self_occlusion(device):
    los = ESDFLineOfSight.__new__(ESDFLineOfSight)
    los.device = device
    los.ray_samples = 0
    los.ray_step_m = 0.1
    los.clearance_m = 0.25
    los.candidate_chunk = 0
    los.endpoint_guard_m = 0.3
    los.query_backend = _ToyLOSBackend()
    observer = torch.zeros(1, 1, 3, device=device)
    risk_points = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]], device=device)
    map_id = torch.zeros(1, dtype=torch.long, device=device)
    visible = los(observer, risk_points, map_id)
    assert bool(visible[0, 0, 0]), visible
    assert not bool(visible[0, 0, 1]), visible
    print('los_endpoint_guard_avoids_surface_self_occlusion ok', visible.detach().cpu().tolist())

def check_no_entry_censoring_excludes_margin_regression(device):
    sampled_time = torch.tensor([[0.0, 0.5, 1.0, 1.5]], device=device)
    sampled_pos = torch.stack([sampled_time, torch.zeros_like(sampled_time), torch.zeros_like(sampled_time)], dim=-1)
    yaw_ref = torch.zeros_like(sampled_time)
    labeler = ReactionMarginLabeler(risk_arrival_radius_m=0.2)
    mixed_points = torch.tensor([[[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], device=device)
    mixed = labeler(sampled_pos, sampled_time, yaw_ref, mixed_points, torch.ones(1, 2, device=device))
    assert mixed['reaction_margin_point_valid'].tolist() == [[True, False]], mixed
    assert bool(mixed['reaction_margin_valid'][0])
    far_points = torch.tensor([[[10.0, 0.0, 0.0]]], device=device)
    far = labeler(sampled_pos, sampled_time, yaw_ref, far_points, torch.ones(1, 1, device=device))
    assert not bool(far['reaction_margin_valid'][0])
    assert bool(far['reaction_margin_censored'][0])
    print('no_entry_censoring_excludes_margin_regression ok')

def check_candidate_device(device):
    batch = 2
    depth = torch.rand(batch, 1, cfg["image_height"], cfg["image_width"], device=device)
    obs_b = torch.randn(batch, 9, device=device)
    obs_b[:, 6:9] = torch.tensor([cfg["goal_length"], 0.0, 0.0], device=device)
    policy = OARMNetwork().to(device)
    candidate = policy.inference(depth, obs_b)
    flat = candidate.flatten()
    assert candidate.end_state_b.device.type == device.type, candidate.end_state_b.device
    assert candidate.traj_time.device.type == device.type, candidate.traj_time.device
    assert "candidate_type" in flat
    assert flat["candidate_type"].numel() == batch * cfg["traj_num"]
    prepared_obs = policy.state_transform.prepare_input(policy.state_transform.normalize_obs(obs_b))
    frontier = policy.frontier_extractor(depth)
    frontier_lattice = candidate_frontier_overlap(frontier, cfg["vertical_num"], cfg["horizon_num"])
    anchors = policy.candidate_generator(batch, frontier_lattice)
    anchor_feature = policy.anchor_features(
        anchors,
        torch.empty(batch, 1, cfg["vertical_num"], cfg["horizon_num"], device=device),
    )
    raw = policy.forward(depth, prepared_obs, anchors)
    assert anchor_feature.shape == (batch, policy.anchor_feature_dim, cfg["vertical_num"], cfg["horizon_num"])
    assert raw.utility_score.shape == (batch, cfg["vertical_num"], cfg["horizon_num"])
    print("candidate_device ok", tuple(candidate.end_state_b.shape), device)


def check_risk_point_guidance(device):
    batch = 2
    q = 6
    depth = torch.rand(batch, 1, cfg["image_height"], cfg["image_width"], device=device)
    obs_b = torch.randn(batch, 9, device=device)
    obs_b[:, 6:9] = torch.tensor([cfg["goal_length"], 0.0, 0.0], device=device)
    pos = torch.zeros(batch, 3, device=device)
    rot = torch.eye(3, device=device).unsqueeze(0).expand(batch, -1, -1)

    policy = OARMNetwork().to(device)
    frontier = policy.frontier_extractor(depth)
    frontier_lattice = candidate_frontier_overlap(frontier, cfg["vertical_num"], cfg["horizon_num"])
    anchors = policy.candidate_generator(batch, frontier_lattice)
    prepared_obs = policy.state_transform.prepare_input(policy.state_transform.normalize_obs(obs_b))
    raw = policy.forward(depth, prepared_obs, anchors)
    raw.yaw_raw.retain_grad()
    raw.time_raw.retain_grad()
    candidate = policy.state_transform.pred_to_candidate(raw, anchors)
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
    risk_points_w = torch.randn(batch, q, 3, device=device)
    risk_points_w[..., 0] += 2.0
    labels = {
        "risk_points_w": risk_points_w,
        "risk_weight": torch.ones(batch, q, device=device),
        "yaw0": torch.zeros(batch, device=device),
    }
    loss = OARMLoss(
        enable_risk_point_guidance=True,
        enable_reaction_margin=True,
        enable_yaw_visibility=True,
        deployed_yaw_mode="predicted",
    )(start_state_w, end_state_w, flat, goal_w, labels)
    assert torch.isfinite(loss["total_loss"])
    assert "yaw_visibility_cost" in loss
    assert "margin_loss" in loss
    policy.zero_grad(set_to_none=True)
    loss["yaw_visibility_cost"].backward(retain_graph=True)
    grad_norm = torch.zeros((), device=device)
    for param in policy.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            grad_norm = grad_norm + param.grad.detach().abs().sum()
    assert grad_norm > 0.0
    assert raw.yaw_raw.grad is not None and torch.isfinite(raw.yaw_raw.grad).all()
    assert raw.yaw_raw.grad.detach().abs().sum() > 0.0
    policy.zero_grad(set_to_none=True)
    raw.time_raw.grad = None
    time_anchors = policy.candidate_generator.yopo_anchors(batch, device=device)
    time_raw = policy.forward(depth, prepared_obs, time_anchors)
    time_raw.time_raw.retain_grad()
    time_candidate = policy.state_transform.pred_to_candidate(time_raw, time_anchors)
    time_candidate.traj_time.sum().backward()
    assert time_raw.time_raw.grad is not None and torch.isfinite(time_raw.time_raw.grad).all()
    assert time_raw.time_raw.grad.detach().abs().sum() > 0.0
    print("risk_point_guidance ok", float(loss["total_loss"].detach().cpu()))


def check_goal_yaw_matches_yopo_calculate_yaw(device):
    n, t = 2, 12
    sampled_time = torch.linspace(0.0, 1.1, t, device=device).unsqueeze(0).expand(n, -1)
    x = sampled_time
    y = 0.2 * torch.sin(2.0 * sampled_time)
    z = torch.zeros_like(x)
    sampled_pos = torch.stack([x, y, z], dim=-1)
    sampled_vel = torch.zeros_like(sampled_pos)
    sampled_vel[:, 1:, :] = (sampled_pos[:, 1:, :] - sampled_pos[:, :-1, :]) / (
        sampled_time[:, 1:, None] - sampled_time[:, :-1, None]
    ).clamp(min=1e-3)
    sampled_vel[:, 0, :] = sampled_vel[:, 1, :]
    goal_w = torch.tensor([[3.0, 1.0, 0.0], [2.5, -0.8, 0.0]], device=device)
    yaw0 = torch.tensor([0.1, -0.2], device=device)
    yaw_ref, _ = OARMLoss.goal_oriented_yaw_like_ros(sampled_pos, sampled_vel, sampled_time, goal_w, yaw0)

    expected = []
    for batch_id in range(n):
        last_yaw = float(yaw0[batch_id].detach().cpu())
        last_time = float(sampled_time[batch_id, 0].detach().cpu())
        goal_pos = (sampled_pos[batch_id, 0] + goal_w[batch_id]).detach().cpu().numpy()
        yaw_values = []
        for step in range(t):
            current_time = float(sampled_time[batch_id, step].detach().cpu())
            dt = max(current_time - last_time, 1e-3)
            vel_dir = sampled_vel[batch_id, step].detach().cpu().numpy()
            goal_dir = goal_pos - sampled_pos[batch_id, step].detach().cpu().numpy()
            last_yaw, _ = calculate_yaw(vel_dir, goal_dir, last_yaw, dt)
            last_time = current_time
            yaw_values.append(last_yaw)
        expected.append(yaw_values)
    expected = torch.tensor(expected, dtype=yaw_ref.dtype, device=device)
    assert torch.allclose(yaw_ref, expected, atol=1e-5), (yaw_ref - expected).abs().max()
    print("goal_yaw_matches_yopo_calculate_yaw ok", float((yaw_ref - expected).abs().max().detach().cpu()))


class CaptureYawLabeler:
    def __call__(
        self,
        sampled_pos_w,
        sampled_time,
        yaw_ref,
        risk_points_w,
        risk_weight=None,
        visibility_mask=None,
    ):
        label = yaw_ref.mean(dim=1)
        return {
            "reaction_margin_softmin": label,
            "reaction_margin_min": label,
            "reaction_margin_valid": torch.ones_like(label, dtype=torch.bool),
            "reaction_margin_censored": torch.zeros_like(label, dtype=torch.bool),
            "arrival_time_min": torch.zeros_like(label),
        }


def check_eval_label_generation_uses_deployed_yaw_mode(device):
    start_state = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device)
    end_state = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device)
    flat = {
        "traj_time": torch.ones(1, device=device),
        "yaw_terminal": torch.zeros(1, device=device),
    }
    flat_labels = {
        "risk_points_w": torch.tensor([[[1.0, 1.0, 0.0]]], device=device),
        "risk_weight": torch.ones(1, 1, device=device),
        "yaw0": torch.zeros(1, device=device),
        "yaw_rate0": torch.zeros(1, device=device),
    }
    args = SimpleNamespace(eval_reaction_margin=True)
    map_id = torch.zeros(1, dtype=torch.long, device=device)
    goal_w = torch.tensor([[0.0, 5.0, 0.0]], device=device)
    goal_labels = maybe_generate_reaction_margin_labels(
        dict(flat_labels),
        flat,
        start_state,
        end_state,
        map_id,
        goal_w,
        args,
        CaptureYawLabeler(),
        None,
        OARMLoss(deployed_yaw_mode="goal"),
    )
    predicted_labels = maybe_generate_reaction_margin_labels(
        dict(flat_labels),
        flat,
        start_state,
        end_state,
        map_id,
        goal_w,
        args,
        CaptureYawLabeler(),
        None,
        OARMLoss(deployed_yaw_mode="predicted"),
    )
    assert "reaction_margin_valid" in goal_labels
    assert not torch.allclose(goal_labels["reaction_margin"], predicted_labels["reaction_margin"])
    print("eval_label_generation_uses_deployed_yaw_mode ok")


def check_dataset_sample(dataset_root=None):
    try:
        dataset = OARMDataset(mode="train", dataset_root=dataset_root)
    except FileNotFoundError as exc:
        print("dataset_sample skipped:", exc)
        return False
    depth, pos, rot, obs_b, map_id, labels = dataset[0]
    assert depth.ndim == 3 and depth.shape[0] == 1, tuple(depth.shape)
    assert labels["occlusion_risk"].shape[-2:] == (cfg["vertical_num"], cfg["horizon_num"])
    assert labels["frontier_backup_feasible"].shape[-2:] == (cfg["vertical_num"], cfg["horizon_num"])
    assert labels["risk_points_w"].shape == (oarm_cfg.risk_point_count, 3)
    assert labels["risk_weight"].shape == (oarm_cfg.risk_point_count,)
    assert labels["risk_esdf"].shape == (oarm_cfg.risk_point_count,)
    print("dataset_sample ok", tuple(depth.shape), labels["occlusion_risk"].shape)
    return True


def check_one_batch_loss(device, batch_size, train_occlusion_risk, train_reaction_margin, train_backup_feasibility, dataset_root=None, use_occlusion_aware_visibility=False, risk_label_source="proxy"):
    try:
        dataset = OARMDataset(mode="train", dataset_root=dataset_root, risk_label_source=risk_label_source)
    except (FileNotFoundError, ValueError) as exc:
        print("one_batch_loss skipped:", exc)
        return False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    depth, pos, rot, obs_b, map_id, labels = next(iter(loader))
    depth = depth.to(device)
    pos = pos.to(device)
    rot = rot.to(device)
    obs_b = obs_b.to(device)
    map_id = map_id.to(device)

    policy = OARMNetwork().to(device)
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

    flat_labels = {}
    if train_occlusion_risk:
        flat_labels["occlusion_risk"] = labels["occlusion_risk"].to(device).reshape(-1)
    if train_reaction_margin and "reaction_margin" in labels:
        flat_labels["reaction_margin"] = labels["reaction_margin"].to(device).reshape(-1)
    if train_backup_feasibility:
        flat_labels["visible_free_distance"] = visible_free_distance_from_depth(depth, flat["end_state_b"][:, 0:3])
    if "risk_points_w" in labels:
        flat_labels["risk_points_w"] = labels["risk_points_w"].to(device)
        flat_labels["risk_weight"] = labels["risk_weight"].to(device)
        flat_labels["yaw0"] = labels["yaw0"].to(device)
        flat_labels["yaw_rate0"] = labels["yaw_rate0"].to(device)

    loss_fn = OARMLoss(
        enable_occlusion_risk=train_occlusion_risk,
        enable_risk_point_guidance=True,
        enable_reaction_margin=train_reaction_margin,
        enable_yaw_visibility=False,
        use_occlusion_aware_visibility=use_occlusion_aware_visibility,
        deployed_yaw_mode="goal",
        enable_yield_feasibility=train_backup_feasibility,
    )
    map_id_expanded = map_id.repeat_interleave(cfg["traj_num"], dim=0) if use_occlusion_aware_visibility else None
    loss = loss_fn(start_state_w, end_state_w, flat, goal_w, flat_labels, map_id_expanded)
    assert torch.isfinite(loss["total_loss"])
    if train_backup_feasibility:
        assert "backup_feasible_rate" in loss
        assert "stop_distance" in loss
    print("one_batch_loss ok", float(loss["total_loss"].detach().cpu()))
    return True


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--dataset-root", type=str, default="")
    p.add_argument("--one-batch-loss", action="store_true")
    p.add_argument("--train-occlusion-risk", action="store_true")
    p.add_argument("--train-reaction-margin", action="store_true")
    p.add_argument("--train-backup-feasibility", action="store_true")
    p.add_argument("--use-occlusion-aware-visibility", action="store_true")
    p.add_argument("--risk-label-source", choices=["proxy", "proxy_esdf", "gt_pointcloud"], default="proxy")
    return p


def main():
    args = parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    check_soft_fov_shapes(device)
    check_occlusion_masked_first_visible_time(device)
    check_first_entry_precedes_closest_approach(device)
    check_los_short_ray_covers_full_ray(device)
    check_los_endpoint_guard_avoids_surface_self_occlusion(device)
    check_no_entry_censoring_excludes_margin_regression(device)
    check_candidate_device(device)
    check_risk_point_guidance(device)
    check_goal_yaw_matches_yopo_calculate_yaw(device)
    check_eval_label_generation_uses_deployed_yaw_mode(device)
    if not args.skip_dataset:
        check_dataset_sample(args.dataset_root or None)
    if args.one_batch_loss:
        check_one_batch_loss(
            device,
            args.batch_size,
            args.train_occlusion_risk,
            args.train_reaction_margin,
            args.train_backup_feasibility,
            args.dataset_root or None,
            args.use_occlusion_aware_visibility,
            args.risk_label_source,
        )


if __name__ == "__main__":
    main()
