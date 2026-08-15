import torch

from OARM.policy.oarm_poly_solver import quintic_coefficients, sample_polynomial


def expand_candidate_label(label, candidate_count, like):
    label = label.to(device=like.device, dtype=like.dtype)
    if label.shape[0] == candidate_count:
        return label
    if candidate_count % label.shape[0] != 0:
        raise ValueError(f"Cannot expand label with first dim {label.shape[0]} to {candidate_count} candidates")
    return label.repeat_interleave(candidate_count // label.shape[0], dim=0)


def sampled_time_grid(traj_time, eval_points, include_zero=True):
    start = 0.0 if include_zero else 1.0 / eval_points
    tau = torch.linspace(start, 1.0, eval_points, device=traj_time.device, dtype=traj_time.dtype)
    return traj_time[:, None] * tau[None, :]


def generate_reaction_margin_labels(
    flat_labels,
    flat,
    start_state_w,
    end_state_w,
    map_id_expanded,
    goal_w,
    *,
    enabled,
    labeler,
    line_of_sight,
    yaw_helper,
    eval_points=30,
    include_diagnostics=False,
    risk_weight_override=None,
):
    """Generate candidate reaction-margin labels from risk points when raw labels are absent."""
    if not enabled:
        return flat_labels
    if "reaction_margin" in flat_labels:
        current = flat_labels["reaction_margin"]
        if current.shape[0] == flat["traj_time"].shape[0]:
            return flat_labels
        for key in ("reaction_margin", "reaction_margin_valid", "reaction_margin_censored"):
            flat_labels.pop(key, None)
    if "risk_points_w" not in flat_labels:
        return flat_labels

    traj_time = flat["traj_time"]
    coeff = quintic_coefficients(start_state_w, end_state_w, traj_time)
    sampled_pos, sampled_vel, _, _ = sample_polynomial(coeff, traj_time, eval_points, include_zero=True)
    sampled_time = sampled_time_grid(traj_time, eval_points, include_zero=True)

    risk_points_w = expand_candidate_label(flat_labels["risk_points_w"], traj_time.shape[0], traj_time)
    risk_weight = risk_weight_override if risk_weight_override is not None else flat_labels.get("risk_weight")
    if risk_weight is None:
        risk_weight = torch.ones(risk_points_w.shape[:-1], device=traj_time.device, dtype=traj_time.dtype)
    else:
        risk_weight = expand_candidate_label(risk_weight, traj_time.shape[0], traj_time)

    yaw0 = flat_labels.get("yaw0")
    if yaw0 is None:
        yaw0 = torch.zeros_like(traj_time)
    else:
        yaw0 = expand_candidate_label(yaw0, traj_time.shape[0], traj_time)

    yaw_rate0 = flat_labels.get("yaw_rate0")
    if yaw_rate0 is None:
        yaw_rate0 = torch.zeros_like(traj_time)
    else:
        yaw_rate0 = expand_candidate_label(yaw_rate0, traj_time.shape[0], traj_time)

    yaw_ref, _ = yaw_helper.deployed_yaw_reference(
        yaw0,
        yaw_rate0,
        flat["yaw_terminal"],
        traj_time,
        sampled_pos,
        sampled_vel,
        sampled_time,
        goal_w,
    )

    visibility_mask = None
    if line_of_sight is not None:
        visibility_mask = line_of_sight(sampled_pos, risk_points_w, map_id_expanded.reshape(-1))

    margin_labels = labeler(
        sampled_pos,
        sampled_time,
        yaw_ref,
        risk_points_w,
        risk_weight,
        visibility_mask=visibility_mask,
    )
    flat_labels["reaction_margin"] = margin_labels["reaction_margin_softmin"].detach()
    flat_labels["reaction_margin_valid"] = margin_labels["reaction_margin_valid"].detach()
    flat_labels["reaction_margin_censored"] = margin_labels["reaction_margin_censored"].detach()
    if include_diagnostics:
        flat_labels["reaction_margin_min"] = margin_labels["reaction_margin_min"].detach()
        flat_labels["reaction_margin_arrival_time_min"] = margin_labels["arrival_time_min"].detach()
    return flat_labels
