import warnings

import torch


def load_oarm_checkpoint(path, map_location=None, weights_only=True):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
        return checkpoint["state_dict"], metadata
    return checkpoint, {}


def make_oarm_checkpoint(
    state_dict,
    candidate_mode,
    backbone_mode,
    training_options=None,
    enable_yield_candidates=None,
    deployed_yaw_mode=None,
):
    training_options = dict(training_options or {})
    if enable_yield_candidates is None:
        enable_yield_candidates = training_options.get("enable_yield_candidates")
    if deployed_yaw_mode is None:
        deployed_yaw_mode = training_options.get("deployed_yaw_mode")
    return {
        "state_dict": state_dict,
        "candidate_mode": candidate_mode,
        "backbone_mode": backbone_mode,
        "enable_yield_candidates": enable_yield_candidates,
        "deployed_yaw_mode": deployed_yaw_mode,
        "stage": training_options.get("stage"),
        "train_yaw_visibility": training_options.get("train_yaw_visibility"),
        "train_margin_ranking": training_options.get("train_margin_ranking"),
        "training_options": training_options,
        "checkpoint_format": "oarm_policy_v2",
    }


def validate_checkpoint_metadata(
    metadata,
    candidate_mode,
    backbone_mode,
    allow_mismatch=False,
    enable_yield_candidates=None,
    deployed_yaw_mode=None,
):
    mismatches = []
    expected = {
        "candidate_mode": candidate_mode,
        "backbone_mode": backbone_mode,
    }
    if enable_yield_candidates is not None:
        expected["enable_yield_candidates"] = bool(enable_yield_candidates)
    if deployed_yaw_mode is not None:
        expected["deployed_yaw_mode"] = deployed_yaw_mode
    for key, value in expected.items():
        stored = metadata.get(key)
        if stored is None:
            training_options = metadata.get("training_options") or {}
            stored = training_options.get(key)
        if stored is not None and str(stored) != str(value):
            mismatches.append(f"{key}: checkpoint={stored}, requested={value}")
    if not mismatches:
        return
    message = "OARM checkpoint metadata mismatch: " + "; ".join(mismatches)
    if allow_mismatch:
        warnings.warn(message, RuntimeWarning)
        return
    raise ValueError(message + ". Pass --allow-checkpoint-mismatch to override intentionally.")
