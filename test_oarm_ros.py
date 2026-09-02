import argparse
import json
import os
import sys
import time
from threading import Lock

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROS_IMPORT_ERROR = None
try:
    import rospy
    import std_msgs.msg
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Odometry, Path
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from visualization_msgs.msg import Marker
except ModuleNotFoundError as exc:
    ROS_IMPORT_ERROR = exc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
YOPO_DIR = os.path.join(REPO_ROOT, "YOPO")
if YOPO_DIR not in sys.path:
    sys.path.insert(0, YOPO_DIR)

from OARM.policy.oarm_network import OARMNetwork
from OARM.policy.oarm_intervention_selector import (
    BrakeInterventionLatch,
    BrakeLatchConfig,
    InterventionSelectorConfig,
    OARMInterventionSelector,
)
from OARM.policy.oarm_brake import brake_visible_clearance_margin, constrained_brake_command, deterministic_brake_endpoint
from OARM.policy.oarm_latency_model import OARMLatencyModel, is_sensor_frame_stale
from OARM.policy.oarm_risk_calibrator import TemperatureCalibration, calibrated_probability, risk_upper_bound
from OARM.policy.oarm_rm_critic import risk_probability_from_window, two_stage_risk_probability
from OARM.utils.checkpoint import load_oarm_checkpoint, validate_checkpoint_metadata
from OARM.policy.oarm_state_transform import OARMStateTransform
from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()
from config.config import cfg
from policy.poly_solver import Poly5Solver, calculate_yaw

try:
    from control_msg import PositionCommand
except ModuleNotFoundError as exc:
    if ROS_IMPORT_ERROR is None:
        ROS_IMPORT_ERROR = exc


class OARMNet:
    """ROS inference node for OARM.

    This mirrors YOPO/test_yopo_ros.py but uses OARM's decoded candidate set:
    utility_score selects the candidate, and each candidate carries its own
    trajectory time.
    """

    def __init__(self, config, weight):
        if ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS Python modules are unavailable. Source your ROS workspace before running "
                "OARM/test_oarm_ros.py, for example: source Controller/devel/setup.bash"
            ) from ROS_IMPORT_ERROR
        self.config = config
        rospy.init_node("oarm_net", anonymous=False)

        cfg["train"] = False
        self.height = cfg["image_height"]
        self.width = cfg["image_width"]
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.array(self.config["goal"], dtype=np.float32)
        self.yopo_preserve_mode = self.config.get("candidate_mode") in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}
        self.goal_received = self.config["start_immediately"]
        self.plan_from_reference = self.config["plan_from_reference"]
        self.verbose = self.config["verbose"]
        self.visualize = self.config["visualize"]
        self.debug = self.config["debug"]
        self.log_candidate_table = bool(self.config.get("log_candidate_table", False))
        self.main_experiment = self.config["main_experiment"]
        self.selector_experiment = self.config.get("selector_experiment", False)
        self.position_control_mode = self.config["position_control_mode"]
        self.fast_sim_mode = self.config["fast_sim_mode"]
        self.progress_bonus_weight = self.config["progress_bonus_weight"]
        self.agile_progress_weight = self.config.get("agile_progress_weight", 0.0)
        self.agile_goal_distance_weight = self.config.get("agile_goal_distance_weight", 0.0)
        self.agile_lateral_penalty = self.config.get("agile_lateral_penalty", 0.0)
        self.agile_time_penalty = self.config.get("agile_time_penalty", 0.0)
        self.agile_stop_penalty = self.config.get("agile_stop_penalty", 0.0)
        self.oarm_margin_alpha = float(self.config.get("oarm_margin_alpha", 0.0))
        self.oarm_risk_beta = float(self.config.get("oarm_risk_beta", 0.0))
        self.selector_min_goal_drop_rate = self.config.get("selector_min_goal_drop_rate", None)
        self.selector_max_lateral_rate = self.config.get("selector_max_lateral_rate", None)
        self.depth_clearance_weight = float(self.config.get("depth_clearance_weight", 0.0))
        self.depth_clearance_min = float(self.config.get("depth_clearance_min", 0.35))
        self.depth_clearance_gate = bool(self.config.get("depth_clearance_gate", False))
        self.depth_clearance_samples = int(self.config.get("depth_clearance_samples", 9))
        self.depth_clearance_pixel_radius = int(self.config.get("depth_clearance_pixel_radius", 1))
        self.use_depth_header_sensor_age = bool(self.config.get("use_depth_header_sensor_age", True))
        self.max_depth_sensor_age_s = max(float(self.config.get("max_depth_sensor_age_ms", 250.0)), 0.0) / 1000.0
        self.stale_depth_drop_count = 0
        self.last_depth_sensor_age_s = None
        self.depth_emergency_stop = bool(self.config.get("depth_emergency_stop", False))
        self.depth_emergency_clearance = float(self.config.get("depth_emergency_clearance", self.depth_clearance_min))
        self.depth_emergency_critical_clearance = float(self.config.get("depth_emergency_critical_clearance", 0.15))
        self.depth_emergency_speed_threshold = float(self.config.get("depth_emergency_speed_threshold", 1.0))
        self.depth_emergency_traj_time = float(self.config.get("depth_emergency_traj_time", 0.45))
        self.depth_emergency_distance_scale = float(self.config.get("depth_emergency_distance_scale", 0.0))
        self.depth_emergency_retreat_distance = float(self.config.get("depth_emergency_retreat_distance", 0.0))
        self.depth_emergency_target_z = self.config.get("depth_emergency_target_z", None)
        self.depth_emergency_z_rate = float(self.config.get("depth_emergency_z_rate", 0.8))
        self.brake_decel_mps2 = float(self.config.get("brake_decel_mps2", 3.0))
        self.brake_max_time = float(self.config.get("brake_max_time", 5.0))
        self.brake_max_accel_mps2 = float(self.config.get("brake_max_accel_mps2", 6.0))
        self.brake_max_jerk_mps3 = float(self.config.get("brake_max_jerk_mps3", 30.0))
        self.brake_max_thrust_accel_mps2 = float(self.config.get("brake_max_thrust_accel_mps2", 18.0))
        self.brake_max_tilt_deg = float(self.config.get("brake_max_tilt_deg", 50.0))
        self.brake_sample_count = int(self.config.get("brake_sample_count", 81))
        self.brake_time_growth = float(self.config.get("brake_time_growth", 1.25))
        self.brake_verified_risk_upper = float(self.config.get("brake_verified_risk_upper", self.config.get("risk_threshold_keep", 0.10)))
        self.brake_require_visible_stop_distance = bool(self.config.get("brake_require_visible_stop_distance", True))
        self.brake_latch = BrakeInterventionLatch(
            BrakeLatchConfig(
                enabled=bool(self.config.get("brake_latch_enabled", True)),
                min_hold_s=float(self.config.get("brake_latch_min_hold_s", 0.6)),
                release_speed_mps=float(self.config.get("brake_latch_release_speed_mps", 0.25)),
                release_frames=int(self.config.get("brake_latch_release_frames", 3)),
                release_risk=float(self.config.get("risk_threshold_keep", 0.10)),
                require_release_evidence=bool(self.config.get("brake_latch_require_release_evidence", True)),
            )
        )
        self.brake_probe_enabled = bool(self.config.get("brake_probe_enabled", True))
        self.brake_probe_yaw_amplitude_rad = np.deg2rad(float(self.config.get("brake_probe_yaw_deg", 30.0)))
        self.brake_probe_period_s = max(float(self.config.get("brake_probe_period_s", 2.0)), 0.1)
        self.brake_probe_min_stationary_s = max(float(self.config.get("brake_probe_min_stationary_s", 2.0)), 0.0)
        self.brake_latch_anchor_w = None
        self.brake_latch_anchor_time_s = None
        self.brake_probe_stationary_start_s = None
        self.brake_probe_yaw_center = None
        self.last_depth_emergency_stop = False
        self.last_depth_emergency_reason = None
        self.last_depth_emergency_target = None
        self.last_deterministic_brake_stop = False
        self.last_deterministic_brake_reason = None
        self.last_deterministic_brake_terminal_speed = None
        self.last_deterministic_brake_terminal_acc_norm = None
        self.last_deterministic_brake_diagnostics = None
        self.last_brake_candidate_info = None
        self.last_brake_command = None
        self.last_selector_force_emergency_stop = False
        self.last_selector_force_emergency_reason = None
        self.last_stop_fallback_count = None
        self.last_stop_fallback_altitude_valid_count = None
        self.selector_min_traj_z = self.config.get("selector_min_traj_z", None)
        self.selector_max_traj_z = self.config.get("selector_max_traj_z", None)
        self.altitude_band_weight = float(self.config.get("altitude_band_weight", 0.0))
        self.altitude_band_samples = int(self.config.get("altitude_band_samples", self.depth_clearance_samples))
        self.camera_fx = self.config.get("camera_fx", None)
        self.camera_fy = self.config.get("camera_fy", None)
        self.camera_cx = self.config.get("camera_cx", None)
        self.camera_cy = self.config.get("camera_cy", None)
        hfov = float(self.config.get("camera_hfov_deg", cfg["horizon_camera_fov"]))
        vfov = float(self.config.get("camera_vfov_deg", cfg["vertical_camera_fov"]))
        self.camera_fx = float(self.camera_fx) if self.camera_fx is not None else (self.width - 1) / (2.0 * np.tan(np.deg2rad(hfov) * 0.5))
        self.camera_fy = float(self.camera_fy) if self.camera_fy is not None else (self.height - 1) / (2.0 * np.tan(np.deg2rad(vfov) * 0.5))
        self.camera_cx = float(self.camera_cx) if self.camera_cx is not None else (self.width - 1) * 0.5
        self.camera_cy = float(self.camera_cy) if self.camera_cy is not None else (self.height - 1) * 0.5
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        self.depth_forward_axis = self.config.get("depth_forward_axis", "x")
        self.depth_horizontal_axis = self.config.get("depth_horizontal_axis", "y")
        self.depth_vertical_axis = self.config.get("depth_vertical_axis", "z")
        self.depth_forward_sign = float(self.config.get("depth_forward_sign", 1.0))
        self.depth_horizontal_sign = float(self.config.get("depth_horizontal_sign", 1.0))
        self.depth_vertical_sign = float(self.config.get("depth_vertical_sign", -1.0))
        depth_axes = (self.depth_forward_axis, self.depth_horizontal_axis, self.depth_vertical_axis)
        if any(axis not in axis_to_index for axis in depth_axes) or len(set(depth_axes)) != 3:
            raise ValueError(
                "Depth projection axes must be a permutation of x/y/z: "
                f"forward={self.depth_forward_axis}, horizontal={self.depth_horizontal_axis}, "
                f"vertical={self.depth_vertical_axis}"
            )
        self.depth_forward_index = axis_to_index[self.depth_forward_axis]
        self.depth_horizontal_index = axis_to_index[self.depth_horizontal_axis]
        self.depth_vertical_index = axis_to_index[self.depth_vertical_axis]
        self.arrival_distance = self.config.get("arrival_distance", 1.0)
        self.accept_rviz_goal = self.config.get("accept_rviz_goal", True)
        self.goal_segment_id = int(self.config.get("goal_segment_id", 0))
        self.hover_on_arrival = self.config.get("hover_on_arrival", True)
        self.first_arrival_time = None
        self.min_goal_distance = None
        self.run_id = self.config.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
        self.data_split = str(self.config.get("data_split", "adhoc"))
        self.episode_id = self.config.get("episode_id") or self.run_id
        self.method = self.config.get("method", "oarm")
        self.scenario = self.config.get("scenario", "unknown")
        self.seed = int(self.config.get("seed", 0))
        self.checkpoint_path = self.config.get("checkpoint", "")
        enable_rm_critic = self.config.get("enable_rm_critic", None)
        if enable_rm_critic is None and self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            try:
                state_for_rm, _metadata_for_rm = load_oarm_checkpoint(self.checkpoint_path, map_location="cpu")
                enable_rm_critic = any(key.startswith("preserve_network.rm_critic.") for key in state_for_rm)
            except Exception as exc:
                rospy.logwarn(f"Could not inspect checkpoint RM critic state; using disabled RM critic: {exc}")
                enable_rm_critic = False
        self.enable_rm_critic = bool(enable_rm_critic)
        self.config["enable_rm_critic"] = self.enable_rm_critic
        self.rm_critic_hazard_bins = int(self.config.get("rm_critic_hazard_bins") or 0)
        self.rm_critic_hazard_max_time_s = float(self.config.get("rm_critic_hazard_max_time_s", 2.5))
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            try:
                state_for_hazard, metadata_for_hazard = load_oarm_checkpoint(self.checkpoint_path, map_location="cpu")
                training_options = metadata_for_hazard.get("training_options") or {}
                meta_bins = metadata_for_hazard.get("rm_critic_hazard_bins", training_options.get("rm_critic_hazard_bins"))
                if meta_bins is not None and self.rm_critic_hazard_bins <= 0:
                    self.rm_critic_hazard_bins = int(meta_bins)
                meta_horizon = metadata_for_hazard.get("rm_critic_hazard_max_time_s", training_options.get("rm_critic_hazard_max_time_s"))
                if meta_horizon is not None and "rm_critic_hazard_max_time_s" not in self.config:
                    self.rm_critic_hazard_max_time_s = float(meta_horizon)
                final_bias = state_for_hazard.get("preserve_network.rm_critic.mlp.4.bias")
                if final_bias is not None:
                    inferred_bins = max(0, int(final_bias.numel()) - 4)
                    if inferred_bins > 0:
                        self.rm_critic_hazard_bins = inferred_bins
            except Exception as exc:
                rospy.logwarn(f"Could not inspect checkpoint RM hazard metadata; using configured hazard bins={self.rm_critic_hazard_bins}: {exc}")
        self.config["rm_critic_hazard_bins"] = self.rm_critic_hazard_bins
        self.config["rm_critic_hazard_max_time_s"] = self.rm_critic_hazard_max_time_s
        latency_aware_risk = self.config.get("enable_latency_aware_risk", None)
        self.enable_latency_aware_risk = self.enable_rm_critic if latency_aware_risk is None else bool(latency_aware_risk)
        self.config["enable_latency_aware_risk"] = self.enable_latency_aware_risk
        self.latency_model = OARMLatencyModel(
            brake_accel_mps2=float(self.config.get("latency_brake_accel_mps2", 6.0)),
            sensor_age_s=float(self.config.get("sensor_age_ms", 0.0)) / 1000.0,
            queue_latency_s=float(self.config.get("queue_latency_ms", 0.0)) / 1000.0,
            selector_latency_s=float(self.config.get("selector_latency_ms", 0.0)) / 1000.0,
            control_latency_s=float(self.config.get("control_latency_ms", 20.0)) / 1000.0,
            actuation_latency_s=float(self.config.get("actuation_latency_ms", 30.0)) / 1000.0,
            reaction_margin_s=float(self.config.get("reaction_budget_margin_ms", 0.0)) / 1000.0,
            latency_window=int(self.config.get("latency_window", 128)),
            quantile=float(self.config.get("latency_quantile", 0.95)),
        )
        self.last_latency_budget = None
        self.warned_latency_risk_unavailable = False
        self.warned_latency_brake_unavailable = False
        validity_fusion_cfg = self.config.get("use_validity_risk_fusion", None)
        self.use_validity_risk_fusion = self.enable_rm_critic if validity_fusion_cfg is None else bool(validity_fusion_cfg)
        self.validity_unknown_risk = float(self.config.get("validity_unknown_risk", 0.5))
        self.use_calibrated_risk = bool(self.config.get("use_calibrated_risk", False))
        calibration_file = self.config.get("calibration_file", "") or ""
        self.calibration_metadata = {}
        if calibration_file:
            with open(calibration_file, "r", encoding="utf-8") as calibration_stream:
                self.calibration_metadata = json.load(calibration_stream)
            self.risk_calibration = TemperatureCalibration.from_file(calibration_file)
            self.calibration_version = os.path.basename(calibration_file)
        else:
            self.risk_calibration = TemperatureCalibration()
            self.calibration_version = "identity"
        if self.config.get("risk_conformal_slack", None) is not None:
            self.risk_calibration.conformal_slack = float(self.config.get("risk_conformal_slack"))
        self.enable_intervention_selector = bool(self.config.get("enable_intervention_selector", False))
        risk_threshold_keep = float(self.config.get("risk_threshold_keep", 0.10))
        risk_threshold_safe = float(self.config.get("risk_threshold_safe", 0.20))
        if not 0.0 <= risk_threshold_keep <= risk_threshold_safe <= 1.0:
            raise ValueError(
                "Intervention risk thresholds must satisfy "
                f"0 <= keep <= safe <= 1, got keep={risk_threshold_keep}, safe={risk_threshold_safe}"
            )
        if self.main_experiment and self.enable_intervention_selector:
            if not calibration_file or not self.use_calibrated_risk:
                raise ValueError(
                    "Formal intervention requires --calibration-file and --use-calibrated-risk. "
                    "Collect a held-out calibration split first; omit --main-experiment only for a diagnostic smoke."
                )
            allowed_labels = {
                "reaction_window_lt_budget",
                "insufficient_reaction_gt",
                "rm_violation_gt",
                "selected_rm_violation_gt",
            }
            calibration_stats = self.calibration_metadata.get("input_stats") or {}
            if (
                int(self.calibration_metadata.get("sample_count", 0)) <= 0
                or self.calibration_metadata.get("label_key") not in allowed_labels
                or self.calibration_metadata.get("calibration_split") != "calibration"
                or not bool(self.calibration_metadata.get("require_split", False))
                or not bool(self.calibration_metadata.get("require_episode_id", False))
                or int(calibration_stats.get("episode_count", 0)) < 2
            ):
                raise ValueError(
                    "Formal intervention calibration must contain candidate-level reaction-risk labels, "
                    "the calibration split, and at least two explicitly identified episodes."
                )
        if self.main_experiment and self.enable_intervention_selector and not self.brake_latch.config.enabled:
            raise ValueError("Formal intervention requires the brake latch; remove --disable-brake-latch.")
        if self.main_experiment and self.enable_intervention_selector and (
            not self.brake_probe_enabled or not self.brake_latch.config.require_release_evidence
        ):
            raise ValueError(
                "Formal intervention requires the stationary visibility probe before releasing BRAKE.")
        self.intervention_selector = OARMInterventionSelector(
            InterventionSelectorConfig(
                delta_keep=risk_threshold_keep,
                delta_safe=risk_threshold_safe,
                risk_improvement_min=float(self.config.get("risk_improvement_min", 0.02)),
                lambda_risk=float(self.config.get("selector_lambda_risk", 1.0)),
            )
        )
        agile_bonus_enabled = any(
            abs(float(value)) > 1e-9
            for value in (
                self.agile_progress_weight,
                self.agile_goal_distance_weight,
                self.agile_lateral_penalty,
                self.agile_time_penalty,
                self.agile_stop_penalty,
                self.depth_clearance_weight,
                self.oarm_margin_alpha,
                self.oarm_risk_beta,
            )
        )
        if self.main_experiment and (self.fast_sim_mode or self.progress_bonus_weight != 0.0 or agile_bonus_enabled):
            raise ValueError(
                "main_experiment requires all online selector bonuses to be zero. "
                "Use --selector-experiment without --main-experiment for selector ablations."
            )
        if self.main_experiment and self.selector_experiment:
            raise ValueError("--main-experiment and --selector-experiment are mutually exclusive.")
        if agile_bonus_enabled and not self.selector_experiment:
            raise ValueError(
                "Online selector bonuses require --selector-experiment. "
                "Use --main-experiment only for learned-utility paper runs."
            )
        self.min_command_z = self.config["min_command_z"]
        self.max_command_z = self.config["max_command_z"]
        self.path_max_points = self.config["path_max_points"]
        self.log_jsonl_path = self.config.get("log_jsonl", "")
        self.exec_log_jsonl_path = self.config.get("exec_log_jsonl", "")
        self.log_jsonl_file = None
        self.exec_log_jsonl_file = None
        log_mode = "a" if self.config.get("append_logs", False) else "w"
        if self.log_jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_jsonl_path)), exist_ok=True)
            self.log_jsonl_file = open(self.log_jsonl_path, log_mode, encoding="utf-8")
        if self.exec_log_jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.exec_log_jsonl_path)), exist_ok=True)
            self.exec_log_jsonl_file = open(self.exec_log_jsonl_path, log_mode, encoding="utf-8")
        if self.log_jsonl_file is not None or self.exec_log_jsonl_file is not None:
            rospy.on_shutdown(self.close_log_jsonl)
        self.Rotation_bc = R.from_euler("ZYX", [0, self.config["pitch_angle_deg"], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.selected_traj_time = cfg["sgm_time"]
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = OARMStateTransform()
        self.lattice_primitive = self.state_transform.lattice_primitive

        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_count = 0
        self.control_count = 0
        self.last_status_time = 0.0
        self.last_nav_viz_time = 0.0
        self.last_goal_distance = None
        self.last_selected_type = None
        self.last_selected_time = None
        self.last_selected_end_norm = None
        self.last_selector_valid_count = None
        self.last_selector_total_count = None
        self.last_depth_clearance_selected = None
        self.last_depth_clearance_min = None
        self.last_depth_clearance_valid_count = None
        self.last_depth_clearance_total_count = None
        self.last_altitude_valid_count = None
        self.last_altitude_total_count = None
        self.last_candidate_min_z = None
        self.last_candidate_max_z = None
        self.warned_no_ctrl_subscriber = False
        self.depth_fps = 30
        self.executed_path = Path()
        self.executed_path.header.frame_id = "world"

        utility_delta_scale = self.config.get("yopo_preserve_utility_delta_scale")
        if utility_delta_scale is None and self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            try:
                _state_dict_for_scale, scale_metadata = load_oarm_checkpoint(self.checkpoint_path, map_location="cpu")
                utility_delta_scale = scale_metadata.get("yopo_preserve_utility_delta_scale")
                if utility_delta_scale is None:
                    utility_delta_scale = (scale_metadata.get("training_options") or {}).get("yopo_preserve_utility_delta_scale")
            except Exception as exc:
                rospy.logwarn(f"Could not inspect checkpoint utility_delta_scale; using default 0.35: {exc}")
        if utility_delta_scale is None:
            utility_delta_scale = 0.35
        self.config["yopo_preserve_utility_delta_scale"] = float(utility_delta_scale)

        self.policy = OARMNetwork(
            candidate_mode=self.config.get("candidate_mode", "typed_frontier"),
            backbone_mode=self.config.get("backbone_mode", "yopo_original"),
            enable_yield_candidates=self.config.get("enable_yield_candidates", False),
            utility_delta_scale=self.config["yopo_preserve_utility_delta_scale"],
            enable_rm_critic=self.enable_rm_critic,
            rm_critic_hazard_bins=self.rm_critic_hazard_bins,
            rm_critic_hazard_max_time_s=self.rm_critic_hazard_max_time_s,
        ).to(self.device)
        self.load_policy(weight)
        self.policy.eval()
        self.warm_up()

        self.lattice_traj_pub = rospy.Publisher("/oarm_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/oarm_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/oarm_net/trajs_visual", PointCloud2, queue_size=1)
        self.executed_path_pub = rospy.Publisher("/oarm_net/executed_path", Path, queue_size=1)
        self.goal_marker_pub = rospy.Publisher("/oarm_net/goal_marker", Marker, queue_size=1)
        self.goal_line_pub = rospy.Publisher("/oarm_net/goal_line", Marker, queue_size=1)
        self.status_text_pub = rospy.Publisher("/oarm_net/status_text", Marker, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)

        self.odom_sub = rospy.Subscriber(
            self.config["odom_topic"], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True
        )
        self.depth_sub = rospy.Subscriber(
            self.config["depth_topic"], Image, self.callback_depth, queue_size=1, buff_size=2 ** 24, tcp_nodelay=True
        )
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)

        rospy.sleep(1.0)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("OARM Net Node Ready!")
        self.print_topic_status()
        rospy.spin()

    def load_policy(self, weight):
        if not weight:
            raise ValueError("OARM checkpoint path is empty. Pass --checkpoint or --trial/--epoch.")
        if not os.path.isfile(weight):
            raise FileNotFoundError(f"OARM checkpoint not found: {weight}")
        state_dict, checkpoint_metadata = load_oarm_checkpoint(weight, map_location=self.device)
        candidate_mode = self.config.get("candidate_mode", "typed_frontier")
        backbone_mode = self.config.get("backbone_mode", "yopo_original")
        if candidate_mode in {"yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"}:
            is_oarm_preserve_checkpoint = any(key.startswith("preserve_network.") for key in state_dict)
            if is_oarm_preserve_checkpoint:
                validate_checkpoint_metadata(
                    checkpoint_metadata,
                    candidate_mode,
                    backbone_mode,
                    allow_mismatch=self.config.get("allow_checkpoint_mismatch", False),
                    enable_yield_candidates=self.config.get("enable_yield_candidates", False),
                    deployed_yaw_mode=self.config.get("deployed_yaw_mode", "goal"),
                    yopo_preserve_utility_delta_scale=self.config.get("yopo_preserve_utility_delta_scale", 0.35),
                )
                self.policy.load_state_dict(state_dict)
                rospy.loginfo(f"Loaded OARM {candidate_mode} checkpoint: {weight}")
                return
            self.policy.preserve_network.load_yopo_state_dict(state_dict, strict=True)
            rospy.loginfo(f"Loaded official YOPO checkpoint into {candidate_mode} policy: {weight}")
            return
        validate_checkpoint_metadata(
            checkpoint_metadata,
            candidate_mode,
            backbone_mode,
            allow_mismatch=self.config.get("allow_checkpoint_mismatch", False),
            enable_yield_candidates=self.config.get("enable_yield_candidates", False),
            deployed_yaw_mode=self.config.get("deployed_yaw_mode", "goal"),
            yopo_preserve_utility_delta_scale=self.config.get("yopo_preserve_utility_delta_scale", 0.35),
        )
        self.policy.load_state_dict(state_dict)

    def callback_set_goal(self, data):
        if not self.accept_rviz_goal:
            rospy.loginfo("Ignoring RViz goal because --disable-rviz-goal is set.")
            return
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2.0], dtype=np.float32)
        self.goal_received = True
        self.arrive = False
        self.goal_segment_id += 1
        self.first_arrival_time = None
        self.min_goal_distance = None
        self.last_goal_distance = None
        self.last_selected_type = None
        self.last_selected_time = None
        self.last_selected_end_norm = None
        self.reset_executed_path()
        self.brake_latch.active = False
        self.brake_latch.hold_until_s = 0.0
        self.brake_latch.safe_release_frames = 0
        self.brake_latch_anchor_w = None
        self.brake_latch_anchor_time_s = None
        self.brake_probe_stationary_start_s = None
        self.brake_probe_yaw_center = None
        if self.odom_init:
            self.desire_pos = np.array(
                (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z)
            )
            self.desire_vel = np.array(
                (self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z)
            )
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            self.ctrl_time = None
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array(
                (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z)
            )
            self.desire_vel = np.array(
                (self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z)
            )
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat(
                [
                    self.odom.pose.pose.orientation.x,
                    self.odom.pose.pose.orientation.y,
                    self.odom.pose.pose.orientation.z,
                    self.odom.pose.pose.orientation.w,
                ]
            ).as_euler("ZYX", degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos, _vel, _yaw = self.get_odom_state()
        self.last_goal_distance = float(np.linalg.norm(pos - self.goal))
        if self.min_goal_distance is None:
            self.min_goal_distance = self.last_goal_distance
        else:
            self.min_goal_distance = min(self.min_goal_distance, self.last_goal_distance)
        self.publish_navigation_visuals(pos)
        if self.goal_received and self.last_goal_distance < self.arrival_distance and not self.arrive:
            print("Arrive!")
            self.arrive = True
            self.first_arrival_time = float(time.time())
        self.write_exec_log(pos)

    def process_odom(self):
        rotation_wb = R.from_quat(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        ).as_matrix()
        self.Rotation_wc = np.dot(rotation_wb, self.Rotation_bc)
        rotation_cw = self.Rotation_wc.T

        vel_w = (
            self.desire_vel
            if self.plan_from_reference
            else np.array(
                [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z]
            )
        )
        vel_c = np.dot(rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(rotation_cw, acc_w)

        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        return torch.from_numpy(obs[None, :])

    def build_constrained_brake_command(self):
        generation_start = time.time()
        start_pos = self.get_start_pos().astype(np.float32)
        start_vel = self.get_start_vel().astype(np.float32)
        start_acc = self.desire_acc if self.desire_acc is not None else np.zeros(3, dtype=np.float32)
        start_acc = np.asarray(start_acc, dtype=np.float32).reshape(3)
        min_brake_time = float(np.clip(self.depth_emergency_traj_time, 0.1, self.brake_max_time))
        command = constrained_brake_command(
            start_pos,
            start_vel,
            start_acc=start_acc,
            goal=self.goal,
            min_time=min_brake_time,
            brake_accel=self.brake_decel_mps2,
            max_time=self.brake_max_time,
            max_accel=self.brake_max_accel_mps2,
            max_jerk=self.brake_max_jerk_mps3,
            max_thrust_accel=self.brake_max_thrust_accel_mps2,
            max_tilt_deg=self.brake_max_tilt_deg,
            sample_count=self.brake_sample_count,
            time_growth=self.brake_time_growth,
            target_z=self.depth_emergency_target_z,
            z_rate=self.depth_emergency_z_rate,
            min_command_z=self.min_command_z,
            max_command_z=self.max_command_z,
        )
        generation_latency_ms = 1000.0 * (time.time() - generation_start)
        return command, start_pos, start_vel, start_acc, generation_latency_ms

    def depth_sensor_age_s(self, depth_msg):
        sensor_age_s = None
        if self.use_depth_header_sensor_age and getattr(depth_msg, "header", None) is not None:
            try:
                stamp = depth_msg.header.stamp
                if stamp is not None and not stamp.is_zero():
                    sensor_age_s = max(0.0, (rospy.Time.now() - stamp).to_sec())
            except Exception:
                sensor_age_s = None
        return sensor_age_s

    def estimate_reaction_budget(
        self, depth_msg, obs, inference_latency_s, brake_bundle=None, brake_generation_latency_ms=0.0,
        sensor_age_s=None,
    ):
        if sensor_age_s is None:
            sensor_age_s = self.depth_sensor_age_s(depth_msg)
        velocity_body = obs.detach().cpu().numpy()[0, :3]
        speed_parallel_mps = None
        brake_duration_s = None
        brake_distance_m = None
        if brake_bundle is not None:
            brake_command, _start_pos, start_vel, _start_acc, _generation_ms = brake_bundle
            speed_parallel_mps = float(np.linalg.norm(start_vel))
            brake_duration_s = float(brake_command.duration)
            brake_distance_m = float(brake_command.diagnostics.stop_distance)
        budget = self.latency_model.estimate(
            speed_parallel_mps=speed_parallel_mps,
            velocity_body_mps=velocity_body,
            inference_latency_s=float(inference_latency_s),
            sensor_age_s=sensor_age_s,
            selector_latency_s=max(float(brake_generation_latency_ms), 0.0) / 1000.0,
            maneuver_latency_s=brake_duration_s,
            brake_distance_m=brake_distance_m,
        )
        self.last_latency_budget = budget
        return budget

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init:
            return
        if not self.goal_received:
            return
        if self.arrive:
            if self.hover_on_arrival and not self.yopo_preserve_mode:
                self.publish_arrival_hover()
            return

        sensor_age_s = self.depth_sensor_age_s(data)
        self.last_depth_sensor_age_s = sensor_age_s
        if is_sensor_frame_stale(sensor_age_s, self.max_depth_sensor_age_s):
            self.stale_depth_drop_count += 1
            rospy.logwarn_throttle(
                2.0,
                "Dropping stale depth frame: age=%.1f ms exceeds max=%.1f ms (dropped=%d)",
                1000.0 * sensor_age_s,
                1000.0 * self.max_depth_sensor_age_s,
                self.stale_depth_drop_count,
            )
            return

        time0 = time.time()
        depth_m = self.decode_depth_m(data)
        depth = self.prepare_depth_input(depth_m)
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)

        time1 = time.time()
        obs = self.process_odom().to(self.device, non_blocking=True)

        time2 = time.time()
        if self.device == "cuda":
            torch.cuda.synchronize()
        forward_start = time.time()
        candidate = self.policy.inference(depth_input, obs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        forward_end = time.time()
        brake_bundle = None
        brake_generation_latency_ms = 0.0
        if self.enable_latency_aware_risk or self.enable_intervention_selector or self.depth_emergency_stop:
            try:
                brake_bundle = self.build_constrained_brake_command()
                brake_generation_latency_ms = float(brake_bundle[-1])
            except Exception as exc:
                if not getattr(self, "warned_latency_brake_unavailable", False):
                    rospy.logwarn(f"Could not compute constrained brake command for this planning frame; using latency fallback: {exc}")
                    self.warned_latency_brake_unavailable = True
        flat = candidate.flatten()
        forward_latency_s = forward_end - forward_start
        latency_budget = self.estimate_reaction_budget(
            data,
            obs,
            forward_latency_s,
            brake_bundle=brake_bundle,
            brake_generation_latency_ms=brake_generation_latency_ms,
            sensor_age_s=sensor_age_s,
        )
        endstate = flat["end_state_b"].detach().cpu().numpy()
        utility = flat["utility_score"].detach().cpu().numpy()
        utility_base = flat.get("utility_base")
        utility_base = None if utility_base is None else utility_base.detach().cpu().numpy()
        utility_delta = flat.get("utility_delta")
        utility_delta = None if utility_delta is None else utility_delta.detach().cpu().numpy()
        traj_time = flat["traj_time"].detach().cpu().numpy()
        candidate_type = flat.get("candidate_type")
        candidate_type = None if candidate_type is None else candidate_type.detach().cpu().numpy()
        risk_logit_prob = torch.sigmoid(flat["risk_logit"]).detach().cpu().numpy()
        margin_pred_t = flat["margin_pred"]
        risk_prob_t = torch.sigmoid(flat["risk_logit"])
        risk_source = "risk_logit"
        reaction_window_mean_t = flat.get("reaction_window_mean")
        reaction_window_logvar_t = flat.get("reaction_window_logvar")
        validity_logit_t = flat.get("validity_logit")
        zero_window_logit_t = flat.get("zero_window_logit", flat.get("rm_insufficient_logit"))
        hazard_logits_t = flat.get("hazard_logits")
        hazard_risk_prob_t = None
        if (
            self.enable_latency_aware_risk
            and validity_logit_t is not None
            and zero_window_logit_t is not None
            and hazard_logits_t is not None
        ):
            hazard_risk_prob_t = two_stage_risk_probability(
                validity_logit_t,
                zero_window_logit_t,
                hazard_logits_t,
                latency_budget.tau_total_s,
                hazard_max_time_s=self.rm_critic_hazard_max_time_s,
            )
            risk_prob_t = hazard_risk_prob_t
            if reaction_window_mean_t is not None:
                margin_pred_t = reaction_window_mean_t - float(latency_budget.tau_total_s)
            risk_source = "two_stage_hazard_latency"
        elif self.enable_latency_aware_risk and reaction_window_mean_t is not None and reaction_window_logvar_t is not None:
            risk_prob_t = risk_probability_from_window(
                reaction_window_mean_t,
                reaction_window_logvar_t,
                latency_budget.tau_total_s,
            )
            margin_pred_t = reaction_window_mean_t - float(latency_budget.tau_total_s)
            risk_source = "reaction_window_latency"
        elif self.enable_latency_aware_risk and not self.warned_latency_risk_unavailable:
            rospy.logwarn(
                "Latency-aware risk was requested, but this policy did not return reaction_window_mean/logvar; "
                "falling back to risk_logit."
            )
            self.warned_latency_risk_unavailable = True
        raw_risk_prob_t = risk_prob_t.clamp(1e-6, 1.0 - 1e-6)
        validity_prob_t = None if validity_logit_t is None else torch.sigmoid(validity_logit_t)
        validity_fused_risk_t = raw_risk_prob_t
        risk_already_interaction_conditioned = hazard_risk_prob_t is not None
        if self.use_validity_risk_fusion and validity_prob_t is not None and not risk_already_interaction_conditioned:
            unknown = torch.full_like(raw_risk_prob_t, self.validity_unknown_risk)
            validity_fused_risk_t = validity_prob_t * raw_risk_prob_t + (1.0 - validity_prob_t) * unknown
            risk_source = risk_source + "+validity"
        if self.use_calibrated_risk:
            calibrated_risk_t = calibrated_probability(torch.logit(validity_fused_risk_t.clamp(1e-6, 1.0 - 1e-6)), self.risk_calibration).to(device=validity_fused_risk_t.device)
            risk_source = risk_source + "+calibrated"
        else:
            calibrated_risk_t = validity_fused_risk_t
        risk_upper_t = risk_upper_bound(calibrated_risk_t, self.risk_calibration).to(device=validity_fused_risk_t.device)
        margin_pred = margin_pred_t.detach().cpu().numpy()
        raw_risk_prob = raw_risk_prob_t.detach().cpu().numpy()
        hazard_risk_prob = None if hazard_risk_prob_t is None else hazard_risk_prob_t.detach().cpu().numpy()
        validity_fused_risk_prob = validity_fused_risk_t.detach().cpu().numpy()
        calibrated_risk_prob = calibrated_risk_t.detach().cpu().numpy()
        risk_upper = risk_upper_t.detach().cpu().numpy()
        risk_prob = risk_upper if (self.enable_intervention_selector or self.use_calibrated_risk) else validity_fused_risk_prob
        reaction_window_mean = None if reaction_window_mean_t is None else reaction_window_mean_t.detach().cpu().numpy()
        reaction_window_logvar = None if reaction_window_logvar_t is None else reaction_window_logvar_t.detach().cpu().numpy()
        validity_prob = None if validity_prob_t is None else validity_prob_t.detach().cpu().numpy()
        yield_logit = flat["yield_logit"] if "yield_logit" in flat else flat["backup_logit"]
        yield_prob = torch.sigmoid(yield_logit).detach().cpu().numpy()
        yaw_terminal = flat["yaw_terminal"].detach().cpu().numpy()
        time3 = time.time()

        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)
        depth_clearance = self.compute_candidate_depth_clearance(depth_m, endstate_w, traj_time)
        altitude_violation = self.compute_candidate_altitude_violation(endstate_w, traj_time)
        action_id, selection_score = self.select_action(
            utility,
            endstate_w,
            traj_time,
            candidate_type,
            margin_pred=margin_pred,
            risk_prob=risk_prob,
            depth_clearance=depth_clearance,
            altitude_violation=altitude_violation,
        )
        original_yopo_top1 = int(np.argmax(utility))
        intervention = None
        intervention_brake = False
        brake_candidate = None
        brake_anchor_created = False
        if self.enable_intervention_selector:
            brake_candidate = self.build_and_evaluate_brake_candidate(
                depth_m,
                brake_bundle=brake_bundle,
                brake_generation_latency_ms=brake_generation_latency_ms,
            )
            action_id, selection_score, intervention = self.apply_intervention_selector(
                utility,
                endstate_w,
                traj_time,
                candidate_type,
                risk_upper,
                depth_clearance,
                altitude_violation,
                original_yopo_top1,
                brake_candidate=brake_candidate,
            )
            intervention_brake = intervention is not None and intervention.intervention_type == "BRAKE"
            if intervention_brake and self.brake_latch.active and self.brake_latch_anchor_w is None:
                if brake_candidate is not None:
                    self.brake_latch_anchor_w = np.asarray(
                        brake_candidate["command"].end_pos, dtype=np.float32
                    ).copy()
                    self.brake_latch_anchor_time_s = time.time()
                    brake_anchor_created = True
        else:
            self.last_brake_candidate_info = None
            self.last_brake_command = None
        selected_time = float(np.clip(traj_time[action_id], 0.1, 10.0))
        depth_emergency_stop = self.should_depth_emergency_stop(depth_clearance, action_id)
        emergency_stop = depth_emergency_stop or intervention_brake
        if depth_emergency_stop and not intervention_brake and self.enable_intervention_selector:
            latch_duration_s = 0.0 if brake_candidate is None else float(brake_candidate.get("duration", 0.0))
            self.brake_latch.arm(time.time(), latch_duration_s)
        with self.lock:
            start_pos = self.get_start_pos()
            start_vel = self.get_start_vel()
            start_acc = self.desire_acc
            if emergency_stop:
                if intervention_brake and brake_candidate is not None:
                    brake_command = brake_candidate["command"]
                    start_pos = brake_candidate.get("start_pos", start_pos)
                    start_vel = brake_candidate.get("start_vel", start_vel)
                    start_acc = brake_candidate.get("start_acc", start_acc)
                else:
                    if brake_bundle is not None:
                        brake_command, start_pos, start_vel, start_acc, _brake_generation_ms = brake_bundle
                    else:
                        brake_command, start_pos, start_vel, start_acc, brake_generation_latency_ms = self.build_constrained_brake_command()
                if self.brake_latch.active and self.brake_latch_anchor_w is None:
                    self.brake_latch_anchor_w = np.asarray(brake_command.end_pos, dtype=np.float32).copy()
                    self.brake_latch_anchor_time_s = time.time()
                    brake_anchor_created = True
                if self.brake_latch.active and self.brake_latch_anchor_w is not None and not brake_anchor_created:
                    selected_time = max(float(self.depth_emergency_traj_time), 0.3)
                    end_pos = np.asarray(self.brake_latch_anchor_w, dtype=np.float32).copy()
                    end_vel = np.zeros(3, dtype=np.float32)
                    end_acc = np.zeros(3, dtype=np.float32)
                    diagnostics = brake_command.diagnostics.to_dict()
                    diagnostics.update(
                        {
                            "duration": float(selected_time),
                            "stop_distance": float(np.linalg.norm(end_pos - start_pos)),
                            "latched_anchor_hold": True,
                        }
                    )
                else:
                    selected_time = brake_command.duration
                    end_pos = brake_command.end_pos
                    end_vel = brake_command.end_vel
                    end_acc = brake_command.end_acc
                    diagnostics = brake_command.diagnostics.to_dict()
                self.last_depth_emergency_target = end_pos.astype(float).tolist()
                self.last_deterministic_brake_stop = True
                self.last_deterministic_brake_reason = (
                    self.last_depth_emergency_reason
                    or self.last_selector_force_emergency_reason
                    or (intervention.intervention_reason if intervention_brake and intervention is not None else "emergency_brake")
                )
                self.last_deterministic_brake_terminal_speed = float(np.linalg.norm(end_vel))
                self.last_deterministic_brake_terminal_acc_norm = float(np.linalg.norm(end_acc))
                self.last_deterministic_brake_diagnostics = diagnostics
            else:
                self.last_depth_emergency_target = None
                self.last_deterministic_brake_stop = False
                self.last_deterministic_brake_reason = None
                self.last_deterministic_brake_terminal_speed = None
                self.last_deterministic_brake_terminal_acc_norm = None
                self.last_deterministic_brake_diagnostics = None
                end_pos = start_pos + endstate_w[action_id, :, 0]
                end_pos[2] = float(np.clip(end_pos[2], self.min_command_z, self.max_command_z))
                end_vel = endstate_w[action_id, :, 1]
                end_acc = endstate_w[action_id, :, 2]
            self.optimal_poly_x = Poly5Solver(
                start_pos[0],
                start_vel[0],
                start_acc[0],
                end_pos[0],
                end_vel[0],
                end_acc[0],
                selected_time,
            )
            self.optimal_poly_y = Poly5Solver(
                start_pos[1],
                start_vel[1],
                start_acc[1],
                end_pos[1],
                end_vel[1],
                end_acc[1],
                selected_time,
            )
            self.optimal_poly_z = Poly5Solver(
                start_pos[2],
                start_vel[2],
                start_acc[2],
                end_pos[2],
                end_vel[2],
                end_acc[2],
                selected_time,
            )
            self.selected_traj_time = selected_time
            self.ctrl_time = 0.0

        time4 = time.time()
        self.depth_count += 1
        self.print_selection_status(action_id, utility, selection_score, endstate_w, traj_time, candidate_type)
        self.visualize_trajectory(utility, endstate_w, traj_time, action_id, candidate_type)
        time5 = time.time()
        self.write_benchmark_log(
            action_id,
            utility,
            selection_score,
            endstate_w,
            traj_time,
            candidate_type,
            margin_pred,
            risk_prob,
            yield_prob,
            yaw_terminal,
            depth_clearance,
            utility_base,
            utility_delta,
            altitude_violation,
            risk_source=risk_source,
            risk_logit_prob=risk_logit_prob,
            raw_risk_prob=raw_risk_prob,
            hazard_risk_prob=hazard_risk_prob,
            validity_fused_risk_prob=validity_fused_risk_prob,
            calibrated_risk_prob=calibrated_risk_prob,
            risk_upper_bound=risk_upper,
            intervention=intervention,
            original_yopo_top1=original_yopo_top1,
            reaction_window_mean=reaction_window_mean,
            reaction_window_logvar=reaction_window_logvar,
            validity_prob=validity_prob,
            latency_budget=latency_budget,
            inference_latency_ms=(forward_end - forward_start) * 1000.0,
            candidate_decode_ms=(time3 - forward_end) * 1000.0,
            selection_ms=(time4 - time3) * 1000.0,
            total_latency_ms=(time5 - time0) * 1000.0,
            brake_generation_latency_ms=brake_generation_latency_ms,
            total_planning_latency_ms=(time5 - time0) * 1000.0,
        )
        self.print_time(time0, time1, time2, time3, time4, time5)

    def decode_depth_m(self, data):
        if data.encoding == "32FC1":
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width).copy()
        elif data.encoding == "16UC1":
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        invalid = np.isnan(depth) | (depth < self.min_dis) | (depth > self.max_dis)
        depth = np.clip(depth, self.min_dis, self.max_dis)
        depth[invalid] = self.max_dis
        return depth.astype(np.float32)

    def prepare_depth_input(self, depth_m):
        depth = np.minimum(depth_m, self.max_dis) / self.max_dis
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        return interpolated_image.reshape([1, 1, self.height, self.width])

    def decode_depth(self, data):
        return self.prepare_depth_input(self.decode_depth_m(data))

    def control_pub(self, _timer):
        if not self.goal_received:
            return
        if self.arrive and self.yopo_preserve_mode and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return
        if self.arrive and self.hover_on_arrival:
            self.publish_arrival_hover()
            return
        if self.ctrl_time is None or self.ctrl_time > self.selected_traj_time:
            return
        if self.ctrl_pub.get_num_connections() == 0:
            if not rospy.is_shutdown() and not self.warned_no_ctrl_subscriber:
                rospy.logwarn(
                    "No subscriber on %s. Start the controller launch before expecting the drone to move.",
                    self.config["ctrl_topic"],
                )
                self.warned_no_ctrl_subscriber = True
            return
        with self.lock:
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            if self.position_control_mode:
                control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
            else:
                control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            position_z = self.optimal_poly_z.get_position(self.ctrl_time)
            if not self.yopo_preserve_mode:
                position_z = float(np.clip(position_z, self.min_command_z, self.max_command_z))
            control_msg.position.z = position_z
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])

            goal_dir = self.goal - self.desire_pos
            probe_active = bool(
                self.brake_probe_enabled
                and self.brake_latch.active
                and self.brake_probe_stationary_start_s is not None
                and self.brake_probe_yaw_center is not None
            )
            if probe_active:
                probe_elapsed_s = max(0.0, time.time() - self.brake_probe_stationary_start_s)
                probe_omega = 2.0 * np.pi / self.brake_probe_period_s
                yaw = self.brake_probe_yaw_center + self.brake_probe_yaw_amplitude_rad * np.sin(
                    probe_omega * probe_elapsed_s
                )
                yaw_dot = self.brake_probe_yaw_amplitude_rad * probe_omega * np.cos(
                    probe_omega * probe_elapsed_s
                )
            else:
                yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)
            self.control_count += 1
            if self.debug and self.control_count % 50 == 1:
                acc_norm = float(np.linalg.norm(self.desire_acc))
                vel_norm = float(np.linalg.norm(self.desire_vel))
                rospy.loginfo(
                    "OARM command flag=%d pos=(%.2f, %.2f, %.2f) vel_norm=%.2f acc_norm=%.2f",
                    control_msg.trajectory_flag,
                    control_msg.position.x,
                    control_msg.position.y,
                    control_msg.position.z,
                    vel_norm,
                    acc_norm,
                )

    def publish_arrival_hover(self):
        control_msg = PositionCommand()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
        control_msg.position.x = float(self.goal[0])
        control_msg.position.y = float(self.goal[1])
        control_msg.position.z = float(np.clip(self.goal[2], self.min_command_z, self.max_command_z))
        control_msg.velocity.x = 0.0
        control_msg.velocity.y = 0.0
        control_msg.velocity.z = 0.0
        control_msg.acceleration.x = 0.0
        control_msg.acceleration.y = 0.0
        control_msg.acceleration.z = 0.0
        control_msg.yaw = self.last_yaw
        control_msg.yaw_dot = 0.0
        self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
        self.desire_vel = np.array([0.0, 0.0, 0.0])
        self.desire_acc = np.array([0.0, 0.0, 0.0])
        self.desire_init = True
        self.last_control_msg = control_msg
        self.ctrl_pub.publish(control_msg)
        self.control_count += 1

    def compute_candidate_altitude_violation(self, endstate_w, traj_time):
        if self.selector_min_traj_z is None and self.selector_max_traj_z is None and self.altitude_band_weight <= 0.0:
            self.last_altitude_valid_count = None
            self.last_altitude_total_count = None
            self.last_candidate_min_z = None
            self.last_candidate_max_z = None
            return None

        start_pos = self.get_start_pos()
        start_vel = self.get_start_vel()
        start_acc = self.desire_acc if self.desire_acc is not None else np.zeros(3, dtype=np.float32)
        samples = max(3, int(self.altitude_band_samples))
        min_z = np.full((endstate_w.shape[0],), np.inf, dtype=np.float32)
        max_z = np.full((endstate_w.shape[0],), -np.inf, dtype=np.float32)
        violation = np.zeros((endstate_w.shape[0],), dtype=np.float32)

        for i in range(endstate_w.shape[0]):
            tf = float(np.clip(traj_time[i], 0.1, 10.0))
            t_values = np.linspace(0.0, tf, samples)
            pz = Poly5Solver(
                start_pos[2],
                start_vel[2],
                start_acc[2],
                endstate_w[i, 2, 0] + start_pos[2],
                endstate_w[i, 2, 1],
                endstate_w[i, 2, 2],
                tf,
            )
            z_values = np.asarray(pz.get_position(t_values), dtype=np.float32)
            min_z[i] = float(np.min(z_values))
            max_z[i] = float(np.max(z_values))
            if self.selector_min_traj_z is not None:
                violation[i] += max(0.0, float(self.selector_min_traj_z) - float(min_z[i]))
            if self.selector_max_traj_z is not None:
                violation[i] += max(0.0, float(max_z[i]) - float(self.selector_max_traj_z))

        altitude_valid = violation <= 1e-6
        self.last_altitude_valid_count = int(np.count_nonzero(altitude_valid))
        self.last_altitude_total_count = int(altitude_valid.size)
        self.last_candidate_min_z = min_z
        self.last_candidate_max_z = max_z
        return violation

    def compute_candidate_depth_clearance(self, depth_m, endstate_w, traj_time):
        if not (self.depth_clearance_gate or self.depth_clearance_weight > 0.0):
            self.last_depth_clearance_selected = None
            self.last_depth_clearance_min = None
            self.last_depth_clearance_valid_count = None
            self.last_depth_clearance_total_count = None
            return None
        if depth_m is None or depth_m.size == 0:
            return None

        start_pos = self.get_start_pos()
        start_vel = self.get_start_vel()
        start_acc = self.desire_acc if self.desire_acc is not None else np.zeros(3, dtype=np.float32)
        rotation_cw = self.Rotation_wc.T
        samples = max(3, int(self.depth_clearance_samples))
        radius = max(0, int(self.depth_clearance_pixel_radius))
        clearances = np.full((endstate_w.shape[0],), np.inf, dtype=np.float32)
        projection_counts = np.zeros((endstate_w.shape[0],), dtype=np.int32)

        for i in range(endstate_w.shape[0]):
            tf = float(np.clip(traj_time[i], 0.1, 10.0))
            # Skip t=0 so the current vehicle body does not self-trigger the depth gate.
            t_values = np.linspace(0.15 * tf, tf, samples)
            px = Poly5Solver(start_pos[0], start_vel[0], start_acc[0], endstate_w[i, 0, 0] + start_pos[0], endstate_w[i, 0, 1], endstate_w[i, 0, 2], tf)
            py = Poly5Solver(start_pos[1], start_vel[1], start_acc[1], endstate_w[i, 1, 0] + start_pos[1], endstate_w[i, 1, 1], endstate_w[i, 1, 2], tf)
            pz = Poly5Solver(start_pos[2], start_vel[2], start_acc[2], endstate_w[i, 2, 0] + start_pos[2], endstate_w[i, 2, 1], endstate_w[i, 2, 2], tf)
            pts_w = np.stack((px.get_position(t_values), py.get_position(t_values), pz.get_position(t_values)), axis=-1)
            pts_c = (rotation_cw @ (pts_w - start_pos[None, :]).T).T
            forward = self.depth_forward_sign * pts_c[:, self.depth_forward_index]
            valid_forward = (forward > self.min_dis) & (forward < self.max_dis)
            if not np.any(valid_forward):
                continue
            pts_c = pts_c[valid_forward]
            forward = forward[valid_forward]
            horizontal = self.depth_horizontal_sign * pts_c[:, self.depth_horizontal_index]
            vertical = self.depth_vertical_sign * pts_c[:, self.depth_vertical_index]
            u = self.camera_cx + self.camera_fx * (horizontal / forward)
            v = self.camera_cy + self.camera_fy * (vertical / forward)
            ui = np.rint(u).astype(np.int32)
            vi = np.rint(v).astype(np.int32)
            inside = (ui >= 0) & (ui < self.width) & (vi >= 0) & (vi < self.height)
            if not np.any(inside):
                continue
            ui = ui[inside]
            vi = vi[inside]
            forward = forward[inside]
            local_clearance = []
            for uu, vv, xx in zip(ui, vi, forward):
                u0 = max(0, int(uu) - radius)
                u1 = min(self.width, int(uu) + radius + 1)
                v0 = max(0, int(vv) - radius)
                v1 = min(self.height, int(vv) + radius + 1)
                observed = float(np.min(depth_m[v0:v1, u0:u1]))
                local_clearance.append(observed - float(xx))
            if local_clearance:
                projection_counts[i] = len(local_clearance)
                clearances[i] = float(np.min(local_clearance))

        finite = np.isfinite(clearances)
        self.last_depth_clearance_min = float(np.min(clearances[finite])) if np.any(finite) else None
        self.last_depth_clearance_valid_count = int(np.count_nonzero(finite & (clearances >= self.depth_clearance_min)))
        self.last_depth_clearance_total_count = int(clearances.size)
        return clearances

    def _evaluate_single_candidate_geometry(self, depth_m, endstate_w, traj_time):
        saved = {
            "last_depth_clearance_selected": self.last_depth_clearance_selected,
            "last_depth_clearance_min": self.last_depth_clearance_min,
            "last_depth_clearance_valid_count": self.last_depth_clearance_valid_count,
            "last_depth_clearance_total_count": self.last_depth_clearance_total_count,
            "last_altitude_valid_count": self.last_altitude_valid_count,
            "last_altitude_total_count": self.last_altitude_total_count,
            "last_candidate_min_z": self.last_candidate_min_z,
            "last_candidate_max_z": self.last_candidate_max_z,
        }
        try:
            depth_clearance = self.compute_candidate_depth_clearance(depth_m, endstate_w, traj_time)
            altitude_violation = self.compute_candidate_altitude_violation(endstate_w, traj_time)
        finally:
            for key, value in saved.items():
                setattr(self, key, value)
        clearance_value = None if depth_clearance is None else float(depth_clearance[0])
        altitude_value = None if altitude_violation is None else float(altitude_violation[0])
        depth_ok = True if depth_clearance is None else bool(np.isfinite(depth_clearance[0]) and depth_clearance[0] >= self.depth_clearance_min)
        altitude_ok = True if altitude_violation is None else bool(altitude_violation[0] <= 1e-6)
        return {
            "depth_clearance": clearance_value,
            "altitude_violation": altitude_value,
            "depth_admissible": depth_ok,
            "altitude_admissible": altitude_ok,
            "geometry_admissible": bool(depth_ok and altitude_ok),
        }

    def build_and_evaluate_brake_candidate(self, depth_m, brake_bundle=None, brake_generation_latency_ms=0.0):
        if brake_bundle is None:
            brake_bundle = self.build_constrained_brake_command()
            brake_generation_latency_ms = float(brake_bundle[-1])
        command, start_pos, start_vel, start_acc, bundle_generation_ms = brake_bundle
        brake_generation_latency_ms = float(brake_generation_latency_ms if brake_generation_latency_ms is not None else bundle_generation_ms)
        endstate_w = np.zeros((1, 3, 3), dtype=np.float32)
        endstate_w[0, :, 0] = command.end_pos - start_pos
        endstate_w[0, :, 1] = command.end_vel
        endstate_w[0, :, 2] = command.end_acc
        traj_time = np.array([command.duration], dtype=np.float32)
        geometry = self._evaluate_single_candidate_geometry(depth_m, endstate_w, traj_time)
        diagnostics = command.diagnostics.to_dict()
        dynamic_feasible = bool(diagnostics.get("feasible", False))
        visible_stop_margin = None
        if geometry.get("depth_clearance") is not None:
            visible_stop_margin = brake_visible_clearance_margin(
                geometry["depth_clearance"],
                self.depth_clearance_min,
            )
        visible_stop_ok = True if visible_stop_margin is None else visible_stop_margin >= 0.0
        geometry_admissible = bool(
            geometry["geometry_admissible"]
            and (visible_stop_ok or not self.brake_require_visible_stop_distance)
        )
        risk_upper = float(self.brake_verified_risk_upper) if dynamic_feasible and geometry_admissible else 1.0
        info = {
            **geometry,
            **diagnostics,
            "command": command,
            "start_pos": start_pos,
            "start_vel": start_vel,
            "start_acc": start_acc,
            "endstate_w": endstate_w,
            "traj_time": traj_time,
            "dynamic_feasible": dynamic_feasible,
            "geometry_admissible": geometry_admissible,
            "visible_stop_margin": visible_stop_margin,
            "visible_stop_distance_ok": bool(visible_stop_ok),
            "feasible": bool(dynamic_feasible and geometry_admissible),
            "risk_upper_bound": risk_upper,
            "risk_source": "geometry_dynamic_proxy",
            "brake_generation_latency_ms": brake_generation_latency_ms,
        }
        self.last_brake_candidate_info = {key: value for key, value in info.items() if key not in {"command", "start_pos", "start_vel", "start_acc", "endstate_w", "traj_time"}}
        self.last_brake_command = command
        return info

    def should_depth_emergency_stop(self, depth_clearance, action_id):
        self.last_depth_emergency_stop = False
        self.last_depth_emergency_reason = None
        if not self.depth_emergency_stop:
            return False
        if self.last_selector_force_emergency_stop:
            self.last_depth_emergency_stop = True
            self.last_depth_emergency_reason = self.last_selector_force_emergency_reason
            return True
        if depth_clearance is None:
            return False
        speed = float(np.linalg.norm(self.get_start_vel()))
        finite = np.isfinite(depth_clearance)
        if not np.any(finite):
            return False
        selected_clearance = None
        if 0 <= int(action_id) < depth_clearance.shape[0] and finite[action_id]:
            selected_clearance = float(depth_clearance[action_id])
        no_safe_candidate = bool(
            self.depth_clearance_gate
            and self.last_depth_clearance_valid_count is not None
            and self.last_depth_clearance_valid_count == 0
        )
        # Do not trigger on the worst unselected candidate. In cluttered scenes some
        # candidates are intentionally bad; emergency should only override the chosen
        # command or a true no-safe-candidate condition.
        selected_too_close = selected_clearance is not None and selected_clearance < self.depth_emergency_clearance
        critical_hazard = selected_clearance is not None and selected_clearance < self.depth_emergency_critical_clearance
        speed_hazard = speed >= self.depth_emergency_speed_threshold
        if critical_hazard or ((no_safe_candidate or selected_too_close) and speed_hazard):
            self.last_depth_emergency_stop = True
            if critical_hazard:
                self.last_depth_emergency_reason = "critical_depth_clearance"
            elif no_safe_candidate:
                self.last_depth_emergency_reason = "no_safe_depth_candidate"
            else:
                self.last_depth_emergency_reason = "selected_depth_clearance"
            return True
        return False

    def select_action(
        self,
        utility,
        endstate_w,
        traj_time,
        candidate_type=None,
        margin_pred=None,
        risk_prob=None,
        depth_clearance=None,
        altitude_violation=None,
    ):
        score = utility.copy()
        self.last_selector_force_emergency_stop = False
        self.last_selector_force_emergency_reason = None
        self.last_stop_fallback_count = None
        self.last_stop_fallback_altitude_valid_count = None
        if self.fast_sim_mode and candidate_type is not None:
            score = score - 0.25 * (candidate_type == 2) - 0.6 * (candidate_type == 3)
        if self.oarm_margin_alpha != 0.0 and margin_pred is not None:
            score = score + self.oarm_margin_alpha * margin_pred
        if self.oarm_risk_beta != 0.0 and risk_prob is not None:
            score = score - self.oarm_risk_beta * risk_prob

        selector_valid = None
        if (
            self.progress_bonus_weight > 0.0
            or self.agile_progress_weight > 0.0
            or self.agile_goal_distance_weight > 0.0
            or self.agile_lateral_penalty > 0.0
            or self.selector_min_goal_drop_rate is not None
            or self.selector_max_lateral_rate is not None
        ):
            start_pos = self.get_start_pos()
            goal_dir = self.goal - start_pos
            goal_norm = np.linalg.norm(goal_dir)
            if goal_norm > 1e-3:
                goal_dir = goal_dir / goal_norm
                endpoint_offset = endstate_w[:, :, 0]
                progress = np.dot(endpoint_offset, goal_dir)
                time_safe = np.clip(traj_time, 0.1, 10.0)
                progress_rate = progress / time_safe
                score = score + (self.progress_bonus_weight + self.agile_progress_weight) * progress_rate
                endpoint_pos = start_pos[None, :] + endpoint_offset
                endpoint_goal_distance = np.linalg.norm(endpoint_pos - self.goal[None, :], axis=1)
                goal_distance_drop = goal_norm - endpoint_goal_distance
                goal_distance_drop_rate = goal_distance_drop / time_safe
                lateral_offset = endpoint_offset - progress[:, None] * goal_dir[None, :]
                lateral_distance = np.linalg.norm(lateral_offset, axis=1)
                lateral_rate = lateral_distance / time_safe
                if self.agile_goal_distance_weight > 0.0:
                    score = score + self.agile_goal_distance_weight * goal_distance_drop_rate
                if self.agile_lateral_penalty > 0.0:
                    score = score - self.agile_lateral_penalty * lateral_rate
                selector_valid = np.ones_like(score, dtype=bool)
                if self.selector_min_goal_drop_rate is not None:
                    selector_valid &= goal_distance_drop_rate >= float(self.selector_min_goal_drop_rate)
                if self.selector_max_lateral_rate is not None:
                    selector_valid &= lateral_rate <= float(self.selector_max_lateral_rate)
                self.last_selector_total_count = int(selector_valid.size)
                self.last_selector_valid_count = int(np.count_nonzero(selector_valid))

        depth_valid = None
        if depth_clearance is not None:
            finite_clearance = np.isfinite(depth_clearance)
            clearance_violation = np.where(
                finite_clearance,
                np.maximum(0.0, self.depth_clearance_min - depth_clearance),
                0.0,
            )
            if self.depth_clearance_weight > 0.0:
                score = score - self.depth_clearance_weight * clearance_violation
            if self.depth_clearance_gate:
                depth_valid = finite_clearance & (depth_clearance >= self.depth_clearance_min)
                self.last_depth_clearance_valid_count = int(np.count_nonzero(depth_valid))
                self.last_depth_clearance_total_count = int(depth_valid.size)

        altitude_valid = None
        if altitude_violation is not None:
            if self.altitude_band_weight > 0.0:
                score = score - self.altitude_band_weight * altitude_violation
            if self.selector_min_traj_z is not None or self.selector_max_traj_z is not None:
                altitude_valid = altitude_violation <= 1e-6
                self.last_altitude_valid_count = int(np.count_nonzero(altitude_valid))
                self.last_altitude_total_count = int(altitude_valid.size)

        valid_mask = selector_valid
        safety_masks = []
        if depth_valid is not None:
            safety_masks.append(depth_valid)
        if altitude_valid is not None:
            safety_masks.append(altitude_valid)
        if safety_masks:
            safety_valid = safety_masks[0].copy()
            for mask in safety_masks[1:]:
                safety_valid &= mask
            if np.any(safety_valid):
                if valid_mask is None:
                    valid_mask = safety_valid
                else:
                    combined = valid_mask & safety_valid
                    # If goal-shaping conflicts with safety, keep safety.
                    valid_mask = combined if np.any(combined) else safety_valid
            else:
                # At least one enabled safety gate found no safe candidate. Do not let
                # another gate resurrect unsafe candidates; fall through to brake/yield.
                valid_mask = np.zeros_like(score, dtype=bool)

        if valid_mask is not None:
            if np.any(valid_mask):
                score = np.where(valid_mask, score, -1.0e9)
            elif candidate_type is not None:
                stop_mask = (candidate_type == 2) | (candidate_type == 3)
                if np.any(stop_mask):
                    fallback_mask = stop_mask.copy()
                    self.last_stop_fallback_count = int(np.count_nonzero(fallback_mask))
                    if altitude_valid is not None:
                        altitude_stop_mask = fallback_mask & altitude_valid
                        self.last_stop_fallback_altitude_valid_count = int(np.count_nonzero(altitude_stop_mask))
                        if np.any(altitude_stop_mask):
                            fallback_mask = altitude_stop_mask
                        else:
                            self.last_selector_force_emergency_stop = True
                            self.last_selector_force_emergency_reason = "no_altitude_safe_stop"
                    score = np.where(fallback_mask, score, -1.0e9)

        if self.agile_time_penalty > 0.0:
            score = score - self.agile_time_penalty * traj_time
        if self.agile_stop_penalty > 0.0 and candidate_type is not None:
            stop_mask = (candidate_type == 2) | (candidate_type == 3)
            hazard = np.zeros_like(score, dtype=bool)
            if margin_pred is not None:
                hazard = hazard | (margin_pred < self.config.get("agile_stop_margin_threshold", -0.35))
            if risk_prob is not None:
                hazard = hazard | (risk_prob > self.config.get("agile_stop_risk_threshold", 0.45))
            score = score - self.agile_stop_penalty * stop_mask * (~hazard)
        return int(np.argmax(score)), score

    def apply_intervention_selector(
        self,
        utility,
        endstate_w,
        traj_time,
        candidate_type,
        risk_upper,
        depth_clearance,
        altitude_violation,
        original_yopo_top1,
        brake_candidate=None,
    ):
        geometry_admissible = np.ones_like(utility, dtype=bool)
        if candidate_type is not None:
            geometry_admissible &= (candidate_type != 2) & (candidate_type != 3)
        if depth_clearance is not None:
            geometry_admissible &= np.isfinite(depth_clearance) & (depth_clearance >= self.depth_clearance_min)
        if altitude_violation is not None:
            geometry_admissible &= altitude_violation <= 1e-6
        top_endpoint = endstate_w[original_yopo_top1, :, 0]
        deviation = np.linalg.norm(endstate_w[:, :, 0] - top_endpoint[None, :], axis=1)
        brake_risk_upper = None if brake_candidate is None else float(brake_candidate.get("risk_upper_bound", 1.0))
        brake_feasible = bool(
            brake_candidate is not None
            and brake_candidate.get("feasible", False)
            and brake_risk_upper is not None
            and brake_risk_upper <= float(self.config.get("risk_threshold_safe", 0.20))
        )
        decision = self.intervention_selector.select(
            risk_upper_bound=risk_upper,
            yopo_cost=-utility,
            geometry_admissible=geometry_admissible,
            deviation_from_top1=deviation,
            brake_feasible=brake_feasible,
            brake_risk_upper_bound=brake_risk_upper,
            top1_index=original_yopo_top1,
        )
        release_index = decision.selected_index
        release_admissible = bool(
            release_index is not None
            and 0 <= int(release_index) < geometry_admissible.size
            and geometry_admissible[int(release_index)]
        )
        brake_duration_s = 0.0 if brake_candidate is None else float(brake_candidate.get("duration", 0.0))
        now_s = time.time()
        speed_mps = float(np.linalg.norm(self.get_start_vel()))
        if self.brake_latch.active and speed_mps <= self.brake_latch.config.release_speed_mps:
            if self.brake_probe_stationary_start_s is None:
                self.brake_probe_stationary_start_s = now_s
                try:
                    self.brake_probe_yaw_center = float(self.get_odom_state()[2])
                except Exception:
                    self.brake_probe_yaw_center = float(self.last_yaw)
        else:
            self.brake_probe_stationary_start_s = None
        release_evidence = bool(
            not self.brake_latch.config.require_release_evidence
            or (
                self.brake_probe_enabled
                and self.brake_probe_stationary_start_s is not None
                and now_s - self.brake_probe_stationary_start_s >= self.brake_probe_min_stationary_s
            )
        )
        latch_was_active = bool(self.brake_latch.active)
        decision = self.brake_latch.update(
            decision=decision,
            now_s=now_s,
            speed_mps=speed_mps,
            selected_admissible=release_admissible,
            brake_duration_s=brake_duration_s,
            brake_risk_upper_bound=brake_risk_upper,
            release_evidence=release_evidence,
        )
        if latch_was_active and not self.brake_latch.active:
            self.brake_latch_anchor_w = None
            self.brake_latch_anchor_time_s = None
            self.brake_probe_stationary_start_s = None
            self.brake_probe_yaw_center = None
        selection_score = utility.astype(np.float32).copy()
        selected = decision.selected_index
        if decision.intervention_type == "BRAKE":
            brake_idx = None
            if candidate_type is not None:
                brake_candidates = np.flatnonzero(candidate_type == 2)
                if brake_candidates.size > 0:
                    brake_idx = int(brake_candidates[np.argmax(utility[brake_candidates])])
            selected = original_yopo_top1 if brake_idx is None else brake_idx
            self.last_selector_force_emergency_stop = True
            self.last_selector_force_emergency_reason = decision.intervention_reason
        elif selected is None:
            selected = original_yopo_top1
        return int(selected), selection_score, decision

    def visualize_trajectory(self, utility, pred_endstate, traj_time, action_id, candidate_type=None):
        start_pos = self.get_start_pos()
        start_vel = self.get_start_vel()
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.linspace(0.0, self.selected_traj_time, 20)
            points_array = np.stack(
                (
                    self.optimal_poly_x.get_position(t_values),
                    self.optimal_poly_y.get_position(t_values),
                    self.optimal_poly_z.get_position(t_values),
                ),
                axis=-1,
            )
            self.publish_xyz_cloud(self.best_traj_pub, points_array)

        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            points_array = self.sample_many_polys(
                start_pos,
                start_vel,
                lattice_endstate,
                np.zeros_like(lattice_endstate),
                np.zeros_like(lattice_endstate),
                np.full(lattice_endstate.shape[0], cfg["sgm_time"]),
            )
            self.publish_xyz_cloud(self.lattice_traj_pub, points_array)

        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            points_array = self.sample_many_polys(
                start_pos,
                start_vel,
                pred_endstate[:, :, 0],
                pred_endstate[:, :, 1],
                pred_endstate[:, :, 2],
                traj_time,
                utility=utility,
                candidate_type=candidate_type,
                selected_id=action_id,
            )
            self.publish_intensity_cloud(self.all_trajs_pub, points_array)

    def sample_many_polys(
        self,
        start_pos,
        start_vel,
        end_pos_offset,
        end_vel,
        end_acc,
        traj_time,
        utility=None,
        candidate_type=None,
        selected_id=None,
    ):
        chunks = []
        for i in range(end_pos_offset.shape[0]):
            tf = float(np.clip(traj_time[i], 0.1, 10.0))
            t_values = np.linspace(0.0, tf, 20)
            px = Poly5Solver(
                start_pos[0],
                start_vel[0],
                self.desire_acc[0],
                end_pos_offset[i, 0] + start_pos[0],
                end_vel[i, 0],
                end_acc[i, 0],
                tf,
            )
            py = Poly5Solver(
                start_pos[1],
                start_vel[1],
                self.desire_acc[1],
                end_pos_offset[i, 1] + start_pos[1],
                end_vel[i, 1],
                end_acc[i, 1],
                tf,
            )
            pz = Poly5Solver(
                start_pos[2],
                start_vel[2],
                self.desire_acc[2],
                end_pos_offset[i, 2] + start_pos[2],
                end_vel[i, 2],
                end_acc[i, 2],
                tf,
            )
            pts = np.stack((px.get_position(t_values), py.get_position(t_values), pz.get_position(t_values)), axis=-1)
            if utility is not None:
                intensity = np.full((pts.shape[0], 1), utility[i], dtype=np.float32)
                if selected_id is not None and i == selected_id:
                    intensity[:] = np.nanmax(utility) + 1.0
                if candidate_type is not None:
                    intensity += 0.05 * candidate_type[i]
                pts = np.column_stack((pts, intensity))
            chunks.append(pts)
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def publish_xyz_cloud(publisher, points_array):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
        publisher.publish(point_cloud_msg)

    @staticmethod
    def publish_intensity_cloud(publisher, points_array):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]
        point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array.astype(np.float32))
        publisher.publish(point_cloud_msg)

    def reset_executed_path(self):
        self.executed_path = Path()
        self.executed_path.header.frame_id = "world"

    def publish_navigation_visuals(self, pos):
        if not self.goal_received:
            return

        now = rospy.Time.now()
        wall_now = time.time()
        if wall_now - self.last_nav_viz_time < 0.1:
            return
        self.last_nav_viz_time = wall_now

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "world"
        pose.pose = self.odom.pose.pose
        self.executed_path.header.stamp = now
        self.executed_path.poses.append(pose)
        if len(self.executed_path.poses) > self.path_max_points:
            self.executed_path.poses = self.executed_path.poses[-self.path_max_points :]
        self.executed_path_pub.publish(self.executed_path)

        self.publish_goal_marker(now)
        self.publish_goal_line(now, pos)
        self.publish_status_text(now, pos)

    def publish_goal_marker(self, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "world"
        marker.ns = "oarm_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.goal[0])
        marker.pose.position.y = float(self.goal[1])
        marker.pose.position.z = float(self.goal[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 0.8
        marker.color.r = 0.0
        marker.color.g = 0.95
        marker.color.b = 0.25
        marker.color.a = 0.9
        self.goal_marker_pub.publish(marker)

    def publish_goal_line(self, stamp, pos):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "world"
        marker.ns = "oarm_goal_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08
        marker.color.r = 0.1
        marker.color.g = 0.75
        marker.color.b = 1.0
        marker.color.a = 0.9
        marker.points = [
            Point(float(pos[0]), float(pos[1]), float(pos[2])),
            Point(float(self.goal[0]), float(self.goal[1]), float(self.goal[2])),
        ]
        self.goal_line_pub.publish(marker)

    def publish_status_text(self, stamp, pos):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "world"
        marker.ns = "oarm_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2] + 1.2)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.8
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95

        speed = float(
            np.linalg.norm(
                [
                    self.odom.twist.twist.linear.x,
                    self.odom.twist.twist.linear.y,
                    self.odom.twist.twist.linear.z,
                ]
            )
        )
        selected = "NA"
        if self.last_selected_type is not None and self.last_selected_time is not None:
            selected = f"type={self.last_selected_type} T={self.last_selected_time:.2f}s"
        distance = 0.0 if self.last_goal_distance is None else self.last_goal_distance
        marker.text = f"dist {distance:.1f} m | speed {speed:.1f} m/s | {selected}"
        self.status_text_pub.publish(marker)

    def get_start_pos(self):
        if self.plan_from_reference:
            return self.desire_pos
        return np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))

    def get_start_vel(self):
        if self.plan_from_reference:
            return self.desire_vel
        return np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))

    def get_odom_state(self):
        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        yaw = R.from_quat(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        ).as_euler("ZYX", degrees=False)[0]
        return pos, vel, float(yaw)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        self.time_interpolation += time1 - time0
        self.time_prepare += time2 - time1
        self.time_forward += time3 - time2
        self.time_process += time4 - time3
        self.time_visualize += time5 - time4
        self.count += 1

        total_time = (time5 - time0) * 1000.0
        tolerance = 1000.0 / self.depth_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {total_time:.2f} ms exceeds {tolerance:.2f} ms.")
        if self.verbose or total_time > tolerance:
            print(
                "\033[34mAverage Time Consuming:\033[0m "
                f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m"
            )

    def print_topic_status(self):
        rospy.loginfo(
            "OARM topics: odom=%s depth=%s ctrl=%s ctrl_subscribers=%d main_experiment=%s fast_sim=%s progress_bonus=%.3f",
            self.config["odom_topic"],
            self.config["depth_topic"],
            self.config["ctrl_topic"],
            self.ctrl_pub.get_num_connections(),
            str(self.main_experiment),
            str(self.fast_sim_mode),
            self.progress_bonus_weight,
        )

    def print_selection_status(self, action_id, utility, selection_score, endstate_w, traj_time, candidate_type):
        now = time.time()
        end_offset = endstate_w[action_id, :, 0]
        end_norm = float(np.linalg.norm(end_offset))
        type_text = self.candidate_type_name(candidate_type[action_id] if candidate_type is not None else None)
        self.last_selected_type = type_text
        self.last_selected_time = float(traj_time[action_id])
        self.last_selected_end_norm = end_norm
        if not self.debug and now - self.last_status_time < 1.0:
            return
        self.last_status_time = now
        rospy.loginfo(
            "OARM selected id=%d type=%s T=%.2f end_norm=%.2f utility=%.3f ctrl_subscribers=%d published=%d",
            action_id,
            type_text,
            float(traj_time[action_id]),
            end_norm,
            float(utility[action_id]),
            self.ctrl_pub.get_num_connections(),
            self.control_count,
        )
        if end_norm < 0.15:
            rospy.logwarn(
                "Selected OARM endpoint is very close to current position. "
                "This checkpoint may be choosing brake/yield candidates."
            )
        if self.debug and self.progress_bonus_weight > 0.0:
            rospy.loginfo(
                "OARM selection score=%.3f progress_bonus_weight=%.2f fast_sim=%s",
                float(selection_score[action_id]),
                self.progress_bonus_weight,
                str(self.fast_sim_mode),
            )

    @staticmethod
    def candidate_type_name(type_id):
        if type_id is None:
            return "NA"
        names = {
            0: "progress",
            1: "probe",
            2: "brake",
            3: "yield",
        }
        return names.get(int(type_id), str(int(type_id)))

    def write_benchmark_log(
        self,
        action_id,
        utility,
        selection_score,
        endstate_w,
        traj_time,
        candidate_type,
        margin_pred,
        risk_prob,
        yield_prob,
        yaw_terminal,
        depth_clearance,
        utility_base,
        utility_delta,
        altitude_violation,
        risk_source="risk_logit",
        risk_logit_prob=None,
        raw_risk_prob=None,
        hazard_risk_prob=None,
        validity_fused_risk_prob=None,
        calibrated_risk_prob=None,
        risk_upper_bound=None,
        intervention=None,
        original_yopo_top1=None,
        reaction_window_mean=None,
        reaction_window_logvar=None,
        validity_prob=None,
        latency_budget=None,
        inference_latency_ms=0.0,
        candidate_decode_ms=0.0,
        selection_ms=0.0,
        total_latency_ms=0.0,
        brake_generation_latency_ms=0.0,
        total_planning_latency_ms=0.0,
    ):
        if self.log_jsonl_file is None:
            return
        selected_type = self.candidate_type_name(candidate_type[action_id] if candidate_type is not None else None)
        selected_depth_clearance = None
        if depth_clearance is not None and np.isfinite(depth_clearance[action_id]):
            selected_depth_clearance = float(depth_clearance[action_id])
        self.last_depth_clearance_selected = selected_depth_clearance
        selected_altitude_violation = None
        selected_min_z = None
        selected_max_z = None
        if altitude_violation is not None:
            selected_altitude_violation = float(altitude_violation[action_id])
        if self.last_candidate_min_z is not None:
            selected_min_z = float(self.last_candidate_min_z[action_id])
        if self.last_candidate_max_z is not None:
            selected_max_z = float(self.last_candidate_max_z[action_id])
        speed = float(
            np.linalg.norm(
                [
                    self.odom.twist.twist.linear.x,
                    self.odom.twist.twist.linear.y,
                    self.odom.twist.twist.linear.z,
                ]
            )
        )
        end_offset = endstate_w[action_id, :, 0]
        start_pos = self.get_start_pos()
        start_vel = self.get_start_vel()
        odom_pos, odom_vel, odom_yaw = self.get_odom_state()
        start_acc = self.desire_acc
        selected_end_pos = start_pos + endstate_w[action_id, :, 0]
        selected_end_vel = endstate_w[action_id, :, 1]
        selected_end_acc = endstate_w[action_id, :, 2]
        deterministic_brake = bool(self.last_deterministic_brake_stop)
        probe_execution = bool(
            deterministic_brake
            and self.brake_probe_enabled
            and self.brake_latch.active
            and self.brake_probe_stationary_start_s is not None
        )
        executed_time = float(self.selected_traj_time) if deterministic_brake else float(traj_time[action_id])
        executed_type = "probe" if probe_execution else ("brake" if deterministic_brake else selected_type)
        if deterministic_brake and self.last_depth_emergency_target is not None:
            commanded_end_pos = np.array(self.last_depth_emergency_target, dtype=np.float32)
        else:
            commanded_end_pos = selected_end_pos.copy()
            commanded_end_pos[2] = float(np.clip(commanded_end_pos[2], self.min_command_z, self.max_command_z))
        commanded_end_vel = np.zeros(3, dtype=np.float32) if deterministic_brake else selected_end_vel
        commanded_end_acc = np.zeros(3, dtype=np.float32) if deterministic_brake else selected_end_acc
        selected_goal_distance_drop = None
        selected_goal_distance_drop_rate = None
        selected_lateral_distance = None
        selected_lateral_rate = None
        selected_progress_rate = None
        goal_vec = self.goal - start_pos
        goal_norm = float(np.linalg.norm(goal_vec))
        if goal_norm > 1e-3:
            goal_dir = goal_vec / goal_norm
            progress = float(np.dot(end_offset, goal_dir))
            selected_progress_rate = progress / max(float(traj_time[action_id]), 0.1)
            endpoint_goal_distance = float(np.linalg.norm(selected_end_pos - self.goal))
            selected_goal_distance_drop = goal_norm - endpoint_goal_distance
            selected_goal_distance_drop_rate = selected_goal_distance_drop / max(float(traj_time[action_id]), 0.1)
            lateral_offset = end_offset - progress * goal_dir
            selected_lateral_distance = float(np.linalg.norm(lateral_offset))
            selected_lateral_rate = selected_lateral_distance / max(float(traj_time[action_id]), 0.1)
        now = float(time.time())
        row = {
            "run_id": self.run_id,
            "split": self.data_split,
            "episode_id": self.episode_id,
            "goal_segment_id": int(self.goal_segment_id),
            "method": self.method,
            "scenario": self.scenario,
            "seed": self.seed,
            "checkpoint": self.checkpoint_path,
            "candidate_mode": self.config.get("candidate_mode", "typed_frontier"),
            "backbone_mode": self.config.get("backbone_mode", "yopo_original"),
            "enable_yield_candidates": bool(self.config.get("enable_yield_candidates", False)),
            "deployed_yaw_mode": self.config.get("deployed_yaw_mode", "goal"),
            "selector_experiment": bool(self.selector_experiment),
            "enable_rm_critic": bool(self.enable_rm_critic),
            "enable_latency_aware_risk": bool(self.enable_latency_aware_risk),
            "risk_source": risk_source,
            "risk_threshold_keep": float(self.config.get("risk_threshold_keep", 0.10)),
            "risk_threshold_safe": float(self.config.get("risk_threshold_safe", 0.20)),
            "risk_improvement_min": float(self.config.get("risk_improvement_min", 0.02)),
            "selector_lambda_risk": float(self.config.get("selector_lambda_risk", 1.0)),
            "use_calibrated_risk": bool(self.use_calibrated_risk),
            "use_validity_risk_fusion": bool(self.use_validity_risk_fusion),
            "reaction_budget_margin_ms": float(self.config.get("reaction_budget_margin_ms", 0.0)),
            "validity_unknown_risk": float(self.validity_unknown_risk),
            "use_depth_header_sensor_age": bool(self.use_depth_header_sensor_age),
            "max_depth_sensor_age_ms": 1000.0 * self.max_depth_sensor_age_s,
            "depth_sensor_age_ms": None if self.last_depth_sensor_age_s is None else 1000.0 * self.last_depth_sensor_age_s,
            "stale_depth_drop_count": int(self.stale_depth_drop_count),
            "calibration_temperature": float(self.risk_calibration.temperature),
            "calibration_conformal_slack": float(self.risk_calibration.conformal_slack),
            "calibration_version": self.calibration_version,
            "enable_intervention_selector": bool(self.enable_intervention_selector),
            "oarm_margin_alpha": float(self.oarm_margin_alpha),
            "oarm_risk_beta": float(self.oarm_risk_beta),
            "agile_progress_weight": float(self.agile_progress_weight),
            "agile_goal_distance_weight": float(self.agile_goal_distance_weight),
            "agile_lateral_penalty": float(self.agile_lateral_penalty),
            "depth_clearance_weight": float(self.depth_clearance_weight),
            "depth_clearance_min": float(self.depth_clearance_min),
            "depth_clearance_gate": bool(self.depth_clearance_gate),
            "depth_emergency_stop_enabled": bool(self.depth_emergency_stop),
            "depth_emergency_clearance": float(self.depth_emergency_clearance),
            "depth_emergency_critical_clearance": float(self.depth_emergency_critical_clearance),
            "depth_emergency_speed_threshold": float(self.depth_emergency_speed_threshold),
            "depth_emergency_traj_time": float(self.depth_emergency_traj_time),
            "depth_emergency_distance_scale": float(self.depth_emergency_distance_scale),
            "depth_emergency_retreat_distance": float(self.depth_emergency_retreat_distance),
            "depth_emergency_target_z": self.depth_emergency_target_z,
            "depth_emergency_z_rate": float(self.depth_emergency_z_rate),
            "brake_decel_mps2": float(self.brake_decel_mps2),
            "brake_max_time": float(self.brake_max_time),
            "brake_max_accel_mps2": float(self.brake_max_accel_mps2),
            "brake_max_jerk_mps3": float(self.brake_max_jerk_mps3),
            "brake_max_thrust_accel_mps2": float(self.brake_max_thrust_accel_mps2),
            "brake_max_tilt_deg": float(self.brake_max_tilt_deg),
            "brake_sample_count": int(self.brake_sample_count),
            "brake_verified_risk_upper": float(self.brake_verified_risk_upper),
            "brake_require_visible_stop_distance": bool(self.brake_require_visible_stop_distance),
            "brake_latch_enabled": bool(self.brake_latch.config.enabled),
            "brake_latch_active": bool(self.brake_latch.active),
            "brake_latch_remaining_s": float(self.brake_latch.remaining_s(time.time())),
            "brake_latch_safe_release_frames": int(self.brake_latch.safe_release_frames),
            "brake_latch_min_hold_s": float(self.brake_latch.config.min_hold_s),
            "brake_latch_release_speed_mps": float(self.brake_latch.config.release_speed_mps),
            "brake_latch_release_frames": int(self.brake_latch.config.release_frames),
            "brake_latch_release_risk": float(self.brake_latch.config.release_risk),
            "brake_latch_require_release_evidence": bool(
                self.brake_latch.config.require_release_evidence),
            "brake_latch_anchor_w": (
                None if self.brake_latch_anchor_w is None
                else np.asarray(self.brake_latch_anchor_w, dtype=float).tolist()),
            "brake_probe_enabled": bool(self.brake_probe_enabled),
            "brake_probe_active": bool(probe_execution),
            "brake_probe_stationary_elapsed_s": (
                None if self.brake_probe_stationary_start_s is None
                else max(0.0, now - self.brake_probe_stationary_start_s)),
            "brake_generation_latency_ms": float(brake_generation_latency_ms),
            "total_planning_latency_ms": float(total_planning_latency_ms),
            "brake_candidate_info": self.last_brake_candidate_info,
            "brake_dynamic_feasible": None if self.last_brake_candidate_info is None else bool(self.last_brake_candidate_info.get("dynamic_feasible", False)),
            "brake_geometry_admissible": None if self.last_brake_candidate_info is None else bool(self.last_brake_candidate_info.get("geometry_admissible", False)),
            "brake_risk_upper": None if self.last_brake_candidate_info is None else float(self.last_brake_candidate_info.get("risk_upper_bound", 1.0)),
            "brake_stop_distance": None if self.last_brake_candidate_info is None else float(self.last_brake_candidate_info.get("stop_distance", 0.0)),
            "brake_peak_accel": None if self.last_brake_candidate_info is None else float(self.last_brake_candidate_info.get("peak_accel", 0.0)),
            "brake_peak_jerk": None if self.last_brake_candidate_info is None else float(self.last_brake_candidate_info.get("peak_jerk", 0.0)),
            "brake_peak_tilt": None if self.last_brake_candidate_info is None else float(self.last_brake_candidate_info.get("peak_tilt_deg", 0.0)),
            "brake_visible_stop_margin": None if self.last_brake_candidate_info is None or self.last_brake_candidate_info.get("visible_stop_margin") is None else float(self.last_brake_candidate_info.get("visible_stop_margin", 0.0)),
            "brake_visible_stop_distance_ok": None if self.last_brake_candidate_info is None else bool(self.last_brake_candidate_info.get("visible_stop_distance_ok", False)),
            "depth_emergency_stop": bool(self.last_depth_emergency_stop),
            "depth_emergency_reason": self.last_depth_emergency_reason,
            "depth_emergency_target_w": self.last_depth_emergency_target,
            "deterministic_brake_stop": bool(deterministic_brake),
            "deterministic_brake_reason": self.last_deterministic_brake_reason,
            "deterministic_brake_terminal_speed": self.last_deterministic_brake_terminal_speed,
            "deterministic_brake_terminal_acc_norm": self.last_deterministic_brake_terminal_acc_norm,
            "deterministic_brake_diagnostics": self.last_deterministic_brake_diagnostics,
            "deterministic_brake_feasible": None if self.last_deterministic_brake_diagnostics is None else bool(self.last_deterministic_brake_diagnostics.get("feasible", False)),
            "deterministic_brake_duration": None if self.last_deterministic_brake_diagnostics is None else float(self.last_deterministic_brake_diagnostics.get("duration", 0.0)),
            "deterministic_brake_stop_distance": None if self.last_deterministic_brake_diagnostics is None else float(self.last_deterministic_brake_diagnostics.get("stop_distance", 0.0)),
            "deterministic_brake_peak_accel": None if self.last_deterministic_brake_diagnostics is None else float(self.last_deterministic_brake_diagnostics.get("peak_accel", 0.0)),
            "deterministic_brake_peak_jerk": None if self.last_deterministic_brake_diagnostics is None else float(self.last_deterministic_brake_diagnostics.get("peak_jerk", 0.0)),
            "deterministic_brake_peak_thrust_accel": None if self.last_deterministic_brake_diagnostics is None else float(self.last_deterministic_brake_diagnostics.get("peak_thrust_accel", 0.0)),
            "deterministic_brake_peak_tilt_deg": None if self.last_deterministic_brake_diagnostics is None else float(self.last_deterministic_brake_diagnostics.get("peak_tilt_deg", 0.0)),
            "selector_force_emergency_stop": bool(self.last_selector_force_emergency_stop),
            "selector_force_emergency_reason": self.last_selector_force_emergency_reason,
            "stop_fallback_count": self.last_stop_fallback_count,
            "stop_fallback_altitude_valid_count": self.last_stop_fallback_altitude_valid_count,
            "depth_forward_axis": self.depth_forward_axis,
            "depth_horizontal_axis": self.depth_horizontal_axis,
            "depth_vertical_axis": self.depth_vertical_axis,
            "depth_forward_sign": float(self.depth_forward_sign),
            "depth_horizontal_sign": float(self.depth_horizontal_sign),
            "depth_vertical_sign": float(self.depth_vertical_sign),
            "depth_clearance_selected": selected_depth_clearance,
            "depth_clearance_min_observed": self.last_depth_clearance_min,
            "depth_clearance_valid_count": self.last_depth_clearance_valid_count,
            "depth_clearance_total_count": self.last_depth_clearance_total_count,
            "selector_min_traj_z": self.selector_min_traj_z,
            "selector_max_traj_z": self.selector_max_traj_z,
            "altitude_band_weight": float(self.altitude_band_weight),
            "altitude_valid_count": self.last_altitude_valid_count,
            "altitude_total_count": self.last_altitude_total_count,
            "selected_altitude_violation": selected_altitude_violation,
            "selected_min_z": selected_min_z,
            "selected_max_z": selected_max_z,
            "selector_min_goal_drop_rate": self.selector_min_goal_drop_rate,
            "selector_max_lateral_rate": self.selector_max_lateral_rate,
            "selector_valid_count": self.last_selector_valid_count,
            "selector_total_count": self.last_selector_total_count,
            "agile_time_penalty": float(self.agile_time_penalty),
            "agile_stop_penalty": float(self.agile_stop_penalty),
            "timestamp": now,
            "time": now,
            "depth_count": int(self.depth_count),
            "map_id": int(self.config.get("map_id", 0)),
            "selected_id": int(action_id),
            "selected_type": executed_type,
            "candidate_type": executed_type,
            "nominal_candidate_type": selected_type,
            "selected_time": executed_time,
            "nominal_selected_time": float(traj_time[action_id]),
            "selected_utility": float(utility[action_id]),
            "selected_selection_score": float(selection_score[action_id]),
            "selected_oarm_margin_bonus": float(self.oarm_margin_alpha * margin_pred[action_id]),
            "selected_oarm_risk_penalty": float(self.oarm_risk_beta * risk_prob[action_id]),
            "selected_margin_pred": float(margin_pred[action_id]),
            "reaction_margin": float(margin_pred[action_id]),
            "selected_risk_prob": float(risk_prob[action_id]),
            "selected_raw_risk_prob": None if raw_risk_prob is None else float(raw_risk_prob[action_id]),
            "selected_hazard_risk_prob": None if hazard_risk_prob is None else float(hazard_risk_prob[action_id]),
            "selected_validity_fused_risk_prob": None if validity_fused_risk_prob is None else float(validity_fused_risk_prob[action_id]),
            "selected_calibrated_risk_prob": None if calibrated_risk_prob is None else float(calibrated_risk_prob[action_id]),
            "selected_risk_upper_bound": None if risk_upper_bound is None else float(risk_upper_bound[action_id]),
            "selected_risk_logit_prob": None if risk_logit_prob is None else float(risk_logit_prob[action_id]),
            "selected_reaction_window_pred": None if reaction_window_mean is None else float(reaction_window_mean[action_id]),
            "selected_reaction_window_logvar": None if reaction_window_logvar is None else float(reaction_window_logvar[action_id]),
            "selected_reaction_window_std": None if reaction_window_logvar is None else float(np.exp(0.5 * reaction_window_logvar[action_id])),
            "selected_validity_prob": None if validity_prob is None else float(validity_prob[action_id]),
            "selected_yield_prob": float(yield_prob[action_id]),
            "yaw0": float(self.last_yaw),
            "selected_yaw_terminal": float(yaw_terminal[action_id]),
            "start_pos_w": start_pos.astype(float).tolist(),
            "start_vel_w": start_vel.astype(float).tolist(),
            "start_acc_w": start_acc.astype(float).tolist(),
            "planner_start_source": "reference" if self.plan_from_reference else "odom",
            "odom_pos_w": odom_pos.astype(float).tolist(),
            "odom_vel_w": odom_vel.astype(float).tolist(),
            "odom_yaw": float(odom_yaw),
            "goal_w": self.goal.astype(float).tolist(),
            "selected_end_offset_w": end_offset.astype(float).tolist(),
            "selected_end_pos_w": selected_end_pos.astype(float).tolist(),
            "selected_end_vel_w": selected_end_vel.astype(float).tolist(),
            "selected_end_acc_w": selected_end_acc.astype(float).tolist(),
            "commanded_end_pos_w": commanded_end_pos.astype(float).tolist(),
            "commanded_end_vel_w": commanded_end_vel.astype(float).tolist(),
            "commanded_end_acc_w": commanded_end_acc.astype(float).tolist(),
            "selected_goal_distance_drop": selected_goal_distance_drop,
            "selected_goal_distance_drop_rate": selected_goal_distance_drop_rate,
            "selected_lateral_distance": selected_lateral_distance,
            "selected_lateral_rate": selected_lateral_rate,
            "selected_progress_rate": selected_progress_rate,
            "speed": speed,
            "inference_latency_ms": float(inference_latency_ms),
            "network_forward_ms": float(inference_latency_ms),
            "candidate_decode_ms": float(candidate_decode_ms),
            "selection_ms": float(selection_ms),
            "total_latency_ms": float(total_latency_ms),
            "planner_total_latency_ms": float(total_planning_latency_ms),
            "goal_distance": None if self.last_goal_distance is None else float(self.last_goal_distance),
            "min_goal_distance": None if self.min_goal_distance is None else float(self.min_goal_distance),
            "arrival_distance": float(self.arrival_distance),
            "arrive": bool(self.arrive),
            "first_arrival_time": self.first_arrival_time,
            "end_norm": float(np.linalg.norm(end_offset)),
            "emergency_brake": bool(selected_type in {"brake", "yield"} or deterministic_brake),
            "collision": False,
            "collision_flag": False,
            "success": bool(self.arrive),
            "success_flag": bool(self.arrive),
            "goal_received": bool(self.goal_received),
            "run_active": bool(self.goal_received),
            "online_inputs": ["depth", "state", "goal"],
            "uses_privileged_online": False,
            "mapless_online_inference": True,
            "original_yopo_top1": None if original_yopo_top1 is None else int(original_yopo_top1),
            "intervention_type": None if intervention is None else intervention.intervention_type,
            "intervention_reason": None if intervention is None else intervention.intervention_reason,
            "risk_before": None if intervention is None or intervention.risk_before is None else float(intervention.risk_before),
            "risk_after": None if intervention is None or intervention.risk_after is None else float(intervention.risk_after),
            "intervention_score": None if intervention is None or intervention.score is None else float(intervention.score),
            "intervention_metadata": None if intervention is None else intervention.metadata,
        }
        if latency_budget is not None:
            for key, value in latency_budget.to_dict().items():
                log_key = "latency_model_inference_latency_ms" if key == "inference_latency_ms" else key
                row[log_key] = float(value)
        if self.log_candidate_table:
            candidates = []
            for i in range(int(len(utility))):
                cand_type = self.candidate_type_name(candidate_type[i] if candidate_type is not None else None)
                cand_end_offset = endstate_w[i, :, 0]
                cand_end_pos = start_pos + cand_end_offset
                cand = {
                    "id": int(i),
                    "type": cand_type,
                    "time": float(traj_time[i]),
                    "utility": float(utility[i]),
                    "utility_base": None if utility_base is None else float(utility_base[i]),
                    "utility_delta": None if utility_delta is None else float(utility_delta[i]),
                    "selection_score": float(selection_score[i]),
                    "margin_pred": float(margin_pred[i]),
                    "risk_prob": float(risk_prob[i]),
                    "raw_risk_prob": None if raw_risk_prob is None else float(raw_risk_prob[i]),
                    "hazard_risk_prob": None if hazard_risk_prob is None else float(hazard_risk_prob[i]),
                    "validity_fused_risk_prob": None if validity_fused_risk_prob is None else float(validity_fused_risk_prob[i]),
                    "calibrated_risk_prob": None if calibrated_risk_prob is None else float(calibrated_risk_prob[i]),
                    "risk_upper_bound": None if risk_upper_bound is None else float(risk_upper_bound[i]),
                    "risk_logit_prob": None if risk_logit_prob is None else float(risk_logit_prob[i]),
                    "reaction_window_pred": None if reaction_window_mean is None else float(reaction_window_mean[i]),
                    "reaction_window_logvar": None if reaction_window_logvar is None else float(reaction_window_logvar[i]),
                    "validity_prob": None if validity_prob is None else float(validity_prob[i]),
                    "yield_prob": float(yield_prob[i]),
                    "yaw_terminal": float(yaw_terminal[i]),
                    "end_offset_w": cand_end_offset.astype(float).tolist(),
                    "end_pos_w": cand_end_pos.astype(float).tolist(),
                    "end_vel_w": endstate_w[i, :, 1].astype(float).tolist(),
                    "end_acc_w": endstate_w[i, :, 2].astype(float).tolist(),
                }
                if depth_clearance is not None and np.isfinite(depth_clearance[i]):
                    cand["depth_clearance"] = float(depth_clearance[i])
                if altitude_violation is not None:
                    cand["altitude_violation"] = float(altitude_violation[i])
                if self.last_candidate_min_z is not None:
                    cand["min_z"] = float(self.last_candidate_min_z[i])
                if self.last_candidate_max_z is not None:
                    cand["max_z"] = float(self.last_candidate_max_z[i])
                candidates.append(cand)
            if self.last_brake_candidate_info is not None:
                brake_info = dict(self.last_brake_candidate_info)
                brake_info.pop("command", None)
                brake_info.update(
                    {
                        "id": int(len(candidates)),
                        "type": "brake_verified",
                        "time": float(brake_info.get("duration", self.selected_traj_time)),
                        "utility": None,
                        "selection_score": None,
                        "risk_prob": float(brake_info.get("risk_upper_bound", 1.0)),
                        "risk_upper_bound": float(brake_info.get("risk_upper_bound", 1.0)),
                        "dynamic_feasible": bool(brake_info.get("dynamic_feasible", False)),
                        "geometry_admissible": bool(brake_info.get("geometry_admissible", False)),
                    }
                )
                candidates.append(brake_info)
            row["candidates"] = candidates
            row["candidate_count"] = len(candidates)
        self.log_jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
        self.log_jsonl_file.flush()

    def write_exec_log(self, pos=None):
        if self.exec_log_jsonl_file is None:
            return
        if not self.goal_received:
            return
        odom_pos, odom_vel, odom_yaw = self.get_odom_state()
        if pos is not None:
            odom_pos = pos
        now = float(time.time())
        speed = float(np.linalg.norm(odom_vel))
        row = {
            "run_id": self.run_id,
            "split": self.data_split,
            "episode_id": self.episode_id,
            "goal_segment_id": int(self.goal_segment_id),
            "method": self.method,
            "scenario": self.scenario,
            "seed": self.seed,
            "checkpoint": self.checkpoint_path,
            "candidate_mode": self.config.get("candidate_mode", "typed_frontier"),
            "backbone_mode": self.config.get("backbone_mode", "yopo_original"),
            "enable_yield_candidates": bool(self.config.get("enable_yield_candidates", False)),
            "deployed_yaw_mode": self.config.get("deployed_yaw_mode", "goal"),
            "selector_experiment": bool(self.selector_experiment),
            "timestamp": now,
            "time": now,
            "map_id": int(self.config.get("map_id", 0)),
            "odom_pos_w": odom_pos.astype(float).tolist(),
            "odom_vel_w": odom_vel.astype(float).tolist(),
            "odom_yaw": float(odom_yaw),
            "speed": speed,
            "goal_w": self.goal.astype(float).tolist(),
            "goal_distance": None if self.last_goal_distance is None else float(self.last_goal_distance),
            "min_goal_distance": None if self.min_goal_distance is None else float(self.min_goal_distance),
            "arrival_distance": float(self.arrival_distance),
            "first_arrival_time": self.first_arrival_time,
            "success": bool(self.arrive),
            "success_flag": bool(self.arrive),
            "arrive": bool(self.arrive),
            "goal_received": bool(self.goal_received),
            "run_active": bool(self.goal_received),
            "selected_type": self.last_selected_type,
            "selected_time": self.last_selected_time,
            "plan_from_reference": bool(self.plan_from_reference),
            "exec_log_source": "odom_callback",
            "online_inputs": ["odom", "goal"],
            "uses_privileged_online": False,
        }
        self.exec_log_jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
        self.exec_log_jsonl_file.flush()

    def close_log_jsonl(self):
        if self.log_jsonl_file is not None:
            self.log_jsonl_file.close()
            self.log_jsonl_file = None
        if self.exec_log_jsonl_file is not None:
            self.exec_log_jsonl_file.close()
            self.exec_log_jsonl_file = None

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs[:, 6] = cfg["goal_length"]
        _ = self.policy.inference(depth, obs)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="explicit OARM checkpoint path")
    parser.add_argument("--trial", type=int, default=0, help="OARM trial number under OARM/saved/OARM_<trial>")
    parser.add_argument("--epoch", type=int, default=50, help="checkpoint epoch number")
    parser.add_argument("--ctrl-topic", type=str, default="/so3_control/pos_cmd")
    parser.add_argument("--odom-topic", type=str, default="/sim/odom")
    parser.add_argument("--depth-topic", type=str, default="/depth_image")
    parser.add_argument("--plan-from-reference", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--position-control-mode",
        action="store_true",
        help="publish a non-READY trajectory flag so network_controller_node uses position feedback",
    )
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="use the built-in default goal immediately instead of waiting for an RViz goal",
    )
    parser.add_argument(
        "--fast-sim-mode",
        action="store_true",
        help="prefer progress candidates online for early simulation debugging",
    )
    parser.add_argument(
        "--main-experiment",
        action="store_true",
        help="reject debug-only selection bonuses so paper runs use learned utility only",
    )
    parser.add_argument(
        "--selector-experiment",
        action="store_true",
        help="allow online selector bonuses as an ablation, separate from --main-experiment",
    )
    parser.add_argument("--oarm-margin-alpha", type=float, default=0.0, help="A2 selector: add alpha * predicted reaction margin to selection score")
    parser.add_argument("--oarm-risk-beta", type=float, default=0.0, help="A2 selector: subtract beta * predicted risk probability from selection score")
    parser.add_argument("--enable-rm-critic", action="store_true", default=None, help="enable the probabilistic reaction-window critic; omitted means auto-detect from checkpoint")
    latency_risk_group = parser.add_mutually_exclusive_group()
    latency_risk_group.add_argument("--enable-latency-aware-risk", dest="enable_latency_aware_risk", action="store_true", default=None)
    latency_risk_group.add_argument("--disable-latency-aware-risk", dest="enable_latency_aware_risk", action="store_false")
    parser.add_argument("--latency-brake-accel-mps2", type=float, default=6.0)
    parser.add_argument("--latency-window", type=int, default=128)
    parser.add_argument("--latency-quantile", type=float, default=0.95)
    parser.add_argument("--rm-critic-hazard-bins", type=int, default=None)
    parser.add_argument("--rm-critic-hazard-max-time-s", type=float, default=2.5)
    parser.add_argument("--sensor-age-ms", type=float, default=0.0)
    parser.add_argument("--queue-latency-ms", type=float, default=0.0)
    parser.add_argument("--selector-latency-ms", type=float, default=0.0)
    depth_age = parser.add_mutually_exclusive_group()
    depth_age.add_argument("--use-depth-header-sensor-age", dest="use_depth_header_sensor_age", action="store_true", default=True)
    depth_age.add_argument("--disable-depth-header-sensor-age", dest="use_depth_header_sensor_age", action="store_false")
    parser.add_argument("--max-depth-sensor-age-ms", type=float, default=250.0, help="drop depth frames older than this; <=0 disables dropping")
    parser.add_argument("--control-latency-ms", type=float, default=20.0)
    parser.add_argument("--actuation-latency-ms", type=float, default=30.0)
    parser.add_argument("--reaction-budget-margin-ms", type=float, default=0.0)
    parser.add_argument("--calibration-file", type=str, default="")
    parser.add_argument("--use-calibrated-risk", action="store_true")
    parser.add_argument("--risk-conformal-slack", type=float, default=None)
    parser.add_argument("--use-validity-risk-fusion", action="store_true", default=None)
    parser.add_argument("--validity-unknown-risk", type=float, default=0.5)
    parser.add_argument("--enable-intervention-selector", action="store_true")
    parser.add_argument("--risk-threshold-keep", type=float, default=0.10)
    parser.add_argument("--risk-threshold-safe", type=float, default=0.20)
    parser.add_argument("--risk-improvement-min", type=float, default=0.02)
    parser.add_argument("--selector-lambda-risk", type=float, default=1.0)
    parser.add_argument("--progress-bonus-weight", type=float, default=0.0)
    parser.add_argument("--agile-progress-weight", type=float, default=0.0)
    parser.add_argument("--agile-goal-distance-weight", type=float, default=0.0)
    parser.add_argument("--agile-lateral-penalty", type=float, default=0.0)
    parser.add_argument("--selector-min-goal-drop-rate", type=float, default=None)
    parser.add_argument("--selector-max-lateral-rate", type=float, default=None)
    parser.add_argument("--depth-clearance-weight", type=float, default=0.0)
    parser.add_argument("--depth-clearance-min", type=float, default=0.35)
    parser.add_argument("--depth-clearance-gate", action="store_true")
    parser.add_argument("--depth-clearance-samples", type=int, default=9)
    parser.add_argument("--depth-clearance-pixel-radius", type=int, default=1)
    parser.add_argument("--depth-emergency-stop", action="store_true")
    parser.add_argument("--depth-emergency-clearance", type=float, default=0.35)
    parser.add_argument("--depth-emergency-critical-clearance", type=float, default=0.15)
    parser.add_argument("--depth-emergency-speed-threshold", type=float, default=1.0)
    parser.add_argument("--depth-emergency-traj-time", type=float, default=0.45)
    parser.add_argument("--depth-emergency-distance-scale", type=float, default=0.0)
    parser.add_argument("--depth-emergency-retreat-distance", type=float, default=0.0)
    parser.add_argument("--depth-emergency-target-z", type=float, default=None)
    parser.add_argument("--depth-emergency-z-rate", type=float, default=0.8)
    parser.add_argument("--brake-decel-mps2", type=float, default=3.0)
    parser.add_argument("--brake-max-time", type=float, default=5.0)
    parser.add_argument("--brake-max-accel-mps2", type=float, default=6.0)
    parser.add_argument("--brake-max-jerk-mps3", type=float, default=30.0)
    parser.add_argument("--brake-max-thrust-accel-mps2", type=float, default=18.0)
    parser.add_argument("--brake-max-tilt-deg", type=float, default=50.0)
    parser.add_argument("--brake-sample-count", type=int, default=81)
    parser.add_argument("--brake-verified-risk-upper", type=float, default=0.10)
    parser.add_argument("--disable-brake-visible-stop-distance-check", dest="brake_require_visible_stop_distance", action="store_false", default=True)
    parser.add_argument("--brake-time-growth", type=float, default=1.25)
    parser.add_argument("--disable-brake-latch", dest="brake_latch_enabled", action="store_false", default=True)
    parser.add_argument("--brake-latch-min-hold-s", type=float, default=0.6)
    parser.add_argument("--brake-latch-release-speed-mps", type=float, default=0.25)
    parser.add_argument("--brake-latch-release-frames", type=int, default=3)
    parser.add_argument(
        "--disable-brake-release-evidence",
        dest="brake_latch_require_release_evidence",
        action="store_false",
        default=True,
    )
    parser.add_argument("--disable-brake-probe", dest="brake_probe_enabled", action="store_false", default=True)
    parser.add_argument("--brake-probe-yaw-deg", type=float, default=30.0)
    parser.add_argument("--brake-probe-period-s", type=float, default=2.0)
    parser.add_argument("--brake-probe-min-stationary-s", type=float, default=2.0)
    parser.add_argument("--selector-min-traj-z", type=float, default=None)
    parser.add_argument("--selector-max-traj-z", type=float, default=None)
    parser.add_argument("--altitude-band-weight", type=float, default=0.0)
    parser.add_argument("--altitude-band-samples", type=int, default=13)
    parser.add_argument("--camera-hfov-deg", type=float, default=90.0)
    parser.add_argument("--camera-vfov-deg", type=float, default=60.0)
    parser.add_argument("--camera-fx", type=float, default=None)
    parser.add_argument("--camera-fy", type=float, default=None)
    parser.add_argument("--camera-cx", type=float, default=None)
    parser.add_argument("--camera-cy", type=float, default=None)
    parser.add_argument("--depth-forward-axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--depth-horizontal-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--depth-vertical-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--depth-forward-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--depth-horizontal-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--depth-vertical-sign", type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument("--agile-time-penalty", type=float, default=0.0)
    parser.add_argument("--agile-stop-penalty", type=float, default=0.0)
    parser.add_argument("--agile-stop-risk-threshold", type=float, default=0.45)
    parser.add_argument("--agile-stop-margin-threshold", type=float, default=-0.35)
    parser.add_argument("--min-command-z", type=float, default=1.0)
    parser.add_argument("--max-command-z", type=float, default=3.0)
    parser.add_argument("--arrival-distance", type=float, default=1.0)
    parser.add_argument("--no-hover-on-arrival", action="store_true")
    parser.add_argument("--goal-x", type=float, default=50.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-z", type=float, default=2.0)
    parser.add_argument("--disable-rviz-goal", action="store_true")
    parser.add_argument("--path-max-points", type=int, default=2000)
    parser.add_argument("--log-jsonl", type=str, default="", help="write per-planning-step benchmark rows")
    parser.add_argument("--exec-log-jsonl", type=str, default="", help="write high-rate odometry execution rows")
    parser.add_argument("--append-logs", action="store_true", help="append to existing JSONL logs instead of overwriting")
    parser.add_argument("--log-candidate-table", action="store_true", help="store all per-candidate scores/endpoints in benchmark JSONL for offline oracle analysis")
    parser.add_argument("--run-id", type=str, default="", help="stable id shared by planner and execution logs")
    parser.add_argument("--data-split", choices=["train", "calibration", "validation", "test", "adhoc"], default="adhoc")
    parser.add_argument("--episode-id", type=str, default="", help="episode identity used to enforce calibration/test separation")
    parser.add_argument("--method", type=str, default="oarm", help="method label for benchmark grouping")
    parser.add_argument("--scenario", type=str, default="unknown", help="scenario label written into logs")
    parser.add_argument("--seed", type=int, default=0, help="run seed or map variant id written into logs")
    parser.add_argument("--map-id", type=int, default=0, help="GT ESDF/pointcloud map id for offline annotation")
    parser.add_argument("--candidate-mode", choices=["yopo", "typed_frontier", "yopo_preserve", "yopo_preserve_rerank", "a4_preserve_brake"], default="typed_frontier")
    parser.add_argument("--backbone-mode", choices=["oarm_light", "yopo_original"], default="yopo_original")
    parser.add_argument("--enable-yield-candidates", action="store_true")
    parser.add_argument("--deployed-yaw-mode", choices=["goal", "hold", "predicted"], default="goal")
    parser.add_argument("--yopo-preserve-utility-delta-scale", type=float, default=None)
    parser.add_argument("--allow-checkpoint-mismatch", action="store_true")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = args.checkpoint or os.path.join(base_dir, "saved", f"OARM_{args.trial}", f"epoch{args.epoch}.pth")
    print("load OARM weight from:", weight)

    settings = {
        "goal": [args.goal_x, args.goal_y, args.goal_z],
        "pitch_angle_deg": 0,
        "odom_topic": args.odom_topic,
        "depth_topic": args.depth_topic,
        "ctrl_topic": args.ctrl_topic,
        "plan_from_reference": args.plan_from_reference or args.position_control_mode,
        "verbose": args.verbose,
        "visualize": not args.no_visualize,
        "debug": args.debug,
        "main_experiment": args.main_experiment,
        "selector_experiment": args.selector_experiment,
        "position_control_mode": args.position_control_mode,
        "start_immediately": args.start_immediately,
        "fast_sim_mode": args.fast_sim_mode,
        "oarm_margin_alpha": args.oarm_margin_alpha,
        "oarm_risk_beta": args.oarm_risk_beta,
        "enable_rm_critic": args.enable_rm_critic,
        "enable_latency_aware_risk": args.enable_latency_aware_risk,
        "latency_brake_accel_mps2": args.latency_brake_accel_mps2,
        "latency_window": args.latency_window,
        "latency_quantile": args.latency_quantile,
        "rm_critic_hazard_bins": args.rm_critic_hazard_bins,
        "rm_critic_hazard_max_time_s": args.rm_critic_hazard_max_time_s,
        "sensor_age_ms": args.sensor_age_ms,
        "queue_latency_ms": args.queue_latency_ms,
        "selector_latency_ms": args.selector_latency_ms,
        "use_depth_header_sensor_age": args.use_depth_header_sensor_age,
        "max_depth_sensor_age_ms": args.max_depth_sensor_age_ms,
        "control_latency_ms": args.control_latency_ms,
        "actuation_latency_ms": args.actuation_latency_ms,
        "reaction_budget_margin_ms": args.reaction_budget_margin_ms,
        "calibration_file": args.calibration_file,
        "use_calibrated_risk": args.use_calibrated_risk,
        "risk_conformal_slack": args.risk_conformal_slack,
        "use_validity_risk_fusion": args.use_validity_risk_fusion,
        "validity_unknown_risk": args.validity_unknown_risk,
        "enable_intervention_selector": args.enable_intervention_selector,
        "risk_threshold_keep": args.risk_threshold_keep,
        "risk_threshold_safe": args.risk_threshold_safe,
        "risk_improvement_min": args.risk_improvement_min,
        "selector_lambda_risk": args.selector_lambda_risk,
        "progress_bonus_weight": args.progress_bonus_weight,
        "agile_progress_weight": args.agile_progress_weight,
        "agile_goal_distance_weight": args.agile_goal_distance_weight,
        "agile_lateral_penalty": args.agile_lateral_penalty,
        "selector_min_goal_drop_rate": args.selector_min_goal_drop_rate,
        "selector_max_lateral_rate": args.selector_max_lateral_rate,
        "depth_clearance_weight": args.depth_clearance_weight,
        "depth_clearance_min": args.depth_clearance_min,
        "depth_clearance_gate": args.depth_clearance_gate,
        "depth_clearance_samples": args.depth_clearance_samples,
        "depth_clearance_pixel_radius": args.depth_clearance_pixel_radius,
        "depth_emergency_stop": args.depth_emergency_stop,
        "depth_emergency_clearance": args.depth_emergency_clearance,
        "depth_emergency_critical_clearance": args.depth_emergency_critical_clearance,
        "depth_emergency_speed_threshold": args.depth_emergency_speed_threshold,
        "depth_emergency_traj_time": args.depth_emergency_traj_time,
        "depth_emergency_distance_scale": args.depth_emergency_distance_scale,
        "depth_emergency_retreat_distance": args.depth_emergency_retreat_distance,
        "depth_emergency_target_z": args.depth_emergency_target_z,
        "depth_emergency_z_rate": args.depth_emergency_z_rate,
        "brake_decel_mps2": args.brake_decel_mps2,
        "brake_max_time": args.brake_max_time,
        "brake_max_accel_mps2": args.brake_max_accel_mps2,
        "brake_max_jerk_mps3": args.brake_max_jerk_mps3,
        "brake_max_thrust_accel_mps2": args.brake_max_thrust_accel_mps2,
        "brake_max_tilt_deg": args.brake_max_tilt_deg,
        "brake_sample_count": args.brake_sample_count,
        "brake_verified_risk_upper": args.brake_verified_risk_upper,
        "brake_require_visible_stop_distance": args.brake_require_visible_stop_distance,
        "brake_time_growth": args.brake_time_growth,
        "brake_latch_enabled": args.brake_latch_enabled,
        "brake_latch_min_hold_s": args.brake_latch_min_hold_s,
        "brake_latch_release_speed_mps": args.brake_latch_release_speed_mps,
        "brake_latch_release_frames": args.brake_latch_release_frames,
        "brake_latch_require_release_evidence": args.brake_latch_require_release_evidence,
        "brake_probe_enabled": args.brake_probe_enabled,
        "brake_probe_yaw_deg": args.brake_probe_yaw_deg,
        "brake_probe_period_s": args.brake_probe_period_s,
        "brake_probe_min_stationary_s": args.brake_probe_min_stationary_s,
        "selector_min_traj_z": args.selector_min_traj_z,
        "selector_max_traj_z": args.selector_max_traj_z,
        "altitude_band_weight": args.altitude_band_weight,
        "altitude_band_samples": args.altitude_band_samples,
        "camera_hfov_deg": args.camera_hfov_deg,
        "camera_vfov_deg": args.camera_vfov_deg,
        "camera_fx": args.camera_fx,
        "camera_fy": args.camera_fy,
        "camera_cx": args.camera_cx,
        "camera_cy": args.camera_cy,
        "depth_forward_axis": args.depth_forward_axis,
        "depth_horizontal_axis": args.depth_horizontal_axis,
        "depth_vertical_axis": args.depth_vertical_axis,
        "depth_forward_sign": args.depth_forward_sign,
        "depth_horizontal_sign": args.depth_horizontal_sign,
        "depth_vertical_sign": args.depth_vertical_sign,
        "agile_time_penalty": args.agile_time_penalty,
        "agile_stop_penalty": args.agile_stop_penalty,
        "agile_stop_risk_threshold": args.agile_stop_risk_threshold,
        "agile_stop_margin_threshold": args.agile_stop_margin_threshold,
        "arrival_distance": args.arrival_distance,
        "hover_on_arrival": not args.no_hover_on_arrival,
        "accept_rviz_goal": not args.disable_rviz_goal,
        "min_command_z": args.min_command_z,
        "max_command_z": args.max_command_z,
        "path_max_points": args.path_max_points,
        "log_jsonl": args.log_jsonl,
        "exec_log_jsonl": args.exec_log_jsonl,
        "append_logs": args.append_logs,
        "log_candidate_table": args.log_candidate_table,
        "run_id": args.run_id,
        "data_split": args.data_split,
        "episode_id": args.episode_id,
        "method": args.method,
        "scenario": args.scenario,
        "seed": args.seed,
        "checkpoint": weight,
        "map_id": args.map_id,
        "candidate_mode": args.candidate_mode,
        "backbone_mode": args.backbone_mode,
        "enable_yield_candidates": args.enable_yield_candidates,
        "yopo_preserve_utility_delta_scale": args.yopo_preserve_utility_delta_scale,
        "deployed_yaw_mode": args.deployed_yaw_mode,
        "allow_checkpoint_mismatch": args.allow_checkpoint_mismatch,
    }
    if args.position_control_mode and not args.plan_from_reference:
        print("position-control-mode enabled: using plan_from_reference=True")
    OARMNet(settings, weight)
