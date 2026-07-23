import argparse
import json
import os
import time
from threading import Lock

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from OARM.utils.yopo_compat import ensure_yopo_path

ensure_yopo_path()

ROS_IMPORT_ERROR = None
try:
    import rospy
    import std_msgs.msg
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import Image, PointCloud2
except ModuleNotFoundError as exc:
    ROS_IMPORT_ERROR = exc

from config.config import cfg
from policy.poly_solver import Poly5Solver, calculate_yaw
from policy.primitive import LatticePrimitive
from policy.state_transform import StateTransform
from policy.yopo_network import YopoNetwork

try:
    from control_msg import PositionCommand
except ModuleNotFoundError as exc:
    if ROS_IMPORT_ERROR is None:
        ROS_IMPORT_ERROR = exc


class ExactYopoAdapter:
    """OARM-side ROS wrapper for the unmodified YOPO policy path.

    This class intentionally does not use OARMNetwork. It mirrors
    YOPO/test_yopo_ros.py for state preparation, network inference,
    prediction decoding, score selection, trajectory time, and control
    publishing, while adding OARM JSONL logs and parity dumps.
    """

    def __init__(self, args):
        if ROS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROS Python modules are unavailable. Source your ROS workspace before running "
                "OARM/test_yopo_exact_adapter_ros.py."
            ) from ROS_IMPORT_ERROR

        cfg["train"] = False
        self.args = args
        self.height = cfg["image_height"]
        self.width = cfg["image_width"]
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.asarray([args.goal_x, args.goal_y, args.goal_z], dtype=np.float64)
        self.goal_received = args.start_immediately
        self.plan_from_reference = args.plan_from_reference
        self.verbose = args.verbose
        self.visualize = not args.no_visualize
        self.Rotation_bc = R.from_euler("ZYX", [0, args.pitch_angle_deg, 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = args.ctrl_dt
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.first_arrival_time = None
        self.min_goal_distance = None
        self.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
        self.depth_count = 0
        self.control_count = 0
        self.parity_dump_written = False

        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = float(self.lattice_primitive.segment_time)

        self.policy = YopoNetwork().to(self.device)
        self.load_policy(args.weight)
        self.policy.eval()
        self.warm_up()

        self.log_file = self.open_log(args.log_jsonl)
        self.exec_log_file = self.open_log(args.exec_log_jsonl)
        self.control_log_file = self.open_log(args.control_log_jsonl)

        rospy.init_node(args.node_name, anonymous=False)
        if self.log_file is not None or self.exec_log_file is not None or self.control_log_file is not None:
            rospy.on_shutdown(self.close_logs)

        self.best_traj_pub = rospy.Publisher("/yopo_exact_adapter/best_traj_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(args.ctrl_topic, PositionCommand, queue_size=1)
        self.odom_sub = rospy.Subscriber(args.odom_topic, Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(args.depth_topic, Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)

        rospy.sleep(1.0)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print(
            json.dumps(
                {
                    "node": args.node_name,
                    "weight": args.weight,
                    "device": self.device,
                    "goal": self.goal.astype(float).tolist(),
                    "traj_time": self.traj_time,
                    "selection": "argmin_yopo_score",
                    "uses_oarm_network": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        rospy.spin()

    def open_log(self, path):
        if not path:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mode = "a" if self.args.append_logs else "w"
        return open(path, mode, encoding="utf-8")

    def close_logs(self):
        for handle in (self.log_file, self.exec_log_file, self.control_log_file):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()

    def write_jsonl(self, handle, row):
        if handle is None:
            return
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()

    def load_policy(self, weight):
        if not weight:
            raise ValueError("--weight is required")
        if not os.path.isfile(weight):
            raise FileNotFoundError(f"YOPO checkpoint not found: {weight}")
        state_dict = torch.load(weight, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(state_dict)

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, _score_pred = self.policy(depth, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)

    def callback_set_goal(self, data):
        if self.args.disable_rviz_goal:
            rospy.loginfo("Ignoring RViz goal because --disable-rviz-goal is set.")
            return
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, self.args.goal_z], dtype=np.float64)
        self.goal_received = True
        self.arrive = False
        self.first_arrival_time = None
        self.min_goal_distance = None
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f}, {self.args.goal_z:.1f})")

    def get_odom_state(self):
        pos = np.asarray(
            [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z],
            dtype=np.float64,
        )
        vel = np.asarray(
            [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z],
            dtype=np.float64,
        )
        quat = [
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w,
        ]
        yaw = float(R.from_quat(quat).as_euler("ZYX", degrees=False)[0])
        return pos, vel, yaw

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.asarray(
                [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z],
                dtype=np.float64,
            )
            self.desire_vel = np.asarray(
                [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z],
                dtype=np.float64,
            )
            self.desire_acc = np.zeros(3, dtype=np.float64)
            _pos, _vel, yaw = self.get_odom_state()
            self.last_yaw = yaw
        self.odom_init = True

        pos, vel, yaw = self.get_odom_state()
        goal_distance = float(np.linalg.norm(pos - self.goal))
        self.min_goal_distance = goal_distance if self.min_goal_distance is None else min(self.min_goal_distance, goal_distance)
        if self.goal_received and goal_distance < self.args.arrival_distance and not self.arrive:
            print("Arrive!")
            self.arrive = True
            self.first_arrival_time = float(time.time())
        self.write_exec_log(pos, vel, yaw, goal_distance)

    def write_exec_log(self, pos, vel, yaw, goal_distance):
        row = {
            "exec_log_source": "oarm_exact_yopo_adapter_odom",
            "time": float(time.time()),
            "timestamp": float(time.time()),
            "run_id": self.run_id,
            "goal_segment_id": 0,
            "run_active": True,
            "method": self.args.method,
            "scenario": self.args.scenario,
            "seed": int(self.args.seed),
            "map_id": int(self.args.map_id),
            "goal_w": self.goal.astype(float).tolist(),
            "success_distance": float(self.args.success_distance),
            "arrival_distance": float(self.args.arrival_distance),
            "odom_pos_w": pos.astype(float).tolist(),
            "odom_vel_w": vel.astype(float).tolist(),
            "odom_yaw": float(yaw),
            "speed": float(np.linalg.norm(vel)),
            "goal_distance": float(goal_distance),
            "min_goal_distance": None if self.min_goal_distance is None else float(self.min_goal_distance),
            "arrive": bool(self.arrive),
            "success": bool(goal_distance <= self.args.success_distance),
            "uses_privileged_online": False,
            "online_inputs": ["depth_image", "odom", "goal"],
        }
        self.write_jsonl(self.exec_log_file, row)

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
            else np.asarray(
                [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z],
                dtype=np.float64,
            )
        )
        vel_c = np.dot(rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(rotation_cw, acc_w)
        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(rotation_cw, goal_w)
        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_raw = torch.from_numpy(obs[None, :])
        obs_norm = self.state_transform.normalize_obs(obs_raw.clone())
        return obs_raw, obs_norm

    def decode_depth(self, data):
        if data.encoding == "32FC1":
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth, self.max_dis) / self.max_dis
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        depth = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS).astype(np.float32) / 255.0
        return depth.reshape(1, 1, self.height, self.width)

    def process_output(self, endstate_pred, score_pred):
        raw_endstate = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        raw_score = score_pred.reshape(self.lattice_primitive.traj_num)
        selected_id = int(np.argmin(raw_score))
        selected_lattice_id = int(self.lattice_primitive.traj_num - 1 - selected_id)
        selected_endstate_b = self.state_transform.pred_to_endstate_cpu(
            raw_endstate[selected_id : selected_id + 1, :], selected_lattice_id
        )[0]
        all_lattice_ids = torch.arange(self.lattice_primitive.traj_num - 1, -1, -1)
        all_endstate_b = self.state_transform.pred_to_endstate_cpu(raw_endstate, all_lattice_ids)
        all_endstate_c = all_endstate_b.reshape(-1, 3, 3).transpose(0, 2, 1)
        all_endstate_w = np.matmul(self.Rotation_wc, all_endstate_c)
        selected_endstate_w = all_endstate_w[selected_id]
        return {
            "raw_endstate": raw_endstate,
            "raw_score": raw_score,
            "selected_id": selected_id,
            "selected_lattice_id": selected_lattice_id,
            "selected_score": float(raw_score[selected_id]),
            "selected_endstate_b": selected_endstate_b,
            "all_endstate_b": all_endstate_b,
            "all_endstate_w": all_endstate_w,
            "selected_endstate_w": selected_endstate_w,
        }

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init or not self.goal_received:
            return
        if self.arrive:
            return

        time0 = time.time()
        depth = self.decode_depth(data)
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        time1 = time.time()
        obs_raw, obs_norm = self.process_odom()
        obs_norm_device = obs_norm.to(self.device, non_blocking=True)
        obs_input_device = self.state_transform.prepare_input(obs_norm_device)
        time2 = time.time()
        if self.device == "cuda":
            torch.cuda.synchronize()
        forward_start = time.time()
        endstate_pred, score_pred = self.policy(depth_input, obs_input_device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        forward_end = time.time()
        decoded = self.process_output(endstate_pred.cpu().numpy(), score_pred.cpu().numpy())
        time3 = time.time()

        with self.lock:
            start_pos = self.desire_pos if self.plan_from_reference else self.get_odom_state()[0]
            start_vel = self.desire_vel if self.plan_from_reference else self.get_odom_state()[1]
            start_acc = self.desire_acc
            end = decoded["selected_endstate_w"]
            self.optimal_poly_x = Poly5Solver(
                start_pos[0], start_vel[0], start_acc[0], end[0, 0] + start_pos[0], end[0, 1], end[0, 2], self.traj_time
            )
            self.optimal_poly_y = Poly5Solver(
                start_pos[1], start_vel[1], start_acc[1], end[1, 0] + start_pos[1], end[1, 1], end[1, 2], self.traj_time
            )
            self.optimal_poly_z = Poly5Solver(
                start_pos[2], start_vel[2], start_acc[2], end[2, 0] + start_pos[2], end[2, 1], end[2, 2], self.traj_time
            )
            self.ctrl_time = 0.0

        time4 = time.time()
        self.depth_count += 1
        self.write_planner_log(decoded, forward_end - forward_start, time4 - time0)
        self.maybe_dump_parity(
            depth,
            obs_raw,
            obs_norm,
            obs_input_device.detach().cpu(),
            decoded,
            start_pos,
            start_vel,
            start_acc,
        )
        self.visualize_best_trajectory()
        time5 = time.time()
        if self.verbose:
            print(
                "exact-yopo frame={} id={} lattice={} score={:.6f} T={:.3f} total_ms={:.2f}".format(
                    self.depth_count,
                    decoded["selected_id"],
                    decoded["selected_lattice_id"],
                    decoded["selected_score"],
                    self.traj_time,
                    (time5 - time0) * 1000.0,
                ),
                flush=True,
            )

    def write_planner_log(self, decoded, forward_seconds, total_seconds):
        pos, vel, yaw = self.get_odom_state()
        goal_distance = float(np.linalg.norm(pos - self.goal))
        selected_end_w = decoded["selected_endstate_w"]
        row = {
            "time": float(time.time()),
            "timestamp": float(time.time()),
            "run_id": self.run_id,
            "goal_segment_id": 0,
            "run_active": True,
            "method": self.args.method,
            "scenario": self.args.scenario,
            "seed": int(self.args.seed),
            "map_id": int(self.args.map_id),
            "checkpoint": self.args.weight,
            "adapter_type": "exact_yopo",
            "uses_oarm_network": False,
            "uses_oarm_head": False,
            "candidate_mode": "official_yopo",
            "selection_rule": "argmin_yopo_score",
            "trajectory_time": float(self.traj_time),
            "selected_traj_time": float(self.traj_time),
            "selected_id": int(decoded["selected_id"]),
            "selected_lattice_id": int(decoded["selected_lattice_id"]),
            "selected_score": float(decoded["selected_score"]),
            "utility_score": float(-decoded["selected_score"]),
            "reaction_margin": 0.0,
            "first_visible_time": 0.0,
            "arrival_time_to_risk": 0.0,
            "candidate_type": "yopo",
            "speed": float(np.linalg.norm(vel)),
            "inference_latency_ms": float(forward_seconds * 1000.0),
            "total_latency_ms": float(total_seconds * 1000.0),
            "emergency_brake": False,
            "position_w": pos.astype(float).tolist(),
            "odom_pos_w": pos.astype(float).tolist(),
            "odom_vel_w": vel.astype(float).tolist(),
            "odom_yaw": float(yaw),
            "goal_w": self.goal.astype(float).tolist(),
            "goal_distance": goal_distance,
            "selected_endstate_w": selected_end_w.astype(float).tolist(),
            "selected_endpoint_w": (pos + selected_end_w[:, 0]).astype(float).tolist(),
            "selected_velocity_w": selected_end_w[:, 1].astype(float).tolist(),
            "selected_acceleration_w": selected_end_w[:, 2].astype(float).tolist(),
            "success_distance": float(self.args.success_distance),
            "arrival_distance": float(self.args.arrival_distance),
        }
        self.write_jsonl(self.log_file, row)

    def parity_samples(self, start_pos, start_vel, start_acc):
        times = np.arange(self.ctrl_dt, self.traj_time + 1e-9, self.ctrl_dt, dtype=np.float64)
        pos = np.stack(
            [
                self.optimal_poly_x.get_position(times),
                self.optimal_poly_y.get_position(times),
                self.optimal_poly_z.get_position(times),
            ],
            axis=-1,
        )
        vel = np.stack(
            [
                self.optimal_poly_x.get_velocity(times),
                self.optimal_poly_y.get_velocity(times),
                self.optimal_poly_z.get_velocity(times),
            ],
            axis=-1,
        )
        acc = np.stack(
            [
                self.optimal_poly_x.get_acceleration(times),
                self.optimal_poly_y.get_acceleration(times),
                self.optimal_poly_z.get_acceleration(times),
            ],
            axis=-1,
        )
        yaws = []
        yaw_dots = []
        yaw = float(self.last_yaw)
        for p, v in zip(pos, vel):
            yaw, yaw_dot = calculate_yaw(v, self.goal - p, yaw, self.ctrl_dt)
            yaws.append(yaw)
            yaw_dots.append(yaw_dot)
        return {
            "control_t": times,
            "control_pos_w": pos,
            "control_vel_w": vel,
            "control_acc_w": acc,
            "control_yaw": np.asarray(yaws, dtype=np.float64),
            "control_yaw_dot": np.asarray(yaw_dots, dtype=np.float64),
            "control_flag": np.full(times.shape, int(PositionCommand.TRAJECTORY_STATUS_READY), dtype=np.int32),
            "start_pos_w": np.asarray(start_pos, dtype=np.float64),
            "start_vel_w": np.asarray(start_vel, dtype=np.float64),
            "start_acc_w": np.asarray(start_acc, dtype=np.float64),
        }

    def maybe_dump_parity(self, depth, obs_raw, obs_norm, obs_input, decoded, start_pos, start_vel, start_acc):
        path = self.args.dump_parity_npz
        if not path or self.parity_dump_written:
            return
        if self.args.dump_parity_frame >= 0 and self.depth_count != self.args.dump_parity_frame:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        samples = self.parity_samples(start_pos, start_vel, start_acc)
        np.savez_compressed(
            path,
            depth_input=depth.astype(np.float32),
            obs_raw=obs_raw.cpu().numpy().astype(np.float32),
            normalized_obs=obs_norm.cpu().numpy().astype(np.float32),
            primitive_obs=obs_input.cpu().numpy().astype(np.float32),
            raw_endstate=decoded["raw_endstate"].astype(np.float32),
            raw_score=decoded["raw_score"].astype(np.float32),
            selected_id=np.asarray(decoded["selected_id"], dtype=np.int64),
            selected_lattice_id=np.asarray(decoded["selected_lattice_id"], dtype=np.int64),
            selected_score=np.asarray(decoded["selected_score"], dtype=np.float32),
            decoded_endstate_b_all=decoded["all_endstate_b"].astype(np.float32),
            decoded_endstate_w_all=decoded["all_endstate_w"].astype(np.float32),
            selected_endstate_b=decoded["selected_endstate_b"].astype(np.float32),
            selected_endstate_w=decoded["selected_endstate_w"].astype(np.float32),
            trajectory_time=np.asarray(self.traj_time, dtype=np.float32),
            goal_w=self.goal.astype(np.float32),
            rotation_wc=self.Rotation_wc.astype(np.float32),
            **samples,
        )
        self.parity_dump_written = True
        print(f"Wrote parity dump: {path}", flush=True)

    def visualize_best_trajectory(self):
        if self.best_traj_pub.get_num_connections() <= 0 or self.optimal_poly_x is None:
            return
        dt = self.traj_time / 20.0
        t_values = np.arange(0, self.traj_time, dt)
        points_array = np.stack(
            [
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values),
            ],
            axis=-1,
        )
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        self.best_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, points_array))

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return
        with self.lock:
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.asarray([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.asarray([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.asarray([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)
            self.write_control_log(control_msg)

    def write_control_log(self, msg):
        row = {
            "time": float(time.time()),
            "run_id": self.run_id,
            "goal_segment_id": 0,
            "method": self.args.method,
            "scenario": self.args.scenario,
            "seed": int(self.args.seed),
            "map_id": int(self.args.map_id),
            "control_index": int(self.control_count),
            "ctrl_time": float(self.ctrl_time),
            "trajectory_time": float(self.traj_time),
            "trajectory_flag": int(msg.trajectory_flag),
            "position_w": [float(msg.position.x), float(msg.position.y), float(msg.position.z)],
            "velocity_w": [float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z)],
            "acceleration_w": [float(msg.acceleration.x), float(msg.acceleration.y), float(msg.acceleration.z)],
            "yaw": float(msg.yaw),
            "yaw_dot": float(msg.yaw_dot),
        }
        self.control_count += 1
        self.write_jsonl(self.control_log_file, row)


def default_weight(args):
    if args.weight:
        return args.weight
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "YOPO", "saved", f"YOPO_{args.trial}", f"epoch{args.epoch}.pth")


def parser():
    p = argparse.ArgumentParser(description="Exact YOPO adapter with OARM logging, without modifying YOPO baseline.")
    p.add_argument("--weight", default="", help="official YOPO checkpoint path; defaults to YOPO/saved/YOPO_<trial>/epoch<epoch>.pth")
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--node-name", default="yopo_exact_adapter")
    p.add_argument("--ctrl-topic", default="/so3_control/pos_cmd")
    p.add_argument("--odom-topic", default="/sim/odom")
    p.add_argument("--depth-topic", default="/depth_image")
    p.add_argument("--pitch-angle-deg", type=float, default=0.0)
    p.add_argument("--plan-from-reference", action="store_true")
    p.add_argument("--ctrl-dt", type=float, default=0.02)
    p.add_argument("--goal-x", type=float, default=50.0)
    p.add_argument("--goal-y", type=float, default=0.0)
    p.add_argument("--goal-z", type=float, default=2.0)
    p.add_argument("--arrival-distance", type=float, default=5.0, help="YOPO internal arrival latch distance; clean YOPO uses 5m")
    p.add_argument("--success-distance", type=float, default=1.0, help="logged success threshold; postprocess can choose another threshold")
    p.add_argument("--start-immediately", action="store_true")
    p.add_argument("--disable-rviz-goal", action="store_true")
    p.add_argument("--no-visualize", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-jsonl", default="")
    p.add_argument("--exec-log-jsonl", default="")
    p.add_argument("--control-log-jsonl", default="")
    p.add_argument("--append-logs", action="store_true")
    p.add_argument("--run-id", default="")
    p.add_argument("--method", default="yopo_exact_adapter")
    p.add_argument("--scenario", default="unknown")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--map-id", type=int, default=0)
    p.add_argument("--dump-parity-npz", default="")
    p.add_argument("--dump-parity-frame", type=int, default=1, help="1-based depth frame to dump; -1 dumps first processed frame")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.weight = default_weight(args)
    ExactYopoAdapter(args)
