import argparse
import json
import os
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R


class RosOdomLogger:
    def __init__(self, args):
        self.args = args
        self.goal = np.asarray([args.goal_x, args.goal_y, args.goal_z], dtype=np.float64)
        self.min_goal_distance = None
        self.first_arrival_time = None
        self.first_message_wall_time = None
        self.message_count = 0
        self.start_time = time.time()
        self.run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        mode = "a" if args.append else "w"
        self.file = open(args.output, mode, encoding="utf-8")
        rospy.init_node(args.node_name, anonymous=True)
        rospy.Subscriber(args.topic, Odometry, self.callback_odom, queue_size=100, tcp_nodelay=True)

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def callback_odom(self, msg):
        now = time.time()
        if self.first_message_wall_time is None:
            self.first_message_wall_time = now
        position = np.asarray(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        velocity = np.asarray(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(velocity))
        goal_distance = float(np.linalg.norm(position - self.goal))
        if self.min_goal_distance is None:
            self.min_goal_distance = goal_distance
        else:
            self.min_goal_distance = min(self.min_goal_distance, goal_distance)
        arrived = goal_distance <= self.args.success_distance
        if arrived and self.first_arrival_time is None:
            self.first_arrival_time = now
            if self.args.print_arrival:
                print("Arrive!", flush=True)

        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        try:
            yaw = float(R.from_quat(quat).as_euler("ZYX", degrees=False)[0])
        except ValueError:
            yaw = None

        row = {
            "exec_log_source": "external_ros_odom_logger",
            "topic": self.args.topic,
            "time": now,
            "timestamp": now,
            "ros_stamp": float(msg.header.stamp.to_sec()) if msg.header.stamp else None,
            "elapsed_wall_time": float(now - self.start_time),
            "message_index": int(self.message_count),
            "run_id": self.run_id,
            "goal_segment_id": int(self.args.goal_segment_id),
            "run_active": True,
            "method": self.args.method,
            "scenario": self.args.scenario,
            "seed": int(self.args.seed),
            "map_id": int(self.args.map_id),
            "goal_w": self.goal.astype(float).tolist(),
            "success_distance": float(self.args.success_distance),
            "arrival_distance": float(self.args.success_distance),
            "odom_pos_w": position.astype(float).tolist(),
            "odom_vel_w": velocity.astype(float).tolist(),
            "odom_yaw": yaw,
            "speed": speed,
            "goal_distance": goal_distance,
            "min_goal_distance": None if self.min_goal_distance is None else float(self.min_goal_distance),
            "first_arrival_time": self.first_arrival_time,
            "arrive": bool(arrived),
            "success": bool(arrived),
            "success_flag": bool(arrived),
            "uses_privileged_online": False,
            "online_inputs": ["odom", "goal"],
        }
        self.file.write(json.dumps(row, sort_keys=True) + "\n")
        if self.args.flush_every <= 1 or self.message_count % self.args.flush_every == 0:
            self.file.flush()
        self.message_count += 1

    def spin(self):
        rate = rospy.Rate(self.args.poll_hz)
        while not rospy.is_shutdown():
            if self.args.duration > 0.0 and time.time() - self.start_time >= self.args.duration:
                break
            rate.sleep()
        self.close()
        print(
            json.dumps(
                {
                    "output": self.args.output,
                    "topic": self.args.topic,
                    "rows": self.message_count,
                    "run_id": self.run_id,
                    "goal": self.goal.astype(float).tolist(),
                    "success_distance": float(self.args.success_distance),
                    "min_goal_distance": self.min_goal_distance,
                    "first_arrival_time": self.first_arrival_time,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )


def parser():
    p = argparse.ArgumentParser(description="External odometry JSONL logger for ROS planners.")
    p.add_argument("--output", required=True, help="output JSONL path")
    p.add_argument("--topic", default="/sim/odom", help="Odometry topic to subscribe")
    p.add_argument("--node-name", default="oarm_external_odom_logger")
    p.add_argument("--duration", type=float, default=60.0, help="seconds; <=0 runs until Ctrl-C")
    p.add_argument("--poll-hz", type=float, default=20.0)
    p.add_argument("--flush-every", type=int, default=10)
    p.add_argument("--append", action="store_true")
    p.add_argument("--run-id", default="")
    p.add_argument("--goal-segment-id", type=int, default=0)
    p.add_argument("--method", default="clean_yopo")
    p.add_argument("--scenario", default="unknown")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--map-id", type=int, default=0)
    p.add_argument("--goal-x", type=float, default=50.0)
    p.add_argument("--goal-y", type=float, default=0.0)
    p.add_argument("--goal-z", type=float, default=2.0)
    p.add_argument("--success-distance", type=float, default=1.0)
    p.add_argument("--print-arrival", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    logger = RosOdomLogger(args)
    try:
        logger.spin()
    finally:
        logger.close()
