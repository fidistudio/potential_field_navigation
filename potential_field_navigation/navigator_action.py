import math
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from velocity_profiler_interfaces.action import Move2D
from potential_field_interfaces.action import NavigateToPose

from .potential_fields import compute_next_position


# -----------------------------
# Configuration constants
# -----------------------------

SCAN_QUEUE_SIZE = 10
ODOM_QUEUE_SIZE = 10

OBSTACLE_MIN_RANGE = 0.05
OBSTACLE_MAX_RANGE = 1.5

CONTROL_RATE_HZ = 2.0
GOAL_TOLERANCE = 0.25


class PotentialFieldNavigator(Node):
    """
    ROS 2 node that navigates toward a target pose using potential fields.
    """

    def __init__(self) -> None:
        super().__init__("potential_field_navigator")

        self._scan: Optional[LaserScan] = None
        self._position = np.zeros(2)

        self._initialize_subscribers()
        self._initialize_actions()

    # -----------------------------
    # Initialization
    # -----------------------------

    def _initialize_subscribers(self) -> None:
        self.create_subscription(
            LaserScan,
            "/scan",
            self._on_scan_received,
            SCAN_QUEUE_SIZE,
        )

        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odometry_received,
            ODOM_QUEUE_SIZE,
        )

    def _initialize_actions(self) -> None:
        self._move_client = ActionClient(self, Move2D, "move_2d")

        self._navigation_server = ActionServer(
            self,
            NavigateToPose,
            "navigate_to_pose",
            self._execute_navigation,
        )

    # -----------------------------
    # Callbacks
    # -----------------------------

    def _on_scan_received(self, scan: LaserScan) -> None:
        self._scan = scan

    def _on_odometry_received(self, odometry: Odometry) -> None:
        position = odometry.pose.pose.position
        self._position = np.array([position.x, position.y])

    # -----------------------------
    # Sensor processing
    # -----------------------------

    def _extract_obstacles(self) -> List[np.ndarray]:
        """
        Converts laser scan data into obstacle positions in world coordinates.
        """
        if self._scan is None:
            return []

        obstacles: List[np.ndarray] = []
        angle = self._scan.angle_min

        for distance in self._scan.ranges:
            if OBSTACLE_MIN_RANGE < distance < OBSTACLE_MAX_RANGE:
                offset_x = distance * math.cos(angle)
                offset_y = distance * math.sin(angle)
                obstacle = self._position + np.array([offset_x, offset_y])
                obstacles.append(obstacle)

            angle += self._scan.angle_increment

        return obstacles

    # -----------------------------
    # Action execution
    # -----------------------------

    def _execute_navigation(
        self, goal_handle: NavigateToPose.Goal
    ) -> NavigateToPose.Result:
        goal_pose = goal_handle.request.goal.pose.position
        goal_position = np.array([goal_pose.x, goal_pose.y])

        rate = self.create_rate(CONTROL_RATE_HZ)
        self._move_client.wait_for_server()

        while rclpy.ok():
            distance_to_goal = np.linalg.norm(goal_position - self._position)

            if distance_to_goal < GOAL_TOLERANCE:
                goal_handle.succeed()
                return NavigateToPose.Result(
                    success=True,
                    message="Goal reached",
                )

            obstacles = self._extract_obstacles()
            next_position = compute_next_position(
                self._position,
                goal_position,
                obstacles,
            )

            self._send_motion_command(next_position)
            self._publish_feedback(goal_handle, distance_to_goal)

            rate.sleep()

        return NavigateToPose.Result(
            success=False,
            message="Navigation aborted",
        )

    # -----------------------------
    # Action helpers
    # -----------------------------

    def _send_motion_command(self, target_position: np.ndarray) -> None:
        goal = Move2D.Goal()
        goal.x = float(target_position[0])
        goal.y = float(target_position[1])

        future = self._move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

    def _publish_feedback(
        self,
        goal_handle: NavigateToPose.Goal,
        distance_to_goal: float,
    ) -> None:
        feedback = NavigateToPose.Feedback()
        feedback.distance_to_goal = float(distance_to_goal)
        feedback.state = "MOVING"
        goal_handle.publish_feedback(feedback)


def main() -> None:
    rclpy.init()
    node = PotentialFieldNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
