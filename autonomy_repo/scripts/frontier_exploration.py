#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from asl_tb3_msgs.msg import TurtleBotState
from nav_msgs.msg import OccupancyGrid
from asl_tb3_lib.grids import StochOccupancyGrid2D
from std_msgs.msg import Bool
import numpy as np
from scipy.signal import convolve2d


class FrontierExploration(Node):
    def __init__(self):
        super().__init__("frontier_exploration")

        self.state = None
        self.occupancy = None

        self.state_sub = self.create_subscription(
            TurtleBotState,
            "/state",
            self.state_callback,
            10,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            10,
        )

        self.goal_pub = self.create_publisher(
            TurtleBotState,
            "/cmd_nav",
            10,
        )

        self.nav_success_sub = self.create_subscription(
            Bool,
            "/nav_success",
            self.nav_success_callback,
            10,
        )

        self.waiting_for_goal = False

        self.current_goal = None
        self.failed_goals = []

        self.exploration_timer = self.create_timer(
            1.0,
            self.publish_frontier_goal,
        )

    def state_callback(self, msg):
        self.state = msg

    def map_callback(self, msg):
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([
                msg.info.origin.position.x,
                msg.info.origin.position.y,
            ]),
            window_size=9,
            probs=msg.data,
        )

    def nav_success_callback(self, msg):
        if not self.waiting_for_goal:
            return

        if msg.data:
            self.get_logger().info("Reached the goal")
        else:
            self.get_logger().info("Navigation planning failed")

            if self.current_goal is not None:
                self.failed_goals.append(self.current_goal.copy())

        self.current_goal = None
        self.waiting_for_goal = False


    def explore(self):
        occupancy = self.occupancy
        window_size = 13
        window_area = window_size ** 2
        kernel = np.ones((window_size, window_size))

        unknown = (occupancy.probs == -1)
        occupied = (occupancy.probs >= 50)
        unoccupied = (
            (occupancy.probs >= 0)
            & (occupancy.probs < 50)
        )

        unknown_counts = convolve2d(
            unknown.astype(int), kernel, mode="same"
        )
        occupied_counts = convolve2d(
            occupied.astype(int), kernel, mode="same"
        )
        unoccupied_counts = convolve2d(
            unoccupied.astype(int), kernel, mode="same"
        )

        frontier = (
            (unknown_counts >= 0.20 * window_area)
            & (occupied_counts == 0)
            & (unoccupied_counts >= 0.30 * window_area)
            & unoccupied
        )

        rows, cols = np.where(frontier)
        grid_xy = np.column_stack((cols, rows))
        return occupancy.grid2state(grid_xy)

    def publish_frontier_goal(self):
        if self.state is None or self.occupancy is None:
            return

        if self.waiting_for_goal:
            return

        if self.goal_pub.get_subscription_count() == 0:
            return

        frontier_states = self.explore()

        if len(frontier_states) == 0:
            self.get_logger().info("No frontier candidates found")
            return

        robot_xy = np.array([self.state.x, self.state.y])
        distances = np.linalg.norm(
            frontier_states - robot_xy, axis=1
        )

        # Avoid goals too close to the robot.
        keep = distances >= 0.5
        frontier_states = frontier_states[keep]
        distances = distances[keep]

        if len(frontier_states) == 0:
            self.get_logger().info("No frontier candidates far enough away")
            return

        # Try nearby candidates first, checking navigator clearance.
        for index in np.argsort(distances):
            target = frontier_states[index]
            if any(
                np.linalg.norm(target - failed) < 0.5
                for failed in self.failed_goals
            ):
                continue

            if not self.occupancy.is_free(target):
                continue

            heading = np.arctan2(
                target[1] - robot_xy[1],
                target[0] - robot_xy[0],
            )

            goal = TurtleBotState(
                x=float(target[0]),
                y=float(target[1]),
                theta=float(heading),
            )

            self.current_goal = target.copy()
            self.waiting_for_goal = True
            self.goal_pub.publish(goal)
            self.get_logger().info(
                f"Sent frontier goal: ({goal.x:.2f}, {goal.y:.2f})"
            )
            return

        self.get_logger().info("No frontier candidates passed clearance checks")


def main():
    rclpy.init()
    node = FrontierExploration()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
            