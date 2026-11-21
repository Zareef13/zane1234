#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class StopSignMonitor(Node):
    """Stop at stop signs while navigating.

    Same stop/cooldown logic as perception_controller.py, but instead of driving
    /cmd_vel itself it publishes /pause_navigation, which the navigator already
    handles (zero velocity while paused, replan to the same goal on resume).

    perception_controller spins in place, so a 3 s cooldown is enough to turn
    away from the sign. While navigating, the robot is driving toward the sign
    and keeps seeing it, so we also wait until the sign has been out of view for
    `clear_time` seconds before stopping again.
    """

    def __init__(self):
        super().__init__("stop_sign_monitor")
        # 5 s required stop + 0.5 s for the robot to come to rest after the pause is sent
        self.declare_parameter("stop_time", 5.5)
        self.declare_parameter("cooldown", 3.0)
        self.declare_parameter("clear_time", 2.0)

        self.stop_start_time = None
        self.cooldown_until = None
        self.last_seen_time = None

        self.pause_pub = self.create_publisher(Bool, "/pause_navigation", 10)
        self.detector_sub = self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)
        self.timer = self.create_timer(0.1, self.update)

    def param(self, name: str) -> float:
        return self.get_parameter(name).value

    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def detector_callback(self, msg: Bool):
        if not msg.data:
            return
        current_time = self.now()
        new_sign = (
            self.last_seen_time is None
            or current_time - self.last_seen_time >= self.param("clear_time")
        )
        self.last_seen_time = current_time

        if self.stop_start_time is not None:
            return
        if self.cooldown_until is not None and current_time < self.cooldown_until:
            return
        if not new_sign:
            return

        self.stop_start_time = current_time
        self.get_logger().info(f"Stop sign detected: stopping for {self.param('stop_time'):.1f} s")

    def update(self):
        # Publish the desired state every tick; the navigator ignores repeats.
        if self.stop_start_time is not None:
            current_time = self.now()
            if current_time - self.stop_start_time >= self.param("stop_time"):
                self.stop_start_time = None
                self.cooldown_until = current_time + self.param("cooldown")
                self.get_logger().info("Stop complete: resuming navigation")
        self.pause_pub.publish(Bool(data=self.stop_start_time is not None))


def main():
    rclpy.init()
    node = StopSignMonitor()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
