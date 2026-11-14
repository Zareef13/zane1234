#!/usr/bin/env python3

import rclpy
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from std_msgs.msg import Bool

class PerceptionController(BaseController):
    def __init__(self):
        super().__init__("perception_controller")
        self.declare_parameter("active", True)
        self.stop_start_time = None
        self.cooldown_until = None
        self.detector_sub = self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)
    
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

    def detector_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if not msg.data:
            return
        if not self.active:
            return
        if self.cooldown_until is not None:
            if current_time < self.cooldown_until:
                return 
        self.set_parameters([rclpy.Parameter("active", value=False)])

    def compute_control(self) -> TurtleBotControl:
        """ Compute control command at the current time
        Returns:
            TurtleBotControl: control message
        """
        control_message = TurtleBotControl()
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.active:
            control_message.omega = 0.5
            self.stop_start_time = None
        else:
            if self.stop_start_time is None:
                self.stop_start_time = current_time

            if current_time - self.stop_start_time < 5:
                control_message.omega = 0.0
            else:
                self.set_parameters([rclpy.Parameter("active", value=True)])
                self.stop_start_time = None
                self.cooldown_until = current_time + 3.0
                control_message.omega = 0.5
        return control_message


if __name__ == "__main__":
    rclpy.init()
    controller = PerceptionController()
    rclpy.spin(controller)
    rclpy.shutdown()

