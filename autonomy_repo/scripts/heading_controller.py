#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg  import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__()
        self.declare_parameter("kp", 2,0)

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) ->  TurtleBotControl:
        wrappedError = wrap_angle( goal.theta - state.theta)
        omega = self.kp * wrappedError
        message = TurtleBotControl()
        message.omega = omega
        return message

        
if __name__ == "__main__":
    rclpy.init()
    controller = HeadingController()
    rclpy.spin(controller)
    rclpy.shutdown()

        
    
