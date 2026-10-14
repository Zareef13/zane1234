#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class ConstantControl(Node):
    def __init__(self) -> None:
        super().__init__('constant_control')
        self.timer = self.create_timer(0.2, self.timer_callback)

        self.constcontrol_pub = self.create_publisher(Twist,"/cmd_vel",10)


        self.kill_sub = self.create_subscription(Bool, "/kill",
                                                  self.kill_callback, 10)

        



    def timer_callback(self) -> None:
        msg = Twist()
        msg.linear.x = 0.5
        msg.angular.z = 0.5
        self.constcontrol_pub.publish(msg)

    def kill_callback(self, msg) -> None:
        if msg.data:
            self.timer.cancel()
            newmsg = Twist()
            newmsg.linear.x = 0.0
            newmsg.angular.z = 0.0
            self.constcontrol_pub.publish(newmsg)
       



if __name__ == "__main__":
    rclpy.init()
    node = ConstantControl()
    rclpy.spin(node)
    rclpy.shutdown()