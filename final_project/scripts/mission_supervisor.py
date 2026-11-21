#!/usr/bin/env python3
"""Mission supervisor for AA 174 Saving Christmas.

Listens for stop signs, pedestrians, and Rudolph (often labeled "teddy bear"),
and publishes /mission_mode for the navigator and the frontier explorer:

  explore    keep searching
  stop       freeze in place (stop sign or pedestrian)
  celebrate  Rudolph was found; spin, then stay put
"""

import rclpy
from rclpy.node import Node
from asl_tb3_msgs.msg import TurtleBotState
from std_msgs.msg import Bool, String

from mission_logic import (
    PERSON,
    REINDEER,
    STOP_SIGN,
    DetectionLatch,
    classify_label,
    MissionLogic,
)

try:
    from vision_msgs.msg import Detection2DArray
except ImportError:  # course images may not have vision_msgs installed
    Detection2DArray = None


class MissionSupervisor(Node):
    def __init__(self):
        super().__init__("mission_supervisor")

        self.declare_parameter("stop_duration", 5.0)
        self.declare_parameter("pedestrian_clear_time", 1.0)
        self.declare_parameter("stop_sign_clear_distance", 0.8)
        self.declare_parameter("confirm_hits", 2)
        self.declare_parameter("detection_timeout", 1.2)
        self.declare_parameter("min_score", 0.4)
        self.declare_parameter("label_topic", "/detector_label")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("stop_sign_topic", "/detector/stop_sign")
        self.declare_parameter("person_topic", "/detector/person")
        self.declare_parameter("reindeer_topic", "/detector/teddy_bear")

        self.logic = MissionLogic(
            stop_duration=float(self.get_parameter("stop_duration").value),
            pedestrian_clear_time=float(self.get_parameter("pedestrian_clear_time").value),
            stop_sign_clear_distance=float(self.get_parameter("stop_sign_clear_distance").value),
        )
        self.latch = DetectionLatch(
            confirm_hits=int(self.get_parameter("confirm_hits").value),
            timeout=float(self.get_parameter("detection_timeout").value),
        )
        self.min_score = float(self.get_parameter("min_score").value)

        self.x = 0.0
        self.y = 0.0

        self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.mode_pub = self.create_publisher(String, "/mission_mode", 10)

        self.create_subscription(
            String,
            self.get_parameter("label_topic").value,
            self.label_callback,
            10,
        )
        self._subscribe_bool("stop_sign_topic", STOP_SIGN)
        self._subscribe_bool("person_topic", PERSON)
        self._subscribe_bool("reindeer_topic", REINDEER)

        if Detection2DArray is not None:
            self.create_subscription(
                Detection2DArray,
                self.get_parameter("detections_topic").value,
                self.detections_callback,
                10,
            )
        else:
            self.get_logger().warn(
                "vision_msgs is not installed, so /detections is ignored. "
                "String and Bool detector topics still work."
            )

        self.create_timer(0.1, self.tick)
        self.get_logger().info("Mission supervisor ready. Searching for Rudolph.")

    def _subscribe_bool(self, param_name, kind):
        topic = self.get_parameter(param_name).value

        def callback(msg, detected_kind=kind):
            if msg.data:
                self.latch.mark(detected_kind, self._now())
            else:
                self.latch.clear(detected_kind)

        self.create_subscription(Bool, topic, callback, 10)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def state_callback(self, msg):
        self.x = msg.x
        self.y = msg.y

    def label_callback(self, msg):
        kind = classify_label(msg.data)
        if kind is not None:
            self.latch.mark(kind, self._now())

    def detections_callback(self, msg):
        now = self._now()
        for detection in msg.detections:
            for kind, score in self._hypotheses(detection):
                if kind is not None and score >= self.min_score:
                    self.latch.mark(kind, now)

    def _hypotheses(self, detection):
        results = getattr(detection, "results", None) or []
        if not results:
            yield classify_label(getattr(detection, "id", "")), 1.0
            return
        for result in results:
            hypothesis = getattr(result, "hypothesis", result)
            class_id = getattr(hypothesis, "class_id", "")
            score = getattr(hypothesis, "score", 1.0)
            yield classify_label(class_id), float(score)

    def tick(self):
        now = self._now()
        mode = self.logic.update(
            now,
            self.x,
            self.y,
            self.latch.confirmed(STOP_SIGN, now),
            self.latch.confirmed(PERSON, now),
            self.latch.confirmed(REINDEER, now),
        )
        self._announce(self.logic.event)
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)

    def _announce(self, event):
        if event == REINDEER:
            line = "FOUND RUDOLPH — mission complete"
            print(line, flush=True)
            self.get_logger().info(line)
        elif event == STOP_SIGN:
            self.get_logger().info("Stop sign. Holding for 5 seconds.")
        elif event == PERSON:
            self.get_logger().info("Pedestrian. Waiting until the crossing is clear.")
        elif event == "resume":
            self.get_logger().info("Coast is clear. Resuming exploration.")


def main():
    rclpy.init()
    node = MissionSupervisor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
