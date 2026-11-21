#!/usr/bin/env python3
"""Stop-sign threshold tuning (real robot).

Sets the robot's detector threshold (default 0.1, so near-miss detections are drawn
with their scores), decompresses the annotated detector feed, and opens rviz with the
"Stop Sign Detector" image panel.

    ros2 launch autonomy_repo detector_tuning.launch.py                 # threshold 0.1
    ros2 launch autonomy_repo detector_tuning.launch.py threshold:=0.3  # use a decimal

The threshold stays set on the detector until it restarts, so set the value you picked
before exploring: ros2 param set /mobilenet_detector threshold 0.3
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    threshold = LaunchConfiguration("threshold")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("threshold", default_value="0.1"),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
                ),
                launch_arguments={
                    "config": PathJoinSubstitution(
                        [
                            FindPackageShare("autonomy_repo"),
                            "rviz",
                            "default.rviz",
                        ]
                    ),
                    "use_sim_time": use_sim_time,
                }.items(),
            ),

            # decompress the robot's annotated detector feed for the "Stop Sign Detector" panel
            Node(
                package="image_transport",
                executable="republish",
                arguments=["compressed", "raw"],
                remappings=[
                    ("in/compressed", "/detector_image/compressed"),
                    ("out", "/detector_image_decompressed"),
                ],
            ),

            # set the detector threshold, retrying until the detector node is up
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-c",
                    [
                        "until ros2 param set /mobilenet_detector threshold ",
                        threshold,
                        "; do echo 'waiting for /mobilenet_detector (camera.launch.py on the robot)...';"
                        " sleep 2; done",
                    ],
                ],
                output="screen",
            ),
        ]
    )
