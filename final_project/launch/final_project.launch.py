#!/usr/bin/env python3
"""Launch the Saving Christmas stack.

Gazebo / RViz (same pattern as the frontier exploration homework):

    ros2 launch final_project final_project.launch.py

On the real robot, when SLAM and the state publisher are already running:

    ros2 launch final_project final_project.launch.py with_sim:=false use_sim_time:=false

The supervisor reads any of these detector topics:

    /detector_label                 std_msgs/String
    /detections                     vision_msgs/Detection2DArray
    /detector/stop_sign             std_msgs/Bool
    /detector/person                std_msgs/Bool
    /detector/teddy_bear            std_msgs/Bool

"teddy bear" is treated as Rudolph.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    with_sim = LaunchConfiguration("with_sim")

    sim_nodes = [
        IncludeLaunchDescription(
            PathJoinSubstitution(
                [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
            ),
            launch_arguments={
                "config": PathJoinSubstitution(
                    [
                        FindPackageShare("final_project"),
                        "rviz",
                        "default.rviz",
                    ]
                ),
                "use_sim_time": use_sim_time,
            }.items(),
            condition=IfCondition(with_sim),
        ),
        Node(
            executable="state_publisher.py",
            package="asl_tb3_lib",
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(with_sim),
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("with_sim", default_value="true"),
            *sim_nodes,
            Node(
                executable="navigator.py",
                package="final_project",
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                executable="frontier_exploration.py",
                package="final_project",
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                executable="mission_supervisor.py",
                package="final_project",
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="image_transport",
                executable="republish",
                arguments=["compressed", "raw"],
                remappings=[
                    ("in/compressed", "/detector_image/compressed"),
                    ("out", "/detector_image_decompressed"),
                ],
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )
