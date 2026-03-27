"""Launch file for the JetAuto Maze Solver — starts navigation + color detection."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='jetauto_maze_solver',
            executable='maze_navigator',
            name='maze_navigator',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='jetauto_maze_solver',
            executable='color_wall_counter',
            name='color_wall_counter',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'camera_hfov': 1.2,
                # Gazebo only: keep false. Set true + rqt_image_view on /jetauto/camera/color_debug to tune vision.
                'publish_debug_image': False,
            }],
        ),
    ])
