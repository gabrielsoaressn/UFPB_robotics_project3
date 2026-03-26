"""Launch file for the JetAuto Maze Solver — starts SLAM + navigation + color detection."""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    slam_params = os.path.join(
        get_package_share_directory('jetauto_maze_solver'),
        'config', 'slam_toolbox.yaml'
    )

    return LaunchDescription([
        # SLAM em modo mapeamento — publica /map e TF map→odom
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params, {'use_sim_time': True}],
        ),
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
            parameters=[{'use_sim_time': True}],
        ),
    ])
