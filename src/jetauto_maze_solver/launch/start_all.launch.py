"""
Single launch file that starts everything:
  1. Robot description (URDF)
  2. Gazebo simulation with labyrinth world
  3. EKF (odometry filtering)
  4. Maze navigator + Color wall counter
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    robotics_subject_dir = get_package_share_directory('robotics_subject')
    maze_solver_dir = get_package_share_directory('jetauto_maze_solver')

    # 1. Robot description (joint/robot state publishers)
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robotics_subject_dir, 'launch', 'robot_description.launch.py')
        ),
    )

    # 2. Gazebo simulation + spawn robot
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robotics_subject_dir, 'launch', 'simulation_world.launch.py')
        ),
    )

    # 3. EKF node (delayed 3s to wait for Gazebo topics)
    ekf = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(robotics_subject_dir, 'launch', 'ekf.launch.py')
                ),
            ),
        ],
    )

    # 4. Maze solver nodes (delayed 5s to let simulation stabilise)
    maze_solver = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(maze_solver_dir, 'launch', 'launch_maze_solver.launch.py')
                ),
            ),
        ],
    )

    return LaunchDescription([
        robot_description,
        simulation,
        ekf,
        maze_solver,
    ])
