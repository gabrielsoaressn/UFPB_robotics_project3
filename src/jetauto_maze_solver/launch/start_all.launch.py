"""
Single launch file that starts everything:
  1. Robot description (URDF)
  2. Gazebo simulation with labyrinth world
  3. EKF (odometry filtering)
  4. RViz (LaserScan + Map do SLAM)
  5. SLAM + Maze navigator + Color wall counter
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

    rviz_config = os.path.join(maze_solver_dir, 'rviz', 'maze_solver.rviz')

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

    # 4. RViz com config de LaserScan + Map (delayed 4s para o robot_description estar pronto)
    rviz = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config],
                parameters=[{'use_sim_time': True}],
                output='screen',
            ),
        ],
    )

    # 5. SLAM + Maze solver nodes (delayed 6s para o Gazebo e EKF estabilizarem)
    maze_solver = TimerAction(
        period=6.0,
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
        rviz,
        maze_solver,
    ])
