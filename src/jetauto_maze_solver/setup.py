import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'jetauto_maze_solver'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joao Gabriel',
    maintainer_email='gab@todo.todo',
    description='Autonomous maze navigation and colored wall detection for JetAuto robot',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'maze_navigator = jetauto_maze_solver.maze_navigator:main',
            'color_wall_counter = jetauto_maze_solver.color_wall_counter:main',
        ],
    },
)
