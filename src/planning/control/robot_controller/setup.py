from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=['robot_controller', 'robot_controller.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='you@example.com',
    description='Unified quadruped controller: Twist in, robot-type adapters out.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller_node = robot_controller.node:main',
            'lite3_status_node = robot_controller.status_node:main',
        ],
    },
)
