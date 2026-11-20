from glob import glob
import os

from setuptools import find_packages, setup

package_name = "scene_memory"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="qi",
    maintainer_email="dev@qi.local",
    description="4D scene memory over an Insta360 + VIO stream: perception, layered topo mapping, spatio-temporal query.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = scene_memory.nodes.detector_node:main",
            "topo_map_node = scene_memory.nodes.topo_map_node:main",
            "view_depth_node = scene_memory.nodes.view_depth_node:main",
            "latency_probe_node = scene_memory.nodes.latency_probe_node:main",
            "lio_watchdog_node = scene_memory.nodes.lio_watchdog_node:main",
            "camera_tf_node = scene_memory.nodes.camera_tf_node:main",
            "goal_server_node = scene_memory.nodes.goal_server_node:main",
            "hydra_scene_node = scene_memory.nodes.hydra_scene_node:main",
            "daaam_live_node = scene_memory.nodes.daaam_live_node:main",
            "infer_image = scene_memory.perception.infer_image:main",
        ],
    },
)
