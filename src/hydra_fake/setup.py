from setuptools import find_packages, setup

package_name = "hydra_fake"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="qi",
    maintainer_email="dev@qi.local",
    description="Hydra-style topo map: pose-graph anchored objects, LIO reset stitching, free-space places.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["hydra_fake_node = hydra_fake.node:main"]},
)
