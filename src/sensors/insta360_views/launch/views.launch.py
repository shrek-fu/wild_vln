from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _models() -> Path | None:
    """Media SDK model directory; absent on Jetson, where the CUDA backend needs none."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        for relative in (
            Path("insta360_driver") / "insta360_media_sdk" / "bin" / "models",
            Path("drivers") / "insta360_media_sdk" / "bin" / "models",
            Path("src") / "sensors" / "drivers" / "insta360_media_sdk" / "bin" / "models",
        ):
            candidate = parent / relative
            if candidate.is_dir():
                return candidate
    return None


def generate_launch_description() -> LaunchDescription:
    pkg = Path(__file__).resolve().parents[1]
    config = pkg / "config" / "insta360_views.yaml"
    if not config.is_file():
        config = pkg / "insta360_views.yaml"
    overrides = {"stitch_backend": LaunchConfiguration("stitch_backend")}
    models = _models()
    if models is not None:
        overrides["model_dir"] = str(models)
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "stitch_backend",
                default_value="auto",
                description="auto (Media SDK if built, else CUDA), mediasdk or cuda",
            ),
            Node(
                package="insta360_views",
                executable="insta360_views",
                name="insta360_views",
                output="screen",
                parameters=[str(config), overrides],
            ),
        ]
    )
