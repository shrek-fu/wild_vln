"""Livox mount shared by the static TF launches and the body-cloud crop."""

import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory


def mount_file():
    return os.path.join(
        get_package_share_directory('cloud_preprocess'),
        'config',
        'base_link_mount.yaml',
    )


def load_mounts():
    with open(mount_file(), 'r', encoding='utf-8') as mount_file_handle:
        return yaml.safe_load(mount_file_handle) or {}


def load_mount(robot_type):
    """Return (mount dict or None, all mounts)."""
    mounts = load_mounts()
    key = (robot_type or '').strip().lower() or 'zsl1'
    mount = mounts.get(key)
    if not isinstance(mount, dict):
        return None, mounts
    return mount, mounts


def livox_to_base_link(lidar_x, lidar_y, lidar_height, pitch_deg):
    """Pose of base_link in livox_frame.

    lidar_x, lidar_y and lidar_height are the livox_frame origin in base_link.
    pitch_deg is positive when livox +X pitches down about base_link +Y.
    """
    pitch = math.radians(float(pitch_deg))
    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    tx = -float(lidar_x) * cosine + float(lidar_height) * sine
    ty = -float(lidar_y)
    tz = -float(lidar_x) * sine - float(lidar_height) * cosine
    half = pitch / 2.0
    return (tx, ty, tz, 0.0, -math.sin(half), 0.0, math.cos(half))


def format_component(value):
    return f'{float(value):.6f}'


def body_box(mount):
    """Length, width, height in meters. Non-positive length disables the crop."""
    return (
        float(mount.get('body_length', 0.0) or 0.0),
        float(mount.get('body_width', 0.0) or 0.0),
        float(mount.get('body_height', 0.0) or 0.0),
    )
