from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_path = get_package_share_directory("localization_cpp")
    default_rviz_config_path = os.path.join(package_path, "rviz", "localization.rviz")

    rviz_use      = LaunchConfiguration("rviz")
    rviz_cfg      = LaunchConfiguration("rviz_cfg")
    pcd_map_topic = LaunchConfiguration("pcd_map_topic")
    pcd_map_path  = LaunchConfiguration("map")
    use_sim_time  = LaunchConfiguration("use_sim_time")
    odom_topic    = LaunchConfiguration("odom_topic")
    cloud_topic   = LaunchConfiguration("cloud_topic")

    localization_fitness_threshold = 0.8
    localization_rmse_threshold    = 0.3   #0.25
    odom_timeout = 0.5  # seconds

    # ── declare arguments ──────────────────────────────────────────────────────
    declare_rviz_cmd = DeclareLaunchArgument(
        "rviz", default_value="false", description="Use RViz to monitor results")
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        "rviz_cfg", default_value=default_rviz_config_path,
        description="RViz config file path")
    declare_pcd_map_path = DeclareLaunchArgument(
        "map", default_value="", description="Path to PCD map file")
    declare_pcd_map_topic = DeclareLaunchArgument(
        "pcd_map_topic", default_value="/localize/map",
        description="Topic to publish PCD map")
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="false",
        description="Use simulation (bag) time")

    # ── global_localization (C++) ──────────────────────────────────────────────
    global_localization_node = Node(
        package="localization_cpp",
        executable="global_localization",
        name="global_localization",
        output="screen",
        parameters=[{
            "map_voxel_size":          0.4,
            "scan_voxel_size":         0.1,
            "freq_localization":       0.5,
            "freq_global_map":         0.25,
            "localization_threshold":  0.8,
            "fitness_threshold":       localization_fitness_threshold,
            "rmse_threshold":          localization_rmse_threshold,
            "odom_timeout":            odom_timeout,
            "max_retry_times":         20,
            "fov":                     6.28319,
            "fov_far":                 300.0,
            "pcd_map_path":            pcd_map_path,
            "pcd_map_topic":           pcd_map_topic,
            "use_sim_time":            use_sim_time,
        }],
        remappings=[
            ("/Odometry",        odom_topic),
            ("/cloud_registered", cloud_topic),
            ("/submap",           "/localize/submap"),
            ("/map_to_odom",      "/localize/map_to_odom"),
            ("/icp_score",        "/localize/icp_score"),
            ("/loc_status",       "/localize/status"),
        ],
    )

    # ── transform_fusion (C++) ─────────────────────────────────────────────────
    transform_fusion_node = Node(
        package="localization_cpp",
        executable="transform_fusion",
        name="transform_fusion",
        output="screen",
        parameters=[{
            "freq_pub_localization":  50.0,
            "freq_pub_for_visualize": 10.0,
            "fitness_threshold":      localization_fitness_threshold,
            "rmse_threshold":         localization_rmse_threshold,
            "odom_timeout":           odom_timeout,
            "max_time_diff":          0.5,
            "max_position_diff":      1.0,
            "use_sim_time":           use_sim_time,
        }],
        remappings=[
            ("/Odometry",                    odom_topic),
            ("/localization_for_visualize",  "/ql/nav/localization"),
            ("/map_to_odom",                 "/localize/map_to_odom"),
            ("/icp_score",                   "/localize/icp_score"),
            ("/amcl_pose",                   "/localize/amcl_pose"),
        ],
    )

    # ── pcd_to_pointcloud (pcl_ros, unchanged) ─────────────────────────────────
    pcd_publisher_node = Node(
        package="pcl_ros",
        executable="pcd_to_pointcloud",
        name="map_publisher",
        output="screen",
        parameters=[{
            "file_name":    pcd_map_path,
            "tf_frame":     "map",
            "cloud_topic":  pcd_map_topic,
            "period_ms_":   500,
            "use_sim_time": use_sim_time,
        }],
        remappings=[("cloud_pcd", pcd_map_topic)],
    )

    # ── cloud_transformer (C++) ────────────────────────────────────────────────
    cloud_transformer_node = Node(
        package="cloud_preprocess",
        executable="cloud_transformer",
        name="cloud_transformer",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "publish_map": True,
            "publish_body": False,
            "crop_body": False,
        }],
        remappings=[
            ("/cloud_registered", cloud_topic),
            ("/cloud_in_map",     "/ql/nav/cloud_in_map"),
        ],
    )

    rviz_node = Node(
        package="rviz2", executable="rviz2",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-d", rviz_cfg], condition=IfCondition(rviz_use))

    ld = LaunchDescription()
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(declare_pcd_map_path)
    ld.add_action(declare_pcd_map_topic)
    ld.add_action(declare_use_sim_time)
    ld.add_action(DeclareLaunchArgument(
        "odom_topic", default_value="/ql/nav/odometry",
        description="base_link odometry in odom (STVLN Super-LIO: /lio/base/odom)"))
    ld.add_action(DeclareLaunchArgument(
        "cloud_topic", default_value="/cloud_registered",
        description="deskewed scan in odom (STVLN Super-LIO: /lio/cloud_world)"))

    ld.add_action(rviz_node)
    ld.add_action(global_localization_node)
    ld.add_action(transform_fusion_node)
    ld.add_action(pcd_publisher_node)
    ld.add_action(cloud_transformer_node)

    return ld
