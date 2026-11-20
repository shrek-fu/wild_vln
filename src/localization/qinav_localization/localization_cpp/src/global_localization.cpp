// global_localization.cpp
// C++ port of global_localization.py
//
// Key differences from Python version:
//   - Open3D ICP  →  PCL IterativeClosestPoint (same algorithm, same parameters)
//   - Open3D VoxelGrid  →  pcl::VoxelGrid
//   - Open3D PCD I/O  →  pcl::io::loadPCDFile
//   - ros2_numpy  →  sensor_msgs::PointCloud2Iterator
//   - fitness (inlier fraction) computed manually after PCL ICP
//
// All parameters, topic names, thresholds and logic are identical to the Python version.

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <atomic>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <tf2_eigen/tf2_eigen.hpp>

using namespace std::chrono_literals;
using PclCloud = pcl::PointCloud<pcl::PointXYZ>;

// ─── helpers ──────────────────────────────────────────────────────────────────

static Eigen::Matrix4f poseToMat(const geometry_msgs::msg::Pose & pose)
{
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    mat(0, 3) = static_cast<float>(pose.position.x);
    mat(1, 3) = static_cast<float>(pose.position.y);
    mat(2, 3) = static_cast<float>(pose.position.z);
    Eigen::Quaternionf q(
        static_cast<float>(pose.orientation.w),
        static_cast<float>(pose.orientation.x),
        static_cast<float>(pose.orientation.y),
        static_cast<float>(pose.orientation.z));
    mat.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    return mat;
}

static Eigen::Matrix4f inverseSE3(const Eigen::Matrix4f & T)
{
    Eigen::Matrix4f inv = Eigen::Matrix4f::Identity();
    inv.block<3, 3>(0, 0) = T.block<3, 3>(0, 0).transpose();
    inv.block<3, 1>(0, 3) = -inv.block<3, 3>(0, 0) * T.block<3, 1>(0, 3);
    return inv;
}

// Voxel downsample a PCL cloud
static PclCloud::Ptr voxelDownsample(const PclCloud::ConstPtr & cloud, float leaf)
{
    PclCloud::Ptr out(new PclCloud);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf, leaf, leaf);
    vg.filter(*out);
    return out;
}

// Convert sensor_msgs::PointCloud2 to PCL cloud (extract xyz)
static PclCloud::Ptr msgToCloud(const sensor_msgs::msg::PointCloud2 & msg)
{
    PclCloud::Ptr out(new PclCloud);
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z)) {
            out->push_back({*iter_x, *iter_y, *iter_z});
        }
    }
    return out;
}

// Run PCL ICP and return (transformation, fitness, inlier_rmse)
// fitness = inlier_fraction (same semantics as Open3D), not PCL's mean-squared-distance
struct IcpResult { Eigen::Matrix4f transformation; double fitness; double inlier_rmse; };

static IcpResult registrationAtScale(
    const PclCloud::ConstPtr & scan,
    const PclCloud::ConstPtr & map,
    const Eigen::Matrix4f & initial,
    float scan_voxel_size,
    float map_voxel_size,
    float scale)
{
    PclCloud::Ptr scan_ds = voxelDownsample(scan, scan_voxel_size * scale);
    PclCloud::Ptr map_ds  = voxelDownsample(map,  map_voxel_size  * scale);

    float max_corr_dist = 1.0f * scale;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(scan_ds);
    icp.setInputTarget(map_ds);
    icp.setMaxCorrespondenceDistance(static_cast<double>(max_corr_dist));
    icp.setMaximumIterations(20);
    icp.setTransformationEpsilon(1e-6);

    PclCloud::Ptr aligned(new PclCloud);
    icp.align(*aligned, initial);

    Eigen::Matrix4f T = icp.getFinalTransformation();

    // Compute fitness (inlier fraction) and inlier RMSE manually
    // (PCL's getFitnessScore returns mean squared distance, not inlier fraction)
    if (aligned->empty() || map_ds->empty()) {
        return {T, 0.0, 1e9};
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(map_ds);

    int inliers = 0;
    double sum_sq = 0.0;
    float max_dist_sq = max_corr_dist * max_corr_dist;

    for (const auto & pt : *aligned) {
        std::vector<int>   idx(1);
        std::vector<float> dist_sq(1);
        if (tree.nearestKSearch(pt, 1, idx, dist_sq) > 0 && dist_sq[0] < max_dist_sq) {
            ++inliers;
            sum_sq += dist_sq[0];
        }
    }

    double fitness = static_cast<double>(inliers) / aligned->size();
    double rmse    = (inliers > 0) ? std::sqrt(sum_sq / inliers) : 1e9;
    return {T, fitness, rmse};
}

// ─── node ─────────────────────────────────────────────────────────────────────

class FastLIOLocalization : public rclcpp::Node
{
public:
    FastLIOLocalization()
    : Node("fast_lio_localization"),
      T_map_to_odom_(Eigen::Matrix4f::Identity())
    {
        // ── parameters ────────────────────────────────────────────────────────
        declare_parameter("map_voxel_size", 0.4);
        declare_parameter("scan_voxel_size", 0.1);
        declare_parameter("freq_localization", 0.5);
        declare_parameter("freq_global_map", 0.25);
        declare_parameter("localization_threshold", 0.8);
        declare_parameter("fov", 6.28319);
        declare_parameter("fov_far", 300.0);
        declare_parameter("pcd_map_topic", std::string("/map"));
        declare_parameter("pcd_map_path", std::string(""));
        declare_parameter("fitness_threshold", 0.8);
        declare_parameter("rmse_threshold", 0.25);
        declare_parameter("odom_timeout", 0.5);
        declare_parameter("max_retry_times", 10);
        // Points (frame map) appended to the global map at runtime, e.g. by map_extender beyond the prior map.
        declare_parameter("map_append_topic", std::string("/localize/map_append"));

        map_voxel_size_    = static_cast<float>(get_parameter("map_voxel_size").as_double());
        scan_voxel_size_   = static_cast<float>(get_parameter("scan_voxel_size").as_double());
        fov_               = get_parameter("fov").as_double();
        fov_far_           = get_parameter("fov_far").as_double();
        fov_half_          = fov_ / 2.0;
        fitness_threshold_ = get_parameter("fitness_threshold").as_double();
        rmse_threshold_    = get_parameter("rmse_threshold").as_double();
        odom_timeout_      = get_parameter("odom_timeout").as_double();
        max_retry_times_   = get_parameter("max_retry_times").as_int();

        // ── publishers ────────────────────────────────────────────────────────
        pub_submap_     = create_publisher<sensor_msgs::msg::PointCloud2>("/submap", 10);
        pub_map_to_odom_= create_publisher<nav_msgs::msg::Odometry>("/map_to_odom", 10);
        pub_icp_score_  = create_publisher<std_msgs::msg::Float32MultiArray>("/icp_score", 10);
        pub_status_     = create_publisher<std_msgs::msg::String>("/loc_status", 10);

        // ── load global map (blocking) ─────────────────────────────────────
        RCLCPP_INFO(get_logger(), "Loading global map...");
        initializeGlobalMap();
        RCLCPP_INFO(get_logger(), "Global map loaded (%zu points after voxel).",
            global_map_->size());

        // ── subscribers ───────────────────────────────────────────────────────
        sub_scan_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/cloud_registered", 10,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cbSaveCurScan(msg); });
        sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry", 10,
            [this](nav_msgs::msg::Odometry::SharedPtr msg) { cbSaveCurOdom(msg); });
        auto init_pose_cb =
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                cbInitializePose(msg);
            };
        sub_init_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10, init_pose_cb);
        // RViz dddmr 3DInitialPose
        sub_init_pose_3d_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initial_3d_pose", 10, init_pose_cb);
        const std::string append_topic = get_parameter("map_append_topic").as_string();
        if (!append_topic.empty()) {
            sub_map_append_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                append_topic, rclcpp::QoS(10).reliable(),
                [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cbMapAppend(msg); });
        }
        double freq = get_parameter("freq_localization").as_double();
        using Ms = std::chrono::milliseconds;
        timer_ = create_wall_timer(
            Ms(static_cast<int>(1000.0 / freq)),
            [this]() { localisationTimerCallback(); });
    }

private:
    // ── state ─────────────────────────────────────────────────────────────────
    PclCloud::Ptr global_map_;
    // Pre-built homogeneous matrix for global map points: shape (N, 4)
    Eigen::MatrixXf global_map_pts_h_;   // row = [x, y, z, 1]

    nav_msgs::msg::Odometry::SharedPtr cur_odom_;
    PclCloud::Ptr cur_scan_;
    std::mutex scan_odom_mutex_;
    geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pending_initial_pose_;

    Eigen::Matrix4f T_map_to_odom_;
    bool initialized_ = false;
    bool localization_ok_ = false;  // last ICP met fitness+rmse thresholds
    int  localize_times_ = 0;

    rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};
    bool has_odom_time_ = false;

    // ── parameters ────────────────────────────────────────────────────────────
    float  map_voxel_size_, scan_voxel_size_;
    double fov_, fov_far_, fov_half_;
    double fitness_threshold_, rmse_threshold_, odom_timeout_;
    int    max_retry_times_;

    // ── ROS handles ───────────────────────────────────────────────────────────
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_submap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_map_to_odom_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_icp_score_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_scan_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pose_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pose_3d_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_map_append_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ── initialization ────────────────────────────────────────────────────────

    void initializeGlobalMap()
    {
        std::string path = get_parameter("pcd_map_path").as_string();
        PclCloud::Ptr raw(new PclCloud);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *raw) < 0) {
            RCLCPP_FATAL(get_logger(), "Failed to load PCD: %s", path.c_str());
            return;
        }
        global_map_ = voxelDownsample(raw, map_voxel_size_);

        // Pre-build homogeneous matrix (N×4)
        size_t N = global_map_->size();
        global_map_pts_h_.resize(N, 4);
        for (size_t i = 0; i < N; ++i) {
            global_map_pts_h_(i, 0) = global_map_->points[i].x;
            global_map_pts_h_(i, 1) = global_map_->points[i].y;
            global_map_pts_h_(i, 2) = global_map_->points[i].z;
            global_map_pts_h_(i, 3) = 1.0f;
        }
    }

    // ── callbacks ─────────────────────────────────────────────────────────────

    void cbSaveCurOdom(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(scan_odom_mutex_);
        cur_odom_ = msg;
        last_odom_time_ = now();
        has_odom_time_ = true;
        if (pending_initial_pose_) {
            auto pending = pending_initial_pose_;
            pending_initial_pose_.reset();
            RCLCPP_INFO(get_logger(),
                "Odometry arrived; applying buffered initial pose.");
            applyInitializePoseLocked(pending);
        }
    }

    // The sender only sends voxels it has not sent before, so this appends without re-voxelizing the whole map.
    void cbMapAppend(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (msg->header.frame_id != "map") {
            RCLCPP_WARN(get_logger(), "map_append in frame '%s', expected 'map': ignored",
                msg->header.frame_id.c_str());
            return;
        }
        PclCloud::Ptr add = voxelDownsample(msgToCloud(*msg), map_voxel_size_);
        if (add->empty() || !global_map_) return;
        std::lock_guard<std::mutex> lk(scan_odom_mutex_);
        const Eigen::Index n0 = global_map_pts_h_.rows();
        global_map_pts_h_.conservativeResize(n0 + static_cast<Eigen::Index>(add->size()), 4);
        for (size_t i = 0; i < add->size(); ++i) {
            const auto & p = add->points[i];
            global_map_->push_back(p);
            global_map_pts_h_.row(n0 + static_cast<Eigen::Index>(i)) << p.x, p.y, p.z, 1.0f;
        }
        RCLCPP_INFO(get_logger(), "Global map extended by %zu points (now %zu).",
            add->size(), global_map_->size());
    }

    void cbSaveCurScan(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        PclCloud::Ptr cloud = msgToCloud(*msg);
        std::lock_guard<std::mutex> lk(scan_odom_mutex_);
        cur_scan_ = cloud;
    }

    void cbInitializePose(geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(scan_odom_mutex_);
        if (!cur_odom_) {
            pending_initial_pose_ = msg;
            RCLCPP_WARN(get_logger(),
                "Received initial pose but no odometry yet; buffering until odometry arrives.");
            return;
        }
        applyInitializePoseLocked(msg);
    }

    // Caller must hold scan_odom_mutex_. Requires cur_odom_.
    void applyInitializePoseLocked(
        const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr & msg)
    {
        // initialpose from RViz is map->base; convert it to map->odom using current odom->base.
        Eigen::Matrix4f T_map_to_base  = poseToMat(msg->pose.pose);
        Eigen::Matrix4f T_odom_to_base = poseToMat(cur_odom_->pose.pose);
        Eigen::Matrix4f initial_map_to_odom = T_map_to_base * inverseSE3(T_odom_to_base);

        // If already localized successfully, reject a low-scoring initial pose.
        if (localization_ok_ && cur_scan_) {
            auto eval = evaluatePose(initial_map_to_odom);
            if (eval) {
                publishIcpScore(eval->fitness, eval->inlier_rmse);
                if (!isLocalizationOk(eval->fitness, eval->inlier_rmse)) {
                    RCLCPP_WARN(get_logger(),
                        "Rejecting initial pose: localization already successful, "
                        "candidate score too low (fitness=%.4f, rmse=%.4f; "
                        "need fitness>%.2f and rmse<%.2f).",
                        eval->fitness, eval->inlier_rmse,
                        fitness_threshold_, rmse_threshold_);
                    return;
                }
            }
        }

        T_map_to_odom_ = initial_map_to_odom;
        initialized_ = true;
        localize_times_ = 0;
        RCLCPP_INFO(get_logger(), "Initial pose received and converted to map->odom.");
        if (cur_scan_) {
            globalLocalization(T_map_to_odom_);
        }
    }

    // ── timer callback ────────────────────────────────────────────────────────

    void localisationTimerCallback()
    {
        if (!initialized_) {
            RCLCPP_INFO(get_logger(), "Waiting for initial pose...");
            return;
        }
        if (checkOdomTimeout()) return;

        std::lock_guard<std::mutex> lk(scan_odom_mutex_);
        if (cur_scan_) {
            globalLocalization(T_map_to_odom_);
        }
    }

    // ── core localization ─────────────────────────────────────────────────────

    bool isLocalizationOk(double fitness, double rmse) const
    {
        return fitness > fitness_threshold_ && rmse < rmse_threshold_;
    }

    // Run coarse+fine ICP for a pose estimate. Returns fine result, or nullopt if FOV empty.
    // Caller must hold scan_odom_mutex_ and have a valid cur_scan_.
    std::optional<IcpResult> evaluatePose(const Eigen::Matrix4f & pose_estimation)
    {
        PclCloud::Ptr scan_tobe_mapped = cur_scan_;
        PclCloud::Ptr map_in_fov = cropGlobalMapInFOV(pose_estimation);

        if (!map_in_fov || map_in_fov->empty()) {
            RCLCPP_WARN(get_logger(), "FOV submap is empty, skipping ICP.");
            return std::nullopt;
        }

        // Coarse ICP (scale=5) — warms correspondence; fine result is authoritative.
        (void)registrationAtScale(
            scan_tobe_mapped, map_in_fov, pose_estimation,
            scan_voxel_size_, map_voxel_size_, 5.0f);

        return registrationAtScale(
            scan_tobe_mapped, map_in_fov, pose_estimation,
            scan_voxel_size_, map_voxel_size_, 1.0f);
    }

    void globalLocalization(const Eigen::Matrix4f & pose_estimation)
    {
        auto fine = evaluatePose(pose_estimation);
        if (!fine) {
            return;
        }

        double fitness = fine->fitness;
        double rmse    = fine->inlier_rmse;
        RCLCPP_INFO(get_logger(), "Fitness score %.4f, rmse %.4f", fitness, rmse);
        publishIcpScore(fitness, rmse);

        if (fitness > fitness_threshold_) {
            T_map_to_odom_ = fine->transformation;
            publishOdom(fine->transformation);
        }

        if (isLocalizationOk(fitness, rmse)) {
            localize_times_ = 0;
            localization_ok_ = true;
            publishStatus("localization_success");
            RCLCPP_INFO(get_logger(), "localize success");
        } else {
            ++localize_times_;
            if (localize_times_ > max_retry_times_) {
                localize_times_ = 0;
                initialized_ = false;
                localization_ok_ = false;
                T_map_to_odom_ = Eigen::Matrix4f::Identity();
                publishStatus("localization_failed");
                RCLCPP_ERROR(get_logger(),
                    "localize failed for %d times, set initialized to False, "
                    "waiting for initial pose again.", max_retry_times_);
            }
        }
    }

    // Crop global map to sensor FOV and publish submap
    PclCloud::Ptr cropGlobalMapInFOV(const Eigen::Matrix4f & pose_estimation)
    {
        if (!cur_odom_) return nullptr;

        Eigen::Matrix4f T_odom_to_base = poseToMat(cur_odom_->pose.pose);
        Eigen::Matrix4f T_map_to_base  = pose_estimation * T_odom_to_base;
        Eigen::Matrix4f T_base_to_map  = inverseSE3(T_map_to_base);

        // Transform all global map points to base_link frame
        // global_map_pts_h_ : (N, 4)
        // result            : (N, 4)  = global_map_pts_h_ * T_base_to_map^T
        Eigen::MatrixXf in_base = (T_base_to_map * global_map_pts_h_.transpose()).transpose();
        // in_base columns: [x_b, y_b, z_b, 1]

        // FOV filter (same logic as Python)
        std::vector<int> indices;
        indices.reserve(1024);
        float fov_far_f = static_cast<float>(fov_far_);
        float fov_half_f = static_cast<float>(fov_half_);

        if (fov_ > M_PI) {
            // Omnidirectional: only distance filter
            for (int i = 0; i < in_base.rows(); ++i) {
                float xb = in_base(i, 0);
                float yb = in_base(i, 1);
                float az = std::atan2(yb, xb);
                if (xb < fov_far_f && std::abs(az) < fov_half_f) {
                    indices.push_back(i);
                }
            }
        } else {
            for (int i = 0; i < in_base.rows(); ++i) {
                float xb = in_base(i, 0);
                float yb = in_base(i, 1);
                float az = std::atan2(yb, xb);
                if (xb > 0 && xb < fov_far_f && std::abs(az) < fov_half_f) {
                    indices.push_back(i);
                }
            }
        }

        PclCloud::Ptr fov_cloud(new PclCloud);
        fov_cloud->reserve(indices.size());
        for (int idx : indices) {
            fov_cloud->push_back({
                global_map_pts_h_(idx, 0),
                global_map_pts_h_(idx, 1),
                global_map_pts_h_(idx, 2)
            });
        }

        // Publish submap (every 10th point for bandwidth)
        publishSubmap(fov_cloud);

        return fov_cloud;
    }

    // ── publishers ────────────────────────────────────────────────────────────

    void publishOdom(const Eigen::Matrix4f & T)
    {
        if (!cur_odom_) return;

        nav_msgs::msg::Odometry msg;
        msg.header.stamp    = cur_odom_->header.stamp;
        msg.header.frame_id = "map";
        msg.pose.pose.position.x = T(0, 3);
        msg.pose.pose.position.y = T(1, 3);
        msg.pose.pose.position.z = T(2, 3);
        Eigen::Quaternionf q(T.block<3, 3>(0, 0));
        q.normalize();
        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        msg.pose.pose.orientation.w = q.w();
        pub_map_to_odom_->publish(msg);
    }

    void publishIcpScore(double fitness, double rmse)
    {
        std_msgs::msg::Float32MultiArray msg;
        msg.data = {static_cast<float>(fitness), static_cast<float>(rmse)};
        pub_icp_score_->publish(msg);
    }

    void publishStatus(const std::string & status)
    {
        std_msgs::msg::String msg;
        msg.data = status;
        pub_status_->publish(msg);
    }

    void publishSubmap(const PclCloud::Ptr & cloud)
    {
        if (!cur_odom_ || cloud->empty()) return;

        sensor_msgs::msg::PointCloud2 out;
        PclCloud sub;
        for (size_t i = 0; i < cloud->size(); i += 10) {
            sub.push_back((*cloud)[i]);
        }
        pcl::toROSMsg(sub, out);
        out.header.stamp    = cur_odom_->header.stamp;
        out.header.frame_id = "map";
        pub_submap_->publish(out);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    bool checkOdomTimeout()
    {
        if (!has_odom_time_) return true;
        double dt = (now() - last_odom_time_).seconds();
        if (dt > odom_timeout_) {
            RCLCPP_WARN(get_logger(),
                "Odometry message timeout: %.3fs (max: %.3fs)", dt, odom_timeout_);
            cur_odom_.reset();
            cur_scan_.reset();
            has_odom_time_ = false;
            return true;
        }
        return false;
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FastLIOLocalization>());
    rclcpp::shutdown();
    return 0;
}
