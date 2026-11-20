// transform_fusion.cpp
// C++ port of transform_fusion.py
// Logic: fuse map->odom (from ICP localization) with odom->base_link (from odometry)
//        to publish map->base_link at high frequency, and broadcast map->odom TF.

#include <chrono>
#include <memory>
#include <string>
#include <optional>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>
#include <tf2_eigen/tf2_eigen.hpp>

using namespace std::chrono_literals;

// ─── helpers ──────────────────────────────────────────────────────────────────

static Eigen::Matrix4d poseToMat(const geometry_msgs::msg::Pose & pose)
{
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    mat(0, 3) = pose.position.x;
    mat(1, 3) = pose.position.y;
    mat(2, 3) = pose.position.z;
    Eigen::Quaterniond q(
        pose.orientation.w,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z);
    mat.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    return mat;
}

static Eigen::Quaterniond matToQuat(const Eigen::Matrix4d & mat)
{
    return Eigen::Quaterniond(mat.block<3, 3>(0, 0)).normalized();
}

// Add nanosecond offset to a builtin_interfaces/Time stamp
static builtin_interfaces::msg::Time addTimeOffset(
    const builtin_interfaces::msg::Time & stamp, double offset_sec)
{
    int64_t total_ns =
        static_cast<int64_t>(stamp.sec) * 1'000'000'000LL +
        static_cast<int64_t>(stamp.nanosec) +
        static_cast<int64_t>(offset_sec * 1e9);
    builtin_interfaces::msg::Time t;
    t.sec     = static_cast<int32_t>(total_ns / 1'000'000'000LL);
    t.nanosec = static_cast<uint32_t>(total_ns % 1'000'000'000LL);
    return t;
}

// ─── node ─────────────────────────────────────────────────────────────────────

class TransformFusion : public rclcpp::Node
{
public:
    TransformFusion()
    : Node("transform_fusion")
    {
        // ── parameters ────────────────────────────────────────────────────────
        declare_parameter("freq_pub_localization", 50.0);
        declare_parameter("freq_pub_for_visualize", 10.0);
        declare_parameter("fitness_threshold", 0.5);
        declare_parameter("rmse_threshold", 0.25);
        declare_parameter("odom_timeout", 0.5);
        declare_parameter("max_time_diff", 0.5);
        declare_parameter("max_position_diff", 1.0);

        freq_pub_localization_  = get_parameter("freq_pub_localization").as_double();
        freq_pub_for_visualize_ = get_parameter("freq_pub_for_visualize").as_double();
        fitness_threshold_      = get_parameter("fitness_threshold").as_double();
        rmse_threshold_         = get_parameter("rmse_threshold").as_double();
        odom_timeout_           = get_parameter("odom_timeout").as_double();
        max_time_diff_          = get_parameter("max_time_diff").as_double();
        max_position_diff_      = get_parameter("max_position_diff").as_double();
        max_pos_diff_sq_        = max_position_diff_ * max_position_diff_;

        // ── publishers ────────────────────────────────────────────────────────
        pub_localization_ = create_publisher<nav_msgs::msg::Odometry>("/localization", 1);
        pub_localization_viz_ = create_publisher<nav_msgs::msg::Odometry>("/localization_for_visualize", 1);
        pub_amcl_pose_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/amcl_pose", 1);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // ── subscribers ───────────────────────────────────────────────────────
        sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/Odometry", 1,
            [this](nav_msgs::msg::Odometry::SharedPtr msg) { cbSaveCurOdom(msg); });
        sub_map_to_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/map_to_odom", 1,
            [this](nav_msgs::msg::Odometry::SharedPtr msg) { cbSaveMapToOdom(msg); });
        sub_icp_score_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            "/icp_score", 1,
            [this](std_msgs::msg::Float32MultiArray::SharedPtr msg) { cbSaveIcpScore(msg); });

        // ── timers ────────────────────────────────────────────────────────────
        using Ms = std::chrono::milliseconds;
        auto period_loc = Ms(static_cast<int>(1000.0 / freq_pub_localization_));
        auto period_viz = Ms(static_cast<int>(1000.0 / freq_pub_for_visualize_));
        timer_fusion_ = create_wall_timer(period_loc, [this]() { transformFusion(); });
        timer_viz_    = create_wall_timer(period_viz, [this]() { publishLocalizationForVisualize(); });

        RCLCPP_INFO(get_logger(), "transform_fusion node started.");
    }

private:
    // ── cached state ──────────────────────────────────────────────────────────
    nav_msgs::msg::Odometry::SharedPtr cur_odom_;
    Eigen::Matrix4d T_map_to_odom_   = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q_map_to_odom_ = Eigen::Quaterniond::Identity();
    float icp_fitness_  = 0.f;
    float icp_rmse_     = 1e9f;
    bool  icp_score_available_ = false;

    nav_msgs::msg::Odometry localization_msg_;
    bool localization_pub_available_ = false;

    rclcpp::Time last_odom_time_{0, 0, RCL_ROS_TIME};
    bool has_odom_time_ = false;

    // For odom validity check
    int64_t prev_stamp_ns_ = -1;
    double prev_x_ = 0, prev_y_ = 0, prev_z_ = 0;

    // ── parameters ────────────────────────────────────────────────────────────
    double freq_pub_localization_, freq_pub_for_visualize_;
    double fitness_threshold_, rmse_threshold_;
    double odom_timeout_, max_time_diff_, max_position_diff_, max_pos_diff_sq_;

    // ── ROS handles ───────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_viz_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_amcl_pose_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_map_to_odom_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_icp_score_;
    rclcpp::TimerBase::SharedPtr timer_fusion_;
    rclcpp::TimerBase::SharedPtr timer_viz_;

    // ── callbacks ─────────────────────────────────────────────────────────────

    void cbSaveCurOdom(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        if (checkOdomValidity(msg)) {
            cur_odom_ = msg;
            last_odom_time_ = now();
            has_odom_time_ = true;
        } else {
            RCLCPP_ERROR(get_logger(), "Odometry message is invalid, stopping relocalization");
            cur_odom_.reset();
        }
    }

    void cbSaveMapToOdom(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Pre-cache matrix and quaternion; transform_fusion() reads cached values
        T_map_to_odom_ = poseToMat(msg->pose.pose);
        q_map_to_odom_ = matToQuat(T_map_to_odom_);
    }

    void cbSaveIcpScore(std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        if (msg->data.size() >= 2) {
            icp_fitness_ = msg->data[0];
            icp_rmse_    = msg->data[1];
            icp_score_available_ = true;
        }
    }

    // ── main fusion timer (high frequency) ────────────────────────────────────

    void transformFusion()
    {
        if (checkOdomTimeout()) return;
        if (!cur_odom_) return;

        // ── broadcast map→odom TF ─────────────────────────────────────────────
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp    = addTimeOffset(cur_odom_->header.stamp, 0.1);
        tf_msg.header.frame_id = "map";
        tf_msg.child_frame_id  = "odom";
        tf_msg.transform.translation.x = T_map_to_odom_(0, 3);
        tf_msg.transform.translation.y = T_map_to_odom_(1, 3);
        tf_msg.transform.translation.z = T_map_to_odom_(2, 3);
        tf_msg.transform.rotation.x = q_map_to_odom_.x();
        tf_msg.transform.rotation.y = q_map_to_odom_.y();
        tf_msg.transform.rotation.z = q_map_to_odom_.z();
        tf_msg.transform.rotation.w = q_map_to_odom_.w();
        tf_broadcaster_->sendTransform(tf_msg);

        // ── fuse map→odom × odom→base_link ───────────────────────────────────
        Eigen::Matrix4d T_odom_to_base = poseToMat(cur_odom_->pose.pose);
        Eigen::Matrix4d T_map_to_base  = T_map_to_odom_ * T_odom_to_base;
        Eigen::Quaterniond q_result(T_map_to_base.block<3, 3>(0, 0));
        q_result.normalize();

        localization_msg_.header.stamp    = cur_odom_->header.stamp;
        localization_msg_.header.frame_id = "map";
        localization_msg_.child_frame_id  = "base_link";
        localization_msg_.pose.pose.position.x    = T_map_to_base(0, 3);
        localization_msg_.pose.pose.position.y    = T_map_to_base(1, 3);
        localization_msg_.pose.pose.position.z    = T_map_to_base(2, 3);
        localization_msg_.pose.pose.orientation.x = q_result.x();
        localization_msg_.pose.pose.orientation.y = q_result.y();
        localization_msg_.pose.pose.orientation.z = q_result.z();
        localization_msg_.pose.pose.orientation.w = q_result.w();
        localization_msg_.twist = cur_odom_->twist;

        pub_localization_->publish(localization_msg_);
        localization_pub_available_ = true;

        if (isLocalizationValid()) {
            auto amcl = geometry_msgs::msg::PoseWithCovarianceStamped();
            amcl.header = localization_msg_.header;
            amcl.pose.pose = localization_msg_.pose.pose;
            amcl.pose.covariance = localization_msg_.pose.covariance;
            pub_amcl_pose_->publish(amcl);
        }
    }

    // ── low-frequency visualization timer ─────────────────────────────────────

    void publishLocalizationForVisualize()
    {
        if (localization_pub_available_) {
            pub_localization_viz_->publish(localization_msg_);
            localization_pub_available_ = false;
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    bool isLocalizationValid() const
    {
        return icp_score_available_ &&
               icp_fitness_ > static_cast<float>(fitness_threshold_) &&
               icp_rmse_ < static_cast<float>(rmse_threshold_);
    }

    bool checkOdomTimeout()
    {
        if (!has_odom_time_) return true;
        double dt = (now() - last_odom_time_).seconds();
        if (dt > odom_timeout_) {
            RCLCPP_WARN(get_logger(),
                "Odometry message timeout: %.3fs (max: %.3fs)", dt, odom_timeout_);
            cur_odom_.reset();
            prev_stamp_ns_ = -1;
            has_odom_time_ = false;
            icp_score_available_ = false;
            return true;
        }
        return false;
    }

    bool checkOdomValidity(const nav_msgs::msg::Odometry::SharedPtr & msg)
    {
        const auto & s = msg->header.stamp;
        int64_t cur_ns = static_cast<int64_t>(s.sec) * 1'000'000'000LL + s.nanosec;
        const auto & p = msg->pose.pose.position;

        if (prev_stamp_ns_ < 0) {
            prev_stamp_ns_ = cur_ns;
            prev_x_ = p.x; prev_y_ = p.y; prev_z_ = p.z;
            return true;
        }

        double time_diff = std::abs(cur_ns - prev_stamp_ns_) * 1e-9;
        if (time_diff > max_time_diff_) {
            RCLCPP_WARN(get_logger(),
                "Odometry time gap too large: %.3fs (max: %.3fs)", time_diff, max_time_diff_);
            return false;
        }

        double dx = p.x - prev_x_, dy = p.y - prev_y_, dz = p.z - prev_z_;
        if (dx*dx + dy*dy + dz*dz > max_pos_diff_sq_) {
            RCLCPP_WARN(get_logger(),
                "Odometry position change too large: %.3fm (max: %.3fm)",
                std::sqrt(dx*dx + dy*dy + dz*dz), max_position_diff_);
            return false;
        }

        prev_stamp_ns_ = cur_ns;
        prev_x_ = p.x; prev_y_ = p.y; prev_z_ = p.z;
        return true;
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TransformFusion>());
    rclcpp::shutdown();
    return 0;
}
