// ground_to_odom.cpp
// C++ port of ground_to_odom.py
// Subscribes to ground and non-ground segmentation topics, transforms to odom
// frame via TF, optionally appends a circle-disk fill layer at ground_height.

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

using namespace std::chrono_literals;

class GroundTrans : public rclcpp::Node
{
public:
    GroundTrans()
    : Node("ground_trans"),
      tf_buffer_(get_clock()),
      tf_listener_(tf_buffer_, this, /*spin_thread=*/false)
    {
        // MultiThreadedExecutor drives TF callbacks in a separate thread;
        // inform the buffer so it won't warn about timeout calls.
        tf_buffer_.setUsingDedicatedThread(true);
        declare_parameter("input_topic",             "/segmentation/ground");
        declare_parameter("output_topic",            "/ground_in_odom");
        declare_parameter("non_ground_input_topic",  "/segmentation/obstacle");
        declare_parameter("non_ground_output_topic", "/non_ground_in_odom");
        declare_parameter("source_frame",    "base_link");
        declare_parameter("target_frame",    "odom");
        declare_parameter("ground_height",   -0.5);
        declare_parameter("fill_radius",      0.8);
        declare_parameter("fill_resolution",  0.15);

        std::string input_topic     = get_parameter("input_topic").as_string();
        std::string output_topic    = get_parameter("output_topic").as_string();
        std::string ng_input_topic  = get_parameter("non_ground_input_topic").as_string();
        std::string ng_output_topic = get_parameter("non_ground_output_topic").as_string();
        source_frame_    = get_parameter("source_frame").as_string();
        target_frame_    = get_parameter("target_frame").as_string();
        ground_height_   = get_parameter("ground_height").as_double();
        fill_radius_     = get_parameter("fill_radius").as_double();
        fill_resolution_ = get_parameter("fill_resolution").as_double();

        if (fill_radius_ >= 0.0) {
            makeFillPoints();
        }

        cb_group_ground_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_ng_     = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions opt_ground, opt_ng;
        opt_ground.callback_group = cb_group_ground_;
        opt_ng.callback_group     = cb_group_ng_;

        pub_    = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic,    10);
        ng_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(ng_output_topic, 10);

        sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic, 5,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { groundCallback(msg); },
            opt_ground);

        ng_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            ng_input_topic, 5,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { ngCallback(msg); },
            opt_ng);

        RCLCPP_INFO(get_logger(),
            "ground_trans ready | fill_pts=%zu, r=%.2f, h=%.2f, res=%.2f",
            fill_pts_.size() / 3, fill_radius_, ground_height_, fill_resolution_);
    }

private:
    std::string source_frame_, target_frame_;
    double ground_height_, fill_radius_, fill_resolution_;

    // Flattened xyz of fill circle-disk points in source (base_link) frame
    std::vector<float> fill_pts_;

    tf2_ros::Buffer                  tf_buffer_;
    tf2_ros::TransformListener       tf_listener_;

    rclcpp::CallbackGroup::SharedPtr cb_group_ground_, cb_group_ng_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_, ng_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_, ng_pub_;

    // ── generate fill circle-disk points ─────────────────────────────────────
    void makeFillPoints()
    {
        const double d = fill_resolution_, r = fill_radius_, h = ground_height_;
        for (double x = -r; x <= r + d * 0.5; x += d) {
            for (double y = -r; y <= r + d * 0.5; y += d) {
                if (x * x + y * y <= r * r) {
                    fill_pts_.push_back(static_cast<float>(x));
                    fill_pts_.push_back(static_cast<float>(y));
                    fill_pts_.push_back(static_cast<float>(h));
                }
            }
        }
    }

    // ── non-blocking TF lookup ────────────────────────────────────────────────
    bool lookupTf(const std::string & source, const rclcpp::Time & stamp,
                  geometry_msgs::msg::TransformStamped & out)
    {
        const tf2::Duration zero = tf2::durationFromSec(0.0);
        try {
            out = tf_buffer_.lookupTransform(target_frame_, source, stamp, zero);
            return true;
        } catch (...) {}
        try {
            out = tf_buffer_.lookupTransform(target_frame_, source, tf2::TimePointZero, zero);
            return true;
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "TF unavailable [%s->%s]: %s", source.c_str(), target_frame_.c_str(), ex.what());
            return false;
        }
    }

    // ── build merged cloud (source cloud + fill points) ───────────────────────
    sensor_msgs::msg::PointCloud2 buildMergedCloud(
        const sensor_msgs::msg::PointCloud2 & msg)
    {
        const size_t n_src  = static_cast<size_t>(msg.width) * msg.height;
        const size_t n_fill = fill_pts_.size() / 3;
        const size_t n_total = n_src + n_fill;

        sensor_msgs::msg::PointCloud2 merged;
        merged.header     = msg.header;
        merged.height     = 1;
        merged.width      = static_cast<uint32_t>(n_total);
        merged.is_bigendian = false;
        merged.point_step = 12;
        merged.row_step   = static_cast<uint32_t>(n_total * 12);
        merged.is_dense   = false;

        merged.fields.resize(3);
        for (int i = 0; i < 3; ++i) {
            merged.fields[i].name     = std::string(1, "xyz"[i]);
            merged.fields[i].offset   = static_cast<uint32_t>(i * 4);
            merged.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
            merged.fields[i].count    = 1;
        }

        merged.data.resize(n_total * 12);
        uint8_t * dst = merged.data.data();

        // Copy xyz from each source point (handles arbitrary point_step)
        const uint8_t * src = msg.data.data();
        for (size_t i = 0; i < n_src; ++i) {
            std::memcpy(dst, src, 12);
            dst += 12;
            src += msg.point_step;
        }
        // Append fill points
        std::memcpy(dst, fill_pts_.data(), n_fill * 12);

        return merged;
    }

    // ── core: transform and publish ───────────────────────────────────────────
    void transformAndPublish(
        const sensor_msgs::msg::PointCloud2 & msg,
        bool append_fill,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub)
    {
        const std::string source = msg.header.frame_id.empty() ? source_frame_ : msg.header.frame_id;
        geometry_msgs::msg::TransformStamped tf_stamped;
        if (!lookupTf(source, rclcpp::Time(msg.header.stamp), tf_stamped)) {
            return;
        }

        const sensor_msgs::msg::PointCloud2 & cloud_to_transform =
            (append_fill && !fill_pts_.empty()) ? buildMergedCloud(msg) : msg;

        // Temporary non-const copy required by tf2::doTransform
        sensor_msgs::msg::PointCloud2 cloud_in = cloud_to_transform;
        sensor_msgs::msg::PointCloud2 cloud_out;
        tf2::doTransform(cloud_in, cloud_out, tf_stamped);
        cloud_out.header.frame_id = target_frame_;
        pub->publish(cloud_out);
    }

    void groundCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        transformAndPublish(*msg, /*append_fill=*/true, pub_);
    }

    void ngCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        transformAndPublish(*msg, /*append_fill=*/false, ng_pub_);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GroundTrans>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
