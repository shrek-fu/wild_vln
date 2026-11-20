// Subscribe a registered cloud or a Livox CustomMsg, transform it into map
// and/or base_link, and optionally drop points inside the robot body box.

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <tf2/time.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <rclcpp/qos.hpp>

using namespace std::chrono_literals;

class CloudTransformer : public rclcpp::Node
{
public:
    CloudTransformer()
    : Node("cloud_transformer"),
      tf_buffer_(get_clock()),
      tf_listener_(tf_buffer_, this, /*spin_thread=*/true)
    {
        declare_parameter("input_msg_type", "pointcloud2");
        declare_parameter("input_topic", "/cloud_registered");
        declare_parameter("publish_map", true);
        declare_parameter("publish_body", true);
        declare_parameter("qos_best_effort", true);
        declare_parameter("robot_type", "zsl1");
        declare_parameter("crop_body", false);
        declare_parameter("body_length", 0.0);
        declare_parameter("body_width", 0.0);
        declare_parameter("body_height", 0.0);

        std::string input_msg_type = get_parameter("input_msg_type").as_string();
        std::string input_topic = get_parameter("input_topic").as_string();
        bool publish_map = get_parameter("publish_map").as_bool();
        bool publish_body = get_parameter("publish_body").as_bool();
        bool qos_best_effort = get_parameter("qos_best_effort").as_bool();
        robot_type_ = get_parameter("robot_type").as_string();
        crop_body_ = get_parameter("crop_body").as_bool();
        body_length_ = get_parameter("body_length").as_double();
        body_width_ = get_parameter("body_width").as_double();
        body_height_ = get_parameter("body_height").as_double();

        rclcpp::QoS qos(10);
        if (qos_best_effort) {
            qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        } else {
            qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        }

        if (input_msg_type == "custom_msg") {
            sub_livox_ = create_subscription<livox_ros_driver2::msg::CustomMsg>(
                input_topic, 10,
                [this](livox_ros_driver2::msg::CustomMsg::SharedPtr msg) { livoxCloudCallback(msg); });
        } else {
            sub_pc2_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                input_topic, 10,
                [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloudCallback(msg); });
        }

        if (publish_map) {
            pub_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_in_map", qos);
        }
        if (publish_body) {
            pub_body_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_in_body", qos);
        }

        timer_ = create_wall_timer(100ms, [this]() { processCloud(); });

        RCLCPP_INFO(get_logger(),
            "cloud_transformer started | input_msg_type=%s input_topic=%s publish_map=%d publish_body=%d "
            "robot_type=%s crop_body=%d box=%.3f x %.3f x %.3f",
            input_msg_type.c_str(), input_topic.c_str(), publish_map, publish_body,
            robot_type_.c_str(), crop_body_, body_length_, body_width_, body_height_);
    }

private:
    sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_;
    bool processed_ = true;
    bool crop_body_ = false;
    double body_length_ = 0.0;
    double body_width_ = 0.0;
    double body_height_ = 0.0;
    std::string robot_type_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc2_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_body_;
    rclcpp::TimerBase::SharedPtr timer_;

    void cloudCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        latest_cloud_ = msg;
        processed_ = false;
    }

    void livoxCloudCallback(livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
        convertLivoxToPointCloud2(*msg, *cloud);
        latest_cloud_ = cloud;
        processed_ = false;
    }

    void convertLivoxToPointCloud2(
        const livox_ros_driver2::msg::CustomMsg & custom_msg,
        sensor_msgs::msg::PointCloud2 & cloud_out)
    {
        cloud_out.header = custom_msg.header;
        cloud_out.height = 1;

        sensor_msgs::PointCloud2Modifier modifier(cloud_out);
        modifier.setPointCloud2Fields(
            4,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::msg::PointField::FLOAT32);
        modifier.resize(custom_msg.point_num);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_out, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_out, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_out, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud_out, "intensity");

        for (uint32_t i = 0; i < custom_msg.point_num; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
            const auto & p = custom_msg.points[i];
            *iter_x = p.x;
            *iter_y = p.y;
            *iter_z = p.z;
            *iter_intensity = static_cast<float>(p.reflectivity);
        }
    }

    void processCloud()
    {
        if (!latest_cloud_ || processed_) return;
        processed_ = true;

        if (pub_map_) transformAndPublish(*latest_cloud_, "map", pub_map_, false);
        if (pub_body_) transformAndPublish(*latest_cloud_, "base_link", pub_body_, crop_body_);
    }

    void cropBody(sensor_msgs::msg::PointCloud2 & cloud) const
    {
        if (body_length_ <= 0.0) return;

        int x_off = -1;
        int y_off = -1;
        int z_off = -1;
        for (const auto & field : cloud.fields) {
            if (field.name == "x") x_off = static_cast<int>(field.offset);
            else if (field.name == "y") y_off = static_cast<int>(field.offset);
            else if (field.name == "z") z_off = static_cast<int>(field.offset);
        }
        if (x_off < 0 || y_off < 0 || z_off < 0 || cloud.point_step == 0) return;

        const float half_x = static_cast<float>(body_length_ * 0.5);
        const float half_y = static_cast<float>(body_width_ * 0.5);
        const float z_min = -0.05f;
        const float z_max = static_cast<float>(body_height_);
        const uint32_t step = cloud.point_step;
        const size_t count = static_cast<size_t>(cloud.width) * cloud.height;
        if (cloud.data.size() < count * step) return;

        std::vector<uint8_t> kept;
        kept.reserve(cloud.data.size());
        size_t kept_count = 0;
        for (size_t i = 0; i < count; ++i) {
            const uint8_t * point = cloud.data.data() + i * step;
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            std::memcpy(&x, point + x_off, sizeof(float));
            std::memcpy(&y, point + y_off, sizeof(float));
            std::memcpy(&z, point + z_off, sizeof(float));
            const bool inside =
                std::isfinite(x) && std::isfinite(y) && std::isfinite(z) &&
                x >= -half_x && x <= half_x &&
                y >= -half_y && y <= half_y &&
                z >= z_min && z <= z_max;
            if (inside) continue;
            kept.insert(kept.end(), point, point + step);
            ++kept_count;
        }

        cloud.data.swap(kept);
        cloud.width = static_cast<uint32_t>(kept_count);
        cloud.height = 1;
        cloud.row_step = cloud.point_step * cloud.width;
    }

    void transformAndPublish(
        const sensor_msgs::msg::PointCloud2 & cloud,
        const std::string & target_frame,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub,
        bool crop)
    {
        sensor_msgs::msg::PointCloud2 cloud_out;
        geometry_msgs::msg::TransformStamped tf_stamped;

        rclcpp::Time stamp = rclcpp::Time(cloud.header.stamp);
        try {
            tf_stamped = tf_buffer_.lookupTransform(
                target_frame, cloud.header.frame_id, stamp,
                tf2::durationFromSec(0.1));
        } catch (const tf2::ExtrapolationException &) {
            try {
                tf_stamped = tf_buffer_.lookupTransform(
                    target_frame, cloud.header.frame_id, tf2::TimePointZero,
                    tf2::durationFromSec(0.1));
            } catch (const tf2::TransformException & ex) {
                RCLCPP_ERROR(get_logger(), "TF lookup failed: %s", ex.what());
                return;
            }
        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(get_logger(), "TF lookup failed: %s", ex.what());
            return;
        }

        tf2::doTransform(cloud, cloud_out, tf_stamped);
        cloud_out.header.frame_id = target_frame;
        if (crop) cropBody(cloud_out);
        pub->publish(cloud_out);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CloudTransformer>());
    rclcpp::shutdown();
    return 0;
}
