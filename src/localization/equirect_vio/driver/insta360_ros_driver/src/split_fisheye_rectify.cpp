#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "cv_bridge/cv_bridge.h"
#include "insta360_ros_driver/msg/camera_preview_info.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

// Rectifies the front and rear lenses with the factory offset. The subscribed
// dual-fisheye frame is not modified. Each output has the pixel size of one
// lens in that frame (half the width, full height). The pinhole focal length
// is the factory equidistant focal length, so the field of view is fixed by
// the calibration and that size rather than by a separate setting.
//
// Offset layout, lens count first:
//   2_r_cx_cy_yaw_pitch_roll_r_cx_cy_yaw_pitch_roll_width_height_lensType
// The factory radius is the 90-degree ring: f = radius * 2 / pi.
namespace {

struct LensCalib {
    double radius = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
    double roll_deg = 0.0;
};

struct DualCalib {
    LensCalib lens[2];
    double width = 0.0;
    double height = 0.0;
    bool valid = false;
};

cv::Mat rotX(double rad) {
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, c, -s, 0, s, c);
}

cv::Mat rotY(double rad) {
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return (cv::Mat_<double>(3, 3) << c, 0, s, 0, 1, 0, -s, 0, c);
}

cv::Mat rotZ(double rad) {
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return (cv::Mat_<double>(3, 3) << c, -s, 0, s, c, 0, 0, 0, 1);
}

bool parseOffset(const std::string& text, DualCalib& out) {
    std::vector<double> values;
    std::string token;
    for (size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == '_') {
            if (!token.empty()) {
                try {
                    values.push_back(std::stod(token));
                } catch (const std::exception&) {
                    return false;
                }
                token.clear();
            }
        } else {
            token.push_back(text[i]);
        }
    }
    if (values.size() < 16 || static_cast<int>(values[0]) != 2) {
        return false;
    }
    for (int lens = 0; lens < 2; ++lens) {
        const int base = 1 + lens * 6;
        out.lens[lens].radius = values[base];
        out.lens[lens].cx = values[base + 1];
        out.lens[lens].cy = values[base + 2];
        out.lens[lens].yaw_deg = values[base + 3];
        out.lens[lens].pitch_deg = values[base + 4];
        out.lens[lens].roll_deg = values[base + 5];
    }
    out.width = values[13];
    out.height = values[14];
    out.valid = out.width > 1.0 && out.height > 1.0 &&
                out.lens[0].radius > 1.0 && out.lens[1].radius > 1.0;
    return out.valid;
}

}  // namespace

class SplitFisheyeRectifyNode : public rclcpp::Node {
public:
    SplitFisheyeRectifyNode() : Node("split_fisheye_rectify") {
        declare_parameter("input_topic", "/dual_fisheye/image");
        declare_parameter("calibration_topic", "/insta360/camera_preview_info");
        declare_parameter("front_topic", "/camera/front/image_rect");
        declare_parameter("rear_topic", "/camera/rear/image_rect");
        declare_parameter("front_info_topic", "/camera/front/camera_info");
        declare_parameter("rear_info_topic", "/camera/rear/camera_info");
        declare_parameter("front_fisheye_topic", "/camera/front/image_fisheye");
        declare_parameter("rear_fisheye_topic", "/camera/rear/image_fisheye");
        declare_parameter("front_fisheye_info_topic", "/camera/front/fisheye_info");
        declare_parameter("rear_fisheye_info_topic", "/camera/rear/fisheye_info");
        declare_parameter("front_on_right", true);

        const auto qos = rclcpp::QoS(1).reliable();
        front_pub_ = create_publisher<sensor_msgs::msg::Image>(
            get_parameter("front_topic").as_string(), qos);
        rear_pub_ = create_publisher<sensor_msgs::msg::Image>(
            get_parameter("rear_topic").as_string(), qos);
        front_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
            get_parameter("front_info_topic").as_string(), qos);
        rear_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
            get_parameter("rear_info_topic").as_string(), qos);
        front_fisheye_pub_ = create_publisher<sensor_msgs::msg::Image>(
            get_parameter("front_fisheye_topic").as_string(), qos);
        rear_fisheye_pub_ = create_publisher<sensor_msgs::msg::Image>(
            get_parameter("rear_fisheye_topic").as_string(), qos);
        front_fisheye_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
            get_parameter("front_fisheye_info_topic").as_string(), qos);
        rear_fisheye_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
            get_parameter("rear_fisheye_info_topic").as_string(), qos);

        rclcpp::QoS calib_qos(1);
        calib_qos.reliable().transient_local();
        calib_sub_ = create_subscription<insta360_ros_driver::msg::CameraPreviewInfo>(
            get_parameter("calibration_topic").as_string(), calib_qos,
            std::bind(&SplitFisheyeRectifyNode::onCalib, this, std::placeholders::_1));
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            get_parameter("input_topic").as_string(), qos,
            std::bind(&SplitFisheyeRectifyNode::onImage, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(),
                    "Factory-rectified front/rear images keep the subscribed lens size");
    }

private:
    struct LensMap {
        cv::Mat map_x;
        cv::Mat map_y;
        sensor_msgs::msg::CameraInfo info;
        sensor_msgs::msg::CameraInfo fisheye_info;
        int origin_x = 0;
        bool front = false;
    };

    void onCalib(const insta360_ros_driver::msg::CameraPreviewInfo::SharedPtr msg) {
        DualCalib parsed;
        bool ok = false;
        for (const auto& offset : msg->camera_offset) {
            if (parseOffset(offset, parsed)) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            RCLCPP_WARN(get_logger(), "No usable factory offset in camera_offset");
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        calib_ = parsed;
        maps_.clear();
        RCLCPP_INFO(get_logger(),
                    "Factory offset for %s: calib %.0fx%.0f, radii %.1f / %.1f",
                    msg->camera_name.c_str(), calib_.width, calib_.height,
                    calib_.lens[0].radius, calib_.lens[1].radius);
    }

    void buildMaps(int image_width, int image_height) {
        maps_.clear();
        const int out_w = image_width / 2;
        const int out_h = image_height;
        const double sx = image_width / calib_.width;
        const double sy = image_height / calib_.height;
        const bool front_on_right = get_parameter("front_on_right").as_bool();
        cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);

        for (int i = 0; i < 2; ++i) {
            const auto& lens = calib_.lens[i];
            const double cx = lens.cx * sx;
            const double cy = lens.cy * sy;
            const double radius = lens.radius * (sx + sy) / 2.0;
            const double f = radius * 2.0 / CV_PI;
            const double hfov_deg =
                2.0 * std::atan((out_w / 2.0) / f) * 180.0 / CV_PI;
            cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx, 0, f, cy, 0, 0, 1);
            cv::Mat P = (cv::Mat_<double>(3, 3) << f, 0, out_w / 2.0,
                         0, f, out_h / 2.0,
                         0, 0, 1);
            const double yaw = lens.yaw_deg * CV_PI / 180.0;
            const double pitch = lens.pitch_deg * CV_PI / 180.0;
            const double roll = lens.roll_deg * CV_PI / 180.0;
            cv::Mat R = rotZ(roll) * rotY(pitch) * rotX(yaw);

            LensMap lens_map;
            cv::fisheye::initUndistortRectifyMap(
                K, D, R, P, cv::Size(out_w, out_h), CV_32FC1,
                lens_map.map_x, lens_map.map_y);
            lens_map.front = front_on_right ? (cx >= image_width / 2.0) : (cx < image_width / 2.0);
            auto& info = lens_map.info;
            info.width = static_cast<uint32_t>(out_w);
            info.height = static_cast<uint32_t>(out_h);
            info.distortion_model = "plumb_bob";
            info.d = {0.0, 0.0, 0.0, 0.0, 0.0};
            info.k = {f, 0.0, out_w / 2.0, 0.0, f, out_h / 2.0, 0.0, 0.0, 1.0};
            info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            info.p = {f, 0.0, out_w / 2.0, 0.0, 0.0, f, out_h / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0};

            lens_map.origin_x = cx >= image_width / 2.0 ? out_w : 0;
            const double cx_fisheye = cx - lens_map.origin_x;
            auto& fisheye = lens_map.fisheye_info;
            fisheye.width = static_cast<uint32_t>(out_w);
            fisheye.height = static_cast<uint32_t>(out_h);
            fisheye.distortion_model = "equidistant";
            fisheye.d = {0.0, 0.0, 0.0, 0.0};
            fisheye.k = {f, 0.0, cx_fisheye, 0.0, f, cy, 0.0, 0.0, 1.0};
            fisheye.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            fisheye.p = {f, 0.0, cx_fisheye, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
            RCLCPP_INFO(get_logger(),
                        "%s fisheye %dx%d equidistant f=%.1f cx=%.1f cy=%.1f; pinhole hfov=%.1f deg",
                        lens_map.front ? "front" : "rear", out_w, out_h, f, cx_fisheye, cy, hfov_deg);
            maps_.push_back(std::move(lens_map));
        }
        map_width_ = image_width;
        map_height_ = image_height;
    }

    void onImage(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImageConstPtr image;
        try {
            image = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        } catch (const cv_bridge::Exception& ex) {
            RCLCPP_ERROR(get_logger(), "cv_bridge: %s", ex.what());
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!calib_.valid) {
            if (!warned_calib_) {
                RCLCPP_WARN(get_logger(), "Waiting for /insta360/camera_preview_info");
                warned_calib_ = true;
            }
            return;
        }
        if (maps_.empty() || map_width_ != image->image.cols || map_height_ != image->image.rows) {
            buildMaps(image->image.cols, image->image.rows);
        }

        for (const auto& lens_map : maps_) {
            cv::Mat rectified;
            cv::remap(image->image, rectified, lens_map.map_x, lens_map.map_y,
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            cv_bridge::CvImage out;
            out.header = msg->header;
            out.header.frame_id = lens_map.front ? "camera_front" : "camera_rear";
            out.encoding = sensor_msgs::image_encodings::BGR8;
            out.image = rectified;
            auto info = lens_map.info;
            info.header = out.header;
            (lens_map.front ? front_pub_ : rear_pub_)->publish(*out.toImageMsg());
            (lens_map.front ? front_info_pub_ : rear_info_pub_)->publish(info);

            cv_bridge::CvImage fisheye_image;
            fisheye_image.header = msg->header;
            fisheye_image.header.frame_id = lens_map.front ? "camera_front_fisheye" : "camera_rear_fisheye";
            fisheye_image.encoding = sensor_msgs::image_encodings::BGR8;
            fisheye_image.image = image->image(
                cv::Rect(lens_map.origin_x, 0, lens_map.map_x.cols, lens_map.map_x.rows)).clone();
            auto fisheye_info = lens_map.fisheye_info;
            fisheye_info.header = fisheye_image.header;
            (lens_map.front ? front_fisheye_pub_ : rear_fisheye_pub_)
                ->publish(*fisheye_image.toImageMsg());
            (lens_map.front ? front_fisheye_info_pub_ : rear_fisheye_info_pub_)->publish(fisheye_info);
        }
    }

    DualCalib calib_;
    std::vector<LensMap> maps_;
    int map_width_ = 0;
    int map_height_ = 0;
    bool warned_calib_ = false;
    std::mutex mutex_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr front_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rear_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr front_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rear_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr front_fisheye_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rear_fisheye_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr front_fisheye_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rear_fisheye_info_pub_;
    rclcpp::Subscription<insta360_ros_driver::msg::CameraPreviewInfo>::SharedPtr calib_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SplitFisheyeRectifyNode>());
    rclcpp::shutdown();
    return 0;
}
