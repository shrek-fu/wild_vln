#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <atomic>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <mutex>

#include <camera/camera.h>
#include <camera/photography_settings.h>
#include <camera/device_discovery.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "factory_calibration.hpp"
#include "insta360_ros_driver/msg/camera_preview_info.hpp"
#include "insta360_ros_driver/msg/factory_calibration.hpp"

// Maps a device timestamp stream onto ROS time. The camera delivers IMU in ~100 ms
// bursts, so arrival time cannot be used as the sample time. The unit (ns/us/ms) is
// detected from the sample spacing; the offset tracks the minimum observed latency.
class DeviceClock {
public:
    explicit DeviceClock(double nominal_period_s) : nominal_period_s_(nominal_period_s) {}

    bool ready() const { return scale_ > 0.0; }
    double scale() const { return scale_; }
    double offset() const { return offset_; }

    // Returns false until the unit is known.
    bool update(int64_t device_ts, double now_s) {
        if (scale_ <= 0.0) {
            if (have_prev_ && device_ts > prev_ts_) {
                const double dt = static_cast<double>(device_ts - prev_ts_);
                for (double s : {1e-9, 1e-6, 1e-3}) {
                    const double period = dt * s;
                    if (period > 0.2 * nominal_period_s_ && period < 20.0 * nominal_period_s_) {
                        scale_ = s;
                        break;
                    }
                }
            }
            prev_ts_ = device_ts;
            have_prev_ = true;
            if (scale_ <= 0.0) {
                return false;
            }
        }
        const double latency = now_s - static_cast<double>(device_ts) * scale_;
        if (!have_offset_) {
            offset_ = latency;
            have_offset_ = true;
            last_now_ = now_s;
        } else {
            // Relax upward slowly (clock drift), snap down to a lower latency at once.
            offset_ += kRelaxPerSecond * std::max(0.0, now_s - last_now_);
            last_now_ = now_s;
            offset_ = std::min(offset_, latency);
        }
        return true;
    }

    double to_ros(int64_t device_ts, double offset) const {
        return static_cast<double>(device_ts) * scale_ + offset;
    }

private:
    static constexpr double kRelaxPerSecond = 1e-4;
    double nominal_period_s_;
    double scale_ = 0.0;
    double offset_ = 0.0;
    double last_now_ = 0.0;
    int64_t prev_ts_ = 0;
    bool have_prev_ = false;
    bool have_offset_ = false;
};

class TestStreamDelegate : public ins_camera::StreamDelegate {
private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    std::mutex clock_mutex_;
    DeviceClock imu_clock_{0.005};
    DeviceClock video_clock_{1.0 / 30.0};
    bool use_device_time_ = true;
    double last_imu_stamp_ = 0.0;
    int64_t last_report_ns_ = 0;

    static builtin_interfaces::msg::Time to_msg(double t) {
        return rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME);
    }

public:
    TestStreamDelegate(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {
        use_device_time_ = node_->declare_parameter<bool>("use_device_time", true);

        // Publisher for the compressed H.264 video stream
        compressed_pub_ = node_->create_publisher<sensor_msgs::msg::CompressedImage>(
            "/dual_fisheye/image/compressed", 
            rclcpp::QoS(10)
        );

        // IMU arrives in bursts of ~20 samples; a depth-5 queue drops most of each burst.
        imu_pub_ = node_->create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", rclcpp::QoS(2000));
        RCLCPP_INFO(node_->get_logger(), "Publisher for compressed images and IMU created (use_device_time=%d).",
                    use_device_time_ ? 1 : 0);
    }

    virtual ~TestStreamDelegate() {}

    void OnAudioData(const uint8_t* data, size_t size, int64_t timestamp) override {}

    void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t streamType, int stream_index) override {
        // We only care about the main video stream (index 0)
        if (stream_index == 0 && size > 0 && compressed_pub_) {
            auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();

            const rclcpp::Time now = node_->get_clock()->now();
            msg->header.stamp = now;
            if (use_device_time_) {
                std::lock_guard<std::mutex> lock(clock_mutex_);
                if (video_clock_.update(timestamp, now.seconds()) && imu_clock_.ready()) {
                    // Same device clock as the IMU when the two offsets agree; then the IMU
                    // offset (lower transport latency) aligns the frame with the IMU samples.
                    const double imu_based = video_clock_.to_ros(timestamp, imu_clock_.offset());
                    const bool shared = video_clock_.scale() == imu_clock_.scale() &&
                                        std::abs(video_clock_.offset() - imu_clock_.offset()) < 0.5;
                    const double t = shared ? imu_based : video_clock_.to_ros(timestamp, video_clock_.offset());
                    msg->header.stamp = to_msg(t);
                    const int64_t now_ns = now.nanoseconds();
                    if (now_ns - last_report_ns_ > 10'000'000'000LL) {
                        last_report_ns_ = now_ns;
                        RCLCPP_INFO(node_->get_logger(),
                                    "Device clock: imu scale=%g offset=%.4f | video scale=%g offset=%.4f | "
                                    "shared=%d video_latency=%.1f ms",
                                    imu_clock_.scale(), imu_clock_.offset(), video_clock_.scale(),
                                    video_clock_.offset(), shared ? 1 : 0, 1e3 * (now.seconds() - t));
                    }
                }
            }
            msg->header.frame_id = "camera_frame";

            // Set the format to H.264
            // The subscriber will need to know this to select the correct decoder.
            msg->format = "h264";

            // Copy the compressed video data directly into the message
            msg->data.assign(data, data + size);

            compressed_pub_->publish(std::move(msg));
        }
    }

    void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
        const rclcpp::Time now = node_->get_clock()->now();
        for (const auto& gyro : data) {
            auto msg = std::make_unique<sensor_msgs::msg::Imu>();
            msg->header.stamp = now;
            if (use_device_time_) {
                std::lock_guard<std::mutex> lock(clock_mutex_);
                if (!imu_clock_.update(gyro.timestamp, now.seconds())) {
                    continue;
                }
                const double t = imu_clock_.to_ros(gyro.timestamp, imu_clock_.offset());
                if (t <= last_imu_stamp_) {
                    continue;
                }
                last_imu_stamp_ = t;
                msg->header.stamp = to_msg(t);
            }
            msg->header.frame_id = "imu_frame";
            msg->angular_velocity.x = gyro.gx;
            msg->angular_velocity.y = gyro.gy;
            msg->angular_velocity.z = gyro.gz;
            
            msg->linear_acceleration.x = gyro.ax * 9.80665;
            msg->linear_acceleration.y = gyro.ay * 9.80665;
            msg->linear_acceleration.z = gyro.az * 9.80665;

            msg->orientation.x = 0.0;
            msg->orientation.y = 0.0;
            msg->orientation.z = 0.0;
            msg->orientation.w = 1.0; // Neutral orientation
            msg->orientation_covariance[0] = -1.0; // No orientation data available

            for (int i = 0; i < 9; i++)
            {
                msg->angular_velocity_covariance[i] = 0;
                msg->linear_acceleration_covariance[i] = 0;
            }
            imu_pub_->publish(std::move(msg));
        }
    }

    void OnExposureData(const ins_camera::ExposureData& data) override {}
};

class CameraWrapper {
private:
    std::shared_ptr<ins_camera::Camera> cam;
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Publisher<insta360_ros_driver::msg::CameraPreviewInfo>::SharedPtr metadata_pub_;
    rclcpp::Publisher<insta360_ros_driver::msg::FactoryCalibration>::SharedPtr factory_pub_;

public:
    CameraWrapper(const std::shared_ptr<rclcpp::Node>& node) : node_(node) {}

    ~CameraWrapper() {
        if (cam) {
            cam->Close();
        }
    }

    int run_camera() {
        // CameraSDK reads ./jsons/camera_conf_<model>.json, including X4, from the cwd.
        if (!std::filesystem::exists("jsons")) {
#ifdef CAMERA_SDK_BIN_DIR
            const std::filesystem::path sdk_bin{CAMERA_SDK_BIN_DIR};
            if (std::filesystem::exists(sdk_bin / "jsons")) {
                std::filesystem::current_path(sdk_bin);
                RCLCPP_INFO(node_->get_logger(), "Using CameraSDK configs from %s", sdk_bin.string().c_str());
            }
#endif
        }

        ins_camera::DeviceDiscovery discovery;

        rclcpp::QoS metadata_qos(1);
        metadata_qos.reliable().transient_local();
        metadata_pub_ = node_->create_publisher<insta360_ros_driver::msg::CameraPreviewInfo>(
            "/insta360/camera_preview_info", metadata_qos);
        factory_pub_ = node_->create_publisher<insta360_ros_driver::msg::FactoryCalibration>(
            "/insta360/factory_calibration", metadata_qos);

        auto list = discovery.GetAvailableDevices();
        if (list.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "No available camera devices found.");
            return -1;
        }

        cam = std::make_shared<ins_camera::Camera>(list[0].info);
        if (!cam->Open()) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to open camera.");
            return -1;
        }
        RCLCPP_INFO(node_->get_logger(), "Camera opened successfully.");

        auto preview = cam->GetPreviewParam();

        auto metadata = std::make_unique<insta360_ros_driver::msg::CameraPreviewInfo>();
        metadata->header.stamp = node_->get_clock()->now();
        metadata->header.frame_id = "camera_frame";
        metadata->camera_name = preview.camera_name;
        metadata->decode_type = preview.encode_type == ins_camera::VideoEncodeType::H265 ? "h265" : "h264";
        metadata->crop_src_width = preview.GetCropSrcWidth();
        metadata->crop_src_height = preview.GetCropSrcHeight();
        metadata->crop_dst_width = preview.GetCropDstWidth();
        metadata->crop_dst_height = preview.GetCropDstHeight();
        metadata->crop_offset_x = preview.GetCropOffsetX();
        metadata->crop_offset_y = preview.GetCropOffsetY();
        std::vector<std::string> metadata_offsets;
        for (size_t i = 0; i < preview.GetCalibrationCount(); ++i) {
            metadata_offsets.push_back(preview.GetCalibration(i));
        }
        metadata->camera_offset = metadata_offsets;
        metadata_pub_->publish(std::move(metadata));

        auto factory = std::make_unique<insta360_ros_driver::msg::FactoryCalibration>();
        factory->header.stamp = node_->get_clock()->now();
        factory->header.frame_id = "imu_frame";
        factory->camera_name = preview.camera_name;
        factory->delay_timestamp = preview.delay_timestamp;
        factory->sweep_time = preview.sweep_time;
        factory->acceleration_range = preview.acceleration_range;
        factory->gyro_range = preview.gyro_range;
        if (!fillFactoryCalibration(metadata_offsets, *factory)) {
            RCLCPP_WARN(node_->get_logger(), "Factory offset v1 was not parsed");
        }
        RCLCPP_INFO(node_->get_logger(),
                    "Factory lenses r=%.1f/%.1f cx=%.1f/%.1f yaw=%.3f/%.3f pitch=%.3f/%.3f roll=%.3f/%.3f",
                    factory->radius[0], factory->radius[1], factory->cx[0], factory->cx[1],
                    factory->yaw_deg[0], factory->yaw_deg[1], factory->pitch_deg[0], factory->pitch_deg[1],
                    factory->roll_deg[0], factory->roll_deg[1]);
        if (factory->has_v2) {
            RCLCPP_INFO(node_->get_logger(),
                        "Factory v2 translation lens0=(%.5f, %.5f, %.5f) lens1=(%.5f, %.5f, %.5f)",
                        factory->v2_tx[0], factory->v2_ty[0], factory->v2_tz[0],
                        factory->v2_tx[1], factory->v2_ty[1], factory->v2_tz[1]);
        }
        RCLCPP_INFO(node_->get_logger(),
                    "IMU timing delay_timestamp=%ld sweep_time=%ld gyro_range=%d accel_range=%d",
                    static_cast<long>(factory->delay_timestamp), static_cast<long>(factory->sweep_time),
                    factory->gyro_range, factory->acceleration_range);
        factory_pub_->publish(std::move(factory));
        RCLCPP_INFO(node_->get_logger(), "Published camera preview metadata for %s", preview.camera_name.c_str());
        discovery.FreeDeviceDescriptors(list);

        std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<TestStreamDelegate>(node_);
        cam->SetStreamDelegate(delegate);

        auto start = time(NULL);

        uint64_t utc_time = static_cast<uint64_t>(start);
        uint32_t offset_time = 0; //no offset from UTC

        cam->SyncLocalTimeToCamera(utc_time,offset_time);       
        ins_camera::LiveStreamParam param;
        param.video_resolution = ins_camera::VideoResolution::RES_1920_960P30; //Change this line to edit the resolution
        //Possible resolutions (results may vary per model) are:
        //RES_3840_1920P30
        //RES_2560_1280P30
        //RES_1152_1152P30 (this will give 2304 x 1152 at 30 FPS)
        //RES_1920_960P30  
        param.lrv_video_resulution = ins_camera::VideoResolution::RES_1440_720P30;
        param.video_bitrate = 1024 * 1024 / 2;
        param.enable_audio = false;
        param.using_lrv = false;

        if (!cam->StartLiveStreaming(param)) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to start live streaming.");
            return -1;
        }
        
        RCLCPP_INFO(node_->get_logger(), "Live streaming started.");
        return 0;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("insta_publisher");
    
    CameraWrapper camera(node);
    if (camera.run_camera() != 0) {
        rclcpp::shutdown();
        return -1;
    }
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}