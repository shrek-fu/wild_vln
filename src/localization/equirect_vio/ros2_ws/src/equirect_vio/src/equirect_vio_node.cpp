// SPDX-License-Identifier: MIT
// Live ROS 2 node: feed equirectangular images + IMU into OpenVIO.

#include "vio/geometry/lie_group.h"
#include "vio/pipeline/config_loader.h"
#include "vio/odometry/visual_inertial_odometry.h"
#include "vio/util/types.h"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <string>

class EquirectVioNode : public rclcpp::Node {
 public:
  EquirectVioNode() : Node("equirect_vio") {
    const std::string config_path = declare_parameter<std::string>("config", "");
    // Defaults: stitched equirectangular + IMU from x4_air_equirect.launch.xml
    image_topic_ = declare_parameter<std::string>("image_topic", "/equirectangular/image");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data_raw");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "imu");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    max_path_size_ = declare_parameter<int>("max_path_size", 5000);
    // Kept for compatibility; gate uses CaptureTime only (OpenVIO offline rule).
    imu_queue_margin_s_ = declare_parameter<double>("imu_queue_margin_s", 0.0);
    max_pending_images_ = declare_parameter<int>("max_pending_images", 8);
    warmup_updated_frames_ = declare_parameter<int>("warmup_updated_frames", 10);
    reset_origin_on_ready_ = declare_parameter<bool>("reset_origin_on_ready", true);
    reset_on_divergence_ = declare_parameter<bool>("reset_on_divergence", true);
    max_speed_ = declare_parameter<double>("max_speed", 10.0);
    stall_timeout_s_ = declare_parameter<double>("stall_timeout_s", 3.0);

    if (config_path.empty()) {
      throw std::runtime_error("Parameter 'config' (OpenVIO YAML path) is required");
    }

    vio::ConfigLoader loader;
    if (!loader.Read(config_path) || !loader.Check()) {
      throw std::runtime_error("Failed to load OpenVIO config: " + config_path);
    }
    options_ = loader.odometry;
    odometry_ = std::make_unique<vio::VisualInertialOdometry>(options_);

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("~/odometry", 50);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("~/path", 10);
    publish_feature_image_ = declare_parameter<bool>("publish_feature_image", true);
    feature_image_scale_ = declare_parameter<double>("feature_image_scale", 0.5);
    feature_pub_ = create_publisher<sensor_msgs::msg::Image>("~/feature_image", rclcpp::SensorDataQoS());
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, rclcpp::QoS(2000).best_effort(),
        std::bind(&EquirectVioNode::ImuCallback, this, std::placeholders::_1));
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&EquirectVioNode::ImageCallback, this, std::placeholders::_1));

    path_.header.frame_id = odom_frame_;

    RCLCPP_INFO(get_logger(), "OpenVIO equirect VIO ready");
    RCLCPP_INFO(get_logger(), "  config: %s", config_path.c_str());
    RCLCPP_INFO(get_logger(), "  image:  %s (%dx%d)", image_topic_.c_str(), options_.camera.width,
                options_.camera.height);
    RCLCPP_INFO(get_logger(), "  imu:    %s", imu_topic_.c_str());
  }

 private:
  struct PendingImage {
    double timestamp = 0.0;
    builtin_interfaces::msg::Time stamp;
    cv::Mat gray;
  };

  void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    vio::ImuSample sample;
    sample.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    sample.gyro = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    sample.accel =
        Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

    std::lock_guard<std::mutex> lock(mutex_);
    imu_buffer_.push_back(sample);
    while (!imu_buffer_.empty() && (sample.timestamp - imu_buffer_.front().timestamp) > 30.0) {
      imu_buffer_.pop_front();
    }
    TryProcessPendingImagesLocked();
  }

  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
        cv_ptr = cv_bridge::toCvShare(msg, "mono8");
      } else {
        // MediaSDK publishes rgba8/bgra8; DIY stitcher uses rgb8/bgr8.
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
      }
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    PendingImage pending;
    pending.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    pending.stamp = msg->header.stamp;
    if (cv_ptr->image.channels() == 1) {
      pending.gray = cv_ptr->image.clone();
    } else {
      cv::cvtColor(cv_ptr->image, pending.gray, cv::COLOR_BGR2GRAY);
    }
    if (pending.gray.cols != options_.camera.width || pending.gray.rows != options_.camera.height) {
      cv::resize(pending.gray, pending.gray, cv::Size(options_.camera.width, options_.camera.height), 0, 0,
                 cv::INTER_AREA);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    image_buffer_.push_back(std::move(pending));
    while (static_cast<int>(image_buffer_.size()) > max_pending_images_) {
      image_buffer_.pop_front();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000.0,
                           "Dropping old image: IMU lag (pending>%d)", max_pending_images_);
    }
    const double wall = now().seconds();
    if (last_processed_wall_ < 0.0) {
      last_processed_wall_ = wall;
    }
    TryProcessPendingImagesLocked();
    // Images keep arriving but none gets processed: the sync gate is deadlocked (e.g. CaptureTime far ahead of
    // the IMU after a divergence). Start over instead of dropping frames forever.
    if (reset_on_divergence_ && odometry_->estimator().IsInitialized() &&
        wall - last_processed_wall_ > stall_timeout_s_) {
      ResetVioLocked("no image processed for " + std::to_string(wall - last_processed_wall_) + " s");
    }
  }

  // Requires mutex_. Fresh OpenVIO instance: it re-initializes from the next images (hold still, then move).
  void ResetVioLocked(const std::string& reason) {
    const auto& est = odometry_->estimator();
    RCLCPP_ERROR(get_logger(),
                 "Resetting VIO: %s (time_offset=%.3f s sensor_offset=%.3f s |v|=%.2f m/s). "
                 "Re-initializing: hold still ~2 s, then move.",
                 reason.c_str(), est.rig().time_offset, est.rig().sensor_time_offset, est.velocity().norm());
    odometry_ = std::make_unique<vio::VisualInertialOdometry>(options_);
    image_buffer_.clear();
    last_imu_fed_ = -1.0;
    output_ready_ = false;
    updated_frames_ = 0;
    origin_.setZero();
    path_.poses.clear();
    last_processed_wall_ = now().seconds();
    ++num_resets_;
  }

  // Requires mutex_. Divergence checks after a processed image.
  bool CheckHealthLocked() {
    const auto& est = odometry_->estimator();
    if (!est.IsInitialized()) {
      return true;
    }
    if (est.time_offset_clamped()) {
      ResetVioLocked("time offset hit the clamp (" + std::to_string(options_.estimator.max_time_offset) + " s)");
      return false;
    }
    if (!est.position().allFinite() || !est.velocity().allFinite()) {
      ResetVioLocked("non-finite state");
      return false;
    }
    if (est.velocity().norm() > max_speed_) {
      ResetVioLocked("speed above " + std::to_string(max_speed_) + " m/s");
      return false;
    }
    return true;
  }

  // Requires mutex_.
  // Image-led sync: for each image stamp, wait until an IMU sample on the camera clock is
  // past CaptureTime, then AddImu through that first-past sample (nearest IMU that closes
  // the OpenVIO propagation interval — same rule as odometry_pipeline offline).
  // After at-rest init, estimator time_ sits ~0.5*window before the init image; drop any
  // queued frame whose capture is not strictly after that time. Soft propagation failures
  // drop the frame and continue (do not abort the live stream).
  void TryProcessPendingImagesLocked() {
    while (!image_buffer_.empty()) {
      const PendingImage& img = image_buffer_.front();
      const double capture = odometry_->CaptureTime(img.timestamp);
      const bool initialized = odometry_->estimator().IsInitialized();

      // Skip frames that cannot advance the filter (at-rest older_end / duplicate stamps).
      if (initialized) {
        const double est_time = odometry_->estimator().timestamp();
        if (capture <= est_time) {
          RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000.0,
              "Skipping stale image (img=%.3f capture=%.3f est_time=%.3f pending=%zu)", img.timestamp, capture,
              est_time, image_buffer_.size());
          image_buffer_.pop_front();
          continue;
        }
      }

      double newest_imu = last_imu_fed_;
      if (!imu_buffer_.empty()) {
        newest_imu = std::max(newest_imu, imu_buffer_.back().timestamp);
      }
      if (odometry_->ToCameraClock(newest_imu) <= capture) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000.0,
                             "Waiting for IMU past image (img=%.3f capture=%.3f newest_imu=%.3f pending=%zu)",
                             img.timestamp, capture, newest_imu, image_buffer_.size());
        return;
      }

      // Feed IMU up through the first sample past capture (image-timestamp → covering IMU).
      while (!imu_buffer_.empty()) {
        const vio::ImuSample sample = imu_buffer_.front();
        imu_buffer_.pop_front();
        odometry_->AddImu(sample);
        last_imu_fed_ = sample.timestamp;
        if (odometry_->ToCameraClock(sample.timestamp) > capture) {
          break;
        }
      }

      PendingImage done = std::move(image_buffer_.front());
      image_buffer_.pop_front();

      if (!odometry_->ProcessImage(done.timestamp, done.gray, image_index_++)) {
        const std::string& reason = odometry_->failure();
        if (reason.find("propagation not forward") != std::string::npos ||
            reason.find("IMU does not cover") != std::string::npos) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000.0,
                               "Dropping frame after soft OpenVIO failure: %s (img=%.3f capture=%.3f est=%.3f)",
                               reason.c_str(), done.timestamp, odometry_->CaptureTime(done.timestamp),
                               odometry_->estimator().timestamp());
          continue;
        }
        RCLCPP_ERROR(get_logger(), "OpenVIO stopped: %s", reason.c_str());
        if (reset_on_divergence_) {
          ResetVioLocked("estimator failure: " + reason);
        }
        return;
      }

      last_processed_wall_ = now().seconds();
      if (reset_on_divergence_ && !CheckHealthLocked()) {
        return;
      }

      const auto& report = odometry_->report();
      if (!odometry_->estimator().IsInitialized()) {
        const auto& init = report.initialization;
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000.0,
            "Processed frame %d feats=%zu | init method=%s success=%d tracks=%d "
            "disp_old=%.1f/%d disp_new=%.1f/%d rot=%.1fdeg | %s",
            image_index_, report.observations.size(), init.method.empty() ? "-" : init.method.c_str(),
            init.success ? 1 : 0, init.num_tracks, init.disparity_old, init.disparity_tracks_old,
            init.disparity_new, init.disparity_tracks_new, init.rotation_deg,
            init.reason.empty() ? "(waiting)" : init.reason.c_str());
      } else {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000.0,
                             "Processed frame %d (has_state=%d pending=%zu est=%.3f)", image_index_,
                             report.has_state ? 1 : 0, image_buffer_.size(),
                             odometry_->estimator().timestamp());
      }

      if (report.initialized) {
        RCLCPP_INFO(get_logger(),
                    "VIO initialized at image_t=%.3f state_t=%.3f (%s); flushing %zu pre-init queued frames",
                    report.timestamp, odometry_->estimator().timestamp(), report.initialization.method.c_str(),
                    image_buffer_.size());
        // Queued frames may sit at/behind at-rest older_end; wait for fresh images.
        image_buffer_.clear();
      }
      if (report.has_state && !output_ready_) {
        // Warm-up after initialization: wait for a full clone window and a few frames of accepted visual updates
        // before anything leaves the node, so the trajectory does not start with the unconstrained IMU-only jump.
        const auto& est = odometry_->estimator();
        if (est.statistics().accepted > 0) {
          ++updated_frames_;
        }
        if (est.NumClones() >= options_.estimator.max_clones && updated_frames_ >= warmup_updated_frames_) {
          output_ready_ = true;
          path_.poses.clear();
          if (reset_origin_on_ready_) {
            origin_ = est.position();
          }
          RCLCPP_INFO(get_logger(), "VIO warm-up done (%d clones, %d updated frames): publishing odometry",
                      est.NumClones(), updated_frames_);
        }
      }
      if (report.has_state && output_ready_) {
        PublishPose(done.stamp);
      }
      PublishFeatureImage(done);
    }
  }

  // Tracked features over the image, colored like the OpenVIO viewer:
  // magenta landmark, green accepted, red gate-rejected, orange triangulation failed,
  // grey too few measurements, light blue tracking only (before an update / initializing).
  void PublishFeatureImage(const PendingImage& frame) {
    if (!publish_feature_image_ || feature_pub_->get_subscription_count() == 0) {
      return;
    }
    const auto& report = odometry_->report();
    const auto& est = odometry_->estimator();
    std::unordered_map<int, vio::TrackOutcome> outcomes;
    if (report.has_state) {
      for (const auto& track : est.track_reports()) {
        outcomes[track.id] = track.outcome;
      }
    }
    auto color_of = [](vio::TrackOutcome outcome) {
      switch (outcome) {
        case vio::TrackOutcome::kAccepted: return cv::Scalar(90, 230, 60);
        case vio::TrackOutcome::kGate: return cv::Scalar(60, 60, 255);
        case vio::TrackOutcome::kTriangulation:
        case vio::TrackOutcome::kProjection: return cv::Scalar(40, 190, 255);
        case vio::TrackOutcome::kLandmark: return cv::Scalar(240, 90, 240);
        default: return cv::Scalar(150, 150, 150);
      }
    };

    cv::Mat canvas;
    cv::cvtColor(frame.gray, canvas, cv::COLOR_GRAY2BGR);
    const double s = feature_image_scale_;
    if (s != 1.0) {
      cv::resize(canvas, canvas, cv::Size(), s, s, cv::INTER_AREA);
    }
    const int radius = std::max(2, static_cast<int>(std::lround(4 * s)));
    int n_landmark = 0;
    int n_accepted = 0;
    for (const auto& obs : report.observations) {
      const auto it = outcomes.find(obs.id);
      const cv::Scalar color = it == outcomes.end() ? cv::Scalar(255, 200, 120) : color_of(it->second);
      if (it != outcomes.end()) {
        n_landmark += it->second == vio::TrackOutcome::kLandmark;
        n_accepted += it->second == vio::TrackOutcome::kAccepted;
      }
      cv::circle(canvas, cv::Point2d(obs.pixel.x() * s, obs.pixel.y() * s), radius, color, -1, cv::LINE_AA);
    }

    std::string status;
    if (report.has_state) {
      status = "tracking  feats=" + std::to_string(report.observations.size()) +
               " landmarks=" + std::to_string(n_landmark) + " accepted=" + std::to_string(n_accepted);
    } else if (est.IsInitialized()) {
      status = report.initialized ? "initialized (" + report.initialization.method + ")"
                                  : "estimator: " + est.failure();
    } else {
      status = "initializing: " + (report.initialization.reason.empty() ? std::string("waiting for data")
                                                                        : report.initialization.reason);
    }
    const double font = std::max(0.4, 0.6 * canvas.cols / 960.0);
    const cv::Point origin(8, static_cast<int>(std::lround(36 * font)));
    cv::putText(canvas, status, origin, cv::FONT_HERSHEY_SIMPLEX, font, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(canvas, status, origin, cv::FONT_HERSHEY_SIMPLEX, font, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    std_msgs::msg::Header header;
    header.stamp = frame.stamp;
    header.frame_id = "camera";
    feature_pub_->publish(*cv_bridge::CvImage(header, "bgr8", canvas).toImageMsg());
  }

  void PublishPose(const builtin_interfaces::msg::Time& stamp) {
    const auto& est = odometry_->estimator();
    const Eigen::Quaterniond q_ItoG(vio::jpl::ToRotation(est.orientation()).transpose());
    const Eigen::Vector3d p = est.position() - origin_;
    const Eigen::Vector3d& v = est.velocity();

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = p.x();
    odom.pose.pose.position.y = p.y();
    odom.pose.pose.position.z = p.z();
    odom.pose.pose.orientation.x = q_ItoG.x();
    odom.pose.pose.orientation.y = q_ItoG.y();
    odom.pose.pose.orientation.z = q_ItoG.z();
    odom.pose.pose.orientation.w = q_ItoG.w();
    odom.twist.twist.linear.x = v.x();
    odom.twist.twist.linear.y = v.y();
    odom.twist.twist.linear.z = v.z();
    odom_pub_->publish(odom);

    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = odom.header;
    pose_stamped.pose = odom.pose.pose;
    path_.header = odom.header;
    path_.poses.push_back(pose_stamped);
    while (static_cast<int>(path_.poses.size()) > max_path_size_) {
      path_.poses.erase(path_.poses.begin());
    }
    path_pub_->publish(path_);

    if (publish_tf_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = odom.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = p.x();
      tf.transform.translation.y = p.y();
      tf.transform.translation.z = p.z();
      tf.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  std::string image_topic_;
  std::string imu_topic_;
  std::string odom_frame_;
  std::string base_frame_;
  bool publish_tf_ = true;
  int max_path_size_ = 5000;
  double imu_queue_margin_s_ = 0.01;
  int max_pending_images_ = 8;

  vio::VisualInertialOdometryOptions options_;
  std::unique_ptr<vio::VisualInertialOdometry> odometry_;

  std::mutex mutex_;
  std::deque<vio::ImuSample> imu_buffer_;
  std::deque<PendingImage> image_buffer_;
  double last_imu_fed_ = -1.0;
  int image_index_ = 0;
  int warmup_updated_frames_ = 10;
  int updated_frames_ = 0;
  bool output_ready_ = false;
  bool reset_origin_on_ready_ = true;
  bool reset_on_divergence_ = true;
  double max_speed_ = 10.0;
  double stall_timeout_s_ = 3.0;
  double last_processed_wall_ = -1.0;
  int num_resets_ = 0;
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feature_pub_;
  bool publish_feature_image_ = true;
  double feature_image_scale_ = 0.5;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  nav_msgs::msg::Path path_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<EquirectVioNode>());
  } catch (const std::exception& e) {
    fprintf(stderr, "equirect_vio_node failed: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
