// Insta360 X4 Air live views over USB.
// Camera SDK preview stream -> Media SDK RealTimeStitcher (equirectangular)
// -> front, rear, left and right pinhole views as JPEG CompressedImage.
//
// No OpenCV here: libMediaSDK.so loads its own opencv_world 4.7, and a
// second OpenCV in the same process is not safe.
#include <camera/camera.h>
#include <camera/device_discovery.h>
#include <ins_realtime_stitcher.h>
#include <ins_stitcher.h>
#include <jpeglib.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

using namespace std::chrono_literals;

namespace {

// FFmpeg AVPixelFormat values reported by the stitcher callback.
constexpr int kAvPixFmtYuv420p = 0;
constexpr int kAvPixFmtRgb24 = 2;
constexpr int kAvPixFmtBgr24 = 3;
constexpr int kAvPixFmtYuvj420p = 12;
constexpr int kAvPixFmtNv12 = 23;
constexpr int kAvPixFmtRgba = 26;
constexpr int kAvPixFmtBgra = 28;

bool supported_format(int f) {
  return f == kAvPixFmtYuv420p || f == kAvPixFmtYuvj420p || f == kAvPixFmtNv12 || f == kAvPixFmtRgb24 ||
         f == kAvPixFmtBgr24 || f == kAvPixFmtRgba || f == kAvPixFmtBgra;
}

// Copy of one stitched frame as delivered by the SDK (planar or packed).
struct RawFrame {
  int format = -1;
  int width = 0;
  int height = 0;
  std::vector<uint8_t> plane[3];
  int linesize[3] = {0, 0, 0};
  rclcpp::Time stamp;
};

inline uint8_t clamp8(float v) { return static_cast<uint8_t>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v + 0.5f)); }

struct Frame {
  std::vector<uint8_t> data;
  int width = 0;
  int height = 0;
  int stride = 0;
  int channels = 0;
  bool bgr = false;
  rclcpp::Time stamp;
};

// Packed RGB, stride width * 3. YUV uses BT.601 (limited range unless YUVJ).
void to_rgb(const RawFrame& raw, Frame& out) {
  const int w = raw.width, h = raw.height;
  out.width = w;
  out.height = h;
  out.stride = w * 3;
  out.channels = 3;
  out.bgr = false;
  out.stamp = raw.stamp;
  out.data.resize(static_cast<size_t>(w) * h * 3);
  const int f = raw.format;
  if (f == kAvPixFmtYuv420p || f == kAvPixFmtYuvj420p || f == kAvPixFmtNv12) {
    const bool full = f == kAvPixFmtYuvj420p;
    const float ys = full ? 1.f : 255.f / 219.f, yo = full ? 0.f : 16.f;
    const float cs = full ? 1.f : 255.f / 224.f;
    for (int y = 0; y < h; ++y) {
      const uint8_t* yr = raw.plane[0].data() + static_cast<size_t>(y) * raw.linesize[0];
      const uint8_t* ur = raw.plane[1].data() + static_cast<size_t>(y / 2) * raw.linesize[1];
      const uint8_t* vr = f == kAvPixFmtNv12 ? ur + 1 : raw.plane[2].data() + static_cast<size_t>(y / 2) * raw.linesize[2];
      const int cstep = f == kAvPixFmtNv12 ? 2 : 1;
      uint8_t* o = out.data.data() + static_cast<size_t>(y) * out.stride;
      for (int x = 0; x < w; ++x, o += 3) {
        const float Y = (yr[x] - yo) * ys;
        const float U = (ur[(x / 2) * cstep] - 128.f) * cs;
        const float V = (vr[(x / 2) * cstep] - 128.f) * cs;
        o[0] = clamp8(Y + 1.402f * V);
        o[1] = clamp8(Y - 0.344136f * U - 0.714136f * V);
        o[2] = clamp8(Y + 1.772f * U);
      }
    }
    return;
  }
  const int ch = (f == kAvPixFmtRgba || f == kAvPixFmtBgra) ? 4 : 3;
  const bool bgr = f == kAvPixFmtBgr24 || f == kAvPixFmtBgra;
  for (int y = 0; y < h; ++y) {
    const uint8_t* s = raw.plane[0].data() + static_cast<size_t>(y) * raw.linesize[0];
    uint8_t* o = out.data.data() + static_cast<size_t>(y) * out.stride;
    for (int x = 0; x < w; ++x, s += ch, o += 3) {
      o[0] = s[bgr ? 2 : 0];
      o[1] = s[1];
      o[2] = s[bgr ? 0 : 2];
    }
  }
}

// Pinhole view sampled from an equirectangular image.
// Body axes: X forward (equirect centre), Y left, Z up. Yaw is positive to the left.
struct View {
  std::string name;
  double yaw_deg = 0.0;
  int size = 0;
  double fx = 0.0;
  int eq_width = 0;
  int eq_height = 0;
  std::vector<int32_t> u0, v0;
  std::vector<float> du, dv;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_image;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pub_info;

  void build(int eq_w, int eq_h, double hfov_deg) {
    eq_width = eq_w;
    eq_height = eq_h;
    const double half = 0.5 * hfov_deg * M_PI / 180.0;
    fx = 0.5 * size / std::tan(half);
    const double c = 0.5 * size;
    const double yaw = yaw_deg * M_PI / 180.0;
    const double fwd[3] = {std::cos(yaw), std::sin(yaw), 0.0};
    const double right[3] = {std::sin(yaw), -std::cos(yaw), 0.0};
    const size_t n = static_cast<size_t>(size) * size;
    u0.resize(n);
    v0.resize(n);
    du.resize(n);
    dv.resize(n);
    for (int y = 0; y < size; ++y) {
      const double b = (y + 0.5 - c) / fx;
      for (int x = 0; x < size; ++x) {
        const double a = (x + 0.5 - c) / fx;
        const double dx = fwd[0] + a * right[0];
        const double dy = fwd[1] + a * right[1];
        const double dz = -b;
        const double lon = std::atan2(dy, dx);
        const double lat = std::atan2(dz, std::hypot(dx, dy));
        const double u = eq_w * (0.5 - lon / (2.0 * M_PI)) - 0.5;
        double v = eq_h * (0.5 - lat / M_PI) - 0.5;
        v = std::min(std::max(v, 0.0), eq_h - 1.0);
        const double fu = std::floor(u);
        const double fv = std::min(std::floor(v), eq_h - 2.0);
        const size_t i = static_cast<size_t>(y) * size + x;
        u0[i] = static_cast<int32_t>(fu);
        v0[i] = static_cast<int32_t>(fv);
        du[i] = static_cast<float>(u - fu);
        dv[i] = static_cast<float>(v - fv);
      }
    }
  }

  void render(const Frame& f, std::vector<uint8_t>& rgb) const {
    rgb.resize(static_cast<size_t>(size) * size * 3);
    const int r = f.bgr ? 2 : 0;
    const int b = f.bgr ? 0 : 2;
    const int ch = f.channels;
    for (size_t i = 0; i < u0.size(); ++i) {
      int ua = u0[i] % f.width;
      if (ua < 0) ua += f.width;
      const int ub = (ua + 1) % f.width;
      const uint8_t* row0 = f.data.data() + static_cast<size_t>(v0[i]) * f.stride;
      const uint8_t* row1 = row0 + f.stride;
      const uint8_t* p00 = row0 + ua * ch;
      const uint8_t* p01 = row0 + ub * ch;
      const uint8_t* p10 = row1 + ua * ch;
      const uint8_t* p11 = row1 + ub * ch;
      const float wu = du[i];
      const float wv = dv[i];
      const float w00 = (1 - wu) * (1 - wv), w01 = wu * (1 - wv);
      const float w10 = (1 - wu) * wv, w11 = wu * wv;
      uint8_t* out = rgb.data() + i * 3;
      const int src[3] = {r, 1, b};
      for (int k = 0; k < 3; ++k) {
        const int s = src[k];
        out[k] = static_cast<uint8_t>(w00 * p00[s] + w01 * p01[s] + w10 * p10[s] + w11 * p11[s] + 0.5f);
      }
    }
  }

  sensor_msgs::msg::CameraInfo camera_info(const std::string& frame_id, const rclcpp::Time& stamp) const {
    sensor_msgs::msg::CameraInfo info;
    info.header.stamp = stamp;
    info.header.frame_id = frame_id;
    info.width = size;
    info.height = size;
    info.distortion_model = "plumb_bob";
    info.d = {0, 0, 0, 0, 0};
    const double c = 0.5 * size - 0.5;
    info.k = {fx, 0, c, 0, fx, c, 0, 0, 1};
    info.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    info.p = {fx, 0, c, 0, 0, fx, c, 0, 0, 0, 1, 0};
    return info;
  }
};

bool encode_jpeg(const uint8_t* rgb, int width, int height, int quality, std::vector<uint8_t>& out) {
  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  unsigned char* buf = nullptr;
  unsigned long len = 0;
  jpeg_mem_dest(&cinfo, &buf, &len);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row = const_cast<uint8_t*>(rgb + static_cast<size_t>(cinfo.next_scanline) * width * 3);
    jpeg_write_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_compress(&cinfo);
  out.assign(buf, buf + len);
  jpeg_destroy_compress(&cinfo);
  free(buf);
  return len > 0;
}

ins_camera::VideoResolution parse_resolution(const std::string& s) {
  static const std::map<std::string, ins_camera::VideoResolution> table = {
      {"3840_1920", ins_camera::VideoResolution::RES_3840_1920P30},
      {"2560_1280", ins_camera::VideoResolution::RES_2560_1280P30},
      {"1920_960", ins_camera::VideoResolution::RES_1920_960P30},
      {"1440_720", ins_camera::VideoResolution::RES_1440_720P30},
      {"960_480", ins_camera::VideoResolution::RES_960_480P30},
  };
  const auto it = table.find(s);
  if (it == table.end()) throw std::invalid_argument("stream_resolution must be one of 3840_1920, 2560_1280, 1920_960, 1440_720, 960_480");
  return it->second;
}

ins::STITCH_TYPE parse_stitch(const std::string& s) {
  if (s == "template") return ins::STITCH_TYPE::TEMPLATE;
  if (s == "optflow") return ins::STITCH_TYPE::OPTFLOW;
  if (s == "dynamic") return ins::STITCH_TYPE::DYNAMICSTITCH;
  if (s == "aiflow") return ins::STITCH_TYPE::AIFLOW;
  throw std::invalid_argument("stitch_type must be template, optflow, dynamic or aiflow");
}

class StitchDelegate : public ins_camera::StreamDelegate {
 public:
  explicit StitchDelegate(std::shared_ptr<ins::RealTimeStitcher> stitcher) : stitcher_(std::move(stitcher)) {}
  void OnAudioData(const uint8_t*, size_t, int64_t) override {}
  void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t stream_type, int stream_index) override {
    stitcher_->HandleVideoData(data, size, timestamp, stream_type, stream_index);
  }
  void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
    std::vector<ins::GyroData> v(data.size());
    std::memcpy(v.data(), data.data(), data.size() * sizeof(ins_camera::GyroData));
    stitcher_->HandleGyroData(v);
  }
  void OnExposureData(const ins_camera::ExposureData& data) override {
    ins::ExposureData e{};
    e.exposure_time = data.exposure_time;
    e.timestamp = data.timestamp;
    stitcher_->HandleExposureData(e);
  }

 private:
  std::shared_ptr<ins::RealTimeStitcher> stitcher_;
};

class Insta360Views : public rclcpp::Node {
 public:
  Insta360Views() : Node("insta360_views") {
    model_dir_ = declare_parameter<std::string>("model_dir", "");
    resolution_ = declare_parameter<std::string>("stream_resolution", "1920_960");
    bitrate_mbps_ = declare_parameter<double>("stream_bitrate_mbps", 4.0);
    stitch_type_ = declare_parameter<std::string>("stitch_type", "dynamic");
    eq_width_ = declare_parameter<int>("equirect_width", 1920);
    view_size_ = declare_parameter<int>("view_size", 640);
    hfov_deg_ = declare_parameter<double>("view_hfov_deg", 90.0);
    front_yaw_deg_ = declare_parameter<double>("front_yaw_deg", 0.0);
    rate_hz_ = declare_parameter<double>("publish_rate_hz", 15.0);
    jpeg_quality_ = declare_parameter<int>("jpeg_quality", 80);
    publish_equirect_ = declare_parameter<bool>("publish_equirect", true);
    software_decode_ = declare_parameter<bool>("software_decode", false);
    frame_prefix_ = declare_parameter<std::string>("frame_prefix", "insta360");
    parse_resolution(resolution_);
    parse_stitch(stitch_type_);
    if (!model_dir_.empty()) {
      // Media SDK requires the trailing slash.
      ins::SetModelFileRootDir(model_dir_.back() == '/' ? model_dir_ : model_dir_ + "/");
    }

    const std::pair<const char*, double> names[] = {{"front", 0.0}, {"left", 90.0}, {"rear", 180.0}, {"right", -90.0}};
    for (const auto& [name, yaw] : names) {
      View v;
      v.name = name;
      v.yaw_deg = front_yaw_deg_ + yaw;
      v.size = view_size_;
      const std::string base = "/" + frame_prefix_ + "/" + name;
      v.pub_image = create_publisher<sensor_msgs::msg::CompressedImage>(base + "/image/compressed", rclcpp::SensorDataQoS());
      v.pub_info = create_publisher<sensor_msgs::msg::CameraInfo>(base + "/camera_info", rclcpp::SensorDataQoS());
      views_.push_back(std::move(v));
    }
    if (publish_equirect_) {
      pub_equirect_ = create_publisher<sensor_msgs::msg::CompressedImage>(
          "/" + frame_prefix_ + "/equirect/image/compressed", rclcpp::SensorDataQoS());
    }
    RCLCPP_INFO(get_logger(), "views front/left/rear/right %dx%d hfov %.0f deg, front_yaw %.1f deg, stitch %s, stream %s, topics /%s/<view>/image/compressed",
                view_size_, view_size_, hfov_deg_, front_yaw_deg_, stitch_type_.c_str(), resolution_.c_str(), frame_prefix_.c_str());

    worker_ = std::thread([this] { work_loop(); });
    camera_thread_ = std::thread([this] { camera_loop(); });
  }

  ~Insta360Views() override {
    running_ = false;
    frame_cv_.notify_all();
    if (camera_thread_.joinable()) camera_thread_.join();
    if (worker_.joinable()) worker_.join();
  }

 private:
  void on_frame(uint8_t* data[4], int linesize[4], int width, int height, int format) {
    last_frame_ns_ = now().nanoseconds();
    // X4 Air reports format 0 (YUV420P) but delivers one packed RGBA plane
    // (linesize 4 * width, no chroma planes), as the Media SDK demo assumes.
    if (data[1] == nullptr && data[0] != nullptr) {
      if (linesize[0] >= 4 * width) {
        format = kAvPixFmtRgba;
      } else if (linesize[0] >= 3 * width) {
        format = kAvPixFmtRgb24;
      }
    }
    if (!supported_format(format)) {
      if (!format_warned_.exchange(true)) {
        RCLCPP_ERROR(get_logger(), "unsupported stitched pixel format %d (AVPixelFormat)", format);
      }
      return;
    }
    if (!format_logged_.exchange(true)) {
      RCLCPP_INFO(get_logger(), "first stitched frame %dx%d format %d linesize %d %d %d planes %p %p %p", width, height,
                  format, linesize[0], linesize[1], linesize[2], static_cast<void*>(data[0]),
                  static_cast<void*>(data[1]), static_cast<void*>(data[2]));
    }
    const bool yuv420 = format == kAvPixFmtYuv420p || format == kAvPixFmtYuvj420p;
    const int planes = yuv420 ? 3 : (format == kAvPixFmtNv12 ? 2 : 1);
    const int bytes_pp = format == kAvPixFmtRgba || format == kAvPixFmtBgra ? 4
                         : (format == kAvPixFmtRgb24 || format == kAvPixFmtBgr24 ? 3 : 1);
    const int chroma_w = format == kAvPixFmtNv12 ? 2 * ((width + 1) / 2) : (width + 1) / 2;
    for (int i = 0; i < planes; ++i) {
      const int need = i == 0 ? width * bytes_pp : chroma_w;
      if (data[i] == nullptr || linesize[i] < need) {
        if (!layout_warned_.exchange(true)) {
          RCLCPP_ERROR(get_logger(), "unexpected plane %d layout (linesize %d, need %d); frame dropped", i, linesize[i], need);
        }
        return;
      }
    }
    std::lock_guard<std::mutex> lk(frame_mutex_);
    latest_.format = format;
    latest_.width = width;
    latest_.height = height;
    latest_.stamp = now();
    for (int i = 0; i < 3; ++i) {
      if (i >= planes) {
        latest_.linesize[i] = 0;
        continue;
      }
      const int rows = i == 0 ? height : (height + 1) / 2;
      latest_.linesize[i] = linesize[i];
      latest_.plane[i].assign(data[i], data[i] + static_cast<size_t>(linesize[i]) * rows);
    }
    has_frame_ = true;
    frame_cv_.notify_one();
  }

  void work_loop() {
    RawFrame raw;
    Frame f;
    std::vector<uint8_t> rgb, jpeg;
    const auto period = std::chrono::duration<double>(1.0 / std::max(rate_hz_, 0.1));
    auto next = std::chrono::steady_clock::now();
    while (running_ && rclcpp::ok()) {
      {
        std::unique_lock<std::mutex> lk(frame_mutex_);
        frame_cv_.wait_for(lk, 500ms, [this] { return has_frame_ || !running_; });
        if (!has_frame_) continue;
        std::swap(raw, latest_);
        has_frame_ = false;
      }
      if (std::chrono::steady_clock::now() < next) continue;
      next = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
      if (raw.width < 2 || raw.height < 2) continue;
      to_rgb(raw, f);
      for (auto& v : views_) {
        if (v.eq_width != f.width || v.eq_height != f.height) v.build(f.width, f.height, hfov_deg_);
        v.render(f, rgb);
        if (!encode_jpeg(rgb.data(), v.size, v.size, jpeg_quality_, jpeg)) continue;
        const std::string frame_id = frame_prefix_ + "_" + v.name + "_optical";
        auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
        msg->header.stamp = f.stamp;
        msg->header.frame_id = frame_id;
        msg->format = "jpeg";
        msg->data = jpeg;
        v.pub_image->publish(std::move(msg));
        v.pub_info->publish(v.camera_info(frame_id, f.stamp));
      }
      if (pub_equirect_ && encode_jpeg(f.data.data(), f.width, f.height, jpeg_quality_, jpeg)) {
        auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
        msg->header.stamp = f.stamp;
        msg->header.frame_id = frame_prefix_;
        msg->format = "jpeg";
        msg->data = jpeg;
        pub_equirect_->publish(std::move(msg));
      }
    }
  }

  // Connect, stream and reconnect when the camera goes away.
  void camera_loop() {
    bool waiting_logged = false;
    while (running_ && rclcpp::ok()) {
      ins_camera::DeviceDiscovery discovery;
      auto list = discovery.GetAvailableDevices();
      if (list.empty()) {
        if (!waiting_logged) {
          RCLCPP_WARN(get_logger(),
                      "SDK found no Insta360 camera; waiting. Check USB 3.0, Android mode on the camera, and USB "
                      "permissions (tools/check_insta360_usb.py, config/udev/99-insta360.rules).");
          waiting_logged = true;
        }
        discovery.FreeDeviceDescriptors(list);
        sleep_while_running(2s);
        continue;
      }
      waiting_logged = false;
      RCLCPP_INFO(get_logger(), "camera %s serial %s fw %s", list[0].camera_name.c_str(),
                  list[0].serial_number.c_str(), list[0].fw_version.c_str());
      auto cam = std::make_shared<ins_camera::Camera>(list[0].info);
      // The connection info points into the descriptors; free them only after Open().
      const bool opened = cam->Open();
      discovery.FreeDeviceDescriptors(list);
      if (!opened) {
        RCLCPP_ERROR(get_logger(), "camera Open() failed; retrying");
        sleep_while_running(2s);
        continue;
      }
      stream(cam);
      cam->Close();
      if (running_) sleep_while_running(2s);
    }
  }

  void stream(const std::shared_ptr<ins_camera::Camera>& cam) {
    auto stitcher = std::make_shared<ins::RealTimeStitcher>();
    const auto preview = cam->GetPreviewParam();
    ins::CameraInfo info;
    info.cameraName = preview.camera_name;
    info.decode_type = static_cast<ins::VideoDecodeType>(preview.encode_type);
    std::vector<std::string> calib;
    for (size_t i = 0; i < preview.GetCalibrationCount(); ++i) calib.push_back(preview.GetCalibration(i));
    info.SetCalibration(calib, preview.GetCropSrcWidth(), preview.GetCropSrcHeight(), preview.GetCropDstWidth(),
                        preview.GetCropDstHeight(), preview.GetCropOffsetX(), preview.GetCropOffsetY());
    stitcher->SetCameraInfo(info);
    stitcher->SetStitchType(parse_stitch(stitch_type_));
    stitcher->EnableFlowState(false);
    stitcher->SetOutputSize(eq_width_, eq_width_ / 2);
    stitcher->SetSoftwareCodecUsage(false, software_decode_);
    stitcher->SetStitchStateCallback([this](int error, const char* text) {
      RCLCPP_ERROR(get_logger(), "stitcher error %d: %s", error, text ? text : "");
    });
    stitcher->SetStitchRealTimeDataCallback(
        [this](uint8_t* data[4], int linesize[4], int width, int height, int format, int64_t) {
          on_frame(data, linesize, width, height, format);
        });
    std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<StitchDelegate>(stitcher);
    cam->SetStreamDelegate(delegate);

    ins_camera::LiveStreamParam param;
    param.video_resolution = parse_resolution(resolution_);
    param.lrv_video_resulution = param.video_resolution;
    param.video_bitrate = static_cast<uint32_t>(bitrate_mbps_ * 1024 * 1024);
    param.enable_audio = false;
    param.using_lrv = false;
    if (!cam->StartLiveStreaming(param)) {
      RCLCPP_ERROR(get_logger(), "StartLiveStreaming failed; retrying");
      return;
    }
    stitcher->StartStitch();
    RCLCPP_INFO(get_logger(), "live stream started (%s, %s), camera %s", resolution_.c_str(), stitch_type_.c_str(),
                preview.camera_name.c_str());

    const int64_t started = now().nanoseconds();
    last_frame_ns_ = 0;
    while (running_ && rclcpp::ok()) {
      std::this_thread::sleep_for(500ms);
      const int64_t last = last_frame_ns_.load();
      const int64_t ref = last > 0 ? last : started;
      if (now().nanoseconds() - ref > 5'000'000'000LL) {
        RCLCPP_WARN(get_logger(), "no stitched frame for 5 s; reconnecting");
        break;
      }
    }
    cam->StopLiveStreaming();
    stitcher->CancelStitch();
  }

  void sleep_while_running(std::chrono::milliseconds d) {
    const auto end = std::chrono::steady_clock::now() + d;
    while (running_ && rclcpp::ok() && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(100ms);
  }

  std::string model_dir_, resolution_, stitch_type_, frame_prefix_;
  double bitrate_mbps_, hfov_deg_, front_yaw_deg_, rate_hz_;
  int eq_width_, view_size_, jpeg_quality_;
  bool publish_equirect_, software_decode_;
  std::vector<View> views_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_equirect_;
  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  RawFrame latest_;
  bool has_frame_ = false;
  std::atomic<bool> running_{true};
  std::atomic<bool> format_logged_{false}, format_warned_{false}, layout_warned_{false};
  std::atomic<int64_t> last_frame_ns_{0};
  std::thread worker_, camera_thread_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  ins::InitEnv();
  ins_camera::SetLogLevel(ins_camera::LogLevel::WARNING);
  ins::SetLogLevel(ins::InsLogLevel::WARNING);
  auto node = std::make_shared<Insta360Views>();
  rclcpp::spin(node);
  node.reset();
  rclcpp::shutdown();
  // libMNN.so (pulled in by libMediaSDK.so) corrupts the heap in its global
  // destructors, so skip them once everything above has shut down.
  std::fflush(nullptr);
  std::_Exit(0);
}
