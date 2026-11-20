// Insta360 X4 Air live views over USB.
// Camera SDK preview stream -> equirectangular stitch -> front, rear, left and right
// pinhole views as JPEG CompressedImage. Stitching (stitch_backend):
//   mediasdk  Media SDK RealTimeStitcher (x86-64 only)
//   cuda      ../insta360_cuda_stitch: GStreamer decode + CUDA remap from the preview
//             calibration (Jetson, or any machine with CUDA)
//
// No OpenCV here: libMediaSDK.so loads its own opencv_world 4.7, and a
// second OpenCV in the same process is not safe.
#include <camera/camera.h>
#include <camera/device_discovery.h>
#ifdef HAVE_MEDIA_SDK
#include <ins_realtime_stitcher.h>
#include <ins_stitcher.h>
#endif
#ifdef HAVE_CUDA_STITCH
#include <insta360_cuda_stitch/live_stitcher.hpp>
#endif
#include <jpeglib.h>
#include <fcntl.h>
#include <linux/usbdevice_fs.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

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
// Tightly packed RGB24 is taken over without a copy, leaving raw.plane[0] with out's old buffer.
void to_rgb(RawFrame& raw, Frame& out) {
  const int w = raw.width, h = raw.height;
  out.width = w;
  out.height = h;
  out.stride = w * 3;
  out.channels = 3;
  out.bgr = false;
  out.stamp = raw.stamp;
  const int f = raw.format;
  if (f == kAvPixFmtRgb24 && raw.linesize[0] == w * 3) {
    out.data.swap(raw.plane[0]);
    return;
  }
  out.data.resize(static_cast<size_t>(w) * h * 3);
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
  std::vector<uint8_t> rgb, jpeg;  // per view so the views can be rendered in parallel
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

#ifdef HAVE_MEDIA_SDK
ins::STITCH_TYPE parse_stitch(const std::string& s) {
  if (s == "template") return ins::STITCH_TYPE::TEMPLATE;
  if (s == "optflow") return ins::STITCH_TYPE::OPTFLOW;
  if (s == "dynamic") return ins::STITCH_TYPE::DYNAMICSTITCH;
  if (s == "aiflow") return ins::STITCH_TYPE::AIFLOW;
  throw std::invalid_argument("stitch_type must be template, optflow, dynamic or aiflow");
}
#endif

std::string pick_backend(const std::string& requested) {
#ifdef HAVE_MEDIA_SDK
  constexpr bool media_sdk = true;
#else
  constexpr bool media_sdk = false;
#endif
#ifdef HAVE_CUDA_STITCH
  constexpr bool cuda = true;
#else
  constexpr bool cuda = false;
#endif
  if (requested == "auto") {
    if (media_sdk) return "mediasdk";
    if (cuda) return "cuda";
    throw std::invalid_argument("built without any stitcher");
  }
  if (requested == "mediasdk" && !media_sdk) throw std::invalid_argument("built without Media SDK; use stitch_backend:=cuda");
  if (requested == "cuda" && !cuda) throw std::invalid_argument("built without insta360_cuda_stitch (CUDA)");
  if (requested != "mediasdk" && requested != "cuda") throw std::invalid_argument("stitch_backend must be auto, mediasdk or cuda");
  return requested;
}

std::string json_escape(const std::string& s) {
  std::ostringstream out;
  for (const unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out << '\\' << c;
    } else if (c < 0x20) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
    } else {
      out << c;
    }
  }
  return out.str();
}

// Writes what the camera sends before stitching (encoded fisheye packets, gyro,
// exposure, preview calibration) so a session can be restitched or fed to SfM
// offline. One seg_NNN directory per camera connection.
class RawRecorder {
 public:
  explicit RawRecorder(std::string root) : root_(std::move(root)) {}

  std::string start_segment(const ins_camera::PreviewParam& preview, const std::string& camera_name,
                            const std::string& serial, const std::string& fw, const std::string& resolution,
                            double bitrate_mbps, double arrival_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    close_locked();
    // skip segments of an earlier process (respawn after a crash) instead of overwriting them
    do {
      std::ostringstream name;
      name << "seg_" << std::setw(3) << std::setfill('0') << segment_++;
      dir_ = std::filesystem::path(root_) / name.str();
    } while (std::filesystem::exists(dir_));
    std::filesystem::create_directories(dir_);
    ext_ = preview.encode_type == ins_camera::VideoEncodeType::H265 ? "h265" : "h264";

    std::ofstream info(dir_ / "camera.json");
    info << std::setprecision(17) << "{\n"
         << "  \"camera_name\": \"" << json_escape(camera_name) << "\",\n"
         << "  \"preview_camera_name\": \"" << json_escape(preview.camera_name) << "\",\n"
         << "  \"serial\": \"" << json_escape(serial) << "\",\n"
         << "  \"firmware\": \"" << json_escape(fw) << "\",\n"
         << "  \"encode\": \"" << ext_ << "\",\n"
         << "  \"stream_resolution\": \"" << resolution << "\",\n"
         << "  \"stream_bitrate_mbps\": " << bitrate_mbps << ",\n"
         << "  \"segment_start_ros_s\": " << arrival_s << ",\n"
         << "  \"delay_timestamp\": " << preview.delay_timestamp << ",\n"
         << "  \"sweep_time\": " << preview.sweep_time << ",\n"
         << "  \"acceleration_range\": " << preview.acceleration_range << ",\n"
         << "  \"gyro_range\": " << preview.gyro_range << ",\n"
         << "  \"crop\": {\"src_width\": " << preview.GetCropSrcWidth() << ", \"src_height\": " << preview.GetCropSrcHeight()
         << ", \"dst_width\": " << preview.GetCropDstWidth() << ", \"dst_height\": " << preview.GetCropDstHeight()
         << ", \"offset_x\": " << preview.GetCropOffsetX() << ", \"offset_y\": " << preview.GetCropOffsetY() << "},\n"
         << "  \"calibration\": [";
    for (size_t i = 0; i < preview.GetCalibrationCount(); ++i) {
      info << (i ? ", " : "") << "\"" << json_escape(preview.GetCalibration(i)) << "\"";
    }
    info << "]\n}\n";

    video_index_.open(dir_ / "video_index.csv");
    video_index_ << "seq,stream_index,stream_type,device_ts,arrival_ros_s,file,offset,size\n";
    gyro_.open(dir_ / "gyro.csv");
    gyro_ << "device_ts,ros_s,arrival_ros_s,ax_g,ay_g,az_g,gx_rad_s,gy_rad_s,gz_rad_s\n";
    exposure_.open(dir_ / "exposure.csv");
    exposure_ << "device_ts,arrival_ros_s,exposure_time\n";
    frames_.open(dir_ / "frames.csv");
    frames_ << "device_ts,ros_s\n";
    for (auto* f : {&video_index_, &gyro_, &exposure_, &frames_}) *f << std::setprecision(17);
    return dir_.string();
  }

  void video(const uint8_t* data, size_t size, int64_t device_ts, uint8_t stream_type, int stream_index,
             double arrival_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!video_index_.is_open()) return;
    auto& out = streams_[stream_index];
    if (!out.file.is_open()) {
      out.name = "stream_" + std::to_string(stream_index) + "." + ext_;
      out.file.open(dir_ / out.name, std::ios::binary);
    }
    video_index_ << seq_++ << ',' << stream_index << ',' << static_cast<int>(stream_type) << ',' << device_ts << ','
                 << arrival_s << ',' << out.name << ',' << out.offset << ',' << size << '\n';
    out.file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    out.offset += size;
    bytes_ += size;
  }

  void gyro(const ins_camera::GyroData& g, double ros_s, double arrival_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!gyro_.is_open()) return;
    gyro_ << g.timestamp << ',' << ros_s << ',' << arrival_s << ',' << g.ax << ',' << g.ay << ',' << g.az << ','
          << g.gx << ',' << g.gy << ',' << g.gz << '\n';
  }

  void exposure(const ins_camera::ExposureData& e, double arrival_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!exposure_.is_open()) return;
    exposure_ << e.timestamp << ',' << arrival_s << ',' << e.exposure_time << '\n';
  }

  void frame(int64_t device_ts, double ros_s) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!frames_.is_open()) return;
    frames_ << device_ts << ',' << ros_s << '\n';
  }

  uint64_t bytes() const { return bytes_.load(); }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    close_locked();
  }

 private:
  struct StreamFile {
    std::string name;
    std::ofstream file;
    uint64_t offset = 0;
  };

  void close_locked() {
    streams_.clear();
    for (auto* f : {&video_index_, &gyro_, &exposure_, &frames_}) {
      if (f->is_open()) f->close();
    }
    seq_ = 0;
  }

  std::string root_;
  std::filesystem::path dir_;
  std::string ext_ = "h264";
  int segment_ = 0;
  uint64_t seq_ = 0;
  std::atomic<uint64_t> bytes_{0};
  std::map<int, StreamFile> streams_;
  std::ofstream video_index_, gyro_, exposure_, frames_;
  std::mutex mutex_;
};

// Camera SDK callbacks -> recorder, IMU publisher and the active stitcher.
class StitchDelegate : public ins_camera::StreamDelegate {
 public:
  using VideoSink = std::function<void(const uint8_t*, size_t, int64_t, uint8_t, int)>;
  using GyroSink = std::function<void(const std::vector<ins_camera::GyroData>&)>;
  using ExposureSink = std::function<void(const ins_camera::ExposureData&)>;
  using Clock = std::function<double()>;
  StitchDelegate(VideoSink video, GyroSink gyro, ExposureSink exposure, GyroSink imu,
                 std::shared_ptr<RawRecorder> recorder = nullptr, Clock clock = nullptr)
      : video_(std::move(video)), gyro_(std::move(gyro)), exposure_(std::move(exposure)), imu_(std::move(imu)),
        recorder_(std::move(recorder)), clock_(std::move(clock)) {}
  void OnAudioData(const uint8_t*, size_t, int64_t) override {}
  void OnVideoData(const uint8_t* data, size_t size, int64_t timestamp, uint8_t stream_type, int stream_index) override {
    if (recorder_) recorder_->video(data, size, timestamp, stream_type, stream_index, clock_());
    if (video_) video_(data, size, timestamp, stream_type, stream_index);
  }
  void OnGyroData(const std::vector<ins_camera::GyroData>& data) override {
    if (gyro_) gyro_(data);
    if (imu_) imu_(data);
  }
  void OnExposureData(const ins_camera::ExposureData& data) override {
    if (recorder_) recorder_->exposure(data, clock_());
    if (exposure_) exposure_(data);
  }

 private:
  VideoSink video_;
  GyroSink gyro_;
  ExposureSink exposure_;
  GyroSink imu_;
  std::shared_ptr<RawRecorder> recorder_;
  Clock clock_;
};

// Maps camera timestamps onto ROS time. IMU arrives in ~100 ms bursts, so arrival
// time is not the sample time. The unit is detected from the sample spacing.
class DeviceClock {
 public:
  explicit DeviceClock(double nominal_period_s) : nominal_period_s_(nominal_period_s) {}

  bool ready() const { return scale_ > 0.0 && have_offset_; }
  double scale() const { return scale_; }
  double offset() const { return offset_; }

  bool update(int64_t device_ts, double now_s) {
    if (scale_ <= 0.0) {
      if (have_prev_ && device_ts > prev_ts_) {
        const double dt = static_cast<double>(device_ts - prev_ts_);
        for (double scale : {1e-9, 1e-6, 1e-3}) {
          const double period = dt * scale;
          if (period > 0.2 * nominal_period_s_ && period < 20.0 * nominal_period_s_) {
            scale_ = scale;
            break;
          }
        }
      }
      prev_ts_ = device_ts;
      have_prev_ = true;
      if (scale_ <= 0.0) return false;
    }
    const double latency = now_s - static_cast<double>(device_ts) * scale_;
    if (!have_offset_) {
      offset_ = latency;
      have_offset_ = true;
      last_now_ = now_s;
    } else {
      offset_ += 1e-4 * std::max(0.0, now_s - last_now_);
      last_now_ = now_s;
      offset_ = std::min(offset_, latency);
    }
    return true;
  }

  double to_ros(int64_t device_ts) const { return to_ros(device_ts, offset_); }
  double to_ros(int64_t device_ts, double offset) const {
    return static_cast<double>(device_ts) * scale_ + offset;
  }

 private:
  double nominal_period_s_;
  double scale_ = 0.0;
  double offset_ = 0.0;
  double last_now_ = 0.0;
  int64_t prev_ts_ = 0;
  bool have_prev_ = false;
  bool have_offset_ = false;
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
    publish_vio_topics_ = declare_parameter<bool>("publish_vio_topics", true);
    vio_image_stride_ = static_cast<int>(std::max<int64_t>(1, declare_parameter<int>("vio_image_stride", 1)));
    software_decode_ = declare_parameter<bool>("software_decode", false);
    frame_prefix_ = declare_parameter<std::string>("frame_prefix", "insta360");
    raw_dump_dir_ = declare_parameter<std::string>("raw_dump_dir", "");
    if (!raw_dump_dir_.empty()) {
      recorder_ = std::make_shared<RawRecorder>(raw_dump_dir_);
      RCLCPP_INFO(get_logger(), "raw camera stream, gyro and exposure -> %s", raw_dump_dir_.c_str());
    }
    parse_resolution(resolution_);
    video_submode_ = declare_parameter<std::string>("video_submode", "keep");
    if (video_submode_ != "keep" && video_submode_ != "normal" && video_submode_ != "liveview")
      throw std::invalid_argument("video_submode must be keep, normal or liveview");
    backend_ = pick_backend(declare_parameter<std::string>("stitch_backend", "auto"));
    cuda_decoder_ = declare_parameter<std::string>("cuda_decoder", "");
#ifdef HAVE_MEDIA_SDK
    if (backend_ == "mediasdk") {
      parse_stitch(stitch_type_);
      if (!model_dir_.empty()) {
        // Media SDK requires the trailing slash.
        ins::SetModelFileRootDir(model_dir_.back() == '/' ? model_dir_ : model_dir_ + "/");
      }
    }
#endif

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
    if (publish_vio_topics_) {
      pub_equirect_raw_ = create_publisher<sensor_msgs::msg::Image>(
          "/" + frame_prefix_ + "/equirect/image", rclcpp::SensorDataQoS());
      pub_imu_ = create_publisher<sensor_msgs::msg::Imu>("/imu/data_raw", rclcpp::QoS(2000));
    }
    RCLCPP_INFO(get_logger(), "views front/left/rear/right %dx%d hfov %.0f deg, front_yaw %.1f deg, stitch %s, stream %s, topics /%s/<view>/image/compressed",
                view_size_, view_size_, hfov_deg_, front_yaw_deg_,
                backend_ == "mediasdk" ? stitch_type_.c_str() : backend_.c_str(), resolution_.c_str(), frame_prefix_.c_str());

    worker_ = std::thread([this] { work_loop(); });
    if (pub_equirect_) equirect_thread_ = std::thread([this] { equirect_loop(); });
    camera_thread_ = std::thread([this] { camera_loop(); });
  }

  ~Insta360Views() override {
    running_ = false;
    frame_cv_.notify_all();
    eq_cv_.notify_all();
    if (camera_thread_.joinable()) camera_thread_.join();
    if (worker_.joinable()) worker_.join();
    if (equirect_thread_.joinable()) equirect_thread_.join();
  }

 private:
  rclcpp::Time stamp_from_device(int64_t device_ts) {
    const rclcpp::Time wall = now();
    if (device_ts <= 0) return wall;
    std::lock_guard<std::mutex> lock(clock_mutex_);
    if (!video_clock_.update(device_ts, wall.seconds())) return wall;
    double t = video_clock_.to_ros(device_ts);
    if (imu_clock_.ready()) {
      const bool shared = video_clock_.scale() == imu_clock_.scale() &&
                          std::abs(video_clock_.offset() - imu_clock_.offset()) < 0.5;
      if (shared) t = video_clock_.to_ros(device_ts, imu_clock_.offset());
    }
    const int64_t now_ns = wall.nanoseconds();
    if (now_ns - last_clock_report_ns_ > 10'000'000'000LL) {
      last_clock_report_ns_ = now_ns;
      RCLCPP_INFO(get_logger(),
                  "device clock: imu scale=%g offset=%.4f | video scale=%g offset=%.4f | latency=%.1f ms",
                  imu_clock_.scale(), imu_clock_.offset(), video_clock_.scale(), video_clock_.offset(),
                  1e3 * (wall.seconds() - t));
    }
    return rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME);
  }

  void on_frame(uint8_t* data[4], int linesize[4], int width, int height, int format, int64_t device_ts) {
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
    latest_.stamp = stamp_from_device(device_ts);
    if (recorder_) recorder_->frame(device_ts, latest_.stamp.seconds());
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
      if (raw.width < 2 || raw.height < 2) continue;
      to_rgb(raw, f);
      if (pub_equirect_) {
        std::lock_guard<std::mutex> lk(eq_mutex_);
        if (!eq_has_ && !eq_busy_) {
          eq_pending_ = f;
          eq_has_ = true;
          eq_cv_.notify_one();
        }
      }
      // JPEG views stay at publish_rate_hz so YOLOE is not flooded. Advance a fixed schedule rather
      // than now + period: with the period close to the frame interval, jitter would skip every
      // other frame.
      const auto t = std::chrono::steady_clock::now();
      if (t >= next) {
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        if (next < t) next = t;
        publish_views(f);
      }
      // VIO needs the stitcher rate by default (vio_image_stride=1). Last, so the frame can be moved.
      if (pub_equirect_raw_ && f.channels == 3 && !f.bgr && (++vio_frame_ % vio_image_stride_) == 0) {
        auto raw_msg = std::make_unique<sensor_msgs::msg::Image>();
        raw_msg->header.stamp = f.stamp;
        raw_msg->header.frame_id = frame_prefix_;
        raw_msg->height = static_cast<uint32_t>(f.height);
        raw_msg->width = static_cast<uint32_t>(f.width);
        raw_msg->encoding = "rgb8";
        raw_msg->is_bigendian = 0;
        raw_msg->step = static_cast<uint32_t>(f.stride);
        raw_msg->data = std::move(f.data);
        f.data.clear();
        pub_equirect_raw_->publish(std::move(raw_msg));
      }
    }
  }

  // A 3840x1920 JPEG takes 50-70 ms on one Orin NX core, so the equirect JPEG has its own thread
  // that always takes the newest frame and never holds up the views or the raw topic.
  void equirect_loop() {
    Frame f;
    std::vector<uint8_t> jpeg;
    while (running_ && rclcpp::ok()) {
      {
        std::unique_lock<std::mutex> lk(eq_mutex_);
        eq_cv_.wait_for(lk, 500ms, [this] { return eq_has_ || !running_; });
        if (!eq_has_) continue;
        std::swap(f, eq_pending_);
        eq_has_ = false;
        eq_busy_ = true;
      }
      if (encode_jpeg(f.data.data(), f.width, f.height, jpeg_quality_, jpeg)) {
        auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
        msg->header.stamp = f.stamp;
        msg->header.frame_id = frame_prefix_;
        msg->format = "jpeg";
        msg->data = jpeg;
        pub_equirect_->publish(std::move(msg));
      }
      std::lock_guard<std::mutex> lk(eq_mutex_);
      eq_busy_ = false;
    }
  }

  // Render, encode and publish each view on its own thread; serially the four 640x640 JPEGs take
  // most of a 30 fps frame budget on Orin NX.
  void publish_views(const Frame& f) {
    std::vector<std::thread> jobs;
    jobs.reserve(views_.size());
    for (auto& view : views_) {
      jobs.emplace_back([this, &f, &v = view] {
        if (v.eq_width != f.width || v.eq_height != f.height) v.build(f.width, f.height, hfov_deg_);
        v.render(f, v.rgb);
        if (!encode_jpeg(v.rgb.data(), v.size, v.size, jpeg_quality_, v.jpeg)) return;
        const std::string frame_id = frame_prefix_ + "_" + v.name + "_optical";
        auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
        msg->header.stamp = f.stamp;
        msg->header.frame_id = frame_id;
        msg->format = "jpeg";
        msg->data = v.jpeg;
        v.pub_image->publish(std::move(msg));
        v.pub_info->publish(v.camera_info(frame_id, f.stamp));
      });
    }
    for (auto& j : jobs) j.join();
  }

  // Runs even when /imu/data_raw is off: the IMU clock also stamps the video (stamp_from_device).
  void publish_imu(const std::vector<ins_camera::GyroData>& data) {
    const rclcpp::Time wall = this->now();
    std::lock_guard<std::mutex> lock(clock_mutex_);
    for (const auto& gyro : data) {
      const bool mapped = imu_clock_.update(gyro.timestamp, wall.seconds());
      const double stamp = mapped ? imu_clock_.to_ros(gyro.timestamp) : -1.0;
      if (recorder_) recorder_->gyro(gyro, stamp, wall.seconds());
      if (!mapped || !pub_imu_) continue;
      if (stamp <= last_imu_stamp_) continue;
      last_imu_stamp_ = stamp;
      auto msg = std::make_unique<sensor_msgs::msg::Imu>();
      const int64_t ns = static_cast<int64_t>(stamp * 1e9);
      msg->header.stamp = rclcpp::Time(ns, RCL_ROS_TIME);
      msg->header.frame_id = "imu";
      msg->angular_velocity.x = gyro.gx;
      msg->angular_velocity.y = gyro.gy;
      msg->angular_velocity.z = gyro.gz;
      msg->linear_acceleration.x = gyro.ax * 9.80665;
      msg->linear_acceleration.y = gyro.ay * 9.80665;
      msg->linear_acceleration.z = gyro.az * 9.80665;
      msg->orientation.w = 1.0;
      msg->orientation_covariance[0] = -1.0;
      pub_imu_->publish(std::move(msg));
    }
  }

  struct UsbCamera {
    std::string pid;   // empty = not enumerated
    std::string node;  // /dev/bus/usb/BBB/DDD
  };

  static std::string read_attr(const std::filesystem::path& p) {
    std::ifstream f(p);
    std::string s;
    f >> s;
    return s;
  }

  static UsbCamera find_usb_camera() {
    std::error_code ec;
    for (const auto& d : std::filesystem::directory_iterator("/sys/bus/usb/devices", ec)) {
      if (read_attr(d.path() / "idVendor") != "2e1a") continue;
      char node[64];
      std::snprintf(node, sizeof(node), "/dev/bus/usb/%03d/%03d", std::atoi(read_attr(d.path() / "busnum").c_str()),
                    std::atoi(read_attr(d.path() / "devnum").c_str()));
      return {read_attr(d.path() / "idProduct"), node};
    }
    return {};
  }

  // USBDEVFS_RESET re-enumerates the camera, which clears a hung SDK handshake (same as toggling
  // sysfs authorized, but needs only rw on the device node, given by 99-insta360.rules).
  bool reset_usb(const std::string& node) {
    const int fd = ::open(node.c_str(), O_WRONLY);
    if (fd < 0) {
      RCLCPP_ERROR(get_logger(), "cannot open %s for a USB reset: %s", node.c_str(), std::strerror(errno));
      return false;
    }
    const int rc = ::ioctl(fd, USBDEVFS_RESET, 0);
    const int err = errno;
    ::close(fd);
    if (rc < 0) RCLCPP_ERROR(get_logger(), "USB reset of %s failed: %s", node.c_str(), std::strerror(err));
    return rc == 0;
  }

  // Connect, stream and reconnect when the camera goes away.
  void camera_loop() {
    std::string waiting_logged;
    int failures = 0;
    auto last_reset = std::chrono::steady_clock::now() - 1h;
    while (running_ && rclcpp::ok()) {
      ins_camera::DeviceDiscovery discovery;
      auto list = discovery.GetAvailableDevices();
      if (list.empty()) {
        discovery.FreeDeviceDescriptors(list);
        const UsbCamera usb = find_usb_camera();
        // 1005 = mass storage: no control interface, only the camera menu can change it.
        const std::string state = usb.pid.empty() ? "absent" : usb.pid == "1005" ? "storage" : "hung";
        if (waiting_logged != state) {
          if (state == "absent")
            RCLCPP_WARN(get_logger(), "no Insta360 on USB; waiting (cable, USB 3.0 port, camera powered on?)");
          else if (state == "storage")
            RCLCPP_ERROR(get_logger(), "camera is in U-disk (mass storage) mode 2e1a:1005; select Android mode "
                                       "on the camera, it connects by itself afterwards");
          else
            RCLCPP_WARN(get_logger(), "camera 2e1a:%s on USB but not answering the SDK handshake", usb.pid.c_str());
          waiting_logged = state;
        }
        if (state == "hung" && ++failures >= 3 && std::chrono::steady_clock::now() - last_reset > 30s) {
          RCLCPP_WARN(get_logger(), "resetting the camera USB device %s", usb.node.c_str());
          reset_usb(usb.node);
          last_reset = std::chrono::steady_clock::now();
          failures = 0;
          waiting_logged.clear();
          sleep_while_running(5s);
          continue;
        }
        sleep_while_running(2s);
        continue;
      }
      waiting_logged.clear();
      failures = 0;
      RCLCPP_INFO(get_logger(), "camera %s serial %s fw %s", list[0].camera_name.c_str(),
                  list[0].serial_number.c_str(), list[0].fw_version.c_str());
      camera_name_ = list[0].camera_name;
      camera_serial_ = list[0].serial_number;
      camera_fw_ = list[0].fw_version;
      auto cam = std::make_shared<ins_camera::Camera>(list[0].info);
      // The connection info points into the descriptors; free them only after Open().
      const bool opened = cam->Open();
      discovery.FreeDeviceDescriptors(list);
      if (!opened) {
        RCLCPP_ERROR(get_logger(), "camera Open() failed; retrying");
        sleep_while_running(2s);
        continue;
      }
      if (video_submode_ != "keep") {
        const auto mode = video_submode_ == "liveview" ? ins_camera::SubVideoMode::VIDEO_LIVEVIEW
                                                       : ins_camera::SubVideoMode::VIDEO_NORMAL;
        if (!cam->SetVideoSubMode(mode)) RCLCPP_WARN(get_logger(), "SetVideoSubMode(%s) failed", video_submode_.c_str());
      }
      stream(cam);
      cam->Close();
      if (running_) sleep_while_running(2s);
    }
  }

  void stream(const std::shared_ptr<ins_camera::Camera>& cam) {
    const auto preview = cam->GetPreviewParam();
    std::vector<std::string> calib;
    for (size_t i = 0; i < preview.GetCalibrationCount(); ++i) calib.push_back(preview.GetCalibration(i));
    StitchDelegate::VideoSink video;
    StitchDelegate::GyroSink gyro;
    StitchDelegate::ExposureSink exposure;
    std::function<void()> start_stitch = [] {}, stop_stitch = [] {};
    std::string stitch_desc = backend_;
#ifdef HAVE_MEDIA_SDK
    if (backend_ == "mediasdk") {
      auto stitcher = std::make_shared<ins::RealTimeStitcher>();
      ins::CameraInfo info;
      info.cameraName = preview.camera_name;
      info.decode_type = static_cast<ins::VideoDecodeType>(preview.encode_type);
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
          [this](uint8_t* data[4], int linesize[4], int width, int height, int format, int64_t timestamp) {
            on_frame(data, linesize, width, height, format, timestamp);
          });
      video = [stitcher](const uint8_t* data, size_t size, int64_t ts, uint8_t type, int index) {
        stitcher->HandleVideoData(data, size, ts, type, index);
      };
      gyro = [stitcher](const std::vector<ins_camera::GyroData>& data) {
        std::vector<ins::GyroData> v(data.size());
        std::memcpy(v.data(), data.data(), data.size() * sizeof(ins_camera::GyroData));
        stitcher->HandleGyroData(v);
      };
      exposure = [stitcher](const ins_camera::ExposureData& data) {
        ins::ExposureData e{};
        e.exposure_time = data.exposure_time;
        e.timestamp = data.timestamp;
        stitcher->HandleExposureData(e);
      };
      start_stitch = [stitcher] { stitcher->StartStitch(); };
      stop_stitch = [stitcher] { stitcher->CancelStitch(); };
      stitch_desc = "mediasdk " + stitch_type_;
    }
#endif
#ifdef HAVE_CUDA_STITCH
    if (backend_ == "cuda") {
      insta360_cuda_stitch::LiveStitcherOptions opt;
      try {
        opt.calibration = insta360_cuda_stitch::parse_calibration(
            calib, preview.GetCropSrcWidth(), preview.GetCropDstWidth(), preview.GetCropOffsetX(), preview.GetCropOffsetY());
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "preview calibration unusable for the CUDA stitcher: %s", e.what());
        return;
      }
      opt.out_width = eq_width_;
      opt.h265 = preview.encode_type == ins_camera::VideoEncodeType::H265;
      opt.decoder = cuda_decoder_;
      auto stitcher = std::make_shared<insta360_cuda_stitch::LiveStitcher>(
          opt,
          [this](const uint8_t* rgb, int width, int height, int64_t device_ts) {
            uint8_t* data[4] = {const_cast<uint8_t*>(rgb), nullptr, nullptr, nullptr};
            int linesize[4] = {width * 3, 0, 0, 0};
            on_frame(data, linesize, width, height, kAvPixFmtRgb24, device_ts);
          },
          [this](const std::string& text) { RCLCPP_ERROR(get_logger(), "cuda stitcher: %s", text.c_str()); });
      video = [stitcher](const uint8_t* data, size_t size, int64_t ts, uint8_t, int index) {
        if (index == 0) stitcher->push(data, size, ts);
      };
      stitch_desc = "cuda: " + stitcher->pipeline();
    }
#endif
    if (recorder_) {
      const std::string seg = recorder_->start_segment(preview, camera_name_, camera_serial_, camera_fw_, resolution_,
                                                       bitrate_mbps_, now().seconds());
      RCLCPP_INFO(get_logger(), "raw segment %s", seg.c_str());
    }
    std::shared_ptr<ins_camera::StreamDelegate> delegate = std::make_shared<StitchDelegate>(
        video, gyro, exposure, [this](const std::vector<ins_camera::GyroData>& data) { publish_imu(data); }, recorder_,
        [this]() { return now().seconds(); });
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
    start_stitch();
    RCLCPP_INFO(get_logger(), "live stream started (%s, %s), camera %s", resolution_.c_str(), stitch_desc.c_str(),
                preview.camera_name.c_str());

    const int64_t started = now().nanoseconds();
    int64_t last_raw_report = started;
    last_frame_ns_ = 0;
    while (running_ && rclcpp::ok()) {
      std::this_thread::sleep_for(500ms);
      if (recorder_ && now().nanoseconds() - last_raw_report > 10'000'000'000LL) {
        last_raw_report = now().nanoseconds();
        RCLCPP_INFO(get_logger(), "raw stream recorded %.1f MB", recorder_->bytes() / 1e6);
      }
      const int64_t last = last_frame_ns_.load();
      const int64_t ref = last > 0 ? last : started;
      if (now().nanoseconds() - ref > 5'000'000'000LL) {
        RCLCPP_WARN(get_logger(), "no stitched frame for 5 s; reconnecting");
        break;
      }
    }
    cam->StopLiveStreaming();
    stop_stitch();
    if (recorder_) recorder_->close();
  }

  void sleep_while_running(std::chrono::milliseconds d) {
    const auto end = std::chrono::steady_clock::now() + d;
    while (running_ && rclcpp::ok() && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(100ms);
  }

  std::string model_dir_, resolution_, stitch_type_, frame_prefix_, raw_dump_dir_, backend_, cuda_decoder_, video_submode_;
  std::string camera_name_, camera_serial_, camera_fw_;
  std::shared_ptr<RawRecorder> recorder_;
  double bitrate_mbps_, hfov_deg_, front_yaw_deg_, rate_hz_;
  int eq_width_, view_size_, jpeg_quality_;
  bool publish_equirect_, publish_vio_topics_, software_decode_;
  int vio_image_stride_ = 1;
  int vio_frame_ = 0;
  std::vector<View> views_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_equirect_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_equirect_raw_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_imu_;
  std::mutex clock_mutex_;
  DeviceClock imu_clock_{0.005};
  DeviceClock video_clock_{1.0 / 30.0};
  double last_imu_stamp_ = 0.0;
  int64_t last_clock_report_ns_ = 0;
  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  RawFrame latest_;
  bool has_frame_ = false;
  std::atomic<bool> running_{true};
  std::atomic<bool> format_logged_{false}, format_warned_{false}, layout_warned_{false};
  std::atomic<int64_t> last_frame_ns_{0};
  std::mutex eq_mutex_;
  std::condition_variable eq_cv_;
  Frame eq_pending_;
  bool eq_has_ = false, eq_busy_ = false;
  std::thread worker_, camera_thread_, equirect_thread_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
#ifdef HAVE_MEDIA_SDK
  ins::InitEnv();
  ins::SetLogLevel(ins::InsLogLevel::WARNING);
#endif
  ins_camera::SetLogLevel(ins_camera::LogLevel::WARNING);
  auto node = std::make_shared<Insta360Views>();
  rclcpp::spin(node);
  node.reset();
  rclcpp::shutdown();
  // libMNN.so (pulled in by libMediaSDK.so) corrupts the heap in its global
  // destructors, so skip them once everything above has shut down.
  std::fflush(nullptr);
  std::_Exit(0);
}
