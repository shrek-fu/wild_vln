// Camera SDK H.264/H.265 packets -> GStreamer decode -> CUDA equirect stitch -> RGB24 callback.
// Drop-in for the part of ins::RealTimeStitcher that insta360_views uses, without Media SDK.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "insta360_cuda_stitch/stitch_model.hpp"

namespace insta360_cuda_stitch {

struct LiveStitcherOptions {
  Calibration calibration;
  int out_width = 1920;
  bool h265 = false;
  // Between the parser and the appsink; must end in raw RGBA, BGRA, RGBx, BGRx, I420 or NV12, or
  // pitch-linear NV12 in NVMM (Jetson builds with HAVE_NVMM). Empty picks nvv4l2decoder + nvvidconv
  // to NVMM NV12 (or I420 in system memory without NVMM) on Jetson, else avdec + videoconvert.
  std::string decoder;
  // Offline: every decoded frame reaches the callback and push() blocks instead of dropping.
  bool lossless = false;
};

class LiveStitcher {
 public:
  // rgb: out_width * 3 bytes per row; valid only during the call.
  using FrameCallback = std::function<void(const uint8_t* rgb, int width, int height, int64_t device_ts)>;
  using ErrorCallback = std::function<void(const std::string&)>;

  LiveStitcher(LiveStitcherOptions options, FrameCallback on_frame, ErrorCallback on_error = nullptr);
  ~LiveStitcher();
  LiveStitcher(const LiveStitcher&) = delete;
  LiveStitcher& operator=(const LiveStitcher&) = delete;

  // One encoded access unit as delivered by ins_camera::StreamDelegate::OnVideoData (stream 0).
  void push(const uint8_t* data, size_t size, int64_t device_ts);
  const std::string& pipeline() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace insta360_cuda_stitch
