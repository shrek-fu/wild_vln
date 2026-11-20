// Dual-fisheye frame (host memory; RGBA/BGRA, I420 or NV12) -> equirect RGB24 (host memory) on the GPU.
#pragma once

#include <cstddef>
#include <cstdint>

#include "insta360_cuda_stitch/stitch_model.hpp"

namespace insta360_cuda_stitch {

// Y'CbCr -> R'G'B' for 8-bit video.
struct YuvMatrix {
  float y_offset, y_scale, c_scale;  // (Y - y_offset) * y_scale, (C - 128) * c_scale
  float rv, gu, gv, bu;              // R = Y + rv V, G = Y - gu U - gv V, B = Y + bu U
  static YuvMatrix make(bool bt709, bool full_range);
};

class CudaRemap {
 public:
  explicit CudaRemap(const StitchMaps& maps);
  ~CudaRemap();
  CudaRemap(const CudaRemap&) = delete;
  CudaRemap& operator=(const CudaRemap&) = delete;

  // in: 4 bytes per pixel, R first unless bgr. out: out_width * 3 bytes per row, RGB.
  void run(const uint8_t* in, size_t in_pitch, bool bgr, uint8_t* out, size_t out_pitch);
  // 4:2:0. I420: u and v are separate planes. NV12: u is the interleaved UV plane and v is ignored.
  void run_yuv(const uint8_t* y, size_t y_pitch, const uint8_t* u, size_t u_pitch, const uint8_t* v, size_t v_pitch,
               bool nv12, const YuvMatrix& m, uint8_t* out, size_t out_pitch);
  // NV12 already in device-accessible memory (a mapped NVMM surface); no upload.
  void run_nv12_device(const uint8_t* y, size_t y_pitch, const uint8_t* uv, size_t uv_pitch, const YuvMatrix& m,
                       uint8_t* out, size_t out_pitch);
  // Page-locked host buffer for run outputs; faster to download into than pageable memory.
  static uint8_t* alloc_host(size_t bytes);
  static void free_host(uint8_t* p);

  int in_width() const { return in_w_; }
  int in_height() const { return in_h_; }
  int out_width() const { return out_w_; }
  int out_height() const { return out_h_; }

 private:
  template <class Source>
  void launch(const Source& src, uint8_t* out, size_t out_pitch);

  int in_w_, in_h_, out_w_, out_h_;
  void* map_ = nullptr;   // float4 (u0, v0, u1, v1)
  float* w0_ = nullptr;
  uint8_t* in_dev_ = nullptr;  // RGBA, or the Y plane
  uint8_t* c0_dev_ = nullptr;  // I420: U | V side by side; NV12: UV
  uint8_t* out_dev_ = nullptr;
  size_t in_dev_pitch_ = 0, c0_pitch_ = 0, out_dev_pitch_ = 0;
  bool yuv_ = false;
  void* stream_ = nullptr;
};

}  // namespace insta360_cuda_stitch
