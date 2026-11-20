#include "insta360_cuda_stitch/cuda_remap.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace insta360_cuda_stitch {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Bilinear sample of channels [0, n) with `step` bytes per pixel, clamped to the lens columns
// [x_lo, x_hi] so the other lens never bleeds in.
template <int n>
__device__ void bilinear(const uint8_t* img, size_t pitch, int step, float x, float y, int x_lo, int x_hi, int h,
                         float* c) {
  x = fminf(fmaxf(x, static_cast<float>(x_lo)), static_cast<float>(x_hi));
  y = fminf(fmaxf(y, 0.f), static_cast<float>(h - 1));
  const int x0 = min(static_cast<int>(floorf(x)), x_hi - 1);
  const int y0 = min(static_cast<int>(floorf(y)), h - 2);
  const float fx = x - x0, fy = y - y0;
  const uint8_t* r0 = img + static_cast<size_t>(y0) * pitch + static_cast<size_t>(x0) * step;
  const uint8_t* r1 = r0 + pitch;
  for (int k = 0; k < n; ++k) {
    const float top = r0[k] + fx * (r0[step + k] - r0[k]);
    const float bot = r1[k] + fx * (r1[step + k] - r1[k]);
    c[k] = top + fy * (bot - top);
  }
}

struct RgbaSource {
  const uint8_t* img;
  size_t pitch;
  bool bgr;
  __device__ float3 operator()(float x, float y, int x_lo, int x_hi, int h) const {
    float c[3];
    bilinear<3>(img, pitch, 4, x, y, x_lo, x_hi, h, c);
    return bgr ? make_float3(c[2], c[1], c[0]) : make_float3(c[0], c[1], c[2]);
  }
};

struct YuvSource {
  const uint8_t* y;
  size_t y_pitch;
  const uint8_t* u;
  const uint8_t* v;
  size_t c_pitch;
  int c_step;  // 1 for I420, 2 for NV12
  YuvMatrix m;
  __device__ float3 operator()(float x, float yy, int x_lo, int x_hi, int h) const {
    float l, cb, cr;
    bilinear<1>(y, y_pitch, 1, x, yy, x_lo, x_hi, h, &l);
    // Chroma sits at the centre of each 2x2 luma block.
    const float cx = (x + 0.5f) * 0.5f - 0.5f, cy = (yy + 0.5f) * 0.5f - 0.5f;
    const int c_lo = x_lo / 2, c_hi = (x_hi + 1) / 2 - 1, ch = h / 2;
    bilinear<1>(u, c_pitch, c_step, cx, cy, c_lo, c_hi, ch, &cb);
    bilinear<1>(v, c_pitch, c_step, cx, cy, c_lo, c_hi, ch, &cr);
    const float Y = (l - m.y_offset) * m.y_scale;
    const float U = (cb - 128.f) * m.c_scale, V = (cr - 128.f) * m.c_scale;
    return make_float3(Y + m.rv * V, Y - m.gu * U - m.gv * V, Y + m.bu * U);
  }
};

__device__ uint8_t to8(float v) { return static_cast<uint8_t>(fminf(fmaxf(v + 0.5f, 0.f), 255.f)); }

template <class Source>
__global__ void stitch_kernel(Source src, int in_w, int in_h, const float4* map, const float* w0, uint8_t* out,
                              size_t out_pitch, int out_w, int out_h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_w || y >= out_h) return;
  const size_t i = static_cast<size_t>(y) * out_w + x;
  const float4 m = map[i];
  const float w = w0[i];
  const int half = in_w / 2;
  float3 c = make_float3(0.f, 0.f, 0.f);
  if (w > 0.f) {
    const float3 a = src(m.x, m.y, 0, half - 1, in_h);
    c = make_float3(w * a.x, w * a.y, w * a.z);
  }
  if (w < 1.f) {
    const float3 b = src(m.z, m.w, half, in_w - 1, in_h);
    const float v = 1.f - w;
    c = make_float3(c.x + v * b.x, c.y + v * b.y, c.z + v * b.z);
  }
  uint8_t* o = out + static_cast<size_t>(y) * out_pitch + static_cast<size_t>(x) * 3;
  o[0] = to8(c.x);
  o[1] = to8(c.y);
  o[2] = to8(c.z);
}

}  // namespace

YuvMatrix YuvMatrix::make(bool bt709, bool full_range) {
  YuvMatrix m{};
  m.y_offset = full_range ? 0.f : 16.f;
  m.y_scale = full_range ? 1.f : 255.f / 219.f;
  m.c_scale = full_range ? 1.f : 255.f / 224.f;
  if (bt709) {
    m.rv = 1.5748f, m.gu = 0.187324f, m.gv = 0.468124f, m.bu = 1.8556f;
  } else {
    m.rv = 1.402f, m.gu = 0.344136f, m.gv = 0.714136f, m.bu = 1.772f;
  }
  return m;
}

CudaRemap::CudaRemap(const StitchMaps& maps)
    : in_w_(maps.in_width), in_h_(maps.in_height), out_w_(maps.out_width), out_h_(maps.out_height) {
  const size_t n = static_cast<size_t>(out_w_) * out_h_;
  std::vector<float4> packed(n);
  for (size_t i = 0; i < n; ++i) packed[i] = make_float4(maps.u0[i], maps.v0[i], maps.u1[i], maps.v1[i]);
  cudaStream_t s;
  check(cudaStreamCreate(&s), "cudaStreamCreate");
  stream_ = s;
  check(cudaMalloc(&map_, n * sizeof(float4)), "cudaMalloc map");
  check(cudaMalloc(&w0_, n * sizeof(float)), "cudaMalloc w0");
  check(cudaMemcpy(map_, packed.data(), n * sizeof(float4), cudaMemcpyHostToDevice), "upload map");
  check(cudaMemcpy(w0_, maps.w0.data(), n * sizeof(float), cudaMemcpyHostToDevice), "upload w0");
  check(cudaMallocPitch(reinterpret_cast<void**>(&in_dev_), &in_dev_pitch_, static_cast<size_t>(in_w_) * 4, in_h_),
        "cudaMallocPitch in");
  check(cudaMallocPitch(reinterpret_cast<void**>(&out_dev_), &out_dev_pitch_, static_cast<size_t>(out_w_) * 3, out_h_),
        "cudaMallocPitch out");
}

CudaRemap::~CudaRemap() {
  cudaFree(map_);
  cudaFree(w0_);
  cudaFree(in_dev_);
  cudaFree(c0_dev_);
  cudaFree(out_dev_);
  if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
}

void CudaRemap::run(const uint8_t* in, size_t in_pitch, bool bgr, uint8_t* out, size_t out_pitch) {
  auto s = static_cast<cudaStream_t>(stream_);
  check(cudaMemcpy2DAsync(in_dev_, in_dev_pitch_, in, in_pitch, static_cast<size_t>(in_w_) * 4, in_h_,
                          cudaMemcpyHostToDevice, s),
        "upload frame");
  launch(RgbaSource{in_dev_, in_dev_pitch_, bgr}, out, out_pitch);
}

void CudaRemap::run_yuv(const uint8_t* y, size_t y_pitch, const uint8_t* u, size_t u_pitch, const uint8_t* v,
                        size_t v_pitch, bool nv12, const YuvMatrix& m, uint8_t* out, size_t out_pitch) {
  const int cw = in_w_ / 2, ch = in_h_ / 2;
  if (!yuv_) {
    check(cudaMallocPitch(reinterpret_cast<void**>(&c0_dev_), &c0_pitch_, static_cast<size_t>(cw) * 2, ch),
          "cudaMallocPitch chroma");
    yuv_ = true;
  }
  auto s = static_cast<cudaStream_t>(stream_);
  check(cudaMemcpy2DAsync(in_dev_, in_dev_pitch_, y, y_pitch, static_cast<size_t>(in_w_), in_h_,
                          cudaMemcpyHostToDevice, s),
        "upload Y");
  YuvSource src{in_dev_, in_dev_pitch_, c0_dev_, nullptr, c0_pitch_, 1, m};
  if (nv12) {
    check(cudaMemcpy2DAsync(c0_dev_, c0_pitch_, u, u_pitch, static_cast<size_t>(cw) * 2, ch, cudaMemcpyHostToDevice, s),
          "upload UV");
    src.v = c0_dev_ + 1;
    src.c_step = 2;
  } else {
    // I420: copy V next to U so both share one pitch.
    check(cudaMemcpy2DAsync(c0_dev_, c0_pitch_, u, u_pitch, static_cast<size_t>(cw), ch, cudaMemcpyHostToDevice, s),
          "upload U");
    check(cudaMemcpy2DAsync(c0_dev_ + cw, c0_pitch_, v, v_pitch, static_cast<size_t>(cw), ch, cudaMemcpyHostToDevice, s),
          "upload V");
    src.v = c0_dev_ + cw;
  }
  launch(src, out, out_pitch);
}

void CudaRemap::run_nv12_device(const uint8_t* y, size_t y_pitch, const uint8_t* uv, size_t uv_pitch,
                                const YuvMatrix& m, uint8_t* out, size_t out_pitch) {
  launch(YuvSource{y, y_pitch, uv, uv + 1, uv_pitch, 2, m}, out, out_pitch);
}

template <class Source>
void CudaRemap::launch(const Source& src, uint8_t* out, size_t out_pitch) {
  auto s = static_cast<cudaStream_t>(stream_);
  const dim3 block(32, 8);
  const dim3 grid((out_w_ + block.x - 1) / block.x, (out_h_ + block.y - 1) / block.y);
  stitch_kernel<<<grid, block, 0, s>>>(src, in_w_, in_h_, static_cast<const float4*>(map_), w0_, out_dev_,
                                       out_dev_pitch_, out_w_, out_h_);
  check(cudaGetLastError(), "stitch_kernel");
  check(cudaMemcpy2DAsync(out, out_pitch, out_dev_, out_dev_pitch_, static_cast<size_t>(out_w_) * 3, out_h_,
                          cudaMemcpyDeviceToHost, s),
        "download frame");
  check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
}

uint8_t* CudaRemap::alloc_host(size_t bytes) {
  void* p = nullptr;
  check(cudaMallocHost(&p, bytes), "cudaMallocHost");
  return static_cast<uint8_t*>(p);
}

void CudaRemap::free_host(uint8_t* p) {
  if (p) cudaFreeHost(p);
}

}  // namespace insta360_cuda_stitch
