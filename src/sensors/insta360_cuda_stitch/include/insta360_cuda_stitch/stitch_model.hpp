// Dual-fisheye -> equirectangular geometry from the Insta360 Camera SDK preview calibration.
//
// The third PreviewParam calibration string ("2_<19 values lens 0>_<19 values lens 1>_<id>")
// holds a unified omnidirectional (Mei) model per lens:
//   xi fx fy cx cy yaw pitch roll a b c k1 k2 k3 p1 p2 W H id
// in pixels of a (2 * src) x src calibration image, src = PreviewParam crop src width.
// The stream shows each lens as a (dst = crop dst width) square cut from the middle of its
// src square, scaled to half the stream width. Fitted against Media SDK "template" output of
// an X4 Air (tools/fit_stitch.py): NCC 0.994-0.999 over the full panorama.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace insta360_cuda_stitch {

struct LensModel {
  double xi = 0, fx = 0, fy = 0, cx = 0, cy = 0;
  double yaw_deg = 0, pitch_deg = 0, roll_deg = 0;
  double k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
};

struct Calibration {
  std::array<LensModel, 2> lens;
  double src_size = 0;  // per-lens calibration square (7680 for the X4 Air)
  double dst_size = 0;  // per-lens square that reaches the stream (7424)
  double offset_x = 0, offset_y = 0;
};

// Throws std::invalid_argument when no usable string is present.
Calibration parse_calibration(const std::vector<std::string>& offsets, int crop_src_width, int crop_dst_width,
                              int crop_offset_x, int crop_offset_y);

// Per output pixel: sample positions in the dual-fisheye stream for lens 0 and lens 1, and the
// lens 0 blend weight (lens 1 gets 1 - w0). Output is out_width x out_width / 2, x forward at the
// image centre, longitude growing to the right, as the Media SDK writes it.
struct StitchMaps {
  int in_width = 0, in_height = 0, out_width = 0, out_height = 0;
  std::vector<float> u0, v0, u1, v1, w0;
};

StitchMaps build_maps(const Calibration& calib, int in_width, int in_height, int out_width,
                      double blend_band = 0.05);

}  // namespace insta360_cuda_stitch
