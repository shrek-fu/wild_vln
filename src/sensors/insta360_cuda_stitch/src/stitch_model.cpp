#include "insta360_cuda_stitch/stitch_model.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace insta360_cuda_stitch {
namespace {

using Mat3 = std::array<std::array<double, 3>, 3>;

Mat3 mul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) r[i][j] += a[i][k] * b[k][j];
  return r;
}

Mat3 rot(char axis, double deg) {
  const double c = std::cos(deg * M_PI / 180.0), s = std::sin(deg * M_PI / 180.0);
  if (axis == 'x') return {{{1, 0, 0}, {0, c, -s}, {0, s, c}}};
  if (axis == 'y') return {{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}};
  return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
}

// Lens axes (columns) in the rig frame: x forward, y left, z up. Lens 0 looks backwards.
Mat3 lens_rotation(const LensModel& m, int lens) {
  static const Mat3 base[2] = {{{{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}}, {{{0, 0, 1}, {0, -1, 0}, {1, 0, 0}}}};
  const double pitch_sign = lens == 0 ? 1.0 : -1.0;
  return mul(mul(mul(base[lens], rot('y', -m.yaw_deg)), rot('x', pitch_sign * m.pitch_deg)), rot('z', m.roll_deg));
}

std::vector<double> split(const std::string& s) {
  std::vector<double> v;
  std::stringstream in(s);
  std::string tok;
  while (std::getline(in, tok, '_')) {
    if (!tok.empty()) v.push_back(std::stod(tok));
  }
  return v;
}

}  // namespace

Calibration parse_calibration(const std::vector<std::string>& offsets, int crop_src_width, int crop_dst_width,
                              int crop_offset_x, int crop_offset_y) {
  for (const auto& s : offsets) {
    std::vector<double> v;
    try {
      v = split(s);
    } catch (const std::exception&) {
      continue;
    }
    if (v.size() < 39 || static_cast<int>(v[0]) != 2) continue;
    Calibration c;
    for (int i = 0; i < 2; ++i) {
      const double* p = v.data() + 1 + 19 * i;
      LensModel& m = c.lens[i];
      m.xi = p[0];
      m.fx = p[1];
      m.fy = p[2];
      m.cx = p[3];
      m.cy = p[4];
      m.yaw_deg = p[5];
      m.pitch_deg = p[6];
      m.roll_deg = p[7];
      m.k1 = p[11];
      m.k2 = p[12];
      m.k3 = p[13];
      m.p1 = p[14];
      m.p2 = p[15];
    }
    const double calib_h = v[1 + 17];
    c.src_size = crop_src_width > 0 ? crop_src_width : calib_h;
    c.dst_size = crop_dst_width > 0 ? crop_dst_width : c.src_size;
    c.offset_x = crop_offset_x;
    c.offset_y = crop_offset_y;
    return c;
  }
  throw std::invalid_argument("no 2-lens omnidirectional calibration string in PreviewParam");
}

StitchMaps build_maps(const Calibration& calib, int in_width, int in_height, int out_width, double blend_band) {
  StitchMaps maps;
  maps.in_width = in_width;
  maps.in_height = in_height;
  maps.out_width = out_width;
  maps.out_height = out_width / 2;
  const size_t n = static_cast<size_t>(maps.out_width) * maps.out_height;
  for (auto* v : {&maps.u0, &maps.v0, &maps.u1, &maps.v1, &maps.w0}) v->resize(n);

  const Mat3 R[2] = {lens_rotation(calib.lens[0], 0), lens_rotation(calib.lens[1], 1)};
  const double half = in_width / 2.0;
  const double scale = half / calib.dst_size;
  const double crop_x = 0.5 * (calib.src_size - calib.dst_size) + calib.offset_x;
  const double crop_y = 0.5 * (calib.src_size - calib.dst_size) + calib.offset_y;
  float* us[2] = {maps.u0.data(), maps.u1.data()};
  float* vs[2] = {maps.v0.data(), maps.v1.data()};

  for (int y = 0; y < maps.out_height; ++y) {
    const double lat = M_PI / 2 - (y + 0.5) / maps.out_height * M_PI;
    for (int x = 0; x < maps.out_width; ++x) {
      const double lon = (x + 0.5) / maps.out_width * 2 * M_PI - M_PI;
      const double d[3] = {std::cos(lat) * std::cos(lon), -std::cos(lat) * std::sin(lon), std::sin(lat)};
      const size_t i = static_cast<size_t>(y) * maps.out_width + x;
      double axis_cos[2];
      for (int l = 0; l < 2; ++l) {
        const LensModel& m = calib.lens[l];
        double p[3];
        for (int k = 0; k < 3; ++k) p[k] = R[l][0][k] * d[0] + R[l][1][k] * d[1] + R[l][2][k] * d[2];
        axis_cos[l] = p[2];
        const double z = p[2] + m.xi;
        const double mx = p[0] / z, my = p[1] / z;
        const double r2 = mx * mx + my * my;
        const double rad = 1 + m.k1 * r2 + m.k2 * r2 * r2 + m.k3 * r2 * r2 * r2;
        const double xd = mx * rad + 2 * m.p1 * mx * my + m.p2 * (r2 + 2 * mx * mx);
        const double yd = my * rad + m.p1 * (r2 + 2 * my * my) + 2 * m.p2 * mx * my;
        const double u = m.fx * xd + m.cx - l * calib.src_size;
        const double v = m.fy * yd + m.cy;
        us[l][i] = static_cast<float>((u - crop_x) * scale + l * half);
        vs[l][i] = static_cast<float>((v - crop_y) * scale);
      }
      maps.w0[i] = static_cast<float>(std::clamp(0.5 + (axis_cos[0] - axis_cos[1]) / (2 * blend_band), 0.0, 1.0));
    }
  }
  return maps;
}

}  // namespace insta360_cuda_stitch
