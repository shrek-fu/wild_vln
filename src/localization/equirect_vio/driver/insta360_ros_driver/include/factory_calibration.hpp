#ifndef INSTA360_FACTORY_CALIBRATION_HPP
#define INSTA360_FACTORY_CALIBRATION_HPP

#include <string>
#include <vector>

#include "insta360_ros_driver/msg/factory_calibration.hpp"

// Fills intrinsics and lens extrinsics from the three CameraSDK offset strings.
// Returns false when the v1 string is missing or not the 2-lens layout.
inline bool fillFactoryCalibration(
    const std::vector<std::string>& offsets,
    insta360_ros_driver::msg::FactoryCalibration& out) {
    auto split = [](const std::string& text) {
        std::vector<double> values;
        std::string token;
        for (size_t i = 0; i <= text.size(); ++i) {
            if (i == text.size() || text[i] == '_') {
                if (!token.empty()) {
                    values.push_back(std::stod(token));
                    token.clear();
                }
            } else {
                token.push_back(text[i]);
            }
        }
        return values;
    };

    bool have_v1 = false;
    for (const auto& offset : offsets) {
        if (offset.empty()) {
            continue;
        }
        std::vector<double> values;
        try {
            values = split(offset);
        } catch (const std::exception&) {
            continue;
        }
        if (values.size() == 16 && static_cast<int>(values[0]) == 2) {
            out.offset_v1 = offset;
            out.calib_width = static_cast<uint32_t>(values[13]);
            out.calib_height = static_cast<uint32_t>(values[14]);
            for (int lens = 0; lens < 2; ++lens) {
                const int base = 1 + lens * 6;
                out.radius[lens] = values[base];
                out.cx[lens] = values[base + 1];
                out.cy[lens] = values[base + 2];
                out.yaw_deg[lens] = values[base + 3];
                out.pitch_deg[lens] = values[base + 4];
                out.roll_deg[lens] = values[base + 5];
            }
            have_v1 = true;
        } else if (values.size() >= 33 && values.size() <= 34 && static_cast<int>(values[0]) == 2) {
            out.offset_v2 = offset;
            out.has_v2 = true;
            for (int lens = 0; lens < 2; ++lens) {
                const int base = 1 + lens * 16;
                out.v2_qx[lens] = values[base + 6];
                out.v2_qy[lens] = values[base + 7];
                out.v2_qz[lens] = values[base + 8];
                out.v2_qw[lens] = values[base + 9];
                out.v2_tx[lens] = values[base + 10];
                out.v2_ty[lens] = values[base + 11];
                out.v2_tz[lens] = values[base + 12];
            }
        } else if (values.size() >= 39 && static_cast<int>(values[0]) == 2) {
            out.offset_v3 = offset;
            out.has_v3 = true;
            for (int lens = 0; lens < 2; ++lens) {
                const int base = 1 + lens * 19;
                out.v3_xi[lens] = values[base];
                out.v3_fx[lens] = values[base + 1];
                out.v3_fy[lens] = values[base + 2];
                out.v3_k1[lens] = values[base + 11];
                out.v3_k2[lens] = values[base + 12];
                out.v3_k3[lens] = values[base + 13];
                out.v3_p1[lens] = values[base + 14];
                out.v3_p2[lens] = values[base + 15];
            }
        }
    }
    return have_v1;
}

#endif
