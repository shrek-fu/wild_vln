# Third-party notice

## OpenVIO

Path: `third_party/openvio`

Upstream: https://github.com/93won/openvio  
License: MIT  
Copyright (c) Seungwon Choi and contributors

Vendored into EquirectangularVIO so the X4 Air VIO stack is self-contained for porting.
Local changes (if any) are limited to portable CMake Release flags for non-x86 hosts.

Bundled inside OpenVIO:

- Ceres Solver — see `openvio/thirdparty/ceres-solver`
- spdlog — see `openvio/thirdparty/spdlog`
- imgui / implot — see `openvio/thirdparty/imgui`, `implot`
