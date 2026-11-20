// Still capture for Insta360 X4 Air via the desktop Camera SDK.
// Build only when INSTA360_SDK_ROOT points at include/ and lib/.
#include <camera/camera.h>
#include <camera/device_discovery.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  const std::string out_dir = argc > 1 ? argv[1] : ".";
  ins_camera::DeviceDiscovery discovery;
  auto list = discovery.GetAvailableDevices();
  if (list.empty()) {
    std::cerr << "no Insta360 camera. Use USB 3.0 and choose Android mode.\n";
    return 1;
  }
  std::cout << "camera " << list[0].camera_name << " serial " << list[0].serial_number
            << " fw " << list[0].fw_version << "\n";
  auto cam = std::make_shared<ins_camera::Camera>(list[0].info);
  // The connection info points into the descriptors; free them only after Open().
  const bool opened = cam->Open();
  discovery.FreeDeviceDescriptors(list);
  if (!opened) {
    std::cerr << "Open() failed\n";
    return 1;
  }
  auto url = cam->TakePhoto();
  const std::string remote = url.GetSingleOrigin();
  if (remote.empty()) {
    std::cerr << "TakePhoto returned an empty path\n";
    cam->Close();
    return 1;
  }
  const auto slash = remote.find_last_of('/');
  const std::string name = slash == std::string::npos ? remote : remote.substr(slash + 1);
  const std::string local = out_dir + "/" + name;
  std::cout << "downloading " << remote << " -> " << local << "\n";
  const bool ok = cam->DownloadCameraFile(
      remote, local, [](int64_t current, int64_t total) {
        std::cout << current << "/" << total << "\n";
      });
  cam->Close();
  if (!ok) {
    std::cerr << "DownloadCameraFile failed\n";
    return 1;
  }
  std::cout << "saved " << local << "\n";
  return 0;
}
