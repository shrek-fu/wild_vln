// Offline check of LiveStitcher on a recorded raw segment (insta360_views raw_dump_dir):
//   stitch_segment <seg_dir> <out_dir> [--width 1920] [--save-seq 0,1332] [--limit N] [--decoder "..."]
//                  [--pipe-every N]
// Feeds stream_0.<h264|h265> packet by packet as the camera would, prints throughput and writes
// the requested frames as <out_dir>/eq_<seq>.ppm.
// --pipe-every N: every Nth stitched frame to stdout as int64 device_ts, int32 width, int32 height,
// width*height*3 RGB bytes (little endian); messages go to stderr. Used by collection_data derive_session.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "insta360_cuda_stitch/live_stitcher.hpp"

using insta360_cuda_stitch::LiveStitcher;
using insta360_cuda_stitch::LiveStitcherOptions;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

int json_int(const std::string& js, const std::string& key) {
  const auto p = js.find("\"" + key + "\"");
  if (p == std::string::npos) throw std::runtime_error("camera.json: no " + key);
  return std::stoi(js.substr(js.find(':', p) + 1));
}

std::vector<std::string> json_calibration(const std::string& js) {
  std::vector<std::string> out;
  auto p = js.find('[', js.find("\"calibration\""));
  const auto end = js.find(']', p);
  while ((p = js.find('"', p + 1)) < end) {
    const auto q = js.find('"', p + 1);
    out.push_back(js.substr(p + 1, q - p - 1));
    p = q;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <seg_dir> <out_dir> [--width 1920] [--save-seq a,b] [--limit N] [--decoder D]\n";
    return 2;
  }
  const std::string seg = argv[1], out_dir = argv[2];
  int width = 1920;
  long limit = -1, pipe_every = 0;
  std::set<long> save;
  std::string decoder;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--width") width = std::stoi(v);
    else if (k == "--limit") limit = std::stol(v);
    else if (k == "--decoder") decoder = v;
    else if (k == "--pipe-every") pipe_every = std::stol(v);
    else if (k == "--save-seq") {
      std::stringstream ss(v);
      std::string t;
      while (std::getline(ss, t, ',')) save.insert(std::stol(t));
    }
  }
  // The Jetson decoder libraries print to fd 1, so the frame stream gets its own copy of it.
  FILE* pipe_out = nullptr;
  if (pipe_every > 0) {
    std::fflush(stdout);
    pipe_out = fdopen(dup(STDOUT_FILENO), "wb");
    dup2(STDERR_FILENO, STDOUT_FILENO);
  }

  const std::string js = read_file(seg + "/camera.json");
  LiveStitcherOptions opt;
  opt.calibration = insta360_cuda_stitch::parse_calibration(
      json_calibration(js), json_int(js, "src_width"), json_int(js, "dst_width"), json_int(js, "offset_x"),
      json_int(js, "offset_y"));
  opt.out_width = width;
  opt.h265 = js.find("\"encode\": \"h265\"") != std::string::npos;
  opt.decoder = decoder;
  opt.lossless = pipe_every > 0;

  struct Packet {
    long seq;
    int64_t ts;
    uint64_t offset;
    size_t size;
  };
  std::vector<Packet> packets;
  std::map<int64_t, long> seq_of_ts;
  {
    std::ifstream idx(seg + "/video_index.csv");
    std::string line;
    std::getline(idx, line);
    while (std::getline(idx, line)) {
      std::stringstream ss(line);
      std::vector<std::string> f;
      std::string t;
      while (std::getline(ss, t, ',')) f.push_back(t);
      if (f.size() < 8 || f[1] != "0") continue;
      packets.push_back({std::stol(f[0]), std::stoll(f[3]), std::stoull(f[6]), std::stoul(f[7])});
      seq_of_ts[packets.back().ts] = packets.back().seq;
    }
  }
  if (limit > 0 && static_cast<long>(packets.size()) > limit) packets.resize(limit);
  std::ifstream stream(seg + (opt.h265 ? "/stream_0.h265" : "/stream_0.h264"), std::ios::binary);

  std::atomic<long> frames{0}, matched{0};
  std::mutex io;
  std::ostream& log = pipe_every > 0 ? std::cerr : std::cout;
  auto t0 = std::chrono::steady_clock::now();
  std::atomic<double> first_frame_s{-1};
  LiveStitcher stitcher(
      opt,
      [&](const uint8_t* rgb, int w, int h, int64_t ts) {
        if (frames++ == 0) {
          first_frame_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        auto it = seq_of_ts.find(ts);
        if (it == seq_of_ts.end()) return;
        const long n = matched++;
        if (pipe_every > 0) {
          if (n % pipe_every) return;
          std::lock_guard<std::mutex> lock(io);
          const int64_t t = ts;
          const int32_t dims[2] = {w, h};
          std::fwrite(&t, sizeof t, 1, pipe_out);
          std::fwrite(dims, sizeof dims, 1, pipe_out);
          std::fwrite(rgb, 1, static_cast<size_t>(w) * h * 3, pipe_out);
          return;
        }
        if (!save.count(it->second)) return;
        std::lock_guard<std::mutex> lock(io);
        const std::string path = out_dir + "/eq_" + std::to_string(it->second) + ".ppm";
        std::ofstream o(path, std::ios::binary);
        o << "P6\n" << w << " " << h << "\n255\n";
        o.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(w) * h * 3);
        log << "wrote " << path << "\n";
      },
      [&](const std::string& e) {
        std::lock_guard<std::mutex> lock(io);
        std::cerr << "stitcher: " << e << "\n";
      });
  log << "pipeline: " << stitcher.pipeline() << "\n";

  std::vector<uint8_t> buf;
  t0 = std::chrono::steady_clock::now();
  for (const auto& p : packets) {
    buf.resize(p.size);
    stream.seekg(static_cast<std::streamoff>(p.offset));
    stream.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(p.size));
    stitcher.push(buf.data(), buf.size(), p.ts);
    if (opt.lossless) continue;  // push() blocks
    // Keep the decoder queue short, as the 30 Hz camera would.
    while (static_cast<long>(&p - packets.data()) - frames.load() > 8) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (opt.lossless) {
    for (long last = -1, idle = 0; idle < 20 && frames.load() < static_cast<long>(packets.size()); ++idle) {
      if (frames.load() != last) idle = 0, last = frames.load();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  for (int i = 0; i < 300 && frames.load() < static_cast<long>(packets.size()) - 2; ++i) {
    if (opt.lossless) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (pipe_out) std::fflush(pipe_out);
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(pipe_every > 0 ? stderr : stdout, "packets %zu frames %ld (timestamp matched %ld) in %.2f s -> %.1f fps, first frame after %.3f s\n",
              packets.size(), frames.load(), matched.load(), dt, frames.load() / dt, first_frame_s.load());
  return 0;
}
