#include "insta360_cuda_stitch/live_stitcher.hpp"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <map>
#include <mutex>
#include <stdexcept>

#include "insta360_cuda_stitch/cuda_remap.hpp"

#ifdef HAVE_NVMM
#include <cuda_egl_interop.h>
#include <cuda_runtime.h>
#include <nvbufsurface.h>
#endif

namespace insta360_cuda_stitch {
namespace {

bool has_element(const char* name) {
  GstElementFactory* f = gst_element_factory_find(name);
  if (!f) return false;
  gst_object_unref(f);
  return true;
}

// Jetson: the decoder's block-linear surface goes through nvvidconv (VIC) to pitch-linear NV12, still
// in NVMM, which CUDA reads in place. Without NVMM support, I420 into system memory is ~1.8x faster
// than RGBA through nvvidconv at 3840x1920 on Orin NX.
std::string default_decoder(bool h265) {
  if (has_element("nvv4l2decoder") && has_element("nvvidconv")) {
#ifdef HAVE_NVMM
    return "nvv4l2decoder enable-max-performance=true disable-dpb=true ! nvvidconv bl-output=false ! "
           "video/x-raw(memory:NVMM),format=NV12";
#else
    return "nvv4l2decoder enable-max-performance=true disable-dpb=true ! nvvidconv ! video/x-raw,format=I420";
#endif
  }
  return std::string(h265 ? "avdec_h265" : "avdec_h264") + " ! videoconvert ! video/x-raw,format=I420";
}

}  // namespace

struct LiveStitcher::Impl {
  LiveStitcherOptions opt;
  FrameCallback on_frame;
  ErrorCallback on_error;
  std::string description;
  GstElement* pipeline = nullptr;
  GstElement* src = nullptr;
  GstElement* sink = nullptr;
  GstBus* bus = nullptr;
  std::unique_ptr<CudaRemap> remap;
  uint8_t* out = nullptr;  // page-locked, out_width * 3 * out_height
  std::mutex ts_mutex;
  std::map<GstClockTime, int64_t> device_ts;  // pts -> camera timestamp
  GstClockTime next_pts = 0;

  ~Impl() { CudaRemap::free_host(out); }

  static GstFlowReturn on_sample(GstAppSink* appsink, gpointer user) {
    auto* self = static_cast<Impl*>(user);
    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;
    try {
      self->stitch(sample);
    } catch (const std::exception& e) {
      if (self->on_error) self->on_error(e.what());
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }

  void ensure_remap(int w, int h) {
    if (remap && remap->in_width() == w && remap->in_height() == h) return;
    remap.reset();
    CudaRemap::free_host(out);
    out = nullptr;
    remap = std::make_unique<CudaRemap>(build_maps(opt.calibration, w, h, opt.out_width));
    out = CudaRemap::alloc_host(static_cast<size_t>(remap->out_width()) * remap->out_height() * 3);
  }

  void stitch(GstSample* sample) {
    GstCaps* caps = gst_sample_get_caps(sample);
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) throw std::runtime_error("decoder caps");
    const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);
    const bool bgr = fmt == GST_VIDEO_FORMAT_BGRA || fmt == GST_VIDEO_FORMAT_BGRx;
    const bool yuv = fmt == GST_VIDEO_FORMAT_I420 || fmt == GST_VIDEO_FORMAT_NV12;
    if (!bgr && !yuv && fmt != GST_VIDEO_FORMAT_RGBA && fmt != GST_VIDEO_FORMAT_RGBx) {
      throw std::runtime_error(std::string("decoder must output RGBA/BGRA/RGBx/BGRx/I420/NV12, got ") +
                               gst_video_format_to_string(fmt));
    }
    const GstVideoColorimetry& c = GST_VIDEO_INFO_COLORIMETRY(&info);
    const auto m = YuvMatrix::make(c.matrix != GST_VIDEO_COLOR_MATRIX_BT601, c.range == GST_VIDEO_COLOR_RANGE_0_255);
    GstBuffer* buf = gst_sample_get_buffer(sample);
    ensure_remap(GST_VIDEO_INFO_WIDTH(&info), GST_VIDEO_INFO_HEIGHT(&info));
    const size_t out_pitch = static_cast<size_t>(remap->out_width()) * 3;
    GstCapsFeatures* features = gst_caps_get_features(caps, 0);
    if (features && gst_caps_features_contains(features, "memory:NVMM")) {
#ifdef HAVE_NVMM
      if (fmt != GST_VIDEO_FORMAT_NV12) throw std::runtime_error("NVMM input must be NV12");
      stitch_nvmm(buf, m, out_pitch);
#else
      throw std::runtime_error("built without NVMM support; end the decoder in system memory");
#endif
    } else {
      GstVideoFrame frame;
      if (!gst_video_frame_map(&frame, &info, buf, GST_MAP_READ)) throw std::runtime_error("map decoded frame");
      const auto plane = [&](int i) { return static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, i)); };
      const auto stride = [&](int i) { return static_cast<size_t>(GST_VIDEO_FRAME_PLANE_STRIDE(&frame, i)); };
      try {
        if (fmt == GST_VIDEO_FORMAT_NV12) {
          remap->run_yuv(plane(0), stride(0), plane(1), stride(1), nullptr, 0, true, m, out, out_pitch);
        } else if (yuv) {
          remap->run_yuv(plane(0), stride(0), plane(1), stride(1), plane(2), stride(2), false, m, out, out_pitch);
        } else {
          remap->run(plane(0), stride(0), bgr, out, out_pitch);
        }
      } catch (...) {
        gst_video_frame_unmap(&frame);
        throw;
      }
      gst_video_frame_unmap(&frame);
    }
    const GstClockTime pts = GST_BUFFER_PTS(buf);
    int64_t ts = -1;
    {
      std::lock_guard<std::mutex> lock(ts_mutex);
      auto it = device_ts.find(pts);
      if (it != device_ts.end()) {
        ts = it->second;
        device_ts.erase(device_ts.begin(), std::next(it));
      }
    }
    on_frame(out, remap->out_width(), remap->out_height(), ts);
  }

#ifdef HAVE_NVMM
  // VIC may change range or matrix without updating the caps; the surface format is authoritative.
  static YuvMatrix surface_matrix(NvBufSurfaceColorFormat f, const YuvMatrix& caps) {
    switch (f) {
      case NVBUF_COLOR_FORMAT_NV12: return YuvMatrix::make(false, false);
      case NVBUF_COLOR_FORMAT_NV12_ER: return YuvMatrix::make(false, true);
      case NVBUF_COLOR_FORMAT_NV12_709: return YuvMatrix::make(true, false);
      case NVBUF_COLOR_FORMAT_NV12_709_ER: return YuvMatrix::make(true, true);
      default: return caps;
    }
  }

  // Decoded surface -> EGLImage -> CUDA pitched pointers; the frame never leaves device memory.
  void stitch_nvmm(GstBuffer* buf, const YuvMatrix& m, size_t out_pitch) {
    GstMapInfo mi;
    if (!gst_buffer_map(buf, &mi, GST_MAP_READ)) throw std::runtime_error("map NVMM buffer");
    auto* surf = reinterpret_cast<NvBufSurface*>(mi.data);
    cudaGraphicsResource_t res = nullptr;
    bool egl_mapped = false;
    try {
      if (NvBufSurfaceMapEglImage(surf, 0) != 0) throw std::runtime_error("NvBufSurfaceMapEglImage failed");
      egl_mapped = true;
      if (cudaGraphicsEGLRegisterImage(&res, surf->surfaceList[0].mappedAddr.eglImage,
                                       cudaGraphicsRegisterFlagsReadOnly) != cudaSuccess) {
        res = nullptr;
        throw std::runtime_error("cudaGraphicsEGLRegisterImage failed");
      }
      cudaEglFrame f;
      if (cudaGraphicsResourceGetMappedEglFrame(&f, res, 0, 0) != cudaSuccess) {
        throw std::runtime_error("cudaGraphicsResourceGetMappedEglFrame failed");
      }
      if (f.frameType != cudaEglFrameTypePitch || f.planeCount < 2) {
        throw std::runtime_error(
            "NVMM surface is block-linear; append 'nvvidconv bl-output=false ! video/x-raw(memory:NVMM),format=NV12'");
      }
      // The EGL frame gives the UV pitch in 2-byte elements; the surface plane params are in bytes.
      const auto& planes = surf->surfaceList[0].planeParams;
      remap->run_nv12_device(static_cast<const uint8_t*>(f.frame.pPitch[0].ptr), planes.pitch[0],
                             static_cast<const uint8_t*>(f.frame.pPitch[1].ptr), planes.pitch[1],
                             surface_matrix(surf->surfaceList[0].colorFormat, m), out, out_pitch);
    } catch (...) {
      if (res) cudaGraphicsUnregisterResource(res);
      if (egl_mapped) NvBufSurfaceUnMapEglImage(surf, 0);
      gst_buffer_unmap(buf, &mi);
      throw;
    }
    cudaGraphicsUnregisterResource(res);
    NvBufSurfaceUnMapEglImage(surf, 0);
    gst_buffer_unmap(buf, &mi);
  }
#endif

  void poll_bus() {
    while (GstMessage* msg = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING))) {
      GError* err = nullptr;
      gchar* dbg = nullptr;
      if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        gst_message_parse_error(msg, &err, &dbg);
      } else {
        gst_message_parse_warning(msg, &err, &dbg);
      }
      if (on_error) on_error(std::string(GST_OBJECT_NAME(msg->src)) + ": " + (err ? err->message : "?"));
      if (err) g_error_free(err);
      g_free(dbg);
      gst_message_unref(msg);
    }
  }
};

LiveStitcher::LiveStitcher(LiveStitcherOptions options, FrameCallback on_frame, ErrorCallback on_error)
    : impl_(std::make_unique<Impl>()) {
  gst_init(nullptr, nullptr);
  Impl& d = *impl_;
  d.opt = std::move(options);
  d.on_frame = std::move(on_frame);
  d.on_error = std::move(on_error);
  const std::string codec = d.opt.h265 ? "h265" : "h264";
  const std::string decoder = d.opt.decoder.empty() ? default_decoder(d.opt.h265) : d.opt.decoder;
  // The remap runs in the appsink streaming thread and waits for a shared GPU; the leaky queue lets the
  // decoder run at camera rate and drops decoded frames instead, so latency stays at one frame rather than
  // the compressed stream piling up in appsrc.
  if (d.opt.lossless) {
    d.description = "appsrc name=src is-live=false format=time block=true max-bytes=1000000 caps=video/x-" + codec +
                    ",stream-format=byte-stream ! " + codec + "parse ! " + decoder +
                    " ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0"
                    " ! appsink name=sink sync=false max-buffers=2 drop=false";
  } else {
    d.description = "appsrc name=src is-live=true format=time block=false max-bytes=16000000 caps=video/x-" + codec +
                    ",stream-format=byte-stream ! " + codec + "parse ! " + decoder +
                    " ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0"
                    " ! appsink name=sink sync=false max-buffers=1 drop=true";
  }
  GError* err = nullptr;
  d.pipeline = gst_parse_launch(d.description.c_str(), &err);
  if (!d.pipeline || err) {
    const std::string msg = err ? err->message : "gst_parse_launch failed";
    if (err) g_error_free(err);
    throw std::runtime_error("GStreamer pipeline '" + d.description + "': " + msg);
  }
  d.src = gst_bin_get_by_name(GST_BIN(d.pipeline), "src");
  d.sink = gst_bin_get_by_name(GST_BIN(d.pipeline), "sink");
  GstAppSinkCallbacks cb{};
  cb.new_sample = &Impl::on_sample;
  gst_app_sink_set_callbacks(GST_APP_SINK(d.sink), &cb, &d, nullptr);
  d.bus = gst_element_get_bus(d.pipeline);
  if (gst_element_set_state(d.pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    throw std::runtime_error("GStreamer pipeline did not start: " + d.description);
  }
}

LiveStitcher::~LiveStitcher() {
  Impl& d = *impl_;
  if (d.src) gst_app_src_end_of_stream(GST_APP_SRC(d.src));
  if (d.pipeline) gst_element_set_state(d.pipeline, GST_STATE_NULL);
  if (d.bus) gst_object_unref(d.bus);
  if (d.src) gst_object_unref(d.src);
  if (d.sink) gst_object_unref(d.sink);
  if (d.pipeline) gst_object_unref(d.pipeline);
}

void LiveStitcher::push(const uint8_t* data, size_t size, int64_t device_ts) {
  Impl& d = *impl_;
  GstBuffer* buf = gst_buffer_new_allocate(nullptr, size, nullptr);
  gst_buffer_fill(buf, 0, data, size);
  {
    std::lock_guard<std::mutex> lock(d.ts_mutex);
    GST_BUFFER_PTS(buf) = d.next_pts;
    d.device_ts[d.next_pts] = device_ts;
    d.next_pts += GST_MSECOND;
    if (d.device_ts.size() > (d.opt.lossless ? 4096u : 256u)) d.device_ts.erase(d.device_ts.begin());
  }
  gst_app_src_push_buffer(GST_APP_SRC(d.src), buf);
  d.poll_bus();
}

const std::string& LiveStitcher::pipeline() const { return impl_->description; }

}  // namespace insta360_cuda_stitch
