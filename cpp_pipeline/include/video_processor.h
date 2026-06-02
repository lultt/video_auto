// Per-video processing: ffmpeg pipe → keyframe decode → feature extraction → CSV

#include "pipeline.h"
#include "feature_extractor.h"
#include "binary_writer.h"
#include "csv_writer.h"

using namespace pipeline;

namespace pipeline {

constexpr int KF_SUBSAMPLE  = 3;       // every Nth keyframe
constexpr int OUT_W         = 640;     // resize width
constexpr int OUT_H         = 360;     // resize height
constexpr int FRAME_BYTES   = OUT_W * OUT_H * 3;  // BGR raw

// ---------------------------------------------------------------------------
// Build ffmpeg command (keyframe-only decode, resized)
// Warning: _popen on Windows needs cmd.exe-friendly quoting.
// Use short-path version: D:/ffmpeg/bin/ffmpeg.exe with forward slashes.
// ---------------------------------------------------------------------------
inline std::string build_ffmpeg_cmd(const std::string& video_path,
                                     bool use_gpu = false,
                                     bool full_frame = false) {
    std::ostringstream ss;
    // Use forward slashes — cmd.exe handles them
    std::string vpath = video_path;
    for (auto& c : vpath) if (c == '\\') c = '/';
    if (use_gpu) {
        // NVDEC path: hevc_cuvid decoder, GPU-side scale, then download to host as BGR24.
        // Note: hevc_cuvid silently ignores -skip_frame nokey (no AVOption exposed),
        // so we use the demuxer-level -discard nokey, which drops non-key *packets*
        // before they ever reach the decoder. This yields ~692 keyframes vs the
        // CPU path's 666 (decoder-level -skip_frame nokey is stricter), a ~4%
        // packet/frame flag mismatch we accept for the benchmark.
        ss << "D:/ffmpeg/bin/ffmpeg.exe";
        if (!full_frame) ss << " -discard nokey";
        ss << " -hwaccel cuda -hwaccel_output_format cuda"
           << " -c:v hevc_cuvid"
           << " -i \"" << vpath << "\""
           << " -vf scale_cuda=" << OUT_W << ":" << OUT_H
           << ",hwdownload,format=nv12,format=bgr24"
           << " -vsync 0"
           << " -f rawvideo -pix_fmt bgr24"
           << " -v quiet"
           << " -";
    } else {
        ss << "D:/ffmpeg/bin/ffmpeg.exe";
        if (!full_frame) ss << " -skip_frame nokey";
        ss << " -i \"" << vpath << "\""
           << " -vf scale=" << OUT_W << ":" << OUT_H
           << " -vsync 0"
           << " -f rawvideo -pix_fmt bgr24"
           << " -v quiet"
           << " -";
    }
    return ss.str();
}

// ---------------------------------------------------------------------------
// Extract a rectangular ROI from the full 640×360 BGR buffer into contiguous
// output buffers (BGR + grayscale).  The copy is cheap (~46k px / ROI) and
// keeps every feature kernel working on contiguous memory for AVX2.
// ---------------------------------------------------------------------------
inline void extract_roi(const uint8_t* full_bgr, const ROIRect& roi,
                         uint8_t* out_bgr, uint8_t* out_gray) {
    int roi_n = roi.w * roi.h;
    for (int y = 0; y < roi.h; ++y) {
        const uint8_t* src_row = full_bgr + ((roi.y0 + y) * OUT_W + roi.x0) * 3;
        uint8_t*       dst_row = out_bgr + y * roi.w * 3;
        memcpy(dst_row, src_row, roi.w * 3);
        // BGR → gray inline (same luminance as the old whole-frame loop)
        for (int x = 0; x < roi.w; ++x) {
            uint8_t b = dst_row[x * 3 + 0];
            uint8_t g = dst_row[x * 3 + 1];
            uint8_t r = dst_row[x * 3 + 2];
            out_gray[y * roi.w + x] = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 8);
        }
    }
    (void)roi_n; // silence unused warning
}

// Read exact number of bytes from FILE* pipe (handles short reads)
inline bool read_exact(void* buf, size_t count, FILE* pipe) {
    size_t off = 0;
    auto* dst = static_cast<uint8_t*>(buf);
    while (off < count) {
        size_t n = fread(dst + off, 1, count - off, pipe);
        if (n == 0) {
            if (feof(pipe) || ferror(pipe)) return false;
            // spurious wakeup on pipe — retry
            continue;
        }
        off += n;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Process one video: decode → per-ROI feature extraction → CBF (+ optional CSV)
// ---------------------------------------------------------------------------
inline VideoResult process_one_video(const std::string& video_path,
                                      const std::string& output_dir,
                                      bool emit_csv = false,
                                      bool no_output = false,
                                      bool use_gpu = false,
                                      bool full_frame = false)
{
    VideoResult result;
    result.video_id = fs::path(video_path).stem().string();

    auto t0 = chr::high_resolution_clock::now();

    std::string cmd = build_ffmpeg_cmd(video_path, use_gpu, full_frame);
    FILE* pipe = POPEN(cmd.c_str(), PIPE_MODE);
    if (!pipe) {
        std::cerr << "ERROR: ffmpeg pipe open failed: " << video_path << std::endl;
        return result;
    }

    // Frame buffer (full 640×360 BGR) — reused for every keyframe
    std::vector<uint8_t> frame_buf(FRAME_BYTES);

    // Per-ROI scratch buffers (BGR + gray current + gray previous)
    const ROIRect* rois = roi_table();
    const int active_rois = NUM_ROIS;
    std::vector<std::vector<uint8_t>> roi_bgr(active_rois);
    std::vector<std::vector<uint8_t>> roi_gray_curr(active_rois);
    std::vector<std::vector<uint8_t>> roi_gray_prev(active_rois);
    std::vector<bool> roi_has_prev(active_rois, false);
    for (int r = 0; r < active_rois; ++r) {
        int n = rois[r].w * rois[r].h;
        roi_bgr[r].resize(n * 3);
        roi_gray_curr[r].resize(n);
        roi_gray_prev[r].resize(n);
    }

    int frame_idx   = 0;
    int sample_idx  = 0;
    const float frame_interval_sec = full_frame ? (1.0f / 25.0f) : 4.0f;

    while (read_exact(frame_buf.data(), FRAME_BYTES, pipe)) {
        // Default mode keeps the production 1/N keyframe subsample; full-frame bypasses it.
        if (!full_frame && frame_idx % KF_SUBSAMPLE != 0) {
            ++frame_idx;
            continue;
        }

        float t_sec = frame_idx * frame_interval_sec;

        for (int r = 0; r < active_rois; ++r) {
            extract_roi(frame_buf.data(), rois[r],
                        roi_bgr[r].data(), roi_gray_curr[r].data());

            auto feat = features::extract_one_frame(
                roi_gray_curr[r].data(), roi_bgr[r].data(),
                roi_has_prev[r] ? roi_gray_prev[r].data() : nullptr,
                rois[r].w, rois[r].h,
                static_cast<uint32_t>(full_frame ? frame_idx : sample_idx), t_sec
            );
            feat.roi_code = rois[r].code;

            if (!no_output) {
                result.features.push_back(feat);
            }
            ++result.total_samples;

            std::swap(roi_gray_prev[r], roi_gray_curr[r]);
            roi_has_prev[r] = true;
        }

        ++frame_idx;
        ++sample_idx;
    }

    PCLOSE(pipe);

    auto t1 = chr::high_resolution_clock::now();
    result.processing_time_sec = chr::duration<double>(t1 - t0).count();
    result.total_keyframes = frame_idx;
    result.video_duration_sec = frame_idx * frame_interval_sec;
    result.active_rois = active_rois;
    result.full_frame_mode = full_frame;

    if (!no_output) {
        // Write columnar binary (.cbf) — primary output
        fs::create_directories(fs::path(output_dir));
        fs::path out_cbf = fs::path(output_dir) / (result.video_id + ".cbf");
        writer::write_features_binary(result.video_id, result.features, out_cbf.string());
        result.parquet_path = out_cbf.string();

        // Optional: human-readable CSV sidecar (for diff with Python parquet)
        if (emit_csv) {
            fs::path out_csv = fs::path(output_dir) / (result.video_id + ".csv");
            writer::write_features_csv(result.video_id, result.features, out_csv.string());
        }
    }

    return result;
}

} // namespace pipeline