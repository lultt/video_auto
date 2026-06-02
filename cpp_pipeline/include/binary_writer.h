// binary_writer.h — lightweight columnar binary format, zero-copy dump.
// No Apache Arrow dependency. No text serialization.
// CBF v2 schema (12 cols, includes roi_code): 80-byte header → column buffers.

#include "pipeline.h"

namespace writer {

#pragma pack(push, 1)
struct ColHeader {
    char     magic[4] = {'C','B','F','\x02'};
    uint32_t num_rows = 0;
    uint32_t num_cols = 12;
    uint32_t col_offsets[13] = {};   // 12 starts + 1 tail
    float    reserved[4] = {};
};
#pragma pack(pop)
static_assert(sizeof(ColHeader) == 80, "ColHeader must be 80 bytes (CBF v2)");

// Write all feature vectors as columnar binary (one column = one contiguous buffer)
inline void write_features_binary(
    const std::string& video_id,
    const std::vector<FrameFeatures>& features,
    const std::string& output_path)
{
    size_t n = features.size();
    if (n == 0) return;

    // Allocate column buffers
    std::vector<uint32_t> col_frame_idx(n);
    std::vector<uint32_t> col_roi_code(n);
    std::vector<float>    col_time(n);
    std::vector<float>    col_brightness(n);
    std::vector<float>    col_brightness_std(n);
    std::vector<float>    col_mean_b(n);
    std::vector<float>    col_mean_g(n);
    std::vector<float>    col_mean_r(n);
    std::vector<float>    col_motion(n);
    std::vector<float>    col_edge(n);
    std::vector<float>    col_lap_var(n);
    std::vector<float>    col_entropy(n);

    // Scatter: struct-of-arrays → columnar (single pass, cache-friendly)
    for (size_t i = 0; i < n; ++i) {
        const auto& f = features[i];
        col_frame_idx[i]      = f.frame_idx;
        col_roi_code[i]       = f.roi_code;
        col_time[i]           = f.timestamp_sec;
        col_brightness[i]     = f.mean_brightness;
        col_brightness_std[i] = f.brightness_std;
        col_mean_b[i]         = f.mean_b;
        col_mean_g[i]         = f.mean_g;
        col_mean_r[i]         = f.mean_r;
        col_motion[i]         = f.motion_intensity;
        col_edge[i]           = f.edge_density;
        col_lap_var[i]        = f.laplacian_variance;
        col_entropy[i]        = f.entropy;
    }

    // Compute column sizes and offsets
    ColHeader hdr;
    hdr.num_rows = static_cast<uint32_t>(n);

    uint32_t col_sizes[12];
    col_sizes[0]  = static_cast<uint32_t>(n * sizeof(uint32_t)); // frame_idx
    col_sizes[1]  = static_cast<uint32_t>(n * sizeof(uint32_t)); // roi_code
    for (int i = 2; i < 12; ++i)
        col_sizes[i] = static_cast<uint32_t>(n * sizeof(float));

    uint32_t off = sizeof(ColHeader);
    for (int i = 0; i < 12; ++i) {
        hdr.col_offsets[i] = off;
        off += col_sizes[i];
    }
    hdr.col_offsets[12] = off;  // total size

    // Write: header → 12 column buffers (one syscall each)
    std::ofstream f(output_path, std::ios::binary);
    if (!f) { std::cerr << "ERROR: cannot write " << output_path << std::endl; return; }

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(ColHeader));
    f.write(reinterpret_cast<const char*>(col_frame_idx.data()),      col_sizes[0]);
    f.write(reinterpret_cast<const char*>(col_roi_code.data()),       col_sizes[1]);
    f.write(reinterpret_cast<const char*>(col_time.data()),           col_sizes[2]);
    f.write(reinterpret_cast<const char*>(col_brightness.data()),     col_sizes[3]);
    f.write(reinterpret_cast<const char*>(col_brightness_std.data()), col_sizes[4]);
    f.write(reinterpret_cast<const char*>(col_mean_b.data()),         col_sizes[5]);
    f.write(reinterpret_cast<const char*>(col_mean_g.data()),         col_sizes[6]);
    f.write(reinterpret_cast<const char*>(col_mean_r.data()),         col_sizes[7]);
    f.write(reinterpret_cast<const char*>(col_motion.data()),         col_sizes[8]);
    f.write(reinterpret_cast<const char*>(col_edge.data()),           col_sizes[9]);
    f.write(reinterpret_cast<const char*>(col_lap_var.data()),        col_sizes[10]);
    f.write(reinterpret_cast<const char*>(col_entropy.data()),        col_sizes[11]);
    f.close();
}

} // namespace writer
