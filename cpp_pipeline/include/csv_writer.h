// CSV writer — lightweight, no Apache Arrow dependency for Phase 1 MVP.
// Output schema matches Python pipeline: one row per ROI per keyframe.
// Phase 3 will upgrade to Arrow Parquet.

#include "pipeline.h"
#include <iomanip>

using namespace pipeline;

namespace writer {

inline std::string escape_csv(const std::string& s) {
    if (s.find(',') == std::string::npos && s.find('"') == std::string::npos)
        return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

inline void write_features_csv(const std::string& video_id,
                                const std::vector<FrameFeatures>& features,
                                const std::string& output_path)
{
    std::ofstream f(output_path);
    if (!f) {
        std::cerr << "ERROR: cannot write " << output_path << std::endl;
        return;
    }
    // Header
    f << "video_id,frame_idx,roi_code,roi_name,timestamp_sec,"
      << "mean_brightness,brightness_std,"
      << "mean_b,mean_g,mean_r,"
      << "motion_intensity,edge_density,laplacian_variance,entropy\n";

    const ROIRect* rois = roi_table();
    f << std::fixed << std::setprecision(6);
    for (const auto& feat : features) {
        f << escape_csv(video_id) << ","
          << feat.frame_idx << ","
          << feat.roi_code << ","
          << rois[feat.roi_code].name << ","
          << feat.timestamp_sec << ","
          << feat.mean_brightness << ","
          << feat.brightness_std << ","
          << feat.mean_b << ","
          << feat.mean_g << ","
          << feat.mean_r << ","
          << feat.motion_intensity << ","
          << feat.edge_density << ","
          << feat.laplacian_variance << ","
          << feat.entropy << "\n";
    }
    f.close();
    std::cout << "  wrote " << features.size() << " rows  →  " << output_path << std::endl;
}

} // namespace writer