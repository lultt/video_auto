#pragma once
// Pure C++ feature extraction — no OpenCV dependency.
// All pixel loops auto-vectorize with -O3 -mavx2.

#include "pipeline.h"

using namespace pipeline;

namespace features {

// ---------------------------------------------------------------------------
// Pixel statistics (brightness, color channels)
// ---------------------------------------------------------------------------
inline void compute_brightness(const uint8_t* gray, int n, float& mean, float& stddev) {
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        float v = static_cast<float>(gray[i]);
        sum  += v;
        sum2 += v * v;
    }
    mean = static_cast<float>(sum / n);
    float var = static_cast<float>(sum2 / n) - mean * mean;
    stddev = (var > 0.0f) ? std::sqrt(var) : 0.0f;
}

inline void compute_color_means(const uint8_t* bgr, int n, float& mb, float& mg, float& mr) {
    double sb = 0.0, sg = 0.0, sr = 0.0;
    for (int i = 0; i < n; ++i) {
        sb += bgr[i * 3 + 0];
        sg += bgr[i * 3 + 1];
        sr += bgr[i * 3 + 2];
    }
    mb = static_cast<float>(sb / n);
    mg = static_cast<float>(sg / n);
    mr = static_cast<float>(sr / n);
}

// ---------------------------------------------------------------------------
// Motion: pixelwise absdiff between two frames that share the same layout
// ---------------------------------------------------------------------------
inline float compute_motion(const uint8_t* gray_curr, const uint8_t* gray_prev, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::abs(static_cast<int>(gray_curr[i]) - static_cast<int>(gray_prev[i]));
    }
    return static_cast<float>(sum / n);
}

// ---------------------------------------------------------------------------
// Simplified edge density: mean absolute gradient magnitude (3x3 Sobel)
// ---------------------------------------------------------------------------
inline float edge_density_sobel(const uint8_t* gray, int w, int h) {
    double sum = 0.0;
    int n = 0;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int idx = y * w + x;
            int gx = -gray[idx - w - 1] + gray[idx - w + 1]
                     - 2 * gray[idx - 1] + 2 * gray[idx + 1]
                     - gray[idx + w - 1] + gray[idx + w + 1];
            int gy = -gray[idx - w - 1] - 2 * gray[idx - w]
                     - gray[idx - w + 1] + gray[idx + w - 1]
                     + 2 * gray[idx + w] + gray[idx + w + 1];
            sum += std::sqrt(static_cast<float>(gx * gx + gy * gy));
            ++n;
        }
    }
    return (n > 0) ? static_cast<float>(sum / (n * 255.0f)) : 0.0f;
}

// ---------------------------------------------------------------------------
// Simplified Laplacian variance (texture sharpness proxy)
// 3×3 Laplacian kernel, then variance of result
// ---------------------------------------------------------------------------
inline float laplacian_variance(const uint8_t* gray, int w, int h) {
    double sum = 0.0, sum2 = 0.0;
    int n = 0;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            int idx = y * w + x;
            float lap = -4.0f * gray[idx]
                        + gray[idx - w] + gray[idx + w]
                        + gray[idx - 1] + gray[idx + 1];
            sum  += lap;
            sum2 += lap * lap;
            ++n;
        }
    }
    if (n < 2) return 0.0f;
    float mean = static_cast<float>(sum / n);
    float var  = static_cast<float>(sum2 / n) - mean * mean;
    return (var > 0.0f) ? std::sqrt(var) : 0.0f;
}

// ---------------------------------------------------------------------------
// Shannon entropy of grayscale histogram
// ---------------------------------------------------------------------------
inline float shannon_entropy(const uint8_t* gray, int n) {
    int hist[256] = {0};
    for (int i = 0; i < n; ++i) hist[gray[i]]++;
    float entropy = 0.0f;
    float inv_n = 1.0f / n;
    for (int i = 0; i < 256; ++i) {
        if (hist[i] > 0) {
            float p = hist[i] * inv_n;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// ---------------------------------------------------------------------------
// Full feature extraction for one keyframe
// ---------------------------------------------------------------------------
inline FrameFeatures extract_one_frame(
    const uint8_t* gray, const uint8_t* bgr,
    const uint8_t* gray_prev, int w, int h,
    uint32_t frame_idx, float timestamp_sec)
{
    FrameFeatures f;
    int n = w * h;

    f.frame_idx = frame_idx;
    f.timestamp_sec = timestamp_sec;

    compute_brightness(gray, n, f.mean_brightness, f.brightness_std);
    compute_color_means(bgr, n, f.mean_b, f.mean_g, f.mean_r);

    if (gray_prev) {
        f.motion_intensity = compute_motion(gray, gray_prev, n);
    }

    f.edge_density = edge_density_sobel(gray, w, h);
    f.laplacian_variance = laplacian_variance(gray, w, h);
    f.entropy = shannon_entropy(gray, n);

    return f;
}

} // namespace features