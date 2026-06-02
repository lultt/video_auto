#pragma once
// cpp_pipeline: C++ ultra-fast video feature extraction backend
//
// Zero Python overhead. Multi-threaded, lock-free, direct ffmpeg pipe decode.
// Target: beat Python's 500x realtime by eliminating pickle/Queue/pandas.

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <memory>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
    #include <psapi.h>
    #include <process.h>   // _popen, _pclose
    #define POPEN  _popen
    #define PCLOSE _pclose
    #define PIPE_MODE "rb"
#else
    #define POPEN  popen
    #define PCLOSE pclose
    #define PIPE_MODE "r"
#endif

namespace fs = std::filesystem;
namespace chr = std::chrono;

namespace pipeline {

// ============================================================================
// Feature vector per frame (tightly packed, no heap indirection)
// ============================================================================
struct alignas(64) FrameFeatures {
    uint32_t frame_idx = 0;
    uint32_t roi_code = 0;            // 0 = left, 1 = right
    float    timestamp_sec = 0.0f;
    float    mean_brightness = 0.0f;
    float    brightness_std = 0.0f;
    float    mean_b = 0.0f;
    float    mean_g = 0.0f;
    float    mean_r = 0.0f;
    float    motion_intensity = 0.0f;
    float    edge_density = 0.0f;
    float    laplacian_variance = 0.0f;
    float    entropy = 0.0f;
};

// ROI definitions — pixel rectangles inside the resized 640x360 frame.
// Mapped from 2560x1440 business zones (divide by 4):
//   left     : (500,900,1500,1440)  → x0=125, y0=225, 250x135
//   transfer : (950,550,1600,900)   → x0=237, y0=137, 163x88
//   right    : (1500,900,2100,1440) → x0=375, y0=225, 150x135
struct ROIRect {
    uint32_t    code;
    const char* name;
    int x0, y0, w, h;
};

inline const ROIRect* roi_table() {
    static const ROIRect rois[] = {
        {0, "left",     125, 225, 250, 135},
        {1, "transfer", 237, 137, 163,  88},
        {2, "right",    375, 225, 150, 135},
    };
    return rois;
}
constexpr int NUM_ROIS = 3;

// Per-video result (collected features + metadata)
struct VideoResult {
    std::string video_id;
    std::string parquet_path;
    std::vector<FrameFeatures> features;
    int    total_keyframes = 0;
    int    total_samples   = 0;
    int    active_rois     = NUM_ROIS;
    bool   full_frame_mode = false;
    double video_duration_sec = 0.0;
    double processing_time_sec = 0.0;
};

// ============================================================================
// Lock-free SPSC (Single Producer Single Consumer) Queue
// Bounded ring buffer for passing decoded frames to feature workers.
// No malloc in push/pop — fully pre-allocated.
// ============================================================================
template <typename T, size_t Capacity = 64>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    T buffer_[Capacity];
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
public:
    SPSCQueue() = default;

    bool try_push(const T& item) {
        size_t w = write_pos_.load(std::memory_order_relaxed);
        size_t r = read_pos_.load(std::memory_order_acquire);
        if (w - r >= Capacity) return false;  // full
        buffer_[w & (Capacity - 1)] = item;
        write_pos_.store(w + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& item) {
        size_t r = read_pos_.load(std::memory_order_relaxed);
        size_t w = write_pos_.load(std::memory_order_acquire);
        if (r >= w) return false;  // empty
        item = buffer_[r & (Capacity - 1)];
        read_pos_.store(r + 1, std::memory_order_release);
        return true;
    }

    size_t size_approx() const {
        size_t w = write_pos_.load(std::memory_order_acquire);
        size_t r = read_pos_.load(std::memory_order_acquire);
        return w - r;
    }
};

// ============================================================================
// Simple thread pool (work-stealing, lock-based — good enough for video count)
// ============================================================================
class ThreadPool {
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    template <typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;

    size_t size() const { return workers_.size(); }
};

// ============================================================================
// Benchmark metrics
// ============================================================================
struct BenchStats {
    int    video_count = 0;
    double total_video_sec = 0.0;
    double total_wall_sec = 0.0;
    double total_frames = 0;
    double peak_memory_mb = 0.0;
    double realtime_factor() const { return total_video_sec / (std::max)(total_wall_sec, 0.001); }
    double videos_per_sec() const { return video_count / (std::max)(total_wall_sec, 0.001); }
    double frames_per_sec() const { return total_frames / (std::max)(total_wall_sec, 0.001); }
};

} // namespace pipeline
