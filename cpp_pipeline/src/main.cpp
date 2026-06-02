// cpp_pipeline: C++ ultra-fast video feature extraction backend
// main.cpp — CLI entry point + thread pool + benchmark
//
// Usage:
//   cpp_pipeline.exe <video_dir> <output_dir> [--threads N] [--benchmark]
//
// Example:
//   cpp_pipeline.exe C:/video J:/video_auto/outputs/cpp_bench --threads 16

#include "pipeline.h"
#include "video_processor.h"

using namespace pipeline;

// Thread pool implementation
ThreadPool::ThreadPool(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop_ = true;
    cv_.notify_all();
    for (auto& w : workers_) if (w.joinable()) w.join();
}

template <typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stop_) throw std::runtime_error("ThreadPool stopped");
        tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return res;
}

// Memory usage probe (Windows)
#ifdef _WIN32
static size_t get_peak_memory_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PeakWorkingSetSize / (1024 * 1024);
    return 0;
}
#else
static size_t get_peak_memory_mb() { return 0; }
#endif

// ============================================================================
int main(int argc, char* argv[]) {
    std::string video_dir;
    std::string output_dir;
    std::string per_video_log;
    int  num_threads = 4;
    bool emit_csv    = false;
    bool no_output   = false;
    bool use_gpu     = false;
    bool full_frame  = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--emit-csv") {
            emit_csv = true;
        } else if (arg == "--no-output") {
            no_output = true;
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--full-frame") {
            full_frame = true;
        } else if (arg == "--per-video-log" && i + 1 < argc) {
            per_video_log = argv[++i];
        } else if (video_dir.empty()) {
            video_dir = arg;
        } else if (output_dir.empty()) {
            output_dir = arg;
        }
    }

    if (video_dir.empty() || (output_dir.empty() && !no_output)) {
        std::cout << "cpp_pipeline - C++ ultra-fast video feature extraction\n\n"
                  << "Usage: cpp_pipeline.exe <video_dir> <output_dir> [--threads N] [--emit-csv] [--no-output] [--gpu] [--full-frame] [--per-video-log PATH]\n\n"
                  << "  video_dir         Path to .mp4 files\n"
                  << "  output_dir        Output directory for .cbf files (required unless --no-output)\n"
                  << "  --threads N       Number of parallel video threads (default 4)\n"
                  << "  --emit-csv        Also write a .csv sidecar per video\n"
                  << "  --no-output       Pure benchmark mode: skip all file I/O (CBF + CSV)\n"
                  << "  --gpu             Use NVDEC / hevc_cuvid decode path\n"
                  << "  --full-frame      Disable keyframe-only decode and process every frame\n"
                  << "  --per-video-log P Append one JSON line per video to this path\n";
        return 1;
    }
    if (no_output && output_dir.empty()) {
        output_dir = ".";
    }

    // Collect videos
    std::vector<std::string> video_paths;
    for (const auto& entry : fs::directory_iterator(video_dir)) {
        if (entry.path().extension() == ".mp4") {
            video_paths.push_back(entry.path().string());
        }
    }
    std::sort(video_paths.begin(), video_paths.end());

    if (video_paths.empty()) {
        std::cerr << "No .mp4 files found in " << video_dir << std::endl;
        return 1;
    }

    std::cout << "\n  ============================================\n"
              <<   "    cpp_pipeline - C++ Feature Extraction\n"
              <<   "  ============================================\n\n"
              << "  Videos:    " << video_paths.size() << "\n"
              << "  Threads:   " << num_threads << "\n"
              << "  Resize:    " << pipeline::OUT_W << "x" << pipeline::OUT_H << "\n"
              << "  ROIs:      LEFT + RIGHT (2 ROIs per keyframe)\n"
              << "  Decode:    " << (full_frame ? "full-frame, every frame" : (std::string("keyframe-only, 1/") + std::to_string(pipeline::KF_SUBSAMPLE) + " subsample")) << "\n"
              << "  Output:    " << (no_output ? "DISABLED (benchmark mode)" : output_dir) << "\n"
              << "  Emit CSV:  " << (emit_csv ? "yes" : "no") << "\n"
              << "  Decoder:   " << (use_gpu ? "GPU (NVDEC / hevc_cuvid)" : "CPU (libavcodec)") << "\n\n";

    // Optional per-video JSONL log (one shared file, mutex-protected)
    std::mutex log_mtx;
    std::ofstream log_file;
    if (!per_video_log.empty()) {
        fs::create_directories(fs::path(per_video_log).parent_path());
        log_file.open(per_video_log, std::ios::out | std::ios::trunc);
    }

    ThreadPool pool(num_threads);
    std::vector<std::future<pipeline::VideoResult>> futures;

    BenchStats stats;
    stats.video_count = static_cast<int>(video_paths.size());

    auto t_global_0 = chr::high_resolution_clock::now();

    // Enqueue one video per thread
    for (const auto& vp : video_paths) {
        futures.push_back(pool.enqueue([vp, &output_dir, emit_csv, no_output, use_gpu, full_frame]() {
            return pipeline::process_one_video(vp, output_dir, emit_csv, no_output, use_gpu, full_frame);
        }));
    }

    // Collect results
    double total_video_sec = 0.0;
    double total_frames    = 0.0;
    int    ok_count        = 0;

    for (auto& fut : futures) {
        auto result = fut.get();
        total_video_sec += result.video_duration_sec;
        total_frames    += result.total_samples;
        if (result.total_samples > 0) ++ok_count;
        std::cout << "  [" << result.video_id.substr(0, 25) << "...]  "
                  << result.total_samples << " samples  "
                  << result.processing_time_sec << "s\n";

        if (log_file.is_open()) {
            std::lock_guard<std::mutex> lock(log_mtx);
            log_file << "{\"video_id\":\"" << result.video_id
                     << "\",\"video_duration_sec\":" << result.video_duration_sec
                     << ",\"processing_time_sec\":" << result.processing_time_sec
                     << ",\"total_samples\":" << result.total_samples
                     << ",\"n_rois\":" << result.active_rois
                     << ",\"full_frame_mode\":" << (result.full_frame_mode ? "true" : "false")
                     << "}\n";
        }
    }
    if (log_file.is_open()) log_file.close();

    auto t_global_1 = chr::high_resolution_clock::now();
    stats.total_wall_sec   = chr::duration<double>(t_global_1 - t_global_0).count();
    stats.total_video_sec  = total_video_sec;
    stats.total_frames     = total_frames;
    stats.peak_memory_mb   = static_cast<double>(get_peak_memory_mb());

    std::cout << "\n  ============================================\n"
              <<   "                  BENCHMARK                   \n"
              <<   "  ============================================\n\n"
              << "  Videos:          " << stats.video_count << "\n"
              << "  Video duration:  " << stats.total_video_sec / 3600.0 << " hrs\n"
              << "  Wall time:       " << stats.total_wall_sec << " s\n"
              << "  Realtime factor: " << stats.realtime_factor() << "x\n"
              << "  Videos/sec:      " << stats.videos_per_sec() << "\n"
              << "  Frames/sec:      " << stats.frames_per_sec() << "\n"
              << "  Peak memory:     " << stats.peak_memory_mb << " MB\n"
              << "  OK / Total:      " << ok_count << " / " << stats.video_count << "\n"
              << std::endl;

    return 0;
}