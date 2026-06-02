"""
视频读取器性能基准测试

测试项目：
- OpenCV 读取速度（CPU/GPU decode）
- ffmpeg pipe 读取速度
- decord 读取速度（如可用）
- NAS 吞吐量
- Adaptive sampling + stabilization 速度
- IO占比 vs CPU占比

输出：FPS、视频处理倍速、瓶颈判断
"""

import os
import sys
import time
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config
from src.video_reader import FastVideoReader


def _find_test_video(cfg):
    test_video = cfg["benchmark"].get("test_video")
    if test_video:
        return test_video
    from pathlib import Path
    videos = sorted(Path(cfg["video_root"]).glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"视频目录为空: {cfg['video_root']}")
    return str(videos[0])


def bench_opencv_cpu(video_path, resize, n_frames):
    """纯 OpenCV CPU 解码。"""
    reader = FastVideoReader(video_path, resize=resize, gpu_decode=False)
    t0 = time.perf_counter()
    count = 0
    for _ in reader.read_fixed_fps(sample_fps=reader.fps, max_frames=n_frames):
        count += 1
    elapsed = time.perf_counter() - t0
    reader.release()
    return count, elapsed


def bench_opencv_gpu(video_path, resize, n_frames):
    """OpenCV GPU (NVDEC) 解码。"""
    reader = FastVideoReader(video_path, resize=resize, gpu_decode=True)
    gpu_ok = reader.gpu_active
    t0 = time.perf_counter()
    count = 0
    for _ in reader.read_fixed_fps(sample_fps=reader.fps, max_frames=n_frames):
        count += 1
    elapsed = time.perf_counter() - t0
    reader.release()
    return count, elapsed, gpu_ok


def bench_ffmpeg_pipe(video_path, resize, n_frames):
    """ffmpeg subprocess pipe 解码。"""
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return 0, 0.0, "ffmpeg not in PATH"

    w, h = resize if resize else (2560, 1440)
    cmd = [
        ffmpeg_path, "-i", video_path,
        "-vf", f"scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-v", "quiet", "-"
    ]
    frame_size = w * h * 3
    t0 = time.perf_counter()
    count = 0
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while count < n_frames:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            _ = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            count += 1
        proc.terminate()
        proc.wait(timeout=5)
    except Exception as e:
        return 0, 0.0, str(e)
    elapsed = time.perf_counter() - t0
    return count, elapsed, None


def bench_decord(video_path, resize, n_frames):
    """decord GPU 解码（如已安装）。"""
    try:
        import decord
        decord.bridge.set_bridge("numpy")
    except ImportError:
        return 0, 0.0, "decord 未安装"

    try:
        w, h = resize if resize else (2560, 1440)
        vr = decord.VideoReader(video_path, width=w, height=h, num_threads=4)
        indices = list(range(min(n_frames, len(vr))))
        t0 = time.perf_counter()
        for i in indices:
            _ = vr[i].asnumpy()
        elapsed = time.perf_counter() - t0
        return len(indices), elapsed, None
    except Exception as e:
        return 0, 0.0, str(e)


def bench_nas_throughput(video_path, chunk_mb=64):
    """原始文件读取吞吐量（测NAS带宽）。"""
    chunk_size = chunk_mb * 1024 * 1024
    file_size = os.path.getsize(video_path)
    read_size = min(file_size, 256 * 1024 * 1024)

    t0 = time.perf_counter()
    bytes_read = 0
    with open(video_path, "rb") as f:
        while bytes_read < read_size:
            data = f.read(chunk_size)
            if not data:
                break
            bytes_read += len(data)
    elapsed = time.perf_counter() - t0
    throughput_mbps = (bytes_read / 1024 / 1024) / elapsed if elapsed > 0 else 0
    return bytes_read, elapsed, throughput_mbps


def bench_adaptive_stabilized(video_path, resize, cfg, n_frames):
    """完整 adaptive + stabilization pipeline。分离 IO/CPU。"""
    reader = FastVideoReader(video_path, resize=resize, gpu_decode=cfg.get("gpu_decode", False))
    adaptive_cfg = cfg["adaptive_sampling"]

    io_time = 0.0
    stab_time = 0.0
    total_count = 0

    # 手动展开循环以分离计时
    normal_step = max(1, round(reader.fps / adaptive_cfg["normal_fps"]))
    frame_idx = 0
    prev_gray = None
    prev_hist = None
    state = "NORMAL"
    burst_start = 0.0
    step = normal_step
    burst_step = max(1, round(reader.fps / adaptive_cfg["burst_fps"]))

    while frame_idx < reader.frame_count and total_count < n_frames:
        # IO: decode + resize
        t_io = time.perf_counter()
        reader.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = reader.cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        import cv2 as _cv2
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        io_time += time.perf_counter() - t_io

        # CPU: stabilization + histogram
        t_cpu = time.perf_counter()
        timestamp_sec = frame_idx / reader.fps

        hist = _cv2.calcHist([gray], [0], None, [256], [0, 256])
        _cv2.normalize(hist, hist)
        if prev_hist is not None:
            scene_corr = float(_cv2.compareHist(hist, prev_hist, _cv2.HISTCMP_CORREL))
        else:
            scene_corr = 1.0

        if adaptive_cfg["enabled"]:
            if state == "NORMAL" and scene_corr < adaptive_cfg["trigger_threshold"]:
                state = "BURST"
                burst_start = timestamp_sec
                step = burst_step
            elif state == "BURST" and (timestamp_sec - burst_start) > adaptive_cfg["burst_duration_sec"]:
                state = "NORMAL"
                step = normal_step

        if prev_gray is not None:
            curr_f = gray.astype(np.float64)
            prev_f = prev_gray.astype(np.float64)
            (dx, dy), _ = _cv2.phaseCorrelate(curr_f, prev_f)
            h, w = gray.shape
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            _ = _cv2.warpAffine(gray, M, (w, h))

        stab_time += time.perf_counter() - t_cpu

        prev_gray = gray
        prev_hist = hist
        total_count += 1
        frame_idx += step

    reader.release()
    return total_count, io_time, stab_time


def run_benchmark(cfg):
    video_path = _find_test_video(cfg)
    n_frames = cfg["benchmark"]["test_frames"]
    resize = (cfg["resize_width"], cfg["resize_height"])

    print("=" * 70)
    print("  VIDEO READER BENCHMARK")
    print("=" * 70)
    print(f"  视频: {os.path.basename(video_path)}")
    print(f"  测试帧数: {n_frames}")
    print(f"  Resize: {resize[0]}x{resize[1]}")
    print()

    # NAS throughput
    print("[1] NAS 文件读取吞吐")
    bytes_read, elapsed, mbps = bench_nas_throughput(video_path)
    print(f"    读取: {bytes_read / 1024 / 1024:.0f} MB / {elapsed:.2f}s = {mbps:.0f} MB/s")
    print()

    # OpenCV CPU
    print("[2] OpenCV CPU 解码")
    count, elapsed = bench_opencv_cpu(video_path, resize, n_frames)
    cpu_fps = count / elapsed if elapsed > 0 else 0
    video_speed = cpu_fps / (cfg["adaptive_sampling"]["normal_fps"])
    print(f"    {count} 帧 / {elapsed:.2f}s = {cpu_fps:.1f} raw fps")
    print(f"    视频处理倍速 (1fps采样): {video_speed:.0f}x 实时")
    print()

    # OpenCV GPU
    print("[3] OpenCV GPU 解码 (NVDEC)")
    count, elapsed, gpu_ok = bench_opencv_gpu(video_path, resize, n_frames)
    gpu_fps = count / elapsed if elapsed > 0 else 0
    if gpu_ok:
        speedup = gpu_fps / cpu_fps if cpu_fps > 0 else 0
        print(f"    {count} 帧 / {elapsed:.2f}s = {gpu_fps:.1f} raw fps")
        print(f"    GPU加速比: {speedup:.2f}x vs CPU")
    else:
        print(f"    GPU解码不可用 (回退CPU): {gpu_fps:.1f} fps")
    print()

    # ffmpeg pipe
    print("[4] ffmpeg pipe 解码")
    count, elapsed, err = bench_ffmpeg_pipe(video_path, resize, n_frames)
    if err:
        print(f"    跳过: {err}")
    else:
        pipe_fps = count / elapsed if elapsed > 0 else 0
        print(f"    {count} 帧 / {elapsed:.2f}s = {pipe_fps:.1f} fps")
    print()

    # decord
    print("[5] decord 解码")
    count, elapsed, err = bench_decord(video_path, resize, n_frames)
    if err:
        print(f"    跳过: {err}")
    else:
        decord_fps = count / elapsed if elapsed > 0 else 0
        print(f"    {count} 帧 / {elapsed:.2f}s = {decord_fps:.1f} fps")
    print()

    # Adaptive + Stabilization (IO vs CPU breakdown)
    print("[6] Adaptive Sampling + Stabilization (IO/CPU 分离)")
    test_n = min(n_frames, 500)
    count, io_t, cpu_t = bench_adaptive_stabilized(video_path, resize, cfg, test_n)
    total_t = io_t + cpu_t
    if total_t > 0 and count > 0:
        effective_fps = count / total_t
        print(f"    采样帧: {count}")
        print(f"    IO (decode+resize+cvtColor): {io_t:.2f}s ({io_t/total_t*100:.0f}%)")
        print(f"    CPU (hist+phaseCorr+warpAffine): {cpu_t:.2f}s ({cpu_t/total_t*100:.0f}%)")
        print(f"    有效速度: {effective_fps:.1f} 采样帧/秒")
        print(f"    瓶颈: {'IO' if io_t > cpu_t else 'CPU计算'}")
    print()

    # 全量预估
    print("[7] 全量处理预估")
    from pathlib import Path
    all_videos = list(Path(cfg["video_root"]).glob("*.mp4"))
    n_videos = len(all_videos)
    reader = FastVideoReader(video_path, resize=resize)
    avg_sampled = reader.frame_count / max(1, round(reader.fps / cfg["adaptive_sampling"]["normal_fps"]))
    reader.release()

    if total_t > 0 and count > 0:
        sec_per_video = avg_sampled * (total_t / count)
        total_hours = (sec_per_video * n_videos) / 3600
        parallel_hours = total_hours / cfg["num_workers"]
        print(f"    视频数: {n_videos}")
        print(f"    每视频采样帧: ~{avg_sampled:.0f}")
        print(f"    单视频耗时: ~{sec_per_video:.0f}s ({sec_per_video/60:.1f}min)")
        print(f"    串行总时间: {total_hours:.1f} 小时")
        print(f"    {cfg['num_workers']}进程并行: {parallel_hours:.1f} 小时")

    print()
    print("=" * 70)


if __name__ == "__main__":
    cfg = load_config()
    run_benchmark(cfg)
