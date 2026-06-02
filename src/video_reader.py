"""
高速视频读取器 — 系统核心模块

解码后端: ffmpeg subprocess pipe（顺序读取，NAS友好）
采样方式: 顺序解码 + 跳帧丢弃（不做 seek）
Stabilization: phaseCorrelate 内置，输出 stabilized frame

输出 FramePacket，下游只需 ROI 裁剪 + 特征计算。
"""

import subprocess
import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path


FFMPEG_PATH = r"D:\ffmpeg\bin\ffmpeg.exe"


@dataclass
class FramePacket:
    frame_idx: int
    timestamp_sec: float
    frame_bgr: np.ndarray
    gray: np.ndarray
    stabilized_gray: np.ndarray
    global_motion_x: float
    global_motion_y: float
    global_motion_mag: float
    sample_fps: float
    scene_transition: float


class FastVideoReader:
    """ffmpeg pipe 顺序解码 + adaptive sampling + stabilization。"""

    def __init__(self, video_path, resize=None, gpu_decode=False):
        self.video_path = str(video_path)
        self.resize = resize
        self._gpu_decode = gpu_decode

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if self.resize:
            self.out_w, self.out_h = self.resize
        else:
            self.out_w, self.out_h = self.width, self.height

    @property
    def duration_sec(self):
        return self.frame_count / self.fps if self.fps > 0 else 0

    def _build_ffmpeg_cmd(self):
        cmd = [FFMPEG_PATH]
        if self._gpu_decode:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += [
            "-i", self.video_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
        ]
        if self.resize:
            cmd += ["-vf", f"scale={self.out_w}:{self.out_h}"]
        cmd += ["-v", "quiet", "-"]
        return cmd

    def _open_pipe(self):
        cmd = self._build_ffmpeg_cmd()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
        )
        return proc

    def _read_frame(self, proc):
        frame_size = self.out_w * self.out_h * 3
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((self.out_h, self.out_w, 3))

    def read_fixed_fps(self, sample_fps=1.0, max_frames=None):
        """固定采样率，顺序解码+跳帧。用于 benchmark。"""
        step = max(1, round(self.fps / sample_fps))
        proc = self._open_pipe()
        frame_idx = 0
        count = 0

        try:
            while frame_idx < self.frame_count:
                frame_bgr = self._read_frame(proc)
                if frame_bgr is None:
                    break
                if frame_idx % step == 0:
                    yield frame_idx, frame_idx / self.fps, frame_bgr
                    count += 1
                    if max_frames and count >= max_frames:
                        break
                frame_idx += 1
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def read_adaptive(self, adaptive_cfg, stabilize=True, max_frames=None):
        """
        自适应采样 + stabilization。

        顺序解码所有帧，按 step 决定哪些帧输出。
        scene_transition 突变时缩小 step。
        """
        normal_fps = adaptive_cfg["normal_fps"]
        burst_fps = adaptive_cfg["burst_fps"]
        threshold = adaptive_cfg["trigger_threshold"]
        burst_duration = adaptive_cfg["burst_duration_sec"]
        enabled = adaptive_cfg.get("enabled", True)

        normal_step = max(1, round(self.fps / normal_fps))
        burst_step = max(1, round(self.fps / burst_fps))

        state = "NORMAL"
        burst_start = 0.0
        step = normal_step
        next_sample_idx = 0

        prev_gray = None
        prev_hist = None
        count = 0

        proc = self._open_pipe()
        frame_idx = 0

        try:
            while frame_idx < self.frame_count:
                frame_bgr = self._read_frame(proc)
                if frame_bgr is None:
                    break

                if frame_idx < next_sample_idx:
                    frame_idx += 1
                    continue

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                timestamp_sec = frame_idx / self.fps
                current_fps = burst_fps if state == "BURST" else normal_fps

                # scene_transition
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist)
                if prev_hist is not None:
                    scene_corr = float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL))
                else:
                    scene_corr = 1.0

                # adaptive state machine
                if enabled:
                    if state == "NORMAL" and scene_corr < threshold:
                        state = "BURST"
                        burst_start = timestamp_sec
                        step = burst_step
                    elif state == "BURST" and (timestamp_sec - burst_start) > burst_duration:
                        state = "NORMAL"
                        step = normal_step

                # stabilization
                if stabilize and prev_gray is not None:
                    curr_f = gray.astype(np.float64)
                    prev_f = prev_gray.astype(np.float64)
                    (dx, dy), _ = cv2.phaseCorrelate(curr_f, prev_f)
                    mag = float(np.sqrt(dx * dx + dy * dy))
                    h, w = gray.shape
                    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    stabilized_gray = cv2.warpAffine(gray, M, (w, h))
                else:
                    dx, dy, mag = 0.0, 0.0, 0.0
                    stabilized_gray = gray.copy()

                yield FramePacket(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    frame_bgr=frame_bgr,
                    gray=gray,
                    stabilized_gray=stabilized_gray,
                    global_motion_x=float(dx),
                    global_motion_y=float(dy),
                    global_motion_mag=mag,
                    sample_fps=current_fps,
                    scene_transition=scene_corr,
                )

                prev_gray = gray
                prev_hist = hist
                next_sample_idx = frame_idx + step
                count += 1
                if max_frames and count >= max_frames:
                    break

                frame_idx += 1
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
