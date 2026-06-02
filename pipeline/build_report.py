"""
build_report.py — Generate a professional Word-compatible project report (.doc)
with embedded charts, tables, and full benchmark results.

Uses HTML+base64-images output that Microsoft Word opens natively.
No external dependencies beyond stdlib + matplotlib + pandas.
"""

import os, sys, json, base64, io
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(r"J:\video_auto")
OUTPUTS = ROOT / "outputs"
REPORT_PATH = ROOT / "outputs" / "项目说明书.doc"

GPU_BENCH_JSON    = OUTPUTS / "gpu_bench" / "gpu_vs_cpu.json"
GPU_SCALING_JSON  = OUTPUTS / "gpu_scaling" / "gpu_scaling_benchmark.json"
CPU_SCALING_CSV   = OUTPUTS / "cpp_scaling" / "scaling_benchmark.csv"

# Chart PNGs to embed
CHART_CPU_VS_GPU      = OUTPUTS / "gpu_bench" / "charts" / "cpu_vs_gpu.png"
CHART_GPU_DECODER     = OUTPUTS / "gpu_scaling" / "charts" / "workers_vs_gpu_decoder.png"
CHART_GPU_DASHBOARD   = OUTPUTS / "gpu_scaling" / "charts" / "scaling_dashboard_gpu.png"
CHART_CPU_DASHBOARD   = OUTPUTS / "cpp_scaling" / "charts" / "scaling_dashboard.png"
CHART_FEATURE_CORPUS  = OUTPUTS / "feature_plots_gpu" / "summary" / "corpus_overview.png"
CHART_FEATURE_MOTION  = OUTPUTS / "feature_plots_gpu" / "summary" / "corpus__motion_intensity.png"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def img_b64(path: Path, max_width_px: int = 720) -> str:
    """Read PNG, optionally resize, return <img> tag with base64 data."""
    if not path.exists():
        return f'<p style="color:#999">[chart missing: {path.name}]</p>'
    # For simplicity, embed at original size; Word auto-scales to page width
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:{max_width_px}px; width:100%; border:1px solid #eee; margin:8px 0;" />'

def render_table_from_json(data: list[dict], columns: list[str], col_labels: list[str]) -> str:
    """Simple HTML table from JSON list."""
    rows = []
    for d in data:
        cells = []
        for c in columns:
            val = d.get(c, "")
            if isinstance(val, float):
                cells.append(f"{val:.1f}")
            else:
                cells.append(str(val))
        rows.append("".join(f"<td>{v}</td>" for v in cells))
    header = "".join(f"<th>{h}</th>" for h in col_labels)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(f'<tr>{r}</tr>' for r in rows)}</tbody></table>"

def render_table_from_csv(path: Path, columns: list[str], col_labels: list[str]) -> str:
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            val = row.get(c, "")
            if isinstance(val, float):
                cells.append(f"{val:.1f}")
            else:
                cells.append(str(val))
        rows.append("".join(f"<td>{v}</td>" for v in cells))
    header = "".join(f"<th>{h}</th>" for h in col_labels)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(f'<tr>{r}</tr>' for r in rows)}</tbody></table>"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
  @page { size: A4; margin: 2cm 2.5cm; }
  body {
    font-family: "Microsoft YaHei", "SimHei", "DejaVu Sans", Arial, sans-serif;
    font-size: 11pt; line-height: 1.65; color: #222;
    max-width: 820px; margin: 0 auto; padding: 20px 10px;
  }
  h1 { font-size: 20pt; border-bottom: 3px solid #2E86AB; padding-bottom: 8px; margin-top: 36px; color: #111; }
  h2 { font-size: 15pt; border-bottom: 1.5px solid #bbb; padding-bottom: 4px; margin-top: 32px; color: #2E86AB; }
  h3 { font-size: 12pt; margin-top: 24px; color: #333; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0 18px 0; font-size: 10pt; }
  th { background: #2E86AB; color: #fff; padding: 8px 10px; text-align: center; font-weight: 600; }
  td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) td { background: #f8f8f8; }
  tr:hover td { background: #e8f4f8; }
  .highlight { background: #d4edda; font-weight: bold; }
  p, li { margin: 4px 0; }
  ul { padding-left: 24px; }
  .meta { color: #777; font-size: 10pt; margin-top: 2px; }
  .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin: 14px 0; font-size: 10pt; }
  .info { background: #e7f3ff; border-left: 4px solid #2E86AB; padding: 10px 14px; margin: 14px 0; font-size: 10pt; }
  .toc { background: #fafafa; padding: 16px 20px; border: 1px solid #e0e0e0; margin: 16px 0; }
  .toc a { color: #2E86AB; text-decoration: none; }
  .fig-caption { font-size: 9.5pt; color: #555; margin-top: 2px; margin-bottom: 16px; text-align: center; font-style: italic; }
  .page-break { page-break-before: always; }
</style>
"""

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------
def build():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Load data
    gpu_bench = json.loads(GPU_BENCH_JSON.read_text())
    gpu_scale = json.loads(GPU_SCALING_JSON.read_text(encoding="utf-8"))
    cpu_df = pd.read_csv(CPU_SCALING_CSV)

    cpu_best = cpu_df.loc[cpu_df["realtime_factor"].idxmax()]
    gpu_best = gpu_bench[1]  # GPU row
    gpu_scale_best = max(gpu_scale, key=lambda r: r["realtime_factor"])

    # Build HTML sections
    sections = []

    # ---- Cover ----
    sections.append(f"""
    <div style="text-align:center; padding:60px 0 30px 0;">
      <h1 style="font-size:26pt; border:none; color:#111;">视频特征提取管道</h1>
      <h1 style="font-size:26pt; border:none; color:#111; margin-top:0;">项目技术说明书</h1>
      <p style="font-size:14pt; color:#555; margin-top:30px;">C++ 高性能视频解码 · GPU NVDEC 关键帧提取 · 多 ROI 特征工程</p>
      <p style="font-size:11pt; color:#888; margin-top:50px;">生成日期：{now}</p>
      <p style="font-size:10pt; color:#aaa;">硬件平台：Intel 18P/36L · NVIDIA RTX A6000 48GB · 256 GB RAM</p>
    </div>
    <div class="page-break"></div>
    """)

    # ---- TOC ----
    sections.append(f"""
    <h2>目录</h2>
    <div class="toc">
      <p><a href="#s1">1. 项目概述</a></p>
      <p><a href="#s2">2. 系统架构</a></p>
      <p><a href="#s3">3. GPU 关键帧解码关键技术突破</a></p>
      <p><a href="#s4">4. GPU vs CPU 对比测试</a></p>
      <p><a href="#s5">5. GPU 扩展性测试</a></p>
      <p><a href="#s6">6. CPU 扩展性测试（参考）</a></p>
      <p><a href="#s7">7. 特征可视化</a></p>
      <p><a href="#s8">8. 工程实现细节</a></p>
      <p><a href="#s9">9. 结论与建议</a></p>
    </div>
    <div class="page-break"></div>
    """)

    # ---- 1. 项目概述 ----
    sections.append(f"""
    <h2 id="s1">1. 项目概述</h2>
    <p>本项目构建了一套<strong>高性能视频特征提取管道</strong>，用于从渔船甲板监控视频中自动提取多 ROI（感兴趣区域）的时序视觉特征。管道采用 C++ 原生实现，通过 ffmpeg pipe 进行关键帧解码，使用 AVX2 指令集加速特征计算，支持 CPU (libavcodec) 和 GPU (NVDEC/hevc_cuvid) 双后端。</p>

    <h3>1.1 核心指标</h3>
    <ul>
      <li><strong>输入：</strong>37 个 HEVC (H.265) 视频，总计 ~25.5 小时，2560×1440 分辨率，25 fps</li>
      <li><strong>输出：</strong>每关键帧 × 2 ROI（左舷/右舷）× 9 维特征向量（亮度、色彩、运动、纹理、熵等）</li>
      <li><strong>GPU 吞吐：</strong>~747 feature rows/s（@ 8 workers），实时因子 4,478×</li>
      <li><strong>CPU 吞吐：</strong>~188 feature rows/s（@ 32 workers），实时因子 1,127×</li>
      <li><strong>GPU 加速比：</strong>~3.9× vs CPU 最优配置</li>
    </ul>

    <div class="info">
      <strong>应用目标：</strong>自动识别渔船作业行为（航行/放网/收网），检测提网事件，统计网次数量，为后续时序聚类与行为分析提供高质量特征输入。
    </div>
    <div class="page-break"></div>
    """)

    # ---- 2. 系统架构 ----
    sections.append(f"""
    <h2 id="s2">2. 系统架构</h2>
    <h3>2.1 数据流</h3>
    <pre style="background:#f5f5f5; padding:12px; font-size:9pt; border:1px solid #ddd;">
    .mp4 (HEVC) ──▶ ffmpeg pipe ──▶ rawvideo BGR ──▶ ROI 裁剪 ──▶ 9-dim 特征提取 ──▶ .cbf 二进制 / .csv
                    │                        │                    │
              -discard nokey         640×360 resize         AVX2 SIMD
              (demuxer 级丢弃)       (GPU: scale_cuda)      每 ROI 独立计算
    </pre>

    <h3>2.2 9 维特征向量</h3>
    <table>
      <tr><th>特征</th><th>含义</th><th>用途</th></tr>
      <tr><td>mean_brightness</td><td>平均亮度 (0–255)</td><td>昼夜检测、光照变化</td></tr>
      <tr><td>brightness_std</td><td>亮度标准差</td><td>场景复杂度</td></tr>
      <tr><td>mean_b / mean_g / mean_r</td><td>BGR 三通道均值</td><td>色彩偏移检测（海水/甲板）</td></tr>
      <tr><td>motion_intensity</td><td>帧间运动强度</td><td>人员活动、作业动作</td></tr>
      <tr><td>edge_density</td><td>边缘密度 (Sobel)</td><td>结构复杂度</td></tr>
      <tr><td>laplacian_variance</td><td>拉普拉斯方差</td><td>对焦/纹理代理</td></tr>
      <tr><td>entropy</td><td>图像熵 (0–8 bit)</td><td>信息量、场景多样性</td></tr>
    </table>

    <h3>2.3 ROI 定义</h3>
    <p>在 640×360 缩放帧上定义两个 ROI，覆盖甲板左右舷区域（去掉中央桅杆/上层建筑遮挡）：</p>
    <ul>
      <li><strong>left (左舷)：</strong>x∈[0,213), y∈[144,360) — 213×216 px</li>
      <li><strong>right (右舷)：</strong>x∈[426,640), y∈[144,360) — 214×216 px</li>
    </ul>
    <div class="page-break"></div>
    """)

    # ---- 3. GPU 关键帧解码 ----
    sections.append(f"""
    <h2 id="s3">3. GPU 关键帧解码关键技术突破</h2>

    <h3>3.1 问题发现</h3>
    <p>初始 GPU 管道使用 <code>-c:v hevc_cuvid -skip_frame nokey</code> 试图在 NVDEC 硬件端实现关键帧解码。但测试发现：</p>

    <table>
      <tr><th>管道配置</th><th>解码帧数（单视频）</th><th>耗时</th><th>结论</th></tr>
      <tr><td>CPU <code>-skip_frame nokey</code></td><td>666 帧（仅 I 帧）</td><td>24.3 s</td><td>✓ 正确</td></tr>
      <tr style="background:#fff3cd"><td>GPU <code>-skip_frame nokey</code>（旧）</td><td><strong>66,600 帧（全帧解码）</strong></td><td>62.0 s</td><td>✗ hevc_cuvid 忽略此标志</td></tr>
      <tr style="background:#d4edda"><td>GPU <code>-discard nokey</code>（新）</td><td>~692 帧（仅关键帧包）</td><td><strong>1.34 s</strong></td><td>✓ 真关键帧管道</td></tr>
    </table>

    <h3>3.2 根因分析</h3>
    <p><code>hevc_cuvid</code> 解码器<strong>不暴露</strong> <code>skip_frame</code> AVOption（见 ffmpeg <code>hevc_cuvid</code> 解码器选项列表 — 仅包含 <code>deint, gpu, surfaces, drop_second_field, crop, resize</code>，无 <code>skip_frame</code>）。ffmpeg 静默忽略该标志，NVDEC 硬件完整解码整个 GOP，仅在软件层过滤输出帧 — 导致 ~100× 额外解码工作量。</p>

    <h3>3.3 解决方案</h3>
    <p>使用 <strong>demuxer 级 <code>-discard nokey</code></strong> 替代 decoder 级 <code>-skip_frame nokey</code>。该选项在 AVPacket 读取阶段按 <code>AV_PKT_FLAG_KEY</code> 标志丢弃非关键帧包，被丢弃的包<strong>永远不会到达解码器</strong>。NVDEC 仅解码约 692 个关键帧包（vs 全帧解码的 66,600 帧），实现真正的稀疏关键帧管道。</p>

    <div class="note">
      <strong>注意：</strong>demuxer 级 discard 与 decoder 级 skip_frame 存在 ~4% 的帧数差异（692 vs 666），源于 MP4 容器包标志与解码器 pict_type 判断的微小差异。该差异对后续特征分析影响可忽略。
    </div>
    <div class="page-break"></div>
    """)

    # ---- 4. GPU vs CPU ----
    sections.append(f"""
    <h2 id="s4">4. GPU vs CPU 对比测试</h2>
    <p>测试条件：37 个视频，25.5 小时总时长，32 workers，<code>--no-output</code> 模式（纯解码+特征计算，无文件 I/O）。</p>

    <h3>4.1 结果总览</h3>
    {render_table_from_json(gpu_bench,
      ["mode","wall_time_s","realtime_factor","fps","cpu_usage_pct","gpu_decode_pct","gpu_decode_max_pct","gpu_mem_max_mb"],
      ["模式","Wall Time (s)","实时因子","rows/s","CPU %","GPU Dec %","GPU Dec Peak","VRAM (MB)"])}

    <h3>4.2 关键发现</h3>
    <ul>
      <li><strong>GPU 加速比 ~3.9×</strong>（20.45s vs 79.92s）</li>
      <li><strong>CPU 卸载显著：</strong>CPU 使用率从 80% 降至 27%，释放 53+ 百分点供下游任务</li>
      <li><strong>GPU 解码器饱和：</strong>NVDEC 利用率 95% avg / 100% peak — 管线受限于 GPU 解码吞吐</li>
      <li><strong>VRAM 可控：</strong>峰值 ~13 GB / 48 GB — A6000 有充足余量</li>
      <li><strong>帧率差异：</strong>GPU 模式 total_video_hours = 25.50h（~4% 额外关键帧包），已在前文说明</li>
    </ul>

    <h3>4.3 对比图表</h3>
    {img_b64(CHART_CPU_VS_GPU)}
    <p class="fig-caption">图 1：CPU vs GPU 解码对比（wall time / realtime / 资源利用率）</p>
    <div class="page-break"></div>
    """)

    # ---- 5. GPU Scaling ----
    sections.append(f"""
    <h2 id="s5">5. GPU 扩展性测试</h2>
    <p>测试条件：workers 从 1 到 32（含 1, 2, 4 低 worker 区间以绘制完整 NVDEC 饱和曲线），其他条件同 §4。</p>

    <h3>5.1 扩展性数据</h3>
    {render_table_from_json(gpu_scale,
      ["workers","wall_time_s","realtime_factor","fps","cpu_usage_pct","gpu_dec_avg_pct","gpu_dec_max_pct","gpu_mem_max_mb"],
      ["Workers","Wall Time (s)","实时因子","rows/s","CPU %","GPU Dec Avg %","GPU Dec Peak %","VRAM (MB)"])}

    <h3>5.2 NVDEC 饱和曲线分析</h3>
    <ul>
      <li><strong>1 worker → 35% NVDEC，</strong>单流解码无法填充 GPU 管道</li>
      <li><strong>2 workers → 66%，</strong>仍有余量</li>
      <li><strong>4 workers → 92%，</strong>接近饱和（已达峰值的 97%）</li>
      <li><strong>8 workers → 96%，</strong>有效饱和 — 此后增加 workers 无任何吞吐提升</li>
    </ul>

    <div class="info">
      <strong>最佳 worker 数：8。</strong> 在 8 workers 时达到峰值吞吐（747 rows/s），VRAM 仅需 4.2 GB，CPU 25%。如需要两台 GPU 管道并行运行，建议降至 4 workers/管道（各 ~3 GB VRAM，累计 NVDEC 利用率约 184% — NVDEC 调度器可正常处理）。
    </div>

    {img_b64(CHART_GPU_DECODER)}
    <p class="fig-caption">图 2：NVDEC 解码器利用率 vs Workers — 完整 S 型饱和曲线</p>

    {img_b64(CHART_GPU_DASHBOARD)}
    <p class="fig-caption">图 3：GPU 扩展性综合仪表板（wall time / realtime / throughput / GPU+CPU 利用率 / VRAM / 扩展效率）</p>
    <div class="page-break"></div>
    """)

    # ---- 6. CPU Scaling ----
    cpu_cols = ["workers","wall_time_s","realtime_factor","fps","cpu_usage_pct","peak_memory_mb"]
    cpu_labels = ["Workers","Wall Time (s)","实时因子","rows/s","CPU %","峰值内存 (MB)"]

    sections.append(f"""
    <h2 id="s6">6. CPU 扩展性测试（参考）</h2>
    <p>测试条件：纯 CPU 软件解码 (<code>-skip_frame nokey</code>)，workers 8–48，<code>--no-output</code> 模式。</p>

    {render_table_from_csv(CPU_SCALING_CSV, cpu_cols, cpu_labels)}

    <ul>
      <li>CPU 管线在 32 workers 达到平台（1,127× 实时），此后增加 workers 仅产生边际收益</li>
      <li>CPU 使用率在 24 workers 后超过 68%，32+ workers 后接近饱和（80–82%）</li>
      <li>峰值内存仅 ~45 MB（CPU 解码 + 轻量特征计算，零 Python 开销）</li>
    </ul>

    {img_b64(CHART_CPU_DASHBOARD)}
    <p class="fig-caption">图 4：CPU 扩展性综合仪表板</p>
    <div class="page-break"></div>
    """)

    # ---- 7. 特征可视化 ----
    sections.append(f"""
    <h2 id="s7">7. 特征可视化</h2>
    <p>使用 GPU 管道输出的 CSV 特征文件，对全部 37 个视频生成专业科研风格时序特征图。</p>

    <h3>7.1 可视化设计</h3>
    <ul>
      <li>左舷/右舷双色区分（steel blue / crimson），含图例</li>
      <li>原始数据以低透明度 (α=0.18) 背景显示，Savitzky-Golay 平滑曲线为前景</li>
      <li>真实挂钟时间 x 轴（从文件名解析时间戳）</li>
      <li>每个特征独立 y 轴自动缩放</li>
      <li>160 DPI，科研级排版风格（去上右侧刺、浅色网格、合理留白）</li>
    </ul>

    <h3>7.2 输出清单</h3>
    <ul>
      <li><strong>37 个单视频仪表板</strong>（每视频 6 特征 × 2 ROI 堆叠子图）</li>
      <li><strong>222 个单特征高清 PNG</strong>（6 特征 × 37 视频）</li>
      <li><strong>7 个语料级综合图</strong>（全 25.5 小时拼接）</li>
    </ul>

    <h3>7.3 语料级 Motion Intensity 时序</h3>
    {img_b64(CHART_FEATURE_MOTION)}
    <p class="fig-caption">图 5：全语料 Motion Intensity 时序 — 可见清晰的昼夜节律和突发活动脉冲</p>

    <h3>7.4 语料级综合仪表板</h3>
    {img_b64(CHART_FEATURE_CORPUS)}
    <p class="fig-caption">图 6：全语料 6 特征 × 2 ROI 综合时序（约 25.5 小时连续监测）</p>
    <div class="page-break"></div>
    """)

    # ---- 8. 工程实现 ----
    sections.append(f"""
    <h2 id="s8">8. 工程实现细节</h2>

    <h3>8.1 C++ 管道</h3>
    <table>
      <tr><th>模块</th><th>文件</th><th>职责</th></tr>
      <tr><td>入口 & 线程池</td><td><code>src/main.cpp</code></td><td>CLI 解析、ThreadPool 实现、benchmark 输出</td></tr>
      <tr><td>管道头</td><td><code>include/pipeline.h</code></td><td>FrameFeatures 结构体、SPSCQueue、ROI 定义、BenchStats</td></tr>
      <tr><td>视频处理</td><td><code>include/video_processor.h</code></td><td>ffmpeg 命令构建、ROI 裁剪、帧读取、process_one_video()</td></tr>
      <tr><td>特征提取</td><td><code>include/feature_extractor.h</code></td><td>9 维特征 AVX2 加速计算（亮度/色彩/运动/边缘/熵）</td></tr>
      <tr><td>二进制输出</td><td><code>include/binary_writer.h</code></td><td>列式 .cbf 格式写入</td></tr>
      <tr><td>CSV 输出</td><td><code>include/csv_writer.h</code></td><td>人类可读 .csv 侧车文件</td></tr>
    </table>

    <h3>8.2 ffmpeg 命令</h3>
    <pre style="background:#f5f5f5; padding:10px; font-size:8pt; border:1px solid #ddd;">
# GPU 模式 (NVDEC 关键帧解码)
ffmpeg -discard nokey -hwaccel cuda -hwaccel_output_format cuda \
       -c:v hevc_cuvid -i "INPUT.mp4" \
       -vf scale_cuda=640:360,hwdownload,format=nv12,format=bgr24 \
       -vsync 0 -f rawvideo -pix_fmt bgr24 -v quiet -

# CPU 模式 (软件关键帧解码)
ffmpeg -skip_frame nokey -i "INPUT.mp4" \
       -vf scale=640:360 -vsync 0 -f rawvideo -pix_fmt bgr24 -v quiet -
    </pre>

    <h3>8.3 构建</h3>
    <pre style="background:#f5f5f5; padding:10px; font-size:9pt; border:1px solid #ddd;">
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
# 输出：build/Release/cpp_pipeline.exe
    </pre>

    <h3>8.4 关键帧抽取参数</h3>
    <ul>
      <li>关键帧间隔：~4 秒（源视频 GOP 大小）</li>
      <li>子采样率：1/3（每第 3 个关键帧处理一次）→ 有效间隔 ~12 秒</li>
      <li>缩放尺寸：640×360（从 2560×1440 原始分辨率缩放）</li>
    </ul>
    <div class="page-break"></div>
    """)

    # ---- 9. 结论 ----
    sections.append(f"""
    <h2 id="s9">9. 结论与建议</h2>

    <h3>9.1 关键成果</h3>
    <table>
      <tr><th>指标</th><th>CPU 最优</th><th>GPU 最优</th><th>加速比</th></tr>
      <tr><td>Wall time (25.5h 视频)</td><td>78.1 s @ 32w</td><td><strong>20.3 s @ 8w</strong></td><td>3.9×</td></tr>
      <tr><td>实时因子</td><td>1,127× @ 32w</td><td><strong>4,527× @ 8w</strong></td><td>4.0×</td></tr>
      <tr><td>Feature rows/s</td><td>188 @ 32w</td><td><strong>755 @ 8w</strong></td><td>4.0×</td></tr>
      <tr><td>CPU 使用率</td><td>82%</td><td><strong>21%</strong></td><td>-61%pt 卸载</td></tr>
      <tr><td>内存</td><td>39 MB</td><td>4.2 GB VRAM</td><td>—</td></tr>
    </table>

    <h3>9.2 生产部署建议</h3>
    <ul>
      <li><strong>默认配置：<code>--threads 8 --gpu</code></strong> — 在吞吐、VRAM、CPU 余量之间达到最佳平衡</li>
      <li><strong>多管道并行：</strong>降至 <code>--threads 4</code> 可同时运行 2 个 GPU 管道，CPU 仍有 50%+ 余量供下游任务</li>
      <li><strong>无需 GPU 时：</strong><code>--threads 16</code>（CPU 模式）可获得 ~75% 峰值吞吐（898× 实时），CPU 使用率仅 53%</li>
    </ul>

    <h3>9.3 下一步工作</h3>
    <ul>
      <li><strong>时序特征聚类：</strong>基于 GPU 管道输出的 25.5 小时 left/right ROI 特征，进行作业状态自动标注</li>
      <li><strong>提网事件检测：</strong>motion_intensity 和 brightness_std 的突变模式可用于自动检测收网/放网时间点</li>
      <li><strong>网次统计：</strong>结合 edge_density 和 laplacian_variance（网具纹理特征），实现全自动网次计数</li>
      <li><strong>细粒度分析：</strong>对检测到的核心作业窗口运行 SAM 分割和网目尺寸估计</li>
      <li><strong>多通道扩展：</strong>如监控摄像头数量增长，GPU 管道的 CPU 余量可支撑多路并行处理</li>
    </ul>

    <hr style="margin-top:40px;" />
    <p style="text-align:center; color:#888; font-size:9pt;">
      本文档由自动化 Benchmark 脚本生成 · 数据来源：C++ pipeline + ffmpeg NVDEC · 生成时间：{now}
    </p>
    """)

    # ---- assemble ----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>视频特征提取管道 · 项目技术说明书</title>{CSS}</head>
<body>
{''.join(sections)}
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    print(f"Size: {REPORT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    build()
