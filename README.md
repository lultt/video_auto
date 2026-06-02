# 灯光罩网渔船前甲板视频自动作业解析系统

## 第一阶段：高速视频时序特征系统

固定摄像头 + 固定前甲板视角 → ROI时序特征 → parquet

## 为什么不使用 YOLO / 深度学习

这是一个**固定机位工业视频状态分析**问题，不是目标检测问题。

核心区别：
- 目标检测回答："画面里有什么物体？在哪里？" 
- 状态分析回答："当前作业处于什么阶段？什么时候发生了状态变化？"

对于固定摄像头、固定场景的工业视频：
1. **场景不变** — 摄像头不动，甲板不动，ROI固定。不需要"找到"目标。
2. **状态由时序决定** — 灯光亮灭、人员活动强度、网具运动，这些是时间序列信号，不是单帧分类问题。
3. **速度要求极高** — 2083个视频 × 每个40-60分钟，深度学习推理速度无法满足全量处理需求。
4. **特征可解释** — 亮度、运动强度、边缘密度等物理量直接对应作业状态，无需黑盒模型。

正确的技术路线：
```
视频 → 采样(1fps) → ROI裁剪 → 时序特征(亮度/运动/边缘) → parquet → 状态分析
```

后续阶段如需精细分类，可在时序特征基础上叠加轻量模型，而非对每帧做目标检测。

## 视频规格

| 参数 | 值 |
|------|-----|
| 分辨率 | 2560×1440 |
| 帧率 | 25fps |
| 单文件时长 | ~40-60分钟 |
| 总文件数 | 2083 |
| 存储位置 | `\\DS224plus\video\viedeo` |

## 项目结构

```
├── configs/config.yaml       # 全局配置
├── data/                     # 视频清单
├── cache/                    # 中间缓存
├── outputs/                  # parquet + 图表
├── logs/                     # 运行日志
├── src/
│   ├── scan_videos.py        # 扫描NAS生成视频清单
│   ├── video_reader.py       # 高速视频读取器
│   ├── extract_features.py   # ROI时序特征提取
│   ├── visualize_features.py # 特征可视化
│   └── benchmark_pipeline.py # 性能基准测试
├── requirements.txt
└── README.md
```

## 快速开始

```bash
conda activate yolonew
pip install -r requirements.txt --quiet

# 1. 扫描视频清单
python src/scan_videos.py

# 2. 性能基准测试（含GPU解码/IO瓶颈分析）
python src/benchmark_pipeline.py

# 3. 提取全量特征
python src/extract_features.py

# 4. 可视化
python src/visualize_features.py
```

## 输出格式

每个视频生成一个 parquet 文件，列结构：

| 列名 | 说明 |
|------|------|
| video_id | 视频文件名 |
| frame_idx | 原始帧号 |
| timestamp_sec | 时间戳(秒) |
| roi_name | ROI区域名 |
| mean_brightness | 亮度均值 |
| brightness_std | 亮度标准差 |
| mean_r / mean_g / mean_b | RGB通道均值 |
| motion_intensity | 运动强度 |
| edge_density | 边缘密度 |
| texture_complexity | 纹理复杂度 |

## 环境要求

- Python 3.9 / PyTorch 2.0.1+CUDA11.7（已有，不动）
- OpenCV 4.12（已有）
- RTX A6000（GPU解码加速）
- ffmpeg 8.1（已配置）
