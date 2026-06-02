import os
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def validate_config(cfg):
    errors = []

    if not isinstance(cfg.get("video_root"), str) or not cfg["video_root"]:
        errors.append("video_root 必须是非空字符串")
    elif not os.path.exists(cfg["video_root"]):
        errors.append(f"video_root 路径不存在: {cfg['video_root']}")

    for d in ["output_dir", "cache_dir", "log_dir", "data_dir"]:
        if not isinstance(cfg.get(d), str):
            errors.append(f"{d} 必须是字符串")

    rois = cfg.get("rois")
    if not isinstance(rois, dict) or len(rois) == 0:
        errors.append("rois 必须定义至少一个区域")
    else:
        for name, roi in rois.items():
            for key in ["x_min", "x_max", "y_min", "y_max"]:
                val = roi.get(key)
                if not isinstance(val, (int, float)):
                    errors.append(f"rois.{name}.{key} 必须是数值")
                elif val < 0.0 or val > 1.0:
                    errors.append(f"rois.{name}.{key}={val} 超出[0,1]范围")
            if roi.get("x_min", 0) >= roi.get("x_max", 1):
                errors.append(f"rois.{name}: x_min >= x_max")
            if roi.get("y_min", 0) >= roi.get("y_max", 1):
                errors.append(f"rois.{name}: y_min >= y_max")

    adaptive = cfg.get("adaptive_sampling", {})
    if adaptive.get("enabled", False):
        nfps = adaptive.get("normal_fps", 1.0)
        bfps = adaptive.get("burst_fps", 5.0)
        if not (0.1 <= nfps <= 25.0):
            errors.append(f"adaptive_sampling.normal_fps={nfps} 超出合理范围[0.1, 25]")
        if not (0.1 <= bfps <= 25.0):
            errors.append(f"adaptive_sampling.burst_fps={bfps} 超出合理范围[0.1, 25]")
        if bfps <= nfps:
            errors.append("adaptive_sampling.burst_fps 必须大于 normal_fps")
        thresh = adaptive.get("trigger_threshold", 0.7)
        if not (0.0 < thresh < 1.0):
            errors.append(f"adaptive_sampling.trigger_threshold={thresh} 必须在(0,1)内")

    rolling = cfg.get("rolling", {})
    ws = rolling.get("window_sec", 30)
    if not isinstance(ws, (int, float)) or ws <= 0:
        errors.append(f"rolling.window_sec={ws} 必须为正数")

    nw = cfg.get("num_workers", 4)
    if not isinstance(nw, int) or nw < 1:
        errors.append(f"num_workers={nw} 必须为正整数")

    return errors


def validate_and_report(config_path="configs/config.yaml"):
    print(f"验证配置: {config_path}")
    try:
        cfg = load_config(config_path)
    except yaml.YAMLError as e:
        print(f"  YAML解析失败: {e}")
        return None
    except FileNotFoundError:
        print(f"  文件不存在: {config_path}")
        return None

    errors = validate_config(cfg)
    if errors:
        print(f"  发现 {len(errors)} 个问题:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  配置验证通过")
        print(f"  video_root: {cfg['video_root']}")
        print(f"  ROI数量: {len(cfg['rois'])}")
        print(f"  采样: {cfg['adaptive_sampling']['normal_fps']}fps (burst: {cfg['adaptive_sampling']['burst_fps']}fps)")
        print(f"  并行: {cfg['num_workers']} workers")
    return cfg


if __name__ == "__main__":
    validate_and_report()
