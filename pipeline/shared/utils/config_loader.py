"""Pipeline config loader — unified YAML config for all layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_pipeline_config(path: str | Path | None = None) -> Dict[str, Any]:
    if path is not None:
        p = Path(path)
    else:
        p = Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_legacy_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)