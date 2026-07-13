from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    weights: Path
    confidence: float
    classes: dict[int, str]


def load_models(path: str | Path) -> list[ModelConfig]:
    """Load local model definitions and validate the small public config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}

    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("配置文件必须包含至少一个 models 条目。")

    models: list[ModelConfig] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("models 中的每一项都必须是对象。")
        try:
            classes = {int(key): str(value) for key, value in entry["classes"].items()}
            weights = Path(entry["weights"])
            if not weights.is_absolute():
                weights = (config_path.parent.parent / weights).resolve()
            models.append(ModelConfig(str(entry["name"]), weights, float(entry.get("confidence", 0.5)), classes))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"无效模型配置：{entry}") from exc
    return models
