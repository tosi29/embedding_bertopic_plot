from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .dim_reduction import DimensionalityConfig
from .embedding import AWSEmbeddingConfig, EmbeddingConfig, GeminiEmbeddingConfig
from .plotting import PlotConfig
from .topic_modeling import BerTopicConfig


@dataclass
class DataConfig:
    input_path: Path
    json_text_field: str = "text"
    json_label_field: str = "label"
    json_details_field: str = "details"
    json_embedding_field: str = "embedding"
    default_label: str = "unknown"


@dataclass
class PipelineConfig:
    data: DataConfig
    embedding: EmbeddingConfig
    topic_model: BerTopicConfig
    reduction: DimensionalityConfig
    plot: PlotConfig
    output_path: Path
    topic_name_overrides: dict[int, str] | None = None


def _coerce_path(value: Any) -> Path:
    if value is None:
        raise ValueError("A file path must be provided in the configuration")
    if isinstance(value, Path):
        return value
    return Path(value)


def load_config(path: Path) -> PipelineConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    embedding_section = raw.get("embedding", {}) or {}
    backend = embedding_section.get("backend", "gemini")
    gemini_kwargs = dict(embedding_section.get("gemini", {}) or {})
    aws_kwargs = dict(embedding_section.get("aws", {}) or {})

    legacy_model = embedding_section.get("model")
    if legacy_model:
        if backend.lower() == "aws":
            aws_kwargs.setdefault("model", legacy_model)
        else:
            gemini_kwargs.setdefault("model", legacy_model)

    legacy_task_type = embedding_section.get("task_type")
    if legacy_task_type:
        gemini_kwargs.setdefault("task_type", legacy_task_type)

    if "retry_delay_sec" in embedding_section:
        gemini_kwargs.setdefault("retry_delay_sec", embedding_section["retry_delay_sec"])

    legacy_region = embedding_section.get("region")
    if legacy_region:
        aws_kwargs.setdefault("region", legacy_region)

    embedding_cfg = EmbeddingConfig(
        backend=backend,
        api_key_env=embedding_section.get("api_key_env"),
        gemini=GeminiEmbeddingConfig(**gemini_kwargs),
        aws=AWSEmbeddingConfig(**aws_kwargs),
    )
    topic_cfg = BerTopicConfig(
        language=raw.get("topic_model", {}).get("language", "japanese"),
        umap_kwargs=raw.get("topic_model", {}).get("umap", {}),
        hdbscan_kwargs=raw.get("topic_model", {}).get("hdbscan", {}),
    )
    reduction_cfg = DimensionalityConfig(**raw.get("reduction", {}))
    plot_cfg = PlotConfig(**raw.get("plot", {}))

    data_section = raw.get("data", {})
    data_cfg = DataConfig(
        input_path=_coerce_path(data_section.get("path")),
        json_text_field=data_section.get("json_text_field", "text"),
        json_label_field=data_section.get("json_label_field", "label"),
        json_details_field=data_section.get("json_details_field", "details"),
        json_embedding_field=data_section.get("json_embedding_field", "embedding"),
        default_label=data_section.get("default_label", "unknown"),
    )

    output_path = _coerce_path(raw.get("output", {}).get("path"))
    topic_names = raw.get("topic_model", {}).get("topic_names")

    return PipelineConfig(
        data=data_cfg,
        embedding=embedding_cfg,
        topic_model=topic_cfg,
        reduction=reduction_cfg,
        plot=plot_cfg,
        output_path=output_path,
        topic_name_overrides=topic_names,
    )
