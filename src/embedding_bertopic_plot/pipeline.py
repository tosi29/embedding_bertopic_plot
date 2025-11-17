from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .config import PipelineConfig
from .data_loader import DataLoaderConfig, dump_records, load_records
from .dim_reduction import create_reducer, reduce_embeddings
from .embedding import create_provider, fill_missing_embeddings
from .models import TextRecord
from .plotting import build_plot
from .topic_modeling import BerTopicModeler, run_topic_modeling

BERTOPIC_OUTLIER_LABEL = "Outliers -1"
logger = logging.getLogger(__name__)


def _format_topic(topic_id: int, overrides: dict[int, str] | None) -> str:
    if overrides and topic_id in overrides:
        return f"Topic {topic_id}: {overrides[topic_id]}"
    if topic_id == -1:
        return BERTOPIC_OUTLIER_LABEL
    return f"Topic {topic_id}"


def _ensure_embeddings(records: Iterable[TextRecord]) -> None:
    missing = [record for record in records if record.embedding is None]
    if missing:
        raise RuntimeError("Embeddings missing for some records; run embedding step first")


@contextmanager
def _log_step(step_name: str):
    logger.info("Starting %s step", step_name)
    try:
        yield
    finally:
        logger.info("Finished %s step", step_name)


@dataclass
class PipelineSteps:
    load: bool = True
    embed: bool = True
    topic_model: bool = True
    reduction: bool = True
    plot: bool = True

    @classmethod
    def from_sequence(cls, names: Iterable[str]) -> "PipelineSteps":
        normalized = {name.strip().lower() for name in names}
        return cls(
            load="load" in normalized,
            embed="embed" in normalized,
            topic_model="topic" in normalized or "topic_model" in normalized,
            reduction="reduction" in normalized or "reduce" in normalized,
            plot="plot" in normalized,
        )


def execute_pipeline(
    config: PipelineConfig,
    *,
    steps: PipelineSteps | None = None,
    records_path: Path | None = None,
    save_records: Path | None = None,
):
    steps = steps or PipelineSteps()

    loader_config = DataLoaderConfig(
        json_text_field=config.data.json_text_field,
        json_label_field=config.data.json_label_field,
        json_details_field=config.data.json_details_field,
        json_embedding_field=config.data.json_embedding_field,
        default_label=config.data.default_label,
    )

    records: List[TextRecord] = []
    if steps.load:
        load_path = records_path or config.data.input_path
        with _log_step("load"):
            records = load_records(load_path, loader_config)
    else:
        raise ValueError("Load step is required to create records")

    if steps.embed and config.embedding.backend.lower() != "none":
        with _log_step("embedding"):
            provider = create_provider(config.embedding)
            fill_missing_embeddings(records, provider)
            if save_records:
                dump_records(records, save_records)
    elif steps.embed:
        logger.info("Embedding step skipped because backend is set to 'none'")

    if steps.topic_model:
        with _log_step("topic modeling"):
            _ensure_embeddings(records)
            topic_result = run_topic_modeling(records, BerTopicModeler(config.topic_model))
            topic_ids = topic_result.topic_ids
    else:
        raise ValueError("Topic modeling step is required for downstream visualization")

    if steps.reduction:
        with _log_step("reduction"):
            _ensure_embeddings(records)
            reducer = create_reducer(config.reduction)
            reduced = reduce_embeddings((record.embedding for record in records), reducer)
    else:
        reduced = np.zeros((len(records), 2))

    if steps.plot:
        with _log_step("plot"):
            topic_labels = [
                _format_topic(topic_id, config.topic_name_overrides)
                for topic_id in topic_ids
            ]
            fig = build_plot(
                coordinates=reduced,
                topic_labels=topic_labels,
                texts=[record.text for record in records],
                details=[record.details for record in records],
                labels=[record.label for record in records],
                config=config.plot,
                source_name=config.data.input_path.name,
            )
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(config.output_path, include_plotlyjs="cdn")
        return fig

    return None
