from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import hdbscan
import numpy as np
import umap
from bertopic import BERTopic

from .models import TextRecord


class TopicModeler:
    def fit_transform(self, texts: Sequence[str], embeddings: np.ndarray) -> tuple[list[int], BERTopic]:
        raise NotImplementedError


@dataclass
class BerTopicConfig:
    language: str = "japanese"
    umap_kwargs: dict | None = None
    hdbscan_kwargs: dict | None = None


@dataclass
class TopicModelResult:
    topic_ids: List[int]
    topic_model: BERTopic


class BerTopicModeler(TopicModeler):
    def __init__(self, config: BerTopicConfig):
        self.config = config

    def fit_transform(self, texts: Sequence[str], embeddings: np.ndarray) -> TopicModelResult:
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            **(self.config.umap_kwargs or {}),
        )
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            **(self.config.hdbscan_kwargs or {}),
        )
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=None,
            language=self.config.language,
            verbose=True,
        )
        topic_ids, _ = topic_model.fit_transform(list(texts), embeddings=embeddings)
        return TopicModelResult(topic_ids=list(topic_ids), topic_model=topic_model)


def run_topic_modeling(records: Iterable[TextRecord], modeler: TopicModeler) -> TopicModelResult:
    texts = [record.text for record in records]
    embeddings = np.array([record.embedding for record in records], dtype=float)
    return modeler.fit_transform(texts, embeddings)
