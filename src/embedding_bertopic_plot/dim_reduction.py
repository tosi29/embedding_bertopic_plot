from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
import umap


class DimensionalityReducer:
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _validate_embeddings(records: Iterable[list[float]]) -> np.ndarray:
    return np.array(list(records), dtype=float)


@dataclass
class UMAPReducer(DimensionalityReducer):
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        return reducer.fit_transform(embeddings)


@dataclass
class PCAReducer(DimensionalityReducer):
    n_components: int = 2

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        return PCA(n_components=self.n_components).fit_transform(embeddings)


@dataclass
class TSNEReducer(DimensionalityReducer):
    n_components: int = 2
    random_state: int = 42

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for t-SNE") from exc
        return TSNE(n_components=self.n_components, random_state=self.random_state).fit_transform(
            embeddings
        )


@dataclass
class DimensionalityConfig:
    method: str = "umap"
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


def create_reducer(config: DimensionalityConfig) -> DimensionalityReducer:
    method = config.method.lower()
    if method == "umap":
        return UMAPReducer(
            n_components=config.n_components,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.random_state,
        )
    if method == "pca":
        return PCAReducer(n_components=config.n_components)
    if method == "tsne":
        return TSNEReducer(n_components=config.n_components, random_state=config.random_state)
    raise ValueError(f"Unsupported reduction method: {config.method}")


def reduce_embeddings(records: Iterable[list[float]], reducer: DimensionalityReducer) -> np.ndarray:
    embeddings = _validate_embeddings(records)
    return reducer.reduce(embeddings)
