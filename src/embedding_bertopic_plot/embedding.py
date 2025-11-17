from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence

import litellm
from google import genai
from google.genai import types as genai_types

from .models import TextRecord


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: Sequence[str]) -> List[list[float]]:
        return [self.embed(text) for text in texts]


@dataclass
class GeminiEmbeddingProvider:
    model: str
    task_type: str
    api_key: str
    retry_delay_sec: float = 0.0

    def __post_init__(self) -> None:
        self.client = genai.Client(api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type=self.task_type),
        )
        if self.retry_delay_sec:
            time.sleep(self.retry_delay_sec)
        return result.embeddings[0].values


@dataclass
class AWSEmbeddingProvider:
    model: str
    api_key: str | None = None
    region_name: str | None = None

    def embed(self, text: str) -> list[float]:
        response = litellm.embedding(
            model=self.model,
            input=text,
            api_key=self.api_key,
            aws_region_name=self.region_name,
        )
        return response["data"][0]["embedding"]


@dataclass
class DummyEmbeddingProvider:
    dimension: int = 5

    def embed(self, text: str) -> list[float]:
        seed = hash(text) % (2**32)
        rnd = random.Random(seed)
        return [float(rnd.random()) for _ in range(self.dimension)]


class NoOpEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("Embedding generation is disabled in this mode")


@dataclass
class EmbeddingConfig:
    backend: str = "gemini"
    model: str = "gemini-embedding-exp-03-07"
    task_type: str = "CLUSTERING"
    retry_delay_sec: float = 0.0
    api_key_env: str | None = None
    region: str | None = None


def create_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    backend = config.backend.lower()
    api_key = os.environ.get(config.api_key_env or "") if config.api_key_env else None
    if backend == "gemini":
        if api_key is None:
            raise RuntimeError("Gemini backend requires api_key_env to be set")
        return GeminiEmbeddingProvider(
            model=config.model,
            task_type=config.task_type,
            retry_delay_sec=config.retry_delay_sec,
            api_key=api_key,
        )
    if backend == "aws":
        return AWSEmbeddingProvider(model=config.model, api_key=api_key, region_name=config.region)
    if backend == "dummy":
        return DummyEmbeddingProvider()
    if backend == "none":
        return NoOpEmbeddingProvider()
    raise ValueError(f"Unsupported embedding backend: {config.backend}")


def fill_missing_embeddings(records: Iterable[TextRecord], provider: EmbeddingProvider) -> None:
    for record in records:
        if record.embedding is None:
            record.embedding = provider.embed(record.text)
