from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
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

    def embed(self, text: str) -> list[float]:
        response = litellm.embedding(
            model=self.model,
            input=text,
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
class GeminiEmbeddingConfig:
    model: str = "gemini-embedding-exp-03-07"
    task_type: str = "CLUSTERING"
    retry_delay_sec: float = 0.0
    api_key_env: str | None = None


@dataclass
class AWSEmbeddingConfig:
    model: str = "amazon.titan-embed-text-v2"


@dataclass
class EmbeddingConfig:
    backend: str = "gemini"
    gemini: GeminiEmbeddingConfig = field(default_factory=GeminiEmbeddingConfig)
    aws: AWSEmbeddingConfig = field(default_factory=AWSEmbeddingConfig)


def create_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    backend = config.backend.lower()
    if backend == "gemini":
        api_key_env = config.gemini.api_key_env
        api_key = os.environ.get(api_key_env or "") if api_key_env else None
        if api_key is None:
            raise RuntimeError("Gemini backend requires api_key_env to be set")
        return GeminiEmbeddingProvider(
            model=config.gemini.model,
            task_type=config.gemini.task_type,
            retry_delay_sec=config.gemini.retry_delay_sec,
            api_key=api_key,
        )
    if backend == "aws":
        return AWSEmbeddingProvider(
            model=config.aws.model,
        )
    if backend == "dummy":
        return DummyEmbeddingProvider()
    if backend == "none":
        return NoOpEmbeddingProvider()
    raise ValueError(f"Unsupported embedding backend: {config.backend}")


def fill_missing_embeddings(records: Iterable[TextRecord], provider: EmbeddingProvider) -> None:
    for record in records:
        if record.embedding is None:
            record.embedding = provider.embed(record.text)
