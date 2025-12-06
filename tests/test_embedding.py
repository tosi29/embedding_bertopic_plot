import pytest

from embedding_bertopic_plot.embedding import (
    AWSEmbeddingProvider,
    DummyEmbeddingProvider,
    EmbeddingConfig,
    create_provider,
)


def test_dummy_embedding_is_deterministic():
    provider = DummyEmbeddingProvider(dimension=3)
    first = provider.embed("hello")
    second = provider.embed("hello")

    assert len(first) == 3
    assert first == second
    assert all(isinstance(value, float) for value in first)


def test_create_provider_requires_aws_env(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")

    config = EmbeddingConfig(backend="aws")

    with pytest.raises(RuntimeError, match="AWS backend requires.*AWS_ACCESS_KEY_ID"):
        create_provider(config)


def test_create_provider_aws_with_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")

    config = EmbeddingConfig(backend="aws")

    provider = create_provider(config)

    assert isinstance(provider, AWSEmbeddingProvider)
    assert provider.model == "amazon.titan-embed-text-v2:0"
