from embedding_bertopic_plot.embedding import DummyEmbeddingProvider


def test_dummy_embedding_is_deterministic():
    provider = DummyEmbeddingProvider(dimension=3)
    first = provider.embed("hello")
    second = provider.embed("hello")

    assert len(first) == 3
    assert first == second
    assert all(isinstance(value, float) for value in first)
