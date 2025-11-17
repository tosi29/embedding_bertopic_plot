from pathlib import Path

from embedding_bertopic_plot.data_loader import DataLoaderConfig, load_records
from embedding_bertopic_plot.models import DEFAULT_LABEL


def test_load_txt(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("first\n\nsecond\n")

    records = load_records(path, DataLoaderConfig(default_label="file"))

    assert [record.text for record in records] == ["first", "second"]
    assert all(record.label == "file" for record in records)
    assert all(record.embedding is None for record in records)


def test_load_json_with_embedding_and_defaults(tmp_path: Path):
    path = tmp_path / "sample.json"
    path.write_text(
        """[
        {"text": "hello", "embedding": [1.0, 0.0], "label": "x", "details": "meta"},
        {"text": "skip me", "embedding": "not valid"},
        {"text": "world"}
    ]"""
    )

    records = load_records(path, DataLoaderConfig())

    assert len(records) == 3
    assert records[0].embedding == [1.0, 0.0]
    assert records[1].embedding is None
    assert records[2].label == DEFAULT_LABEL
