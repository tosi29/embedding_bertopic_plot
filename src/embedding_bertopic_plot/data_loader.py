from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import DEFAULT_LABEL, TextRecord, is_valid_embedding


class DataLoaderConfig:
    def __init__(
        self,
        *,
        json_text_field: str = "text",
        json_label_field: str = "label",
        json_details_field: str = "details",
        json_embedding_field: str = "embedding",
        default_label: str = DEFAULT_LABEL,
    ) -> None:
        self.json_text_field = json_text_field
        self.json_label_field = json_label_field
        self.json_details_field = json_details_field
        self.json_embedding_field = json_embedding_field
        self.default_label = default_label


def _load_txt(path: Path, default_label: str) -> List[TextRecord]:
    with path.open("r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]
    return [TextRecord(text=text, label=default_label) for text in texts]


def _load_json(path: Path, config: DataLoaderConfig) -> List[TextRecord]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON data must be a list of objects")

    records: List[TextRecord] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            record = TextRecord.from_dict(
                item,
                text_field=config.json_text_field,
                label_field=config.json_label_field,
                details_field=config.json_details_field,
                embedding_field=config.json_embedding_field,
                default_label=config.default_label,
            )
        except ValueError:
            continue

        if record.embedding is not None and not is_valid_embedding(record.embedding):
            record.embedding = None
        records.append(record)

    return records


def load_records(path: Path, config: DataLoaderConfig) -> List[TextRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_txt(path, config.default_label)
    if suffix == ".json":
        return _load_json(path, config)
    raise ValueError(f"Unsupported input extension: {suffix}")


def dump_records(records: Iterable[TextRecord], path: Path) -> None:
    payload = [record.to_dict() for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
