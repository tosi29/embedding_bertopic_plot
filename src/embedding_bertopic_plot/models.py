from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_LABEL = "unknown"


@dataclass
class TextRecord:
    """Container for a single text sample and optional metadata."""

    text: str
    label: str = DEFAULT_LABEL
    details: str = ""
    embedding: Optional[list[float]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "text": self.text,
            "label": self.label,
            "details": self.details,
            "embedding": self.embedding,
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        text_field: str,
        label_field: str,
        details_field: str,
        embedding_field: str,
        default_label: str = DEFAULT_LABEL,
    ) -> "TextRecord":
        text = data.get(text_field)
        if text is None or not isinstance(text, str):
            raise ValueError(f"Missing or invalid text in field '{text_field}'")

        label = data.get(label_field, default_label) or default_label
        if not isinstance(label, str):
            label = default_label

        details = data.get(details_field, "")
        if not isinstance(details, str):
            details = ""

        embedding = data.get(embedding_field)
        extra = {
            key: value
            for key, value in data.items()
            if key
            not in {
                text_field,
                label_field,
                details_field,
                embedding_field,
            }
        }

        return cls(
            text=text,
            label=label,
            details=details,
            embedding=embedding if isinstance(embedding, list) else None,
            extra=extra,
        )


def is_valid_embedding(embedding: Any) -> bool:
    return isinstance(embedding, list) and all(
        isinstance(value, (float, int)) for value in embedding
    )
