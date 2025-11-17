from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Iterable, Sequence

import plotly.express as px


@dataclass
class PlotConfig:
    title_prefix: str = "BERTopic Clustering (UMAP)"
    text_wrap_width: int = 40


def wrap_text(value: str, width: int) -> str:
    if not value:
        return ""
    return textwrap.fill(value, width=width).replace("\n", "<br>")


def build_plot(
    *,
    coordinates: Sequence[Sequence[float]],
    topic_labels: Sequence[str],
    texts: Iterable[str],
    details: Iterable[str],
    labels: Iterable[str],
    config: PlotConfig,
    source_name: str,
):
    x_coords = [point[0] for point in coordinates]
    y_coords = [point[1] for point in coordinates]

    wrapped_texts = [wrap_text(text, config.text_wrap_width) for text in texts]
    wrapped_details = [wrap_text(detail, config.text_wrap_width) for detail in details]
    label_list = list(labels)
    topic_label_list = list(topic_labels)

    data_frame = {
        "x": x_coords,
        "y": y_coords,
        "topic": topic_label_list,
        "text": wrapped_texts,
        "details": wrapped_details,
        "label": label_list,
    }

    fig = px.scatter(
        data_frame=data_frame,
        x="x",
        y="y",
        color="topic",
        custom_data=["text", "details", "label", "topic"],
        title=f"{config.title_prefix} from {source_name}",
        labels={"x": "Dimension 1", "y": "Dimension 2", "color": "Topic"},
    )

    hovertemplate = (
        "<b>text:</b><br>%{customdata[0]}<br><br>"
        "<b>details:</b><br>%{customdata[1]}<br><br>"
        "<b>label:</b> %{customdata[2]}<br>"
        "<b>Topic:</b> %{customdata[3]}<br>"
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=hovertemplate)
    return fig
