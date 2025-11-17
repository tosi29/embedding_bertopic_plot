import json
from pathlib import Path

import pytest

from embedding_bertopic_plot.config import DataConfig, PipelineConfig
from embedding_bertopic_plot.dim_reduction import DimensionalityConfig
from embedding_bertopic_plot.embedding import EmbeddingConfig
from embedding_bertopic_plot.pipeline import PipelineSteps, execute_pipeline
from embedding_bertopic_plot.plotting import PlotConfig
from embedding_bertopic_plot.topic_modeling import BerTopicConfig

pytestmark = [
    # 固定シードの再現性を優先したいので random_state を使うが、並列化に関する UMAP の警告は無視する
    pytest.mark.filterwarnings("ignore:n_jobs value .* overridden .* random_state:UserWarning"),
    # 依存ライブラリ内の将来非推奨警告は現状抑止するしかない
    pytest.mark.filterwarnings("ignore:'force_all_finite' was renamed .*:FutureWarning"),
]


def _write_sample_input(path: Path) -> None:
    samples = [
        {"text": "おはよう", "label": "greeting", "details": "朝の挨拶"},
        {"text": "こんばんは", "label": "greeting", "details": "夜の挨拶"},
        {"text": "今日はいい天気", "label": "smalltalk", "details": "雑談"},
        {"text": "雨が降りそうです", "label": "smalltalk", "details": "天気の話"},
    ]
    path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")


def test_execute_pipeline_all_steps_with_dummy_embedding(tmp_path):
    input_path = tmp_path / "input.json"
    _write_sample_input(input_path)
    output_path = tmp_path / "plot.html"

    config = PipelineConfig(
        data=DataConfig(input_path=input_path, default_label="test"),
        embedding=EmbeddingConfig(backend="dummy"),
        topic_model=BerTopicConfig(
            language="japanese",
            umap_kwargs={
                "n_neighbors": 2,
                "n_components": 2,
                "min_dist": 0.0,
                "metric": "cosine",
                "random_state": 42,
            },
            hdbscan_kwargs={
                "min_cluster_size": 2,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "prediction_data": True,
            },
        ),
        reduction=DimensionalityConfig(method="pca", n_components=2),
        plot=PlotConfig(title_prefix="Test Plot", text_wrap_width=20, show_point_text=False),
        output_path=output_path,
        topic_name_overrides={0: "挨拶"},
    )

    fig = execute_pipeline(config, steps=PipelineSteps.from_sequence(["load", "embed", "topic", "reduction", "plot"]))

    assert fig is not None
    assert output_path.exists()
    saved_html = output_path.read_text(encoding="utf-8")
    assert "Test Plot" in saved_html
