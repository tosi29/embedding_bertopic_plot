from pathlib import Path

from embedding_bertopic_plot.config import load_config


def test_load_config_paths(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        data:
          path: input.json
        embedding:
          backend: dummy
        topic_model:
          language: english
        reduction:
          method: pca
        plot:
          title_prefix: Test
        output:
          path: output.html
        """
    )

    config = load_config(config_path)

    assert config.data.input_path.name == "input.json"
    assert config.embedding.backend == "dummy"
    assert config.reduction.method == "pca"
    assert config.output_path.name == "output.html"
