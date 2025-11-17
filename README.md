# Embedding BERTopic Plot Pipeline

A modular, config-driven pipeline for loading text data, generating embeddings (Gemini, AWS via LiteLLM, dummy, or no-op), running BERTopic, applying dimensionality reduction, and exporting Plotly visualizations.

## Project layout

- `config.example.yaml`: Example configuration demonstrating all tunable parameters.
- `src/embedding_bertopic_plot/`: Modular pipeline code split by responsibility.
- `sample.py`: Original monolithic script (kept for reference, untouched).

## Requirements & environment

- Python 3.12+
- Dependencies are managed with [uv](https://docs.astral.sh/uv/). Create the environment via `uv sync`.
- Set API keys as environment variables when needed (e.g., `GEMINI_API_KEY` for Gemini, AWS credentials for LiteLLM/Bedrock models).

### AWS Bedrock via LiteLLM

When `embedding.backend` is `aws`, LiteLLM expects the standard AWS credentials in your environment. At minimum set the following (optionally `AWS_SESSION_TOKEN` when using temporary credentials):

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION_NAME=us-east-1
```

These variables match the [LiteLLM Bedrock embedding docs](https://docs.litellm.ai/docs/embedding/supported_embedding#bedrock-embedding). Without them the embedding step will fail because LiteLLM cannot create a Bedrock client.

## Running the pipeline

1. Copy `config.example.yaml` to a new file (e.g., `config.yaml`) and adjust paths and parameters.
2. Run the orchestrator:

```bash
uv run topic-pipeline --config config.yaml \
  --steps load,embed,topic,reduction,plot \
  --save-records outputs/records_with_embeddings.json
```

Key flags:
- `--input` / `--output`: Override paths defined in the config file.
- `--steps`: Choose which stages to execute; helpful for reusing cached embeddings (e.g., `--steps load,topic,reduction,plot`).
- `--embedding-backend`: Switch providers (`gemini`, `aws`, `dummy`, `none`).
- `--embedding-api-key-env`, `--embedding-model`: Override provider details at runtime.

## Configuration reference

```yaml
data:
  path: data/input.json        # .txt or .json supported
  json_text_field: text
  json_label_field: label
  json_details_field: details
  json_embedding_field: embedding
  default_label: unknown

embedding:
  backend: gemini              # gemini | aws | dummy | none
  api_key_env: GEMINI_API_KEY  # env var containing the API key
  gemini:
    model: gemini-embedding-exp-03-07
    task_type: CLUSTERING      # Gemini task type
    retry_delay_sec: 1
  aws:
    model: amazon.titan-embed-text-v2:0

topic_model:
  language: japanese
  umap:
    n_neighbors: 15
    n_components: 5
    min_dist: 0.0
    metric: cosine
  hdbscan:
    min_cluster_size: 5
    metric: euclidean
    cluster_selection_method: eom
  topic_names:
    -1: "その他・外れ値"         # Optional display label overrides

reduction:
  method: umap                 # umap | pca | tsne
  n_components: 2
  n_neighbors: 15
  min_dist: 0.1
  metric: cosine
  random_state: 42

plot:
  title_prefix: "BERTopic Clustering (UMAP)"
  text_wrap_width: 40
  show_point_text: false       # true to display wrapped text next to each point

output:
  path: outputs/bertopic_umap.html
```

## Notes
- When `embedding.backend` is `none`, embeddings must already be present in the input JSON; the pipeline will error if any are missing.
- Use `--save-records` to persist embeddings and speed up subsequent runs or testing.
- The modular design allows future providers (e.g., local sentence-transformers) or topic modelers to be added without touching the orchestrator.
