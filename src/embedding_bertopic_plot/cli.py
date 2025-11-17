from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig, load_config
from .pipeline import PipelineSteps, execute_pipeline


def _parse_steps(value: str) -> PipelineSteps:
    names = [part.strip() for part in value.split(",") if part.strip()]
    return PipelineSteps.from_sequence(names)


def _override_embedding(config: PipelineConfig, args: argparse.Namespace) -> None:
    if args.embedding_backend:
        config.embedding.backend = args.embedding_backend
    if args.embedding_model:
        backend = config.embedding.backend.lower()
        if backend == "aws":
            config.embedding.aws.model = args.embedding_model
        else:
            config.embedding.gemini.model = args.embedding_model
    if args.embedding_task_type:
        config.embedding.gemini.task_type = args.embedding_task_type
    if args.embedding_api_key_env:
        config.embedding.api_key_env = args.embedding_api_key_env


def _apply_path_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    if args.input:
        config.data.input_path = args.input
    if args.output:
        config.output_path = args.output



def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the embedding -> topic -> UMAP -> Plot pipeline")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML configuration file")
    parser.add_argument("--input", type=Path, help="Override input path from config")
    parser.add_argument("--output", type=Path, help="Override output HTML path from config")
    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default=PipelineSteps(),
        help="Comma separated steps to run (load,embed,topic,reduction,plot)",
    )
    parser.add_argument(
        "--save-records",
        type=Path,
        help="Optional path to save records with embeddings after the embedding step",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        help="Override embedding backend (gemini, aws, dummy, none)",
    )
    parser.add_argument("--embedding-model", type=str, help="Override embedding model name")
    parser.add_argument(
        "--embedding-task-type",
        type=str,
        help="Override embedding task type (for Gemini)",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        type=str,
        help="Environment variable that stores the embedding API key",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    _apply_path_overrides(config, args)
    _override_embedding(config, args)

    execute_pipeline(
        config,
        steps=args.steps if isinstance(args.steps, PipelineSteps) else PipelineSteps(),
        save_records=args.save_records,
    )


if __name__ == "__main__":
    main()
