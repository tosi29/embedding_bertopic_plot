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
