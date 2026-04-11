"""CLI for generating candidate images (generation phase only)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate identity-consistent candidate images from a reference face."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to reference face image")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--run-name", type=str, default=None, help="Name for output run directory (auto-generated if not set)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: Input file {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file {args.config} not found.", file=sys.stderr)
        sys.exit(1)

    from src.config import load_settings

    settings = load_settings(str(config_path))

    if args.seed is not None:
        settings = settings.model_copy(
            update={
                "runtime": settings.runtime.model_copy(update={"seed": args.seed}),
            }
        )

    run_name = args.run_name or f"gen_{input_path.stem}"

    from src.reporting.report_builder import build_run_paths

    paths = build_run_paths(Path(settings.output.base_dir), run_name)

    from src.validation.face_analyzer import FaceAnalyzer

    analyzer = FaceAnalyzer(settings)
    ref_result = analyzer.analyze_file(str(input_path))

    if ref_result.face_info is None:
        print(f"Error: No face detected in reference image {args.input}", file=sys.stderr)
        sys.exit(1)

    from src.prompts.prompt_bank import get_prompt_specs

    specs = get_prompt_specs()

    from src.generation.instantid_generator import InstantIDGenerator

    generator = InstantIDGenerator(settings)
    try:
        records = generator.generate_batch(
            face_embedding=ref_result.face_info.embedding,
            face_landmarks=ref_result.face_info.landmarks,
            prompt_specs=specs,
            base_seed=settings.runtime.seed,
            output_dir=paths.candidates_dir,
        )

        manifest_path = paths.candidates_dir / "candidate_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for rec in records:
                line = json.dumps(
                    {
                        "image_path": rec.image_path,
                        "prompt_id": rec.prompt_id,
                        "seed": rec.seed,
                        "guidance_scale": rec.guidance_scale,
                        "num_inference_steps": rec.num_inference_steps,
                    },
                    ensure_ascii=False,
                )
                f.write(line + "\n")

        print(f"Generated {len(records)} candidates in {paths.candidates_dir}")
        print(f"Manifest: {manifest_path}")
    finally:
        generator.release()


if __name__ == "__main__":
    main()
