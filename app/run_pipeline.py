import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import OneShotIDPipeline


def main():
    parser = argparse.ArgumentParser(
        description="OneShot-ID Identity-Consistent Generation & Validation"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to reference face image")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional name for output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    pipeline = OneShotIDPipeline(config_path=args.config)
    success = pipeline.run(
        ref_image_path=args.input,
        output_dir_name=args.output_name,
        seed=args.seed,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
