"""CLI for validating candidate images against a reference face (validation phase only)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path — must come before any src.* imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_settings
from src.reporting.report_builder import write_all_reports
from src.validation.face_analyzer import FaceAnalyzer
from src.validation.validator import summarize_results, validate_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated candidate images against a reference face."
    )
    parser.add_argument("--reference", type=str, required=True, help="Path to reference face image")
    parser.add_argument("--candidate-dir", type=str, required=True, help="Directory containing candidate images")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: candidate-dir/../reports/)",
    )
    parser.add_argument("--run-name", type=str, default="validation", help="Run name for report header")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref_path = Path(args.reference)
    cand_dir = Path(args.candidate_dir)

    if not ref_path.exists():
        print(f"Error: Reference image not found: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not cand_dir.is_dir():
        print(f"Error: Candidate directory not found: {cand_dir}", file=sys.stderr)
        sys.exit(1)

    settings = load_settings(args.config)

    report_dir = Path(args.report_dir) if args.report_dir else cand_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    analyzer = FaceAnalyzer(settings)
    try:
        ref_embedding, _, _ = analyzer.analyze_reference(str(ref_path))
    except ValueError:
        print(f"Error: No face detected in reference image {ref_path}", file=sys.stderr)
        sys.exit(1)

    manifest_path = cand_dir / "candidate_manifest.jsonl"
    prompt_id_map: dict[str, str] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                img_name = Path(entry["image_path"]).name
                prompt_id_map[img_name] = entry.get("prompt_id", "unknown")

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    candidate_files = sorted(
        f for f in cand_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    )

    if not candidate_files:
        print(f"Error: No image files found in {cand_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    threshold = settings.validation.similarity_threshold

    for img_path in candidate_files:
        analysis = analyzer.analyze_file(str(img_path))
        prompt_id = prompt_id_map.get(img_path.name, img_path.stem)
        result = validate_single(ref_embedding, analysis, str(img_path), prompt_id, threshold)
        results.append(result)
        status_icon = "✓" if result.status == "passed" else "✗"
        sim_str = f"{result.similarity:.4f}" if result.similarity is not None else "N/A"
        print(f"  {status_icon} {img_path.name}: similarity={sim_str} [{result.status}]")

    summary = summarize_results(results)
    write_all_reports(results, summary, report_dir, run_name=args.run_name)

    print("\n--- Validation Summary ---")
    print(f"Total: {summary.total}, Passed: {summary.passed}, Failed: {summary.failed}")
    if summary.mean_similarity is not None:
        print(
            f"Similarity: mean={summary.mean_similarity:.4f}, "
            f"min={summary.min_similarity:.4f}, max={summary.max_similarity:.4f}"
        )
    print(f"\nReports written to: {report_dir}")

    sys.exit(0 if summary.passed > 0 else 1)


if __name__ == "__main__":
    main()
