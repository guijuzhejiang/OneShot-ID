"""Build run output paths and write validation reports (CSV, JSON, Markdown)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.validation.result_schema import ValidationResult, ValidationSummary


@dataclass
class RunPaths:
    """Standard output directory structure for a single run."""

    run_dir: Path  # outputs/runs/<run_name>/
    candidates_dir: Path  # outputs/runs/<run_name>/candidates/
    kept_dir: Path  # outputs/runs/<run_name>/kept/
    rejected_dir: Path  # outputs/runs/<run_name>/rejected/
    report_dir: Path  # outputs/runs/<run_name>/reports/
    faces_dir: Path  # outputs/runs/<run_name>/faces/


def build_run_paths(base_dir: Path, run_name: str) -> RunPaths:
    """Build the standard output path structure. Creates directories."""
    run_dir = base_dir / "runs" / run_name
    paths = RunPaths(
        run_dir=run_dir,
        candidates_dir=run_dir / "candidates",
        kept_dir=run_dir / "kept",
        rejected_dir=run_dir / "rejected",
        report_dir=run_dir / "reports",
        faces_dir=run_dir / "faces",
    )
    for d in [paths.candidates_dir, paths.kept_dir, paths.rejected_dir, paths.report_dir, paths.faces_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv_report(results: list[ValidationResult], output_path: Path) -> None:
    """Write per-image validation results as CSV.

    Columns: image_path, prompt_id, face_count, status, failure_reason, similarity
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "prompt_id",
        "face_count",
        "status",
        "failure_reason",
        "similarity",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "image_path": r.image_path,
                    "prompt_id": r.prompt_id,
                    "face_count": r.face_count,
                    "status": r.status,
                    "failure_reason": r.failure_reason or "",
                    "similarity": "" if r.similarity is None else r.similarity,
                }
            )


def write_json_summary(summary: ValidationSummary, output_path: Path) -> None:
    """Write aggregated statistics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(mode="json")
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fmt_float(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.6g}"


def write_markdown_report(
    results: list[ValidationResult],
    summary: ValidationSummary,
    output_path: Path,
    run_name: str = "",
) -> None:
    """Write human-readable Markdown report.

    Content:
    - Header with run name
    - Summary stats table (total, passed, failed, mean/min/max/std similarity)
    - Failure reason breakdown (if any failures)
    - Per-image results table
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Validation report",
        "",
    ]
    if run_name:
        lines.extend([f"**Run:** `{run_name}`", ""])
    lines.extend(
        [
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total images | {summary.total} |",
            f"| Passed (kept) | {summary.passed} |",
            f"| Failed | {summary.failed} |",
            f"| Mean similarity | {_fmt_float(summary.mean_similarity)} |",
            f"| Min similarity | {_fmt_float(summary.min_similarity)} |",
            f"| Max similarity | {_fmt_float(summary.max_similarity)} |",
            f"| Std similarity | {_fmt_float(summary.std_similarity)} |",
            "",
        ]
    )

    if summary.failure_reasons:
        lines.extend(
            [
                "## Failure reason breakdown",
                "",
                "| Reason | Count |",
                "| --- | --- |",
            ]
        )
        for reason, count in sorted(summary.failure_reasons.items(), key=lambda x: (-x[1], x[0])):
            safe = reason.replace("|", "\\|")
            lines.append(f"| {safe} | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Per-image results",
            "",
            "| Image | Prompt | Faces | Status | Failure reason | Similarity |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in results:
        sim = "—" if r.similarity is None else _fmt_float(r.similarity)
        reason = (r.failure_reason or "").replace("|", "\\|")
        lines.append(
            f"| `{r.image_path}` | `{r.prompt_id}` | {r.face_count} | `{r.status}` | {reason} | {sim} |"
        )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_all_reports(
    results: list[ValidationResult],
    summary: ValidationSummary,
    report_dir: Path,
    run_name: str = "",
) -> dict[str, Path]:
    """Write all three report types. Returns dict mapping report type to file path.

    Returns: {"csv": path, "json": path, "markdown": path}
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "csv": report_dir / "validation_results.csv",
        "json": report_dir / "validation_summary.json",
        "markdown": report_dir / "validation_report.md",
    }
    write_csv_report(results, out["csv"])
    write_json_summary(summary, out["json"])
    write_markdown_report(results, summary, out["markdown"], run_name=run_name)
    return out
