"""Tests for reporting path layout and report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.reporting.report_builder import (
    build_run_paths,
    write_all_reports,
    write_csv_report,
    write_json_summary,
    write_markdown_report,
)
from src.validation.result_schema import ValidationResult, ValidationSummary

mock_results = [
    ValidationResult(
        image_path="img1.png",
        prompt_id="front_neutral",
        face_count=1,
        status="passed",
        similarity=0.72,
    ),
    ValidationResult(
        image_path="img2.png",
        prompt_id="left_15_smile",
        face_count=0,
        status="failed_no_face",
        failure_reason="No face detected",
    ),
    ValidationResult(
        image_path="img3.png",
        prompt_id="right_45_neutral",
        face_count=2,
        status="failed_multi_face_low_similarity",
        failure_reason="Multiple faces, low similarity",
        similarity=0.38,
    ),
]
mock_summary = ValidationSummary(
    total=3,
    passed=1,
    failed=2,
    mean_similarity=0.55,
    min_similarity=0.38,
    max_similarity=0.72,
    std_similarity=0.17,
    failure_reasons={
        "No face detected": 1,
        "Multiple faces, low similarity": 1,
    },
)


def test_build_run_paths_creates_directories(tmp_path: Path) -> None:
    base = tmp_path / "out"
    paths = build_run_paths(base, "run_a")
    assert paths.run_dir == base / "runs" / "run_a"
    assert paths.candidates_dir == paths.run_dir / "candidates"
    assert paths.kept_dir == paths.run_dir / "kept"
    assert paths.rejected_dir == paths.run_dir / "rejected"
    assert paths.report_dir == paths.run_dir / "reports"
    for d in (
        paths.candidates_dir,
        paths.kept_dir,
        paths.rejected_dir,
        paths.report_dir,
    ):
        assert d.is_dir()


def test_build_run_paths_report_dir_posix_relative() -> None:
    assert (
        build_run_paths(Path("outputs"), "smoke").report_dir.as_posix()
        == "outputs/runs/smoke/reports"
    )


def test_write_csv_report_headers_and_rows(tmp_path: Path) -> None:
    p = tmp_path / "validation_results.csv"
    write_csv_report(mock_results, p)
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["image_path"] == "img1.png"
    assert rows[0]["prompt_id"] == "front_neutral"
    assert rows[0]["face_count"] == "1"
    assert rows[0]["status"] == "passed"
    assert rows[0]["failure_reason"] == ""
    assert rows[0]["similarity"] == "0.72"
    assert rows[1]["failure_reason"] == "No face detected"
    assert rows[1]["similarity"] == ""
    assert rows[2]["similarity"] == "0.38"


def test_write_json_summary_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "validation_summary.json"
    write_json_summary(mock_summary, p)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    again = ValidationSummary.model_validate(loaded)
    assert again == mock_summary


def test_write_markdown_report_contains_tables(tmp_path: Path) -> None:
    p = tmp_path / "validation_report.md"
    write_markdown_report(mock_results, mock_summary, p, run_name="smoke")
    text = p.read_text(encoding="utf-8")
    assert "# Validation report" in text
    assert "smoke" in text
    assert "## Summary" in text
    assert "| Total images | 3 |" in text
    assert "| Passed (kept) | 1 |" in text
    assert "## Failure reason breakdown" in text
    assert "No face detected" in text
    assert "## Per-image results" in text
    assert "img1.png" in text
    assert "img2.png" in text


def test_write_all_reports_creates_three_files(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    paths = write_all_reports(mock_results, mock_summary, report_dir, run_name="r1")
    assert set(paths.keys()) == {"csv", "json", "markdown"}
    assert paths["csv"].name == "validation_results.csv"
    assert paths["json"].name == "validation_summary.json"
    assert paths["markdown"].name == "validation_report.md"
    for f in paths.values():
        assert f.is_file()
