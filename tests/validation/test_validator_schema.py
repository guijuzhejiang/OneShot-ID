"""Tests for validation result schema, cosine similarity, and summarizers."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.face_selection import FaceInfo
from src.validation.face_analyzer import AnalysisResult
from src.validation.result_schema import ValidationResult, ValidationSummary
from src.validation.validator import (
    cosine_similarity,
    summarize_results,
    summarize_scores,
    validate_single,
)


def _unit_embedding(dim: int, index: int) -> np.ndarray:
    e = np.zeros(dim, dtype=np.float32)
    e[index] = 1.0
    return e


def test_cosine_similarity_identical() -> None:
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_validate_single_no_face() -> None:
    ref = _unit_embedding(512, 0)
    analysis = AnalysisResult(None, 0, "no_face", "No face detected")
    r = validate_single(ref, analysis, "/tmp/a.png", "p1", 0.5)
    assert r.status == "failed_no_face"
    assert r.similarity is None
    assert r.face_count == 0


def test_validate_single_ok_above_threshold() -> None:
    ref = _unit_embedding(512, 0)
    gen = _unit_embedding(512, 0)
    fi = FaceInfo(
        bbox=np.zeros(4, dtype=np.float32),
        embedding=gen,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=0.99,
        bbox_area=1.0,
    )
    analysis = AnalysisResult(fi, 1, "ok", None)
    r = validate_single(ref, analysis, "/tmp/b.png", "p2", 0.5)
    assert r.status == "passed"
    assert r.similarity == pytest.approx(1.0)
    assert r.failure_reason is None


def test_validate_single_ok_below_threshold() -> None:
    ref = _unit_embedding(512, 0)
    gen = _unit_embedding(512, 1)
    fi = FaceInfo(
        bbox=np.zeros(4, dtype=np.float32),
        embedding=gen,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=0.99,
        bbox_area=1.0,
    )
    analysis = AnalysisResult(fi, 1, "ok", None)
    r = validate_single(ref, analysis, "/tmp/c.png", "p3", 0.5)
    assert r.status == "failed_low_similarity"
    assert r.similarity == pytest.approx(0.0)


def test_validate_single_multi_face_above_threshold() -> None:
    ref = _unit_embedding(512, 0)
    gen = _unit_embedding(512, 0)
    fi = FaceInfo(
        bbox=np.zeros(4, dtype=np.float32),
        embedding=gen,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=0.99,
        bbox_area=1.0,
    )
    analysis = AnalysisResult(fi, 3, "multi_face", "Multiple faces detected (3), using largest")
    r = validate_single(ref, analysis, "/tmp/d.png", "p4", 0.5)
    assert r.status == "passed"
    assert r.similarity == pytest.approx(1.0)


def test_validate_single_multi_face_below_threshold() -> None:
    ref = _unit_embedding(512, 0)
    gen = _unit_embedding(512, 1)
    fi = FaceInfo(
        bbox=np.zeros(4, dtype=np.float32),
        embedding=gen,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        det_score=0.99,
        bbox_area=1.0,
    )
    analysis = AnalysisResult(fi, 2, "multi_face", "Multiple faces detected (2), using largest")
    r = validate_single(ref, analysis, "/tmp/e.png", "p5", 0.5)
    assert r.status == "failed_multi_face_low_similarity"
    assert r.similarity == pytest.approx(0.0)


def test_summarize_scores_three_values() -> None:
    s = summarize_scores([0.50, 0.60, 0.70])
    assert s["mean"] == pytest.approx(0.6)
    assert s["min"] == pytest.approx(0.5)
    assert s["max"] == pytest.approx(0.7)
    assert "std" in s


def test_summarize_scores_empty() -> None:
    assert summarize_scores([]) == {}


def test_summarize_results_aggregates() -> None:
    results = [
        ValidationResult(
            image_path="a.png",
            prompt_id="1",
            face_count=1,
            status="passed",
            similarity=0.9,
        ),
        ValidationResult(
            image_path="b.png",
            prompt_id="2",
            face_count=0,
            status="failed_no_face",
            failure_reason="No face detected",
            similarity=None,
        ),
        ValidationResult(
            image_path="c.png",
            prompt_id="3",
            face_count=1,
            status="failed_low_similarity",
            failure_reason="Similarity 0.1000 below threshold 0.5",
            similarity=0.1,
        ),
    ]
    summary = summarize_results(results)
    assert isinstance(summary, ValidationSummary)
    assert summary.total == 3
    assert summary.passed == 1
    assert summary.failed == 2
    assert summary.mean_similarity == pytest.approx((0.9 + 0.1) / 2)
    assert summary.min_similarity == pytest.approx(0.1)
    assert summary.max_similarity == pytest.approx(0.9)
    assert summary.failure_reasons.get("No face detected") == 1
    assert summary.failure_reasons.get("Similarity 0.1000 below threshold 0.5") == 1
