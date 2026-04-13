"""Cosine similarity, per-image validation, and batch score/summary helpers."""

from __future__ import annotations

import statistics
from typing import Optional

import numpy as np

from src.validation.face_analyzer import AnalysisResult
from src.validation.result_schema import ValidationResult, ValidationSummary


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def validate_single(
    ref_embedding: np.ndarray,
    analysis: AnalysisResult,
    image_path: str,
    prompt_id: str,
    threshold: float,
) -> ValidationResult:
    """Validate a single generated image against reference embedding."""
    if analysis.status == "no_face" or analysis.face_info is None:
        return ValidationResult(
            image_path=image_path,
            prompt_id=prompt_id,
            face_count=analysis.face_count,
            status="failed_no_face",
            failure_reason=analysis.failure_reason or "No face detected",
            similarity=None,
        )

    sim = cosine_similarity(ref_embedding, analysis.face_info.embedding)
    if sim >= threshold:
        return ValidationResult(
            image_path=image_path,
            prompt_id=prompt_id,
            face_count=analysis.face_count,
            status="passed",
            failure_reason=None,
            similarity=sim,
        )

    if analysis.status == "multi_face":
        status = "failed_multi_face_low_similarity"
        reason = (
            f"Multiple faces ({analysis.face_count}); similarity {sim:.4f} below threshold {threshold}"
        )
    else:
        status = "failed_low_similarity"
        reason = f"Similarity {sim:.4f} below threshold {threshold}"

    return ValidationResult(
        image_path=image_path,
        prompt_id=prompt_id,
        face_count=analysis.face_count,
        status=status,
        failure_reason=reason,
        similarity=sim,
    )


def summarize_scores(scores: list[float]) -> dict:
    """Compute mean, min, max, std for similarity scores. Empty input -> {}."""
    if not scores:
        return {}
    xs = [float(x) for x in scores]
    mean = statistics.mean(xs)
    lo = min(xs)
    hi = max(xs)
    if len(xs) == 1:
        std = 0.0
    else:
        std = statistics.pstdev(xs)
    return {"mean": mean, "min": lo, "max": hi, "std": std}


def summarize_results(results: list[ValidationResult]) -> ValidationSummary:
    """Build a ValidationSummary from a list of individual results."""
    total = len(results)
    passed = sum(1 for r in results if r.status == "passed")
    failed = total - passed

    scored = [r.similarity for r in results if r.similarity is not None]
    stats: dict[str, Optional[float]] = {}
    if scored:
        agg = summarize_scores(scored)
        stats = {
            "mean_similarity": agg["mean"],
            "min_similarity": agg["min"],
            "max_similarity": agg["max"],
            "std_similarity": agg["std"],
        }
    else:
        stats = {
            "mean_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
            "std_similarity": None,
        }

    reasons: dict[str, int] = {}
    for r in results:
        if r.status == "passed":
            continue
        key = r.failure_reason or r.status
        reasons[key] = reasons.get(key, 0) + 1

    return ValidationSummary(
        total=total,
        passed=passed,
        failed=failed,
        mean_similarity=stats["mean_similarity"],
        min_similarity=stats["min_similarity"],
        max_similarity=stats["max_similarity"],
        std_similarity=stats["std_similarity"],
        failure_reasons=reasons,
    )
