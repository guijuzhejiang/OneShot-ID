"""Pydantic models for per-image validation and batch summaries."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of validating one generated image against the reference."""

    image_path: str
    prompt_id: str
    face_count: int
    status: str
    # "passed" | "failed_no_face" | "failed_low_similarity" | "failed_multi_face_low_similarity"
    failure_reason: Optional[str] = None
    similarity: Optional[float] = None  # None if no face detected


class ValidationSummary(BaseModel):
    """Aggregate statistics for a batch of validation results."""

    total: int
    passed: int
    failed: int
    mean_similarity: Optional[float] = None
    min_similarity: Optional[float] = None
    max_similarity: Optional[float] = None
    std_similarity: Optional[float] = None
    failure_reasons: dict[str, int] = Field(default_factory=dict)
