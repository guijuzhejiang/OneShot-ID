"""Face similarity and identity validation."""

from src.validation.face_analyzer import AnalysisResult, FaceAnalyzer
from src.validation.result_schema import ValidationResult, ValidationSummary
from src.validation.validator import (
    summarize_results,
    summarize_scores,
    validate_single,
)

__all__ = [
    "AnalysisResult",
    "FaceAnalyzer",
    "ValidationResult",
    "ValidationSummary",
    "summarize_results",
    "summarize_scores",
    "validate_single",
]
