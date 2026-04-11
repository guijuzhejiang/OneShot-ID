"""Shared face analysis engine (InsightFace) for generation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.face_selection import FaceInfo, select_largest_face


def _parse_ctx_id(device: str) -> int:
    d = device.strip().lower()
    if not d.startswith("cuda"):
        return -1
    if ":" not in d:
        return 0
    try:
        return int(d.split(":", 1)[1])
    except ValueError:
        return 0


def _onnx_providers(device: str) -> list[str]:
    if device.strip().lower().startswith("cuda"):
        return ["CUDAExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclass
class AnalysisResult:
    """Result of analyzing a single image for face identity."""

    face_info: Optional[FaceInfo]  # None if no face detected
    face_count: int  # total faces detected in image
    status: str  # "ok" | "no_face" | "multi_face"
    failure_reason: Optional[str]  # human-readable reason if not ok, or None


class FaceAnalyzer:
    """Shared face analysis engine used by both generation and validation."""

    def __init__(self, settings) -> None:
        self.device = settings.runtime.device
        self._ctx_id = _parse_ctx_id(self.device)
        self._insightface_dir = settings.models.insightface_dir
        self._app = None  # Lazy initialization

    def _ensure_loaded(self) -> None:
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name="antelopev2",
                root=self._insightface_dir,
                providers=_onnx_providers(self.device),
            )
            self._app.prepare(ctx_id=self._ctx_id)

    def analyze(self, image: np.ndarray) -> AnalysisResult:
        """Detect faces in a BGR image and return standardized result."""
        self._ensure_loaded()
        faces = self._app.get(image)
        n = len(faces)
        if n == 0:
            return AnalysisResult(None, 0, "no_face", "No face detected")
        info = select_largest_face(faces)
        if n == 1:
            return AnalysisResult(info, 1, "ok", None)
        return AnalysisResult(
            info,
            n,
            "multi_face",
            f"Multiple faces detected ({n}), using largest",
        )

    def analyze_file(self, image_path: str | Path) -> AnalysisResult:
        """Load image from file path and analyze (BGR via cv2)."""
        import cv2

        path = Path(image_path)
        img = cv2.imread(str(path))
        if img is None:
            return AnalysisResult(
                None,
                0,
                "no_face",
                f"Failed to load image: {path}",
            )
        return self.analyze(img)
