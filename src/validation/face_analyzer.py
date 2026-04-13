"""Shared face analysis engine (InsightFace) for generation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from diffusers.utils import load_image

from src.generation.pipeline_stable_diffusion_xl_instantid import draw_kps, resize_img
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
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclass
class AnalysisResult:
    """Result of analyzing a single image for face identity."""

    face_info: Optional[FaceInfo]  # None if no face detected
    face_count: int  # total faces detected in image
    status: str  # "ok" | "no_face" | "multi_face"
    failure_reason: Optional[str]  # human-readable reason if not ok, or None
    face_image: Optional[np.ndarray] = None  # Cropped face image (BGR)


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
            self._app.prepare(ctx_id=self._ctx_id, det_size=(640, 640))

    def analyze(self, image: np.ndarray, save_path: Optional[str | Path] = None) -> AnalysisResult:
        """Detect faces in a BGR image and return standardized result."""
        self._ensure_loaded()
        faces = self._app.get(image)
        n = len(faces)
        if n == 0:
            return AnalysisResult(None, 0, "no_face", "No face detected")
        
        info = select_largest_face(faces)
        face_crop = None
        if info:
            x1, y1, x2, y2 = info.bbox.astype(int)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = image[y1:y2, x1:x2].copy()
            
            if save_path and face_crop.size > 0:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), face_crop)

        if n == 1:
            return AnalysisResult(info, 1, "ok", None, face_image=face_crop)
        return AnalysisResult(
            info,
            n,
            "multi_face",
            f"Multiple faces detected ({n}), using largest",
            face_image=face_crop,
        )

    def analyze_reference(self, image_path: str | Path):
        """Analyze reference face following the InstantID-compatible workflow.

        Loads the image via PIL, resizes to standard dimensions, converts to
        BGR, detects the largest face, and returns the raw embedding plus a
        keypoints image produced by ``draw_kps`` (as required by InstantID's
        ControlNet).

        Returns:
            Tuple of (embedding, kps_image, resized_face_image):
            - embedding: np.ndarray — raw face embedding from InsightFace
            - kps_image: PIL Image with drawn keypoints (ControlNet input)
            - face_image: the resized PIL Image

        Raises:
            ValueError: if no face is detected in the image.
        """
        self._ensure_loaded()

        face_image = load_image(str(image_path))
        face_image = resize_img(face_image)

        bgr = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        faces = self._app.get(bgr)
        if not faces:
            raise ValueError(f"No face detected in reference image: {image_path}")

        largest = sorted(
            faces,
            key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
        )[-1]
        embedding = largest['embedding']
        kps_image = draw_kps(face_image, largest['kps'])

        return embedding, kps_image, face_image

    def analyze_file(self, image_path: str | Path, save_face_to: Optional[str | Path] = None) -> AnalysisResult:
        """Load image from file path and analyze (BGR via cv2)."""

        path = Path(image_path)
        img = cv2.imread(str(path))
        if img is None:
            return AnalysisResult(
                None,
                0,
                "no_face",
                f"Failed to load image: {path}",
            )
        return self.analyze(img, save_path=save_face_to)
