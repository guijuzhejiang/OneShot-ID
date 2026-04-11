"""Pure face selection utilities (no InsightFace import)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class FaceInfo:
    """Standardized face detection result."""

    bbox: np.ndarray  # shape (4,) — x1, y1, x2, y2
    embedding: np.ndarray  # shape (512,) — L2-normalized face embedding
    landmarks: np.ndarray  # shape (5, 2) — 5-point facial landmarks (for ControlNet)
    det_score: float  # detection confidence
    bbox_area: float  # computed area of bounding box


def _face_to_faceinfo(face: Any) -> FaceInfo:
    bbox = np.asarray(face.bbox, dtype=np.float64).reshape(-1)
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    area = (x2 - x1) * (y2 - y1)
    emb = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(emb))
    if n > 0:
        emb = emb / n
    kps = np.asarray(face.kps, dtype=np.float32).reshape(5, 2)
    return FaceInfo(
        bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32),
        embedding=emb,
        landmarks=kps,
        det_score=float(face.det_score),
        bbox_area=float(area),
    )


def select_largest_face(faces: list) -> Optional[FaceInfo]:
    """Select the face with largest bounding box area from InsightFace detection results.

    Args:
        faces: list of insightface Face objects (from app.get())

    Returns:
        FaceInfo for the largest face, or None if faces is empty.
    """
    if not faces:
        return None
    best: Any = None
    best_area = -1.0
    for face in faces:
        bbox = np.asarray(face.bbox, dtype=np.float64).reshape(-1)
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = face
    return _face_to_faceinfo(best)
