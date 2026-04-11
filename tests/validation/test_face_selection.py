"""Unit tests for face selection and FaceAnalyzer wiring (no GPU / model load)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.face_selection import FaceInfo, select_largest_face
from src.validation.face_analyzer import FaceAnalyzer


def _mock_face(
    bbox: tuple[float, float, float, float],
    *,
    det_score: float = 0.99,
    emb_scale: float = 1.0,
) -> SimpleNamespace:
    x1, y1, x2, y2 = bbox
    # Non-normalized raw vector; select_largest_face should L2-normalize
    raw = np.arange(512, dtype=np.float32) * emb_scale + 0.1
    kps = np.zeros((5, 2), dtype=np.float32)
    return SimpleNamespace(
        bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32),
        embedding=raw,
        kps=kps,
        det_score=det_score,
    )


def test_select_largest_face_empty() -> None:
    assert select_largest_face([]) is None


def test_select_largest_face_single() -> None:
    f = _mock_face((10.0, 10.0, 50.0, 90.0))
    info = select_largest_face([f])
    assert isinstance(info, FaceInfo)
    np.testing.assert_allclose(info.bbox, [10.0, 10.0, 50.0, 90.0], rtol=0, atol=1e-5)
    assert info.bbox_area == pytest.approx(40.0 * 80.0)
    assert info.det_score == pytest.approx(0.99)
    assert info.landmarks.shape == (5, 2)
    assert info.embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(info.embedding), 1.0, atol=1e-5)


def test_select_largest_face_picks_largest_area() -> None:
    small = _mock_face((0.0, 0.0, 10.0, 10.0), emb_scale=1.0)
    large = _mock_face((0.0, 0.0, 20.0, 20.0), emb_scale=2.0)
    info = select_largest_face([small, large])
    assert info is not None
    assert info.bbox_area == pytest.approx(400.0)
    np.testing.assert_allclose(info.bbox, [0.0, 0.0, 20.0, 20.0], rtol=0, atol=1e-5)
    # embedding should come from the large face (different scale -> different normalized direction check)
    raw_large = np.asarray(large.embedding, dtype=np.float32).reshape(-1)
    expected = raw_large / np.linalg.norm(raw_large)
    np.testing.assert_allclose(info.embedding, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("device", "expected_ctx"),
    [
        ("cuda:1", 1),
        ("cuda:0", 0),
        ("CUDA:2", 2),
        ("cpu", -1),
        ("CPU", -1),
    ],
)
def test_face_analyzer_ctx_id(device: str, expected_ctx: int) -> None:
    settings = SimpleNamespace(
        runtime=SimpleNamespace(device=device),
        models=SimpleNamespace(insightface_dir="/tmp/insightface-models"),
    )
    a = FaceAnalyzer(settings)
    assert a.device == device
    assert a._ctx_id == expected_ctx


def test_face_analyzer_default_cuda_index() -> None:
    settings = SimpleNamespace(
        runtime=SimpleNamespace(device="cuda"),
        models=SimpleNamespace(insightface_dir="/x"),
    )
    assert FaceAnalyzer(settings)._ctx_id == 0
