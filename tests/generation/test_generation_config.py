"""Tests for generation config wiring and lightweight InstantIDGenerator behavior."""

from pathlib import Path

import numpy as np

from src.config import load_settings
from src.generation import GenerationRecord, InstantIDGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def test_generator_device_without_weights() -> None:
    settings = load_settings(str(DEFAULT_CONFIG))
    g = InstantIDGenerator(settings, load_weights=False)
    assert g.device == "cuda:1"


def test_generation_record_fields() -> None:
    rec = GenerationRecord(
        image_path="/tmp/out.png",
        prompt_id="front_neutral_portrait",
        seed=42,
        guidance_scale=3.5,
        num_inference_steps=6,
    )
    assert rec.image_path == "/tmp/out.png"
    assert rec.prompt_id == "front_neutral_portrait"
    assert rec.seed == 42
    assert rec.guidance_scale == 3.5
    assert rec.num_inference_steps == 6


def test_draw_keypoints_size() -> None:
    settings = load_settings(str(DEFAULT_CONFIG))
    g = InstantIDGenerator(settings, load_weights=False)
    kps = np.array(
        [[100.0, 100.0], [200.0, 100.0], [150.0, 200.0], [120.0, 280.0], [180.0, 280.0]],
        dtype=np.float32,
    )
    img = g._draw_keypoints(kps, size=(640, 480))
    assert img.size == (640, 480)


def test_release_when_pipe_none() -> None:
    settings = load_settings(str(DEFAULT_CONFIG))
    g = InstantIDGenerator(settings, load_weights=False)
    assert g._pipe is None
    g.release()
