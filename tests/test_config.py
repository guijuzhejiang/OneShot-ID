"""Tests for configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def test_load_default_config() -> None:
    s = load_settings(str(DEFAULT_CONFIG))
    assert isinstance(s, Settings)
    assert s.runtime.device == "cuda:1"
    assert s.runtime.seed == 42
    assert s.generation.min_keep == 8
    assert s.generation.max_keep == 12
    assert s.generation.candidates_per_round == 12
    assert s.generation.max_rounds == 3
    assert s.validation.similarity_threshold == 0.45
    assert s.output.base_dir == "outputs"
    assert "insightface" in s.models.insightface_dir
    assert "InstantID" in s.models.instantid_dir
    assert s.models.sdxl_path.endswith(".safetensors")


def test_generation_max_lt_min_rejected() -> None:
    data = {
        "runtime": {"device": "cpu", "seed": 0},
        "models": {
            "insightface_dir": "/a",
            "instantid_dir": "/b",
            "sdxl_path": "/c.ckpt",
        },
        "generation": {
            "min_keep": 10,
            "max_keep": 5,
            "candidates_per_round": 4,
            "max_rounds": 2,
        },
        "validation": {"similarity_threshold": 0.5},
        "output": {"base_dir": "out"},
    }
    with pytest.raises(ValidationError):
        Settings.model_validate(data)


def test_similarity_out_of_range_rejected() -> None:
    data = {
        "runtime": {"device": "cpu", "seed": 0},
        "models": {
            "insightface_dir": "/a",
            "instantid_dir": "/b",
            "sdxl_path": "/c.ckpt",
        },
        "generation": {
            "min_keep": 1,
            "max_keep": 2,
            "candidates_per_round": 4,
            "max_rounds": 2,
        },
        "validation": {"similarity_threshold": 1.5},
        "output": {"base_dir": "out"},
    }
    with pytest.raises(ValidationError):
        Settings.model_validate(data)
