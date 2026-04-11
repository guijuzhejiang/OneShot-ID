"""YAML configuration loading and validation (Pydantic v2)."""

from __future__ import annotations

from pathlib import Path
import yaml
from pydantic import BaseModel, Field, model_validator


class RuntimeSettings(BaseModel):
    device: str = Field(..., min_length=1, description="Torch device string, e.g. cuda:0 or cpu")
    seed: int = Field(..., ge=0, description="Base random seed")


class ModelPaths(BaseModel):
    insightface_dir: str = Field(..., min_length=1)
    instantid_dir: str = Field(..., min_length=1)
    sdxl_path: str = Field(..., min_length=1)


class GenerationSettings(BaseModel):
    min_keep: int = Field(..., ge=1, description="Minimum number of images to keep")
    max_keep: int = Field(..., ge=1, description="Maximum number of images to keep")
    candidates_per_round: int = Field(..., ge=1, description="Candidates generated per round")
    max_rounds: int = Field(..., ge=1, description="Maximum regeneration rounds")

    @model_validator(mode="after")
    def max_ge_min(self) -> GenerationSettings:
        if self.max_keep < self.min_keep:
            raise ValueError("max_keep must be greater than or equal to min_keep")
        return self


class ValidationSettings(BaseModel):
    similarity_threshold: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity gate")


class OutputSettings(BaseModel):
    base_dir: str = Field(..., min_length=1, description="Root directory for run outputs")


class Settings(BaseModel):
    runtime: RuntimeSettings
    models: ModelPaths
    generation: GenerationSettings
    validation: ValidationSettings
    output: OutputSettings


def load_settings(config_path: str) -> Settings:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None or not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return Settings.model_validate(raw)
