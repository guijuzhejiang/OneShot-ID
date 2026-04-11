"""Canonical prompt variants for identity-consistent SDXL + InstantID generation."""

from __future__ import annotations

from pydantic import BaseModel, Field

SHARED_NEGATIVE_PROMPT = (
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, "
    "extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), "
    "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, "
    "watermark, text"
)


class PromptSpec(BaseModel):
    """One pose/expression/camera variant with recommended Lightning-friendly sampling."""

    prompt_id: str = Field(..., min_length=1, description="Stable id, e.g. front_neutral")
    positive_prompt: str = Field(..., min_length=1)
    negative_prompt: str = Field(..., min_length=1)
    guidance_scale: float = Field(..., ge=0.0, description="Recommended CFG for Lightning SDXL")
    num_inference_steps: int = Field(..., ge=1, le=32)
    tags: list[str] = Field(default_factory=list)
    retryable: bool = Field(
        ...,
        description="If False, skip this id in regeneration rounds to avoid repeating hard poses",
    )


def _specs() -> list[PromptSpec]:
    q = (
        "single person, realistic skin texture, natural lighting, sharp focus, "
        "8k uhd, dslr, high quality, masterpiece, best quality"
    )
    return [
        PromptSpec(
            prompt_id="front_neutral_portrait",
            positive_prompt=(
                f"portrait photo of one person facing the camera, neutral relaxed expression, "
                f"head straight, shoulders visible, standard portrait framing, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=3.5,
            num_inference_steps=6,
            tags=["pose:front", "expression:neutral", "camera:portrait"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="left_15_subtle_smile",
            positive_prompt=(
                f"portrait of one person turned slightly to their left, about 15 degrees, "
                f"subtle gentle smile, eyes toward camera, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=4.0,
            num_inference_steps=6,
            tags=["pose:left_slight", "expression:subtle_smile", "camera:portrait"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="right_15_subtle_smile",
            positive_prompt=(
                f"portrait of one person turned slightly to their right, about 15 degrees, "
                f"subtle gentle smile, eyes toward camera, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=4.0,
            num_inference_steps=6,
            tags=["pose:right_slight", "expression:subtle_smile", "camera:portrait"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="left_45_neutral",
            positive_prompt=(
                f"portrait of one person in left profile, head turned about 45 degrees from camera, "
                f"neutral expression, clear jawline and ear visible, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=2.5,
            num_inference_steps=7,
            tags=["pose:left_profile", "expression:neutral", "camera:portrait"],
            retryable=False,
        ),
        PromptSpec(
            prompt_id="right_45_neutral",
            positive_prompt=(
                f"portrait of one person in right profile, head turned about 45 degrees from camera, "
                f"neutral expression, clear jawline and ear visible, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=2.5,
            num_inference_steps=7,
            tags=["pose:right_profile", "expression:neutral", "camera:portrait"],
            retryable=False,
        ),
        PromptSpec(
            prompt_id="head_up_confident",
            positive_prompt=(
                f"portrait of one person facing camera, chin slightly raised, head tilted up a little, "
                f"confident composed expression, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=3.5,
            num_inference_steps=6,
            tags=["pose:front", "expression:confident", "camera:portrait", "head:tilt_up"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="head_down_contemplative",
            positive_prompt=(
                f"portrait of one person facing camera, head tilted slightly downward, "
                f"contemplative thoughtful expression, eyes may look slightly down, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=3.5,
            num_inference_steps=6,
            tags=["pose:front", "expression:contemplative", "camera:portrait", "head:tilt_down"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="extreme_close_serious",
            positive_prompt=(
                f"extreme close-up portrait of one person, face fills most of frame, "
                f"serious intense expression, sharp eyes, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=2.0,
            num_inference_steps=8,
            tags=["pose:front", "expression:serious", "camera:extreme_close"],
            retryable=False,
        ),
        PromptSpec(
            prompt_id="medium_shot_warm_smile",
            positive_prompt=(
                f"medium shot portrait of one person from chest up, facing camera, "
                f"warm genuine smile, relaxed posture, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=4.5,
            num_inference_steps=5,
            tags=["pose:front", "expression:warm_smile", "camera:medium"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="front_closed_eyes_peaceful",
            positive_prompt=(
                f"portrait of one person facing camera, eyes gently closed, peaceful serene expression, "
                f"soft natural light, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=3.0,
            num_inference_steps=6,
            tags=["pose:front", "expression:peaceful", "camera:portrait", "eyes:closed"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="three_quarter_left_laughing",
            positive_prompt=(
                f"three-quarter view portrait, person angled to show more left cheek, "
                f"open joyful laugh, teeth may show, candid energy, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=4.0,
            num_inference_steps=6,
            tags=["pose:three_quarter_left", "expression:laughing", "camera:portrait"],
            retryable=True,
        ),
        PromptSpec(
            prompt_id="three_quarter_right_pensive",
            positive_prompt=(
                f"three-quarter view portrait, person angled to show more right cheek, "
                f"pensive introspective look, soft gaze, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=3.5,
            num_inference_steps=6,
            tags=["pose:three_quarter_right", "expression:pensive", "camera:portrait"],
            retryable=True,
        ),
    ]


_SPECS_CACHE: list[PromptSpec] | None = None


def get_prompt_specs() -> list[PromptSpec]:
    """Return the full list of 12 prompt specifications."""
    global _SPECS_CACHE
    if _SPECS_CACHE is None:
        _SPECS_CACHE = _specs()
    return list(_SPECS_CACHE)


def get_retryable_specs() -> list[PromptSpec]:
    """Return only prompt specs that are retryable for regeneration rounds."""
    return [s for s in get_prompt_specs() if s.retryable]
