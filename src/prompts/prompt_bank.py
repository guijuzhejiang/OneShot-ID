"""Canonical prompt variants for identity-consistent SDXL + InstantID generation."""

from __future__ import annotations

from pydantic import BaseModel, Field

# 共享负面提示词：强制禁止NSFW内容、多人、畸形等
SHARED_NEGATIVE_PROMPT = (
    # NSFW / 裸露 / 暴露 — 最高权重强制禁止
    "(nsfw, nude, naked, topless, bare chest, bare skin, cleavage, lingerie, underwear, "
    "bikini, swimsuit, sexually suggestive, erotic, provocative, revealing clothing:1.8), "
    # 多人 — 禁止出现第二个人
    "(multiple people, two people, group, crowd, couple, duo, several people, "
    "background person, bystander:1.6), "
    # 畸形 / 解剖错误
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, "
    "extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), "
    "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, "
    "watermark, text, signature, logo"
)

# 共享正面质量词：单人 + 穿着得体 + 高画质
_QUALITY = (
    # 单人：强制只出现一个人
    "(solo, single person, alone, only one person:1.5), "
    # 穿着：确保穿着得体的日常服装
    "fully clothed, wearing casual everyday clothes, "
    # 画质
    "realistic skin texture, natural lighting, sharp focus, "
    "8k uhd, high quality, masterpiece, best quality"
)


class PromptSpec(BaseModel):
    """One pose/expression/camera variant with recommended sampling parameters."""

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
    q = _QUALITY
    return [
        # ── 1. 剧烈怒吼（硬侧光、风吹发丝）, 轻微左转── 测试强烈情绪与动态发丝对身份保持的影响
        PromptSpec(
            prompt_id="dramatic_scream_side_light",
            positive_prompt=(
                f"(slight head turn to the left:1.3), screaming with mouth wide open, intense anger, "
                f"harsh side lighting, outdoor stormy background, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["expression:angry", "lighting:side", "motion:hair_blown"],
            retryable=True,
        ),
        # ── 2. 霓虹街头夸张笑（赛博风、彩色 Rim light）, 右侧脸── 测试极端光色下的人脸一致性
        PromptSpec(
            prompt_id="neon_cyberpunk_exaggerated_laugh",
            positive_prompt=(
                f"(right profile:1.3), exaggerated wide laugh, head thrown back, teeth visible, "
                f"neon-lit alley, colored rim lights, wet pavement reflections, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["style:cyberpunk", "expression:exaggerated_laugh", "lighting:neon"],
            retryable=True,
        ),
        # ── 3. 跳跃动作（空中动感）, 从下往上看── 测试身体大幅位移与面部保持一致性
        PromptSpec(
            prompt_id="action_jump_midair",
            positive_prompt=(
                f"jump, (viewed from below:1.3)', {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["action:jump", "motion:dynamic", "camera:action"],
            retryable=False,
        ),
        # ── 4. 大喊/呼喊（开口极大、面部扭曲）, 从上往下看── 测试张嘴极大时人脸嵌入稳定性
        PromptSpec(
            prompt_id="open_mouth_shout",
            positive_prompt=(
                f"shouting loudly, (viewed from above:1.3), mouth fully open, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["expression:shout", "lighting:spotlight", "distortion:face"],
            retryable=True,
        ),
        # ── 5. 摇头狂笑（头部大幅摆动）,戴眼镜── 测试头部极端角度与关键点变化
        PromptSpec(
            prompt_id="head_shake_wild_laugh",
            positive_prompt=(
                f"laughing wildly , exaggerated facial creases, wearing glasses, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["expression:wild_laugh", "motion:head_shake", "lighting:strobe"],
            retryable=True,
        ),
        # ── 6. 痛哭流涕── 测试液体遮挡与表情变化对识别的影响
        PromptSpec(
            prompt_id="crying",
            positive_prompt=(
                f"crying, visible tears, tearful eyes, sad and emotional expression, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["expression:crying", "camera:close", "effect:tears"],
            retryable=False,
        ),
        # ── 7. 极端惊讶── 测试眼部形变对嵌入的影响
        PromptSpec(
            prompt_id="extreme_surprised_wide_eyes",
            positive_prompt=(
                f"with extremely surprised expression, mouth agape, "
                f"bright frontal lighting, cluttered urban background, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["expression:extreme_surprised", "lighting:frontal"],
            retryable=True,
        ),
        # ── 8. 戏剧舞台表演（夸张化妆与表情）, 戴帽子── 测试化妆/面饰对身份保持的影响
        PromptSpec(
            prompt_id="theatrical_makeup_exaggerated",
            positive_prompt=(
                f"in theatrical makeup, wearing a hat, exaggerated facial features (heavy blush), dramatic shadows, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["style:theatrical", "expression:exaggerated", "makeup:heavy"],
            retryable=False,
        ),
        # ── 9. 雨中狂奔（湿发、衣物贴身）── 测试湿润光泽与遮挡对识别的影响
        PromptSpec(
            prompt_id="running_in_rain_expressive",
            positive_prompt=(
                f"running in heavy rain, hair plastered, clothes clinging, intense determined expression, "
                f"dramatic wet reflections, motion blur, urban night scene, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["action:running", "weather:rain", "expression:determined"],
            retryable=False,
        ),
        # ── 10. 高举双手胜利（夸张肢体动作）── 测试全身大幅动作情况下的人脸稳定性
        PromptSpec(
            prompt_id="victory_arms_raised",
            positive_prompt=(
                f"with both arms raised high in victory, chest forward, triumphant shout, "
                f"wide open mouth, uplifted face toward sky, strong backlight rim, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["gesture:arms_raised", "expression:triumphant", "lighting:backlight"],
            retryable=True,
        ),
        # ── 11. 抽象面部扭曲（艺术化夸张）── 测试在明显艺术化变形下是否还能识别同一人物
        PromptSpec(
            prompt_id="artistic_face_distortion",
            positive_prompt=(
                f"with artistically exaggerated facial distortion (stretched mouth) but clearly the same subject, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["style:artistic", "effect:distortion", "expression:warped"],
            retryable=False,
        ),
        # ── 12. 恐惧和惊慌的表情
        PromptSpec(
            prompt_id="fearful_expression",
            positive_prompt=(
                f"a frightened person, terrified expression, dilated pupils, raised eyebrows, tense facial muscles, "
                f"slightly open mouth, {q}"
            ),
            negative_prompt=SHARED_NEGATIVE_PROMPT,
            guidance_scale=5,
            num_inference_steps=20,
            tags=["camera:action", "expression:fearful", "detail:micro"],
            retryable=False,
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
