"""InstantID + SDXL generation adapter (diffusers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw
import torch

if TYPE_CHECKING:
    from src.config import Settings
    from src.prompts.prompt_bank import PromptSpec


@dataclass
class GenerationRecord:
    """Metadata for one generated candidate image."""

    image_path: str
    prompt_id: str
    seed: int
    guidance_scale: float
    num_inference_steps: int


class InstantIDGenerator:
    def __init__(self, settings: Settings, load_weights: bool = True) -> None:
        """
        Args:
            settings: Settings object from config
            load_weights: If False, skip loading heavy model weights (for testing)
        """
        self.device = settings.runtime.device
        self._settings = settings
        self._pipe: Any = None
        if load_weights:
            self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load SDXL + ControlNet + IP-Adapter pipeline."""
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        instantid_dir = self._settings.models.instantid_dir
        sdxl_path = self._settings.models.sdxl_path

        controlnet = ControlNetModel.from_pretrained(
            instantid_dir,
            subfolder="ControlNetModel",
            torch_dtype=dtype,
        )
        load_kw: dict[str, Any] = {}
        if dtype == torch.float16:
            load_kw["variant"] = "fp16"
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            sdxl_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            **load_kw,
        )
        pipe.load_ip_adapter_instantid(
            instantid_dir,
            subfolder="",
            weight_name="ip-adapter.bin",
        )
        pipe.set_ip_adapter_scale(0.8)
        pipe.to(self.device)
        self._pipe = pipe

    def _draw_keypoints(
        self,
        landmarks: np.ndarray,
        size: tuple[int, int] = (1024, 1024),
    ) -> Image.Image:
        """Draw facial keypoints as ControlNet condition image."""
        kps = np.asarray(landmarks, dtype=np.float64)
        if kps.shape != (5, 2):
            raise ValueError(f"landmarks must have shape (5, 2), got {kps.shape}")

        img = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]
        radius = 5
        w, h = size
        for (x, y), color in zip(kps, colors):
            xi = float(np.clip(x, 0.0, w - 1.0))
            yi = float(np.clip(y, 0.0, h - 1.0))
            draw.ellipse(
                [xi - radius, yi - radius, xi + radius, yi + radius],
                fill=color,
            )
        return img

    def _ensure_pipeline(self) -> None:
        if self._pipe is None:
            raise RuntimeError(
                "Pipeline is not loaded; construct InstantIDGenerator with load_weights=True "
                "before calling generate_single or generate_batch."
            )

    def generate_single(
        self,
        face_embedding: np.ndarray,
        face_landmarks: np.ndarray,
        prompt_spec: PromptSpec,
        seed: int,
        output_path: Path,
    ) -> GenerationRecord:
        """Generate a single candidate image and save it to disk."""
        self._ensure_pipeline()

        emb = np.asarray(face_embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != 512:
            raise ValueError(f"face_embedding must have 512 elements, got {emb.shape[0]}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        kps_image = self._draw_keypoints(face_landmarks)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        out = self._pipe(
            prompt=prompt_spec.positive_prompt,
            negative_prompt=prompt_spec.negative_prompt,
            image_embeds=emb,
            image=kps_image,
            num_inference_steps=prompt_spec.num_inference_steps,
            guidance_scale=prompt_spec.guidance_scale,
            generator=generator,
        )
        image = out.images[0]
        image.save(output_path)

        return GenerationRecord(
            image_path=str(output_path),
            prompt_id=prompt_spec.prompt_id,
            seed=int(seed),
            guidance_scale=float(prompt_spec.guidance_scale),
            num_inference_steps=int(prompt_spec.num_inference_steps),
        )

    def generate_batch(
        self,
        face_embedding: np.ndarray,
        face_landmarks: np.ndarray,
        prompt_specs: list[PromptSpec],
        base_seed: int,
        output_dir: Path,
    ) -> list[GenerationRecord]:
        """Generate one image per prompt spec with seeds base_seed + index."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records: list[GenerationRecord] = []
        for i, spec in enumerate(prompt_specs):
            path = output_dir / f"{spec.prompt_id}.png"
            records.append(
                self.generate_single(
                    face_embedding,
                    face_landmarks,
                    spec,
                    base_seed + i,
                    path,
                )
            )
        return records

    def release(self) -> None:
        """Release GPU memory by deleting the pipeline."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
