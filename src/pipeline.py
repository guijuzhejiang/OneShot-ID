"""End-to-end orchestration: analyze reference, generate, validate, select, report."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.generation.instantid_generator import InstantIDGenerator
from src.prompts.prompt_bank import PromptSpec, get_prompt_specs, get_retryable_specs
from src.reporting.report_builder import build_run_paths, write_all_reports
from src.validation.face_analyzer import FaceAnalyzer
from src.validation.result_schema import ValidationResult
from src.validation.validator import summarize_results, validate_single

if TYPE_CHECKING:
    from src.config import Settings


def select_kept_images(
    passed_results: list[ValidationResult],
    min_keep: int,
    max_keep: int,
) -> tuple[list[ValidationResult], list[ValidationResult], bool]:
    """Select which images to keep based on similarity scores.

    Returns: (kept, rejected_extras, success)
    - kept: images to keep in kept_dir
    - rejected_extras: passed images beyond max_keep
    - success: True if len(kept) >= min_keep
    """
    if len(passed_results) > max_keep:
        sorted_results = sorted(
            passed_results,
            key=lambda r: (-(r.similarity or 0.0), r.prompt_id),
        )
        kept = sorted_results[:max_keep]
        rejected = sorted_results[max_keep:]
    else:
        kept = list(passed_results)
        rejected = []

    success = len(kept) >= min_keep
    return kept, rejected, success


def _specs_for_round(
    round_idx: int,
    passed_ids: set[str],
    settings: Settings,
) -> list[PromptSpec]:
    if round_idx == 0:
        specs = get_prompt_specs()
    else:
        specs = [s for s in get_retryable_specs() if s.prompt_id not in passed_ids]
    cap = settings.generation.candidates_per_round
    return specs[:cap]


class OneShotIDPipeline:
    def __init__(self, config_path: str = "configs/default.yaml") -> None:
        from src.config import load_settings

        self.settings = load_settings(config_path)

    def run(
        self,
        ref_image_path: str | Path,
        output_dir_name: str | None = None,
        seed: int | None = None,
    ) -> bool:
        settings = self.settings
        if seed is not None:
            settings = settings.model_copy(
                update={
                    "runtime": settings.runtime.model_copy(update={"seed": int(seed)}),
                }
            )

        run_name = output_dir_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        paths = build_run_paths(Path(settings.output.base_dir), run_name)

        analyzer = FaceAnalyzer(settings)
        ref_analysis = analyzer.analyze_file(ref_image_path)
        if ref_analysis.face_info is None:
            print(
                "Error: No usable face in reference image: "
                f"{ref_analysis.failure_reason or 'unknown'}"
            )
            return False

        face_info = ref_analysis.face_info
        ref_embedding = face_info.embedding
        ref_landmarks = face_info.landmarks
        threshold = settings.validation.similarity_threshold

        results_by_prompt: dict[str, ValidationResult] = {}
        generator = InstantIDGenerator(settings)

        try:
            for round_idx in range(settings.generation.max_rounds):
                passed_ids = {pid for pid, r in results_by_prompt.items() if r.status == "passed"}
                if len(passed_ids) >= settings.generation.min_keep:
                    break

                specs = _specs_for_round(round_idx, passed_ids, settings)
                if not specs:
                    break

                round_base_seed = settings.runtime.seed + round_idx * 10_000
                records = generator.generate_batch(
                    ref_embedding,
                    ref_landmarks,
                    specs,
                    round_base_seed,
                    paths.candidates_dir,
                )

                for rec in records:
                    analysis = analyzer.analyze_file(rec.image_path)
                    vr = validate_single(
                        ref_embedding,
                        analysis,
                        rec.image_path,
                        rec.prompt_id,
                        threshold,
                    )
                    results_by_prompt[rec.prompt_id] = vr

                passed_now = sum(1 for r in results_by_prompt.values() if r.status == "passed")
                if passed_now >= settings.generation.min_keep:
                    break
        finally:
            generator.release()

        all_results = sorted(results_by_prompt.values(), key=lambda r: r.prompt_id)
        summary = summarize_results(all_results)
        write_all_reports(all_results, summary, paths.report_dir, run_name=run_name)

        passed_only = [r for r in all_results if r.status == "passed"]
        kept, rejected_extras, success = select_kept_images(
            passed_only,
            settings.generation.min_keep,
            settings.generation.max_keep,
        )

        for p in paths.kept_dir.glob("*"):
            if p.is_file():
                p.unlink()
        for p in paths.rejected_dir.glob("*"):
            if p.is_file():
                p.unlink()

        for r in kept:
            dest = paths.kept_dir / Path(r.image_path).name
            shutil.copy2(r.image_path, dest)

        for r in rejected_extras:
            dest = paths.rejected_dir / Path(r.image_path).name
            shutil.copy2(r.image_path, dest)

        for r in all_results:
            if r.status != "passed":
                dest = paths.rejected_dir / Path(r.image_path).name
                shutil.copy2(r.image_path, dest)

        print(
            f"Run `{run_name}`: passed={summary.passed}/{summary.total}, "
            f"kept={len(kept)}, success_meets_min_keep={success}"
        )
        print(f"Output: {paths.run_dir}")
        return success
