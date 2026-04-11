"""Tests for the canonical prompt variant bank."""

from src.prompts.prompt_bank import (
    SHARED_NEGATIVE_PROMPT,
    PromptSpec,
    get_prompt_specs,
    get_retryable_specs,
)


def test_prompt_spec_count_and_unique_ids() -> None:
    specs = get_prompt_specs()
    assert len(specs) == 12
    ids = [s.prompt_id for s in specs]
    assert len(ids) == len(set(ids))


def test_all_specs_use_shared_negative_and_lightning_ranges() -> None:
    for s in get_prompt_specs():
        assert s.negative_prompt == SHARED_NEGATIVE_PROMPT
        assert 1.5 <= s.guidance_scale <= 5.0
        assert 5 <= s.num_inference_steps <= 8
        assert s.positive_prompt.strip()
        assert s.tags


def test_retryable_subset_non_empty_and_strict() -> None:
    all_specs = get_prompt_specs()
    retry = get_retryable_specs()
    assert retry
    assert len(retry) < len(all_specs)
    retry_ids = {s.prompt_id for s in retry}
    assert all(s.prompt_id in retry_ids for s in retry)
    for s in retry:
        assert s.retryable is True
    non_retry_ids = {s.prompt_id for s in all_specs if not s.retryable}
    assert non_retry_ids
    assert retry_ids.isdisjoint(non_retry_ids)


def test_non_retryable_are_extreme_variants() -> None:
    """Hard poses should not repeat in regeneration rounds."""
    hard_ids = {"left_45_neutral", "right_45_neutral", "extreme_close_serious"}
    for s in get_prompt_specs():
        if s.prompt_id in hard_ids:
            assert s.retryable is False


def test_prompt_spec_model_roundtrip() -> None:
    s = get_prompt_specs()[0]
    assert isinstance(s, PromptSpec)
    data = s.model_dump()
    again = PromptSpec.model_validate(data)
    assert again == s
