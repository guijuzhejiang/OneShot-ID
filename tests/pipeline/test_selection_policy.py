from src.pipeline import select_kept_images
from src.validation.result_schema import ValidationResult


def _make_result(prompt_id: str, similarity: float, status: str = "passed") -> ValidationResult:
    return ValidationResult(
        image_path=f"{prompt_id}.png",
        prompt_id=prompt_id,
        face_count=1,
        status=status,
        similarity=similarity,
    )


def test_keep_all_when_within_range():
    results = [_make_result(f"p{i}", 0.5 + i * 0.02) for i in range(10)]
    kept, rejected, success = select_kept_images(results, min_keep=8, max_keep=12)
    assert len(kept) == 10
    assert len(rejected) == 0
    assert success is True


def test_trim_to_max_keep():
    results = [_make_result(f"p{i}", 0.5 + i * 0.01) for i in range(15)]
    kept, rejected, success = select_kept_images(results, min_keep=8, max_keep=12)
    assert len(kept) == 12
    assert len(rejected) == 3
    assert success is True
    assert kept[0].similarity >= kept[-1].similarity


def test_insufficient_returns_false():
    results = [_make_result(f"p{i}", 0.6) for i in range(5)]
    kept, rejected, success = select_kept_images(results, min_keep=8, max_keep=12)
    assert len(kept) == 5
    assert success is False


def test_empty_returns_false():
    kept, rejected, success = select_kept_images([], min_keep=8, max_keep=12)
    assert len(kept) == 0
    assert success is False


def test_exact_min_keep():
    results = [_make_result(f"p{i}", 0.5) for i in range(8)]
    kept, rejected, success = select_kept_images(results, min_keep=8, max_keep=12)
    assert len(kept) == 8
    assert success is True
