import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_validate_help():
    result = subprocess.run(
        [sys.executable, "app/run_validate.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    for flag in ["--reference", "--candidate-dir", "--config", "--report-dir"]:
        assert flag in result.stdout, f"Missing flag: {flag}"
