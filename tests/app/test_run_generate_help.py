import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_generate_help():
    result = subprocess.run(
        [sys.executable, "app/run_generate.py", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode == 0
    for flag in ["--input", "--config", "--run-name", "--seed"]:
        assert flag in result.stdout
