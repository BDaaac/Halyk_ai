import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_score_reports_score_for_ready_files():
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "score",
            "tests/fixtures/submission_template.json",
            "tests/fixtures/ground_truth.json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Средний балл по ячейке" in result.stdout


def test_eval_reports_unimplemented_pipeline():
    result = subprocess.run(
        [sys.executable, "main.py", "eval"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "eval requires stage 0" in result.stderr


def test_run_writes_baseline_before_later_stages(tmp_path: Path):
    # Empty-string ANTHROPIC_API_KEY in the child's env. config.load_dotenv
    # sees the key already present and skips it — so the .env file cannot
    # inject the developer's real key here, and the pipeline's missing-key
    # guard leaves every cell at the stage-0 baseline instead of dialing
    # the API 24 times over the wire.
    env = {**os.environ, "WORKSPACE_DIR": str(tmp_path), "ANTHROPIC_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, "main.py", "run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0
    assert (tmp_path / "submission.json").exists()
