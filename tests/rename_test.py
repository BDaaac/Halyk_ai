"""Rename B1→T1, P3→T7 in template + ledger + cache; verify score unchanged.

Run once, not part of pytest — it copies 200 PDFs and scores the pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DATA = ROOT / "agentic-bank-public"
SRC_WS = ROOT / "workspace"

RENAMES = {"B1": "T1", "P3": "T7"}


def rename_txn_ids(text: str) -> str:
    for old, new in RENAMES.items():
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])", new, text)
    return text


def main() -> int:
    fixture = ROOT / "rename_test_fixture"
    if fixture.exists():
        shutil.rmtree(fixture)
    data = fixture / "data"
    ws = fixture / "workspace"
    (data).mkdir(parents=True)
    (ws / "extractions").mkdir(parents=True)
    (ws / "selections").mkdir(parents=True)
    (ws / "vision").mkdir(parents=True)

    template = json.loads((SRC_DATA / "submission_template.json").read_text(encoding="utf-8"))
    template["answers"] = {RENAMES.get(k, k): v for k, v in template["answers"].items()}
    (data / "submission_template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ledger_text = (SRC_DATA / "master_ledger_2025.csv").read_text(encoding="utf-8")
    (data / "master_ledger_2025.csv").write_text(rename_txn_ids(ledger_text), encoding="utf-8")

    ground_truth = json.loads((SRC_DATA / "ground_truth.json").read_text(encoding="utf-8"))
    ground_truth["scenarios"] = {
        RENAMES.get(k, k): v for k, v in ground_truth["scenarios"].items()
    }
    ground_truth = json.loads(rename_txn_ids(json.dumps(ground_truth, ensure_ascii=False)))
    (data / "ground_truth.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shutil.copytree(SRC_DATA / "documents", data / "documents")

    for src_file in (SRC_WS / "extractions").glob("*.json"):
        sid = src_file.stem
        new_sid = RENAMES.get(sid, sid)
        text = rename_txn_ids(src_file.read_text(encoding="utf-8"))
        (ws / "extractions" / f"{new_sid}.json").write_text(text, encoding="utf-8")

    for src_file in (SRC_WS / "selections").glob("*.json"):
        sid = src_file.stem
        new_sid = RENAMES.get(sid, sid)
        text = rename_txn_ids(src_file.read_text(encoding="utf-8"))
        (ws / "selections" / f"{new_sid}.json").write_text(text, encoding="utf-8")

    for src_file in (SRC_WS / "vision").glob("*.txt"):
        shutil.copy(src_file, ws / "vision" / src_file.name)

    env = {
        **os.environ,
        "DATA_DIR": str(data),
        "WORKSPACE_DIR": str(ws),
        "ANTHROPIC_API_KEY": "",
    }
    run = subprocess.run(
        [sys.executable, "main.py", "run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    print("=== run output ===")
    print(run.stdout)
    if run.stderr:
        print("stderr:", run.stderr[:500])
    assert run.returncode == 0, run.stderr

    score = subprocess.run(
        [
            sys.executable,
            "main.py",
            "score",
            str(ws / "submission.json"),
            str(data / "ground_truth.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    print("=== score output ===")
    print(score.stdout)
    return score.returncode


if __name__ == "__main__":
    sys.exit(main())
