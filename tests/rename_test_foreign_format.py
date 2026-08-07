"""Rename B1→ACME-01, P3→ACME-02 with a completely different txn_id shape.

Stronger version of rename_test.py: the original one kept the TXN- prefix
and only swapped the scenario token. This one drops the TXN- prefix
entirely (`TXN-B1-0001` → `ACME-01-0001`, `TXN-P3-0001` → `ACME-02-0001`)
so `scenario_membership_mask`, `build_mapping` and `validate_selection`
have to survive a form of transaction ID they have never seen. Anything
in the pipeline that quietly assumed 'TXN-{scenario}-{n}' will now break
loudly.

Run once, not part of pytest — copies the ledger, cache and 200 PDFs.
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

# scenario key → new scenario key
SCENARIO_RENAMES = {"B1": "ACME-01", "P3": "ACME-02"}

# txn_id substring rewrites: drop TXN- prefix and use the new key.
TXN_REWRITES = [
    (re.compile(rf"\bTXN-{re.escape(old)}-"), f"{new}-")
    for old, new in SCENARIO_RENAMES.items()
]


def rewrite_txn_ids(text: str) -> str:
    for pattern, replacement in TXN_REWRITES:
        text = pattern.sub(replacement, text)
    return text


def rename_scenario_key(key: str) -> str:
    return SCENARIO_RENAMES.get(key, key)


def rewrite_json_text(text: str) -> str:
    """Rewrite txn_id occurrences inside a JSON blob."""
    return rewrite_txn_ids(text)


def main() -> int:
    fixture = ROOT / "rename_test_foreign_fixture"
    if fixture.exists():
        shutil.rmtree(fixture)
    data = fixture / "data"
    ws = fixture / "workspace"
    data.mkdir(parents=True)
    (ws / "extractions").mkdir(parents=True)
    (ws / "selections").mkdir(parents=True)
    (ws / "vision").mkdir(parents=True)

    # Template: rename scenario keys.
    template = json.loads((SRC_DATA / "submission_template.json").read_text(encoding="utf-8"))
    template["answers"] = {rename_scenario_key(k): v for k, v in template["answers"].items()}
    (data / "submission_template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Ledger: rewrite every TXN-{OLD}-N into {NEW}-N (drops the TXN- prefix).
    ledger_text = (SRC_DATA / "master_ledger_2025.csv").read_text(encoding="utf-8")
    (data / "master_ledger_2025.csv").write_text(rewrite_txn_ids(ledger_text), encoding="utf-8")

    # Ground truth: rename scenario keys AND rewrite txn_id references in
    # the evidence_txn_id field.
    ground_truth = json.loads((SRC_DATA / "ground_truth.json").read_text(encoding="utf-8"))
    ground_truth["scenarios"] = {
        rename_scenario_key(k): v for k, v in ground_truth["scenarios"].items()
    }
    ground_truth = json.loads(rewrite_json_text(json.dumps(ground_truth, ensure_ascii=False)))
    (data / "ground_truth.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Documents: cannot rewrite PDFs easily. They still reference
    # TXN-B1-*/TXN-P3-* inside audit notes. That only matters for stage-5
    # targeted retrieval on missing-amount transactions, and B1/P3 have
    # none in the public set — so the run should still complete cleanly.
    shutil.copytree(SRC_DATA / "documents", data / "documents")

    for src_file in (SRC_WS / "extractions").glob("*.json"):
        sid = src_file.stem
        new_sid = rename_scenario_key(sid)
        text = rewrite_txn_ids(src_file.read_text(encoding="utf-8"))
        (ws / "extractions" / f"{new_sid}.json").write_text(text, encoding="utf-8")

    for src_file in (SRC_WS / "selections").glob("*.json"):
        sid = src_file.stem
        new_sid = rename_scenario_key(sid)
        text = rewrite_txn_ids(src_file.read_text(encoding="utf-8"))
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
        print("stderr:", run.stderr[:800])
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
