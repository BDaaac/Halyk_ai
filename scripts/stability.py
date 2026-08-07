"""Cross-run per-cell stability analysis for cold submissions.

Reads workspace/run{1,2,3}_submission.json plus ground_truth.json and
prints:
  * per-run score
  * per-cell classification: stable-correct / stable-wrong / coin-flip
  * cost of one retry attempt (soft_warnings inspection)

Run as:  python scripts/stability.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from scorer import score_cell


ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspace"
GT_PATH = ROOT / "agentic-bank-public" / "ground_truth.json"


def _load_runs() -> list[dict]:
    runs = []
    for i in (1, 2, 3):
        path = WS / f"run{i}_submission.json"
        runs.append(json.loads(path.read_text(encoding="utf-8"))["answers"])
    return runs


def _ground_truth() -> dict:
    return json.loads(GT_PATH.read_text(encoding="utf-8")).get("scenarios", {})


def _score(run: dict, ground_truth: dict) -> tuple[float, int]:
    total = 0.0
    correct_status = 0
    n = 0
    for sid, clauses in run.items():
        gt_clauses = (ground_truth.get(sid) or {}).get("covenants", {})
        for cid, cell in clauses.items():
            key = gt_clauses.get(cid)
            if not key:
                continue
            detail = score_cell(sid, cid, cell, key)
            total += detail.total
            if detail.status_pts > 0:
                correct_status += 1
            n += 1
    return total / n, correct_status


def _cell_signature(actual):
    if isinstance(actual, (int, float)):
        return round(float(actual), 2)
    return actual


def main() -> None:
    runs = _load_runs()
    gt = _ground_truth()

    scores: list[float] = []
    corrects: list[int] = []
    for i, run in enumerate(runs, 1):
        mean, correct = _score(run, gt)
        scores.append(mean)
        corrects.append(correct)
        print(f"run {i}: score={mean:.4f}, status_correct={correct}/36")

    print()
    print(f"mean:  {statistics.mean(scores):.4f}")
    print(f"stdev: {statistics.stdev(scores):.4f}")
    print(f"range: {min(scores):.4f} … {max(scores):.4f}  (spread {max(scores) - min(scores):.4f})")

    stable_correct: list[str] = []
    stable_wrong: list[tuple[str, str]] = []
    unstable: list[tuple[str, list, list]] = []
    for sid, clauses in runs[0].items():
        for cid in clauses:
            key = (gt.get(sid) or {}).get("covenants", {}).get(cid) or {}
            statuses = [runs[i].get(sid, {}).get(cid, {}).get("status") for i in range(3)]
            actuals = [_cell_signature(runs[i].get(sid, {}).get(cid, {}).get("actual")) for i in range(3)]
            unique_statuses = set(statuses)
            unique_actuals = set(actuals)

            gt_status = key.get("status")
            gt_actual = key.get("actual")

            if len(unique_statuses) == 1 and len(unique_actuals) == 1:
                if statuses[0] != gt_status:
                    stable_wrong.append((f"{sid} {cid}", f"stable {statuses[0]}, GT {gt_status}"))
                    continue
                actual = actuals[0]
                if isinstance(actual, (int, float)) and isinstance(gt_actual, (int, float)):
                    if gt_actual != 0:
                        rel_err = abs(actual - gt_actual) / abs(gt_actual)
                        if rel_err <= 0.05:
                            stable_correct.append(f"{sid} {cid}")
                        else:
                            stable_wrong.append((f"{sid} {cid}", f"actual off {actual} vs {gt_actual}"))
                    else:
                        (stable_correct if actual == 0 else stable_wrong).append(
                            f"{sid} {cid}" if actual == 0 else (f"{sid} {cid}", f"actual {actual} vs 0")
                        )
                else:
                    stable_correct.append(f"{sid} {cid}")
            else:
                unstable.append((f"{sid} {cid}", statuses, actuals))

    print()
    print(f"stable-correct : {len(stable_correct)} cells")
    print(f"stable-wrong   : {len(stable_wrong)} cells")
    print(f"coin-flip      : {len(unstable)} cells")

    if stable_wrong:
        print()
        print("STABLE-WRONG (Haiku/Sonnet consistently wrong — need extraction fix, not retry):")
        for cell, reason in sorted(stable_wrong):
            print(f"  {cell:8s}  {reason}")

    if unstable:
        print()
        print("COIN-FLIP (varies between runs — where retry could help most):")
        for cell, statuses, actuals in sorted(unstable):
            print(f"  {cell:8s}  status={statuses}  actual={actuals}")


if __name__ == "__main__":
    main()
