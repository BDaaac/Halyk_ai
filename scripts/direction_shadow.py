"""Shadow analysis: enumerate cells where role_description implies a
direction (outgoing / incoming) that contradicts the sign of a selected
transaction. Reports whether the current cell scored correctly against GT.

No code change is performed. Output feeds the decision whether to promote
direction mismatch into a semantic validation error and route it through
clause-level retry.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OUTGOING_MARKERS = (
    r"в\s+пользу",
    r"платеж\w*\s+.{0,20}\s*в\s+пользу",
    r"paid\s+to",
    r"payment\s+to",
    r"payments?\s+to",
    r"distributed\s+to",
    r"payable\s+to",
    r"\boutgoing\b",
    r"restricted\s+payment",
    r"дивиденд",
    r"исходящ",
)
INCOMING_MARKERS = (
    r"поступлен",
    r"receipts?",
    r"received\s+from",
    r"\bincoming\b",
    r"выручк",
    r"revenue\s+from",
    r"proceeds\s+from",
)


def direction_hint(description: str) -> str | None:
    text = description.lower()
    for pattern in OUTGOING_MARKERS:
        if re.search(pattern, text):
            return "outgoing"
    for pattern in INCOMING_MARKERS:
        if re.search(pattern, text):
            return "incoming"
    return None


def sign(amount: float) -> str:
    if amount > 0:
        return "+"
    if amount < 0:
        return "-"
    return "0"


def main() -> int:
    ledger = pd.read_csv(ROOT / "agentic-bank-public" / "master_ledger_2025.csv")
    ledger_amounts = dict(zip(ledger["txn_id"].astype(str), ledger["amount"]))

    gt = json.loads((ROOT / "agentic-bank-public" / "ground_truth.json").read_text(encoding="utf-8"))["scenarios"]
    submission = json.loads((ROOT / "workspace" / "submission.json").read_text(encoding="utf-8"))

    mismatches: list[dict] = []
    for sid in sorted(gt):
        extraction_path = ROOT / "workspace" / "extractions" / f"{sid}.json"
        selection_path = ROOT / "workspace" / "selections" / f"{sid}.json"
        if not (extraction_path.exists() and selection_path.exists()):
            continue
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))["output"]
        selection = json.loads(selection_path.read_text(encoding="utf-8"))["output"]
        role_descriptions = {}
        for covenant in extraction.get("covenants", []):
            cid = str(covenant.get("clause_id"))
            for role, desc in (covenant.get("role_descriptions") or {}).items():
                role_descriptions[(cid, role)] = desc

        for cid, roles in selection.items():
            if cid == "uncertain" or not isinstance(roles, dict):
                continue
            for role, txn_ids in roles.items():
                if not isinstance(txn_ids, list):
                    continue
                desc = role_descriptions.get((cid, role), "")
                dhint = direction_hint(desc)
                if dhint is None:
                    continue
                for txn_id in txn_ids:
                    amount = ledger_amounts.get(txn_id)
                    if amount is None or pd.isna(amount):
                        continue
                    sig = sign(amount)
                    conflict = (
                        (dhint == "outgoing" and sig == "+")
                        or (dhint == "incoming" and sig == "-")
                    )
                    if not conflict:
                        continue

                    our_cell = submission.get("answers", {}).get(sid, {}).get(cid, {})
                    gt_cell = (gt.get(sid) or {}).get("covenants", {}).get(cid) or {}
                    status_match = our_cell.get("status") == gt_cell.get("status")
                    mismatches.append({
                        "cell": f"{sid} {cid}",
                        "role": role,
                        "direction": dhint,
                        "txn_id": txn_id,
                        "amount": float(amount),
                        "our_status": our_cell.get("status"),
                        "gt_status": gt_cell.get("status"),
                        "status_correct": status_match,
                        "desc": desc[:60],
                    })

    header = (
        f"{'cell':<8s} {'role':<24s} {'dir':<8s} {'txn_id':<15s} "
        f"{'amount':>15s} {'ours':<10s} {'gt':<10s} {'ok?':<4s}  desc"
    )
    print(header)
    print("-" * len(header))
    for row in mismatches:
        print(
            f"{row['cell']:<8s} {row['role']:<24s} {row['direction']:<8s} "
            f"{row['txn_id']:<15s} {row['amount']:>15.2f} "
            f"{str(row['our_status']):<10s} {str(row['gt_status']):<10s} "
            f"{'yes' if row['status_correct'] else 'no':<4s}  {row['desc']}"
        )

    print()
    total = len(mismatches)
    correct_now = sum(1 for r in mismatches if r["status_correct"])
    wrong_now = total - correct_now
    print(f"total direction mismatches: {total}")
    print(f"  in cells currently CORRECT vs GT: {correct_now}")
    print(f"  in cells currently WRONG   vs GT: {wrong_now}")

    cells_currently_correct = sorted({(r["cell"], r["role"]) for r in mismatches if r["status_correct"]})
    if cells_currently_correct:
        print()
        print("cells that would be perturbed (currently GT-correct):")
        for cell, role in cells_currently_correct:
            print(f"  {cell:<8s} role={role}")

    cells_currently_wrong = sorted({(r["cell"], r["role"]) for r in mismatches if not r["status_correct"]})
    if cells_currently_wrong:
        print()
        print("cells where direction mismatch appears AND we are wrong (candidates to fix):")
        for cell, role in cells_currently_wrong:
            print(f"  {cell:<8s} role={role}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
