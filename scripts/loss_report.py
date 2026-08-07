"""Per-cell weighted loss decomposition against ground_truth.

Splits the standard scorer's per-cell 1.0 budget into three drains:

  status_lost    = 1.00 * (status_wrong)               # wrong status zeroes the cell
  actual_lost    = actual_weight * (1 - scale)         # only counted when status right
  evidence_lost  = 0.20 if gt_evidence not-null and wrong

where actual_weight is 0.50 when gt_evidence is null (evidence budget rolls into
actual scaling) and 0.30 otherwise, mirroring scorer.score_cell exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scorer import actual_scale  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def loss_row(sid: str, cid: str, cell: dict | None, key: dict) -> dict:
    gt_status = key["status"]
    gt_actual = key["actual"]
    gt_ev = key["evidence_txn_id"]

    ours_status = None
    ours_actual = None
    ours_ev = None
    if cell is not None:
        ours_status = cell.get("status")
        ours_actual = cell.get("actual")
        ours_ev = cell.get("evidence_txn_id")

    status_wrong = ours_status != gt_status
    status_lost = 1.0 if status_wrong else 0.0

    if status_wrong:
        actual_lost = 0.0
        evidence_lost = 0.0
        scale = 0.0
    else:
        scale = actual_scale(ours_actual, gt_actual)
        actual_weight = 0.50 if gt_ev is None else 0.30
        actual_lost = actual_weight * (1.0 - scale)
        if gt_ev is None:
            evidence_lost = 0.0
        else:
            evidence_lost = 0.0 if ours_ev == gt_ev else 0.20

    known_loss = status_lost + actual_lost + evidence_lost

    if isinstance(ours_actual, (int, float)) and gt_actual not in (None, 0):
        rel_err = abs(ours_actual - gt_actual) / abs(gt_actual)
    else:
        rel_err = None

    if status_wrong:
        cause = f"status {ours_status!r} != {gt_status!r}"
    elif evidence_lost > 0:
        cause = f"evidence {ours_ev!r} != {gt_ev!r}"
    elif actual_lost > 0:
        cause = f"actual off by {(1 - scale) * 100:.1f}% of tolerance"
    else:
        cause = "perfect"

    return {
        "cell": f"{sid} {cid}",
        "gt_status": gt_status,
        "ours_status": ours_status,
        "gt_actual": gt_actual,
        "ours_actual": ours_actual,
        "rel_error": rel_err,
        "status_lost": status_lost,
        "actual_lost": actual_lost,
        "evidence_lost": evidence_lost,
        "known_loss": known_loss,
        "cause": cause,
    }


def main() -> int:
    submission = load_json(ROOT / "workspace" / "submission.json")
    ground_truth = load_json(ROOT / "agentic-bank-public" / "ground_truth.json")

    rows: list[dict] = []
    for sid, sc in ground_truth["scenarios"].items():
        for cid, key in sc["covenants"].items():
            cell = submission.get("answers", {}).get(sid, {}).get(cid)
            rows.append(loss_row(sid, cid, cell, key))

    rows.sort(key=lambda r: r["known_loss"], reverse=True)

    header = (
        f"{'cell':<10s} {'gt_st':<10s} {'our_st':<10s} "
        f"{'gt_actual':>14s} {'our_actual':>14s} {'rel_err':>8s} "
        f"{'st_lost':>7s} {'act_lost':>8s} {'ev_lost':>7s} {'total':>7s}  cause"
    )
    print(header)
    print("-" * len(header))
    total_status = total_actual = total_evidence = 0.0
    for r in rows:
        rel = f"{r['rel_error']:.4f}" if r["rel_error"] is not None else "—"
        actual_s = f"{r['ours_actual']:.4f}" if isinstance(r["ours_actual"], (int, float)) else str(r["ours_actual"])
        gt_actual_s = f"{r['gt_actual']:.4f}" if isinstance(r["gt_actual"], (int, float)) else str(r["gt_actual"])
        print(
            f"{r['cell']:<10s} {r['gt_status']:<10s} {str(r['ours_status']):<10s} "
            f"{gt_actual_s:>14s} {actual_s:>14s} {rel:>8s} "
            f"{r['status_lost']:>7.3f} {r['actual_lost']:>8.3f} {r['evidence_lost']:>7.3f} "
            f"{r['known_loss']:>7.3f}  {r['cause']}"
        )
        total_status += r["status_lost"]
        total_actual += r["actual_lost"]
        total_evidence += r["evidence_lost"]

    total_loss = total_status + total_actual + total_evidence
    total_cells = len(rows)
    max_points = total_cells * 1.0
    scored = max_points - total_loss

    print()
    print(f"total cells: {total_cells}")
    print(f"max score:   {max_points:.2f}")
    print(f"our score:   {scored:.2f}   (mean per cell: {scored/total_cells:.4f})")
    print()
    print("Loss breakdown:")
    print(f"  wrong status:                  {total_status:6.2f}  ({total_status/max_points*100:5.1f}% of total)")
    print(f"  actual off (status was right): {total_actual:6.2f}  ({total_actual/max_points*100:5.1f}% of total)")
    print(f"  evidence wrong or null-vs-id:  {total_evidence:6.2f}  ({total_evidence/max_points*100:5.1f}% of total)")
    print(f"  total known loss:              {total_loss:6.2f}  ({total_loss/max_points*100:5.1f}% of total)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
