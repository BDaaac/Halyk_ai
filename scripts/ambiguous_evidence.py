"""Enumerate cells whose evidence pool contains >1 counterfactual flipper.

The current find_counterfactual_evidence returns None for any BREACH cell
where more than one candidate transaction independently flips the verdict
to COMPLIANT. That silently loses 0.20 points per such cell in scoring
even though a deterministic tie-breaker (first flipper) would have earned
the full 0.20 whenever the GT evidence is any of the candidates.

Prints one line per BREACH scenario/clause with more than one flipper.
Zero ambiguous cells → topic closed. More → the report will name them
and the operator can decide whether the fix is worth the loss of the
"ambiguous → null" invariant.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import _read_ledger, build_document_index, resolve_documents  # noqa: E402
from lib.consolidated_retrieval import patch_group_capex  # noqa: E402
from stages.s8_compute import (  # noqa: E402
    build_clause_selection,
    evaluate_covenant,
    specifications_from_extraction,
)
from stages.s9_evidence import (  # noqa: E402
    counterfactual_kind,
    evidence_candidates,
    reverted_adjustments,
)


def _all_flippers(spec, selected_roles, adjustments, ledger, base_status) -> list[str]:
    candidates = list(dict.fromkeys(evidence_candidates(selected_roles, adjustments, ledger)))
    flippers: list[str] = []
    for txn_id in candidates:
        kind = counterfactual_kind(txn_id, adjustments, ledger)
        if kind == "revert_adjustment":
            candidate_roles = selected_roles
            candidate_adjustments = reverted_adjustments(txn_id, adjustments, ledger)
        else:
            candidate_roles = {
                role: [candidate for candidate in ids if candidate != txn_id]
                for role, ids in selected_roles.items()
            }
            candidate_adjustments = adjustments
        try:
            values = build_clause_selection(spec, candidate_roles, ledger, candidate_adjustments)
            result = evaluate_covenant(spec, values)
        except (ArithmeticError, ValueError):
            continue
        if result.status != base_status:
            flippers.append(txn_id)
    return flippers


def main() -> int:
    ledger = _read_ledger()
    document_index = build_document_index()
    submission = json.loads((ROOT / "workspace" / "submission.json").read_text(encoding="utf-8"))
    gt = json.loads((ROOT / "agentic-bank-public" / "ground_truth.json").read_text(encoding="utf-8"))["scenarios"]

    print(
        f"{'cell':<10s} {'gt_ev':<18s} {'ours_ev':<18s} "
        f"{'#cands':>7s} {'#flippers':>10s}  flippers  -&gt;gt_in_pool?"
    )
    print("-" * 110)
    total_ambiguous = 0
    total_break = 0
    for sid, clauses in submission["answers"].items():
        extraction_path = ROOT / "workspace" / "extractions" / f"{sid}.json"
        selection_path = ROOT / "workspace" / "selections" / f"{sid}.json"
        if not (extraction_path.exists() and selection_path.exists()):
            continue
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))["output"]
        selection = json.loads(selection_path.read_text(encoding="utf-8"))["output"]
        documents = resolve_documents(sid)
        if documents.kyc:
            patch_group_capex(extraction, document_index, document_index[documents.kyc].text)
        try:
            specs = specifications_from_extraction(sid, extraction)
        except Exception:
            continue
        for cid, cell in clauses.items():
            if cell.get("status") != "BREACH":
                continue
            total_break += 1
            spec = specs.get(cid)
            if spec is None:
                continue
            roles = selection.get(cid, {})
            if not isinstance(roles, dict):
                continue
            candidates = list(dict.fromkeys(
                evidence_candidates(roles, extraction.get("adjustments", []), ledger)
            ))
            flippers = _all_flippers(spec, roles, extraction.get("adjustments", []), ledger, cell["status"])
            if len(flippers) > 1:
                total_ambiguous += 1
                gt_ev = ((gt.get(sid) or {}).get("covenants", {}).get(cid) or {}).get("evidence_txn_id")
                gt_in = "yes" if gt_ev in flippers else "no" if gt_ev else "gt=null"
                print(
                    f"{sid+' '+cid:<10s} {str(gt_ev):<18s} {str(cell.get('evidence_txn_id')):<18s} "
                    f"{len(candidates):>7d} {len(flippers):>10d}  {flippers}  -&gt;{gt_in}"
                )

    print()
    print(f"BREACH cells scanned:  {total_break}")
    print(f"ambiguous (>1 flipper): {total_ambiguous}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
