"""Recall verdict for three top-loss cells: was the right transaction
in the Stage-7 context, or did the model never get to see it?

Reads only on-disk state (workspace/extractions, workspace/selections,
workspace/selections/rejected, ground_truth, ledger). No API calls.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import scenario_membership_mask  # noqa: E402


CELLS = [("B4", "6.1"), ("P1", "6.1"), ("P3", "6.1")]


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_for(sid: str, ledger: pd.DataFrame) -> pd.DataFrame:
    return ledger[scenario_membership_mask(ledger, sid)].reset_index(drop=True)


BILINGUAL_GLOSSARY = {
    "revenue":              ["revenue", "sales", "settlement", "throughput", "recover", "выручк", "поступлен", "продаж"],
    "revenue_q4":           ["revenue", "sales", "settlement", "throughput", "четверт", "quarter", "q4", "q 4"],
    "opex":                 ["operat", "maintenance", "servic", "cleaning", "utilities", "материал", "эксплуатац", "рабоч", "administrative"],
    "rent":                 ["rent", "lease", "leasing", "аренд"],
    "capital_expenditure":  ["capital", "equipment", "purchase", "installation", "construction", "капитальн", "оборудован"],
    "funding_receipts":     ["financing", "loan", "credit", "proceeds", "facility", "escrow", "financ", "финансирован"],
    "interest":             ["interest", "проценты", "процентн"],
    "related_party":        ["management", "advisory", "retainer", "advis", "аффилир"],
    "taxes":                ["tax", "levy", "vat", "нaлoг", "excise"],
    "utilities":            ["utility", "utilities", "power", "electric", "water", "коммунал", "электр"],
    "payroll":              ["payroll", "salary", "wages", "оплат", "заработ"],
    "insurance_premiums":   ["insurance", "premium", "страхов"],
    "insurance":            ["insurance", "страхов"],
}


def _keyword_matches(role: str, description: str, ledger_slice: pd.DataFrame) -> list[dict]:
    """Ledger rows whose description matches bilingual glossary for the role."""
    glossary_terms = BILINGUAL_GLOSSARY.get(role, [])
    # Also pull raw 4+char words from the role description itself.
    desc_words = [w.lower() for w in re.findall(r"[\w-]{4,}", description, flags=re.UNICODE)]
    all_terms = set(t.lower() for t in glossary_terms) | set(desc_words)
    if not all_terms:
        return []
    hits = []
    for _, row in ledger_slice.iterrows():
        row_desc = str(row.get("description", "")).lower()
        matches = sorted({t for t in all_terms if t in row_desc})
        if matches:
            hits.append({
                "txn_id": str(row["txn_id"]),
                "amount": float(row["amount"]) if pd.notna(row["amount"]) else None,
                "desc": row_desc[:70],
                "matched": matches[:5],
            })
    return hits


def _attempts_from_cached(sid: str) -> list[dict]:
    """Reconstruct the attempts we can see: cached (accepted) selection
    plus, if present, rejected/{sid}.json with its attempts list."""
    attempts: list[dict] = []
    cached = _load_json(ROOT / "workspace" / "selections" / f"{sid}.json")
    if cached is not None:
        attempts.append({
            "status": "accepted",
            "output": cached["output"],
            "soft_warnings": cached.get("soft_warnings", []),
        })
    rejected = _load_json(ROOT / "workspace" / "selections" / "rejected" / f"{sid}.json")
    if rejected is not None:
        for i, a in enumerate(rejected.get("attempts", [])):
            attempts.append({
                "status": f"rejected_{i}",
                "output": a.get("output"),
                "error": a.get("error", ""),
            })
    return attempts


def main() -> int:
    ledger = pd.read_csv(ROOT / "agentic-bank-public" / "master_ledger_2025.csv")
    gt = json.loads((ROOT / "agentic-bank-public" / "ground_truth.json").read_text(encoding="utf-8"))["scenarios"]

    for sid, cid in CELLS:
        print(f"\n===================  {sid} {cid}  ===================")
        gt_cell = gt[sid]["covenants"][cid]
        print(f"GT:  status={gt_cell['status']}  actual={gt_cell['actual']}  evidence={gt_cell['evidence_txn_id']}")

        extraction = _load_json(ROOT / "workspace" / "extractions" / f"{sid}.json")
        cov = next((c for c in extraction["output"]["covenants"] if str(c["clause_id"]) == cid), None) if extraction else None
        if cov is None:
            print(f"NO COVENANT extracted for {sid} {cid}")
            continue
        role_descriptions = cov.get("role_descriptions", {}) or {}
        print(f"operator={cov['operator']}  threshold={cov['threshold']}")
        print(f"roles: {list(role_descriptions.keys())}")
        for role, desc in role_descriptions.items():
            print(f"  role {role!r}: {desc!r}")

        ledger_slice = _slice_for(sid, ledger)
        print(f"ledger slice for {sid}: {len(ledger_slice)} rows")

        gt_ev = gt_cell.get("evidence_txn_id")
        if gt_ev:
            gt_in_slice = gt_ev in set(ledger_slice["txn_id"].astype(str))
            print(f"GT evidence {gt_ev} in slice? {gt_in_slice}")

        # Per role, list keyword-matching candidates in the slice.
        for role, desc in role_descriptions.items():
            hits = _keyword_matches(role, desc, ledger_slice)
            print(f"  role {role!r}: {len(hits)} keyword-matching candidate rows in slice")
            for h in hits[:6]:
                print(f"    {h['txn_id']}  amt={h['amount']}  match={h['matched']}  desc={h['desc']}")

        # Attempts
        attempts = _attempts_from_cached(sid)
        print(f"\nAttempts observed on disk: {len(attempts)}")
        for a in attempts:
            print(f"  --- {a['status']} ---")
            output = a.get("output") or {}
            picks = output.get(cid, {})
            print(f"    picks for {cid}: {picks}")
            u = output.get("uncertain", [])
            u_for_clause = [
                item for item in u
                if isinstance(item, dict) and item.get("role") in role_descriptions
            ]
            if u_for_clause:
                print(f"    uncertain items with role in this clause: {u_for_clause}")
            if a["status"].startswith("rejected"):
                print(f"    rejection error: {a.get('error', '')[:180]}")
            if a["status"] == "accepted" and a.get("soft_warnings"):
                for w in a["soft_warnings"]:
                    if cid in w:
                        print(f"    warning: {w[:200]}")

        # What did Sonnet pick for OTHER clauses in the same scenario?
        other_picks: dict[str, list[str]] = {}
        cached = _load_json(ROOT / "workspace" / "selections" / f"{sid}.json")
        if cached is not None:
            for other_cid, other_roles in cached["output"].items():
                if other_cid == "uncertain" or other_cid == cid:
                    continue
                if not isinstance(other_roles, dict):
                    continue
                for role, txn_ids in other_roles.items():
                    if txn_ids:
                        other_picks[f"{other_cid}/{role}"] = list(txn_ids)
        if other_picks:
            print(f"\nOther clauses in {sid} that DID pick candidates from the same slice:")
            for k, v in other_picks.items():
                print(f"  {k}: {v}")

        # Verdict
        print("\nverdict:")
        # If for every role, at least one candidate exists in the slice,
        # then Sonnet had the option to construct a non-empty selection.
        # If GT evidence is a specific txn, that txn's presence in the
        # slice is the strongest single signal.
        all_roles_have_candidates = True
        empty_roles = []
        for role, desc in role_descriptions.items():
            if not _keyword_matches(role, desc, ledger_slice):
                all_roles_have_candidates = False
                empty_roles.append(role)

        if gt_ev:
            if gt_ev in set(ledger_slice["txn_id"].astype(str)):
                print(f"  IN_CONTEXT — GT evidence {gt_ev} IS in the slice; retrieval delivered the candidate")
            else:
                print(f"  NOT_IN_CONTEXT — GT evidence {gt_ev} NOT in the slice; retrieval never gave Sonnet the row")
        else:
            if all_roles_have_candidates:
                print(f"  IN_CONTEXT — every extracted role has at least one keyword-matching candidate in the slice; without GT evidence hint we cannot prove the exact right txn was there, but retrieval was not the empty-set failure mode")
            else:
                print(f"  NOT_IN_CONTEXT — roles with zero keyword matches in slice: {empty_roles}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
