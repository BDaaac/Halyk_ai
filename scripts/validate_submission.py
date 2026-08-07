"""One-shot validator for workspace/submission.json against the CASE format.

Exit code 0 = green; any red line prints to stderr and exits non-zero. Use
before every upload on the challenge day:

    python scripts/validate_submission.py \
        --submission workspace/submission.json \
        --template  agentic-bank-public/submission_template.json

Checks (fail-loud, no warnings):
  * file parses as JSON;
  * top-level string fields team / contact_email / model are non-empty;
  * scenario keys under 'answers' match the template exactly;
  * clause keys per scenario match the template exactly;
  * every cell has 'status', 'actual', 'evidence_txn_id';
  * status in {COMPLIANT, BREACH};
  * actual is a positive number;
  * evidence_txn_id is either null or a non-empty string;
  * no extra top-level keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ALLOWED_STATUSES = {"COMPLIANT", "BREACH"}
TOP_LEVEL_STRINGS = ("team", "contact_email", "model")
CELL_KEYS = ("status", "actual", "evidence_txn_id")


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate(submission_path: Path, template_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"submission not valid JSON: {error}"]
    try:
        template = json.loads(template_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return [f"template unreadable: {error}"]

    for field in TOP_LEVEL_STRINGS:
        value = submission.get(field)
        if not isinstance(value, str) or not value.strip():
            _fail(errors, f"top-level {field!r} must be a non-empty string; got {value!r}")

    if "answers" not in submission or not isinstance(submission["answers"], dict):
        return errors + ["top-level 'answers' must be an object"]

    expected_scenarios = set(template["answers"].keys())
    actual_scenarios = set(submission["answers"].keys())
    if expected_scenarios != actual_scenarios:
        missing = expected_scenarios - actual_scenarios
        extra = actual_scenarios - expected_scenarios
        if missing:
            _fail(errors, f"scenarios missing from submission: {sorted(missing)}")
        if extra:
            _fail(errors, f"unexpected scenarios in submission: {sorted(extra)}")

    for scenario_id, template_clauses in template["answers"].items():
        clauses = submission["answers"].get(scenario_id)
        if not isinstance(clauses, dict):
            _fail(errors, f"scenario {scenario_id}: 'clauses' must be an object")
            continue
        expected_clauses = set(template_clauses.keys())
        actual_clauses = set(clauses.keys())
        if expected_clauses != actual_clauses:
            missing = expected_clauses - actual_clauses
            extra = actual_clauses - expected_clauses
            if missing:
                _fail(errors, f"scenario {scenario_id}: missing clauses {sorted(missing)}")
            if extra:
                _fail(errors, f"scenario {scenario_id}: unexpected clauses {sorted(extra)}")

        for clause_id, cell in clauses.items():
            if not isinstance(cell, dict):
                _fail(errors, f"{scenario_id} {clause_id}: cell must be an object")
                continue
            for key in CELL_KEYS:
                if key not in cell:
                    _fail(errors, f"{scenario_id} {clause_id}: cell missing key {key!r}")

            status = cell.get("status")
            if status not in ALLOWED_STATUSES:
                _fail(errors, f"{scenario_id} {clause_id}: status must be one of {ALLOWED_STATUSES}, got {status!r}")

            actual = cell.get("actual")
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                _fail(errors, f"{scenario_id} {clause_id}: actual must be numeric, got {actual!r}")
            elif actual <= 0:
                _fail(errors, f"{scenario_id} {clause_id}: actual must be > 0, got {actual!r}")

            evidence = cell.get("evidence_txn_id")
            if evidence is not None and (not isinstance(evidence, str) or not evidence.strip()):
                _fail(
                    errors,
                    f"{scenario_id} {clause_id}: evidence_txn_id must be null or a non-empty string, got {evidence!r}",
                )

    expected_top = set(TOP_LEVEL_STRINGS) | {"answers"}
    extra_top = set(submission.keys()) - expected_top
    if extra_top:
        _fail(errors, f"unexpected top-level keys in submission: {sorted(extra_top)}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate submission.json against CASE format.")
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("workspace") / "submission.json",
        help="path to submission.json (default: workspace/submission.json)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("agentic-bank-public") / "submission_template.json",
        help="path to submission_template.json",
    )
    args = parser.parse_args()

    errors = validate(args.submission, args.template)
    if errors:
        print(f"{len(errors)} problem(s) with {args.submission}:", file=sys.stderr)
        for error in errors:
            print(f"  ✗ {error}", file=sys.stderr)
        return 1
    print(f"OK: {args.submission} is submittable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
