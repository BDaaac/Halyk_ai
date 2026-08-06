import json
from copy import deepcopy
from pathlib import Path

from scorer import score_submission


FIXTURES = Path(__file__).parent / "fixtures"


def baseline_submission(template: dict) -> dict:
    """Строит худший стабильный baseline без чисел из модели."""
    result = deepcopy(template)
    for scenario in result["answers"].values():
        for cell in scenario.values():
            cell.update(status="COMPLIANT", actual=0.01, evidence_txn_id=None)
    return result


def test_scorer_baseline_floor():
    template = json.loads((FIXTURES / "submission_template.json").read_text(encoding="utf-8"))
    ground_truth = json.loads((FIXTURES / "ground_truth.json").read_text(encoding="utf-8"))

    score = score_submission(baseline_submission(template), ground_truth).mean
    assert round(score, 4) == 0.2639
    assert 0.26 <= score <= 0.27
