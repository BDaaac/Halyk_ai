"""
Локальный скорер, повторяющий официальную формулу Halyk AI Challenge.

Правила (CASE.ru.md §4):
  status           0.50  — точное совпадение "COMPLIANT"/"BREACH"
  actual           0.30  — по убывающей шкале: 0.30 * max(0, 1 - e/0.05)
  evidence_txn_id  0.20  — точное совпадение; если в ключе null,
                           эти 0.20 убывают по той же шкале, что и actual
  Неверный status  -> вся ячейка 0.
  Пропущенная/переименованная ячейка -> 0.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field


ERR_TOLERANCE = 0.05
W_STATUS = 0.50
W_ACTUAL = 0.30
W_EVIDENCE = 0.20


def actual_scale(answer_actual, key_actual) -> float:
    """Доля от полного балла: 1.0 при точном совпадении, 0.0 при ошибке >= 5%."""
    if answer_actual is None or isinstance(answer_actual, bool):
        return 0.0
    if not isinstance(answer_actual, (int, float)):
        return 0.0
    if key_actual == 0:
        return 1.0 if answer_actual == 0 else 0.0
    rel_err = abs(answer_actual - key_actual) / abs(key_actual)
    return max(0.0, 1.0 - rel_err / ERR_TOLERANCE)


@dataclass
class CellScore:
    scenario_id: str
    clause_id: str
    total: float
    status_pts: float
    actual_pts: float
    evidence_pts: float
    reason: str = ""


def score_cell(scenario_id: str, clause_id: str, answer: dict | None, key: dict) -> CellScore:
    zero = CellScore(scenario_id, clause_id, 0.0, 0.0, 0.0, 0.0)

    if answer is None:
        zero.reason = "ячейка отсутствует"
        return zero

    if answer.get("status") != key["status"]:
        zero.reason = f"status {answer.get('status')!r} != {key['status']!r}"
        return zero

    status_pts = W_STATUS
    scale = actual_scale(answer.get("actual"), key["actual"])
    actual_pts = W_ACTUAL * scale

    if key["evidence_txn_id"] is None:
        # В ключе null — баллы за evidence убывают вместе с actual.
        evidence_pts = W_EVIDENCE * scale
        reason = "" if scale == 1.0 else f"actual off by {1 - scale:.1%} of tolerance"
    else:
        exact = answer.get("evidence_txn_id") == key["evidence_txn_id"]
        evidence_pts = W_EVIDENCE if exact else 0.0
        reason = "" if exact else f"evidence {answer.get('evidence_txn_id')!r} != {key['evidence_txn_id']!r}"

    return CellScore(
        scenario_id, clause_id,
        status_pts + actual_pts + evidence_pts,
        status_pts, actual_pts, evidence_pts, reason,
    )


@dataclass
class RunScore:
    cells: list[CellScore] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(c.total for c in self.cells) / len(self.cells) if self.cells else 0.0

    @property
    def status_accuracy(self) -> float:
        if not self.cells:
            return 0.0
        return sum(1 for c in self.cells if c.status_pts > 0) / len(self.cells)

    def report(self) -> str:
        lines = [
            f"Средний балл по ячейке : {self.mean:.4f}",
            f"Точность status        : {self.status_accuracy:.1%} "
            f"({sum(1 for c in self.cells if c.status_pts > 0)}/{len(self.cells)})",
            f"Сумма                  : {sum(c.total for c in self.cells):.2f} / {len(self.cells)}",
            "",
        ]
        losers = sorted((c for c in self.cells if c.total < 1.0), key=lambda c: c.total)
        if losers:
            lines.append("Ячейки с потерями:")
            for c in losers:
                lines.append(f"  {c.scenario_id:>4} {c.clause_id}  {c.total:.3f}   {c.reason}")
        else:
            lines.append("Все ячейки на 1.000")
        return "\n".join(lines)


def score_submission(submission: dict, ground_truth: dict) -> RunScore:
    answers = submission.get("answers", {})
    run = RunScore()
    for scenario_id, sc in ground_truth["scenarios"].items():
        for clause_id, key in sc["covenants"].items():
            answer = answers.get(scenario_id, {}).get(clause_id)
            run.cells.append(score_cell(scenario_id, clause_id, answer, key))
    return run


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python scorer.py submission.json ground_truth.json")
        raise SystemExit(2)
    submission = json.load(open(sys.argv[1], encoding="utf-8"))
    ground_truth = json.load(open(sys.argv[2], encoding="utf-8"))
    print(score_submission(submission, ground_truth).report())


if __name__ == "__main__":
    main()
