"""CLI последовательного пайплайна проверки ковенантов."""

import argparse
import json
from pathlib import Path

from pipeline import run_pipeline
from scorer import score_submission


def score_command(submission_path: Path, ground_truth_path: Path) -> None:
    """Считает локальную оценку готового submission без запуска пайплайна."""
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    print(score_submission(submission, ground_truth).report())


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка кредитных ковенантов")
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="оценить готовый submission")
    score.add_argument("submission", type=Path)
    score.add_argument("ground_truth", type=Path)
    commands.add_parser("run", help="запустить пайплайн")
    commands.add_parser("eval", help="запустить пайплайн и оценить результат")
    commands.add_parser("diff", help="сравнить два прогона")

    args = parser.parse_args()
    if args.command == "score":
        score_command(args.submission, args.ground_truth)
    elif args.command in {"run", "eval"}:
        run_pipeline()
    else:
        raise NotImplementedError("diff")


if __name__ == "__main__":
    main()
