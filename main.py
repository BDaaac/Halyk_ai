"""CLI последовательного пайплайна проверки ковенантов."""

import argparse
import json
import os
import shutil
from decimal import Decimal
from pathlib import Path

from pipeline import run_pipeline
from scorer import score_submission


def score_command(submission_path: Path, ground_truth_path: Path) -> None:
    """Считает локальную оценку готового submission без запуска пайплайна."""
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    print(score_submission(submission, ground_truth).report())


def eval_command() -> None:
    """Резервирует отдельный путь для будущего запуска с оценкой."""
    raise NotImplementedError("eval requires stage 0")


def _drop_llm_caches(workspace_dir: Path) -> None:
    """Delete stage-6/7 caches so the next run re-issues LLM calls.

    The vision cache is intentionally preserved: vision responses are
    deterministic per page image and re-issuing them only burns money.
    """
    for name in ("extractions", "selections"):
        target = workspace_dir / name
        if target.exists():
            shutil.rmtree(target)


def run_command(*, fresh: bool, data_dir: str | None) -> None:
    if data_dir:
        os.environ["DATA_DIR"] = data_dir
    from config import get_settings

    settings = get_settings()
    if fresh:
        _drop_llm_caches(settings.workspace_dir)
    run_pipeline()
    timings = getattr(run_pipeline, "last_timings", {}) or {}
    usage = getattr(run_pipeline, "last_usage", {}) or {}
    cost = getattr(run_pipeline, "last_cost_usd", {}) or {}
    if timings or usage or cost:
        print("timings (seconds):")
        for stage, value in sorted(timings.items()):
            print(f"  {stage}: {value:.2f}")
        for stage in ("stage_6_extract", "stage_7_select"):
            totals = usage.get(stage, {})
            if totals.get("calls"):
                stage_cost = cost.get(stage, Decimal("0"))
                print(
                    f"{stage}: calls={totals['calls']} "
                    f"input={totals['input_tokens']} output={totals['output_tokens']} "
                    f"cost=${stage_cost}"
                )
        total_cost = sum(cost.values(), Decimal("0"))
        print(f"llm cost total: ${total_cost}")


def view_command(*, ground_truth: Path | None, data_dir: str | None, output: Path) -> None:
    """Emit a self-contained HTML trace viewer under reports/."""
    if data_dir:
        os.environ["DATA_DIR"] = data_dir
    from config import get_settings
    from lib.viewer import build_view

    settings = get_settings()
    template_path = Path(__file__).resolve().parent / "templates" / "viewer.html"
    gt_path = ground_truth if ground_truth and ground_truth.exists() else None
    output_path = build_view(
        workspace_dir=settings.workspace_dir,
        data_dir=settings.data_dir,
        template_path=template_path,
        output_path=output,
        ground_truth_path=gt_path,
    )
    print(f"viewer: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка кредитных ковенантов")
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="оценить готовый submission")
    score.add_argument("submission", type=Path)
    score.add_argument("ground_truth", type=Path)
    run = commands.add_parser("run", help="запустить пайплайн")
    run.add_argument(
        "--fresh",
        action="store_true",
        help="удалить workspace/extractions и workspace/selections перед запуском (vision-кэш сохраняется)",
    )
    run.add_argument(
        "--data",
        type=str,
        default=None,
        help="путь к папке с датасетом (submission_template.json, documents/, master_ledger_2025.csv)",
    )
    commands.add_parser("eval", help="запустить пайплайн и оценить результат")
    commands.add_parser("diff", help="сравнить два прогона")
    view = commands.add_parser("view", help="сгенерировать HTML-вьювер трассы")
    view.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="путь к ground_truth.json (необязателен — без него колонка со счётом не показывается)",
    )
    view.add_argument(
        "--data",
        type=str,
        default=None,
        help="путь к папке с датасетом (переопределяет DATA_DIR)",
    )
    view.add_argument(
        "--out",
        type=Path,
        default=Path("reports") / "view.html",
        help="куда положить HTML (по умолчанию reports/view.html)",
    )

    args = parser.parse_args()
    if args.command == "score":
        score_command(args.submission, args.ground_truth)
    elif args.command == "run":
        run_command(fresh=args.fresh, data_dir=args.data)
    elif args.command == "eval":
        eval_command()
    elif args.command == "view":
        view_command(ground_truth=args.ground_truth, data_dir=args.data, output=args.out)
    else:
        raise NotImplementedError("diff")


if __name__ == "__main__":
    main()
