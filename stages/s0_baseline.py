"""Стадия 0: валидный baseline записывается до любой дальнейшей работы."""

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path


def _write_json_atomically(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def run(data_dir: Path, workspace_dir: Path) -> dict:
    template = json.loads((data_dir / "submission_template.json").read_text(encoding="utf-8"))
    submission = deepcopy(template)
    # Top-level identification fields required by CASE format: empty
    # strings are a format-rejection cause. Filled from env (TEAM_NAME,
    # CONTACT_EMAIL, SUBMISSION_MODEL) so a fresh clone with a filled
    # .env produces a submittable file without any manual edit.
    from config import get_settings

    settings = get_settings()
    if settings.team_name:
        submission["team"] = settings.team_name
    if settings.contact_email:
        submission["contact_email"] = settings.contact_email
    if settings.submission_model:
        submission["model"] = settings.submission_model
    for clauses in submission["answers"].values():
        for cell in clauses.values():
            cell.update(status="COMPLIANT", actual=0.01, evidence_txn_id=None)
    _write_json_atomically(workspace_dir / "submission.json", submission)
    return submission
