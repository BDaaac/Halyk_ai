import json

from config import get_settings
from pipeline import build_document_index, resolve_documents
from stages.s6_extract import _article_six


def test_article_six_has_every_template_clause_for_every_scenario():
    settings = get_settings()
    template = json.loads((settings.data_dir / "submission_template.json").read_text(encoding="utf-8"))
    index = build_document_index()

    for scenario_id, clauses in template["answers"].items():
        agreement = resolve_documents(scenario_id).agreement
        assert agreement is not None
        article = _article_six(index[agreement].text)
        assert 800 <= len(article) <= 2500, scenario_id
        assert all(clause_id in article for clause_id in clauses), scenario_id
