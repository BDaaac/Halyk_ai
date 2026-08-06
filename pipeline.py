"""Публичные границы будущего последовательного пайплайна."""


def run_pipeline(*, stop_after_stage: int | None = None):
    raise NotImplementedError("stage 0")


def gap_scan():
    raise NotImplementedError("stage 1")


def build_document_index():
    raise NotImplementedError("stage 2")


def read_document_text(doc_id: str):
    raise NotImplementedError("stage 2")


def get_account_to_scenario():
    raise NotImplementedError("stage 1")


def get_corrections():
    raise NotImplementedError("stage 5")


def resolve_documents(scenario_id: str):
    raise NotImplementedError("stage 4")


def get_specifications():
    raise NotImplementedError("stage 6")


def related_party_threshold(scenario_id: str):
    raise NotImplementedError("stage 6")


def related_parties(scenario_id: str):
    raise NotImplementedError("stage 8")


def run_scenario(scenario_id: str):
    raise NotImplementedError("stage 8")


def recompute_without(scenario_id: str, clause_id: str, txn_id: str):
    raise NotImplementedError("stage 9")
