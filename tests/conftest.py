"""Session-wide safety net: no pytest run may reach the live Anthropic API.

Individual tests that want to exercise LLM-facing code must inject a
FakeClient (see tests/test_stage6_extract.py and tests/test_doc_classify.py).
Anything that would otherwise pick up ANTHROPIC_API_KEY from the developer's
.env is disarmed here.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _scrub_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
