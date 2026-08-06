from pathlib import Path

from config import get_settings


def test_data_dir_can_be_overridden(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "private-data"))

    assert get_settings().data_dir == tmp_path / "private-data"
