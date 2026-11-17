from pathlib import Path

from bioinspired.experiments.snapshot import capture_repro_snapshots


def test_capture_writes_three_files(tmp_path: Path) -> None:
    capture_repro_snapshots(tmp_path)
    assert (tmp_path / "git.txt").exists()
    assert (tmp_path / "system.txt").exists()
    sys_txt = (tmp_path / "system.txt").read_text()
    assert "hostname:" in sys_txt and "python:" in sys_txt


def test_capture_git_fallback_when_no_repo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")
    capture_repro_snapshots(tmp_path)
    assert (tmp_path / "git.txt").read_text() == "(git no disponible)"
