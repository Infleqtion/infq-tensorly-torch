"""Receipt ownership and atomic publication need no execution dependencies."""

import json
import os
from pathlib import Path

import pytest

import runner_status


def test_receipt_detects_another_owners_replacement_before_update(tmp_path, monkeypatch):
    path = tmp_path / "runner_status.json"
    receipt = runner_status.start_status(path, tmp_path / "config.yaml", b"example: true\n",
                                         {"uri": "https://tracking.example.invalid", "workspace": "test"})
    record = json.loads(path.read_text())
    assert record["state"] == "started" and record["run_id"] is None
    original_inode = path.stat().st_ino
    runner_status.update_status(receipt, state="running", run_id="synthetic-run")
    assert path.stat().st_ino != original_inode
    assert json.loads(path.read_text())["run_id"] == "synthetic-run"
    replacement = tmp_path / "replacement"
    replacement.write_text("another owner's bytes\n")
    replacement.replace(path)
    with pytest.raises(RuntimeError, match="no longer owned"):
        runner_status.update_status(receipt, state="succeeded")
    assert path.read_text() == "another owner's bytes\n"
    assert not list(tmp_path.glob(".runner_status.json.*"))


def test_initial_publication_refuses_a_file_created_after_validation(tmp_path, monkeypatch):
    path = tmp_path / "runner_status.json"
    link = os.link
    def concurrent_owner(source, destination):
        Path(destination).write_text("concurrent owner's bytes\n")
        link(source, destination)
    monkeypatch.setattr(runner_status.os, "link", concurrent_owner)
    with pytest.raises(FileExistsError):
        runner_status.start_status(path, tmp_path / "config.yaml", b"test: true\n",
                                   {"uri": "https://tracking.example.invalid", "workspace": "test"})
    assert path.read_text() == "concurrent owner's bytes\n"
    assert not list(tmp_path.glob(".runner_status.json.*"))


def test_status_path_validation_is_read_only_and_rejects_invalid_destinations(tmp_path):
    with pytest.raises(ValueError, match="absolute path"):
        runner_status.validate_status_path("relative.json")
    with pytest.raises(ValueError, match="existing writable directory"):
        runner_status.validate_status_path(tmp_path / "missing-parent/runner_status.json")
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="already exists"):
        runner_status.validate_status_path(link)
    assert link.is_symlink() and not (tmp_path / "missing-parent").exists()
