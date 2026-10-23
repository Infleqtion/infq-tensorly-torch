"""Offline checks of execution snapshots, using disposable Git repositories only."""

import hashlib
from pathlib import Path
import subprocess
import zipfile

import pytest

from execution_inputs import snapshot_execution_inputs, verify_execution_inputs
import run_experiment as runner


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def snapshot(root, experiment):
    config = experiment / "configs/test.yaml"
    return snapshot_execution_inputs(root, experiment, config, config.read_bytes(), "tiny-test")


def test_snapshot_commits_current_inputs_and_preserves_unrelated_staging(execution_repo):
    root, experiment = execution_repo
    before = git(root, "rev-parse", "HEAD")
    source = root / "tltorch/new_source.py"
    source.write_text("value = 1\n")
    (experiment / "new_helper.py").write_text("helper = 2\n")
    selected = experiment / "configs/new-selected.yaml"
    selected.write_text("selected: new configuration\n")
    (experiment / "configs/personal.yaml").write_text("other: config\n")
    (root / "personal.ipynb").write_text("unrelated notebook")
    generated = experiment / "results/generated.py"
    generated.parent.mkdir()
    generated.write_text("must_not_be_committed = True\n")
    unrelated = root / "unrelated.txt"
    unrelated.write_text("staged change\n")
    git(root, "add", "--", "unrelated.txt")
    unrelated.write_text("unstaged change after staging\n")
    staged = git(root, "diff", "--cached", "--", "unrelated.txt")
    unstaged = git(root, "diff", "--", "unrelated.txt")
    revision = snapshot_execution_inputs(root, experiment, selected, selected.read_bytes(), "tiny-test")
    assert git(root, "rev-parse", "HEAD") != before
    assert git(root, "diff", "--cached", "--", "unrelated.txt") == staged
    assert git(root, "diff", "--", "unrelated.txt") == unstaged
    changed = set(git(root, "diff", "--name-only", "HEAD^", "HEAD").decode().splitlines())
    assert changed == {"tltorch/new_source.py", "experiments/imagenet-compression/new_helper.py",
                       "experiments/imagenet-compression/configs/new-selected.yaml"}
    assert "unrelated.txt" in git(root, "status", "--short").decode()
    assert revision["commit"] == git(root, "rev-parse", "HEAD").decode().strip()
    for name, digest in revision["source_files_sha256"].items():
        assert hashlib.sha256(git(root, "show", f"HEAD:{name}")).hexdigest() == digest
    # A later change to the selected files invalidates the provenance claim.
    selected.write_text("changed: during execution\n")
    with pytest.raises(RuntimeError, match="changed after the recorded commit.*new-selected.yaml"):
        verify_execution_inputs(root, experiment, selected, revision)


def test_unchanged_scope_reuses_head_even_with_unrelated_staged_work(execution_repo):
    root, experiment = execution_repo
    (root / "unrelated.txt").write_text("staged only\n")
    git(root, "add", "--", "unrelated.txt")
    before, index = git(root, "rev-parse", "HEAD"), git(root, "ls-files", "--stage", "-z")
    revision = snapshot(root, experiment)
    assert git(root, "rev-parse", "HEAD") == before
    assert git(root, "ls-files", "--stage", "-z") == index
    assert revision["commit"] == before.decode().strip()


def test_selected_scope_handles_deleted_sources_and_rejects_external_yaml(execution_repo, tmp_path):
    root, experiment = execution_repo
    helper = experiment / "execution_inputs.py"
    helper.unlink()
    revision = snapshot(root, experiment)
    assert revision["source_files_sha256"][str(helper.relative_to(root))] is None
    assert git(root, "show", "--format=", "--name-status", "HEAD").decode().startswith("D\t")
    outside = tmp_path / "external.yaml"
    outside.write_text("not: committed\n")
    with pytest.raises(ValueError, match="configs/ directory"):
        snapshot_execution_inputs(root, experiment, outside, outside.read_bytes(), "tiny")


def test_dry_run_preparation_and_help_do_not_mutate_git(execution_repo, monkeypatch):
    root, experiment = execution_repo
    config = experiment / "configs/resnet18_exploration.yaml"
    monkeypatch.setattr(runner, "EXPERIMENT_DIR", experiment)
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", root)
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation, dry-run, and help must not snapshot or track")
    monkeypatch.setattr(runner, "snapshot_execution_inputs", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr(runner, "prepare_data", lambda *a, **k: {"splits": {}})
    before = (git(root, "rev-parse", "HEAD"), git(root, "ls-files", "--stage", "-z"),
              git(root, "status", "--porcelain", "--untracked-files=all"))
    assert runner.main(["--config", str(config), "--dry-run"]) == 0
    assert runner.main(["--prepare-data"]) == 0
    with pytest.raises(SystemExit) as caught:
        runner.main(["--help"])
    assert caught.value.code == 0
    after = (git(root, "rev-parse", "HEAD"), git(root, "ls-files", "--stage", "-z"),
             git(root, "status", "--porcelain", "--untracked-files=all"))
    assert after == before


@pytest.mark.parametrize("operation", ["staged_deletion", "staged_rename", "unstaged_rename"])
def test_snapshot_records_removed_head_paths_and_preserves_unrelated_work(execution_repo, monkeypatch, tmp_path, operation):
    root, experiment = execution_repo
    original = "tltorch/snapshot_source.py"
    renamed = "tltorch/renamed_source.py"
    (root / original).write_text("value = 7\n")
    git(root, "add", "--", original)
    git(root, "commit", "-m", "Source before deletion or rename")
    before = git(root, "rev-parse", "HEAD")
    if operation == "staged_deletion":
        git(root, "rm", "--", original)
    elif operation == "staged_rename":
        git(root, "mv", "--", original, renamed)
    else:
        (root / original).rename(root / renamed)
    unrelated = root / "unrelated.txt"
    unrelated.write_text("staged unrelated change\n")
    git(root, "add", "--", "unrelated.txt")
    unrelated.write_text("further unstaged change\n")
    staged = git(root, "diff", "--cached", "--", "unrelated.txt")
    unstaged = git(root, "diff", "--", "unrelated.txt")

    revision = snapshot(root, experiment)
    assert git(root, "rev-parse", "HEAD") != before
    assert revision["source_files_sha256"][original] is None
    assert subprocess.run(["git", "cat-file", "-e", f"HEAD:{original}"], cwd=root,
                          capture_output=True).returncode != 0
    if operation != "staged_deletion":
        assert git(root, "show", f"HEAD:{renamed}") == (root / renamed).read_bytes()
        assert revision["source_files_sha256"][renamed] == hashlib.sha256((root / renamed).read_bytes()).hexdigest()
    assert git(root, "diff", "--cached", "--", "unrelated.txt") == staged
    assert git(root, "diff", "--", "unrelated.txt") == unstaged
    # The archive must omit the old path and contain the renamed contents, if any.
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", root)
    output = tmp_path / "archive"
    output.mkdir()
    runner.save_source(output, revision)
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert original not in archive.namelist()
        if operation != "staged_deletion":
            assert archive.read(renamed) == (root / renamed).read_bytes()
