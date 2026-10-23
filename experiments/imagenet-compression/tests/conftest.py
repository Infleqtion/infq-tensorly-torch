"""Import the file-based workflow without treating its hyphenated directory as a package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import shutil
import subprocess
import pytest


@pytest.fixture
def execution_repo(tmp_path):
    """A disposable repository; automatic commits must never touch the real checkout."""
    import run_experiment as runner

    root = tmp_path / "repository"
    experiment = root / "experiments/imagenet-compression"
    experiment.mkdir(parents=True)
    for name in ("setup.py", "README.rst", "pyproject.toml", "uv.lock", ".python-version"):
        shutil.copyfile(runner.REPOSITORY_ROOT / name, root / name)
    for directory in (runner.EXPERIMENT_DIR, runner.REPOSITORY_ROOT / "tltorch"):
        for source in directory.rglob("*.py"):
            relative = source.relative_to(runner.REPOSITORY_ROOT)
            if "tests" in relative.parts:
                continue
            # Only top-level experiment helpers and library Python files are in scope.
            if directory == runner.EXPERIMENT_DIR and source.parent != directory:
                continue
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    (experiment / "configs").mkdir()
    shutil.copyfile(runner.EXPERIMENT_DIR / "configs/resnet18_exploration.yaml",
                    experiment / "configs/resnet18_exploration.yaml")
    (experiment / "configs/test.yaml").write_text("original: configuration\n")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n.venv/\nexperiments/imagenet-compression/results/\n")
    (root / "unrelated.txt").write_text("original\n")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    git("init", "--initial-branch=experiment-test")
    git("config", "user.name", "Offline test")
    git("config", "user.email", "offline@example.invalid")
    git("add", "--", "setup.py", "README.rst", "pyproject.toml", "uv.lock", ".python-version",
        ".gitignore", "unrelated.txt", "tltorch", "experiments")
    git("commit", "-m", "Initial test inputs")
    return root, experiment
