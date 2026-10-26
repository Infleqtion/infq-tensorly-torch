"""Commit and verify the explicit execution inputs for one configured experiment."""

import hashlib
import os
from pathlib import Path
import subprocess


def git(root, *args, input=None):
    """Run Git without interpreting filenames as path patterns."""
    return subprocess.check_output(
        ["git", *args], cwd=root, input=input,
        env={**os.environ, "GIT_LITERAL_PATHSPECS": "1"}, stderr=subprocess.PIPE,
    )


def configuration_files(root, experiment, paths):
    """Validate explicitly named YAMLs and return unique repository-relative paths."""
    root, experiment = Path(root), Path(experiment)
    configs = experiment / "configs"
    candidates = set(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0"))
    names = set()
    for value in paths:
        path = Path(value).expanduser()
        if path.is_symlink():
            raise ValueError(f"Execution YAML must be an existing regular file, not a symbolic link: {path}")
        path = path.resolve()
        if not path.is_relative_to(configs) or path.suffix not in (".yaml", ".yml"):
            raise ValueError("Execution YAML must be inside this experiment's configs/ directory; "
                             "copy an external configuration there before execution")
        if not path.is_file():
            raise ValueError(f"Execution YAML must be an existing regular file: {path}")
        name = path.relative_to(root).as_posix()
        if name not in candidates:
            raise ValueError(f"The selected YAML is ignored by Git; move it to a trackable configs/ path: {name}")
        names.add(name)
    return sorted(names)


def execution_files(root, experiment, config_path, provenance_configs=()):
    """Select source, shared build inputs, and only explicitly named configurations."""
    root, experiment = Path(root), Path(experiment)
    configurations = configuration_files(root, experiment, [config_path, *provenance_configs])
    # A staged deletion is absent from the index; HEAD still supplies its old path.
    candidates = set(git(root, "ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0"))
    candidates.update(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0"))
    experiment_name = experiment.relative_to(root).as_posix()
    files = {"setup.py", "README.rst", "pyproject.toml", "uv.lock", ".python-version", *configurations}
    for name in candidates:
        path = Path(name)
        library_source = path.parts[:1] == ("tltorch",) and "tests" not in path.parts
        experiment_source = path.parent.as_posix() == experiment_name
        if path.suffix == ".py" and (library_source or experiment_source):
            files.add(name)
    for name in files:
        if (root / name).is_symlink():
            raise ValueError(f"Execution input must not be a symbolic link: {name}")
    return sorted(files)


def file_hashes(root, files):
    """Include deletions so that restoring a removed source file is also detected."""
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            if (root / name).is_file() else None for name in files}


def snapshot_execution_inputs(root, experiment, config_path, original_config, run_name, *, provenance_configs=None):
    """Commit current scoped contents, preserving unrelated staged and unstaged work."""
    root = Path(root)
    if Path(config_path).expanduser().read_bytes() != original_config:
        raise RuntimeError("Selected YAML changed since configuration loading; reload it before execution")
    provenance_configs = provenance_configs or {}
    extra_paths = [root / name for name in provenance_configs]
    files = execution_files(root, experiment, config_path, extra_paths)
    for name, contents in provenance_configs.items():
        if (root / name).read_bytes() != contents:
            raise RuntimeError(f"Provenance YAML changed since configuration loading: {name}")
    hashes = file_hashes(root, files)
    before = git(root, "rev-parse", "HEAD").decode().strip()
    # Git's --only uses the named working-tree files and preserves staging elsewhere.
    # New files need an intent-to-add entry before --only can name them.
    tracked = set(git(root, "ls-files", "-z").decode().split("\0"))
    pathspec = b"\0".join(name.encode() for name in files) + b"\0"
    try:
        head_hashes = {}
        for name in files:
            try:
                data = git(root, "show", f"{before}:{name}")
            except subprocess.CalledProcessError:
                head_hashes[name] = None
            else:
                head_hashes[name] = hashlib.sha256(data).hexdigest()
        if hashes != head_hashes:
            new_files = [name for name in files if name not in tracked and hashes[name] is not None]
            if new_files:
                git(root, "add", "--intent-to-add", "--", *new_files)
            git(root, "commit", "--only", "--message",
                f"experiment: snapshot code and configuration for {run_name}",
                "--pathspec-from-file=-", "--pathspec-file-nul", input=pathspec)
        commit = git(root, "rev-parse", "HEAD").decode().strip()
        if commit != before:
            changed = set(git(root, "diff", "--name-only", "-z", before, commit).decode().split("\0")) - {""}
            if changed - set(files):
                raise RuntimeError("A Git hook committed files outside the execution scope; review the commit manually")
        for name, digest in hashes.items():
            try:
                committed = hashlib.sha256(git(root, "show", f"{commit}:{name}")).hexdigest()
            except subprocess.CalledProcessError:
                committed = None
            if committed != digest:
                raise RuntimeError(f"Execution input does not match the recorded commit: {name}")
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Cannot safely snapshot execution inputs: {detail}. "
                           "Resolve Git identity, hooks, index locks, or merge conflicts and retry; "
                           "no MLflow run was created") from error
    revision = {"commit": commit, "source_files_sha256": hashes,
                "configuration_path": Path(config_path).resolve().relative_to(root).as_posix()}
    if provenance_configs:
        revision["provenance_configuration_paths"] = sorted(provenance_configs)
    verify_execution_inputs(root, experiment, config_path, revision)
    return revision


def verify_execution_inputs(root, experiment, config_path, revision):
    """Fail if the selected source or configuration has changed since the snapshot."""
    extra_paths = [Path(root) / name for name in revision.get("provenance_configuration_paths", [])]
    files = execution_files(root, experiment, config_path, extra_paths)
    current = file_hashes(Path(root), files)
    expected = revision["source_files_sha256"]
    changed = sorted(name for name in current.keys() | expected.keys()
                     if current.get(name) != expected.get(name))
    if changed:
        raise RuntimeError("Execution inputs changed after the recorded commit: " + ", ".join(changed))
    return {"matches_recorded_commit": True, "commit": revision["commit"],
            "current_head": git(root, "rev-parse", "HEAD").decode().strip(),
            "working_tree_status": git(root, "status", "--short", "--untracked-files=all").decode()}
