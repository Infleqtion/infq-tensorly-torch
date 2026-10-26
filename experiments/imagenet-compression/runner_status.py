"""Optional local receipts for supervising one invocation, without execution imports."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import uuid


def validate_status_path(path):
    """Check a new absolute destination without creating it, including for dry-run."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("--status-file must be an absolute path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Status destination already exists: {path}")
    if not path.parent.is_dir() or not os.access(path.parent, os.W_OK | os.X_OK):
        raise ValueError(f"Status parent must be an existing writable directory: {path.parent}")
    return path


def start_status(path, config_path, original_config, tracking):
    """Reserve a new receipt atomically and detect ownership changes observed before updates."""
    path = validate_status_path(path)
    receipt = {"path": path, "identity": None, "contents": None, "record": {
        "schema_version": 1, "invocation_id": uuid.uuid4().hex,
        "state": "started", "phase": "preflight",
        "configuration_path": str(Path(config_path).expanduser().resolve()),
        "configuration_sha256": hashlib.sha256(original_config).hexdigest(),
        "output_directory": None,
        "process": {"pid": os.getpid(), "parent_pid": os.getppid(),
                    "hostname": socket.gethostname(),
                    "started_utc": datetime.now(timezone.utc).isoformat()},
        "tracking_uri": tracking["uri"], "workspace": tracking["workspace"],
        "experiment_id": None, "run_id": None, "model_id": None, "model_uri": None,
        "failure": None,
    }}
    update_status(receipt)
    return receipt


def update_status(receipt, **changes):
    """Publish complete JSON through a sibling temporary file, or do nothing when disabled."""
    if receipt is None:
        return
    path = receipt["path"]
    record = {**receipt["record"], **changes}
    contents = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if receipt["identity"] is None:
            # Linking, unlike replace, refuses a destination created since validation.
            os.link(temporary, path)
        else:
            stat = path.lstat()
            if (stat.st_dev, stat.st_ino) != receipt["identity"] or path.read_bytes() != receipt["contents"]:
                raise RuntimeError(f"Status receipt is no longer owned by this invocation: {path}")
            # The check and replace are not an atomic compare-and-swap.
            # The supervising controller reserves this directory and allows one writer.
            os.replace(temporary, path)
        stat = path.lstat()
        receipt.update(record=record, contents=contents, identity=(stat.st_dev, stat.st_ino))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fail_status(receipt, error):
    """Keep the original failure authoritative if its diagnostic receipt cannot be updated."""
    if receipt is None:
        return
    cause_types, seen = [], {id(error)}
    cause = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        cause_types.append(f"{type(cause).__module__}.{type(cause).__qualname__}")
        cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
    try:
        update_status(receipt, state="failed", failure={
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "cause_types": cause_types,
        })
    except Exception as diagnostic_error:
        print(f"Failure receipt could not be updated ({type(diagnostic_error).__name__})", file=sys.stderr)


def finish_status(receipt, output):
    """Reference verified artifacts only after the runner and its run context have returned."""
    if receipt is None:
        return
    output = Path(output).resolve()
    run = json.loads((output / "run.json").read_text())
    for name, field in (("model_verification.json", "passed"),
                        ("mlflow_model_verification.json", "passed"),
                        ("execution_inputs_verification.json", "matches_recorded_commit")):
        if json.loads((output / name).read_text()).get(field) is not True:
            raise RuntimeError(f"Cannot report success without verification: {name}")
    if (output / "failure.json").exists():
        raise RuntimeError("Cannot report success for an output containing failure.json")
    names = ("measurements.json", "results.csv", "run.json", "model_verification.json",
             "mlflow_model_verification.json", "execution_inputs_verification.json",
             "source.json", "provenance.json", "original_config.yaml", "effective_config.json",
             "evaluation_manifest.json")
    artifacts = {name: {"path": str(output / name),
                        "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
                 for name in names}
    update_status(receipt, state="succeeded", phase="completed", output_directory=str(output),
                  experiment_id=run["experiment_id"], run_id=run["run_id"],
                  model_id=run["mlflow_model"]["model_id"], model_uri=run["mlflow_model"]["model_uri"],
                  artifacts=artifacts)
