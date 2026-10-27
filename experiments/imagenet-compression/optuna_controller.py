"""Choose pilot configurations and supervise the existing runner, one child at a time.

No models or tracking clients are constructed here. Study state belongs on trusted,
persistent local disk; the runner alone owns scientific work and execution commits.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

import yaml

import execution_inputs as inputs
import run_experiment as runner


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    """Keep number types and list order in compatibility and request identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode() + b"\n")


def atomic_bytes(path, contents):
    """Publish one controller-owned record; these paths are protected by our locks."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def layer_type(name):
    return "linear" if name == "fc" else "convolution"


def load_study(path):
    """Validate study controls without inspecting data, loading models, or importing Optuna."""
    path = Path(path).expanduser().absolute()
    experiment, root = runner.EXPERIMENT_DIR, runner.REPOSITORY_ROOT
    inputs.configuration_files(root, experiment, [path])
    definition = yaml.load(path.read_bytes(), Loader=runner.UniqueSafeLoader)
    required = ("schema_version", "study_name", "mode", "base_config", "state_dir", "trial_config_dir",
                "enabled_layers", "methods", "storage_targets", "convolution_tucker_fixed_rank_modes",
                "classifier_tensorization", "fitting", "objectives", "sampler", "limits", "pilot_cases")
    runner._keys(definition, required, (), "study")
    if type(definition["schema_version"]) is not int or definition["schema_version"] != 1 or definition["mode"] != "individual_pilot":
        raise ValueError("Only schema_version 1 individual_pilot is implemented")
    if not isinstance(definition["study_name"], str) or not definition["study_name"].strip():
        raise ValueError("study_name must be nonempty")
    layers = definition["enabled_layers"]
    if not isinstance(layers, list) or not layers or any(not isinstance(n, str) or not n for n in layers) or len(set(layers)) != len(layers):
        raise ValueError("enabled_layers must contain distinct nonempty names in conversion order")
    for name in layers:
        if name != "fc" and not (re.fullmatch(r"layer[1-4]\.[01]\.conv[12]", name)):
            raise ValueError(f"Declare a supported ResNet18 convolution name or fc: {name}")
    expected_methods = {"convolution": ["Dense", "CP", "Tucker", "TT"],
                        "linear": ["Dense", "CP", "Tucker", "BlockTT"]}
    if definition["methods"] != expected_methods:
        raise ValueError("methods must declare the reviewed legal ordered choices for convolution and linear")
    targets = definition["storage_targets"]
    if not isinstance(targets, list) or not targets or any(type(v) is not float or not 0 < v <= 1 for v in targets) or len(set(targets)) != len(targets):
        raise ValueError("storage_targets must be distinct floating-point fractions in (0, 1]")
    if definition["convolution_tucker_fixed_rank_modes"] != "spatial":
        raise ValueError("Convolution Tucker fixed_rank_modes must be spatial")
    tensorization = definition["classifier_tensorization"]
    if tensorization != {"in_tensorized_features": [8, 8, 8], "out_tensorized_features": [10, 10, 10]}:
        raise ValueError("Classifier tensorization must preserve the reviewed 512-to-1000 geometry")
    expected_fitting = {"CP": {"init": "random", "n_iter_max": 20, "tol": 1e-8,
                               "random_state": 0, "l2_reg": 1e-5},
                        "Tucker": {"init": "svd", "n_iter_max": 25, "tol": 1e-4, "random_state": None}}
    if canonical(definition["fitting"]) != canonical(expected_fitting):
        raise ValueError("fitting must match the reviewed CP/Tucker controls")
    if definition["objectives"] != [{"name": "top1_percent", "direction": "maximize"},
                                     {"name": "parameter_count", "direction": "minimize"}]:
        raise ValueError("Objectives must maximize top1_percent then minimize parameter_count")
    expected_sampler = {"name": "TPE", "seed": 20261026, "n_startup_trials": 5, "n_ei_candidates": 24,
                        "multivariate": False, "group": False, "constant_liar": False}
    if canonical(definition["sampler"]) != canonical(expected_sampler):
        raise ValueError("sampler must match the reviewed explicit TPE settings")
    limits = definition["limits"]
    runner._keys(limits, ("total_attempts", "per_trial_seconds", "study_deadline_hours",
                         "consecutive_candidate_failures", "workers", "shutdown_seconds"), (), "limits")
    for name in limits:
        runner._integer(limits[name], 1, f"limits.{name}")
    if limits["workers"] != 1 or limits["shutdown_seconds"] > 120:
        raise ValueError("Only one worker and at most 120 shutdown seconds are supported")
    cases = definition["pilot_cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("pilot_cases must be a nonempty list")
    for case in cases:
        runner._keys(case, ("layer", "factorization"), ("storage_target",), "pilot case")
        name, method = case["layer"], case["factorization"]
        if method == "Dense":
            if name is not None or "storage_target" in case:
                raise ValueError("Dense pilot case has layer null and no storage target")
        elif name not in layers or method not in expected_methods[layer_type(name)] or type(case.get("storage_target")) is not float or case["storage_target"] not in targets:
            raise ValueError("Pilot case must name an enabled layer, legal method, and floating target")
    base_path = runner.experiment_path(definition["base_config"])
    inputs.configuration_files(root, experiment, [base_path])
    base, _ = runner.load_config(base_path)
    if base.get("compression") is not None:
        raise ValueError("base_config must describe the dense single-model execution")
    exploration = runner.experiment_path("data/processed/imagenetv2/exploration.json")
    if Path(base["data"]["manifest"]) != exploration:
        raise ValueError("Controller studies may use only the exploration manifest")
    for name in ("state_dir", "trial_config_dir"):
        if not isinstance(definition[name], str) or not definition[name].strip():
            raise ValueError(f"{name} must be a nonempty path")
    trial_dir = runner.experiment_path(definition["trial_config_dir"])
    if not trial_dir.is_relative_to(experiment / "configs") or trial_dir == experiment / "configs":
        raise ValueError("trial_config_dir must be a nested configs/ directory")
    if base_path.is_relative_to(trial_dir) or path.is_relative_to(trial_dir):
        raise ValueError("Trial directory must not contain the study or base template")
    state_dir = runner.experiment_path(definition["state_dir"])
    if state_dir.is_relative_to(root):
        raise ValueError("state_dir must be outside the repository on trusted persistent local disk")
    return {"definition": definition, "path": path, "base_path": base_path, "base": base,
            "base_raw": yaml.load(base_path.read_bytes(), Loader=runner.UniqueSafeLoader),
            "state_dir": state_dir, "trial_dir": trial_dir}


def pilot_parameters(definition, case):
    """Fully specify each queued case; no unspecified method is sampled at random."""
    params = {f"{name}.method": case["factorization"] if name == case["layer"] else "Dense"
              for name in definition["enabled_layers"]}
    if case["factorization"] != "Dense":
        params[f"{case['layer']}.{case['factorization']}.storage_target"] = case["storage_target"]
    return params


def suggest_layers(trial, definition):
    """Conditional choices also work for a later separately reviewed joint search."""
    layers = {}
    for name in definition["enabled_layers"]:
        method = trial.suggest_categorical(f"{name}.method", definition["methods"][layer_type(name)])
        if method == "Dense":
            continue
        target = trial.suggest_categorical(f"{name}.{method}.storage_target", definition["storage_targets"])
        options = {"factorization": method, "rank": target}
        if name == "fc":
            options.update(deepcopy(definition["classifier_tensorization"]))
        elif method == "Tucker":
            options["fixed_rank_modes"] = definition["convolution_tucker_fixed_rank_modes"]
        if method in definition["fitting"]:
            options["decomposition_kwargs"] = deepcopy(definition["fitting"][method])
        layers[name] = options
    return layers


def render_config(context, layers, run_name):
    config = deepcopy(context["base_raw"])
    config["compression"] = {"implementation": "factorized", "layers": layers} if layers else None
    config["tracking"]["run_name"] = run_name
    return yaml.safe_dump(config, sort_keys=False).encode()


@contextmanager
def exclusive_locks(context):
    """Hold both a study lock and a checkout lock shared by different studies."""
    state = context["state_dir"]
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    check_trusted(state)
    lock_root = Path(tempfile.gettempdir()) / f"tltorch-optuna-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    check_trusted(lock_root)
    checkout = digest(str(runner.REPOSITORY_ROOT.resolve()).encode())
    streams = []
    try:
        for path in (lock_root / f"{checkout}.lock", state / "controller.lock"):
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            stream = os.fdopen(descriptor, "a+")
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another controller owns the checkout or study lock: {path}") from error
        yield
    finally:
        for stream in reversed(streams):
            stream.close()


def check_trusted(path):
    """Sampler pickle is executable data: accept only owned, non-shared local state."""
    stat = path.lstat()
    if path.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o022:
        raise RuntimeError(f"State must be owned by this user and not writable by others: {path}")


def current_head():
    return inputs.git(runner.REPOSITORY_ROOT, "rev-parse", "HEAD").decode().strip()


def execution_identity(context):
    """Hash the existing execution scope; trial YAMLs are the only variable source inputs."""
    names = inputs.execution_files(runner.REPOSITORY_ROOT, runner.EXPERIMENT_DIR,
                                  context["base_path"], [context["path"]])
    metadata_path = context["state_dir"] / "study.json"
    if metadata_path.exists():
        check_trusted(metadata_path)
        # HEAD drops a deletion after its snapshot. Keep its tombstone so restoring
        # that path later is still detected, including a now-ignored restoration.
        previous = read_json(metadata_path)["compatibility"]["source_files_sha256"]
        names = sorted(set(names) | {name for name, value in previous.items() if value is None})
    return inputs.file_hashes(runner.REPOSITORY_ROOT, names)


def compatibility(context):
    """Inspect inputs without inference; paths and machine identity deliberately restrict reuse."""
    base = context["base"]
    manifest, manifest_bytes = runner.read_manifest(Path(base["data"]["raw_dir"]), Path(base["data"]["manifest"]))
    # Resolve the configured cache without mutating the caller's global Hub cache.
    import torch.hub
    old_hub = torch.hub._hub_dir
    try:
        cache = runner.configure_weight_cache(base)
        _, weights = runner.cached_weights()
    finally:
        torch.hub._hub_dir = old_hub
    gpu_identity = None
    if base["runtime"]["device"].startswith("cuda"):
        try:
            gpu_identity = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name", "--format=csv,noheader"],
                timeout=10, stderr=subprocess.PIPE).decode().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("Cannot establish the configured GPU identity before a study") from error
    return {"source_files_sha256": execution_identity(context), "definition": context["definition"],
            "fixed_config": base, "manifest_sha256": digest(manifest_bytes),
            "manifest_images_sha256": digest(canonical(manifest)), "weights": weights, "cache": cache,
            "packages": {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version for d in importlib.metadata.distributions()},
            "python": sys.version, "machine": socket.gethostname(), "gpu_identity": gpu_identity,
            "runtime_environment": {name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")}}


def check_lineage(context, metadata):
    """Allow the first scoped snapshot, then only generated-configuration commits."""
    root = runner.REPOSITORY_ROOT
    previous, head = metadata["last_execution_head"], current_head()
    try:
        inputs.git(root, "merge-base", "--is-ancestor", previous, head)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("HEAD left the recorded execution lineage") from error
    first = not metadata["has_execution_snapshot"]
    allowed = set(metadata["compatibility"]["source_files_sha256"])
    prefix = context["trial_dir"].relative_to(root).as_posix() + "/"
    for commit in inputs.git(root, "rev-list", "--reverse", f"{previous}..{head}").decode().splitlines():
        if len(inputs.git(root, "rev-list", "--parents", "-n", "1", commit).split()) != 2:
            raise RuntimeError("A merge entered the study lineage; explicit reconciliation is required")
        names = set(inputs.git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).decode().splitlines())
        if any(not name.startswith(prefix) and (not first or name not in allowed) for name in names):
            raise RuntimeError("Unexpected changes in snapshot lineage; start a new study or review explicitly")
        first = False
    return head


def history_hash(study):
    return digest(canonical([{ "number": t.number, "state": t.state.name, "params": t.params,
                               "values": t.values, "user_attrs": t.user_attrs, "system_attrs": t.system_attrs}
                             for t in study.get_trials(deepcopy=False)]))


def save_sampler(study, state):
    data = pickle.dumps(study.sampler, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_bytes(state / "sampler.pickle", data)
    atomic_json(state / "sampler.json", {"optuna_version": "4.9.0", "history_sha256": history_hash(study),
                                       "pickle_sha256": digest(data)})


def open_study(context, contract):
    """Queue once and resume only a matching database/sampler pair, never a fresh RNG."""
    import optuna
    if optuna.__version__ != "4.9.0":
        raise RuntimeError("This protocol requires optuna==4.9.0")
    state, definition = context["state_dir"], context["definition"]
    metadata_path, database = state / "study.json", state / "study.sqlite3"
    creating = not metadata_path.exists()
    if creating and any((state / name).exists() for name in ("study.sqlite3", "sampler.json", "sampler.pickle")):
        raise RuntimeError("Incomplete state creation; reconcile existing files rather than overwrite them")
    if not creating:
        check_trusted(metadata_path)
        metadata = read_json(metadata_path)
        if canonical(metadata["compatibility"]) != canonical(contract):
            raise RuntimeError("Study code, inputs, environment, or definition changed; incompatible resume")
        check_lineage(context, metadata)
        for path in (database, state / "sampler.json", state / "sampler.pickle"):
            check_trusted(path)
        checkpoint = read_json(state / "sampler.json")
        data = (state / "sampler.pickle").read_bytes()
        if checkpoint["optuna_version"] != optuna.__version__ or digest(data) != checkpoint["pickle_sha256"]:
            raise RuntimeError("Untrusted or inconsistent sampler checkpoint")
        sampler = None
    else:
        settings = {k: v for k, v in definition["sampler"].items() if k != "name"}
        sampler = optuna.samplers.TPESampler(**settings)
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{database}",
                                        engine_kwargs={"connect_args": {"timeout": 20}})
    if creating:
        study = optuna.create_study(study_name=definition["study_name"], storage=storage, sampler=sampler,
                                    directions=[o["direction"] for o in definition["objectives"]])
        study.set_metric_names([o["name"] for o in definition["objectives"]])
        study.set_user_attr("compatibility_sha256", digest(canonical(contract)))
        for case in definition["pilot_cases"]:
            study.enqueue_trial(pilot_parameters(definition, case))
        now = time.time()
        metadata = {"schema_version": 1, "created_at": now,
                    "deadline": now + definition["limits"]["study_deadline_hours"] * 3600,
                    "compatibility": contract, "initial_head": current_head(),
                    "last_execution_head": current_head(), "has_execution_snapshot": False,
                    "consecutive_candidate_failures": 0}
        atomic_json(metadata_path, metadata)
        save_sampler(study, state)
    else:
        study = optuna.load_study(study_name=definition["study_name"], storage=storage)
        if study.user_attrs.get("compatibility_sha256") != digest(canonical(contract)) or history_hash(study) != checkpoint["history_sha256"]:
            raise RuntimeError("Database and sampler history disagree; stop for reconciliation")
        # Check the database before executing trusted pickle data.
        study.sampler = pickle.loads(data)
    return study, metadata


def reserve_attempt(context, trial, layers):
    directory = context["state_dir"] / "attempts"
    directory.mkdir(exist_ok=True)
    directory = directory / f"trial_{trial.number:04d}_{uuid.uuid4().hex}"
    directory.mkdir(mode=0o700)
    config_dir = context["trial_dir"]
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"trial_{trial.number:04d}_{directory.name.rsplit('_', 1)[1]}.yaml"
    contents = render_config(context, layers, f"{context['definition']['study_name']}-{trial.number:04d}")
    with path.open("xb") as stream:
        stream.write(contents)
    # Apply precisely the same path/config validation as a manual runner invocation.
    runner.load_config(path)
    inputs.configuration_files(runner.REPOSITORY_ROOT, runner.EXPERIMENT_DIR, [path])
    (directory / "configuration.yaml").write_bytes(contents)
    record = {"trial_number": trial.number, "directory": str(directory), "configuration_path": str(path),
              "configuration_sha256": digest(contents), "layers": layers,
              "status_path": str(directory / "runner_status.json"), "created_at": time.time(),
              "exit_code": None, "process": None, "outcome": "reserved"}
    atomic_json(directory / "attempt.json", record)
    trial.set_user_attr("attempt_path", str(directory / "attempt.json"))
    return record


def process_identity(pid):
    """Linux start ticks and boot ID distinguish a live child from a reused PID."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return {"pid": pid, "pgid": int(fields[2]), "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "hostname": socket.gethostname()}
    except FileNotFoundError:
        return None


def group_members(identity):
    """Return live members of an owned session, excluding already exited zombies."""
    if identity["hostname"] != socket.gethostname() or identity["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise RuntimeError("Cannot reconcile process identity on a different host or boot")
    leader = process_identity(identity["pid"])
    if leader is not None and leader != identity:
        raise RuntimeError("PID identity changed; refusing to signal a potentially unrelated process")
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
            fields = text[text.rfind(")") + 2:].split()
            if int(fields[2]) == identity["pgid"] and fields[0] != "Z":
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def terminate_group(identity, process, deadline):
    """Escalate within the caller's budget and reap our child; never signal by bare PID."""
    if identity is None:
        raise RuntimeError("Missing owned process identity; manual reconciliation is required")
    remaining = max(0, deadline - time.monotonic())
    for sig, fraction in ((signal.SIGINT, 2 / 3), (signal.SIGTERM, 1 / 6), (signal.SIGKILL, 1 / 6)):
        members = group_members(identity)
        if not members:
            break
        os.killpg(identity["pgid"], sig)
        end = min(deadline, time.monotonic() + remaining * fraction)
        while time.monotonic() < end and group_members(identity):
            if process is not None:
                process.poll()
            time.sleep(min(0.05, max(0, end - time.monotonic())))
    if process is not None:
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    if group_members(identity):
        raise RuntimeError("Owned process group did not terminate within the cleanup budget")


# Tracking calls can hang in transports. A separate process, rather than a timeout
# around a thread, allows the supervisor to enforce a real upper bound.
TRACKING_CLEANUP = '''
import json, sys
import mlflow
record = json.loads(sys.stdin.read())
mlflow.set_tracking_uri(record["tracking_uri"])
mlflow.set_workspace(record["workspace"])
client = mlflow.tracking.MlflowClient()
run_id = record["run_id"]
status = client.get_run(run_id).info.status
if status == "RUNNING":
    client.set_terminated(run_id, status="KILLED")
    status = client.get_run(run_id).info.status
if status not in ("FINISHED", "FAILED", "KILLED"):
    raise RuntimeError("Run finalization is unconfirmed")
print(json.dumps({"run_id": run_id, "status": status}))
'''


def cleanup_tracking(record, context, seconds):
    """Finalize only the saved run, at its saved server/workspace, in a bounded child."""
    receipt = read_json(record["status_path"])
    tracking = context["base"]["tracking"]
    if (receipt.get("configuration_sha256") != record["configuration_sha256"]
            or receipt.get("tracking_uri") != tracking["uri"] or receipt.get("workspace") != tracking["workspace"]
            or not receipt.get("run_id")):
        raise RuntimeError("Missing or mismatched saved run ID; manual tracking reconciliation is required")
    saved = {key: receipt[key] for key in ("tracking_uri", "workspace", "run_id")}
    directory = Path(record["directory"])
    deadline = time.monotonic() + max(0.01, seconds)
    reap_reserve = min(2, max(0.01, seconds) / 4)
    with (directory / "cleanup.stdout").open("wb") as out, (directory / "cleanup.stderr").open("wb") as err:
        child = subprocess.Popen([sys.executable, "-B", "-c", TRACKING_CLEANUP], stdin=subprocess.PIPE,
                                 stdout=out, stderr=err, start_new_session=True)
        try:
            child.communicate(canonical(saved), timeout=max(0.001, deadline - time.monotonic() - reap_reserve))
        except subprocess.TimeoutExpired as error:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=max(0.001, deadline - time.monotonic()))
            raise RuntimeError("Tracking cleanup timed out; needs_cleanup") from error
    if child.returncode != 0:
        raise RuntimeError("Tracking finalization failed; retain cleanup logs and reconcile before resuming")
    result = read_json(directory / "cleanup.stdout")
    if result.get("run_id") != saved["run_id"] or result.get("status") not in ("FINISHED", "FAILED", "KILLED"):
        raise RuntimeError("Tracking cleanup did not confirm the saved run is terminal")
    return result


@contextmanager
def termination_signals():
    """Request normal-flow shutdown; preserve inherited ignores and restore handlers."""
    requested = {"signal": None}
    previous = {}

    def request_stop(signum, frame):
        # Never interrupt Popen, atomic publication, or cleanup with an exception.
        # Repeated signals retain the first reason and let bounded cleanup finish.
        if requested["signal"] is None:
            requested["signal"] = signal.Signals(signum).name

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handler = signal.getsignal(signum)
            if handler != signal.SIG_IGN:
                previous[signum] = handler
                signal.signal(signum, request_stop)
        yield requested
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def supervise(context, record, seconds, *, termination=None):
    """Run exactly one existing runner, keeping a durable exit separate from its receipt."""
    directory = Path(record["directory"])
    argv = [sys.executable, "-B", str(runner.EXPERIMENT_DIR / "run_experiment.py"),
            "--config", record["configuration_path"], "--status-file", record["status_path"],
            "--provenance-config", str(context["path"]), "--provenance-config", str(context["base_path"])]
    record["argv"] = argv
    process = None
    deadline = time.monotonic() + seconds
    interrupted = None
    shutdown = None
    try:
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            process = subprocess.Popen(argv, cwd=runner.REPOSITORY_ROOT, stdout=out, stderr=err,
                                       start_new_session=True, env={**os.environ,
                                       "PYTHONPATH": str(runner.REPOSITORY_ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
            record.update(process=process_identity(process.pid), outcome="running")
            atomic_json(directory / "attempt.json", record)
            while True:
                if termination is not None and termination["signal"] is not None:
                    record["termination_signal"] = termination["signal"]
                    interrupted = "interrupted"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    interrupted = "timeout"
                    break
                try:
                    record["exit_code"] = process.wait(timeout=min(.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
                except KeyboardInterrupt:
                    interrupted = "interrupted"
                    break
            # A stop request may arrive just as wait returns a successful exit.
            if termination is not None and termination["signal"] is not None:
                record["termination_signal"] = termination["signal"]
                interrupted = "interrupted"
            # A runner that exits but leaves children computing is not a completed attempt.
            if interrupted or group_members(record["process"]):
                interrupted = interrupted or "orphaned_child"
                shutdown = time.monotonic() + context["definition"]["limits"]["shutdown_seconds"]
                terminate_group(record["process"], process, shutdown - min(30, context["definition"]["limits"]["shutdown_seconds"] / 4))
                record["exit_code"] = process.returncode
                record["interruption"] = interrupted
                atomic_json(directory / "attempt.json", record)
                try:
                    record["cleanup"] = cleanup_tracking(record, context, max(0.01, shutdown - time.monotonic()))
                except Exception as error:
                    record.update(needs_cleanup=True, cleanup_error=f"{type(error).__name__}: {error}")
            record["outcome"] = "exited"
    except BaseException as error:
        record.update(outcome="supervision_failed", needs_cleanup=True,
                      supervisor_error=f"{type(error).__name__}: {error}")
        # Even an unexpected diagnostic failure must not leave our owned child computing.
        if process is not None and process.poll() is None:
            try:
                cleanup_deadline = min(shutdown or time.monotonic() + 5, time.monotonic() + 5)
                terminate_group(record["process"], process, cleanup_deadline)
                record["exit_code"] = process.returncode
            except Exception as cleanup_error:
                record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        atomic_json(directory / "attempt.json", record)
        raise
    record["exited_at"] = time.time()
    atomic_json(directory / "attempt.json", record)
    return record


def validate_input_provenance(context, contract, record, output):
    """Check the same recorded inputs for successful results and candidate-memory failures."""
    path = Path(record["configuration_path"])
    effective, _ = runner.load_config(path)
    if ((output / "original_config.yaml").read_bytes() != path.read_bytes()
            or read_json(output / "effective_config.json") != effective):
        raise RuntimeError("Saved configuration differs from the requested configuration")
    run = read_json(output / "run.json")
    provenance = read_json(output / "provenance.json")
    source = read_json(output / "source.json")
    for record_with_hashes in (run, provenance):
        if (record_with_hashes.get("original_config_file_sha256") != record["configuration_sha256"]
                or record_with_hashes.get("effective_config_file_sha256") != runner.file_sha256(output / "effective_config.json")):
            raise RuntimeError("Configuration provenance hashes disagree")
    if (provenance.get("manifest_sha256") != contract["manifest_sha256"]
            or runner.file_sha256(output / "evaluation_manifest.json") != contract["manifest_sha256"]
            or provenance.get("weights") != contract["weights"] or provenance.get("cache") != contract["cache"]
            or provenance.get("seeds") != {"runtime": effective["runtime"]["seed"], "timing": effective["timing"]["seed"]}):
        raise RuntimeError("Input, cache, or seed provenance differs from study inputs")
    environment = provenance.get("environment", {})
    if environment.get("cpu") is None or environment.get("cuda_available") is None:
        raise RuntimeError("Missing runtime environment provenance")
    for package, version in environment.get("packages", {}).items():
        if version is not None and contract["packages"].get(re.sub(r"[-_.]+", "-", package).lower()) != version:
            raise RuntimeError(f"Runtime package differs from compatible environment: {package}")
    if Path(environment.get("tltorch_path", "")).resolve() != runner.REPOSITORY_ROOT / "tltorch/__init__.py":
        raise RuntimeError("Runtime library imports did not resolve to this checkout")
    if environment.get("repository_revision") != source.get("commit"):
        raise RuntimeError("Runtime environment revision differs from source snapshot")
    hashes = source.get("source_files_sha256", {})
    config_name = path.relative_to(runner.REPOSITORY_ROOT).as_posix()
    expected_hashes = {**contract["source_files_sha256"], config_name: record["configuration_sha256"]}
    normalized_hashes = {**{n: None for n, value in expected_hashes.items() if value is None}, **hashes}
    if normalized_hashes != expected_hashes or source.get("execution_inputs_match_commit") is not True:
        raise RuntimeError("Recorded source scope differs from compatible execution inputs")
    if provenance.get("git") != source:
        raise RuntimeError("Source commit and provenance disagree")
    for name, expected in hashes.items():
        try:
            committed = digest(inputs.git(runner.REPOSITORY_ROOT, "show", f"{source['commit']}:{name}"))
        except subprocess.CalledProcessError:
            committed = None
        if committed != expected:
            raise RuntimeError(f"Source record differs from its committed bytes: {name}")
    if runner.file_sha256(output / "source.zip") != source.get("source_archive_sha256"):
        raise RuntimeError("Source archive hash disagrees with its provenance")
    with zipfile.ZipFile(output / "source.zip") as archive:
        expected_entries = {name for name, value in hashes.items() if value is not None}
        if set(archive.namelist()) != expected_entries or len(archive.namelist()) != len(expected_entries):
            raise RuntimeError("Source archive has missing, duplicate, or unexpected entries")
        if any(digest(archive.read(name)) != hashes[name] for name in expected_entries):
            raise RuntimeError("Archived source bytes disagree with committed source hashes")


def collect_result(context, contract, record):
    """Accept only a zero exit plus a complete, consistent chain of existing evidence."""
    if record.get("exit_code") != 0 or record.get("interruption") or record.get("needs_cleanup"):
        raise RuntimeError("No confirmed successful child exit")
    receipt = read_json(record["status_path"])
    path = Path(record["configuration_path"])
    tracking = context["base"]["tracking"]
    if (receipt.get("schema_version") != 1 or receipt.get("state") != "succeeded" or receipt.get("phase") != "completed"
            or receipt.get("configuration_path") != str(path)
            or receipt.get("configuration_sha256") != record["configuration_sha256"]
            or digest(path.read_bytes()) != record["configuration_sha256"]
            or receipt.get("tracking_uri") != tracking["uri"] or receipt.get("workspace") != tracking["workspace"]):
        raise RuntimeError("Stale, partial, or mismatched runner receipt")
    identity = record.get("process")
    if not identity or receipt.get("process", {}).get("pid") != identity["pid"] or receipt["process"].get("hostname") != identity["hostname"]:
        raise RuntimeError("Receipt process does not match the owned child")
    output = Path(receipt["output_directory"])
    if not output.is_absolute() or not output.resolve().is_relative_to(Path(context["base"]["output_root"])):
        raise RuntimeError("Output is outside the configured output root")
    if (output / "failure.json").exists():
        raise RuntimeError("Failure record exists despite successful receipt")
    names = ("measurements.json", "results.csv", "run.json", "model_verification.json",
             "mlflow_model_verification.json", "execution_inputs_verification.json", "source.json",
             "provenance.json", "original_config.yaml", "effective_config.json", "evaluation_manifest.json")
    for name in names:
        reference = receipt.get("artifacts", {}).get(name, {})
        if reference.get("path") != str(output / name) or reference.get("sha256") != runner.file_sha256(output / name):
            raise RuntimeError(f"Missing or changed result artifact: {name}")
    if (output / "original_config.yaml").read_bytes() != path.read_bytes():
        raise RuntimeError("Saved original configuration differs from the requested configuration")
    effective, _ = runner.load_config(path)
    if read_json(output / "effective_config.json") != effective:
        raise RuntimeError("Saved effective configuration differs from the requested configuration")
    run = read_json(output / "run.json")
    local = read_json(output / "model_verification.json")
    remote = read_json(output / "mlflow_model_verification.json")
    source = read_json(output / "source.json")
    verified = read_json(output / "execution_inputs_verification.json")
    for name in ("experiment_id", "run_id"):
        if not receipt.get(name) or run.get(name) != receipt[name]:
            raise RuntimeError(f"Inconsistent {name}")
    if run.get("workspace") != tracking["workspace"]:
        raise RuntimeError("Run workspace differs from the requested workspace")
    for name in ("model_id", "model_uri"):
        if not receipt.get(name) or run.get("mlflow_model", {}).get(name) != receipt[name] or remote.get(name) != receipt[name]:
            raise RuntimeError(f"Inconsistent {name}")
    if local.get("passed") is not True or remote.get("passed") is not True or verified.get("matches_recorded_commit") is not True:
        raise RuntimeError("Model or execution-input verification did not pass")
    for filename, expected in (("model.pt", local.get("checkpoint_sha256")),
                               ("verification_input.pt", local.get("fixed_input_sha256")),
                               ("source.zip", source.get("source_archive_sha256"))):
        if expected is None or runner.file_sha256(output / filename) != expected:
            raise RuntimeError(f"Changed verified artifact: {filename}")
    if verified.get("commit") != source.get("commit"):
        raise RuntimeError("Source commit and verification disagree")
    validate_input_provenance(context, contract, record, output)
    measurements = read_json(output / "measurements.json")
    if measurements.get("variant") != ("compressed" if record["layers"] else "dense"):
        raise RuntimeError("Measurement variant differs from the request")
    top1, count = measurements["accuracy"]["top1_percent"], measurements["storage"]["parameter_count"]
    if type(top1) not in (int, float) or not math.isfinite(top1) or not 0 <= top1 <= 100 or type(count) is not int or count <= 0:
        raise RuntimeError("Invalid measured objectives")
    if record["layers"]:
        report = read_json(output / "compression_report.json")
        if report.get("model", {}).get("compressed") != measurements["storage"]:
            raise RuntimeError("Measured whole-model storage disagrees with the compression report")
    return {"values": [top1, count], "output_directory": str(output),
            "run_id": receipt["run_id"], "model_id": receipt["model_id"], "model_uri": receipt["model_uri"],
            "source_commit": source["commit"], "receipt_path": record["status_path"],
            "receipt_sha256": runner.file_sha256(Path(record["status_path"])),
            "trial_number": record["trial_number"], "attempt_path": str(Path(record["directory"]) / "attempt.json")}


def rank_signature(layers):
    """Shape-only diagnostics for the pilot geometry, using the library's own validators."""
    from tensorly.cp_tensor import validate_cp_rank
    from tensorly.tucker_tensor import validate_tucker_rank
    from tltorch.factorized_tensors.tt_rank import resolve_tt_rank
    from tltorch.factorized_tensors.tensorized_matrices import validate_block_tt_rank
    signature = {}
    for name, options in layers.items():
        # Geometry is known for these pilot layers. Other enabled layers get no inferred signature.
        if name not in ("layer4.0.conv2", "layer4.1.conv2", "fc"):
            return None
        method, rank = options["factorization"], options["rank"]
        shape = (10, 10, 10, 8, 8, 8) if name == "fc" else (512, 512, 3, 3)
        if method == "CP":
            actual = validate_cp_rank(shape, rank)
        elif method == "Tucker":
            actual = validate_tucker_rank(shape, rank, fixed_modes=None if name == "fc" else [2, 3])
        elif method == "TT":
            actual = resolve_tt_rank((512, 3, 3, 512), rank)
        else:
            actual = validate_block_tt_rank(((10, 10, 10), (8, 8, 8)), rank)
        signature[name] = {"method": method, "rank": int(actual) if method == "CP" else [int(v) for v in actual]}
    return signature


def confirm_candidate_oom(context, contract, record, receipt):
    """Require a dead owned child, terminal tracking, and intact inputs before continuing."""
    if (record.get("interruption") or record.get("cleanup_error") or record.get("supervisor_error")
            or type(record.get("exit_code")) is not int or record["exit_code"] == 0):
        raise RuntimeError("Candidate OOM has no uninterrupted nonzero exit")
    saved = read_json(Path(record["directory"]) / "attempt.json")
    keys = ("trial_number", "directory", "configuration_path", "configuration_sha256", "status_path",
            "layers", "exit_code", "process")
    if (saved.get("outcome") != "exited" or saved.get("interruption") or saved.get("needs_cleanup")
            or any(saved.get(key) != record.get(key) for key in keys)):
        raise RuntimeError("Candidate OOM lacks consistent durable child exit evidence")
    identity = record.get("process")
    if not identity or group_members(identity):
        raise RuntimeError("Owned child identity is missing or its process group remains alive")
    path = Path(record["configuration_path"])
    tracking = context["base"]["tracking"]
    if (receipt.get("schema_version") != 1 or receipt.get("state") != "failed"
            or receipt.get("configuration_path") != str(path)
            or receipt.get("configuration_sha256") != record["configuration_sha256"]
            or digest(path.read_bytes()) != record["configuration_sha256"]
            or receipt.get("tracking_uri") != tracking["uri"] or receipt.get("workspace") != tracking["workspace"]
            or receipt.get("process", {}).get("pid") != identity["pid"]
            or receipt.get("process", {}).get("hostname") != identity["hostname"]):
        raise RuntimeError("Failed receipt does not match the selected configuration and owned child")
    effective, _ = runner.load_config(path)
    compression = effective["compression"]
    if not record["layers"] or not compression or compression["layers"] != record["layers"]:
        raise RuntimeError("Candidate OOM requires the requested compressed model")
    cleanup = record.get("cleanup") or {}
    if (not receipt.get("run_id") or cleanup.get("run_id") != receipt["run_id"]
            or cleanup.get("status") not in ("FAILED", "FINISHED", "KILLED")):
        raise RuntimeError("Tracking cleanup did not confirm the saved run is terminal")
    output = Path(receipt["output_directory"])
    if not output.is_absolute() or not output.resolve().is_relative_to(Path(context["base"]["output_root"])):
        raise RuntimeError("Output is outside the configured output root")
    run = read_json(output / "run.json")
    failure = read_json(output / "failure.json")
    if (not receipt.get("experiment_id") or run.get("experiment_id") != receipt["experiment_id"]
            or run.get("run_id") != receipt["run_id"] or run.get("workspace") != tracking["workspace"]
            or failure.get("run_id") != receipt["run_id"] or failure.get("error_type") != "OutOfMemoryError"
            or "execution_input_error" not in failure or failure["execution_input_error"] is not None):
        raise RuntimeError("Failure/run evidence disagrees or execution-input verification failed")
    validate_input_provenance(context, contract, record, output)
    source = read_json(output / "source.json")
    runner.verify_execution_inputs(runner.REPOSITORY_ROOT, runner.EXPERIMENT_DIR, path, source)
    if canonical(compatibility(context)) != canonical(contract):
        raise RuntimeError("Execution context changed during the failed candidate")


def classify_failure(record, context, contract):
    """Only recognized fitting failures can consume a candidate and permit another launch."""
    if record.get("needs_cleanup"):
        return "infrastructure"
    try:
        receipt = read_json(record["status_path"])
    except (OSError, ValueError):
        return "infrastructure"
    phase = receipt.get("phase")
    failure = receipt.get("failure") or {}
    types = [failure.get("exception_type"), *failure.get("cause_types", [])]
    fitting_types = ("torch.OutOfMemoryError", "torch.cuda.OutOfMemoryError", "torch._C._LinAlgError",
                     "numpy.linalg.LinAlgError", "torch.linalg.LinAlgError")
    if phase == "conversion" and (record.get("interruption") == "timeout" or any(t in fitting_types for t in types)):
        return "candidate"
    if phase in ("evaluation", "timing") and failure.get("exception_type") in fitting_types[:2]:
        try:
            confirm_candidate_oom(context, contract, record, receipt)
        except Exception as error:
            record["candidate_failure_rejection"] = f"{type(error).__name__}: {error}"
        else:
            return "candidate"
    return "infrastructure"


def complete_attempt(context, contract, study, trial, record, metadata):
    """Tell measured objectives once, or fail without substituting a fabricated score."""
    import optuna
    directory = Path(record["directory"])
    head = check_lineage(context, metadata)
    if record.get("exit_code") == 0 and not record.get("interruption"):
        result = collect_result(context, contract, record)
        record.update(outcome="executed", result=result)
        trial.set_user_attr("result", result)
        study.tell(trial.number, result["values"])
        metadata["consecutive_candidate_failures"] = 0
    else:
        if not record.get("cleanup") and not record.get("needs_cleanup"):
            try:
                record["cleanup"] = cleanup_tracking(record, context, min(30, context["definition"]["limits"]["shutdown_seconds"]))
            except Exception as error:
                record.update(needs_cleanup=True, cleanup_error=f"{type(error).__name__}: {error}")
        try:
            failed_receipt = read_json(record["status_path"])
            record["runner_failure"] = {key: failed_receipt.get(key) for key in ("phase", "failure")}
        except (OSError, ValueError):
            record["runner_failure"] = None
        category = classify_failure(record, context, contract)
        record.update(outcome="failed", failure_category=category)
        trial.set_user_attr("failure", {key: record.get(key) for key in (
            "failure_category", "interruption", "exit_code", "needs_cleanup", "cleanup_error")})
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        metadata["consecutive_candidate_failures"] += 1
        if category != "candidate" or record.get("interruption") == "interrupted":
            metadata["stopped_reason"] = "Shared, unknown, or interrupted failure; explicit reconciliation required"
    metadata["last_execution_head"] = head
    metadata["has_execution_snapshot"] = metadata["last_execution_head"] != metadata["initial_head"]
    atomic_json(directory / "attempt.json", record)
    atomic_json(context["state_dir"] / "study.json", metadata)
    save_sampler(study, context["state_dir"])


def reconcile(context, contract, study, metadata):
    """Never adopt a receipt alone or restart a live/reused/unknown process identity."""
    import optuna
    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        attempt_path = trial.user_attrs.get("attempt_path")
        if not attempt_path:
            raise RuntimeError("RUNNING trial has no durable attempt; reconcile before continuing")
        record = read_json(attempt_path)
        identity = record.get("process")
        if identity and group_members(identity):
            raise RuntimeError("Owned child is still alive; do not start another controller worker")
        if record.get("exit_code") is None or not identity:
            raise RuntimeError("No durable child exit/process identity; manual reconciliation required")
        if record.get("needs_cleanup"):
            raise RuntimeError("Attempt needs_cleanup; resolve it explicitly before resuming")
        complete_attempt(context, contract, study, optuna.trial.Trial(study, trial._trial_id), record, metadata)


def run_controller(context):
    """The study loop chooses, supervises, and collects; the runner performs all computation."""
    import optuna
    if os.environ.get("MLFLOW_RUN_ID"):
        raise ValueError("Unset MLFLOW_RUN_ID: each child must create its own run")
    with termination_signals() as termination, exclusive_locks(context):
        contract = compatibility(context)
        study, metadata = open_study(context, contract)
        reconcile(context, contract, study, metadata)
        limits = context["definition"]["limits"]
        while True:
            if termination["signal"] is not None:
                raise KeyboardInterrupt(f"Controller received {termination['signal']}")
            if metadata.get("stopped_reason"):
                raise RuntimeError(metadata["stopped_reason"])
            trials = study.get_trials(deepcopy=False)
            asked = sum(t.state != optuna.trial.TrialState.WAITING for t in trials)
            if (asked >= limits["total_attempts"] or time.time() >= metadata["deadline"]
                    or metadata["consecutive_candidate_failures"] >= limits["consecutive_candidate_failures"]
                    or not any(t.state == optuna.trial.TrialState.WAITING for t in trials)):
                break
            if canonical(compatibility(context)) != canonical(contract):
                raise RuntimeError("Execution context changed before launch")
            check_lineage(context, metadata)
            trial = study.ask()
            layers = suggest_layers(trial, context["definition"])
            request = digest(canonical({"layers": list(layers.items()), "compatibility": digest(canonical(contract))}))
            trial.set_user_attr("request_sha256", request)
            record = reserve_attempt(context, trial, layers)
            # Persist the full suggestion state before launching a child.
            save_sampler(study, context["state_dir"])
            previous = [t for t in trials if t.user_attrs.get("request_sha256") == request and t.state == optuna.trial.TrialState.COMPLETE]
            if previous:
                original = previous[0].user_attrs.get("result")
                original_record = read_json(original["attempt_path"])
                result = collect_result(context, contract, original_record)
                if result != original:
                    raise RuntimeError("Previously successful evidence changed; refuse exact-request reuse")
                record.update(outcome="reused", reused_from=result)
                trial.set_user_attr("reused_from", result)
                trial.set_user_attr("result", result)
                study.tell(trial, result["values"])
                metadata["consecutive_candidate_failures"] = 0
                atomic_json(Path(record["directory"]) / "attempt.json", record)
                atomic_json(context["state_dir"] / "study.json", metadata)
                save_sampler(study, context["state_dir"])
                continue
            if any(t.user_attrs.get("request_sha256") == request and t.state == optuna.trial.TrialState.FAIL for t in trials):
                record.update(outcome="duplicate_failed", reason="No automatic retry of a failed request")
                trial.set_user_attr("failure", record["reason"])
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                atomic_json(Path(record["directory"]) / "attempt.json", record)
                metadata["stopped_reason"] = record["reason"]
                atomic_json(context["state_dir"] / "study.json", metadata)
                save_sampler(study, context["state_dir"])
                raise RuntimeError(record["reason"])
            signature = rank_signature(layers)
            trial.set_user_attr("rank_signature", signature)
            if signature and any(t.user_attrs.get("rank_signature") == signature for t in trials):
                warning = "An earlier request has the same validated ranks; this is a warning, not result reuse"
                record["rank_warning"] = warning
                print(warning, file=sys.stderr)
            save_sampler(study, context["state_dir"])
            seconds = min(limits["per_trial_seconds"], metadata["deadline"] - time.time())
            if seconds <= 0:
                record.update(outcome="deadline_before_launch", exit_code=None)
                atomic_json(Path(record["directory"]) / "attempt.json", record)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                save_sampler(study, context["state_dir"])
                break
            if termination["signal"] is not None:
                raise KeyboardInterrupt(f"Controller received {termination['signal']}")
            record = supervise(context, record, seconds, termination=termination)
            try:
                complete_attempt(context, contract, study, trial, record, metadata)
            except Exception as error:
                # Invalid evidence is a shared interface failure, never poor model accuracy.
                record.update(outcome="needs_reconciliation", error=f"{type(error).__name__}: {error}")
                atomic_json(Path(record["directory"]) / "attempt.json", record)
                if study.get_trials(deepcopy=False)[trial.number].state == optuna.trial.TrialState.RUNNING:
                    trial.set_user_attr("failure", record["error"])
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                metadata["stopped_reason"] = record["error"]
                atomic_json(context["state_dir"] / "study.json", metadata)
                save_sampler(study, context["state_dir"])
                raise
        trials = study.get_trials(deepcopy=False)
        summary = {"asked": sum(t.state != optuna.trial.TrialState.WAITING for t in trials),
                   "newly_executed": [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE and "reused_from" not in t.user_attrs],
                   "reused": [t.number for t in trials if "reused_from" in t.user_attrs],
                   "failed": [t.number for t in trials if t.state == optuna.trial.TrialState.FAIL],
                   "pareto_trials": [t.number for t in study.best_trials], "deadline": metadata["deadline"]}
        atomic_json(context["state_dir"] / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue existing compatible state within its original budget")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        context = load_study(args.study)
        if args.dry_run:
            print(json.dumps({"study": context["definition"], "base_config": str(context["base_path"]),
                              "state_dir": str(context["state_dir"]), "trial_config_dir": str(context["trial_dir"]),
                              "note": "Validation only. No database, trial YAML, receipt, commit, tracking request, model, or download."}, indent=2))
        else:
            exists = (context["state_dir"] / "study.json").exists()
            if exists and not args.resume:
                raise ValueError("Study already exists; use --resume to retain its original budget and deadline")
            if args.resume and not exists:
                raise ValueError("--resume requires an existing study; refusing to create a replacement")
            print(json.dumps(run_controller(context), indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"Controller stopped ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
