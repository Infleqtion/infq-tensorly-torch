"""Offline controller examples: no pretrained data, shared tracking, or real experiments."""

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

optuna = pytest.importorskip("optuna")
import yaml

import execution_inputs as inputs
import optuna_controller as controller
import run_experiment as runner
import runner_status


@pytest.fixture
def study_context(execution_repo, tmp_path, monkeypatch):
    root, experiment = execution_repo
    source = runner.EXPERIMENT_DIR
    definition = yaml.safe_load((source / "configs/studies/resnet18_pilot.yaml").read_text())
    base = yaml.safe_load((source / "configs/resnet18_exploration_dense.yaml").read_text())
    base["runtime"]["device"] = "cpu"
    (experiment / "configs/base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    definition["base_config"] = "configs/base.yaml"
    definition["state_dir"] = str(tmp_path / "state")
    study_path = experiment / "configs/study.yaml"
    study_path.write_text(yaml.safe_dump(definition, sort_keys=False))
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(runner, "EXPERIMENT_DIR", experiment)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    return controller.load_study(study_path)


def contract_for(context):
    return {"source_files_sha256": controller.execution_identity(context),
            "definition": context["definition"], "fixed_config": context["base"],
            "manifest_sha256": controller.digest(b'{"images": []}\n'),
            "weights": {"sha256": "synthetic-weight-hash"}, "cache": {"torch_home": "synthetic"},
            "packages": {}, "machine": "offline-fixture"}


def fake_success(context, record, seconds):
    """Make the real existing artifact/receipt protocol, without model computation."""
    config_path = Path(record["configuration_path"])
    config, original = runner.load_config(config_path)
    output = Path(config["output_root"]) / Path(record["directory"]).name
    output.mkdir(parents=True)
    (output / "original_config.yaml").write_bytes(original)
    runner.write_json(output / "effective_config.json", config)
    (output / "evaluation_manifest.json").write_bytes(b'{"images": []}\n')
    extras = {p.relative_to(runner.REPOSITORY_ROOT).as_posix(): p.read_bytes()
              for p in (context["path"], context["base_path"])}
    snapshot = inputs.snapshot_execution_inputs(runner.REPOSITORY_ROOT, runner.EXPERIMENT_DIR,
                                               config_path, original, "offline-only", provenance_configs=extras)
    revision = runner.save_source(output, snapshot)
    hashes = {"original_config_file_sha256": controller.digest(original),
              "effective_config_file_sha256": runner.file_sha256(output / "effective_config.json")}
    contract = contract_for(context)
    run = {"experiment_id": "synthetic-experiment", "run_id": f"synthetic-run-{record['trial_number']}",
           "workspace": config["tracking"]["workspace"], **hashes,
           "mlflow_model": {"model_id": "synthetic-model", "model_uri": "models:/synthetic"}}
    runner.write_json(output / "run.json", run)
    runner.write_json(output / "provenance.json", {**hashes, "git": revision,
        "manifest_sha256": contract["manifest_sha256"], "weights": contract["weights"], "cache": contract["cache"],
        "seeds": {"runtime": config["runtime"]["seed"], "timing": config["timing"]["seed"]},
        "environment": {"cpu": "synthetic", "cuda_available": False, "packages": {},
                        "repository_revision": revision["commit"],
                        "tltorch_path": str(runner.REPOSITORY_ROOT / "tltorch/__init__.py")}})
    (output / "model.pt").write_bytes(b"not a model; never loaded")
    (output / "verification_input.pt").write_bytes(b"synthetic verification bytes")
    runner.write_json(output / "model_verification.json", {"passed": True,
        "checkpoint_sha256": runner.file_sha256(output / "model.pt"),
        "fixed_input_sha256": runner.file_sha256(output / "verification_input.pt")})
    runner.write_json(output / "mlflow_model_verification.json", {"passed": True, **run["mlflow_model"]})
    runner.write_json(output / "execution_inputs_verification.json", {"matches_recorded_commit": True,
                                                                  "commit": revision["commit"]})
    measurement = {"variant": "compressed" if record["layers"] else "dense",
                   "accuracy": {"top1_percent": 0.0}, "storage": {"parameter_count": 12345}}
    runner.write_json(output / "measurements.json", measurement)
    if record["layers"]:
        runner.write_json(output / "compression_report.json", {"model": {"compressed": measurement["storage"]}})
    (output / "results.csv").write_text("deliberately not used\n")
    receipt = runner_status.start_status(Path(record["status_path"]), config_path, original, config["tracking"])
    runner_status.finish_status(receipt, output)
    record.update(exit_code=0, process=controller.process_identity(os.getpid()), outcome="exited")
    controller.atomic_json(Path(record["directory"]) / "attempt.json", record)
    return record


def create_pending(context):
    context["state_dir"].mkdir(mode=0o700)
    contract = contract_for(context)
    study, metadata = controller.open_study(context, contract)
    trial = study.ask()
    layers = controller.suggest_layers(trial, context["definition"])
    record = controller.reserve_attempt(context, trial, layers)
    controller.save_sampler(study, context["state_dir"])
    return contract, study, metadata, trial, record


def test_eight_pilot_configs_keep_fixed_science_and_explicit_conditional_parameters(study_context):
    context, definition = study_context, study_context["definition"]
    class FixedTrial:
        def __init__(self, params):
            self.params, self.seen = params, []
        def suggest_categorical(self, name, choices):
            self.seen.append(name)
            assert self.params[name] in choices
            return self.params[name]
    rendered = []
    for case in definition["pilot_cases"]:
        params = controller.pilot_parameters(definition, case)
        trial = FixedTrial(params)
        layers = controller.suggest_layers(trial, definition)
        assert set(trial.seen) == set(params)
        assert [n for n in layers] == [n for n in definition["enabled_layers"] if n == case["layer"]]
        config = yaml.safe_load(controller.render_config(context, layers, "pilot-test"))
        rendered.append(config)
        for name in ("data", "model", "weights", "cache", "runtime", "timing", "output_root"):
            assert config[name] == context["base_raw"][name]
        assert config["tracking"] == {**context["base_raw"]["tracking"], "run_name": "pilot-test"}
        if not layers:
            assert config["compression"] is None
        for name, options in layers.items():
            assert type(options["rank"]) is float
            assert ("fixed_rank_modes" in options) == (name != "fc" and options["factorization"] == "Tucker")
            assert ("decomposition_kwargs" in options) == (options["factorization"] in ("CP", "Tucker"))
            if name == "fc":
                assert options["in_tensorized_features"] == [8, 8, 8]
                assert options["out_tensorized_features"] == [10, 10, 10]
    assert len(rendered) == 8
    assert controller.canonical({"rank": 1}) != controller.canonical({"rank": 1.0})


def test_rendered_options_agree_with_existing_public_validator(study_context):
    from tltorch.utils.model_compression import _normalize_decomposition_kwargs
    definition = study_context["definition"]
    for method in ("CP", "Tucker"):
        assert _normalize_decomposition_kwargs(method, definition["fitting"][method]) == definition["fitting"][method]
    # Shape-only diagnostics use the existing implementations, with no model constructor.
    assert controller.rank_signature({"fc": {"factorization": "Tucker", "rank": .75}}) == controller.rank_signature({"fc": {"factorization": "Tucker", "rank": .9}})
    assert controller.rank_signature({"fc": {"factorization": "CP", "rank": .5}})["fc"]["rank"] == 4741
    assert controller.rank_signature({"layer4.0.conv2": {"factorization": "TT", "rank": .75}})


def test_full_pilot_exhausts_queue_without_fallback_and_resumes_without_new_budget(study_context, monkeypatch):
    context = study_context
    monkeypatch.setattr(controller, "compatibility", contract_for)
    calls = []
    def execute(context, record, seconds, *, termination=None):
        calls.append(record["trial_number"])
        assert 0 < seconds <= 1200
        return fake_success(context, record, seconds)
    monkeypatch.setattr(controller, "supervise", execute)
    before = inputs.git(runner.REPOSITORY_ROOT, "show", "HEAD:unrelated.txt")
    (runner.REPOSITORY_ROOT / "unrelated.txt").write_text("staged unrelated\n")
    inputs.git(runner.REPOSITORY_ROOT, "add", "unrelated.txt")
    (runner.REPOSITORY_ROOT / "unrelated.txt").write_text("working unrelated\n")
    staged = inputs.git(runner.REPOSITORY_ROOT, "show", ":unrelated.txt")
    result = controller.run_controller(context)
    assert result["asked"] == 8 and result["newly_executed"] == list(range(8)) and not result["reused"]
    metadata = controller.read_json(context["state_dir"] / "study.json")
    assert controller.run_controller(context) == result
    assert calls == list(range(8))
    assert controller.read_json(context["state_dir"] / "study.json")["deadline"] == metadata["deadline"]
    assert inputs.git(runner.REPOSITORY_ROOT, "show", "HEAD:unrelated.txt") == before
    assert inputs.git(runner.REPOSITORY_ROOT, "show", ":unrelated.txt") == staged
    assert (runner.REPOSITORY_ROOT / "unrelated.txt").read_text() == "working unrelated\n"


def test_exact_reuse_consumes_attempt_links_original_and_does_not_execute(study_context, monkeypatch):
    context = study_context
    definition = context["definition"]
    definition["pilot_cases"] = [definition["pilot_cases"][0]] * 2
    monkeypatch.setattr(controller, "compatibility", contract_for)
    calls = []
    def execute(context, record, seconds, *, termination=None):
        calls.append(record["trial_number"])
        return fake_success(context, record, seconds)
    monkeypatch.setattr(controller, "supervise", execute)
    summary = controller.run_controller(context)
    assert calls == [0]
    assert summary["asked"] == 2 and summary["newly_executed"] == [0] and summary["reused"] == [1]
    study, _ = controller.open_study(context, contract_for(context))
    original, reused = study.trials
    assert reused.values == original.values == [0, 12345]
    assert reused.user_attrs["reused_from"] == original.user_attrs["result"]
    assert reused.user_attrs["result"]["run_id"] == "synthetic-run-0"
    assert not Path(controller.read_json(reused.user_attrs["attempt_path"])["status_path"]).exists()
    # Reuse is contingent on the original evidence still being intact.
    original_record = controller.read_json(original.user_attrs["attempt_path"])
    Path(original_record["status_path"]).write_text("{}")
    with pytest.raises(RuntimeError, match="receipt"):
        controller.collect_result(context, contract_for(context), original_record)


def test_collector_rejects_partial_or_mismatched_evidence_and_preserves_measured_zero(study_context):
    contract, _, _, _, record = create_pending(study_context)
    record = fake_success(study_context, record, 1)
    assert controller.collect_result(study_context, contract, record)["values"] == [0.0, 12345]
    receipt_path = Path(record["status_path"])
    receipt = receipt_path.read_bytes()
    for changes, match in (({"state": "running"}, "receipt"), ({"run_id": "wrong"}, "run_id"),
                           ({"configuration_sha256": "stale"}, "receipt")):
        controller.atomic_json(receipt_path, {**json.loads(receipt), **changes})
        with pytest.raises(RuntimeError, match=match):
            controller.collect_result(study_context, contract, record)
    receipt_path.write_bytes(receipt)
    with pytest.raises(RuntimeError, match="exit"):
        controller.collect_result(study_context, contract, {**record, "exit_code": None})
    with pytest.raises(RuntimeError, match="exit"):
        controller.collect_result(study_context, contract, {**record, "exit_code": 1})
    output = Path(json.loads(receipt)["output_directory"])
    (output / "failure.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Failure record"):
        controller.collect_result(study_context, contract, record)
    (output / "failure.json").unlink()
    (output / "measurements.json").write_text("{}")
    with pytest.raises(RuntimeError, match="measurements.json"):
        controller.collect_result(study_context, contract, record)


@pytest.mark.parametrize("filename,field,value", [
    ("model_verification.json", "passed", False),
    ("mlflow_model_verification.json", "passed", False),
    ("execution_inputs_verification.json", "matches_recorded_commit", False),
    ("provenance.json", "manifest_sha256", "changed-input"),
])
def test_self_consistent_receipt_cannot_hide_failed_verification_or_changed_inputs(study_context, filename, field, value):
    contract, _, _, _, record = create_pending(study_context)
    fake_success(study_context, record, 1)
    receipt = controller.read_json(record["status_path"])
    path = Path(receipt["output_directory"]) / filename
    data = controller.read_json(path)
    data[field] = value
    controller.atomic_json(path, data)
    receipt["artifacts"][filename]["sha256"] = runner.file_sha256(path)
    controller.atomic_json(Path(record["status_path"]), receipt)
    with pytest.raises(RuntimeError, match="verification|provenance"):
        controller.collect_result(study_context, contract, record)


def test_sampler_checkpoint_restores_rng_and_detects_history_mismatch(study_context):
    context = study_context
    contract, study, metadata, trial, _ = create_pending(context)
    study.tell(trial, [0, 12])
    controller.save_sampler(study, context["state_dir"])
    before = study.sampler._rng.rng.get_state()
    resumed, resumed_metadata = controller.open_study(context, contract)
    after = resumed.sampler._rng.rng.get_state()
    assert before[0] == after[0] and (before[1] == after[1]).all() and before[2:] == after[2:]
    assert resumed_metadata["deadline"] == metadata["deadline"]
    resumed.ask()
    with pytest.raises(RuntimeError, match="history disagree"):
        controller.open_study(context, contract)


def test_changed_source_input_and_lineage_stop_resume(study_context):
    context = study_context
    contract, _, metadata, _, _ = create_pending(context)
    (runner.EXPERIMENT_DIR / "run_experiment.py").write_text("# changed science\n")
    with pytest.raises(RuntimeError, match="incompatible resume"):
        controller.open_study(context, contract_for(context))
    changed = deepcopy(contract)
    changed["manifest_sha256"] = "different"
    with pytest.raises(RuntimeError, match="incompatible resume"):
        controller.open_study(context, changed)
    (runner.REPOSITORY_ROOT / "unrelated.txt").write_text("unrelated commit\n")
    inputs.git(runner.REPOSITORY_ROOT, "add", "unrelated.txt")
    inputs.git(runner.REPOSITORY_ROOT, "commit", "-m", "Unreviewed lineage change")
    with pytest.raises(RuntimeError, match="Unexpected changes"):
        controller.check_lineage(context, metadata)


def test_deadline_includes_downtime_and_waiting_cases_do_not_count_as_asked(study_context, monkeypatch):
    context = study_context
    context["state_dir"].mkdir(mode=0o700)
    contract = contract_for(context)
    study, metadata = controller.open_study(context, contract)
    metadata["deadline"] = time.time() - 1
    controller.atomic_json(context["state_dir"] / "study.json", metadata)
    monkeypatch.setattr(controller, "compatibility", contract_for)
    monkeypatch.setattr(controller, "supervise", lambda *args: pytest.fail("Expired study launched a child"))
    result = controller.run_controller(context)
    assert result["asked"] == 0 and len(study.trials) == 8
    assert result["deadline"] == metadata["deadline"]


def test_checkout_lock_is_shared_across_different_studies(study_context, tmp_path):
    first = study_context
    second = {**first, "state_dir": tmp_path / "different-study"}
    with controller.exclusive_locks(first):
        with pytest.raises(RuntimeError, match="Another controller"):
            with controller.exclusive_locks(second):
                pytest.fail("Two studies acquired the same checkout")
    with controller.exclusive_locks(second):
        pass


def test_reconcile_requires_durable_exit_not_just_success_receipt(study_context, monkeypatch):
    context = study_context
    contract, study, metadata, trial, record = create_pending(context)
    fake_success(context, record, 1)
    # The real child is gone; the fixture wrote its records from this test process.
    monkeypatch.setattr(controller, "group_members", lambda identity: [])
    saved = deepcopy(record)
    record["exit_code"] = None
    controller.atomic_json(Path(record["directory"]) / "attempt.json", record)
    with pytest.raises(RuntimeError, match="durable child exit"):
        controller.reconcile(context, contract, study, metadata)
    controller.atomic_json(Path(record["directory"]) / "attempt.json", saved)
    controller.reconcile(context, contract, study, metadata)
    assert study.trials[trial.number].state == optuna.trial.TrialState.COMPLETE
    controller.reconcile(context, contract, study, metadata)
    assert study.trials[trial.number].values == [0, 12345]


def test_resume_does_not_trust_live_or_reused_pid(study_context, monkeypatch):
    contract, study, metadata, _, record = create_pending(study_context)
    record["process"] = controller.process_identity(os.getpid())
    controller.atomic_json(Path(record["directory"]) / "attempt.json", record)
    with pytest.raises(RuntimeError, match="still alive"):
        controller.reconcile(study_context, contract, study, metadata)
    stale = {**record["process"], "start_ticks": "0"}
    with pytest.raises(RuntimeError, match="PID identity changed"):
        controller.group_members(stale)


def child_record(tmp_path, script):
    directory = tmp_path / "attempt"
    directory.mkdir()
    experiment = tmp_path / "fake-experiment"
    experiment.mkdir()
    (experiment / "run_experiment.py").write_text(script)
    record = {"directory": str(directory), "configuration_path": str(tmp_path / "trial.yaml"),
              "configuration_sha256": "synthetic", "status_path": str(directory / "runner_status.json"),
              "exit_code": None, "process": None}
    context = {"path": tmp_path / "study.yaml", "base_path": tmp_path / "base.yaml",
               "base": {"tracking": {"uri": "https://offline.invalid", "workspace": "test"}},
               "definition": {"limits": {"shutdown_seconds": 2}}}
    return context, record, experiment


def test_supervision_records_nonzero_exit_and_uses_actual_interpreter(tmp_path, monkeypatch):
    context, record, experiment = child_record(tmp_path, "raise SystemExit(7)\n")
    monkeypatch.setattr(runner, "EXPERIMENT_DIR", experiment)
    result = controller.supervise(context, record, 2)
    assert result["exit_code"] == 7 and result["argv"][0] == sys.executable
    assert controller.read_json(Path(record["directory"]) / "attempt.json")["exit_code"] == 7
    assert not controller.group_members(result["process"])


def test_timeout_terminates_owned_descendant_and_keeps_cleanup_uncertainty(tmp_path, monkeypatch):
    script = '''import subprocess,sys,time,signal
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
print(child.pid, flush=True)
time.sleep(60)
'''
    context, record, experiment = child_record(tmp_path, script)
    monkeypatch.setattr(runner, "EXPERIMENT_DIR", experiment)
    start = time.monotonic()
    result = controller.supervise(context, record, .2)
    assert time.monotonic() - start < 5
    assert result["exit_code"] == -signal.SIGKILL and result["interruption"] == "timeout"
    assert result["needs_cleanup"] is True  # no run ID: absence is not proof of no run
    assert not controller.group_members(result["process"])


def test_tracking_cleanup_is_bounded_even_when_transport_hangs(tmp_path, monkeypatch):
    context, record, _ = child_record(tmp_path, "")
    controller.atomic_json(Path(record["status_path"]), {"configuration_sha256": "synthetic",
        "tracking_uri": "https://offline.invalid", "workspace": "test", "run_id": "owned-run"})
    monkeypatch.setattr(controller, "TRACKING_CLEANUP", "import time; time.sleep(60)")
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        controller.cleanup_tracking(record, context, .1)
    assert time.monotonic() - start < 3
    monkeypatch.setattr(controller, "TRACKING_CLEANUP", 'print(\'{"run_id":"owned-run","status":"FINISHED"}\')')
    assert controller.cleanup_tracking(record, context, 2)["status"] == "FINISHED"


def test_candidate_failures_stop_after_two_without_objectives(study_context, monkeypatch):
    context = study_context
    monkeypatch.setattr(controller, "compatibility", contract_for)
    monkeypatch.setattr(controller, "cleanup_tracking", lambda *args: {"status": "FAILED"})
    def fail(context, record, seconds, *, termination=None):
        receipt = {"phase": "conversion", "failure": {"exception_type": "torch.OutOfMemoryError"}}
        controller.atomic_json(Path(record["status_path"]), receipt)
        record.update(exit_code=1, process={"pid": 123}, outcome="exited")
        return record
    monkeypatch.setattr(controller, "supervise", fail)
    result = controller.run_controller(context)
    assert result["asked"] == 2 and result["failed"] == [0, 1]
    study, metadata = controller.open_study(context, contract_for(context))
    assert all(t.values is None for t in study.trials[:2])
    assert metadata["consecutive_candidate_failures"] == 2
    assert controller.run_controller(context)["asked"] == 2


def test_unknown_failure_stops_and_preserves_original_and_cleanup_details(study_context, monkeypatch):
    monkeypatch.setattr(controller, "compatibility", contract_for)
    def fail(context, record, seconds, *, termination=None):
        controller.atomic_json(Path(record["status_path"]), {"phase": "model_loading",
            "failure": {"exception_type": "builtins.RuntimeError", "cause_types": []}})
        record.update(exit_code=1)
        return record
    monkeypatch.setattr(controller, "supervise", fail)
    monkeypatch.setattr(controller, "cleanup_tracking", lambda *args: (_ for _ in ()).throw(RuntimeError("offline transport failure")))
    with pytest.raises(RuntimeError, match="Shared, unknown"):
        controller.run_controller(study_context)
    records = list((study_context["state_dir"] / "attempts").glob("*/attempt.json"))
    assert len(records) == 1
    saved = controller.read_json(records[0])
    assert saved["exit_code"] == 1 and saved["needs_cleanup"]
    assert "offline transport failure" in saved["cleanup_error"]


def test_controller_and_manual_dry_runs_do_not_import_optuna_or_execute(study_context, monkeypatch, tmp_path):
    context = study_context
    for name in ("run_controller", "compatibility", "exclusive_locks"):
        monkeypatch.setattr(controller, name, lambda *args: pytest.fail("Dry-run executed"))
    before = inputs.git(runner.REPOSITORY_ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    assert controller.main(["--study", str(context["path"]), "--dry-run"]) == 0
    assert not context["state_dir"].exists() and not context["trial_dir"].exists()
    assert inputs.git(runner.REPOSITORY_ROOT, "status", "--porcelain=v1", "--untracked-files=all") == before
    # A fresh interpreter makes import isolation observable despite pytest's imports.
    script = '''import builtins,sys
original = builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('optuna','torch','mlflow','torchvision'):
        raise AssertionError('execution dependency imported: ' + name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
import optuna_controller,run_experiment
assert optuna_controller.main(['--study',sys.argv[1],'--dry-run']) == 0
assert run_experiment.main(['--config',sys.argv[2],'--dry-run']) == 0
'''
    subprocess.run([sys.executable, "-B", "-c", script, str(context["path"]), str(context["base_path"])],
        cwd=runner.REPOSITORY_ROOT, env={**os.environ, "PYTHONPATH": str(runner.EXPERIMENT_DIR)}, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_keyboard_interrupt_cleans_owned_child_and_records_interruption(tmp_path, monkeypatch):
    context, record, experiment = child_record(tmp_path, "import time; time.sleep(60)\n")
    monkeypatch.setattr(runner, "EXPERIMENT_DIR", experiment)
    monkeypatch.setattr(controller, "cleanup_tracking", lambda *args: {"status": "KILLED"})
    original_wait = subprocess.Popen.wait
    interrupted = []
    def wait(process, *args, **kwargs):
        if not interrupted:
            interrupted.append(process.pid)
            time.sleep(.1)
            raise KeyboardInterrupt()
        return original_wait(process, *args, **kwargs)
    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    result = controller.supervise(context, record, 10)
    assert result["interruption"] == "interrupted"
    assert result["exit_code"] is not None and result["cleanup"]["status"] == "KILLED"
    assert not controller.group_members(result["process"])


def test_failed_duplicate_is_not_retried_or_given_objectives(study_context, monkeypatch):
    definition = study_context["definition"]
    definition["pilot_cases"] = [definition["pilot_cases"][1]] * 2
    monkeypatch.setattr(controller, "compatibility", contract_for)
    monkeypatch.setattr(controller, "cleanup_tracking", lambda *args: {"status": "FAILED"})
    calls = []
    def fail(context, record, seconds, *, termination=None):
        calls.append(record["trial_number"])
        controller.atomic_json(Path(record["status_path"]), {"phase": "conversion",
            "failure": {"exception_type": "torch.OutOfMemoryError"}})
        return {**record, "exit_code": 1}
    monkeypatch.setattr(controller, "supervise", fail)
    with pytest.raises(RuntimeError, match="No automatic retry"):
        controller.run_controller(study_context)
    assert calls == [0]
    study, _ = controller.open_study(study_context, contract_for(study_context))
    assert [t.values for t in study.trials] == [None, None]
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials)


def test_sampler_files_must_be_trusted_and_unchanged(study_context):
    context = study_context
    contract, _, _, _, _ = create_pending(context)
    checkpoint = context["state_dir"] / "sampler.pickle"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"tampered")
    with pytest.raises(RuntimeError, match="inconsistent sampler"):
        controller.open_study(context, contract)
    checkpoint.write_bytes(original)
    checkpoint.chmod(0o666)
    with pytest.raises(RuntimeError, match="not writable by others"):
        controller.open_study(context, contract)


def test_staged_source_deletion_remains_in_compatibility_after_snapshot(study_context):
    context = study_context
    path = runner.EXPERIMENT_DIR / "obsolete.py"
    path.write_text("# deleted before study creation\n")
    inputs.git(runner.REPOSITORY_ROOT, "add", str(path))
    inputs.git(runner.REPOSITORY_ROOT, "commit", "-m", "Disposable obsolete helper")
    inputs.git(runner.REPOSITORY_ROOT, "rm", str(path))
    contract, _, _, _, record = create_pending(context)
    name = path.relative_to(runner.REPOSITORY_ROOT).as_posix()
    assert contract["source_files_sha256"][name] is None
    fake_success(context, record, 1)
    assert controller.execution_identity(context) == contract["source_files_sha256"]
    controller.collect_result(context, contract, record)
    path.write_text("# restored unexpectedly\n")
    assert controller.execution_identity(context)[name] is not None


def test_tracking_cleanup_queries_only_saved_run_and_preserves_terminal_status(tmp_path, monkeypatch):
    context, record, _ = child_record(tmp_path, "")
    controller.atomic_json(Path(record["status_path"]), {"configuration_sha256": "synthetic",
        "tracking_uri": "https://offline.invalid", "workspace": "test", "run_id": "owned-run"})
    fake = tmp_path / "mlflow.py"
    fake.write_text('''import json
from types import SimpleNamespace
calls = []
def set_tracking_uri(value):
    assert value == 'https://offline.invalid'
    calls.append(['uri', value])
def set_workspace(value):
    assert value == 'test'
    calls.append(['workspace', value])
class Client:
    status = 'RUNNING'
    def get_run(self, run_id):
        assert run_id == 'owned-run'
        calls.append(['get_run', run_id])
        return SimpleNamespace(info=SimpleNamespace(status=self.status))
    def set_terminated(self, run_id, status):
        assert run_id == 'owned-run' and status == 'KILLED'
        calls.append(['set_terminated', run_id, status])
        self.status = status
tracking = SimpleNamespace(MlflowClient=Client)
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    assert controller.cleanup_tracking(record, context, 3)["status"] == "KILLED"
    fake.write_text(fake.read_text().replace("status = 'RUNNING'", "status = 'FINISHED'").replace(
        "assert run_id == 'owned-run' and status == 'KILLED'", "raise AssertionError('terminal run was modified')"))
    assert controller.cleanup_tracking(record, context, 3)["status"] == "FINISHED"


@pytest.mark.parametrize("change,message", [
    ({"unexpected": True}, "unknown options"),
    ({"storage_targets": [1]}, "floating-point"),
    ({"mode": "joint"}, "individual_pilot"),
    ({"enabled_layers": ["layer9.0.conv2"]}, "supported ResNet18"),
])
def test_invalid_study_does_not_create_state_or_trial_configs(study_context, change, message):
    context = study_context
    definition = {**context["definition"], **change}
    context["path"].write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match=message):
        controller.load_study(context["path"])
    assert not context["state_dir"].exists() and not context["trial_dir"].exists()


def test_cli_requires_explicit_resume_and_will_not_recreate_missing_state(study_context, monkeypatch):
    context = study_context
    monkeypatch.setattr(controller, "run_controller", lambda *args: pytest.fail("Unexpected study execution"))
    assert controller.main(["--study", str(context["path"]), "--resume"]) == 1
    assert not context["state_dir"].exists()
    context["state_dir"].mkdir()
    (context["state_dir"] / "study.json").write_text("{}")
    assert controller.main(["--study", str(context["path"])]) == 1
    assert (context["state_dir"] / "study.json").read_text() == "{}"



def test_compatibility_hashes_synthetic_inputs_without_model_loading_or_cache_mutation(study_context, tmp_path, monkeypatch):
    import torch.hub
    context = study_context
    raw = Path(context["base"]["data"]["raw_dir"])
    image = raw / "10/example.jpeg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"artificial image bytes; never decoded")
    manifest = Path(context["base"]["data"]["manifest"])
    manifest.parent.mkdir(parents=True)
    controller.atomic_json(manifest, {"examples": [{"path": "10/example.jpeg", "label": 10,
                                                   "sha256": controller.digest(image.read_bytes())}]})
    checkpoint = tmp_path / "synthetic-checkpoint"
    checkpoint.write_bytes(b"not loaded")
    monkeypatch.setattr(runner, "cached_weights", lambda: (checkpoint, {
        "checkpoint_path": str(checkpoint), "sha256": runner.file_sha256(checkpoint)}))
    monkeypatch.setattr(runner, "load_cached_baseline", lambda *args: pytest.fail("A model was loaded"))
    previous_hub = torch.hub._hub_dir
    first = controller.compatibility(context)
    assert torch.hub._hub_dir == previous_hub
    assert first["manifest_sha256"] == runner.file_sha256(manifest)
    checkpoint.write_bytes(b"changed artificial weights")
    second = controller.compatibility(context)
    assert first["weights"]["sha256"] != second["weights"]["sha256"]
    assert torch.hub._hub_dir == previous_hub
    image.write_bytes(b"changed selected image")
    with pytest.raises(ValueError, match="image hash mismatch"):
        controller.compatibility(context)


@pytest.mark.parametrize("stop_signal,ignore_hup,tracking_confirmed", [
    (signal.SIGTERM, False, True), (signal.SIGHUP, False, True),
    (signal.SIGINT, False, True), (signal.SIGTERM, True, True),
    (signal.SIGTERM, False, False),
])
def test_termination_signal_stops_controller_and_owned_runner(study_context, tmp_path, stop_signal,
                                                              ignore_hup, tracking_confirmed):
    """A real OS signal must not leave the runner alive past its trial deadline."""
    import ctypes
    import textwrap

    context = study_context
    definition = deepcopy(context["definition"])
    definition["limits"].update(per_trial_seconds=1, shutdown_seconds=2)
    context["path"].write_text(yaml.safe_dump(definition, sort_keys=False))
    ready = tmp_path / "runner-ready.json"
    reaped = tmp_path / "descendant-reaped.json"
    fake_runner = textwrap.dedent('''\
        import hashlib,json,os,signal,subprocess,sys,time
        from pathlib import Path
        if __name__ == '__main__':
            config = Path(sys.argv[sys.argv.index('--config') + 1])
            status = Path(sys.argv[sys.argv.index('--status-file') + 1])
            child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'])
            def stop(signum, frame):
                child.kill()
                code = child.wait(timeout=1)
                Path(REAPED).write_text(json.dumps({'pid': child.pid, 'exit_code': code}))
                raise SystemExit(130)
            signal.signal(signal.SIGINT, stop)
            signal.signal(signal.SIGTERM, stop)
            status.write_text(json.dumps({'state': 'running', 'phase': 'conversion', 'run_id': 'isolated-run',
                                         'configuration_sha256': hashlib.sha256(config.read_bytes()).hexdigest()}))
            Path(READY).write_text(json.dumps({'pid': os.getpid(), 'descendant': child.pid}))
            time.sleep(60)
    ''').replace('READY', repr(str(ready))).replace('REAPED', repr(str(reaped)))
    (runner.EXPERIMENT_DIR / "run_experiment.py").write_text(fake_runner)
    driver = tmp_path / "controller-driver.py"
    source_directory = Path(controller.__file__).parent
    driver.write_text(textwrap.dedent('''\
        import signal,sys
        from pathlib import Path
        sys.path.insert(0, SOURCE)
        import optuna_controller as controller
        import run_experiment as runner
        runner.REPOSITORY_ROOT = Path(ROOT)
        runner.EXPERIMENT_DIR = Path(EXPERIMENT)
        def compatible(context):
            return {'source_files_sha256': controller.execution_identity(context),
                    'definition': context['definition'], 'fixed_config': context['base']}
        controller.compatibility = compatible
        controller.rank_signature = lambda layers: None
        def cleanup(*args):
            if not TRACKING_CONFIRMED:
                raise OSError('Isolated tracking finalization unavailable')
            return {'run_id': 'isolated-run', 'status': 'KILLED'}
        controller.cleanup_tracking = cleanup
        if IGNORE_HUP:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        raise SystemExit(controller.main(['--study', STUDY]))
    ''').replace('SOURCE', repr(str(source_directory))).replace('Path(ROOT)', f'Path({str(runner.REPOSITORY_ROOT)!r})')
         .replace('Path(EXPERIMENT)', f'Path({str(runner.EXPERIMENT_DIR)!r})').replace('STUDY', repr(str(context["path"])))
         .replace('TRACKING_CONFIRMED', repr(tracking_confirmed)).replace('IGNORE_HUP', repr(ignore_hup)))

    # Adopt a runner orphaned by the uncorrected controller so this regression can
    # clean up and reap its processes even on a container whose PID 1 does not.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0  # PR_GET_CHILD_SUBREAPER
    assert libc.prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER, test harness only
    process = None
    owned = None
    observation = {}
    try:
        with (tmp_path / "controller.stdout").open("wb") as out, (tmp_path / "controller.stderr").open("wb") as err:
            process = subprocess.Popen([sys.executable, "-B", str(driver)], cwd=runner.REPOSITORY_ROOT,
                stdout=out, stderr=err, start_new_session=True,
                env={key: value for key, value in os.environ.items() if key != "MLFLOW_RUN_ID"})
            until = time.monotonic() + 10
            while not ready.exists() and process.poll() is None and time.monotonic() < until:
                time.sleep(.02)
            assert ready.exists(), (tmp_path / "controller.stderr").read_text()
            owned = controller.read_json(ready)
            identity = controller.process_identity(owned["pid"])
            assert identity is not None
            if ignore_hup:
                process.send_signal(signal.SIGHUP)
                time.sleep(.15)
                assert process.poll() is None and controller.group_members(identity)
            sent_at = time.monotonic()
            process.send_signal(stop_signal)
            code = process.wait(timeout=5)
            # If orphaned, observe it beyond the configured one-second trial limit.
            if controller.group_members(identity):
                time.sleep(max(0, sent_at + 1.2 - time.monotonic()))
            survivors = controller.group_members(identity)
            observation = {"signal": stop_signal.name, "ignored_hup": ignore_hup,
                "tracking_confirmed": tracking_confirmed, "controller_exit_code": code,
                "seconds_after_signal": time.monotonic() - sent_at,
                "trial_seconds": 1, "shutdown_seconds": 2,
                "live_group_after_controller_exit": survivors,
                "survived_trial_deadline": bool(survivors) and time.monotonic() - sent_at > 1}
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
        if owned is not None:
            if Path(f"/proc/{owned['pid']}").exists():
                try:
                    os.killpg(owned["pid"], signal.SIGINT)
                except ProcessLookupError:
                    pass
                until = time.monotonic() + 2
                while not reaped.exists() and time.monotonic() < until:
                    time.sleep(.02)
                if not reaped.exists():
                    try:
                        os.killpg(owned["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            for pid in (owned["pid"], owned["descendant"]):
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass  # Already reaped by its actual parent.
            observation["all_test_processes_reaped"] = all(not Path(f"/proc/{pid}").exists()
                                                           for pid in (owned["pid"], owned["descendant"]))
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0
        controller.atomic_json(tmp_path / "signal-observation.json", observation)
        print(json.dumps(observation, sort_keys=True))
    assert observation["controller_exit_code"] == 1
    assert not observation["live_group_after_controller_exit"]
    assert observation["seconds_after_signal"] < 2.5
    assert observation["all_test_processes_reaped"]
    record_paths = list((context["state_dir"] / "attempts").glob("*/attempt.json"))
    assert len(record_paths) == 1
    attempt = controller.read_json(record_paths[0])
    assert attempt["interruption"] == "interrupted" and attempt["exit_code"] is not None
    assert attempt["termination_signal"] == stop_signal.name
    if tracking_confirmed:
        assert attempt["cleanup"]["status"] == "KILLED"
    else:
        assert attempt["needs_cleanup"] is True
        assert 'Isolated tracking finalization unavailable' in attempt["cleanup_error"]
    study = optuna.load_study(study_name=definition["study_name"], storage=f"sqlite:///{context['state_dir'] / 'study.sqlite3'}")
    assert study.trials[0].state == optuna.trial.TrialState.FAIL and study.trials[0].values is None
    assert all(t.state == optuna.trial.TrialState.WAITING for t in study.trials[1:])



def test_signal_handlers_are_lifecycle_scoped_and_restore_on_failure():
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        with pytest.raises(RuntimeError, match="ordinary control-flow failure"):
            with controller.termination_signals() as termination:
                assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGINT)  # Repeated requests do not interrupt cleanup.
                assert termination["signal"] == "SIGTERM"
                raise RuntimeError("ordinary control-flow failure")
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == previous[signal.SIGTERM]
        assert signal.getsignal(signal.SIGINT) == previous[signal.SIGINT]
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
