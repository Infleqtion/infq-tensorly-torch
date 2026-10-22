"""Offline workflow checks: tiny synthetic images/models and a recording MLflow client."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch
from torch import nn
yaml = pytest.importorskip("yaml", reason="Install the experiments dependency group")
pytest.importorskip("torchvision", reason="Install the experiments dependency group")
if importlib.util.find_spec("mlflow") is None:
    pytest.skip("Install the experiments dependency group", allow_module_level=True)

import run_experiment as runner
import tltorch
from tltorch import compress_model


@pytest.fixture
def config_file(tmp_path):
    config = yaml.safe_load((runner.EXPERIMENT_DIR / "configs/resnet18_exploration.yaml").read_text())
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config


def test_config_keeps_rank_types_and_uses_experiment_paths(config_file, tmp_path, monkeypatch):
    path, config = config_file
    config["compression"]["layers"]["fc"]["rank"] = 2
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    effective, original = runner.load_config(path.name)
    assert original == path.read_bytes()
    assert type(effective["compression"]["layers"]["fc"]["rank"]) is int
    assert type(effective["compression"]["layers"]["layer4.0.conv2"]["rank"]) is float
    assert effective["data"]["raw_dir"] == str(runner.EXPERIMENT_DIR / "data/raw/imagenetv2")
    assert effective["output_root"] == str(runner.EXPERIMENT_DIR / "results/runs")


@pytest.mark.parametrize("change,message", [
    (lambda c: c.update(unknown=True), "unknown options"),
    (lambda c: c["tracking"].pop("uri"), "missing options.*uri"),
    (lambda c: c["tracking"].update(uri=""), "tracking.uri must be a nonempty string"),
    (lambda c: c["tracking"].update(uri=None), "tracking.uri must be a nonempty string"),
    (lambda c: c["tracking"].pop("workspace"), "missing options.*workspace"),
    (lambda c: c["tracking"].update(workspace="  "), "tracking.workspace must be a nonempty string"),
    (lambda c: c["tracking"].update(workspace=None), "tracking.workspace must be a nonempty string"),
    (lambda c: c["runtime"].update(batch_size=True), "batch_size must be an integer"),
    (lambda c: c["runtime"].update(device="cuda"), "explicit CUDA"),
    (lambda c: c["compression"]["layers"]["fc"].update(rnak=2), "unknown options"),
    (lambda c: c["compression"]["layers"]["fc"].update(rank=True), "rank must be"),
    (lambda c: c["timing"].update(repeats=1), "repeats must be"),
])
def test_invalid_configuration(config_file, change, message):
    path, config = config_file
    change(config)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        runner.load_config(path)


def test_yaml_is_safe_and_duplicate_options_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model: first\nmodel: second\n")
    with pytest.raises(ValueError, match="unique strings"):
        runner.load_config(path)
    path.write_text("!!python/object/apply:os.system ['touch should-not-exist']")
    with pytest.raises(yaml.constructor.ConstructorError):
        runner.load_config(path)


def test_dry_run_imports_no_execution_modules_and_creates_no_outputs(config_file, tmp_path):
    path, config = config_file
    config["output_root"] = str(tmp_path / "outputs")
    config["data"] = {"raw_dir": str(tmp_path / "missing-data"), "manifest": str(tmp_path / "missing.json")}
    path.write_text(yaml.safe_dump(config))
    code = '''
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import run_experiment
with patch("socket.socket.connect", side_effect=AssertionError("no network")):
    assert run_experiment.main(["--config", sys.argv[2], "--dry-run"]) == 0
assert not any(name in sys.modules for name in ("torch", "torchvision", "mlflow", "tltorch"))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(runner.EXPERIMENT_DIR), path.name],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "Execution must still check" in result.stdout
    assert '"workspace": "quirc"' in result.stdout
    # The file-based CLI obeys the same caller-relative config rule as main().
    command = [sys.executable, "-B", str(runner.EXPERIMENT_DIR / "run_experiment.py"),
               "--config", path.name, "--dry-run"]
    direct = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert direct.stdout == result.stdout
    assert not (tmp_path / "outputs").exists()


@pytest.fixture
def images(tmp_path):
    from PIL import Image

    raw = tmp_path / "raw"
    entries = []
    # Two images per class, in an explicit order different from filesystem order.
    for label, index in ((5, 1), (0, 1), (5, 0), (0, 0)):
        path = raw / str(label) / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(label * 20, index * 40, 10)).save(path)
        entries.append({"path": path.relative_to(raw).as_posix(), "label": label,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset": "synthetic fixture", "seed": 9, "examples": entries}))
    return raw, manifest


def test_manifest_loader_preserves_multiple_images_per_class_and_order(images):
    from torchvision.transforms import ToTensor

    raw, path = images
    manifest, original = runner.read_manifest(raw, path)
    assert original == path.read_bytes()
    loader = runner.make_loader(raw, manifest, ToTensor(), batch_size=3)
    assert [labels.tolist() for _, labels in loader] == [[5, 0, 5], [0]]


@pytest.mark.parametrize("problem", ["label", "missing", "hash", "unreadable"])
def test_manifest_errors_never_skip_selected_images(images, problem):
    raw, path = images
    manifest = json.loads(path.read_text())
    selected = raw / manifest["examples"][0]["path"]
    if problem == "label":
        manifest["examples"][0]["label"] = 0
    elif problem == "missing":
        selected.unlink()
    else:
        selected.write_bytes(b"not an image")
        if problem == "unreadable":
            manifest["examples"][0]["sha256"] = hashlib.sha256(selected.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    if problem != "unreadable":
        with pytest.raises(ValueError):
            runner.read_manifest(raw, path)
    else:
        checked, _ = runner.read_manifest(raw, path)
        with pytest.raises(RuntimeError, match="Cannot read.*5/1.png"):
            next(iter(runner.make_loader(raw, checked, lambda x: x)))


def test_metrics_count_images_and_include_absent_classes_in_macro_average():
    # All three predictions choose class 0. One is correct; two are in the top five.
    logits = torch.tensor([[6., 5., 4., 3., 2., 1.]] * 3)
    labels = torch.tensor([0, 4, 5])
    batches = [(logits[:2], labels[:2]), (logits[2:], labels[2:])]
    result = runner.evaluate(nn.Identity(), batches, "cpu")
    assert result["top1_percent"] == pytest.approx(100 / 3)
    assert result["top5_percent"] == pytest.approx(200 / 3)
    assert result["cross_entropy_mean"] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    # Class 0: precision 1/3, recall 1, F1 1/2. Other five classes contribute zero.
    assert result["macro_precision"] == pytest.approx(1 / 18)
    assert result["macro_recall"] == pytest.approx(1 / 6)
    assert result["macro_f1"] == pytest.approx(1 / 12)
    assert result["macro_classes"] == 6


def tiny_model():
    torch.manual_seed(8)
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32, 6)).eval()
    model.register_buffer("remember", torch.tensor([7.]), persistent=False)
    layers = {"0": {"factorization": "Tucker", "rank": [2, 3, 1, 1]},
              "3": {"factorization": "BlockTT", "rank": 2,
                    "in_tensorized_features": [4, 8], "out_tensorized_features": [2, 3]}}
    return model, layers


def test_real_checkpoint_round_trip_without_loading_weights_or_decomposing(tmp_path, monkeypatch):
    model, layers = tiny_model()
    compressed, report = compress_model(model, layers)
    inputs = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 48
    verification = runner.save_and_verify_model(compressed, report, inputs, tmp_path)
    assert verification["passed"] and verification["parameters_and_buffers_exact"]
    assert "TuckerTensor" in verification["layers"]["0"]["weight_type"]
    assert "BlockTT" in verification["layers"]["3"]["weight_type"]
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(tltorch, "compress_model", lambda *a, **k: pytest.fail("decomposition"))
    loaded = runner.load_compressed_model(tmp_path / "compressed_model.pt")
    assert loaded is not compressed
    assert torch.equal(loaded.remember, model.remember)
    torch.testing.assert_close(loaded(inputs), compressed(inputs), rtol=0, atol=0)
    # A fresh process also has to load the artifact without constructors or decomposition.
    code = """
import sys
from pathlib import Path
from unittest.mock import patch
import torch
sys.path.insert(0, sys.argv[1])
from run_experiment import load_compressed_model
root = Path(sys.argv[2])
with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("download")), \
     patch("tltorch.compress_model", side_effect=AssertionError("decomposition")):
    model = load_compressed_model(root / "compressed_model.pt")
check = torch.load(root / "verification_input.pt", weights_only=True)
with torch.inference_mode():
    torch.testing.assert_close(model(check["inputs"]), check["expected_outputs"], rtol=1e-5, atol=1e-6)
assert type(model[0].weight).__name__ == "TuckerTensor"
assert type(model[3].weight).__name__ == "BlockTT"
assert model.remember.item() == 7
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(runner.EXPERIMENT_DIR), str(tmp_path)],
                   cwd=tmp_path, check=True)


class RecordingMlflow:
    """Record fluent calls and model the context closing before diagnostics are uploaded."""

    def __init__(self):
        self.events, self.fail = [], None
        self.metrics = {}
        self.params, self.tags, self.uploads = {}, {}, {}
        self.uri = self.workspace = self.active_run_id = None
        self.run_count = 0

    def set_tracking_uri(self, uri):
        self.events.append(f"uri {uri}")
        self.uri = uri

    def set_workspace(self, workspace):
        assert self.uri is not None
        self.events.append(f"workspace {workspace}")
        self.workspace = workspace

    def set_experiment(self, name):
        assert self.uri is not None and self.workspace is not None
        self.events.append(f"experiment {name}")
        return SimpleNamespace(experiment_id="experiment-1")

    @contextmanager
    def start_run(self, *, experiment_id, run_name, tags):
        assert experiment_id == "experiment-1" and self.active_run_id is None
        self.run_count += 1
        self.active_run_id = "run-1"
        self.events.append("start")
        self.tags = dict(tags)
        self.run_name = run_name
        try:
            yield SimpleNamespace(info=SimpleNamespace(run_id="run-1", experiment_id=experiment_id))
        except BaseException:
            self.end_run("FAILED")
            raise
        else:
            self.end_run("FINISHED")

    def end_run(self, status):
        # MLflow clears its active run before sending the status update, even if it fails.
        self.active_run_id = None
        self.set_terminated("run-1", status)

    def log_artifact(self, path, *, run_id=None):
        path = Path(path)
        assert (run_id or self.active_run_id) == "run-1" and path.is_file()
        self.events.append(f"upload {path.name}")
        if self.fail == "early upload" and path.name == "effective_config.json":
            raise OSError("configuration upload failed")
        if self.fail in ("failure upload", "both") and path.name == "failure.json":
            raise OSError("diagnostic upload failed")
        self.uploads[path.name] = path.read_bytes()

    def log_param(self, key, value, **kwargs):
        assert kwargs["synchronous"] and isinstance(value, str)
        self.params[key] = value

    def log_metric(self, key, value, **kwargs):
        self.metrics[key] = value
        assert kwargs["synchronous"]

    def log_artifacts(self, path):
        assert self.active_run_id == "run-1"
        self.events.append("all artifacts")
        assert (Path(path) / "compressed_model.pt").is_file()
        assert json.loads((Path(path) / "model_verification.json").read_text())["passed"]
        if self.fail == "late upload":
            raise OSError("artifact upload failed")
        self.uploads.update({p.name: p.read_bytes() for p in Path(path).iterdir() if p.is_file()})

    def set_terminated(self, run_id, status):
        assert run_id == "run-1"
        self.events.append(status)
        if self.fail in ("status update", "both") or (self.fail == "finish" and status == "FINISHED"):
            raise OSError("tracking connection lost")


@pytest.fixture
def tracking(monkeypatch):
    recording = RecordingMlflow()
    for name in ("set_tracking_uri", "set_workspace", "set_experiment", "start_run",
                 "log_param", "log_metric", "log_artifact", "log_artifacts"):
        monkeypatch.setattr(f"mlflow.{name}", getattr(recording, name))

    def failure_client(*, tracking_uri):
        assert tracking_uri == recording.uri
        return recording

    monkeypatch.setattr("mlflow.MlflowClient", failure_client)
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("No tracking server in offline tests"))
    return recording


@pytest.fixture
def execution(config_file, images, tmp_path, monkeypatch, request, tracking):
    from torchvision.transforms import ToTensor

    path, config = config_file
    baseline, layers = tiny_model()
    config["tracking"]["uri"] = "https://configured.example.invalid"
    config["data"] = {"raw_dir": str(images[0]), "manifest": str(images[1])}
    config["compression"]["layers"] = layers
    config["runtime"].update(device=getattr(request, "param", "cpu"), batch_size=3, threads=1)
    config["timing"].update(input_shape=[1, 3, 4, 4], warmup=1, repeats=2)
    config["output_root"] = str(tmp_path / "outputs")
    path.write_text(yaml.safe_dump(config))
    effective, original = runner.load_config(path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://conflicting.example.invalid")
    events = tracking.events
    monkeypatch.setattr(runner, "cached_weights", lambda: (None, {"identifier": "tiny fixture"}))

    def load(path, device):
        events.append("load baseline")
        return baseline.to(device), ToTensor()

    monkeypatch.setattr(runner, "load_cached_baseline", load)
    convert = tltorch.compress_model

    def convert_once(*args, **kwargs):
        events.append("compress")
        assert not torch.is_inference_mode_enabled()
        return convert(*args, **kwargs)

    monkeypatch.setattr(tltorch, "compress_model", convert_once)
    return effective, original, baseline, events


@pytest.mark.parametrize("execution", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))], indirect=True)
def test_complete_execution_tracks_before_loading_and_preserves_baseline(execution, capsys, monkeypatch, tracking):
    config, original, baseline, events = execution
    fingerprint = runner.model_fingerprint(baseline)
    device = torch.device(config["runtime"]["device"])
    forward_devices, timing_devices = [], []
    evaluate, measure_latency = runner.evaluate, runner.measure_latency

    def check_forward_device(model, args):
        assert args[0].device == device
        assert all(t.device == device for t in [*model.parameters(), *model.buffers()])
        forward_devices.append(args[0].device)

    def checked_evaluation(model, batches, selected_device):
        handle = model.register_forward_pre_hook(check_forward_device)
        try:
            return evaluate(model, batches, selected_device)
        finally:
            handle.remove()

    def checked_latency(model, inputs, **kwargs):
        check_forward_device(model, (inputs,))
        timing_devices.append(inputs.device)
        return measure_latency(model, inputs, **kwargs)

    monkeypatch.setattr(runner, "evaluate", checked_evaluation)
    monkeypatch.setattr(runner, "measure_latency", checked_latency)
    client = tracking
    output = runner.run_experiment(config, original)
    assert events == ["uri https://configured.example.invalid", "workspace quirc", "experiment model-compression", "start", "upload effective_config.json", "upload original_config.yaml",
                      "upload run.json", "upload source.json", "upload source.zip", "load baseline",
                      "upload evaluation_manifest.json", "upload provenance.json", "compress",
                      "all artifacts", "FINISHED"]
    assert tracking.run_count == 1 and tracking.active_run_id is None
    assert tracking.uri == config["tracking"]["uri"]
    assert tracking.run_name == config["tracking"]["run_name"]
    run = json.loads(client.uploads["run.json"])
    assert run["workspace"] == "quirc"
    assert json.loads(client.uploads["effective_config.json"])["tracking"] == config["tracking"]
    assert yaml.safe_load(client.uploads["original_config.yaml"])["tracking"]["workspace"] == "quirc"
    assert run["experiment_id"] == "experiment-1" and run["run_id"] == "run-1"
    printed = capsys.readouterr().out
    assert "quirc" in printed and "experiment-1" in printed and "run-1" in printed and str(output) in printed
    source = json.loads(client.uploads["source.json"])
    assert client.tags == {"tracking.workspace": "quirc",
                           "git.branch": runner.git("branch", "--show-current").strip(),
                           "git.commit": runner.git("rev-parse", "HEAD").strip(),
                           "git.dirty": str(bool(source["working_tree_status"].strip())).lower()}
    provenance = json.loads(client.uploads["provenance.json"])
    # The fixture uses a custom YAML, so hashing the repository example would be wrong.
    assert original != (runner.EXPERIMENT_DIR / "configs/resnet18_exploration.yaml").read_bytes()
    for filename, key in (("original_config.yaml", "original_config_file_sha256"),
                          ("effective_config.json", "effective_config_file_sha256")):
        digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert hashlib.sha256(client.uploads[filename]).hexdigest() == digest
        assert run[key] == provenance[key] == client.params[key] == digest
    assert client.params["manifest_sha256"] == hashlib.sha256(client.uploads["evaluation_manifest.json"]).hexdigest()
    assert client.params["source_archive_sha256"] == hashlib.sha256(client.uploads["source.zip"]).hexdigest()
    assert runner.model_fingerprint(baseline) == fingerprint
    assert client.metrics["dense.examples"] == client.metrics["compressed.examples"] == 4
    assert "conversion_seconds" in client.metrics
    assert (output / "original_config.yaml").read_bytes() == original
    assert (output / "evaluation_manifest.json").read_bytes() == Path(config["data"]["manifest"]).read_bytes()
    with zipfile.ZipFile(output / "source.zip") as archive:
        names = archive.namelist()
        assert "experiments/imagenet-compression/run_experiment.py" in names
        assert "experiments/imagenet-compression/configs/resnet18_exploration.yaml" in names
        assert not any(n.startswith(("experiments/imagenet-compression/data/raw/",
                                                "experiments/imagenet-compression/data/processed/",
                                                ".venv/", "experiments/imagenet-compression/results/")) for n in names)
    assert len((output / "results.csv").read_text().splitlines()) == 3
    assert len(forward_devices) == 6  # Two evaluation batches and one timing check per model.
    assert timing_devices == [device, device]
    assert client.params["device"] == str(device)
    verification = json.loads(client.uploads["model_verification.json"])
    assert verification["passed"] and verification["device"] == str(device)


@pytest.mark.parametrize("failure", ["workspace", "experiment"])
def test_tracking_setup_errors_do_not_fall_back(execution, monkeypatch, tracking, failure):
    config, original, _, events = execution
    original_error = RuntimeError("workspace access denied")

    def fail(value):
        events.append(f"{failure} {value}")
        raise original_error

    monkeypatch.setattr(f"mlflow.set_{failure}", fail)
    with pytest.raises(RuntimeError) as caught:
        runner.run_experiment(config, original)
    assert caught.value is original_error
    expected = ["uri https://configured.example.invalid", "workspace quirc"]
    if failure == "experiment":
        expected += ["experiment model-compression"]
    assert events == expected
    assert tracking.run_count == 0
    failure_path, = Path(config["output_root"]).glob("*/failure.json")
    assert json.loads(failure_path.read_text())["error"] == "workspace access denied"


def test_environment_cannot_resume_an_existing_run(execution, monkeypatch, tracking):
    config, original, _, events = execution
    monkeypatch.setenv("MLFLOW_RUN_ID", "another-run")
    with pytest.raises(ValueError, match="each execution must create a new run"):
        runner.run_experiment(config, original)
    assert events == [] and tracking.run_count == 0


@pytest.mark.parametrize("failure", ["early upload", "conversion", "late upload"])
def test_failures_preserve_cause_and_never_finish_a_run(execution, failure, monkeypatch, tracking):
    config, original, baseline, events = execution
    fingerprint = runner.model_fingerprint(baseline)
    client = tracking
    tracking.fail = failure
    if failure == "conversion":
        def fail(*args, **kwargs):
            raise RuntimeError("deliberate conversion failure")
        monkeypatch.setattr(tltorch, "compress_model", fail)
    with pytest.raises((RuntimeError, OSError), match="failure|failed"):
        runner.run_experiment(config, original)
    assert tracking.run_count == 1 and tracking.active_run_id is None
    assert events[-1] == "FAILED" and "FINISHED" not in events
    if failure == "early upload":
        assert "load baseline" not in events
    assert runner.model_fingerprint(baseline) == fingerprint
    failures = list(Path(config["output_root"]).glob("*/failure.json"))
    assert len(failures) == 1
    assert client.uploads["failure.json"] == failures[0].read_bytes()
    failure_record = json.loads(client.uploads["failure.json"])
    assert failure_record["run_id"] == "run-1"
    assert "Traceback (most recent call last)" in failure_record["traceback"]
    if failure == "conversion":
        assert "deliberate conversion failure" in failure_record["traceback"]
        assert "provenance.json" in client.uploads and "source.zip" in client.uploads


def test_existing_output_is_not_overwritten(tmp_path):
    first = runner.create_output(tmp_path, "run-1")
    (first / "keep.txt").write_text("existing result")
    with pytest.raises(FileExistsError):
        runner.create_output(tmp_path, "run-1")
    assert (first / "keep.txt").read_text() == "existing result"


@pytest.mark.parametrize("address", ["sqlite:///mlflow.db", "file:///tmp/mlruns", "https://user:secret@server",
                                     "https://@server", "https://server?token=secret", "http://server:bad",
                                     "http://[broken", "http://server:99999", "http://", "https://server/path with spaces"])
def test_tracking_uri_validation_runs_without_tracking_requests(config_file, address):
    path, config = config_file
    config["tracking"]["uri"] = address
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=r"tracking.uri must be an HTTP\(S\)"):
        runner.load_config(path)


def test_cli_returns_nonzero_after_execution_failure(config_file, monkeypatch):
    import run_experiment

    def fail(*args):
        raise RuntimeError("deliberate execution failure")
    monkeypatch.setattr(runner, "run_experiment", fail)
    assert run_experiment.main(["--config", str(config_file[0])]) == 1


@pytest.mark.parametrize("tracking_failure", ["failure upload", "status update", "both"])
def test_diagnostic_tracking_errors_do_not_hide_conversion_error(execution, monkeypatch, tracking_failure, tracking):
    config, original, _, events = execution
    client = tracking
    tracking.fail = tracking_failure
    original_error = RuntimeError("original conversion error")
    cause = ValueError("invalid decomposition input")

    def fail_conversion(*args, **kwargs):
        raise original_error from cause

    monkeypatch.setattr(tltorch, "compress_model", fail_conversion)
    with pytest.raises(RuntimeError) as caught:
        runner.run_experiment(config, original)
    assert caught.value is original_error and caught.value.__cause__ is cause
    assert tracking.run_count == 1 and tracking.active_run_id is None
    assert events[-2:] == ["upload failure.json", "FAILED"]
    assert "FINISHED" not in events
    failure_path, = Path(config["output_root"]).glob("*/failure.json")
    failure_record = json.loads(failure_path.read_bytes())
    assert failure_record["error"] == "original conversion error"
    assert "fail_conversion" in failure_record["traceback"]
    if tracking_failure == "status update":
        assert client.uploads["failure.json"] == failure_path.read_bytes()
    else:
        assert "failure.json" not in client.uploads


def test_serialization_failure_keeps_measurements_and_fails_the_run(execution, monkeypatch, tracking):
    config, original, _, events = execution

    def fail_save(model, report, inputs, output):
        (output / "compressed_model.pt").write_bytes(b"incomplete checkpoint")
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(runner, "save_and_verify_model", fail_save)
    client = tracking
    with pytest.raises(OSError, match="checkpoint write failed"):
        runner.run_experiment(config, original)
    assert tracking.run_count == 1 and tracking.active_run_id is None
    assert events[-1] == "FAILED" and "FINISHED" not in events
    assert "compressed_model.pt" not in client.uploads and "all artifacts" not in events
    assert len(list(Path(config["output_root"]).glob("*/results.csv"))) == 1


def test_context_finish_failure_keeps_diagnostics_and_marks_the_same_run_failed(execution, tracking):
    config, original, _, events = execution
    tracking.fail = "finish"
    with pytest.raises(OSError, match="tracking connection lost"):
        runner.run_experiment(config, original)
    assert events[-4:] == ["all artifacts", "FINISHED", "upload failure.json", "FAILED"]
    assert tracking.run_count == 1 and tracking.active_run_id is None
    failure_path, = Path(config["output_root"]).glob("*/failure.json")
    assert tracking.uploads["failure.json"] == failure_path.read_bytes()
    assert json.loads(failure_path.read_text())["error"] == "tracking connection lost"


def test_missing_weights_explain_cache_setup_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("download"))
    with pytest.raises(FileNotFoundError, match="TORCH_HOME"):
        runner.cached_weights()


def test_requested_parameters_keep_rank_meaning_and_supplied_controls(config_file, tracking):
    path, config = config_file
    config["compression"]["layers"] = {
        "fc": {"factorization": "BlockTT", "rank": 1,
               "in_tensorized_features": [4, 8], "out_tensorized_features": [2, 3],
               "max_parameters": 128},
        "layer4.0.conv2": {"factorization": "Tucker", "rank": 1.0, "fixed_rank_modes": "spatial",
                           "decomposition_kwargs": {"random_state": 7, "tol": 0.0,
                                                    "n_iter_max": 2, "init": "svd"}},
    }
    path.write_text(yaml.safe_dump(config))
    effective, _ = runner.load_config(path)
    client = tracking
    runner.log_requested_settings(effective)
    assert client.params == {
        "model": "torchvision.resnet18", "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "seed": "2026", "timing_seed": "2027", "device": "cuda:0", "dtype": "float32",
        "implementation": "factorized", "compression.layers": '["fc", "layer4.0.conv2"]',
        "compression.fc.factorization": "BlockTT", "compression.fc.rank": "1",
        "compression.fc.in_tensorized_features": "[4, 8]",
        "compression.fc.out_tensorized_features": "[2, 3]", "compression.fc.max_parameters": "128",
        "compression.layer4.0.conv2.factorization": "Tucker", "compression.layer4.0.conv2.rank": "1.0",
        "compression.layer4.0.conv2.fixed_rank_modes": "spatial",
        "compression.layer4.0.conv2.decomposition_kwargs.init": "svd",
        "compression.layer4.0.conv2.decomposition_kwargs.n_iter_max": "2",
        "compression.layer4.0.conv2.decomposition_kwargs.tol": "0.0",
        "compression.layer4.0.conv2.decomposition_kwargs.random_state": "7",
        "evaluation.batch_size": "16", "evaluation.workers": "0", "runtime.threads": "4",
        "timing.input_shape": "[1, 3, 224, 224]", "timing.warmup": "20", "timing.repeats": "100",
    }
    # YAML order must not change parameter strings or add library defaults.
    effective["compression"]["layers"] = dict(reversed(list(effective["compression"]["layers"].items())))
    first = dict(client.params)
    client.params.clear()
    runner.log_requested_settings(effective)
    assert client.params == first


def test_unavailable_cuda_is_rejected_without_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="Requested device cuda:0.*no fallback"):
        runner.configure_runtime("cuda:0", threads=1)
    assert runner.configure_runtime("cpu", threads=1) == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_latency_synchronizes_before_start_and_before_stop(monkeypatch):
    import run_experiment as experiment

    device = runner.configure_runtime("cuda:0", threads=1)
    model = nn.Linear(2, 2).to(device).eval()
    inputs = torch.ones(1, 2, device=device)
    events = []
    synchronize, clock = torch.cuda.synchronize, experiment.time.perf_counter_ns

    def checked_sync(selected_device):
        assert selected_device == device
        synchronize(selected_device)
        events.append("sync")

    def checked_clock():
        events.append("clock")
        return clock()

    monkeypatch.setattr(torch.cuda, "synchronize", checked_sync)
    monkeypatch.setattr(experiment.time, "perf_counter_ns", checked_clock)
    handle = model.register_forward_hook(lambda *args: events.append("forward"))
    try:
        result = runner.measure_latency(model, inputs, warmup=1, repeats=2)
    finally:
        handle.remove()
    assert events == ["forward"] + ["sync", "clock", "forward", "sync", "clock"] * 2
    assert len(result["samples_ms"]) == 2 and all(ms > 0 for ms in result["samples_ms"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_same_cuda_checkpoint_loads_on_cpu_and_cuda(tmp_path, monkeypatch):
    from tltorch import FactorizedConv, FactorizedLinear

    device = runner.configure_runtime("cuda:0", threads=1)
    model, layers = tiny_model()
    compressed, report = compress_model(model.to(device), layers)
    inputs = torch.arange(48, dtype=torch.float32, device=device).reshape(1, 3, 4, 4) / 48
    runner.save_and_verify_model(compressed, report, inputs, tmp_path)
    expected_state = runner.model_fingerprint(compressed)
    expected_identity = runner.layer_identity(compressed, report)

    def forbidden(*args, **kwargs):
        pytest.fail("Loading must neither download weights nor repeat decomposition")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(FactorizedConv, "from_conv", forbidden)
    monkeypatch.setattr(FactorizedLinear, "from_linear", forbidden)
    outputs = []
    for location in ("cpu", "cuda:0"):
        loaded = runner.load_compressed_model(tmp_path / "compressed_model.pt", device=location)
        assert runner.model_fingerprint(loaded) == expected_state
        assert runner.layer_identity(loaded, report) == expected_identity
        assert all(t.device == torch.device(location) for t in [*loaded.parameters(), *loaded.buffers()])
        with torch.inference_mode():
            outputs.append(loaded(inputs.to(location)).cpu())
    # Identical float32 weights may accumulate in different orders on CPU and CUDA.
    # Use the existing checkpoint forward tolerance, not factor-by-factor decomposition equality.
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
