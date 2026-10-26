"""Offline workflow checks: tiny synthetic images/models and a recording MLflow client."""

from contextlib import contextmanager
import csv
import hashlib
import io
import importlib.util
import json
import os
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


@pytest.fixture(autouse=True)
def restore_hub_cache(monkeypatch):
    """torch.hub.set_dir changes a global value; restore it after every test."""
    monkeypatch.setattr(torch.hub, "_hub_dir", torch.hub._hub_dir)


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
    (lambda c: c.update(compression=False), "compression must be a mapping"),
    (lambda c: c.update(compression=[]), "compression must be a mapping"),
    (lambda c: c.update(compression="dense"), "compression must be a mapping"),
    (lambda c: c.update(compression={}), "compression: missing options"),
    (lambda c: c["compression"].update(layers={}), "compression.layers must be a nonempty mapping"),
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


@pytest.mark.parametrize("compression", ["absent", "null"])
def test_optional_compression_normalizes_dense_configuration(config_file, compression):
    path, config = config_file
    if compression == "absent":
        config.pop("compression")
    else:
        config["compression"] = None
    path.write_text(yaml.safe_dump(config))
    effective, original = runner.load_config(path)
    assert effective["compression"] is None and original == path.read_bytes()
    _, model_parameters = runner.requested_parameters(effective)
    assert model_parameters == {"model": config["model"], "weights": config["weights"]}
    assert "one dense pretrained ResNet-18 model" in runner.describe(effective)


def test_dense_example_preserves_existing_scientific_settings():
    directory = runner.EXPERIMENT_DIR / "configs"
    compressed = yaml.safe_load((directory / "resnet18_exploration.yaml").read_text())
    dense = yaml.safe_load((directory / "resnet18_exploration_dense.yaml").read_text())
    compressed.pop("compression")
    compressed["tracking"]["run_name"] = "resnet18-exploration-dense"
    assert dense == compressed


def test_yaml_is_safe_and_duplicate_options_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model: first\nmodel: second\n")
    with pytest.raises(ValueError, match="unique strings"):
        runner.load_config(path)
    path.write_text("!!python/object/apply:os.system ['touch should-not-exist']")
    with pytest.raises(yaml.constructor.ConstructorError):
        runner.load_config(path)


@pytest.mark.parametrize("compressed", [False, True])
def test_dry_run_imports_no_execution_modules_and_creates_no_outputs(config_file, tmp_path, compressed):
    path, config = config_file
    if not compressed:
        config.pop("compression")
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


@pytest.mark.parametrize("factorization", [None, "Tucker", "TT"])
def test_real_checkpoint_round_trip_without_loading_weights_or_decomposing(tmp_path, monkeypatch, factorization):
    model, layers = tiny_model()
    if factorization == "TT":
        layers["0"] = {"factorization": "TT", "rank": 2}
    compressed, report = (model, None) if factorization is None else compress_model(model, layers)
    inputs = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 48
    verification = runner.save_and_verify_model(compressed, report, inputs, tmp_path)
    assert verification["passed"] and verification["parameters_and_buffers_exact"]
    if factorization is None:
        assert verification["layers"] == {}
    else:
        assert factorization in verification["layers"]["0"]["weight_type"]
        assert "BlockTT" in verification["layers"]["3"]["weight_type"]
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(tltorch, "compress_model", lambda *a, **k: pytest.fail("decomposition"))
    loaded = runner.load_model(tmp_path / "model.pt")
    # Historical loading instructions remain valid through the compatibility helper.
    assert runner.model_fingerprint(runner.load_compressed_model(tmp_path / "model.pt")) == runner.model_fingerprint(loaded)
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
    model = load_compressed_model(root / "model.pt")
check = torch.load(root / "verification_input.pt", weights_only=True)
with torch.inference_mode():
    torch.testing.assert_close(model(check["inputs"]), check["expected_outputs"], rtol=1e-5, atol=1e-6)
if sys.argv[3] == "dense":
    assert isinstance(model[0], torch.nn.Conv2d) and isinstance(model[3], torch.nn.Linear)
else:
    assert type(model[0].weight).__name__ == sys.argv[3] + "Tensor"
    assert type(model[3].weight).__name__ == "BlockTT"
assert model.remember.item() == 7
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(runner.EXPERIMENT_DIR), str(tmp_path), factorization or "dense"],
                   cwd=tmp_path, check=True)


class RecordingMlflow:
    """Record fluent calls and model the context closing before diagnostics are uploaded."""

    def __init__(self):
        self.events, self.fail = [], None
        self.metrics, self.metric_models, self.model_params = {}, {}, {}
        self.model_count = 0
        self.active_model_id = None
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

    def log_artifact(self, path, *, run_id=None, artifact_path=None):
        path = Path(path)
        assert (run_id or self.active_run_id) == "run-1" and path.is_file()
        self.events.append(f"upload {path.name}")
        if self.fail == "early upload" and path.name == "effective_config.json":
            raise OSError("configuration upload failed")
        if self.fail in ("failure upload", "both") and path.name == "failure.json":
            raise OSError("diagnostic upload failed")
        key = f"{artifact_path}/{path.name}" if artifact_path else path.name
        self.uploads[key] = path.read_bytes()

    def log_param(self, key, value, **kwargs):
        assert kwargs["synchronous"] and isinstance(value, str)
        self.params[key] = value

    def initialize_logged_model(self, *, name, params):
        assert self.active_run_id == "run-1" and name == "model"
        self.events.append("initialize model")
        if self.fail == "model initialization":
            raise RuntimeError("MLflow model initialization failed")
        self.model_count += 1
        self.model_params = dict(params)
        return SimpleNamespace(model_id="model-1")

    def log_metric(self, *args, **kwargs):
        pytest.fail("Use metric batches with explicit model ownership")

    def clear_active_model(self):
        self.active_model_id = None

    def log_batch(self, run_id, *, metrics, synchronous):
        assert run_id == self.active_run_id == "run-1" and synchronous
        assert self.active_model_id is None
        for metric in metrics:
            assert metric.run_id == run_id
            self.metrics[metric.key] = metric.value
            self.metric_models[metric.key] = metric.model_id

    def log_artifacts(self, path):
        assert self.active_run_id == "run-1"
        self.events.append("all artifacts")
        assert (Path(path) / "model.pt").is_file()
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
def tracking(monkeypatch, tmp_path):
    import mlflow.pytorch
    recording = RecordingMlflow()
    save_model, load_model = mlflow.pytorch.save_model, mlflow.pytorch.load_model

    def log_model(model, **kwargs):
        assert recording.active_run_id == "run-1"
        recording.events.append("log model")
        if recording.fail == "model logging":
            raise RuntimeError("MLflow model logging failed")
        assert kwargs.pop("name") == "model"
        assert kwargs.pop("model_id") == "model-1" and recording.model_count == 1
        package = tmp_path / "mlflow-package"
        save_model(model, str(package), **kwargs)
        recording.active_model_id = "model-1"
        return SimpleNamespace(model_uri=str(package), model_id="model-1")

    def reload_model(*args, **kwargs):
        recording.events.append("reload model")
        if recording.fail == "model verification":
            raise RuntimeError("MLflow model reload failed")
        return load_model(*args, **kwargs)

    monkeypatch.setattr(mlflow.pytorch, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "load_model", reload_model)
    for name in ("set_tracking_uri", "set_workspace", "set_experiment", "start_run",
                 "initialize_logged_model", "clear_active_model", "log_param", "log_metric", "log_artifact", "log_artifacts"):
        monkeypatch.setattr(f"mlflow.{name}", getattr(recording, name))

    def failure_client(*, tracking_uri=None):
        assert tracking_uri is None or tracking_uri == recording.uri
        return recording

    monkeypatch.setattr("mlflow.MlflowClient", failure_client)
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: pytest.fail("No tracking server in offline tests"))
    return recording


@pytest.fixture
def execution(config_file, images, tmp_path, monkeypatch, request, tracking, execution_repo):
    from torchvision.transforms import ToTensor

    _, config = config_file
    repository, experiment = execution_repo
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runner, "EXPERIMENT_DIR", experiment)
    path = experiment / "configs/test.yaml"
    baseline, layers = tiny_model()
    config["tracking"]["uri"] = "https://configured.example.invalid"
    config["data"] = {"raw_dir": str(images[0]), "manifest": str(images[1])}
    config["compression"]["layers"] = layers
    config["runtime"].update(device=getattr(request, "param", "cpu"), batch_size=3, threads=1)
    config["timing"].update(input_shape=[1, 3, 4, 4], warmup=1, repeats=2)
    config["output_root"] = str(tmp_path / "outputs")
    config["cache"] = {"torch_home": str(tmp_path / "configured-torch")}
    path.write_text(yaml.safe_dump(config))
    effective, original = runner.load_config(path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://conflicting.example.invalid")
    events = tracking.events
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "conflicting-cache"))
    hub = Path(effective["cache"]["torch_home"]) / "hub"
    weight_path = hub / "checkpoints/resnet18-f37072fd.pth"
    def weights():
        assert Path(torch.hub.get_dir()) == hub  # Configure before preflight.
        return weight_path, {"identifier": "tiny fixture", "checkpoint_path": str(weight_path)}
    monkeypatch.setattr(runner, "cached_weights", weights)

    def load(path, device):
        assert path == weight_path and Path(torch.hub.get_dir()) == hub
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
@pytest.mark.parametrize("compression", ["absent", None, "Tucker", "TT"])
def test_complete_execution_tracks_and_measures_one_model(execution, capsys, monkeypatch, tracking, compression):
    config, original, baseline, events = execution
    selected_yaml = runner.EXPERIMENT_DIR / "configs/test.yaml"
    requested = yaml.safe_load(original)
    if compression == "absent":
        requested.pop("compression")
    elif compression is None:
        requested["compression"] = None
    elif compression == "TT":
        requested["compression"]["layers"]["0"] = {"factorization": "TT", "rank": 2}
    selected_yaml.write_text(yaml.safe_dump(requested))
    config, original = runner.load_config(selected_yaml)
    compressed = config["compression"] is not None
    if not compressed:
        monkeypatch.setattr(tltorch, "compress_model", lambda *a, **k: pytest.fail("Dense execution must not convert"))
    fingerprint = runner.model_fingerprint(baseline)
    device = torch.device(config["runtime"]["device"])
    forward_devices, timing_devices, evaluated_models, timed_models = [], [], [], []
    evaluate, measure_latency = runner.evaluate, runner.measure_latency

    def check_forward_device(model, args):
        assert args[0].device == device
        assert all(t.device == device for t in [*model.parameters(), *model.buffers()])
        forward_devices.append(args[0].device)

    def checked_evaluation(model, batches, selected_device):
        assert (model is baseline) == (not compressed)
        evaluated_models.append(model)
        handle = model.register_forward_pre_hook(check_forward_device)
        try:
            return evaluate(model, batches, selected_device)
        finally:
            handle.remove()

    def checked_latency(model, inputs, **kwargs):
        assert (model is baseline) == (not compressed)
        timed_models.append(model)
        check_forward_device(model, (inputs,))
        timing_devices.append(inputs.device)
        return measure_latency(model, inputs, **kwargs)

    monkeypatch.setattr(runner, "evaluate", checked_evaluation)
    monkeypatch.setattr(runner, "measure_latency", checked_latency)
    client = tracking
    output = runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
    assert events == ["uri https://configured.example.invalid", "workspace quirc", "experiment model-compression", "start", "upload effective_config.json", "upload original_config.yaml",
                      "upload run.json", "upload source.json", "upload source.zip", "load baseline",
                      "upload evaluation_manifest.json", "upload provenance.json"] + (["compress"] if compressed else []) + [
                      "initialize model", "log model", "reload model", "all artifacts", "FINISHED"]
    assert len(evaluated_models) == len(timed_models) == 1
    assert evaluated_models[0] is timed_models[0]
    assert tracking.run_count == tracking.model_count == 1 and tracking.active_run_id is None
    assert tracking.uri == config["tracking"]["uri"]
    assert tracking.run_name == config["tracking"]["run_name"]
    run = json.loads(client.uploads["run.json"])
    assert run["workspace"] == "quirc"
    assert json.loads(client.uploads["effective_config.json"])["tracking"] == config["tracking"]
    assert yaml.safe_load(client.uploads["original_config.yaml"])["tracking"]["workspace"] == "quirc"
    assert run["experiment_id"] == "experiment-1" and run["run_id"] == "run-1"
    assert run["mlflow_model"]["model_id"] == "model-1"
    assert json.loads(client.uploads["mlflow_model_verification.json"])["passed"]
    assert json.loads(client.uploads["execution_inputs_verification.json"])["matches_recorded_commit"]
    printed = capsys.readouterr().out
    assert "quirc" in printed and "experiment-1" in printed and "run-1" in printed and str(output) in printed
    source = json.loads(client.uploads["source.json"])
    assert client.tags == {"tracking.workspace": "quirc",
                           "git.branch": runner.git("branch", "--show-current").strip(),
                           "git.commit": runner.git("rev-parse", "HEAD").strip()}
    assert source["working_tree_status"] == runner.git("status", "--short", "--untracked-files=all")
    provenance = json.loads(client.uploads["provenance.json"])
    assert provenance["git"]["working_tree_status"] == source["working_tree_status"]
    assert provenance["cache"] == {"torch_home": config["cache"]["torch_home"],
                                   "hub_dir": str(Path(config["cache"]["torch_home"]) / "hub")}
    assert provenance["weights"]["checkpoint_path"] == str(
        Path(provenance["cache"]["hub_dir"]) / "checkpoints/resnet18-f37072fd.pth")
    # The fixture uses a custom YAML, so hashing the repository example would be wrong.
    assert original != (runner.EXPERIMENT_DIR / "configs/resnet18_exploration.yaml").read_bytes()
    for filename, key in (("original_config.yaml", "original_config_file_sha256"),
                          ("effective_config.json", "effective_config_file_sha256")):
        digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        assert hashlib.sha256(client.uploads[filename]).hexdigest() == digest
        assert run[key] == provenance[key] == digest
    assert provenance["manifest_sha256"] == hashlib.sha256(client.uploads["evaluation_manifest.json"]).hexdigest()
    assert source["source_archive_sha256"] == hashlib.sha256(client.uploads["source.zip"]).hexdigest()
    assert runner.model_fingerprint(baseline) == fingerprint
    run_parameters, model_parameters = runner.requested_parameters(config)
    assert client.params == run_parameters and client.model_params == model_parameters
    assert not client.params.keys() & client.model_params.keys()
    assert provenance["seeds"]["timing"] == config["timing"]["seed"]
    assert json.loads(client.uploads["effective_config.json"])["runtime"]["workers"] == config["runtime"]["workers"]
    absolute = ("top1_percent", "top5_percent", "cross_entropy_mean", "parameter_count",
                "parameter_bytes", "latency_median_ms", "latency_p25_ms", "latency_p75_ms")
    measurements = json.loads(client.uploads["measurements.json"])
    # Conv: 2*3 + 2; linear: 6*32 + 6. The nonpersistent float32 buffer adds 4 bytes.
    original_storage = {"parameter_count": 206, "parameter_bytes": 824, "buffer_bytes": 4}
    if compressed:
        report = json.loads(client.uploads["compression_report.json"])
        assert measurements["storage"] == report["model"]["compressed"]
        assert report["model"]["original"] == original_storage
        assert all(set(entry["weight_error"]) == {"absolute", "relative"} for entry in report["replacements"])
    else:
        assert "compression_report.json" not in client.uploads
        assert measurements["storage"] == original_storage
        assert client.model_params == {"model": config["model"], "weights": config["weights"]}
    row, = csv.DictReader(client.uploads["results.csv"].decode().splitlines())
    assert measurements["variant"] == row["variant"] == ("compressed" if compressed else "dense")
    expected_metrics = {}
    for key in absolute:
        if key.startswith("latency_"):
            value = measurements["latency"][key.removeprefix("latency_")]
        elif key in ("parameter_count", "parameter_bytes"):
            value = measurements["storage"][key]
        else:
            value = measurements["accuracy"][key]
        if key.startswith("latency_"):
            assert float(row[key]) == round(value, 3)
        elif key.endswith("_percent"):
            assert float(row[key]) == round(value, 2)
        else:
            assert float(row[key]) == value
        expected_metrics[key] = float(value)
    expected_owners = {key: "model-1" for key in absolute}
    if compressed:
        expected_metrics["conversion_seconds"] = measurements["conversion_seconds"]
        expected_owners["conversion_seconds"] = None
        assert float(row["conversion_seconds"]) == measurements["conversion_seconds"]
    else:
        assert "conversion_seconds" not in measurements and "conversion_seconds" not in row
    assert client.metrics == expected_metrics and client.metric_models == expected_owners
    assert len(client.metrics) == (9 if compressed else 8)
    accuracy = measurements["accuracy"]
    assert accuracy["examples"] == 4
    for key in ("examples", "top1_correct", "top5_correct", "macro_precision", "macro_recall", "macro_f1", "macro_classes"):
        assert float(row[key]) == accuracy[key]
    assert float(row["buffer_bytes"]) == measurements["storage"]["buffer_bytes"]
    assert json.loads(client.uploads["latency_samples.json"]) == measurements["latency"]["samples_ms"]
    assert len(measurements["latency"]["samples_ms"]) == config["timing"]["repeats"]
    assert "models" not in measurements
    assert not any("change_pp" in key for key in row)
    assert "Δ" not in client.uploads["report.md"].decode()
    assert (output / "original_config.yaml").read_bytes() == original
    assert (output / "evaluation_manifest.json").read_bytes() == Path(config["data"]["manifest"]).read_bytes()
    with zipfile.ZipFile(output / "source.zip") as archive:
        names = archive.namelist()
        assert "experiments/imagenet-compression/run_experiment.py" in names
        assert "experiments/imagenet-compression/configs/test.yaml" in names
        assert source["execution_inputs_match_commit"]
        for name in names:
            contents = archive.read(name)
            assert hashlib.sha256(contents).hexdigest() == source["source_files_sha256"][name]
            assert contents == subprocess.check_output(
                ["git", "show", f"{source['commit']}:{name}"], cwd=runner.REPOSITORY_ROOT)
        assert not any(n.startswith(("experiments/imagenet-compression/data/raw/",
                                                "experiments/imagenet-compression/data/processed/",
                                                ".venv/", "experiments/imagenet-compression/results/")) for n in names)
    assert len((output / "results.csv").read_text().splitlines()) == 2
    assert len(forward_devices) == 3  # Two evaluation batches and one timing check for the selected model.
    assert timing_devices == [device]
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
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
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
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
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
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
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

    def fail(*args, **kwargs):
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
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
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


@pytest.mark.parametrize("compressed", [False, True])
def test_serialization_failure_keeps_measurements_and_fails_the_run(execution, monkeypatch, tracking, compressed):
    config, original, _, events = execution
    if not compressed:
        selected = runner.EXPERIMENT_DIR / "configs/test.yaml"
        requested = yaml.safe_load(original)
        requested.pop("compression")
        selected.write_text(yaml.safe_dump(requested))
        config, original = runner.load_config(selected)

    def fail_save(model, report, inputs, output):
        (output / "model.pt").write_bytes(b"incomplete checkpoint")
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(runner, "save_and_verify_model", fail_save)
    client = tracking
    with pytest.raises(OSError, match="checkpoint write failed"):
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
    assert tracking.run_count == 1 and tracking.active_run_id is None
    assert events[-1] == "FAILED" and "FINISHED" not in events
    assert "model.pt" not in client.uploads and "all artifacts" not in events
    assert len(list(Path(config["output_root"]).glob("*/results.csv"))) == 1


def test_context_finish_failure_keeps_diagnostics_and_marks_the_same_run_failed(execution, tracking):
    config, original, _, events = execution
    tracking.fail = "finish"
    with pytest.raises(OSError, match="tracking connection lost"):
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
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


def test_requested_parameters_keep_rank_meaning_and_supplied_controls(config_file):
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
    run_parameters, model_parameters = runner.requested_parameters(effective)
    assert run_parameters == {
        "seed": "2026", "device": "cuda:0", "dtype": "float32", "evaluation.batch_size": "16",
        "runtime.threads": "4", "timing.input_shape": "[1, 3, 224, 224]", "timing.warmup": "20",
        "timing.repeats": "100",
    }
    assert model_parameters == {
        "model": "torchvision.resnet18", "weights": "ResNet18_Weights.IMAGENET1K_V1",
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
    }
    # YAML order must not change parameter strings or add library defaults.
    effective["compression"]["layers"] = dict(reversed(list(effective["compression"]["layers"].items())))
    assert runner.requested_parameters(effective) == (run_parameters, model_parameters)


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
        loaded = runner.load_compressed_model(tmp_path / "model.pt", device=location)
        assert runner.model_fingerprint(loaded) == expected_state
        assert runner.layer_identity(loaded, report) == expected_identity
        assert all(t.device == torch.device(location) for t in [*loaded.parameters(), *loaded.buffers()])
        with torch.inference_mode():
            outputs.append(loaded(inputs.to(location)).cpu())
    # Identical float32 weights may accumulate in different orders on CPU and CUDA.
    # Use the existing checkpoint forward tolerance, not factor-by-factor decomposition equality.
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("failure", ["model initialization", "model logging", "model verification"])
def test_mlflow_model_errors_keep_failure_record_and_never_finish(execution, tracking, failure):
    config, original, baseline, events = execution
    tracking.fail = failure
    before = runner.model_fingerprint(baseline)
    with pytest.raises(RuntimeError, match="MLflow model"):
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
    assert "FINISHED" not in events and events[-2:] == ["upload failure.json", "FAILED"]
    assert "failure.json" in tracking.uploads and "all artifacts" not in events
    assert "model_verification.json" not in tracking.uploads  # No final artifact batch.
    assert runner.model_fingerprint(baseline) == before


def test_commit_failure_prevents_any_tracking_or_model_execution(execution):
    config, original, _, events = execution
    hook = runner.REPOSITORY_ROOT / ".git/hooks/pre-commit"
    hook.write_text("#!/bin/sh\necho 'review required' >&2\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(RuntimeError, match="Cannot safely snapshot.*review required"):
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
    assert events == []
    failure, = Path(config["output_root"]).glob("*/failure.json")
    assert json.loads(failure.read_text())["run_id"] is None


def test_modified_execution_source_is_reported_without_hiding_original_failure(execution, monkeypatch, tracking):
    config, original, _, events = execution
    original_error = ValueError("decomposition failed")
    def fail(*args, **kwargs):
        (runner.EXPERIMENT_DIR / "new_helper.py").write_text("changed = True\n")
        raise original_error
    monkeypatch.setattr(tltorch, "compress_model", fail)
    with pytest.raises(ValueError) as caught:
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
    assert caught.value is original_error
    failure = json.loads(tracking.uploads["failure.json"])
    assert "new_helper.py" in failure["execution_input_error"]
    assert events[-1] == "FAILED"


@pytest.mark.parametrize("factorization", [None, "Tucker", "TT"])
def test_real_mlflow_package_contains_code_and_reloads_in_fresh_process(tmp_path, monkeypatch, tracking, factorization):
    model, layers = tiny_model()
    if factorization == "TT":
        layers["0"] = {"factorization": "TT", "rank": 2}
    compressed, report = (model, None) if factorization is None else compress_model(model, layers)
    inputs = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 48
    before = runner.model_fingerprint(compressed)
    modes = [m.training for m in compressed.modules()]
    runner.write_json(tmp_path / "run.json", {"run_id": "run-1"})
    runner.save_and_verify_model(compressed, report, inputs, tmp_path)
    tracking.active_run_id = "run-1"
    check = runner.log_and_verify_mlflow_model(compressed, report, inputs, tmp_path)
    assert check["passed"] and check["model_id"] == "model-1"
    assert runner.model_fingerprint(compressed) == before
    assert [m.training for m in compressed.modules()] == modes
    assert all(t.device.type == "cpu" and t.dtype == torch.float32
               for t in [*compressed.parameters(), *compressed.buffers()])
    package = Path(check["model_uri"])
    metadata = yaml.safe_load((package / "MLmodel").read_text())
    assert metadata["signature"]["inputs"] and metadata["signature"]["outputs"]
    assert (package / "input_example.json").is_file()
    assert not list(package.rglob("*.pt2"))
    assert (package / "code/tltorch/__init__.py").is_file()
    assert not any("tests" in path.parts or "__pycache__" in path.parts
                   for path in (package / "code").rglob("*"))
    assert "tensorly==0.9.0" in (package / "requirements.txt").read_text()
    code = """
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import mlflow.pytorch
with patch("socket.socket.connect", side_effect=AssertionError("offline")):
    model = mlflow.pytorch.load_model(sys.argv[1], map_location="cpu", weights_only=False)
    import tltorch
    assert Path(tltorch.__file__).is_relative_to(Path(sys.argv[1]) / "code")
    with patch("tltorch.compress_model", side_effect=AssertionError("no decomposition")):
        check = torch.load(sys.argv[2], weights_only=True)
        with torch.inference_mode():
            torch.testing.assert_close(model(check["inputs"]), check["expected_outputs"], rtol=1e-5, atol=1e-6)
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(package), str(tmp_path / "verification_input.pt")],
                   cwd=tmp_path, check=True)


def test_new_source_and_selected_yaml_are_committed_before_tracking(execution, monkeypatch, tracking):
    config, original, _, events = execution
    helper = runner.EXPERIMENT_DIR / "new_helper.py"
    helper.write_text("value = 5\n")
    selected = runner.EXPERIMENT_DIR / "configs/test.yaml"
    before = runner.git("rev-parse", "HEAD").strip()
    select_uri = tracking.set_tracking_uri
    def check_commit(uri):
        assert runner.git("rev-parse", "HEAD").strip() != before
        assert runner.git("show", "HEAD:experiments/imagenet-compression/new_helper.py") == helper.read_text()
        committed_yaml = subprocess.check_output(
            ["git", "show", "HEAD:experiments/imagenet-compression/configs/test.yaml"], cwd=runner.REPOSITORY_ROOT)
        assert committed_yaml == original
        select_uri(uri)
    monkeypatch.setattr("mlflow.set_tracking_uri", check_commit)
    output = runner.run_experiment(config, original, config_path=selected)
    assert json.loads((output / "source.json").read_text())["commit"] == runner.git("rev-parse", "HEAD").strip()
    assert events[-1] == "FINISHED"


def test_source_change_during_final_upload_prevents_success(execution, monkeypatch, tracking):
    config, original, _, events = execution
    upload = tracking.log_artifacts
    def change_after_upload(path):
        upload(path)
        (runner.EXPERIMENT_DIR / "new_helper.py").write_text("changed = True\n")
    monkeypatch.setattr("mlflow.log_artifacts", change_after_upload)
    with pytest.raises(RuntimeError, match="changed after the recorded commit.*new_helper.py"):
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
    assert "FINISHED" not in events
    assert "new_helper.py" in json.loads(tracking.uploads["failure.json"])["execution_input_error"]


@pytest.mark.parametrize("change,message", [
    (None, "cache must be a mapping"),
    ({}, "cache: missing options.*torch_home"),
    ({"torch_home": "cache", "unknown": 1}, "cache:.*unknown options.*unknown"),
    ({"torch_home": None}, "cache.torch_home must be a nonempty path string"),
    ({"torch_home": 7}, "cache.torch_home must be a nonempty path string"),
    ({"torch_home": "  "}, "cache.torch_home must be a nonempty path string"),
    ({"torch_home": "bad\x00path"}, "Invalid cache.torch_home"),
])
def test_invalid_cache_settings_fail_during_configuration_loading(config_file, change, message):
    path, config = config_file
    config["cache"] = change
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        runner.load_config(path)


@pytest.mark.parametrize("configured", ["~/.cache/tltorch-model-compression/torch", "other-cache/torch"])
def test_cache_path_resolution_is_pure_and_uses_home_or_experiment_directory(config_file, configured, monkeypatch):
    path, config = config_file
    config["cache"] = {"torch_home": configured}
    path.write_text(yaml.safe_dump(config))
    original_hub = torch.hub.get_dir()
    monkeypatch.setattr(torch.hub, "set_dir", lambda *a: pytest.fail("Configuration loading must not set the cache"))
    effective, _ = runner.load_config(path)
    expected = (Path.home() / configured[2:] if configured.startswith("~/")
                else runner.EXPERIMENT_DIR / configured)
    assert effective["cache"]["torch_home"] == str(expected.resolve())
    assert torch.hub.get_dir() == original_hub


@pytest.mark.parametrize("ambient", [None, "conflicting-torch-home"])
def test_explicit_yaml_cache_wins_before_cached_weight_checks(config_file, tmp_path, monkeypatch, ambient):
    path, config = config_file
    config["cache"] = {"torch_home": str(tmp_path / "selected-cache")}
    path.write_text(yaml.safe_dump(config))
    if ambient is None:
        monkeypatch.delenv("TORCH_HOME", raising=False)
    else:
        monkeypatch.setenv("TORCH_HOME", str(tmp_path / ambient))
    # An existing explicit Hub override must also yield to the selected YAML.
    torch.hub.set_dir(str(tmp_path / "old-hub"))
    effective, _ = runner.load_config(path)
    cache = runner.configure_weight_cache(effective)
    hub = tmp_path / "selected-cache/hub"
    assert torch.hub.get_dir() == str(hub)
    assert cache == {"torch_home": str(hub.parent), "hub_dir": str(hub)}
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("No download"))
    with pytest.raises(FileNotFoundError) as caught:
        runner.cached_weights()
    assert str(hub / "checkpoints/resnet18-f37072fd.pth") in str(caught.value)
    assert "cache.torch_home" in str(caught.value)
    # Hash the small stand-in without loading a model or relaxing checksum validation.
    checkpoint = hub / "checkpoints/resnet18-f37072fd.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    with pytest.raises(ValueError, match="published checksum"):
        runner.cached_weights()
    monkeypatch.setattr(runner, "file_sha256", lambda p: "f37072fd" + "0" * 56)
    actual, provenance = runner.cached_weights()
    assert actual == checkpoint and provenance["checkpoint_path"] == str(checkpoint)


@pytest.mark.parametrize("explicit_hub", [False, True])
def test_absent_cache_section_preserves_pytorch_resolution(config_file, tmp_path, monkeypatch, explicit_hub):
    path, config = config_file
    config.pop("cache", None)
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "environment-cache"))
    monkeypatch.setattr(torch.hub, "_hub_dir", str(tmp_path / "caller-hub") if explicit_hub else None)
    effective, _ = runner.load_config(path)
    before = torch.hub.get_dir()
    cache = runner.configure_weight_cache(effective)
    assert "cache" not in effective
    assert torch.hub.get_dir() == before
    assert cache == {"torch_home": None, "hub_dir": str(Path(before).resolve())}
    expected = tmp_path / ("caller-hub" if explicit_hub else "environment-cache/hub")
    assert Path(before) == expected


def test_nonexecution_commands_do_not_change_global_hub_cache(config_file, tmp_path, monkeypatch):
    path, _ = config_file
    torch.hub.set_dir(str(tmp_path / "caller-hub"))
    before = torch.hub.get_dir()
    def forbidden(*args, **kwargs):
        pytest.fail("Nonexecution commands must not set the cache or contact tracking")
    monkeypatch.setattr(torch.hub, "set_dir", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    runner.load_config(path)
    assert runner.main(["--config", str(path), "--dry-run"]) == 0
    monkeypatch.setattr(runner, "prepare_data", lambda *a, **k: {"splits": {}})
    assert runner.main(["--prepare-data"]) == 0
    with pytest.raises(SystemExit) as caught:
        runner.main(["--help"])
    assert caught.value.code == 0
    assert torch.hub.get_dir() == before


@pytest.mark.parametrize("compressed", [False, True])
def test_real_local_mlflow_separates_parameters_and_metric_ownership(config_file, tmp_path, compressed):
    path, config = config_file
    config["compression"]["layers"]["fc"]["rank"] = 1
    config["compression"]["layers"]["layer4.0.conv2"]["rank"] = 1.0
    if not compressed:
        config.pop("compression")
    path.write_text(yaml.safe_dump(config))
    # A subprocess isolates MLflow's tracking URI, active model, and experiment state.
    # File storage is an opt-in fixture only; the runner still requires HTTP(S).
    environment = {**os.environ, "MLFLOW_TRACKING_URI": (tmp_path / "tracking").as_uri(),
                   "MLFLOW_ALLOW_FILE_STORE": "true", "MLFLOW_ENABLE_TELEMETRY": "false",
                   "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("MLFLOW_RUN_ID", None)
    code = """
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, sys.argv[1])
with patch("socket.socket.connect", side_effect=AssertionError("Temporary local storage only")):
    import torch
    import mlflow
    import run_experiment as runner

    config, _ = runner.load_config(sys.argv[2])
    run_parameters, model_parameters = runner.requested_parameters(config)
    experiment = mlflow.set_experiment("offline-dashboard")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        for key, value in run_parameters.items():
            mlflow.log_param(key, value, synchronous=True)
        output = Path(sys.argv[3]) / "output"
        output.mkdir()
        runner.write_json(output / "run.json", {"run_id": run.info.run_id})
        # This checks tracking destinations, not decomposition or scientific measurements.
        check = runner.log_and_verify_mlflow_model(
            torch.nn.Linear(2, 2).eval(), {"replacements": []}, torch.ones(1, 2), output,
            parameters=model_parameters)
        client = mlflow.MlflowClient()
        model_id = check["model_id"]
        record = client.get_logged_model(model_id)
        assert record.source_run_id == run.info.run_id
        assert record.params == model_parameters
        assert record.name == "model"
        if config["compression"] is not None:
            assert record.params["compression.fc.rank"] == "1"
            assert record.params["compression.layer4.0.conv2.rank"] == "1.0"
        else:
            assert record.params == {"model": config["model"], "weights": config["weights"]}
        assert client.get_run(run.info.run_id).data.params == run_parameters
        assert not run_parameters.keys() & record.params.keys()

        absolute = ("top1_percent", "top5_percent", "cross_entropy_mean", "parameter_count",
                    "parameter_bytes", "latency_median_ms", "latency_p25_ms", "latency_p75_ms")
        row = {key: index + 1 for index, key in enumerate(absolute)}
        expected_model_metrics = {key: float(value) for key, value in row.items()}
        expected_metrics = dict(expected_model_metrics)
        if config["compression"] is not None:
            row["conversion_seconds"] = 0.25
            expected_metrics["conversion_seconds"] = 0.25
        # Deliberately enable the SDK's implicit model default to exercise the guard.
        mlflow.set_active_model(model_id=model_id)
        runner.log_model_metrics(row, run.info.run_id, model_id)
        assert mlflow.get_active_model_id() is None
        assert client.get_run(run.info.run_id).data.metrics == expected_metrics
        model_metrics = client.get_logged_model(model_id).metrics
        assert {metric.key: metric.value for metric in model_metrics} == expected_model_metrics
        assert all(metric.model_id == model_id and metric.run_id == run.info.run_id for metric in model_metrics)
    assert client.get_run(run.info.run_id).info.status == "FINISHED"
    models = client.search_logged_models(experiment_ids=[experiment.experiment_id])
    assert len(models) == 1 and models[0].model_id == model_id
    assert len(client.search_runs([experiment.experiment_id])) == 1
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(runner.EXPERIMENT_DIR), str(path), str(tmp_path)],
                   env=environment, cwd=tmp_path, check=True)


def test_metric_upload_failure_keeps_original_error_and_diagnostics(execution, monkeypatch, tracking):
    config, original, _, events = execution
    error = OSError("metric upload failed")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(tracking, "log_batch", fail)
    with pytest.raises(OSError) as caught:
        runner.run_experiment(config, original, config_path=runner.EXPERIMENT_DIR / "configs/test.yaml")
    assert caught.value is error
    assert events[-2:] == ["upload failure.json", "FAILED"] and "FINISHED" not in events
    assert json.loads(tracking.uploads["failure.json"])["error"] == "metric upload failed"



def test_optional_receipt_and_extra_configurations_complete_after_run_exit(execution, monkeypatch, tracking, tmp_path):
    config, original, _, events = execution
    selected = runner.EXPERIMENT_DIR / "configs/test.yaml"
    status_path = tmp_path / "runner_status.json"
    extras = []
    for folder, contents in (("studies", "study: example\n"), ("templates", "template: dense\n")):
        path = runner.EXPERIMENT_DIR / f"configs/{folder}/settings.yaml"
        path.parent.mkdir()
        path.write_text(contents)
        extras.append(path)
    unrelated = runner.EXPERIMENT_DIR / "configs/personal.yaml"
    unrelated.write_text("personal: do not collect\n")
    load = runner.load_cached_baseline
    def check_early_uploads(*args):
        receipt = json.loads(status_path.read_text())
        assert receipt["state"] == "running" and receipt["phase"] == "model_loading"
        archive = zipfile.ZipFile(io.BytesIO(tracking.uploads["source.zip"]))
        for path in extras:
            name = path.relative_to(runner.REPOSITORY_ROOT).as_posix()
            assert tracking.uploads[f"provenance_configs/{name}"] == path.read_bytes()
            assert archive.read(name) == path.read_bytes()
            assert archive.namelist().count(name) == 1
        assert archive.namelist().count(selected.relative_to(runner.REPOSITORY_ROOT).as_posix()) == 1
        assert unrelated.relative_to(runner.REPOSITORY_ROOT).as_posix() not in archive.namelist()
        return load(*args)
    monkeypatch.setattr(runner, "load_cached_baseline", check_early_uploads)
    upload = tracking.log_artifacts
    def check_final_upload(path):
        receipt = json.loads(status_path.read_text())
        assert receipt["state"] == "running" and receipt["phase"] == "artifact_upload"
        assert json.loads((Path(path) / "model_verification.json").read_text())["passed"]
        assert json.loads((Path(path) / "mlflow_model_verification.json").read_text())["passed"]
        upload(path)
    monkeypatch.setattr("mlflow.log_artifacts", check_final_upload)
    terminate = tracking.set_terminated
    def check_context_exit(run_id, status):
        receipt = json.loads(status_path.read_text())
        assert status == "FINISHED" and receipt["state"] == "running"
        assert receipt["phase"] == "run_finalization" and events[-1] == "all artifacts"
        terminate(run_id, status)
    monkeypatch.setattr(tracking, "set_terminated", check_context_exit)
    arguments = ["--config", str(selected), "--status-file", str(status_path)]
    for path in [*extras, extras[0], selected]:
        arguments += ["--provenance-config", str(path)]
    assert runner.main(arguments) == 0
    receipt = json.loads(status_path.read_text())
    assert receipt["schema_version"] == 1 and receipt["state"] == "succeeded"
    assert receipt["phase"] == "completed" and receipt["failure"] is None
    assert receipt["configuration_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["configuration_path"] == str(selected)
    assert receipt["process"]["pid"] == os.getpid() and receipt["process"]["hostname"]
    assert receipt["tracking_uri"] == config["tracking"]["uri"] and receipt["workspace"] == "quirc"
    assert (receipt["experiment_id"], receipt["run_id"], receipt["model_id"]) == ("experiment-1", "run-1", "model-1")
    output = Path(receipt["output_directory"])
    assert receipt["model_uri"] == json.loads((output / "run.json").read_text())["mlflow_model"]["model_uri"]
    for name, entry in receipt["artifacts"].items():
        assert Path(entry["path"]) == output / name
        assert entry["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()
    assert tracking.run_count == tracking.model_count == 1 and events[-1] == "FINISHED"
    assert set(tracking.params) == {"device", "dtype", "evaluation.batch_size", "runtime.threads",
                                    "seed", "timing.input_shape", "timing.repeats", "timing.warmup"}
    assert set(tracking.tags) == {"tracking.workspace", "git.branch", "git.commit"}


@pytest.mark.parametrize("failure,phase", [
    ("preflight", "preflight"), ("workspace", "tracking_setup"),
    ("early upload", "early_uploads"),
    ("conversion", "conversion"), ("checkpoint", "checkpoint_verification"),
    ("model initialization", "mlflow_model_verification"),
    ("model logging", "mlflow_model_verification"),
    ("model verification", "mlflow_model_verification"),
    ("late upload", "artifact_upload"), ("finish", "run_finalization"),
])
def test_failed_execution_never_publishes_success_receipt(execution, monkeypatch, tracking, tmp_path, failure, phase):
    config, _, _, events = execution
    tracking.fail = failure
    if failure == "preflight":
        def missing_weights():
            raise FileNotFoundError("missing cached weights")
        monkeypatch.setattr(runner, "cached_weights", missing_weights)
    elif failure == "workspace":
        def inaccessible_workspace(value):
            raise PermissionError("workspace unavailable")
        monkeypatch.setattr("mlflow.set_workspace", inaccessible_workspace)
    elif failure == "conversion":
        def fail_conversion(*args, **kwargs):
            raise RuntimeError("conversion failed") from ValueError("fit failed")
        monkeypatch.setattr(tltorch, "compress_model", fail_conversion)
    elif failure == "checkpoint":
        def fail_checkpoint(*args):
            raise RuntimeError("checkpoint verification failed")
        monkeypatch.setattr(runner, "save_and_verify_model", fail_checkpoint)
    status_path = tmp_path / "runner_status.json"
    assert runner.main(["--config", str(runner.EXPERIMENT_DIR / "configs/test.yaml"),
                        "--status-file", str(status_path)]) == 1
    receipt = json.loads(status_path.read_text())
    assert receipt["state"] == "failed" and receipt["phase"] == phase
    assert "artifacts" not in receipt and receipt["failure"]["exception_type"].startswith("builtins.")
    if failure == "preflight":
        assert receipt["output_directory"] is None and not events
        assert all(receipt[key] is None for key in ("experiment_id", "run_id", "model_id", "model_uri"))
    elif failure == "workspace":
        assert receipt["output_directory"] is not None and receipt["run_id"] is None
        assert receipt["experiment_id"] is None and receipt["model_id"] is None
        assert "start" not in events
    else:
        assert receipt["run_id"] == "run-1" and events[-1] == "FAILED"
        assert json.loads((Path(receipt["output_directory"]) / "failure.json").read_text())["run_id"] == "run-1"
        if failure in ("early upload", "conversion", "checkpoint", "model initialization"):
            assert receipt["model_id"] is None and receipt["model_uri"] is None
        elif failure == "model logging":
            assert receipt["model_id"] == "model-1" and receipt["model_uri"] is None
        else:
            assert receipt["model_id"] == "model-1" and receipt["model_uri"]
    if failure == "conversion":
        assert receipt["failure"]["cause_types"] == ["builtins.ValueError"]


def test_failure_receipt_write_error_preserves_original_exception_and_cause(execution, monkeypatch, tmp_path, capsys):
    import runner_status
    original, cause = RuntimeError("original conversion error"), ValueError("original cause")
    def fail_conversion(*args, **kwargs):
        raise original from cause
    monkeypatch.setattr(tltorch, "compress_model", fail_conversion)
    update = runner_status.update_status
    def reject_failure(receipt, **changes):
        if changes.get("state") == "failed":
            raise OSError("receipt write failed")
        update(receipt, **changes)
    monkeypatch.setattr(runner_status, "update_status", reject_failure)
    fail = runner.fail_status
    observed = []
    def record_error(receipt, error):
        observed.append(error)
        fail(receipt, error)
    monkeypatch.setattr(runner, "fail_status", record_error)
    status_path = tmp_path / "runner_status.json"
    assert runner.main(["--config", str(runner.EXPERIMENT_DIR / "configs/test.yaml"),
                        "--status-file", str(status_path)]) == 1
    assert observed == [original] and original.__cause__ is cause
    assert json.loads(status_path.read_text())["state"] == "running"
    stderr = capsys.readouterr().err
    assert "Failure receipt could not be updated" in stderr and "original conversion error" in stderr


def test_final_receipt_write_failure_keeps_finished_run_and_artifacts(execution, monkeypatch, tracking, tmp_path):
    import runner_status
    update = runner_status.update_status
    def reject_success(receipt, **changes):
        if changes.get("state") == "succeeded":
            raise OSError("final receipt write failed")
        update(receipt, **changes)
    monkeypatch.setattr(runner_status, "update_status", reject_success)
    status_path = tmp_path / "runner_status.json"
    assert runner.main(["--config", str(runner.EXPERIMENT_DIR / "configs/test.yaml"),
                        "--status-file", str(status_path)]) == 1
    receipt = json.loads(status_path.read_text())
    assert receipt["state"] == "failed" and receipt["phase"] == "completion_receipt"
    assert receipt["failure"]["exception_type"] == "builtins.OSError"
    assert tracking.events[-1] == "FINISHED" and "FAILED" not in tracking.events
    output = Path(receipt["output_directory"])
    assert not (output / "failure.json").exists() and (output / "model.pt").is_file()
    assert "model.pt" in tracking.uploads


def test_existing_status_destination_is_preserved_before_snapshot(execution, tmp_path, tracking):
    status_path = tmp_path / "runner_status.json"
    status_path.write_text("existing owner's file\n")
    before = runner.git("rev-parse", "HEAD")
    assert runner.main(["--config", str(runner.EXPERIMENT_DIR / "configs/test.yaml"),
                        "--status-file", str(status_path)]) == 1
    assert status_path.read_text() == "existing owner's file\n"
    assert runner.git("rev-parse", "HEAD") == before and tracking.events == []


def test_extra_configuration_changed_during_preflight_prevents_snapshot(execution, monkeypatch, tracking, tmp_path):
    extra = runner.EXPERIMENT_DIR / "configs/study.yaml"
    extra.write_text("study: original\n")
    weights = runner.cached_weights
    def change_extra():
        extra.write_text("study: changed before snapshot\n")
        return weights()
    monkeypatch.setattr(runner, "cached_weights", change_extra)
    before = runner.git("rev-parse", "HEAD")
    status_path = tmp_path / "runner_status.json"
    assert runner.main(["--config", str(runner.EXPERIMENT_DIR / "configs/test.yaml"),
                        "--provenance-config", str(extra), "--status-file", str(status_path)]) == 1
    assert runner.git("rev-parse", "HEAD") == before and tracking.events == []
    assert json.loads(status_path.read_text())["state"] == "failed"


def test_extra_configuration_changed_after_upload_prevents_success(execution, monkeypatch, tracking, tmp_path):
    extra = runner.EXPERIMENT_DIR / "configs/study.yaml"
    extra.write_text("study: original\n")
    upload = tracking.log_artifacts
    def change_extra(path):
        upload(path)
        extra.write_text("study: changed during final upload\n")
    monkeypatch.setattr("mlflow.log_artifacts", change_extra)
    status_path = tmp_path / "runner_status.json"
    assert runner.main(["--config", str(runner.EXPERIMENT_DIR / "configs/test.yaml"),
                        "--provenance-config", str(extra), "--status-file", str(status_path)]) == 1
    assert json.loads(status_path.read_text())["state"] == "failed"
    assert "configs/study.yaml" in json.loads(tracking.uploads["failure.json"])["execution_input_error"]
    assert tracking.events[-1] == "FAILED" and "FINISHED" not in tracking.events


@pytest.mark.parametrize("variant,late_failure", [
    ("dense", None), ("compressed", None),
    ("compressed", "late upload"), ("dense", "finish"),
])
def test_actual_runner_cli_artifacts_reach_controller_collector(execution_repo, tmp_path, monkeypatch,
                                                               variant, late_failure):
    """Use actual producer files, real child exits, and the existing offline execution fixtures."""
    pytest.importorskip("optuna")
    import optuna_controller as controller

    root, experiment = execution_repo
    tests_directory = Path(__file__).parent
    work = tmp_path / "handoff"
    work.mkdir()
    driver = work / "runner-driver.py"
    driver.write_text(r'''import json,sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
sys.path[:0] = [sys.argv[1], sys.argv[2], sys.argv[3]]
import pytest,torch,yaml
import test_run_experiment as fixtures
import optuna_controller as controller
runner = fixtures.runner
root,experiment,work = Path(sys.argv[2]),Path(sys.argv[1]),Path(sys.argv[4])
variant,late_failure = sys.argv[5],json.loads(sys.argv[6])
assert Path(runner.__file__) == experiment / 'run_experiment.py'
assert Path(fixtures.tltorch.__file__) == root / 'tltorch/__init__.py'
with pytest.MonkeyPatch.context() as patch:
    tracking = fixtures.tracking.__wrapped__(patch, work)
    config_file = fixtures.config_file.__wrapped__(work)
    images = fixtures.images.__wrapped__(work)
    _,original,baseline,_ = fixtures.execution.__wrapped__(
        config_file, images, work, patch, SimpleNamespace(param='cpu'), tracking, (root,experiment))
    # Three artificial examples, one with label zero. A dominant, preserved bias
    # makes that one correct in both variants, giving exactly 100/3 percent.
    with torch.no_grad():
        baseline[3].bias.zero_()
        baseline[3].bias[0] = 100
    manifest = json.loads(images[1].read_text())
    manifest['examples'] = manifest['examples'][:3]
    images[1].write_text(json.dumps(manifest))
    requested = yaml.safe_load(original)
    if variant == 'dense':
        requested['compression'] = None
    selected = experiment / 'configs/test.yaml'
    selected.write_text(yaml.safe_dump(requested,sort_keys=False))
    base = deepcopy(requested)
    base['compression'] = None
    template = experiment / 'configs/template.yaml'
    template.write_text(yaml.safe_dump(base,sort_keys=False))
    study = experiment / 'configs/study.yaml'
    study.write_text('purpose: isolated runner-to-collector integration\n')
    context = {'path': study, 'base_path': template, 'base': runner.load_config(template)[0],
               'definition': {'purpose': 'synthetic handoff'}, 'state_dir': work / 'unused-state'}
    contract = controller.compatibility(context)
    runner.write_json(work / 'preflight.json', {'contract': contract,
        'base': context['base'], 'study': str(study), 'template': str(template)})
    tracking.fail = late_failure
    code = runner.main(['--config',str(selected),'--status-file',str(work / 'runner_status.json'),
                        '--provenance-config',str(study),'--provenance-config',str(template)])
    runner.write_json(work / 'tracking-events.json',tracking.events)
raise SystemExit(code)
''')
    command = [sys.executable, "-B", str(driver), str(experiment), str(root), str(tests_directory),
               str(work), variant, json.dumps(late_failure)]
    with (work / "runner.stdout").open("wb") as out, (work / "runner.stderr").open("wb") as err:
        process = subprocess.Popen(command, cwd=root, stdout=out, stderr=err, start_new_session=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root)})
        identity = controller.process_identity(process.pid)
        try:
            code = process.wait(timeout=80)
        finally:
            if process.poll() is None:
                import signal
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
    assert code == (0 if late_failure is None else 1), (work / "runner.stderr").read_text()
    metadata = controller.read_json(work / "preflight.json")
    context = {"base": metadata["base"], "path": Path(metadata["study"]),
               "base_path": Path(metadata["template"])}
    selected = experiment / "configs/test.yaml"
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(runner, "EXPERIMENT_DIR", experiment)
    effective, _ = runner.load_config(selected)
    record = {"directory": str(work), "trial_number": 0, "configuration_path": str(selected),
              "configuration_sha256": runner.file_sha256(selected), "status_path": str(work / "runner_status.json"),
              "exit_code": code, "process": identity,
              "layers": effective["compression"]["layers"] if effective["compression"] else {}}
    controller.atomic_json(work / "attempt.json", record)
    receipt = controller.read_json(work / "runner_status.json")
    output = Path(receipt["output_directory"])
    measurements = controller.read_json(output / "measurements.json")
    assert measurements["accuracy"]["top1_percent"] == 100 / 3
    assert measurements["storage"]["parameter_count"] == (206 if variant == "dense" else 93)
    events = controller.read_json(work / "tracking-events.json")
    assert "reload model" in events and "all artifacts" in events
    if late_failure is not None:
        assert receipt["state"] == "failed" and (output / "failure.json").is_file()
        with pytest.raises(RuntimeError, match="No confirmed successful child exit"):
            controller.collect_result(context, metadata["contract"], record)
        # A forged exit zero still cannot promote the actual failed receipt.
        with pytest.raises(RuntimeError, match="partial.*receipt"):
            controller.collect_result(context, metadata["contract"], {**record, "exit_code": 0})
        return
    assert receipt["state"] == "succeeded" and events[-1] == "FINISHED"
    result = controller.collect_result(context, metadata["contract"], record)
    assert result["values"] == [100 / 3, 206 if variant == "dense" else 93]
    assert result["values"][0] != round(result["values"][0], 2)
    # Independently count the actual trusted checkpoint's parameters, without any fit.
    loaded = runner.load_model(output / "model.pt", "cpu")
    assert sum(p.numel() for p in loaded.parameters()) == result["values"][1]
    assert result["source_commit"] == runner.git("rev-parse", "HEAD").strip()
    assert result["run_id"] == receipt["run_id"] == "run-1"
    assert result["model_id"] == receipt["model_id"] == "model-1"
    assert result["model_uri"] == receipt["model_uri"]
    with zipfile.ZipFile(output / "source.zip") as archive:
        for path in (context["path"], context["base_path"]):
            name = path.relative_to(root).as_posix()
            assert archive.read(name) == path.read_bytes()
    for path in (context["path"], context["base_path"]):
        assert f"upload {path.name}" in events[:events.index("load baseline")]
    controller.atomic_json(work / "collected-result.json", result)
