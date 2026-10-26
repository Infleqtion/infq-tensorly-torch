#!/usr/bin/env python3
"""Prepare ImageNetV2 or evaluate one configured model in a tracked run.

Importing this file only defines helpers. Preparation and dry-run do not import
PyTorch, load models, or contact tracking. Optional notebooks reuse these helpers.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import tempfile
import traceback
from urllib.parse import urlparse
import uuid
import zipfile

import yaml

from execution_inputs import configuration_files, snapshot_execution_inputs, verify_execution_inputs
from runner_status import validate_status_path, start_status, update_status, fail_status, finish_status

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
DATASET = "ImageNetV2 matched-frequency (imagenetv2-b-33)"
EXPLORATION = EXPERIMENT_DIR / "results/cpu_1000/selected_examples.json"


DATASET_REVISION = "d626240be2538720e83103a0e1178d24aca8b12c"
DATASET_URL = ("https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/"
               f"{DATASET_REVISION}/imagenetv2-matched-frequency.tar.gz")
ARCHIVE_SHA256 = "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c"
VARIANTS = (("dense", None), ("fraction_0.75", 0.75), ("fraction_0.50", 0.50))
FLOPS = {
    "total": None,
    "counter": None,
    "multiply_add_convention": None,
    "unsupported_operations": None,
    "coverage": "not assessed",
    "reason": "No counter with verified coverage of factorized einsum/tensordot contractions "
              "was used. No complete or partial FLOP total is reported.",
    "intentional_exclusions": ["loading", "conversion", "preprocessing", "device transfers"],
}



class UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently replacing an earlier setting."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("YAML keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _keys(value, required, optional, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{where}: missing options {sorted(missing)}; unknown options {sorted(unknown)}")


def _integer(value, minimum, where, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{where} must be an integer >= {minimum}" +
                         (f" and <= {maximum}" if maximum is not None else ""))


def experiment_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be nonempty strings")
    path = Path(value).expanduser()
    return (EXPERIMENT_DIR / path).resolve()


def load_config(path):
    """Return the effective configuration and original bytes; YAML paths use EXPERIMENT_DIR; the config filename uses the caller's directory."""
    original = Path(path).expanduser().read_bytes()
    config = yaml.load(original, Loader=UniqueSafeLoader)
    _keys(config, ("model", "weights", "data", "runtime", "timing", "tracking", "output_root"), ("compression", "cache"), "configuration")
    if config["model"] != "torchvision.resnet18" or config["weights"] != "ResNet18_Weights.IMAGENET1K_V1":
        raise ValueError("This runner supports torchvision.resnet18 with ResNet18_Weights.IMAGENET1K_V1")
    _keys(config["data"], ("raw_dir", "manifest"), (), "data")
    for key in ("raw_dir", "manifest"):
        config["data"][key] = str(experiment_path(config["data"][key]))
    config["output_root"] = str(experiment_path(config["output_root"]))
    if "cache" in config:
        _keys(config["cache"], ("torch_home",), (), "cache")
        value = config["cache"]["torch_home"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("cache.torch_home must be a nonempty path string")
        try:
            config["cache"]["torch_home"] = str(experiment_path(value))
        except (ValueError, RuntimeError, OSError) as error:
            raise ValueError(f"Invalid cache.torch_home: {error}") from error

    compression = config.setdefault("compression", None)
    if compression is not None:
        _keys(compression, ("implementation", "layers"), (), "compression")
        if compression["implementation"] not in ("factorized", "reconstructed"):
            raise ValueError("compression.implementation must be factorized or reconstructed")
        if not isinstance(compression["layers"], dict) or not compression["layers"]:
            raise ValueError("compression.layers must be a nonempty mapping")
        # Catch misspellings here; layer compatibility and decomposition rules belong to compress_model.
        optional = ("fixed_rank_modes", "in_tensorized_features", "out_tensorized_features",
                    "max_parameters", "decomposition_kwargs")
        for name, settings in compression["layers"].items():
            if not isinstance(name, str):
                raise ValueError("Layer names must be strings")
            _keys(settings, ("factorization", "rank"), optional, f"layer {name!r}")
            if settings["factorization"] not in ("CP", "Tucker", "TT", "BlockTT"):
                raise ValueError(f"Unknown factorization for layer {name!r}")
            rank = settings["rank"]
            valid_rank = ((type(rank) in (int, float) and math.isfinite(rank) and rank > 0)
                          or rank == "same" or (isinstance(rank, list) and bool(rank)
                                                and all(type(n) is int and n > 0 for n in rank)))
            if not valid_rank:
                raise ValueError(f"Layer {name!r}: rank must be positive numeric, an integer list, or 'same'")
            if "decomposition_kwargs" in settings:
                _keys(settings["decomposition_kwargs"], (), ("init", "n_iter_max", "tol", "random_state", "l2_reg"),
                      f"layer {name!r} decomposition_kwargs")

    runtime = config["runtime"]
    _keys(runtime, ("seed", "device", "dtype", "batch_size", "workers", "threads"), (), "runtime")
    if not isinstance(runtime["device"], str) or not re.fullmatch(r"cpu|cuda:\d+", runtime["device"]):
        raise ValueError("runtime.device must be cpu or an explicit CUDA device such as cuda:0")
    if runtime["dtype"] != "float32":
        raise ValueError("This initial runner supports float32 only")
    for key, minimum in (("seed", 0), ("batch_size", 1), ("workers", 0), ("threads", 1)):
        _integer(runtime[key], minimum, f"runtime.{key}", 2**32 - 1 if key == "seed" else None)
    timing = config["timing"]
    _keys(timing, ("seed", "input_shape", "warmup", "repeats"), (), "timing")
    for key, minimum in (("seed", 0), ("warmup", 0), ("repeats", 2)):
        _integer(timing[key], minimum, f"timing.{key}", 2**32 - 1 if key == "seed" else None)
    shape = timing["input_shape"]
    if (not isinstance(shape, list) or len(shape) != 4
            or any(type(n) is not int or n < 1 for n in shape) or shape[1] != 3):
        raise ValueError("timing.input_shape must be four positive integers [batch, 3, height, width]")
    _keys(config["tracking"], ("uri", "workspace", "experiment_name", "run_name"), (), "tracking")
    for key, value in config["tracking"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"tracking.{key} must be a nonempty string")
    validate_tracking_uri(config["tracking"]["uri"])
    return config, original


def describe(config):
    """Explain the plan and distinguish structural validation from execution checks."""
    import json

    selected = "compressed" if config["compression"] is not None else "dense pretrained"
    return (f"Configuration is valid. Planned evaluation: one {selected} ResNet-18 model.\n"
            "Uses the supplied manifest without resampling; eager execution, opt_einsum disabled.\n"
            "No model/weights loaded, images decoded, outputs created, or tracking calls made.\n"
            "Execution must still check input files/hashes, cached weights, CUDA availability, layer compatibility, "
            "MLflow access/uploads, and checkpoint loading.\n" + json.dumps(config, indent=2))


# Preparation is an explicit action; evaluation only consumes existing manifests.


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_inventory(directory, *, class_ids=range(1000)):
    """Validate numeric class folders and hash their original bytes in stable order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist: {directory}")
    folders = list(directory.iterdir())
    expected = {str(label) for label in class_ids}
    actual = {folder.name for folder in folders}
    if actual != expected:
        missing = sorted(expected - actual, key=int)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{directory}: expected {len(expected)} numeric class folders; "
                         f"missing {missing[:8]}, unexpected {unexpected[:8]} (first eight shown)")
    inventory = []
    for folder in sorted(folders, key=lambda path: int(path.name)):
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Expected an actual class directory: {folder}")
        paths = sorted(folder.iterdir())
        if len(paths) != 10:
            raise ValueError(f"Class {folder.name}: expected 10 images, found {len(paths)}")
        for path in paths:
            if (not path.is_file() or path.is_symlink()
                    or path.suffix.lower() not in (".jpeg", ".jpg", ".png")):
                raise ValueError(f"Expected an original image file, not a link or other entry: {path}")
            inventory.append({"path": path.relative_to(directory).as_posix(),
                              "label": int(folder.name), "sha256": file_sha256(path)})
    return inventory


def build_splits(inventory, exploration_bytes, *, seed=20261019):
    """Keep the evaluated images in exploration and split the other nine per class 4/5."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    by_path = {entry["path"]: entry for entry in inventory}
    exploration = json.loads(exploration_bytes)
    if (not isinstance(exploration, dict) or exploration.get("dataset") != DATASET
            or not isinstance(exploration.get("examples"), list)):
        raise ValueError("Exploration must identify ImageNetV2 matched-frequency and contain an examples list")
    exploration_paths = set()
    for entry in exploration["examples"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Every exploration entry needs a dataset-relative path")
        name = entry["path"]
        if name in exploration_paths:
            raise ValueError(f"Duplicate exploration path: {name}")
        if name not in by_path:
            raise ValueError(f"Exploration path is not in the raw dataset: {name}")
        expected = by_path[name]
        if type(entry.get("label")) is not int or entry["label"] != expected["label"]:
            raise ValueError(f"Exploration label for {name} must be integer {expected['label']}")
        if entry.get("sha256") != expected["sha256"]:
            raise ValueError(f"Missing or mismatched SHA-256 for exploration image {name}")
        exploration_paths.add(name)
    labels = sorted({entry["label"] for entry in inventory})
    if Counter(entry["label"] for entry in exploration["examples"]) != dict.fromkeys(labels, 1):
        raise ValueError("Exploration must contain exactly one image per class")

    candidates = {label: [] for label in labels}
    for entry in inventory:
        if entry["path"] not in exploration_paths:
            candidates[entry["label"]].append(entry["path"])
    confirmation, final = [], []
    rng = random.Random(seed)
    for paths in candidates.values():
        remaining = sorted(paths)
        rng.shuffle(remaining)
        confirmation.extend(by_path[name] for name in sorted(remaining[:4]))
        final.extend(by_path[name] for name in sorted(remaining[4:]))
    splits = {"exploration": exploration,
              "confirmation": {"dataset": DATASET, "selection": "four-per-class",
                               "seed": seed, "examples": confirmation},
              "final_evaluation": {"dataset": DATASET, "selection": "five-per-class",
                                   "seed": seed, "examples": final}}
    seen_paths, hash_owners, counts = set(), {}, {}
    for name, per_class in (("exploration", 1), ("confirmation", 4), ("final_evaluation", 5)):
        entries = splits[name]["examples"]
        paths = {entry["path"] for entry in entries}
        if len(paths) != len(entries) or seen_paths & paths:
            raise ValueError(f"Duplicate or overlapping paths in {name}")
        if Counter(entry["label"] for entry in entries) != dict.fromkeys(labels, per_class):
            raise ValueError(f"{name} must contain {per_class} images per class")
        for entry in entries:
            owner = hash_owners.setdefault(entry["sha256"], (name, entry["path"]))
            if owner[0] != name:
                raise ValueError(f"Exact duplicate content crosses split boundaries: {owner[1]} "
                                 f"in {owner[0]} and {entry['path']} in {name}")
        seen_paths.update(paths)
        counts[name] = {"images": len(entries), "classes": len(labels), "images_per_class": per_class}
    if seen_paths != by_path.keys():
        raise ValueError("Selections must cover the complete dataset")
    return splits, counts


def prepare_data(raw_dir, processed_dir, exploration_manifest, *, source_dir=None,
                 seed=20261019, class_ids=range(1000)):
    """Validate before copying, reuse matching raw data, and write only new processed outputs."""
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    exploration_manifest = Path(exploration_manifest)
    source_dir = Path(source_dir) if source_dir is not None else None
    if processed_dir.exists() or processed_dir.is_symlink():
        raise ValueError(f"Processed directory already exists: {processed_dir}; choose a new --processed-dir")
    raw, processed = raw_dir.resolve(), processed_dir.resolve()
    if raw == processed or raw in processed.parents or processed in raw.parents:
        raise ValueError("Raw and processed directories must be separate, not nested")
    if source_dir is not None:
        source = source_dir.resolve()
        if raw != source and (source in raw.parents or raw in source.parents):
            raise ValueError("Source and raw directories must not be nested")
        if source == processed or source in processed.parents or processed in source.parents:
            raise ValueError("Source and processed directories must be separate, not nested")
    if raw_dir.is_symlink():
        raise ValueError("Raw destination must contain actual files, not be a symbolic link")
    class_ids = tuple(class_ids)
    if source_dir is None and not raw_dir.exists():
        raise ValueError("Raw data is missing; pass --source-dir with the existing extracted dataset")
    inventory = read_inventory(source_dir if source_dir is not None else raw_dir, class_ids=class_ids)
    exploration_bytes = exploration_manifest.read_bytes()
    splits, counts = build_splits(inventory, exploration_bytes, seed=seed)

    if raw_dir.exists():
        if source_dir is not None and read_inventory(raw_dir, class_ids=class_ids) != inventory:
            raise ValueError("Existing raw data differs from the source; no files were overwritten. "
                             "Choose a different --raw-dir or inspect the existing data.")
    else:
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        # copytree creates independent files and refuses an existing destination.
        shutil.copytree(source_dir, raw_dir)
        if read_inventory(raw_dir, class_ids=class_ids) != inventory:
            raise ValueError("Copied raw data does not match the source inventory; inspect it before retrying")

    outputs = {name: (exploration_bytes if name == "exploration"
                      else (json.dumps(manifest, indent=2) + "\n").encode())
               for name, manifest in splits.items()}
    inventory_hash = hashlib.sha256(json.dumps(
        inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Keep project paths portable while retaining external input paths as provenance.
    settings = {}
    for name, path in (("raw_dir", raw_dir), ("processed_dir", processed_dir),
                       ("source_dir", source_dir), ("exploration_manifest", exploration_manifest)):
        if path is None:
            settings[name] = None
        else:
            path = path.resolve()
            settings[name] = str(path.relative_to(REPOSITORY_ROOT)) if path.is_relative_to(REPOSITORY_ROOT) else str(path)
    metadata = {
        "dataset": DATASET, "hosting_revision": DATASET_REVISION,
        "seed": seed,
        "algorithm": "Numeric class order; preserve exploration; shuffle sorted remaining paths "
                     "with one local random.Random(seed); take four for confirmation and five for "
                     "final evaluation. Sort output paths within each class.",
        "settings": settings,
        "exploration_input_sha256": hashlib.sha256(exploration_bytes).hexdigest(),
        "dataset_inventory_sha256": inventory_hash,
        "inventory_hash_format": "UTF-8 JSON of all examples in numeric class/path order, "
                                 "sort_keys=True, separators=(',', ':')",
        "dataset_images": len(inventory),
        "unique_image_hashes": len({entry["sha256"] for entry in inventory}),
        "splits": counts,
        "output_manifest_sha256": {name: hashlib.sha256(contents).hexdigest()
                                   for name, contents in outputs.items()},
        "code": {"path": Path(__file__).relative_to(REPOSITORY_ROOT).as_posix(), "sha256": file_sha256(__file__)},
        "python": sys.version.split()[0],
        "prior_exposure": "The original evaluated selection remains in exploration. "
                          "No recorded exposure outside it was identified in the retained audit. "
                          "Undocumented evaluations cannot be ruled out; review exposure before reuse.",
    }
    processed_dir.mkdir(parents=True, exist_ok=False)
    for name, contents in outputs.items():
        with (processed_dir / f"{name}.json").open("xb") as stream:
            stream.write(contents)
    with (processed_dir / "preparation.json").open("x") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    return metadata


def read_manifest(raw_dir, path):
    """Check only the requested images, preserving their order and numeric labels."""
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data is missing at {raw_dir}; see data/README.md")
    original = Path(path).read_bytes()
    manifest = json.loads(original)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("examples"), list) or not manifest["examples"]:
        raise ValueError("The evaluation manifest must contain a nonempty examples list")
    seen = set()
    for entry in manifest["examples"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Each manifest entry needs a relative path, integer label, and SHA-256")
        name, label = entry["path"], entry.get("label")
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2
                or type(label) is not int or not 0 <= label < 1000 or relative.parts[0] != str(label)
                or relative.as_posix() != name):
            raise ValueError(f"Manifest path/label mismatch: {name!r}, label={label!r}")
        if name in seen:
            raise ValueError(f"Duplicate manifest path: {name}")
        seen.add(name)
        image = (raw_dir / name).resolve()
        if not image.is_relative_to(raw_dir) or not image.is_file():
            raise ValueError(f"Selected image is missing or outside the raw directory: {name}")
        if file_sha256(image) != entry.get("sha256"):
            raise ValueError(f"Selected image hash mismatch: {name}")
    return manifest, original


# DataLoader accepts a map-style object with __len__ and __getitem__; no torch import is needed here.


class ManifestImages:
    """Read every selected image or raise an error naming the failed example."""

    def __init__(self, dataset_dir, manifest, transform):
        self.root = Path(dataset_dir)
        self.examples = manifest["examples"]
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        from PIL import Image

        example = self.examples[index]
        path = self.root / example["path"]
        try:
            with Image.open(path) as image:
                inputs = self.transform(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"Cannot read or preprocess selected image {path}: {error}") from error
        return inputs, example["label"]


def make_loader(dataset_dir, manifest, transform, *, batch_size=16, workers=0, seed=2026):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(ManifestImages(dataset_dir, manifest, transform), batch_size=batch_size,
                      shuffle=False, drop_last=False, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed))


def configure_runtime(device, threads=4):
    """Use explicit float32 backend settings for every variant in a fresh process."""
    import torch
    import torch.backends.opt_einsum as torch_opt_einsum
    from tensorly import plugins

    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested device {device} is unavailable or unsupported; no fallback is used")
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    # Installing an optional optimizer must not change baseline contractions.
    torch_opt_einsum.enabled = False
    plugins.use_default_einsum()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return device


def model_fingerprint(model):
    """Hash all parameter and buffer values, including nonpersistent buffers."""
    digest = hashlib.sha256()
    for category, tensors in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in tensors:
            digest.update(f"{category}:{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prediction_counts(logits, labels):
    """Count correct examples; averaging batch percentages would bias a short batch."""
    import torch

    if logits.ndim != 2 or logits.shape[1] < 5 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected N-by-C predictions with C >= 5 and N integer labels")
    if not torch.isfinite(logits).all() or (labels < 0).any() or (labels >= logits.shape[1]).any():
        raise ValueError("Nonfinite predictions or class labels outside the model output range")
    predicted = logits.topk(5, dim=1).indices
    correct = predicted.eq(labels[:, None])
    return int(correct[:, 0].sum()), int(correct.any(dim=1).sum()), labels.numel()


def measure_latency(model, inputs, *, warmup=20, repeats=100):
    """Time completed forward passes in milliseconds per batch, after warm-up."""
    import torch

    with torch.inference_mode():
        if warmup < 0 or repeats < 2:
            raise ValueError("Use nonnegative warmup and at least two timed passes")
        model.eval()
        for _ in range(warmup):
            model(inputs)
        samples = []
        for _ in range(repeats):
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            start = time.perf_counter_ns()
            model(inputs)
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        quartiles = statistics.quantiles(samples, n=4, method="inclusive")
        return {"median_ms": statistics.median(samples), "p25_ms": quartiles[0], "p75_ms": quartiles[2],
                "samples_ms": samples}


def write_csv(path, rows):
    """Format the summary for readers without rounding the saved measurements."""
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in row.items():
                if key.endswith(("_percent", "_change_pp")):
                    formatted[key] = f"{value:.2f}"
                elif key.startswith("latency_") and key.endswith("_ms"):
                    formatted[key] = f"{value:.3f}"
            writer.writerow(formatted)


def results_table(rows):
    headings = ["Variant", "Top-1 %", "Top-5 %", "Top-1 Δ pp", "Top-5 Δ pp", "Parameters", "Param MiB", "Latency ms [p25, p75]"]
    lines = [" | ".join(headings), " | ".join("---" for _ in headings)]
    for row in rows:
        lines.append(" | ".join([row["variant"], f"{float(row['top1_percent']):.2f}",
            f"{float(row['top5_percent']):.2f}", f"{float(row['top1_change_pp']):+.2f}",
            f"{float(row['top5_change_pp']):+.2f}", str(row["parameter_count"]),
            f"{int(row['parameter_bytes']) / 2**20:.2f}",
            f"{float(row['latency_median_ms']):.2f} [{float(row['latency_p25_ms']):.2f}, {float(row['latency_p75_ms']):.2f}]"]))
    return "\n".join(lines)


def environment_info():
    """Capture the code and runtime used for a run, before measurements start."""
    import torch
    import torch.backends.opt_einsum as torch_opt_einsum
    from tensorly import plugins

    import tltorch

    root = Path(__file__).resolve().parents[2]
    packages = {}
    for name in ("torch", "torchvision", "tensorly", "numpy", "scipy", "Pillow", "nbformat", "nbclient", "ipykernel", "opt-einsum"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    return {"python": platform.python_version(), "packages": packages, "tltorch_version": tltorch.__version__,
            "tltorch_path": tltorch.__file__, "platform": platform.platform(), "cpu": cpu,
            "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "backend": {"opt_einsum": {"available": torch_opt_einsum.is_available(),
                                       "enabled": torch_opt_einsum.enabled,
                                       "strategy": torch_opt_einsum.strategy},
                        "tensorly_plugin_inactive": plugins.PREVIOUS_EINSUM is None,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "matmul_precision": torch.get_float32_matmul_precision(),
                        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "mkldnn_enabled": torch.backends.mkldnn.enabled,
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS")},
            "repository_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
            "experiment_sha256": file_sha256(__file__), "started_utc": datetime.now(timezone.utc).isoformat()}


def evaluate(model, batches, device):
    """Average loss by image and macro metrics across every model output class, zeros included."""
    import torch
    from torch.nn import functional as F

    with torch.inference_mode():
        model.eval()
        top1 = top5 = total = 0
        loss_sum = 0.0
        confusion = None
        for inputs, labels in batches:
            labels = labels.to(device)
            logits = model(inputs.to(device))
            one, five, count = prediction_counts(logits, labels)
            classes = logits.shape[1]
            if confusion is None:
                confusion = torch.zeros(classes, classes, dtype=torch.int64)
            if confusion.shape[0] != classes:
                raise ValueError("Model output class count changed between batches")
            top1 += one
            top5 += five
            total += count
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            # Rows are true classes; columns are top-1 predictions.
            pairs = (labels * classes + logits.argmax(dim=1)).cpu()
            confusion += torch.bincount(pairs, minlength=classes * classes).reshape(classes, classes)
        if total == 0:
            raise ValueError("Evaluation needs at least one example")
        true_positive = confusion.diag().double()
        predicted = confusion.sum(dim=0)
        actual = confusion.sum(dim=1)
        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / actual.clamp_min(1)
        f1 = 2 * true_positive / (predicted + actual).clamp_min(1)
        return {"examples": total, "top1_correct": top1, "top5_correct": top5,
                "top1_percent": 100 * top1 / total, "top5_percent": 100 * top5 / total,
                "cross_entropy_mean": loss_sum / total,
                "macro_precision": precision.mean().item(), "macro_recall": recall.mean().item(),
                "macro_f1": f1.mean().item(), "macro_classes": confusion.shape[0]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True,
                                   env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})


def create_output(root, name):
    """Reserve a new run directory; even an empty existing directory is protected."""
    output = Path(root) / name
    if output.is_relative_to(REPOSITORY_ROOT):
        ignored = subprocess.run(["git", "check-ignore", "-q", str(output)], cwd=REPOSITORY_ROOT)
        if ignored.returncode != 0:
            raise ValueError("An output directory inside the repository must be ignored by Git")
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_directory_name():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]


def save_source(output, snapshot):
    """Archive the exact committed execution scope, excluding deleted source files."""
    hashes = snapshot["source_files_sha256"]
    with zipfile.ZipFile(output / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, digest in hashes.items():
            if digest is not None:
                contents = subprocess.check_output(
                    ["git", "show", f"{snapshot['commit']}:{name}"], cwd=REPOSITORY_ROOT)
                if hashlib.sha256(contents).hexdigest() != digest:
                    raise RuntimeError(f"Committed source hash mismatch: {name}")
                archive.writestr(name, contents)
    revision = {**snapshot, "execution_inputs_match_commit": True,
                "branch": git("branch", "--show-current").strip(),
                "working_tree_status": git("status", "--short", "--untracked-files=all"),
                "source_archive_sha256": file_sha256(output / "source.zip")}
    write_json(output / "source.json", revision)
    return revision


def layer_identity(model, report):
    """Record the concrete factor types and ranks, not just the requested target."""
    identity = {}
    if report is None:
        return identity
    for entry in report["replacements"]:
        layer = model.get_submodule(entry["name"])
        rank = layer.rank
        identity[entry["name"]] = {
            "layer_type": f"{type(layer).__module__}.{type(layer).__qualname__}",
            "weight_type": f"{type(layer.weight).__module__}.{type(layer.weight).__qualname__}",
            "actual_rank": int(rank) if entry["factorization"] == "CP" else [int(n) for n in rank],
        }
    return identity


def load_model(path, device="cpu"):
    """Load our trusted full-model checkpoint; requires matching repository code and packages.

    This uses pickle (weights_only=False). Only load artifacts from trusted runs.
    No model constructor, pretrained-weight loader, or decomposition is called.
    """
    import torch

    model = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Checkpoint does not contain a PyTorch model")
    return model.eval()


def load_compressed_model(path, device="cpu"):
    """Compatibility loader for trusted historical compressed_model.pt checkpoints."""
    return load_model(path, device)


def save_and_verify_model(model, report, inputs, output):
    """Reload independently and check stored state, factorization identity, and a fixed forward pass."""
    import torch

    path = output / "model.pt"
    model.eval()
    torch.save(model, path)
    loaded = load_model(path, inputs.device)
    before, after = layer_identity(model, report), layer_identity(loaded, report)
    if before != after or model_fingerprint(model) != model_fingerprint(loaded):
        raise RuntimeError("Reloaded model differs in stored parameters, buffers, factorization types, or ranks")
    with torch.inference_mode():
        expected = model(inputs)
        actual = loaded(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.save({"inputs": inputs.cpu(), "expected_outputs": expected.cpu()}, output / "verification_input.pt")
    verification = {"passed": True, "rtol": 1e-5, "atol": 1e-6,
                    "max_absolute_output_error": (actual - expected).abs().max().item(),
                    "parameters_and_buffers_exact": True, "layers": before,
                    "checkpoint_sha256": file_sha256(path),
                    "fixed_input_sha256": file_sha256(output / "verification_input.pt"),
                    "device": str(inputs.device), "format": "torch.save full nn.Module; trusted pickle"}
    write_json(output / "model_verification.json", verification)
    return verification


def log_and_verify_mlflow_model(model, report, inputs, output, *, parameters=None, status=None):
    """Log the measured model as trusted pickle, then verify an independent reload."""
    import torch
    import mlflow.pytorch
    from mlflow.models import infer_signature

    before = model_fingerprint(model)
    placement = [(t.device, t.dtype) for t in [*model.parameters(), *model.buffers()]]
    modes = [module.training for module in model.modules()]
    with torch.inference_mode():
        expected = model(inputs).detach()
    example = inputs.detach().cpu().numpy()
    signature = infer_signature(example, expected.cpu().numpy())
    packages = ("mlflow-skinny", "torch", "torchvision", "tensorly", "numpy", "scipy", "cloudpickle", "pandas")
    requirements = [f"{name}=={importlib.metadata.version(name)}" for name in packages]
    # Copy only importable library source, avoiding tests, caches, and editable-install paths.
    with tempfile.TemporaryDirectory(prefix="compression-mlflow-") as temporary:
        library = Path(temporary) / "tltorch"
        source_names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "tltorch")
        for name in sorted(set(source_names.split("\0")) - {""}):
            source = REPOSITORY_ROOT / name
            relative = source.relative_to(REPOSITORY_ROOT / "tltorch")
            if source.suffix == ".py" and "tests" not in relative.parts and source.is_file():
                destination = library / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
        # Passing an existing model ID avoids log_model's automatic run-parameter copy.
        logged_model = mlflow.initialize_logged_model(name="model", params=parameters or {})
        update_status(status, model_id=logged_model.model_id)
        info = mlflow.pytorch.log_model(
            model, name="model", model_id=logged_model.model_id, serialization_format="pickle",
            input_example=example, signature=signature, pip_requirements=requirements,
            code_paths=[str(library)],
        )
        metadata = {"model_uri": info.model_uri, "model_id": logged_model.model_id,
                    "serialization_format": "pickle", "pip_requirements": requirements}
        # Save the URI immediately, so a failed reload still leaves a usable diagnostic.
        run = json.loads((output / "run.json").read_text())
        run["mlflow_model"] = metadata
        write_json(output / "run.json", run)
        update_status(status, model_uri=info.model_uri)
        print(f"MLflow model: {info.model_uri}", flush=True)
        loaded = mlflow.pytorch.load_model(
            info.model_uri, dst_path=temporary, map_location=inputs.device, weights_only=False)
        if (model_fingerprint(loaded) != before
                or layer_identity(loaded, report) != layer_identity(model, report)):
            raise RuntimeError("MLflow model reload changed parameters, buffers, types, or ranks")
        with torch.inference_mode():
            actual = loaded(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if (model_fingerprint(model) != before
            or placement != [(t.device, t.dtype) for t in [*model.parameters(), *model.buffers()]]
            or modes != [module.training for module in model.modules()]):
        raise RuntimeError("MLflow packaging changed the measured model's state, placement, or mode")
    verification = {**metadata, "passed": True, "parameters_and_buffers_exact": True,
                    "rtol": 1e-5, "atol": 1e-6, "device": str(inputs.device),
                    "max_absolute_output_error": (actual - expected).abs().max().item()}
    write_json(output / "mlflow_model_verification.json", verification)
    return verification


def validate_tracking_uri(address):
    """Require a configured HTTP(S) server; credentials stay in MLflow's own setup."""
    message = ("tracking.uri must be an HTTP(S) server URI without credentials, "
               "whitespace, query parameters, or fragments")
    if not isinstance(address, str) or not address.strip():
        raise ValueError(message)
    try:
        parsed = urlparse(address)
        parsed.port  # Reject malformed or out-of-range ports before any tracking request.
    except ValueError:
        raise ValueError(message) from None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(c.isspace() for c in address)):
        raise ValueError(message)


def configure_weight_cache(config):
    """Apply an explicit cache only during execution; otherwise keep PyTorch's choice."""
    import torch

    torch_home = config.get("cache", {}).get("torch_home")
    if torch_home is not None:
        torch.hub.set_dir(str(Path(torch_home) / "hub"))
    return {"torch_home": torch_home,
            "hub_dir": str(Path(torch.hub.get_dir()).expanduser().resolve())}


def cached_weights():
    import torch

    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1
    path = Path(torch.hub.get_dir()).expanduser().resolve() / "checkpoints" / Path(urlparse(weights.url).path).name
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights are missing at {path}. Follow the explicit weight-cache "
                                "setup in experiments/imagenet-compression/README.md. Set cache.torch_home in YAML, "
                                "or use TORCH_HOME when the cache section is absent.")
    digest = file_sha256(path)
    if not digest.startswith("f37072fd"):
        raise ValueError("Cached ResNet-18 weights do not match the published checksum prefix f37072fd")
    return path, {"identifier": "ResNet18_Weights.IMAGENET1K_V1", "url": weights.url, "sha256": digest,
                  "checkpoint_path": str(path)}


def load_cached_baseline(path, device):
    """Use a local state dict directly so execution has no download path."""
    import torch

    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device=device, dtype=torch.float32).eval(), ResNet18_Weights.IMAGENET1K_V1.transforms()


def comparison_rows(observations, report):
    """Historical two-model rows; configured executions use single_model_row."""
    rows = []
    for name, side in (("dense", "original"), ("compressed", "compressed")):
        measurement = observations[name]
        row = {"variant": name, **measurement["accuracy"], **report["model"][side]}
        for metric in ("top1", "top5"):
            row[f"{metric}_change_pp"] = row[f"{metric}_percent"] - observations["dense"]["accuracy"][f"{metric}_percent"]
        row.update({f"replaced_{key}": value for key, value in report["replaced_layers"][side].items()})
        row.update({f"latency_{key}": value for key, value in measurement["latency"].items() if key != "samples_ms"})
        row.update(flops_total=None, flops_reason=FLOPS["reason"])
        rows.append(row)
    return rows


def single_model_row(variant, measurement):
    """Keep one model's measurements in a flat CSV row, without cross-run differences."""
    row = {"variant": variant, **measurement["accuracy"], **measurement["storage"]}
    row.update({f"latency_{key}": value for key, value in measurement["latency"].items() if key != "samples_ms"})
    if "conversion_seconds" in measurement:
        row["conversion_seconds"] = measurement["conversion_seconds"]
    row.update(flops_total=None, flops_reason=FLOPS["reason"])
    return row


def single_model_table(row):
    """Show the selected model's accuracy, storage, and latency without a baseline row."""
    headings = ["Model", "Top-1 %", "Top-5 %", "Mean cross-entropy", "Parameters", "Param MiB", "Buffer bytes", "Latency ms [p25, p75]"]
    values = [row["variant"], f"{row['top1_percent']:.2f}", f"{row['top5_percent']:.2f}",
              f"{row['cross_entropy_mean']:.6f}", str(row["parameter_count"]),
              f"{row['parameter_bytes'] / 2**20:.2f}", str(row["buffer_bytes"]),
              f"{row['latency_median_ms']:.2f} [{row['latency_p25_ms']:.2f}, {row['latency_p75_ms']:.2f}]"]
    return "\n".join((" | ".join(headings), " | ".join("---" for _ in headings), " | ".join(values)))


def requested_parameters(config):
    """Return separate encoded run settings and requested model settings."""
    runtime, timing = config["runtime"], config["timing"]
    compression = config["compression"]
    run_parameters = {
        "device": runtime["device"], "dtype": runtime["dtype"], "seed": runtime["seed"],
        "evaluation.batch_size": runtime["batch_size"], "runtime.threads": runtime["threads"],
        "timing.input_shape": timing["input_shape"],
        "timing.warmup": timing["warmup"], "timing.repeats": timing["repeats"],
    }
    model_parameters = {"model": config["model"], "weights": config["weights"]}
    if compression is not None:
        layers = compression["layers"]
        model_parameters["implementation"] = compression["implementation"]
        model_parameters["compression.layers"] = sorted(layers)
        for name, settings in sorted(layers.items()):
            for option, value in sorted(settings.items()):
                prefix = f"compression.{name}.{option}"
                if option == "decomposition_kwargs":
                    for control, requested in sorted(value.items()):
                        model_parameters[f"{prefix}.{control}"] = requested
                else:
                    model_parameters[prefix] = value
    for parameters in (run_parameters, model_parameters):
        for key, value in parameters.items():
            # JSON keeps rank 1 distinct from target 1.0 and makes lists unambiguous.
            parameters[key] = value if isinstance(value, str) else json.dumps(value, sort_keys=True, allow_nan=False)
    return run_parameters, model_parameters


def log_model_metrics(row, run_id, model_id):
    """Associate eight absolute metrics with this model; conversion time stays run-only."""
    import mlflow
    from mlflow.entities import Metric

    absolute_metrics = ("top1_percent", "top5_percent", "cross_entropy_mean", "parameter_count",
                        "parameter_bytes", "latency_median_ms", "latency_p25_ms", "latency_p75_ms")
    timestamp = int(time.time() * 1000)
    metrics = [Metric(key, float(row[key]), timestamp, 0, model_id=model_id, run_id=run_id)
               for key in absolute_metrics if math.isfinite(row[key])]
    if "conversion_seconds" in row:
        metrics.append(Metric("conversion_seconds", row["conversion_seconds"], timestamp, 0, run_id=run_id))
    # Client batches otherwise infer missing IDs from the active model. Log after
    # packaging, which would also backfill existing step-zero metrics onto a model.
    mlflow.clear_active_model()
    mlflow.MlflowClient().log_batch(run_id, metrics=metrics, synchronous=True)


def run_experiment(config, original_config, *, config_path, provenance_configs=None, status=None):
    """Validate inputs, commit sources, then track, select, measure, verify, and upload one model.

    Tests record the MLflow calls offline; execution requires the configured server.
    """
    import torch
    import numpy as np
    from tltorch import compress_model
    from tltorch.utils.model_compression import _measure_storage

    provenance_configs = provenance_configs or {}
    update_status(status, phase="preflight")
    tracking = config["tracking"]
    validate_tracking_uri(tracking["uri"])
    manifest, manifest_bytes = read_manifest(config["data"]["raw_dir"], config["data"]["manifest"])
    cache_info = configure_weight_cache(config)
    weight_path, weight_info = cached_weights()
    runtime, timing = config["runtime"], config["timing"]
    device = configure_runtime(runtime["device"], runtime["threads"])
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA device {device} is unavailable; no fallback is used")
    random.seed(runtime["seed"])
    np.random.seed(runtime["seed"])
    torch.manual_seed(runtime["seed"])
    output = create_output(config["output_root"], run_directory_name())
    update_status(status, output_directory=str(output.resolve()))
    (output / "original_config.yaml").write_bytes(original_config)
    write_json(output / "effective_config.json", config)
    (output / "evaluation_manifest.json").write_bytes(manifest_bytes)
    configuration_hashes = {
        "original_config_file_sha256": file_sha256(output / "original_config.yaml"),
        "effective_config_file_sha256": file_sha256(output / "effective_config.json"),
    }
    run_id = None
    operation_error = None
    revision = None
    try:
        import mlflow.pytorch  # Check optional packaging dependencies before committing or tracking.

        # start_run otherwise treats this environment variable as a request to resume.
        if os.environ.get("MLFLOW_RUN_ID"):
            raise ValueError("Unset MLFLOW_RUN_ID: each execution must create a new run")
        if mlflow.active_run() is not None:
            raise ValueError("End the active MLflow run before starting this execution")
        if load_config(config_path)[0] != config:
            raise ValueError("Effective configuration differs from the selected YAML; reload it before execution")
        update_status(status, phase="snapshot")
        snapshot = snapshot_execution_inputs(REPOSITORY_ROOT, EXPERIMENT_DIR, config_path,
                                             original_config, tracking["run_name"],
                                             provenance_configs=provenance_configs)
        revision = save_source(output, snapshot)
        update_status(status, phase="tracking_setup")
        mlflow.set_tracking_uri(tracking["uri"])
        mlflow.set_workspace(tracking["workspace"])
        experiment = mlflow.set_experiment(tracking["experiment_name"])
        experiment_id = experiment.experiment_id
        update_status(status, experiment_id=experiment_id)
        tags = {"tracking.workspace": tracking["workspace"], "git.branch": revision["branch"],
                "git.commit": revision["commit"]}
        with mlflow.start_run(experiment_id=experiment_id, run_name=tracking["run_name"], tags=tags) as run:
            run_id = run.info.run_id
            try:
                update_status(status, state="running", phase="early_uploads", run_id=run_id)
                write_json(output / "run.json", {"workspace": tracking["workspace"],
                                                 "experiment_id": experiment_id, "run_id": run_id,
                                                 **configuration_hashes})
                print(f"Starting MLflow workspace {tracking['workspace']}, experiment {experiment_id}, "
                      f"run {run_id}. Local output: {output}", flush=True)
                # Fail early if the server accepts runs but cannot store artifacts.
                mlflow.log_artifact(str(output / "effective_config.json"))
                for filename in ("original_config.yaml", "run.json", "source.json", "source.zip"):
                    mlflow.log_artifact(str(output / filename))
                for name, contents in provenance_configs.items():
                    destination = output / "provenance_configs" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(contents)
                    mlflow.log_artifact(str(destination), artifact_path=destination.parent.relative_to(output).as_posix())
                run_parameters, model_parameters = requested_parameters(config)
                for key, value in run_parameters.items():
                    mlflow.log_param(key, value, synchronous=True)

                update_status(status, phase="model_loading")
                source, preprocessing = load_cached_baseline(weight_path, device)
                original_state = model_fingerprint(source)
                environment = environment_info()
                for package in ("mlflow-skinny", "PyYAML"):
                    environment["packages"][package] = importlib.metadata.version(package)
                provenance = {"environment": environment, "git": revision, "model": config["model"],
                              "weights": weight_info, "cache": cache_info,
                              "preprocessing": {"description": str(preprocessing),
                                                **{key: getattr(preprocessing, key, None) for key in
                                                   ("resize_size", "crop_size", "mean", "std", "antialias")},
                                                "interpolation": str(getattr(preprocessing, "interpolation", None))},
                              "dataset": manifest.get("dataset"), "selection_seed": manifest.get("seed"),
                              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                              **configuration_hashes,
                              "seeds": {"runtime": runtime["seed"], "timing": timing["seed"]},
                              "flops": FLOPS, "execution": "eager; no torch.compile",
                              "macro_metrics": "Unweighted mean across all model output classes. Undefined precision, recall, and F1 are zero.",
                              "latency_units": "milliseconds per batch; excludes loading, preprocessing, transfers, and conversion"}
                update_status(status, phase="input_provenance")
                write_json(output / "provenance.json", provenance)
                for filename in ("evaluation_manifest.json", "provenance.json"):
                    mlflow.log_artifact(str(output / filename))

                compression = config["compression"]
                model, report = source, None
                conversion_seconds = None
                if compression is not None:
                    update_status(status, phase="conversion")
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    # Decomposition creates trainable parameters; keep it outside inference mode.
                    model, report = compress_model(source, compression["layers"],
                                                   implementation=compression["implementation"])
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    conversion_seconds = time.perf_counter() - started
                    if model_fingerprint(source) != original_state:
                        raise RuntimeError("Conversion changed pretrained source parameters or buffers")
                    write_json(output / "compression_report.json", report)
                    storage = report["model"]["compressed"]
                else:
                    # Reuse the library's tensor accounting without copying or converting dense models.
                    storage = _measure_storage([model])

                update_status(status, phase="measurement")
                batches = make_loader(config["data"]["raw_dir"], manifest, preprocessing,
                                      batch_size=runtime["batch_size"], workers=runtime["workers"], seed=runtime["seed"])
                inputs = torch.randn(*timing["input_shape"], device=device, dtype=torch.float32,
                                     generator=torch.Generator(device=device).manual_seed(timing["seed"]))
                measurement = {"accuracy": evaluate(model, batches, device),
                               "latency": measure_latency(model, inputs, warmup=timing["warmup"], repeats=timing["repeats"]),
                               "storage": storage}
                if model_fingerprint(source) != original_state:
                    raise RuntimeError("Evaluation changed pretrained source parameters or buffers")
                if conversion_seconds is not None:
                    measurement["conversion_seconds"] = conversion_seconds
                variant = "compressed" if compression is not None else "dense"
                write_json(output / "measurements.json", {"variant": variant, **measurement})
                row = single_model_row(variant, measurement)
                write_csv(output / "results.csv", [row])
                write_json(output / "latency_samples.json", measurement["latency"]["samples_ms"])
                (output / "report.md").write_text(single_model_table(row) + "\n\n" + provenance["macro_metrics"] +
                                                  "\n\nFLOPs unavailable: " + FLOPS["reason"] + "\n")
                update_status(status, phase="checkpoint_verification")
                save_and_verify_model(model, report, inputs, output)
                verify_execution_inputs(REPOSITORY_ROOT, EXPERIMENT_DIR, config_path, revision)
                update_status(status, phase="mlflow_model_verification")
                model_verification = log_and_verify_mlflow_model(
                    model, report, inputs, output, parameters=model_parameters, status=status)
                verification = verify_execution_inputs(REPOSITORY_ROOT, EXPERIMENT_DIR, config_path, revision)
                write_json(output / "execution_inputs_verification.json", verification)
                update_status(status, phase="metric_upload")
                log_model_metrics(row, run_id, model_verification["model_id"])
                # The run remains open until checkpoint verification and every required upload succeed.
                update_status(status, phase="artifact_upload")
                mlflow.log_artifacts(str(output))
                verify_execution_inputs(REPOSITORY_ROOT, EXPERIMENT_DIR, config_path, revision)
                update_status(status, phase="run_finalization")
            except BaseException as error:
                # MLflow's context exit can itself fail while updating run status.
                operation_error = error
                raise
        return output
    except BaseException as error:
        if operation_error is not None:
            error = operation_error
        input_change_error = None
        if revision is not None:
            try:
                verify_execution_inputs(REPOSITORY_ROOT, EXPERIMENT_DIR, config_path, revision)
            except Exception as changed:
                input_change_error = str(changed)
                print(f"Execution input verification failed: {changed}", file=sys.stderr)
        try:
            write_json(output / "failure.json", {"run_id": run_id, "error_type": type(error).__name__,
                                                "error": str(error), "execution_input_error": input_change_error,
                                                "traceback": "".join(traceback.format_exception(error))})
        except Exception:
            pass  # Diagnostic write failures must not hide the original failure.
        if run_id is not None:
            # The context may already be closed: target this run explicitly, never start another.
            try:
                mlflow.log_artifact(str(output / "failure.json"), run_id=run_id)
            except Exception:
                pass
            try:
                mlflow.MlflowClient(tracking_uri=tracking["uri"]).set_terminated(run_id, status="FAILED")
            except Exception:
                pass
        raise error


# The saved notebooks compare multiple variants. They share the runtime, loading,
# measurement and reporting helpers above; their separate sequence stays optional.


def layer_settings(fraction):
    """The same three trained weights are selected for each storage target."""
    return {
        "layer4.0.conv2": {"factorization": "Tucker", "rank": fraction, "fixed_rank_modes": "spatial"},
        "layer4.1.conv2": {"factorization": "Tucker", "rank": fraction, "fixed_rank_modes": "spatial"},
        "fc": {"factorization": "BlockTT", "rank": fraction,
               "in_tensorized_features": (8, 8, 8), "out_tensorized_features": (10, 10, 10)},
    }


def select_examples(dataset_dir, *, seed=2026, full=False, manifest_path=None,
                    class_ids=range(1000), images_per_class=10):
    """Validate numeric class folders, then select sorted paths with a local RNG.

    The class/count arguments allow tiny offline fixtures; the experiment uses
    the complete release: 1,000 classes with ten images each, even for a quick run.
    Saved manifests preserve order and include content hashes, not just filenames.
    """
    root = Path(dataset_dir).resolve()
    expected_labels = set(class_ids)
    candidates = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.isdecimal() or str(int(folder.name)) != folder.name:
            raise ValueError(f"Expected numeric class directory, got {folder.name!r}")
        label = int(folder.name)
        paths = sorted(path.relative_to(root).as_posix() for path in folder.iterdir()
                       if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(paths) != images_per_class:
            raise ValueError(f"Class {label}: expected {images_per_class} images, found {len(paths)}")
        candidates[label] = paths
    if set(candidates) != expected_labels:
        raise ValueError(f"Dataset classes differ: missing {sorted(expected_labels - candidates.keys())}, "
                         f"unexpected {sorted(candidates.keys() - expected_labels)}")

    if manifest_path is None:
        rng = random.Random(seed)
        examples = []
        for label in sorted(candidates):
            paths = candidates[label] if full else [rng.choice(candidates[label])]
            examples.extend({"path": path, "label": label} for path in paths)
    else:
        manifest = json.loads(Path(manifest_path).read_text())
        examples = manifest["examples"]
    seen, counts = set(), Counter()
    for example in examples:
        path, label = example["path"], example["label"]
        if type(label) is not int or label not in candidates or path not in candidates[label]:
            raise ValueError(f"Manifest path/label does not match the dataset: {example}")
        if path in seen:
            raise ValueError(f"Duplicate manifest path: {path}")
        seen.add(path)
        counts[label] += 1
        digest = file_sha256(root / path)
        if "sha256" in example and example["sha256"] != digest:
            raise ValueError(f"Image content changed since manifest creation: {path}")
        example["sha256"] = digest
    expected_count = images_per_class if full else 1
    if counts != Counter({label: expected_count for label in expected_labels}):
        raise ValueError(f"Manifest must contain exactly {expected_count} image(s) per class")
    return {"dataset": "ImageNetV2 matched-frequency (imagenetv2-b-33)",
            "selection": "all" if full else "one-per-class", "seed": seed if manifest_path is None else manifest["seed"],
            "examples": examples}


def load_baseline(device="cpu", *, allow_download=False):
    """Load the specified torchvision weights; downloading requires explicit opt-in."""
    import torch

    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not cached.is_file() and not allow_download:
        raise FileNotFoundError(f"Weights are not cached at {cached}. Follow the explicit weight-cache "
                                "setup in experiments/imagenet-compression/README.md, or set TORCH_HOME to an existing cache.")
    baseline = resnet18(weights=weights).to(device=device, dtype=torch.float32).eval()
    return baseline, weights.transforms()


def evaluate_accuracy(model, batches, device="cpu", *, progress=False):
    import torch

    with torch.inference_mode():
        model.eval()
        top1 = top5 = total = 0
        for batch_index, (inputs, labels) in enumerate(batches):
            counts = prediction_counts(model(inputs.to(device)), labels.to(device))
            top1 += counts[0]
            top5 += counts[1]
            total += counts[2]
            if progress and (batch_index + 1) % 10 == 0:
                print(f"  evaluated {total} images", flush=True)
        if total == 0:
            raise ValueError("Accuracy needs at least one example")
        return {"top1_correct": top1, "top5_correct": top5, "examples": total,
                "top1_percent": 100 * top1 / total, "top5_percent": 100 * top5 / total}


def measure_variant(model, batches, inputs, *, warmup=20, repeats=100, progress=False):
    return {"accuracy": evaluate_accuracy(model, batches, inputs.device, progress=progress),
            "latency": measure_latency(model, inputs, warmup=warmup, repeats=repeats)}


def run_comparison(baseline, batches, inputs, *, seed=2026, warmup=20, repeats=100):
    """Always convert the untouched baseline; keep construction outside inference mode."""
    import torch
    from tltorch import compress_model

    original = model_fingerprint(baseline)
    observations, reports = {}, {}
    for name, fraction in VARIANTS:
        print(f"Starting {name}", flush=True)
        if fraction is None:
            model = baseline
        else:
            torch.manual_seed(seed)
            model, report = compress_model(baseline, layer_settings(fraction), implementation="factorized")
            if model_fingerprint(baseline) != original:
                raise RuntimeError(f"Conversion of {name} changed baseline parameters or buffers")
            reports[name] = report
            print(f"  converted; baseline unchanged; parameters {report['model']['compressed']['parameter_count']}",
                  flush=True)
        observations[name] = measure_variant(model, batches, inputs, warmup=warmup, repeats=repeats, progress=True)
        accuracy = observations[name]["accuracy"]
        print(f"  top-1 {accuracy['top1_percent']:.2f}%, top-5 {accuracy['top5_percent']:.2f}%", flush=True)
        del model
    if model_fingerprint(baseline) != original:
        raise RuntimeError("Evaluation changed baseline parameters or buffers")
    return observations, reports


def notebook_comparison_rows(observations, reports):
    """Reuse conversion reports for all storage totals, including the dense baseline."""
    reference = reports["fraction_0.75"]
    baseline_accuracy = observations["dense"]["accuracy"]
    for report in reports.values():
        if report["model"]["original"] != reference["model"]["original"]:
            raise ValueError("Conversion reports disagree about baseline storage")
    rows = []
    for name, fraction in VARIANTS:
        measurement = observations[name]
        if measurement["accuracy"]["examples"] != baseline_accuracy["examples"]:
            raise ValueError("Variants were evaluated on different numbers of examples")
        report = reference if fraction is None else reports[name]
        side = "original" if fraction is None else "compressed"
        row = {"variant": name, "weight_fraction": fraction, **measurement["accuracy"]}
        for metric in ("top1", "top5"):
            row[f"{metric}_change_pp"] = row[f"{metric}_percent"] - baseline_accuracy[f"{metric}_percent"]
        row.update(report["model"][side])
        row.update({f"replaced_{key}": value for key, value in report["replaced_layers"][side].items()})
        row.update({f"latency_{key}": value for key, value in measurement["latency"].items() if key != "samples_ms"})
        row.update(flops_total=None, flops_reason=FLOPS["reason"])
        rows.append(row)
    return rows


def save_results(output_dir, observations, reports, manifest, settings, environment, preprocessing):
    """Save small review artifacts; never overwrite a previous comparison."""
    import torch

    from torchvision.models import ResNet18_Weights

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Results directory is not empty: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    weight_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    archive = settings.get("archive")
    provenance = {"release": "imagenetv2-b-33", "hosting_revision": DATASET_REVISION, "source_url": DATASET_URL,
                  "archive_sha256": file_sha256(archive) if archive else None,
                  "selected_image_checksums": "SHA-256 for each image in selected_examples.json"}
    settings = {**settings, "selection_seed": manifest["seed"], "evaluation_size": len(manifest["examples"])}
    transform_settings = {"description": str(preprocessing), "resize_size": preprocessing.resize_size,
                          "crop_size": preprocessing.crop_size, "mean": preprocessing.mean, "std": preprocessing.std,
                          "interpolation": str(preprocessing.interpolation), "antialias": preprocessing.antialias}
    configuration = {"settings": settings, "environment": environment, "model": "torchvision.resnet18",
                     "weights": {"identifier": "ResNet18_Weights.IMAGENET1K_V1", "url": weights.url,
                                 "sha256": file_sha256(weight_file)},
                     "preprocessing": transform_settings, "dataset": provenance,
                     "layers": {name: layer_settings(fraction) for name, fraction in VARIANTS if fraction is not None},
                     "dtype": "float32", "timing_input_shape": [1, 3, 224, 224], "timing_units": "milliseconds per batch",
                     "flops": FLOPS, "completed_utc": datetime.now(timezone.utc).isoformat()}
    rows = notebook_comparison_rows(observations, reports)
    write_csv(output / "results.csv", rows)
    artifacts = {"configuration.json": configuration, "selected_examples.json": manifest,
                 "compression_reports.json": reports, "measurements.json": observations,
                 "latency_samples.json": {name: item["latency"]["samples_ms"] for name, item in observations.items()}}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--config", type=Path, help="YAML filename, relative to the caller's working directory")
    action.add_argument("--prepare-data", action="store_true", help="Prepare data only, without models or tracking")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without execution")
    parser.add_argument("--status-file", type=Path, help="New absolute receipt path in an existing writable directory")
    parser.add_argument("--provenance-config", type=Path, action="append", default=[],
                        help="Additional YAML under configs/ to snapshot and archive; repeat as needed")
    parser.add_argument("--source-dir", type=Path, help="Existing extracted dataset to copy; no downloads")
    parser.add_argument("--raw-dir", type=Path, help="Preparation output; defaults to this experiment's data/raw/imagenetv2")
    parser.add_argument("--processed-dir", type=Path, help="New preparation output; defaults to data/processed/imagenetv2")
    parser.add_argument("--exploration-manifest", type=Path, help="Audited original selection; defaults to results/cpu_1000/selected_examples.json")
    parser.add_argument("--seed", type=int, help="Preparation seed; defaults to 20261019")
    args = parser.parse_args(argv)
    if args.prepare_data:
        if args.status_file is not None or args.provenance_config:
            parser.error("Status and provenance options require --config")
        if args.dry_run:
            parser.error("--dry-run describes a --config; it cannot be combined with --prepare-data")
        raw = args.raw_dir or EXPERIMENT_DIR / "data/raw/imagenetv2"
        processed = args.processed_dir or EXPERIMENT_DIR / "data/processed/imagenetv2"
        try:
            metadata = prepare_data(raw, processed, args.exploration_manifest or EXPLORATION,
                                    source_dir=args.source_dir,
                                    seed=20261019 if args.seed is None else args.seed)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(metadata["splits"], indent=2))
        print(f"Raw images: {raw}")
        print(f"Manifests and metadata: {processed}")
        return 0
    if any(value is not None for value in (args.source_dir, args.raw_dir, args.processed_dir,
                                           args.exploration_manifest, args.seed)):
        parser.error("Preparation options require --prepare-data")
    status = None
    try:
        if args.status_file is not None:
            validate_status_path(args.status_file)
        config, original = load_config(args.config)
        extra_names = (configuration_files(REPOSITORY_ROOT, EXPERIMENT_DIR, args.provenance_config)
                       if args.provenance_config else [])
        selected = args.config.expanduser().resolve()
        provenance_configs = {name: (REPOSITORY_ROOT / name).read_bytes() for name in extra_names
                              if (REPOSITORY_ROOT / name).resolve() != selected}
        if args.dry_run:
            print(describe(config))
            return 0
        if args.status_file is not None:
            status = start_status(args.status_file, args.config, original, config["tracking"])
        output = run_experiment(config, original, config_path=args.config,
                                provenance_configs=provenance_configs, status=status)
        # A finished tracking run stays intact if publishing this local receipt fails.
        update_status(status, phase="completion_receipt")
        finish_status(status, output)
        print(f"Completed model evaluation; all required artifacts uploaded. Local files: {output}")
        return 0
    except BaseException as error:
        fail_status(status, error)
        if not isinstance(error, Exception):
            raise
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
