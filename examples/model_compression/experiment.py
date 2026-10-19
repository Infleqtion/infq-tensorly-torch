"""Compare pretrained ResNet-18 with two factorized copies on ImageNetV2.

Imports and offline checks never download data or weights. Run from the repository
root with ``python -m examples.model_compression.experiment --help``.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from urllib.parse import urlparse

import torch
from torch.utils.data import DataLoader, Dataset

from tltorch import compress_model

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


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


class ManifestImages(Dataset):
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
    return DataLoader(ManifestImages(dataset_dir, manifest, transform), batch_size=batch_size,
                      shuffle=False, drop_last=False, num_workers=workers,
                      generator=torch.Generator().manual_seed(seed))


def configure_runtime(device, threads=4):
    """Use explicit float32 backend settings for every variant in a fresh process."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested device {device} is unavailable or unsupported; no fallback is used")
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return device


def load_baseline(device="cpu", *, allow_download=False):
    """Load the specified torchvision weights; downloading requires explicit opt-in."""
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not cached.is_file() and not allow_download:
        raise FileNotFoundError(f"Weights are not cached at {cached}. Follow README setup or use "
                                "--allow-weight-download (approximately 45 MB).")
    baseline = resnet18(weights=weights).to(device=device, dtype=torch.float32).eval()
    return baseline, weights.transforms()


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
    if logits.ndim != 2 or logits.shape[1] < 5 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected N-by-C predictions with C >= 5 and N integer labels")
    if not torch.isfinite(logits).all() or (labels < 0).any() or (labels >= logits.shape[1]).any():
        raise ValueError("Nonfinite predictions or class labels outside the model output range")
    predicted = logits.topk(5, dim=1).indices
    correct = predicted.eq(labels[:, None])
    return int(correct[:, 0].sum()), int(correct.any(dim=1).sum()), labels.numel()


@torch.inference_mode()
def evaluate_accuracy(model, batches, device="cpu", *, progress=False):
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


@torch.inference_mode()
def measure_latency(model, inputs, *, warmup=20, repeats=100):
    """Time completed forward passes in milliseconds per batch, after warm-up."""
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


def measure_variant(model, batches, inputs, *, warmup=20, repeats=100, progress=False):
    return {"accuracy": evaluate_accuracy(model, batches, inputs.device, progress=progress),
            "latency": measure_latency(model, inputs, warmup=warmup, repeats=repeats)}


def run_comparison(baseline, batches, inputs, *, seed=2026, warmup=20, repeats=100):
    """Always convert the untouched baseline; keep construction outside inference mode."""
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


def comparison_rows(observations, reports):
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
    import tltorch

    root = Path(__file__).resolve().parents[2]
    packages = {}
    for name in ("torch", "torchvision", "tensorly", "numpy", "scipy", "Pillow", "nbformat", "nbclient", "ipykernel"):
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
            "backend": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
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


def save_results(output_dir, observations, reports, manifest, settings, environment, preprocessing):
    """Save small review artifacts; never overwrite a previous comparison."""
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
    rows = comparison_rows(observations, reports)
    write_csv(output / "results.csv", rows)
    artifacts = {"configuration.json": configuration, "selected_examples.json": manifest,
                 "compression_reports.json": reports, "measurements.json": observations,
                 "latency_samples.json": {name: item["latency"]["samples_ms"] for name, item in observations.items()}}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extracted numeric class folders; no dataset auto-download")
    parser.add_argument("--output", required=True, help="New/empty directory for review artifacts")
    parser.add_argument("--manifest", help="Replay selected_examples.json, checking image content hashes")
    parser.add_argument("--archive", help="Optional original archive for additional provenance")
    parser.add_argument("--full", action="store_true", help="Evaluate all 10,000 images, not one per class")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16, help="Accuracy batch size, identical for all variants")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026, help="Model construction/conversion seed")
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--timing-seed", type=int, default=2027)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--allow-weight-download", action="store_true", help="Opt in to approximately 45 MB of pretrained weights")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.warmup < 0 or args.repeats < 2:
        parser.error("Use positive batch size, nonnegative workers/warmup, and at least two timed passes")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error("--output must be a new or empty directory")
    if args.archive and file_sha256(args.archive) != ARCHIVE_SHA256:
        parser.error("--archive does not match the pinned ImageNetV2 matched-frequency archive")
    device = configure_runtime(args.device, args.threads)
    torch.manual_seed(args.seed)
    manifest = select_examples(args.dataset, seed=args.selection_seed, full=args.full, manifest_path=args.manifest)
    baseline, preprocessing = load_baseline(device, allow_download=args.allow_weight_download)
    batches = make_loader(args.dataset, manifest, preprocessing, batch_size=args.batch_size, workers=args.workers, seed=args.seed)
    inputs = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32,
                         generator=torch.Generator(device=device).manual_seed(args.timing_seed))
    environment = environment_info()
    observations, reports = run_comparison(baseline, batches, inputs, seed=args.seed, warmup=args.warmup, repeats=args.repeats)
    rows = save_results(args.output, observations, reports, manifest, vars(args), environment, preprocessing)
    print("\n" + results_table(rows))
    print("Latency is per batch of one; FLOPs unavailable: " + FLOPS["reason"])
    print(f"Saved actual measurements to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
