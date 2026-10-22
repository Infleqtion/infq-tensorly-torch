"""One fresh-process, timing-only round; run rounds 1, 2, and 3 separately.

Both compressed copies exist before timing. Reuse the original timing helper,
rotate variant order, then measure dense once more. No evaluation images are read.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

import run_experiment as experiment
from tltorch import compress_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; never overwrite a round")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; retain it and choose a new path")
    saved = Path(__file__).parent / "results/cpu_1000"
    configuration = json.loads((saved / "configuration.json").read_text())
    expected_reports = json.loads((saved / "compression_reports.json").read_text())
    settings = {key: configuration["settings"][key]
                for key in ("device", "threads", "seed", "timing_seed", "warmup", "repeats")}
    assert settings == dict(device="cpu", threads=4, seed=2026, timing_seed=2027, warmup=20, repeats=100)
    device = experiment.configure_runtime(settings["device"], settings["threads"])
    environment = experiment.environment_info()
    torch.manual_seed(settings["seed"])
    baseline, _ = experiment.load_baseline(device)  # Cached weights only.
    original = experiment.model_fingerprint(baseline)
    weight_file = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    assert experiment.file_sha256(weight_file) == configuration["weights"]["sha256"]
    inputs = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32,
                         generator=torch.Generator(device=device).manual_seed(settings["timing_seed"]))
    models, conversions = {"dense": baseline}, {}
    for variant, fraction in experiment.VARIANTS[1:]:
        torch.manual_seed(settings["seed"])
        layers = experiment.layer_settings(fraction)
        model, report = compress_model(baseline, layers, implementation="factorized")
        assert experiment.model_fingerprint(baseline) == original
        assert not any(module.training for module in model.modules())
        assert all(p.device == device and p.dtype == torch.float32 for p in model.parameters())
        details = []
        for replacement, expected in zip(report["replacements"], expected_reports[variant]["replacements"]):
            name = replacement["name"]
            layer = model.get_submodule(name)
            rank = list(layer.weight.rank)
            assert name == expected["name"]
            assert replacement["requested_rank"] == fraction
            assert rank == list(layer.rank) == list(replacement["actual_rank"]) == expected["actual_rank"]
            assert layer.implementation == "factorized"
            assert layer.weight._name == layers[name]["factorization"]
            shapes = [list(factor.shape) for factor in layer.weight.factors]
            if name == "fc":
                assert layer.in_tensorized_features == (8, 8, 8)
                assert layer.out_tensorized_features == (10, 10, 10)
                assert shapes == [[rank[i], 10, 8, rank[i + 1]] for i in range(3)]
                core_shape = None
            else:
                core_shape = list(layer.weight.core.shape)
                assert core_shape == rank and rank[2:] == [3, 3]
                assert shapes == [[size, r] for size, r in zip((512, 512, 3, 3), rank)]
            details.append({"name": name, "settings": layers[name], "actual_rank": rank,
                            "implementation": layer.implementation, "factor_shapes": shapes,
                            "core_shape": core_shape})
        assert report["model"] == expected_reports[variant]["model"]
        models[variant] = model
        conversions[variant] = {"layers": details, "storage": report["model"]}
        print(f"Constructed {variant}; ranks and factor shapes verified", flush=True)

    # Rotate timing order, not construction order, to isolate this bounded question.
    names = list(models)
    offset = args.round - 1
    order = names[offset:] + names[:offset] + ["dense"]
    record = {"round": args.round, "pid": os.getpid(), "command": [sys.executable, *sys.orig_argv[1:]],
              "working_directory": str(Path.cwd()), "environment": environment,
              "probe_sha256": experiment.file_sha256(__file__), "settings": settings,
              "dtype": "float32", "input_shape": list(inputs.shape),
              "input_sha256": hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
              "weights": configuration["weights"], "conversions": conversions,
              "baseline_before_sha256": original, "order": order, "blocks": []}
    for position, variant in enumerate(order, start=1):
        model = models[variant]
        assert not any(module.training for module in model.modules())
        runtime = experiment.environment_info()
        block = {"position": position, "variant": variant, "dense_recheck": position == 4,
                 "started_utc": runtime["started_utc"], "threads": runtime["threads"],
                 "interop_threads": runtime["interop_threads"], "backend": runtime["backend"],
                 "evaluation_mode": True, "inference_mode": True}
        block["latency"] = experiment.measure_latency(model, inputs, warmup=settings["warmup"],
                                                       repeats=settings["repeats"])
        block["completed_utc"] = datetime.now(timezone.utc).isoformat()
        block["baseline_after_sha256"] = experiment.model_fingerprint(baseline)
        assert block["baseline_after_sha256"] == original
        assert block["threads"] == 4 and block["interop_threads"] == 1
        record["blocks"].append(block)
        latency = block["latency"]
        print(f"Round {args.round}, block {position}, {variant}: {latency['median_ms']:.3f} ms "
              f"[{latency['p25_ms']:.3f}, {latency['p75_ms']:.3f}]", flush=True)
    record["baseline_after_sha256"] = experiment.model_fingerprint(baseline)
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
