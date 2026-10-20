"""Small helpers for the optional, paired opt_einsum notebook experiment."""

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import torch
import torch.backends.opt_einsum as torch_opt_einsum
import tensorly as tl
from tensorly import plugins, tenalg

from . import experiment


@contextmanager
def execution_mode(enabled):
    """Temporarily select disabled or auto, restoring both flags even on failure."""
    if not torch_opt_einsum.is_available():
        raise RuntimeError("Use the separate comparison environment with opt_einsum installed")
    previous_enabled, previous_strategy = torch_opt_einsum.enabled, torch_opt_einsum.strategy
    try:
        # PyTorch only permits setting a strategy while its integration is enabled.
        torch_opt_einsum.enabled = True
        torch_opt_einsum.strategy = "auto"
        torch_opt_einsum.enabled = enabled
        yield
    finally:
        torch_opt_einsum.enabled = True
        torch_opt_einsum.strategy = previous_strategy
        torch_opt_einsum.enabled = previous_enabled


def backend_settings():
    """Record actual settings; installation alone does not establish use."""
    return {"available": torch_opt_einsum.is_available(), "enabled": torch_opt_einsum.enabled,
            "strategy": torch_opt_einsum.strategy, "tensorly_backend": tl.get_backend(),
            "tensorly_tenalg_backend": tenalg.get_backend(),
            "tensorly_default_einsum": tl.backend.current_backend().einsum is torch.einsum,
            "tensorly_plugin_inactive": plugins.PREVIOUS_EINSUM is None,
            "tensorly_plugin_cache_entries": len(plugins.OPT_EINSUM_PATH_CACHE),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "autocast_enabled": torch.is_autocast_enabled("cpu")}


@torch.inference_mode()
def classifier_features(model, inputs):
    """Capture the classifier's real input once; remove the hook before timing."""
    captured = []
    handle = model.fc.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    try:
        with execution_mode(False):
            model(inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one classifier call")
    return captured[0]


@torch.inference_mode()
def verify_optimizer_use(classifier, features):
    """Observe path selection in two untimed forwards, then remove the wrapper."""
    import opt_einsum

    original = opt_einsum.contract_path
    records = []
    for enabled in (False, True):
        with execution_mode(enabled), patch.object(opt_einsum, "contract_path", wraps=original) as traced:
            classifier(features)
            calls = [{"equation": call.args[0], "operand_shapes": [list(x.shape) for x in call.args[1:]],
                      "strategy": call.kwargs["optimize"]} for call in traced.call_args_list]
            records.append({"enabled": enabled, "calls": calls, "backend": backend_settings()})
        if len(calls) != int(enabled):
            raise AssertionError(f"Expected {int(enabled)} path searches, observed {len(calls)}")
    assert opt_einsum.contract_path is original
    return records


@torch.inference_mode()
def compare_outputs(module, inputs, *, rtol, atol):
    """Check execution agreement, retaining unrounded maximum absolute differences."""
    with execution_mode(False):
        disabled = module(inputs)
    with execution_mode(True):
        enabled = module(inputs)
    torch.testing.assert_close(enabled, disabled, rtol=rtol, atol=atol)
    return {"max_absolute_difference": (enabled - disabled).abs().max().item(),
            "reference_max_absolute_value": disabled.abs().max().item(),
            "rtol": rtol, "atol": atol, "output_shape": list(disabled.shape)}


def measure_rounds(models, features, inputs, *, rounds=4, warmup=20, repeats=100):
    """Keep each disabled/auto pair adjacent, alternating order and rotating models."""
    original = {name: experiment.model_fingerprint(model) for name, model in models.items()}
    names = list(models)
    blocks = []
    for round_index in range(rounds):
        offset = round_index % len(names)
        model_order = names[offset:] + names[:offset]
        condition_order = (False, True) if round_index % 2 == 0 else (True, False)
        for name in model_order:
            model = models[name]
            for scope, module, data in (("classifier", model.fc, features[name]), ("whole_model", model, inputs)):
                for enabled in condition_order:
                    with execution_mode(enabled):
                        actual = backend_settings()
                        started = datetime.now(timezone.utc).isoformat()
                        latency = experiment.measure_latency(module, data, warmup=warmup, repeats=repeats)
                    blocks.append({"round": round_index + 1, "block": len(blocks) + 1,
                                   "model": name, "scope": scope, "condition": "auto" if enabled else "disabled",
                                   "model_order": model_order, "condition_order": list(condition_order),
                                   "backend": actual, "started_utc": started, "latency": latency})
                print(f"Round {round_index + 1}: {name}, {scope} pair complete", flush=True)
        for name, model in models.items():
            if experiment.model_fingerprint(model) != original[name]:
                raise RuntimeError(f"Round {round_index + 1} changed {name}'s parameters or buffers")
    return blocks


def paired_results(blocks):
    """Summarize each adjacent pair; speedup is disabled median divided by auto median."""
    if len(blocks) % 2:
        raise ValueError("Timing records must contain complete pairs")
    rows = []
    for index in range(0, len(blocks), 2):
        pair = blocks[index:index + 2]
        key = tuple(pair[0][name] for name in ("round", "model", "scope"))
        if tuple(pair[1][name] for name in ("round", "model", "scope")) != key:
            raise ValueError("Adjacent blocks must describe the same round, model, and scope")
        conditions = {block["condition"]: block for block in pair}
        if set(conditions) != {"disabled", "auto"}:
            raise ValueError("Each pair needs disabled and auto measurements")
        row = dict(zip(("round", "model", "scope"), key))
        row["first_condition"] = pair[0]["condition"]
        for condition, block in conditions.items():
            for metric in ("median_ms", "p25_ms", "p75_ms"):
                row[f"{condition}_{metric}"] = block["latency"][metric]
        row["speedup"] = row["disabled_median_ms"] / row["auto_median_ms"]
        rows.append(row)
    return rows
