"""Diagnostics and paired CPU timings for the focused torch.compile notebook."""

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import traceback

import torch
from torch._dynamo.utils import counters, guard_failures
from torch._inductor import metrics
from torch._logging import _internal as torch_logging

from . import experiment
from .opt_einsum_comparison import backend_settings


def checked_backend_settings():
    """Require the same non-einsum execution settings in every experiment phase."""
    actual = backend_settings()
    assert not actual["enabled"]
    assert actual["tensorly_backend"] == "pytorch"
    assert actual["tensorly_tenalg_backend"] == "core"
    assert actual["tensorly_default_einsum"] and actual["tensorly_plugin_inactive"]
    assert actual["tensorly_plugin_cache_entries"] == 0
    assert not torch._dynamo.config.suppress_errors
    return actual


def compiler_snapshot():
    """Read built-in counters outside timers; do not attach hooks to inference."""
    return {
        "frames_started": counters["frames"]["total"],
        "unique_graphs": counters["stats"]["unique_graphs"],
        "inductor_cache_hits": counters["inductor"]["fxgraph_cache_hit"],
        "inductor_cache_misses": counters["inductor"]["fxgraph_cache_miss"],
        "generated_kernels": metrics.generated_kernel_count,
        "guard_failures": sum(len(reasons) for reasons in guard_failures.values()),
    }


def compiler_changes(before, after):
    """Keep changed counters, including guard failures that warrant investigation."""
    return {key: after[key] - value for key, value in before.items() if after[key] != value}


@contextmanager
def diagnostic_log(path):
    """Save PyTorch 2.5 diagnostics temporarily, restoring logging afterward."""
    previous_state = copy.deepcopy(torch_logging._get_log_state())
    previous_output = os.environ.get("TORCH_LOGS_OUT")
    try:
        os.environ["TORCH_LOGS_OUT"] = str(path)
        torch._logging.set_logs(dynamo=logging.INFO, inductor=logging.INFO,
                                graph_breaks=True, recompiles=True)
        # Keep verbose diagnostics in artifacts, not in the teaching notebook output.
        for name in torch_logging.log_registry.get_log_qnames():
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                if type(handler) is logging.StreamHandler and torch_logging._is_torch_handler(handler):
                    logger.removeHandler(handler)
        yield
    finally:
        if previous_output is None:
            os.environ.pop("TORCH_LOGS_OUT", None)
        else:
            os.environ["TORCH_LOGS_OUT"] = previous_output
        torch_logging._set_log_state(previous_state)
        torch_logging._init_logs()


@torch.inference_mode()
def explain_capture(model, inputs):
    """Describe graph capture before Inductor compilation; explain itself resets Dynamo."""
    explanation = torch._dynamo.explain(model)(inputs)
    return {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "operation_count": explanation.op_count,
        "break_reasons": [
            {"reason": reason.reason,
             "user_stack": [{"file": frame.filename, "line": frame.lineno,
                             "function": frame.name, "code": frame.line}
                            for frame in reason.user_stack]}
            for reason in explanation.break_reasons
        ],
        "graphs": [{"index": index, "code": graph.code,
                    "operations": [str(op) for op in explanation.ops_per_graph[index]]}
                   for index, graph in enumerate(explanation.graphs)],
        "note": "Dynamo capture diagnostic only; optimized execution is verified separately with Inductor.",
    }


def cache_inventory(directory):
    """Record private cache contents and external operators without saving binaries."""
    files = [path for path in Path(directory).rglob("*") if path.is_file()]
    sources = []
    for path in sorted(files):
        if path.suffix != ".py":
            continue
        code = path.read_text()
        sources.append({"path": str(path.relative_to(directory)),
                        "sha256": experiment.file_sha256(path),
                        "external_calls": sorted(set(re.findall(r"extern_kernels\.([a-zA-Z_]+)", code))),
                        "torch_operator_calls": sorted(set(re.findall(r"torch\.ops\.[\w.]+", code)))})
    return {"files": len(files), "bytes": sum(path.stat().st_size for path in files),
            "shared_libraries": sum(path.suffix == ".so" for path in files), "python_sources": sources}


@torch.inference_mode()
def compare_outputs(eager, compiled, inputs, *, rtol, atol):
    """Retain every input's error and assertion result, without changing tolerance."""
    results = []
    for index, data in enumerate(inputs):
        reference = eager(data)
        actual = compiled(data)
        error = None
        try:
            torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        except AssertionError as failure:
            error = str(failure)
        results.append({"input_index": index, "max_absolute_difference": (actual - reference).abs().max().item(),
                        "reference_max_absolute_value": reference.abs().max().item(),
                        "rtol": rtol, "atol": atol, "passed": error is None, "error": error})
    return results


def measure_rounds(models, compiled_models, inputs, *, rounds=4, warmup=20, repeats=100):
    """Keep pairs adjacent and retain invalid blocks if compiler activity occurs."""
    original = {name: experiment.model_fingerprint(model) for name, model in models.items()}
    names = list(models)
    blocks = []
    for round_index in range(rounds):
        offset = round_index % len(names)
        model_order = names[offset:] + names[:offset]
        condition_order = ["eager", "compiled"] if round_index % 2 == 0 else ["compiled", "eager"]
        for name in model_order:
            for condition in condition_order:
                model = models[name] if condition == "eager" else compiled_models.get(name)
                if model is None:
                    continue
                actual_settings = checked_backend_settings()
                before = compiler_snapshot()
                block = {"round": round_index + 1, "block": len(blocks) + 1,
                         "model": name, "condition": condition, "model_order": model_order,
                         "condition_order": condition_order, "backend": actual_settings,
                         "started_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    block["latency"] = experiment.measure_latency(model, inputs, warmup=warmup, repeats=repeats)
                    block["error"] = None
                except Exception:
                    block["latency"] = None
                    block["error"] = traceback.format_exc()
                after = compiler_snapshot()
                changes = compiler_changes(before, after)
                block.update(compiler_before=before, compiler_after=after, compiler_changes=changes,
                             valid_after_compilation=not changes and block["error"] is None)
                blocks.append(block)
            print(f"Round {round_index + 1}: {name} complete", flush=True)
        for name, model in models.items():
            if experiment.model_fingerprint(model) != original[name]:
                raise RuntimeError(f"Round {round_index + 1} changed {name}'s parameters or buffers")
    return blocks


def paired_results(blocks):
    """Report unavailable comparisons honestly; never turn an invalid block into a speedup."""
    groups = {}
    for block in blocks:
        key = (block["round"], block["model"])
        conditions = groups.setdefault(key, {})
        if block["condition"] in conditions:
            raise ValueError("Duplicate condition for the same round and model")
        conditions[block["condition"]] = block
    rows = []
    for (round_number, name), conditions in groups.items():
        row = {"round": round_number, "model": name, "first_condition": next(iter(conditions))}
        for condition in ("eager", "compiled"):
            block = conditions.get(condition)
            valid = block is not None and block["valid_after_compilation"]
            row[f"{condition}_status"] = "ok" if valid else ("unavailable" if block is None else "invalid")
            for metric in ("median_ms", "p25_ms", "p75_ms"):
                row[f"{condition}_{metric}"] = block["latency"][metric] if valid else None
        eager, compiled = row["eager_median_ms"], row["compiled_median_ms"]
        row["speedup"] = eager / compiled if eager is not None and compiled is not None else None
        rows.append(row)
    dense_by_round = {row["round"]: row["compiled_median_ms"] for row in rows if row["model"] == "dense"}
    for row in rows:
        dense, compiled = dense_by_round.get(row["round"]), row["compiled_median_ms"]
        row["compiled_vs_dense_latency_ratio"] = compiled / dense if dense is not None and compiled is not None else None
    return rows
