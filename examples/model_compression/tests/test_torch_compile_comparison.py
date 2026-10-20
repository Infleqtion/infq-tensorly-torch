"""Offline checks for the compile experiment; no actual compilation or model downloads."""

import pytest
import torch
from torch import nn

from examples.model_compression import torch_compile_comparison as comparison


def test_counter_changes_include_guard_failures():
    assert comparison.compiler_changes({"graphs": 4, "guard_failures": 1},
                                       {"graphs": 4, "guard_failures": 2}) == {"guard_failures": 1}
    assert comparison.compiler_changes({"graphs": 4}, {"graphs": 4}) == {}


def test_paired_results_keep_rounds_and_unavailable_cases():
    blocks = []
    for round_number, order in ((1, ["eager", "compiled"]), (2, ["compiled", "eager"])):
        for name, timings in (("dense", {"eager": 4, "compiled": 2}),
                              ("compressed", {"eager": 8, "compiled": 3})):
            for condition in order:
                median = timings[condition]
                blocks.append({"round": round_number, "model": name, "condition": condition,
                               "valid_after_compilation": True,
                               "latency": {"median_ms": median, "p25_ms": median - 1, "p75_ms": median + 1}})
    rows = comparison.paired_results(blocks)
    assert [row["first_condition"] for row in rows] == ["eager", "eager", "compiled", "compiled"]
    assert rows[0]["speedup"] == 2
    assert rows[1]["compiled_vs_dense_latency_ratio"] == 1.5
    blocks[-2]["valid_after_compilation"] = False
    assert comparison.paired_results(blocks)[-1]["speedup"] is None
    assert comparison.paired_results(blocks)[-1]["compiled_status"] == "invalid"
    rows = comparison.paired_results([block for block in blocks if block["condition"] == "eager"])
    assert all(row["compiled_status"] == "unavailable" and row["speedup"] is None for row in rows)
    with pytest.raises(ValueError, match="Duplicate"):
        comparison.paired_results(blocks + [blocks[0]])


def test_round_schedule_and_recompilation_marking(monkeypatch):
    models = {name: nn.Linear(2, 2).eval() for name in ("dense", "first", "second")}
    snapshot = {"graphs": 0}
    calls = []
    monkeypatch.setattr(comparison, "compiler_snapshot", lambda: dict(snapshot))
    monkeypatch.setattr(comparison, "checked_backend_settings", lambda: {"enabled": False})
    def fake_latency(model, inputs, *, warmup, repeats):
        assert (warmup, repeats) == (20, 100)
        calls.append(model)
        if len(calls) == 2:
            snapshot["graphs"] += 1
        return {"median_ms": 1, "p25_ms": 1, "p75_ms": 1, "samples_ms": [1] * repeats}
    monkeypatch.setattr(comparison.experiment, "measure_latency", fake_latency)
    blocks = comparison.measure_rounds(models, models, torch.zeros(1, 2))
    assert len(blocks) == 24
    assert [block["condition"] for block in blocks] == (["eager", "compiled"] * 3 + ["compiled", "eager"] * 3) * 2
    assert [blocks[index]["model_order"] for index in (0, 6, 12, 18)] == [
        ["dense", "first", "second"], ["first", "second", "dense"],
        ["second", "dense", "first"], ["dense", "first", "second"]]
    assert not blocks[1]["valid_after_compilation"]
    assert blocks[1]["compiler_changes"] == {"graphs": 1}
    assert len(blocks[1]["latency"]["samples_ms"]) == 100
    assert comparison.paired_results(blocks)[0]["speedup"] is None


def test_agreement_failure_is_reported_without_relaxing_tolerance():
    results = comparison.compare_outputs(nn.Identity(), lambda x: x + 0.01,
                                         [torch.ones(1, 2)], rtol=1e-4, atol=1e-5)
    assert not results[0]["passed"]
    assert results[0]["rtol"] == 1e-4 and results[0]["atol"] == 1e-5
    assert results[0]["max_absolute_difference"] == pytest.approx(0.01)
    assert "not close" in results[0]["error"]
