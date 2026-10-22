"""Offline checks for pairing and flag restoration; no pretrained models or timing thresholds."""

import pytest
import torch
from torch import nn
import torch.backends.opt_einsum as backend

import opt_einsum_comparison as comparison


@pytest.mark.skipif(not backend.is_available(), reason="Optional opt_einsum comparison environment required")
@pytest.mark.parametrize("initial_enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_execution_flags_restored_on_success_and_failure(initial_enabled, fail):
    original = backend.enabled, backend.strategy
    try:
        backend.enabled = True
        backend.strategy = "greedy"
        backend.enabled = initial_enabled
        try:
            with comparison.execution_mode(not initial_enabled):
                assert backend.enabled == (not initial_enabled)
                assert backend.strategy == "auto"
                if fail:
                    raise RuntimeError("deliberate failure")
        except RuntimeError as error:
            assert fail and str(error) == "deliberate failure"
        assert (backend.enabled, backend.strategy) == (initial_enabled, "greedy")
    finally:
        backend.enabled = True
        backend.strategy = original[1]
        backend.enabled = original[0]


def test_pairs_use_each_round_and_handle_reversed_order():
    blocks = []
    for round_number, timings in ((1, (("disabled", 4), ("auto", 2))),
                                  (2, (("auto", 6), ("disabled", 3)))):
        for condition, median in timings:
            blocks.append({"round": round_number, "model": "example", "scope": "classifier",
                           "condition": condition, "latency": {"median_ms": median, "p25_ms": median - 1,
                                                               "p75_ms": median + 1}})
    rows = comparison.paired_results(blocks)
    assert [row["speedup"] for row in rows] == [2, 0.5]
    assert [row["first_condition"] for row in rows] == ["disabled", "auto"]
    assert rows[1]["auto_p25_ms"] == 5
    blocks[1]["scope"] = "whole_model"
    with pytest.raises(ValueError, match="same round, model, and scope"):
        comparison.paired_results(blocks)


@pytest.mark.skipif(not backend.is_available(), reason="Optional opt_einsum comparison environment required")
def test_round_schedule_and_state_preservation(monkeypatch):
    model = nn.Module()
    model.fc = nn.Linear(2, 2).eval()
    models = {name: model for name in ("dense", "first", "second")}
    flags = backend.enabled, backend.strategy
    observed = []
    def fake_latency(module, inputs, *, warmup, repeats):
        assert (warmup, repeats) == (20, 100)
        observed.append(backend.enabled)
        return {"median_ms": 1, "p25_ms": 1, "p75_ms": 1, "samples_ms": [1] * repeats}
    monkeypatch.setattr(comparison.experiment, "measure_latency", fake_latency)
    blocks = comparison.measure_rounds(models, {name: torch.zeros(1, 2) for name in models}, torch.zeros(1, 2))
    assert len(blocks) == 48
    assert observed == ([False, True] * 6 + [True, False] * 6) * 2
    assert [blocks[index]["model_order"] for index in (0, 12, 24, 36)] == [
        ["dense", "first", "second"], ["first", "second", "dense"],
        ["second", "dense", "first"], ["dense", "first", "second"]]
    assert (backend.enabled, backend.strategy) == flags
