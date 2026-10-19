"""Offline example checks: no torchvision, pretrained weights, or dataset download."""

import csv
import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from examples.model_compression import experiment


def make_folders(root):
    # Deliberately create names in an order different from numeric or lexical order.
    for label in (10, 2):
        folder = root / str(label)
        folder.mkdir()
        for filename in ("c.jpeg", "a.jpeg", "b.jpeg"):
            (folder / filename).write_bytes(f"fixture {label} {filename}".encode())


def test_numeric_labels_sorted_selection_and_manifest_replay(tmp_path):
    make_folders(tmp_path)
    options = {"class_ids": (2, 10), "images_per_class": 3}
    first = experiment.select_examples(tmp_path, seed=7, **options)
    second = experiment.select_examples(tmp_path, seed=7, **options)
    assert first == second
    # Random(7) chooses index 1 then 0 from the sorted candidates a,b,c.
    assert [(row["path"], row["label"]) for row in first["examples"]] == [("2/b.jpeg", 2), ("10/a.jpeg", 10)]
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(json.dumps(first))
    assert experiment.select_examples(tmp_path, seed=999, manifest_path=manifest_path, **options) == first
    full = experiment.select_examples(tmp_path, full=True, **options)
    assert len(full["examples"]) == 6
    assert [row["label"] for row in full["examples"]] == [2, 2, 2, 10, 10, 10]


@pytest.mark.parametrize("failure", ["missing-class", "missing-image", "wrong-label", "duplicate", "changed-content"])
def test_incomplete_or_changed_dataset_fails(tmp_path, failure):
    make_folders(tmp_path)
    options = {"class_ids": (2, 10), "images_per_class": 3}
    manifest = experiment.select_examples(tmp_path, seed=7, **options)
    if failure == "missing-class":
        options["class_ids"] = (2, 10, 11)
    elif failure == "missing-image":
        (tmp_path / "2/c.jpeg").unlink()
    elif failure == "wrong-label":
        manifest["examples"][0]["label"] = 10
    elif failure == "duplicate":
        manifest["examples"][1] = manifest["examples"][0]
    else:
        (tmp_path / manifest["examples"][0]["path"]).write_bytes(b"changed")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        experiment.select_examples(tmp_path, manifest_path=path, **options)


def test_accuracy_aggregates_counts_across_unequal_batches():
    # First row: top-1 correct; second: only top-5 correct; third: neither.
    logits = torch.tensor([[6., 5., 4., 3., 2., 1.],
                           [6., 5., 4., 3., 2., 1.],
                           [6., 5., 4., 3., 2., 1.]])
    labels = torch.tensor([0, 4, 5])
    result = experiment.evaluate_accuracy(nn.Identity(), [(logits[:2], labels[:2]), (logits[2:], labels[2:])])
    assert result["top1_correct"] == 1
    assert result["top5_correct"] == 2
    assert result["examples"] == 3
    assert result["top1_percent"] == pytest.approx(100 / 3)
    assert result["top5_percent"] == pytest.approx(200 / 3)


def test_unreadable_selected_image_is_not_skipped(tmp_path):
    pytest.importorskip("PIL.Image")
    (tmp_path / "bad.jpeg").write_bytes(b"not an image")
    images = experiment.ManifestImages(tmp_path, {"examples": [{"path": "bad.jpeg", "label": 2}]}, lambda x: x)
    with pytest.raises(RuntimeError, match="Cannot read.*bad.jpeg") as error:
        images[0]
    assert error.value.__cause__ is not None


def test_import_does_not_import_torchvision_or_download():
    code = '''
import sys
from unittest.mock import patch
with patch("urllib.request.urlopen", side_effect=AssertionError("No network on import")):
    from examples.model_compression import experiment
assert "torchvision" not in sys.modules
'''
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_small_measurement_and_conversion_integration(monkeypatch):
    torch.manual_seed(4)
    baseline = nn.Sequential(nn.Linear(4, 6)).eval()
    original = experiment.model_fingerprint(baseline)
    # Keep this integration check small; the real experiment supplies ResNet layer names.
    monkeypatch.setattr(experiment, "layer_settings", lambda fraction: {
        "0": {"factorization": "BlockTT", "rank": fraction,
              "in_tensorized_features": (2, 2), "out_tensorized_features": (2, 3)}})
    batches = [(torch.ones(2, 4), torch.tensor([0, 1])), (torch.zeros(1, 4), torch.tensor([2]))]
    observations, reports = experiment.run_comparison(baseline, batches, torch.ones(1, 4), warmup=1, repeats=3)
    rows = experiment.comparison_rows(observations, reports)
    assert len(rows) == 3 and all(row["examples"] == 3 for row in rows)
    assert rows[0]["parameter_count"] == 30  # 6x4 trained weights plus six biases.
    assert all(row["flops_total"] is None for row in rows)
    assert all(len(item["latency"]["samples_ms"]) == 3 for item in observations.values())
    assert experiment.model_fingerprint(baseline) == original
    assert "dense" in experiment.results_table(rows)


def test_csv_formats_summaries_without_changing_measurements(tmp_path):
    row = {"variant": "example", "top1_percent": 51.6, "top5_percent": 76.1,
           "top1_change_pp": 51.6 - 56.8, "top5_change_pp": 76.1 - 80.3,
           "latency_median_ms": 25.5705185, "latency_p25_ms": 25.3763935,
           "latency_p75_ms": 25.77409975, "parameter_count": 10376276,
           "parameter_bytes": 41505104, "buffer_bytes": 38560, "flops_total": None}
    original = dict(row)
    path = tmp_path / "results.csv"
    experiment.write_csv(path, [row])
    with path.open() as stream:
        saved = next(csv.DictReader(stream))
    assert saved == {"variant": "example", "top1_percent": "51.60", "top5_percent": "76.10",
                     "top1_change_pp": "-5.20", "top5_change_pp": "-4.20",
                     "latency_median_ms": "25.571", "latency_p25_ms": "25.376",
                     "latency_p75_ms": "25.774", "parameter_count": "10376276",
                     "parameter_bytes": "41505104", "buffer_bytes": "38560", "flops_total": ""}
    assert row == original
    assert b"\r" not in path.read_bytes()
