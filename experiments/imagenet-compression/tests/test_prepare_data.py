"""Offline preparation checks using two classes of small files, without image decoding."""

from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

import run_experiment as preparation


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"
    # Numeric order (2, 10) differs from lexical order; creation order is reversed too.
    for label in (10, 2):
        folder = source / str(label)
        folder.mkdir(parents=True)
        for index in reversed(range(10)):
            (folder / f"{index}.jpeg").write_bytes(f"class {label}, image {index}".encode())
    exploration = tmp_path / "original.json"
    entries = [{"path": f"{label}/0.jpeg", "label": label,
                "sha256": preparation.file_sha256(source / f"{label}/0.jpeg")} for label in (2, 10)]
    exploration.write_text(json.dumps({"dataset": preparation.DATASET, "selection": "one-per-class",
                                       "seed": 2026, "examples": entries}, indent=2) + "\n")
    return source, exploration


def build(source, exploration):
    inventory = preparation.read_inventory(source, class_ids=(2, 10))
    return preparation.build_splits(inventory, exploration.read_bytes())


def test_numeric_labels_counts_separation_and_reproducible_choices(dataset, monkeypatch):
    source, exploration = dataset
    before = random.getstate()
    splits, counts = build(source, exploration)
    assert random.getstate() == before
    assert splits["exploration"] == json.loads(exploration.read_text())
    seen_paths, seen_hashes = set(), set()
    for name, per_class in (("exploration", 1), ("confirmation", 4), ("final_evaluation", 5)):
        entries = splits[name]["examples"]
        assert Counter(entry["label"] for entry in entries) == {2: per_class, 10: per_class}
        assert counts[name] == {"images": 2 * per_class, "classes": 2, "images_per_class": per_class}
        assert all(entry["label"] == int(entry["path"].split("/")[0]) for entry in entries)
        paths = {entry["path"] for entry in entries}
        hashes = {entry["sha256"] for entry in entries}
        assert not seen_paths & paths and not seen_hashes & hashes
        seen_paths.update(paths)
        seen_hashes.update(hashes)
    assert len(seen_paths) == len(seen_hashes) == 20
    # Permuting the filesystem's enumeration must not change the seeded assignment.
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(reversed(list(original_iterdir(path)))))
    assert build(source, exploration) == (splits, counts)


def test_copy_preserves_source_and_exploration_and_reuses_matching_raw_data(dataset, tmp_path):
    source, exploration = dataset
    source_before = {p.relative_to(source): p.read_bytes() for p in source.glob("*/*")}
    exploration_before = exploration.read_bytes()
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    metadata = preparation.prepare_data(raw, processed, exploration, source_dir=source, class_ids=(2, 10))
    assert (processed / "exploration.json").read_bytes() == exploration_before
    assert exploration.read_bytes() == exploration_before
    for relative, contents in source_before.items():
        assert (source / relative).read_bytes() == (raw / relative).read_bytes() == contents
        assert not (raw / relative).is_symlink()
        assert not (raw / relative).samefile(source / relative)
    assert metadata["dataset_images"] == metadata["unique_image_hashes"] == 20
    assert metadata["exploration_input_sha256"] == preparation.file_sha256(exploration)
    assert metadata["code"]["sha256"] == preparation.file_sha256(preparation.__file__)
    for name, digest in metadata["output_manifest_sha256"].items():
        assert preparation.file_sha256(processed / f"{name}.json") == digest
    raw_before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in raw.glob("*/*")}
    for name, options in (("second", {"source_dir": source}), ("third", {})):
        repeated = preparation.prepare_data(raw, tmp_path / name, exploration, class_ids=(2, 10), **options)
        assert repeated["output_manifest_sha256"] == metadata["output_manifest_sha256"]
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in raw.glob("*/*")} == raw_before


@pytest.mark.parametrize("problem", ["wrong-label", "wrong-hash", "missing-file", "duplicate-path", "class-name"])
def test_invalid_inputs_fail_before_creating_destinations(dataset, tmp_path, problem):
    source, exploration = dataset
    manifest = json.loads(exploration.read_text())
    if problem == "wrong-label":
        manifest["examples"][1]["label"] = "10"
        message = "label.*must be integer 10"
    elif problem == "wrong-hash":
        manifest["examples"][0]["sha256"] = "0" * 64
        message = "mismatched SHA-256"
    elif problem == "missing-file":
        (source / "2/9.jpeg").unlink()
        message = "Class 2: expected 10 images, found 9"
    elif problem == "class-name":
        (source / "10").rename(source / "010")
        message = "numeric class folders.*unexpected.*010"
    else:
        manifest["examples"].append(manifest["examples"][0])
        message = "Duplicate exploration path"
    exploration.write_text(json.dumps(manifest))
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    with pytest.raises(ValueError, match=message):
        preparation.prepare_data(raw, processed, exploration, source_dir=source, class_ids=(2, 10))
    assert not raw.exists() and not processed.exists()


def test_duplicate_content_cannot_cross_exploration_boundary(dataset):
    source, exploration = dataset
    # A differently named copy of an evaluated image cannot enter either later selection.
    (source / "2/1.jpeg").write_bytes((source / "2/0.jpeg").read_bytes())
    with pytest.raises(ValueError, match="Exact duplicate content crosses split boundaries"):
        build(source, exploration)


def test_existing_raw_conflict_is_never_overwritten(dataset, tmp_path):
    source, exploration = dataset
    raw = tmp_path / "raw"
    preparation.prepare_data(raw, tmp_path / "first", exploration, source_dir=source, class_ids=(2, 10))
    (raw / "2/9.jpeg").write_bytes(b"local data must survive")
    with pytest.raises(ValueError, match="Existing raw data differs from the source"):
        preparation.prepare_data(raw, tmp_path / "second", exploration,
                                 source_dir=source, class_ids=(2, 10))
    assert (raw / "2/9.jpeg").read_bytes() == b"local data must survive"
    assert (source / "2/9.jpeg").read_bytes() == b"class 2, image 9"
    assert not (tmp_path / "second").exists()


def test_command_protects_existing_processed_directory_before_reading_data(tmp_path, capsys):
    processed = tmp_path / "existing"
    processed.mkdir()
    sentinel = processed / "keep.txt"
    sentinel.write_text("existing work")
    with pytest.raises(SystemExit) as error:
        preparation.main(["--prepare-data", "--source-dir", str(tmp_path / "missing-source"),
                          "--processed-dir", str(processed), "--seed", "12"])
    assert error.value.code == 2
    assert "Processed directory already exists" in capsys.readouterr().err
    assert sentinel.read_text() == "existing work"
    assert list(processed.iterdir()) == [sentinel]


def test_nested_destinations_cannot_modify_the_source(dataset):
    source, exploration = dataset
    before = sorted(source.rglob("*"))
    with pytest.raises(ValueError, match="Source and raw directories must not be nested"):
        preparation.prepare_data(source / "raw", source.parent / "processed", exploration,
                                 source_dir=source, class_ids=(2, 10))
    assert sorted(source.rglob("*")) == before


def test_preparation_action_uses_tiny_fixture_without_models_or_tracking(dataset, tmp_path):
    source, exploration = dataset
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    code = """
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import run_experiment
prepare = run_experiment.prepare_data
# Only shrink the dataset class universe; use the real preparation and CLI code.
def tiny_prepare(*args, **kwargs):
    return prepare(*args, class_ids=(2, 10), **kwargs)
run_experiment.prepare_data = tiny_prepare
with patch("socket.socket.connect", side_effect=AssertionError("no network")):
    assert run_experiment.main(["--prepare-data", "--source-dir", sys.argv[2],
                               "--exploration-manifest", sys.argv[3],
                               "--raw-dir", sys.argv[4], "--processed-dir", sys.argv[5]]) == 0
assert not any(name in sys.modules for name in ("torch", "torchvision", "numpy", "mlflow", "tltorch"))
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(preparation.EXPERIMENT_DIR),
                    str(source), str(exploration), str(raw), str(processed)],
                   cwd=tmp_path, capture_output=True, text=True, check=True)
    assert (processed / "exploration.json").read_bytes() == exploration.read_bytes()
    assert len(list(raw.glob("*/*.jpeg"))) == 20
    metadata = json.loads((processed / "preparation.json").read_text())
    assert metadata["splits"] == {
        "exploration": {"images": 2, "classes": 2, "images_per_class": 1},
        "confirmation": {"images": 8, "classes": 2, "images_per_class": 4},
        "final_evaluation": {"images": 10, "classes": 2, "images_per_class": 5},
    }


@pytest.mark.parametrize("arguments", [
    ["--prepare-data", "--config", "unused.yaml"],
    ["--prepare-data", "--dry-run"],
    ["--config", "unused.yaml", "--source-dir", "unused-data"],
])
def test_cli_does_not_mix_preparation_and_execution(arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Mixed CLI actions must fail before reading inputs")

    monkeypatch.setattr(preparation, "prepare_data", forbidden)
    monkeypatch.setattr(preparation, "load_config", forbidden)
    with pytest.raises(SystemExit) as error:
        preparation.main(arguments)
    assert error.value.code == 2
