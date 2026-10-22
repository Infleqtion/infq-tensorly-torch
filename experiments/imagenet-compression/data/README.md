# Prepare ImageNetV2 for compression experiments

Run `run_experiment.py --prepare-data` from the experiment directory to copy an existing extracted ImageNetV2
matched-frequency dataset, validate the files, and prepare three selections.
The preparation functions use only Python's standard library; the shared entry point also imports the installed YAML parser. It does not load models, download
weights, decode images, or require a GPU.

- `data/raw/imagenetv2/` contains independent copies of the original image files,
  arranged in numeric class folders. Folder `10` means class 10.
- `data/processed/imagenetv2/` contains selection manifests and preparation
  metadata. It does **not** contain resized or normalized copies of the images.

Preparation, compression, evaluation, and reporting are defined in the parent
directory's `run_experiment.py`. They are separate explicit CLI actions: data
preparation never loads a model or creates a tracking run. The study compares
pretrained models without training or fine-tuning.

Resizing, cropping, and normalization remain in the evaluation loader, using the
preprocessing associated with `ResNet18_Weights.IMAGENET1K_V1`. Neither raw images
nor generated processed outputs belong in Git. The two default directories are
ignored; historical split audits remain under `../splits/` in the experiment directory.

From the repository root, using the existing uv environment:

```bash
uv run --no-sync python -B experiments/imagenet-compression/run_experiment.py --prepare-data \
  --source-dir "$HOME/.cache/tltorch-model-compression/data/imagenetv2-matched-frequency-format-val"
```

The source must contain classes `0` through `999`, with ten original image files
per class. Files are copied, not linked, and the source is never changed. Allow
about 1.3 GB of additional disk space. The script checks file paths and hashes;
this is data preparation, not evaluation of a model on the final set.

If the dataset is not available, follow the existing
[explicit download and checksum instructions](../HISTORICAL-RUNS.md#explicit-downloads-and-image-use-guidance).
Use the dataset archive download, checksum check, and extraction steps; the
pretrained-weight download in that section is unnecessary for preparation.
Those instructions also explain image-use terms. There is no automatic download.

| Selection | Images | Per class | Purpose |
| --- | ---: | ---: | --- |
| Exploration | 1,000 | 1 | Compare compression choices |
| Confirmation | 4,000 | 4 | Check a small shortlist |
| Final evaluation | 5,000 | 5 | Evaluate the chosen scheme after its settings are fixed |

The original 1,000 images have already been evaluated. They stay in exploration:
the default input is the committed
[`cpu_1000/selected_examples.json`](../results/cpu_1000/selected_examples.json),
and the output preserves its bytes exactly. For each class, the script shuffles
the other nine sorted paths with one local random generator, takes four for
confirmation, and leaves five for final evaluation. The default seed is
`20261019`; caller random state is unchanged. No predictions or accuracy values
influence the choices. Cross-selection duplicates, by path or file-content hash,
are rejected. Hashes detect identical files, not visually similar images.

Earlier recorded evaluations used the original 1,000-image selection, which stays
in exploration. The reviewed later rerun manifests contained the same images.
The review found no recorded evaluation outside that selection, but could not
rule out undocumented evaluations. Final-set freshness is relative to that
recorded use. Use confirmation to check a small shortlist; reserve final
evaluation for configurations whose settings are already fixed. Review any
additional prior use before treating these sets as unevaluated.

Four files are produced: `exploration.json`, `confirmation.json`,
`final_evaluation.json`, and `preparation.json`. Manifest entries use
dataset-relative paths, integer labels, and SHA-256 image hashes. Metadata records
dataset identity, the exploration input hash, paths/settings, seed and algorithm,
counts, the complete inventory hash, output hashes, Python version, and the
preparation script's hash. External source paths are recorded as provenance;
image paths inside the manifests remain portable.

To choose other locations or a seed:

```bash
uv run --no-sync python -B experiments/imagenet-compression/run_experiment.py --prepare-data \
  --source-dir /path/to/extracted-imagenetv2 \
  --raw-dir /path/to/project-data/raw/imagenetv2 \
  --processed-dir /path/to/project-data/processed/imagenetv2-run-002 \
  --seed 20261019
```

`--exploration-manifest` can point to another location of the audited original
manifest. It must still contain the already-evaluated selection. The default
paths are under the experiment directory containing the script. Custom relative
arguments resolve from the current working directory; keep custom generated
locations outside Git or add a suitable ignore rule.

An existing raw directory must pass the same structure and hash checks. When
`--source-dir` is supplied, it must also match that source exactly. Without that
argument, the script validates and uses the existing raw directory. It refuses
to overwrite conflicting raw data or **any existing processed directory**, even
for an identical request. To prepare another output, choose a new
`--processed-dir`; the existing raw files can be reused. An interrupted copy may
leave an incomplete raw directory; it will fail validation on retry rather than
be silently overwritten or repaired.

The configured runner consumes the supplied manifest directly, preserving order
and supporting all three partition sizes. The older notebook selection helper
retains its one-or-ten-images-per-class behavior. The already prepared raw and
processed directories were moved here unchanged: no split was regenerated and
the recorded preparation paths and hashes still describe the original execution.
See [the experiment README](../README.md) for the configured runner; connecting to
the existing MLflow server on Cheetah is still a separate step.

Run the small offline tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -B -m pytest \
  -q -p no:cacheprovider experiments/imagenet-compression/tests/test_prepare_data.py
```
