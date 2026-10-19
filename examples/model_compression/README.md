# ResNet-18 model-compression comparison

This example compares a pretrained dense model with two independently compressed
copies on the same ImageNetV2 matched-frequency images. It measures task accuracy,
registered tensor storage, and inference latency. No training or fine-tuning occurs.

The example uses torchvision `ResNet18_Weights.IMAGENET1K_V1` and its own preprocessing:
resize to 256 with bilinear interpolation, center crop to 224, and ImageNet
normalization. See the [weight documentation](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html).

For the current uv environment, follow the [repository setup](../../README.rst#using-uv-for-this-checkout-and-its-notebooks).
The main notebook and CLI now explicitly disable PyTorch's opt_einsum integration
in `configure_runtime`, even if the optional package is installed. Newly generated
`configuration.json` files record the actual flags in `environment.backend.opt_einsum`
and the inactive TensorLy plugin in `environment.backend.tensorly_plugin_inactive`.
The dedicated opt_einsum comparison still tests both conditions; the compile
comparison keeps it disabled. Saved results and the historical setup below have
not been rewritten. The main notebook's saved outputs precede these setup changes.

## Setup outside the repository

Run commands from the repository root. Python 3.12 was used. These optional example
dependencies are separate from the library requirements. The CPU wheels below are
an [official compatible pairing](https://pytorch.org/get-started/previous-versions/).
The shared `MPO_Compression` environment does not need to change.

The reusable commands use `$HOME/.cache/tltorch-model-compression`, matching the
notebooks' home-relative cache default. Historical commands and saved environment
records retain the paths used for the recorded runs.

The notebook uses Matplotlib **3.10.8**, pinned in `requirements.txt`. It supports
Python 3.12 ([release metadata](https://pypi.org/project/matplotlib/3.10.8/)). For an
existing example environment that already has the other dependencies, add only
plotting support with:

```bash
~/.cache/tltorch-model-compression/venv/bin/python -m pip install matplotlib==3.10.8
```

```bash
EXAMPLE_CACHE="$HOME/.cache/tltorch-model-compression"
python3 -m venv "$EXAMPLE_CACHE/venv"
EXAMPLE_PY="$EXAMPLE_CACHE/venv/bin/python"
"$EXAMPLE_PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
"$EXAMPLE_PY" -m pip install -r examples/model_compression/requirements.txt
export TORCH_HOME="${TORCH_HOME:-$EXAMPLE_CACHE/torch}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD"
```

For CUDA, use a matching CUDA wheel pair from the official installation page in a
separate environment, then pass `--device cuda`. The saved comparison uses CPU;
CUDA was not verified. There is no automatic device fallback.

## Explicit downloads and image-use guidance

Allow approximately **45 MB for weights** and **1.26 GB for the dataset archive**,
plus roughly another 1.3 GB for extracted images and space for the environment.
Nothing is downloaded by importing the module or running offline checks.
The dataset and weights stay under the external cache, outside Git.

The [ImageNetV2 authors](https://github.com/modestyachts/ImageNetV2#license) explain
that their MIT license covers the code, **not the image data**. Images originate
from Flickr, which provides image-specific licensing information; the authors
refer users to the same usage conditions as the original ImageNet dataset. Check
those conditions for your use. The hosting site's MIT metadata does not license
all images under MIT. No images are included in the saved review artifacts.

Download the [matched-frequency archive](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
from a pinned hosting revision and verify its published SHA-256:

```bash
mkdir -p "$EXAMPLE_CACHE/data"
curl --fail --location --retry 3 \
  --output "$EXAMPLE_CACHE/data/imagenetv2-matched-frequency.tar.gz" \
  https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/d626240be2538720e83103a0e1178d24aca8b12c/imagenetv2-matched-frequency.tar.gz
"$EXAMPLE_PY" - "$EXAMPLE_CACHE/data/imagenetv2-matched-frequency.tar.gz" <<'PY'
import hashlib
from pathlib import Path
import sys
import tarfile
archive = Path(sys.argv[1])
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == 'f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c'
with tarfile.open(archive) as bundle:
    bundle.extractall(archive.parent, filter='data')
PY
"$EXAMPLE_PY" - <<'PY'
from torchvision.models import ResNet18_Weights
ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
PY
```

If you already have an extracted dataset, pass its directory with `--dataset`
to the command-line experiment and skip the archive download. For the main
notebook, export `IMAGENETV2_DIR="/path/to/imagenetv2-matched-frequency-format-val"`
before starting Jupyter. `--archive` is optional; point it at an existing archive
if available, or omit it and retain the saved per-image hashes as provenance.

To reuse weights stored elsewhere, export `TORCH_HOME="/path/to/torch-cache"`
before running setup or starting Jupyter. Set it to the cache root containing
`hub/checkpoints/resnet18-f37072fd.pth`, not to the checkpoint file itself. The
reusable commands preserve an existing `TORCH_HOME`, and all three notebooks
respect it. Cached weights are reused; a missing cache raises an error unless
the command-line experiment explicitly receives `--allow-weight-download`.

## Run the comparison

The quick comparison requires the complete release (1,000 numeric class folders,
ten images per folder), then selects one image per class. Candidate paths are
sorted before sampling with seed 2026. Folder `10` means class **10**, regardless
of alphabetical order. No missing or unreadable selected images are skipped.

```bash
"$EXAMPLE_PY" -B -m examples.model_compression.experiment \
  --dataset "$EXAMPLE_CACHE/data/imagenetv2-matched-frequency-format-val" \
  --archive "$EXAMPLE_CACHE/data/imagenetv2-matched-frequency.tar.gz" \
  --output examples/model_compression/results/cpu_1000 \
  --device cpu --threads 4 --batch-size 16 --workers 0 \
  --seed 2026 --selection-seed 2026 --timing-seed 2027 --warmup 20 --repeats 100
```

The saved directory is already populated in this checkout. To repeat the measured
selection, choose a **new output directory** and add:

```bash
--manifest examples/model_compression/results/cpu_1000/selected_examples.json
```

For all 10,000 images, add `--full`, omit the quick manifest, and choose another
output directory. A full manifest can subsequently be replayed with `--full`.
Manifest replay checks image paths, integer labels, content hashes, uniqueness,
and the requested number of images per class. Previous result directories are
never overwritten.

Both compressed copies start from the untouched dense baseline. The selected
layers are `layer4.0.conv2`, `layer4.1.conv2` (Tucker with spatial ranks fixed), and
`fc` (BlockTT, input modes `(8,8,8)`, output modes `(10,10,10)`). The public
`compress_model` API uses factorized execution. The module verifies the baseline's
parameter and buffer hashes after conversion and evaluation.

`rank=0.75` and `rank=0.50` request approximate fractions of each selected **weight's**
parameter count. Biases, unchanged layers, and integer ranks mean whole-model
savings differ from these targets. Storage is taken from the compression reports;
it describes registered parameter/buffer payloads, not peak inference memory or
checkpoint file sizes.

Accuracy uses total correct predictions divided by total examples, including the
last short batch. Percentage-point changes refer to the measured ImageNetV2
baseline on the same examples, not published ImageNet accuracy.

Timing uses float32 inputs shaped `(1,3,224,224)`, 20 warm-ups, and 100 completed
forward passes. It excludes loading, conversion, preprocessing, and transfers.
Models and inputs are placed on the chosen device first. CUDA timing synchronizes
before and after each timed forward pass. Results are milliseconds **per batch**,
with median and 25th/75th percentiles. Accuracy batch size is separately fixed at
16 for every variant. CPU thread counts and backend settings are recorded.

FLOP totals are unavailable: no counter with audited coverage of the factorized
`einsum`/`tensordot` operations was used. Coverage, unsupported-operation list, and
multiply-add convention are recorded as unassessed/unavailable; no partial total
or published baseline estimate is presented as a complete measurement.

## Saved CPU comparison

These are actual measurements on the same 1,000-image manifest, using an Intel
Core Ultra 9 285H with four PyTorch CPU threads. Latency is milliseconds per batch
of one; brackets are the 25th and 75th percentiles.

| Variant | Top-1 % | Top-5 % | Top-1 change (pp) | Top-5 change (pp) | Parameters | Parameter MiB | Median latency [p25, p75] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 56.80 | 80.30 | 0.00 | 0.00 | 11,689,512 | 44.59 | 10.91 [10.73, 11.18] |
| Target 0.75 | 51.60 | 76.10 | -5.20 | -4.20 | 10,376,276 | 39.58 | 25.57 [25.38, 25.77] |
| Target 0.50 | 41.70 | 69.60 | -15.10 | -10.70 | 9,083,476 | 34.65 | 23.81 [23.66, 23.99] |

Whole-model parameter storage fell by 11.2% and 22.3%, respectively. Top-1 accuracy
fell from 56.8% to 51.6% and 41.7% on these same 1,000 images: changes of -5.2 and
-15.1 percentage points. These are initial compression results without fine-tuning.

**The timing magnitude is uncertain.** An earlier comparison had a dense median
of 20.71 ms, versus 10.91 ms in the table, while the compressed medians barely
changed. Its raw measurements and configuration remain in `validation_runs.json`.
The CSV line-ending correction happened after measurement and does not explain
this difference. Its cause is unknown. The follow-up below is separate evidence;
it does not replace either original run or establish a stable slowdown ratio.

## Bounded timing follow-up

We ran three fresh processes, with the first-three-variant order rotated in each.
Both compressed models were constructed from the untouched pretrained dense model
**before any timing block**, and all three models remained in memory. Each round
finished with an additional dense measurement. No images were read and no accuracy
evaluation ran. Each block used the existing `measure_latency` helper: CPU float32,
four PyTorch threads and one inter-op thread, the same seeded `(1,3,224,224)` input,
evaluation/inference mode, 20 warm-ups, and 100 timed forwards. No machine-wide
settings were changed.

All block medians are shown below, in milliseconds per batch of one. Every block's
raw samples, quartiles, order, and actual thread/backend settings are retained.

| Round | Order of first three blocks | Dense | Target 0.75 | Target 0.50 | Dense recheck |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Dense → 0.75 → 0.50 | 17.764 | 27.426 | 25.144 | 21.816 |
| 2 | 0.75 → 0.50 → Dense | 22.113 | 27.724 | 25.506 | 21.929 |
| 3 | 0.50 → Dense → 0.75 | 21.606 | 27.322 | 25.045 | 21.489 |

The dense median changed from 17.764 to 21.816 ms within round 1. Its first block's
interquartile range was [17.584, 22.059] ms, so even those 100 passes had substantial
variation. Other blocks had narrower quartiles; that does not establish stability
between processes. Across all six follow-up dense blocks, medians ranged from
17.764 to 22.113 ms. The 0.75 medians ranged from 27.322 to 27.724 ms and the 0.50
medians from 25.045 to 25.506 ms. All recorded backend settings and actual thread
counts matched. Every baseline parameter/buffer hash remained unchanged; actual
ranks and factor/core shapes agreed with the original compression reports.

This follow-up did **not** reproduce the earlier 10.91 ms dense result or identify
its cause. Rotating order did not reveal a consistent explanation. It also differs
from the original comparison by keeping both compressed copies resident and
omitting accuracy work before timing. CPU frequency and competing system activity
were not measured or controlled; they are not established causes. Factorized
variants took longer in every measured round on this CPU, but these observations
do not establish a general performance rule. Within-block quartiles describe the
spread of 100 passes, not uncertainty across independent runs. No fastest run was
chosen as representative.

From the repository root, the exact three rounds used the existing environment
and cached weights:

```bash
export TORCH_HOME=/home/rafael_magaldi/.cache/tltorch-model-compression/torch
export PYTHONDONTWRITEBYTECODE=1
EXAMPLE_PY=/home/rafael_magaldi/.cache/tltorch-model-compression/venv/bin/python
for round in 1 2 3; do
  "$EXAMPLE_PY" -B -u -m examples.model_compression.timing_followup \
    --round "$round" \
    --output "examples/model_compression/results/timing_followup/round_${round}.json"
done
```

The block above preserves the actual command and paths used for the saved rounds.
To repeat it on another machine, use `EXAMPLE_PY` and `TORCH_HOME` from the portable
setup above and choose a new output directory; the probe refuses to overwrite a
round. `timing_followup.py` records
its own hash and the measurement module's current hash, package versions, Git
revision/dirty state, weights and input hashes, requested settings, and verified
ranks/shapes. Construction and verification are outside each timed block.

## Review artifacts and notebook

`results/cpu_1000/` contains:

- `results.csv`: one row per variant, accuracy counts/percentages/deltas, model and
  replaced-layer storage, latency summaries, and the reason FLOPs are unavailable.
  Percentages and percentage-point changes use two decimal places; latency
  summaries use three. Counts and byte totals remain exact integers.
- `compression_reports.json`: requested settings, actual ranks, per-layer storage,
  weight errors, and aggregate storage from the public API.
- `latency_samples.json`: all 100 measured latencies for each variant.
- `measurements.json`: accuracy counts and full-precision timing measurements.
- `presentation.json`: the separate record of regenerating the CSV from saved
  measurements and reports, including before/after CSV hashes. No accuracy or
  latency measurements were repeated for formatting. The original configuration
  and its source hash still describe the code that performed the original run;
  they were not updated to match later presentation changes.
- `selected_examples.json`: relative paths, integer labels, and SHA-256 image hashes.
- `validation_runs.json`: first-execution configuration and raw measurements,
  retained to expose timing variability and explain the CSV-formatting rerun.
- `configuration.json`: seeds, settings, preprocessing, data/weight provenance,
  hardware, backend settings, package versions, Git revision/dirty state, and the
  experiment module hash. The dirty state is expected while these example files
  remain uncommitted.

`results/timing_followup/round_1.json` through `round_3.json` contain the separate
follow-up evidence. They do not revise the original comparison's provenance.

To regenerate only a CSV after a presentation change (no images or weights):

```bash
"$EXAMPLE_PY" -B - <<'PYCSV'
import json
from pathlib import Path
from examples.model_compression import experiment
saved = Path('examples/model_compression/results/cpu_1000')
measurements = json.loads((saved / 'measurements.json').read_text())
reports = json.loads((saved / 'compression_reports.json').read_text())
experiment.write_csv(saved / 'results.csv', experiment.comparison_rows(measurements, reports))
PYCSV
```

## Run the teaching notebook

Opening the executed `model_compression.ipynb` lets a colleague read its explanations,
tables, and plot without running anything. **Executing its cells runs the real
experiment**: it loads cached weights and images, measures the dense baseline,
makes two visible `compress_model` calls, evaluates the compressed copies, plots
the new measurements, and saves them. There is no report-only mode.

The displayed layer dictionaries control the two conversions. Both calls start
from the original dense model. Loading, accuracy, and timing reuse `experiment.py`;
the notebook assembles its own rows and configuration so its variant names and
actual selected settings are saved consistently.

The delivered configuration replays `results/cpu_1000/selected_examples.json`: the
same 1,000 real images and preprocessing as the earlier comparison. It uses CPU
float32, four PyTorch threads, accuracy batch size 16, zero loader workers, conversion
seed 2026, and timing seed 2027. Timing uses batch size one, 20 warm-ups, and 100
measurements. Cached weights and the extracted dataset are reused; weight downloads
stay disabled. `IMAGENETV2_DIR` can override the dataset path.

New measurements go to `results/notebook_run_001/`, separate from `results/cpu_1000/`
and `results/timing_followup/`. For another execution, edit `output_directory` to
an unused name. The notebook refuses to overwrite an existing run. **Save notebook
configuration edits before executing**: the provenance hash covers the saved code
cells, while saved outputs and Markdown edits do not change that hash.

Run all cells in order from a fresh **TensorLy model compression** kernel. From the
repository root, the following command executes and saves the notebook, including
its plots and output tables:

```bash
EXAMPLE_CACHE="$HOME/.cache/tltorch-model-compression"
EXAMPLE_PY="$EXAMPLE_CACHE/venv/bin/python"
"$EXAMPLE_PY" -m ipykernel install --prefix "$EXAMPLE_CACHE/venv" \
  --name tltorch-model-compression --display-name "TensorLy model compression"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="${TORCH_HOME:-$EXAMPLE_CACHE/torch}"
export JUPYTER_PATH="$EXAMPLE_CACHE/venv/share/jupyter"
export IPYTHONDIR="$EXAMPLE_CACHE/ipython"
export JUPYTER_RUNTIME_DIR="$EXAMPLE_CACHE/jupyter-runtime"
export MPLCONFIGDIR="$EXAMPLE_CACHE/matplotlib"
"$EXAMPLE_PY" -B - <<'PYNOTEBOOK'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path('examples/model_compression/model_compression.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=600, kernel_name='tltorch-model-compression',
               resources={'metadata': {'path': str(path.parent.resolve())}}).execute()
nbformat.write(notebook, path)
PYNOTEBOOK
```

The notebook run saves `results.csv`, `compression_reports.json`, `measurements.json`,
`latency_samples.json`, `selected_examples.json`, `configuration.json`, and
`comparison.png`. The configuration includes the actual layer dictionaries,
variant labels, preprocessing, seeds, package versions (including Matplotlib),
weights and code hashes, and baseline fingerprints before/after execution. It
records per-image hashes without claiming that the dataset archive was rechecked.
All storage totals come from the compression reports. Full-precision measurements
and raw latency samples remain in JSON; only CSV presentation is rounded.

The earlier timing discrepancy remains historical evidence, separate from the
notebook's new measurements. Its cause is unknown. Quartiles within one timing
block do not establish stability across independent runs. Full-dataset accuracy,
CUDA timing, and fine-tuning remain unverified; FLOP totals remain unavailable.

## Executed notebook comparison

All **19 code cells** completed from a fresh kernel, including the two visible
compression calls, evaluation of all three models on the replayed 1,000-image
manifest, timing, plots, and saving. The executed notebook contains its outputs;
`results/notebook_run_001/comparison.png` is the standalone three-panel plot.
The dense baseline's parameter and buffer fingerprints matched before and after.

| Model | Top-1 % | Top-5 % | Parameters | Parameter bytes | Buffer bytes | Median ms [p25, p75] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | 56.80 | 80.30 | 11,689,512 | 46,758,048 | 38,560 | 21.540 [21.432, 21.673] |
| Compressed 1 | 51.60 | 76.10 | 10,376,276 | 41,505,104 | 38,560 | 28.060 [27.898, 28.254] |
| Compressed 2 | 41.70 | 69.60 | 9,083,476 | 36,333,904 | 38,560 | 25.633 [25.490, 25.830] |

Compressed 1 uses the 0.75 target; Compressed 2 uses 0.50. Accuracy counts and the
complete compression reports exactly match the earlier `cpu_1000` comparison,
with the notebook's variant names mapped to the corresponding targets. Thus
storage savings and weight errors also match. Latencies are new measurements,
not replacements for the historical runs. This run used the same Intel Core
Ultra 9 285H, four CPU threads, and float32 settings. It does not resolve the
historical timing discrepancy or establish a portable performance ratio.

## Offline checks

```bash
OMP_NUM_THREADS=1 "$EXAMPLE_PY" -B -m pytest -q -p no:cacheprovider \
  examples/model_compression/tests/test_experiment.py
```

These checks cover numeric labels, reproducible selection/replay, incomplete or
changed data, known top-1/top-5 counts across unequal batches, an unreadable image,
import without torchvision/network use, a tiny conversion/measurement run, and
CSV formatting that leaves the numerical measurements unchanged.
They download nothing. The image-decoding check skips if Pillow is absent. Library
tests under `tltorch/` have no new torchvision, notebook, or dataset requirement.

## Optional opt_einsum comparison

`opt_einsum_comparison.ipynb` runs a separate synthetic-input experiment: does
PyTorch's opt_einsum integration change the BlockTT classifier's latency, and is
that change visible in the complete model? It compares disabled against enabled
with strategy `"auto"`, using the same stored weights and model objects. TensorLy's
separate opt_einsum plugin stays inactive. No accuracy evaluation or download runs.

Use a **new environment**. Adding opt_einsum to the original environment could
automatically enable PyTorch's integration and change subsequent results. The
commands below reproduce this run: Python **3.12.3**, torch **2.5.1+cpu**,
torchvision **0.20.1+cpu**, TensorLy **0.9.0**, NumPy **2.5.2**, SciPy **1.17.1**,
and optional **opt_einsum 3.4.0**. `requirements-opt-einsum.txt` pins all other
packages to the versions inspected in the original environment. Neither the
original notebook environment nor the shared MPO environment is modified.

From the repository root:

```bash
EXAMPLE_CACHE="$HOME/.cache/tltorch-model-compression"
COMPARISON_PY="$EXAMPLE_CACHE/venv-opt-einsum/bin/python"
"$EXAMPLE_CACHE/venv/bin/python" -B -m venv "$EXAMPLE_CACHE/venv-opt-einsum"
"$COMPARISON_PY" -B -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
"$COMPARISON_PY" -B -m pip install -r examples/model_compression/requirements-opt-einsum.txt
"$COMPARISON_PY" -B -m ipykernel install --prefix "$EXAMPLE_CACHE/venv-opt-einsum" \
  --name tltorch-opt-einsum --display-name "TensorLy opt_einsum comparison"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="${TORCH_HOME:-$EXAMPLE_CACHE/torch}"
export JUPYTER_PATH="$EXAMPLE_CACHE/venv-opt-einsum/share/jupyter"
export IPYTHONDIR="$EXAMPLE_CACHE/opt-einsum-comparison/ipython"
export JUPYTER_RUNTIME_DIR="$EXAMPLE_CACHE/opt-einsum-comparison/jupyter-runtime"
export MPLCONFIGDIR="$EXAMPLE_CACHE/opt-einsum-comparison/matplotlib"
"$COMPARISON_PY" -B - <<'PYNOTEBOOK'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path('examples/model_compression/opt_einsum_comparison.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=600, kernel_name='tltorch-opt-einsum',
               resources={'metadata': {'path': str(path.parent.resolve())}}).execute()
nbformat.write(notebook, path)
PYNOTEBOOK
```

The executed notebook retains its tables and two-panel plot. To repeat it, change
`output_directory` to an unused name and **save the notebook before execution**;
`results/opt_einsum_run_001/` already contains the delivered run and cannot be
overwritten. Loading and conversion explicitly disable the integration. Each
classifier's features are captured once, before any timing. Temporary tracing
proves enabled forwards use opt_einsum, and is removed before measurements.

Four rounds run in one fresh kernel with alternating condition order and rotating
model order. Each block uses CPU float32, four threads, one inter-op thread,
evaluation/inference mode, batch one, 20 warm-ups, and 100 measured passes. The
notebook reports every block's median and quartiles. Its summary takes the median
of four **per-round speedup ratios**, not pooled samples or the fastest round.
PyTorch's per-forward path-search cost is included in the enabled timings.

The new result directory contains `results.csv`, all raw samples in
`timing_blocks.json`, `output_agreement.json`, `optimizer_use.json`,
`compression_reports.json`, `configuration.json`, and `comparison.png`.
`validation.json` records the post-execution checks, full working-tree status,
and exact environment setup commands. Configuration records actual backend flags, package versions, hardware, seeds,
code hashes, and unchanged model/feature fingerprints. The correctness tolerance
is fixed at `rtol=1e-4, atol=1e-5`; this checks execution agreement, not task accuracy.

The small offline helper checks require no cached weights or timing thresholds:

```bash
OMP_NUM_THREADS=1 "$COMPARISON_PY" -B -m pytest -q -p no:cacheprovider \
  examples/model_compression/tests/test_opt_einsum_comparison.py \
  examples/model_compression/tests/test_experiment.py
```

The flag-dependent checks skip when opt_einsum is absent. It remains optional for
the example and is not added to library dependencies or ordinary library tests.

In the saved four-round run, auto was slower for both compressed classifiers in
every round: 65–69% longer for target 0.75 and 78–84% for target 0.50. Whole-model
latency increased by 2.1–2.6%. The median of four per-round speedup ratios was
0.599× / 0.551× for the classifiers and about 0.978× for each complete model.
Dense control ratios stayed near one. All output checks passed; the largest
absolute difference was 1.43e-6. Keep the integration disabled for this CPU,
batch-one experiment. Path-search cost is included, but was not isolated from
the cost of executing the chosen path. These measurements do not establish
performance on other hardware or new task accuracy. The executed notebook shows
every round and its quartiles; no prior result was replaced.

## Optional torch.compile comparison

`torch_compile_comparison.ipynb` compares ordinary eager execution with
`torch.compile(model, backend="inductor", mode="default")` for the dense model
and the same two factorized copies. It runs the comparison using synthetic inputs,
with opt_einsum explicitly disabled and TensorLy's plugin inactive. It measures
whole-model execution only, with no dataset download or task-accuracy evaluation.

This run reuses the original environment unchanged: Python 3.12.3, PyTorch
2.5.1+cpu, torchvision 0.20.1+cpu, TensorLy 0.9.0, NumPy 2.5.2, and SciPy 1.17.1.
The available C++ compiler is `/usr/bin/g++`, version 13.3.0. A small Inductor
smoke check confirmed that the installed tools generate and run a CPU kernel;
no additional Python or system packages were needed. The shared MPO environment
and the separate opt_einsum environment remain unchanged.

Use the existing **TensorLy model compression** kernel. From the repository root:

```bash
EXAMPLE_CACHE="$HOME/.cache/tltorch-model-compression"
EXAMPLE_PY="$EXAMPLE_CACHE/venv/bin/python"
mkdir -p "$EXAMPLE_CACHE/torch-compile/ipython" \
  "$EXAMPLE_CACHE/torch-compile/jupyter-runtime" "$EXAMPLE_CACHE/torch-compile/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="${TORCH_HOME:-$EXAMPLE_CACHE/torch}"
export JUPYTER_PATH="$EXAMPLE_CACHE/venv/share/jupyter"
export IPYTHONDIR="$EXAMPLE_CACHE/torch-compile/ipython"
export JUPYTER_RUNTIME_DIR="$EXAMPLE_CACHE/torch-compile/jupyter-runtime"
export MPLCONFIGDIR="$EXAMPLE_CACHE/torch-compile/matplotlib"
"$EXAMPLE_PY" -B - <<'PYNOTEBOOK'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path('examples/model_compression/torch_compile_comparison.ipynb')
notebook = nbformat.read(path, as_version=4)
try:
    NotebookClient(notebook, timeout=900, kernel_name='tltorch-model-compression',
                   resources={'metadata': {'path': str(path.parent.resolve())}}).execute()
finally:
    nbformat.write(notebook, path)
PYNOTEBOOK
```

Before another execution, change `output_directory` to an unused name and save
the notebook. The delivered run uses `results/torch_compile_run_001/`; existing
runs cannot be overwritten. Each execution allocates a private Inductor cache
under `$EXAMPLE_CACHE/torch-compile/`, without clearing shared caches. First-use
latency includes the first forward and compilation, and is not claimed to be a
completely cold compiler measurement. Later models may reuse earlier artifacts.

The notebook records capture graphs and graph breaks separately from evidence of
actual Inductor execution. Detailed diagnostics use installed PyTorch 2.5 APIs,
including private diagnostic counters; compatibility with other versions is not
assumed. No compiler errors are suppressed. Correctness uses three seeded inputs
and fixed `rtol=1e-4, atol=1e-5`. Failed comparisons or unsupported models are
reported without substituted eager timings.

All models are prepared before four paired timing rounds. CPU float32, four
threads, one inter-op thread, evaluation/inference mode, batch one, 20 warm-ups,
and 100 samples per block match the prior CPU setup. Condition order alternates
and model order rotates. Compiler-counter changes invalidate a block for
post-compilation speedup summaries while preserving its samples. Tables show
each round's medians and quartiles; the plot retains all rounds.

Saved files include `setup.json`, `configuration.json`, `compression_reports.json`,
`capture_diagnostics.json`, `first_use.json`, `output_agreement.json`,
`preparation.json`, `timing_blocks.json`, `results.csv`, `comparison.png`, and
`diagnostics/` logs. Environment records include package/compiler versions, actual
settings, revision/status, code hashes, and model/input fingerprints. Cache
binaries remain outside the repository. Compilation changes no stored weights or
ranks; model storage reports do not count compiler code, workspace, or peak memory.

Offline checks exercise reporting, paired scheduling, and detection of compiler
activity using simulated measurements. They do not compile models or download data:

```bash
OMP_NUM_THREADS=1 "$EXAMPLE_PY" -B -m pytest -q -p no:cacheprovider \
  examples/model_compression/tests/test_torch_compile_comparison.py \
  examples/model_compression/tests/test_experiment.py
```

All 11 code cells completed in a fresh kernel. All nine output checks passed,
with maximum differences of 3.34e-6 (dense), 4.29e-6 (target 0.75), and 2.98e-6
(target 0.50). Model/input fingerprints stayed unchanged. The capture diagnostic
reported one graph / zero breaks for dense and 13 graphs / 12 breaks for each
compressed model. Actual Inductor logs also show a recoverable stride-`.tolist()`
fake-tensor failure and ordinary Python execution around factor iteration. Tensor
regions still compiled; no library changes or error suppression were used.

First-use latency was 6.156 s, 23.249 s, and 6.319 s in model order. Later models
could reuse cache artifacts, and additional compilation occurred during dense
verification. These are not cold or total preparation costs. The repeated-call
probe and all 24 timing blocks had no compiler-counter changes.

Dense and target 0.50 improved in all four paired comparisons: median per-round
speedup ratios were 1.107× and 1.056×. Target 0.75 had no reliable benefit, with
ratios from 0.533× to 1.010×. Absolute latencies shifted substantially within the
process, including the dense control. Every compiled compressed model was slower
than compiled dense in the same round. Keep compilation optional; these results
do not justify enabling it universally or claiming stable deployment ratios.

A shared-axis plotting limit was corrected from saved samples after execution;
no benchmarks were repeated. `measurement_code.txt` preserves the original code
matching the run's notebook-code hash. `presentation.json` records the corrected
notebook-code and plot hashes. The executed notebook explains every round and the
remaining timing uncertainty, without replacing any earlier experiment.

`validation.json` records the completed checks and exact execution command.
Because this repository ignores `*.log`, `compiler_logs.zip` also contains all
12 diagnostic logs, verified byte-for-byte against `diagnostics/`. This lossless
review archive was packaged after execution; raw logs retain the compiler's
original text. To package a new run from its saved logs, choose its new directory:

```bash
"$EXAMPLE_PY" -B - <<'PYLOGS'
from pathlib import Path
import zipfile
saved = Path('examples/model_compression/results/torch_compile_run_001')
with zipfile.ZipFile(saved / 'compiler_logs.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((saved / 'diagnostics').glob('*.log')):
        archive.write(path, arcname=str(path.relative_to(saved)))
PYLOGS
```
