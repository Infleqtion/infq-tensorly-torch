# ImageNet model compression

`run_experiment.py` compares a dense pretrained ResNet-18 with one independently
compressed copy. It evaluates both on the same saved manifest, measures forward
latency, verifies a saved compressed model, and uploads the results to one MLflow
run. Normal execution does not train, fine-tune, search, compile, or generate splits.
Preparation is a separate explicit action. The notebooks and optional comparison
workflows remain available alongside this script.

The [experiment plan](Model-Compression-Experiment-Plan.md) describes the research
stages, prepared partitions, implemented measurements, and remaining work before
the pilot. The wider study has not run.

This directory contains the workflow, its configuration, data, notebooks, tests,
and results. Shared uv files and the compression library stay at the repository
root. Commands below assume that root unless stated otherwise.

```text
experiments/imagenet-compression/
  Model-Compression-Experiment-Plan.md
  run_experiment.py                # preparation, validation, and tracked execution
  configs/resnet18_exploration.yaml
  data/raw/imagenetv2/             # ignored original images
  data/processed/imagenetv2/       # ignored manifests and preparation metadata
  results/                        # retained historical records
    runs/                         # ignored new timestamp/UUID runs
  splits/imagenetv2_20261019/      # retained original split audit and manifests
  tests/
  model_compression.ipynb
  opt_einsum_comparison.ipynb       # separate optional benchmark
  torch_compile_comparison.ipynb   # historical PyTorch 2.5.1 CPU benchmark
  timing_followup.py
  HISTORICAL-RUNS.md
```

## Environment and local CPU/CUDA setup

Use the existing project `.venv`. To synchronize from the lockfile while retaining
the notebook and plotting tools:

```bash
uv sync --python 3.12 --locked --inexact \
  --group experiments --group notebooks --group opt-einsum --group plot-styles
uv run --no-sync jupyter lab
```

Only `dev` is enabled by default. Include each optional group you want to manage
when syncing; `--inexact` also retains other installed tools. `plot-styles` uses
the locked revision of `https://github.com/Infleqtion/matplotlib-styles` and needs
Git read access. Without it, the main notebook uses ordinary Matplotlib. The
editable TensorLy-Torch install points at this checkout. These groups do not
change the published library requirements.

The tested stack is Python 3.12.3, uv 0.12.8, PyTorch `2.7.1+cu128`, TorchVision
`0.22.1+cu128`, TensorLy 0.9.0, CUDA runtime 12.8, and cuDNN 9.7.1. NumPy 2.5.2,
SciPy 1.17.1, and Matplotlib 3.10.8 remain pinned. The CUDA wheels use the explicit
`pytorch` index at `https://download.pytorch.org/whl/cu128`. The matching release
pair and Blackwell support are documented in the [PyTorch versions](https://pytorch.org/get-started/previous-versions/)
and [2.7 release notes](https://pytorch.org/blog/pytorch-2-7/).

Both CPU and CUDA execution use this same environment. The local RTX PRO 2000
Blackwell laptop GPU has compute capability 12.0, driver 595.99.02, and kernel
7.0.0-38-generic. NVIDIA lists Linux driver 570.26 for CUDA 12.8 GA; the driver
must also support the specific GPU. See [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).
The environment installs its CUDA runtime libraries; these eager checks need no
system CUDA toolkit. `nvidia-smi` describes driver capability, while
`torch.version.cuda` identifies the PyTorch runtime.

```bash
uv run --no-sync python -c "import torch, tltorch; print(tltorch.__file__); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run --no-sync python -c "import torch; x = torch.ones(2, 2, device='cuda:0'); y = x @ x; torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), y)"
```

In VSCode use **Python: Select Interpreter**, then choose this checkout's
`.venv/bin/python`. Select the same environment in the notebook kernel picker.
An optional named Jupyter kernel can be registered inside this environment:

```bash
uv run --no-sync python -m ipykernel install --sys-prefix \
  --name tltorch-uv --display-name "TensorLy-Torch (uv)"
```

For a separate CPU-only installation, change the named `pytorch` index URL to
`https://download.pytorch.org/whl/cpu`, run `uv lock`, and sync the desired groups.
That produces a different lockfile and installation. It is unnecessary for CPU
experiments in the tested CUDA environment; the CPU-only alternative has not been
installed during this validation. Cheetah still needs its own compatibility check.

## Prepare once, then run a configuration

Preparation is an explicit, model-free action:

```bash
uv run --no-sync python experiments/imagenet-compression/run_experiment.py \
  --prepare-data \
  --source-dir "$HOME/.cache/tltorch-model-compression/data/imagenetv2-matched-frequency-format-val"
```

The existing raw images and four prepared files were moved here unchanged; do
not rerun this command to migrate them. Preparation refuses any existing processed
destination. For a new preparation output, supply a new `--processed-dir`, with
`--raw-dir` if needed. See [data/README.md](data/README.md) for safeguards, selection
purposes, and the distinction between recorded exposure and undocumented use.
Preparation only checks and copies files and hashes; it does not evaluate images
or create an MLflow run. Normal execution only consumes the supplied manifest.

One YAML describes one dense-versus-compressed comparison. Reusing it creates a
new MLflow run and a new local timestamp/UUID directory on every execution.
The original YAML and effective configuration are saved for each run.

The example [resnet18_exploration.yaml](configs/resnet18_exploration.yaml) selects
`layer4.0.conv2` and `layer4.1.conv2` for Tucker conversion with fixed spatial
ranks, and `fc` for BlockTT conversion. Each requests a floating-point storage
target of `0.75`; actual ranks and storage come from the compression report.
An integer YAML rank such as `2` means a rank size, while `2.0` is a storage
target. These familiar settings test the workflow; they are not an optimal scheme.

Install the optional experiment dependencies with the existing uv setup:

```bash
uv sync --locked --inexact --group experiments
uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration.yaml --dry-run
```

`--inexact` retains optional packages already installed for notebooks. The group
adds the MLflow client and YAML parser and explicitly includes the existing
TorchVision/Pillow versions. Plotting styles are optional and unused here.
The current lock uses Python 3.12, PyTorch 2.7.1 and TorchVision 0.22.1 with
CUDA 12.8, and TensorLy 0.9.0. This one environment supports both CPU and GPU
execution. In your configuration, set `runtime.device: cpu` for CPU execution or
`runtime.device: cuda:0` for the first NVIDIA GPU, then run the same command below.
No package reinstall is needed when changing that setting. Unavailable CUDA
raises an error; the runner never substitutes CPU.

Small offline checks passed on the local RTX PRO 2000 Blackwell laptop GPU with
driver 595.99.02, including completed GPU operations, conversion, the recording
MLflow client, and checkpoint loading. The setup instructions above include synchronization and CUDA checks.
Before execution on Cheetah, still check its GPU/driver, package versions, MLflow
server/client compatibility, and artifact access. Local checks do not establish
Cheetah compatibility or real-data accuracy and performance.

Relative paths **inside YAML** resolve from this experiment directory, derived
from the runner's location. Absolute data/output paths are also accepted. The
CLI's `--config` filename instead follows normal shell conventions: a relative
filename resolves from the caller's working directory. From another directory,
use an absolute runner path and either an absolute configuration path or a path
relative to that working directory. Changing the shell directory does not change
the meaning of `data/raw/imagenetv2` inside the YAML.

New output locations inside the repository must be ignored by Git;
`results/runs/` under this experiment is the default. Each run gets a new timestamp/UUID directory; existing directories
are never overwritten. A dirty checkout is allowed and does not trigger commits.

The dry run checks safe YAML syntax, duplicate/unknown options, supported runner
settings, and path resolution. It prints the effective configuration. It does
not import execution modules, load models/weights, inspect image contents, create
outputs, or contact MLflow. File availability and hashes, layer compatibility,
CUDA availability, artifact access, and serialization are checked during execution.

After local review and the separate Cheetah setup check, the execution command is:

```bash
uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration.yaml
```

The required `tracking.uri` in YAML supplies the HTTP(S) server address; the
example uses `http://127.0.0.1:9361` on Cheetah. No `MLFLOW_TRACKING_URI` export is
needed, and an existing value cannot override the configuration. URI validation
rejects embedded credentials, whitespace, malformed ports, query parameters, and
fragments before any tracking request. The runner calls `mlflow.set_tracking_uri`,
`mlflow.set_workspace`, and `mlflow.set_experiment` in that order, then uses the
returned experiment ID in one `with mlflow.start_run(...)` block. The required
workspace is `quirc` in the example. Access errors stop execution without another
workspace or local tracking fallback. `MLFLOW_RUN_ID` must be unset so MLflow
cannot silently resume an earlier run. The workspace is saved in both configurations, `run.json`,
and the `tracking.workspace` run tag, and printed with the experiment/run IDs.
Credentials belong in MLflow-supported
environment variables or external credential configuration, never in YAML or the
address. The runner does not print environment credentials or archive credential
files. It requires a server address and never falls back to a local tracking store.
The server must accept a new run and a small configuration artifact before model
loading or compression starts. The runner prints the experiment ID, run ID, and
local output directory at startup; both IDs are also saved in `run.json`.
Configuration and source-provenance files are uploaded before model loading.
The manifest and environment/input provenance follow before compression, so a
failed conversion still has useful context on the server.

MLflow parameters make the requested settings directly comparable: `compression.layers`
lists selected names, and `compression.<layer>.<option>` records each factorization,
rank, tensorization, fixed-rank mode, and parameter cap when supplied. Explicit
decomposition controls use `compression.<layer>.decomposition_kwargs.<control>`.
Integer rank `1` is recorded as `1`, while floating-point target `1.0` stays `1.0`;
lists use deterministic JSON. The runner does not add decomposition defaults to
these requested parameters. Actual ranks, effective decomposition settings,
storage, and reconstruction errors remain in `compression_report.json`.

Evaluation settings use `evaluation.batch_size` and `evaluation.workers`; thread
count is `runtime.threads`. Timing settings use `timing.input_shape`,
`timing.warmup`, and `timing.repeats`. Existing model, weights, device, dtype,
implementation, and seed parameters remain available. Tags `git.branch`,
`git.commit`, and `git.dirty` describe the actual checkout; `git.dirty` is `true`
or `false`, and does not prevent a run.

Parameters, metrics, and artifacts use MLflow's fluent logging functions, which
write to the active run. Uploads remain synchronous. The run context marks success
only after model verification and all required successful-run uploads complete. On
failure, the runner retains `failure.json` with the original error and traceback,
attempts to upload it, and independently attempts to mark the run failed. Neither
tracking error hides the original exception or removes local evidence, including
an error raised while closing the run context. Once that context has closed,
diagnostics explicitly target the saved run ID; the status retry also targets
that ID rather than opening another run. Failure handling uploads only the
diagnostic, never a partly written or unverified model.

Prepare inputs using [data/README.md](data/README.md), and use the
[explicit weight-cache setup](HISTORICAL-RUNS.md#explicit-downloads-and-image-use-guidance).
`TORCH_HOME` may point to an existing cache containing
`hub/checkpoints/resnet18-f37072fd.pth`. Execution loads that file directly and
does not download missing weights. It validates every selected image's path,
numeric folder label, and SHA-256. Missing, mismatched, or unreadable images fail
the run rather than being skipped. The supplied manifest controls selection and
order, with no one-or-ten-images-per-class restriction. Its seed is recorded as
input provenance; selection is never repeated. Confirmation and final partitions
must still be used according to their intended purposes.

Preprocessing is the transform associated with
`ResNet18_Weights.IMAGENET1K_V1`. Runtime seed `2026` seeds Python, NumPy, and
PyTorch; timing input uses a separate local generator with seed `2027`. Both
models execute eagerly in float32 with opt_einsum disabled. The runner checks
the baseline's parameters and buffers before and after conversion and evaluation.

Accuracy is reported as top-1/top-5 percentages and changes in percentage points.
Cross-entropy is summed over images and divided by the total, so a smaller final
batch is weighted correctly. Macro precision, recall, and F1 are the unweighted
means across all output classes, including classes absent from the manifest.
Undefined class metrics contribute zero. These three metrics range from 0 to 1.

Timing uses a preallocated input on the selected device, evaluation/inference
mode, 20 warm-ups, and 100 measurements of batches of one. CUDA is synchronized
around each measurement. Median and inclusive quartiles are milliseconds per
batch; raw samples are retained. Loading, image transforms, transfers, and
conversion are excluded; conversion time is recorded separately. Storage is
taken directly from the public compression report, including biases, buffers,
frozen parameters, and shared-object counting. FLOPs remain unavailable (`null`,
not zero): coverage of factorized contractions has not been validated.

Each output directory contains original/effective configuration, the exact
manifest, a two-row CSV, a readable report, detailed measurements, raw timings,
compression/storage/error reports, and provenance. Provenance includes manifest
and cached-weight hashes, preprocessing, seeds, versions, hardware, backends,
Git branch/commit/status, and a source archive. That archive uses an explicit
allowlist of current runner/preparation/library code and environment files,
including uncommitted contents and hashes. It excludes datasets, environments,
historical runs, and unrelated personal files. It identifies the code beyond
the commit hash; it is not an archive of the entire checkout.

`original_config_file_sha256` and `effective_config_file_sha256` identify the exact
saved configuration files. They are recorded as MLflow parameters and in `run.json`
and `provenance.json`, alongside manifest/source provenance. The effective file
contains resolved absolute paths: its hash is **not** a machine-independent ID
for equivalent experiments. Use the individual settings to compare experiments;
use these hashes to identify exact artifacts. `manifest_sha256` and
`source_archive_sha256` are also available as parameters.

`compressed_model.pt` contains the actual model architecture, parameters, and
buffers, not instructions to decompose again. PyTorch's full-model serialization
was checked with Tucker convolutions and BlockTT linear layers. The runner loads
an independent copy, compares every parameter and buffer exactly (including
nonpersistent buffers), checks factorization types/ranks, and compares outputs
on a fixed input with `rtol=1e-5`, `atol=1e-6`. It records the result in
`model_verification.json` and saves the input/expected output in
`verification_input.pt`.

Load a trusted artifact in the recorded Python/package environment with this
repository code available (or the matching `source.zip` contents over its recorded
commit):

```python
from pathlib import Path
import sys

# Use the absolute experiment path if running outside the repository root.
experiment_directory = Path("experiments/imagenet-compression").resolve()
sys.path.insert(0, str(experiment_directory))
from run_experiment import load_compressed_model

model = load_compressed_model("/path/to/run/compressed_model.pt", device="cpu")
```

Loading calls neither pretrained-weight loading nor decomposition. The checkpoint
uses pickle with `weights_only=False`; load only trusted artifacts. It depends on
matching PyTorch, TorchVision, TensorLy, and TensorLy-Torch code, rather than being
a portable interchange format. Downloading it from the real MLflow server and
verifying it in Cheetah's environment remain part of the later Cheetah test.
Locally, the same tiny Tucker/BlockTT checkpoint was loaded on both CPU and CUDA:
parameters, buffers (including nonpersistent buffers), types, and ranks were
preserved exactly. Outputs agreed with `rtol=1e-5`, `atol=1e-6`; float32
accumulation order can differ between devices, even with identical saved weights.
Use `device="cuda:0"` to load onto CUDA. These checks use the existing explicit
`weights_only=False` trusted-model path,
which does not depend on PyTorch's changed default. They do not promise general
checkpoint portability across package versions. Source archives and Git provenance remain anchored to the repository root,
independent of the caller's working directory. The archive includes the consolidated
runner, example configuration, documentation, shared environment files, and library
code; it excludes images, prepared manifests, caches, and result directories.

Run offline validation without pretrained models, downloads, or a tracking server.
CUDA cases run when a GPU is available and skip otherwise; the CPU cases still run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -B -m pytest \
  -q -p no:cacheprovider experiments/imagenet-compression/tests
```


## Manual execution on Cheetah

Use the dedicated checkout at `/home/rafael/projects/infq-tensorly-torch` on
branch `rafael/exp`. The YAML's server address, `http://127.0.0.1:9361`, is local
to Cheetah; running this command on a laptop does not send computation to Cheetah.
Change `tracking.uri` in a configuration intended for another server.

The shared server is managed separately. MLflow automatically reads the existing
`/home/rafael/.mlflow/credentials`; do not copy credentials into the checkout,
YAML, command line, or artifacts. An authenticated workspace-listing request has
succeeded for `quirc`. Creating experiments/runs and uploading/downloading
artifacts still await the first real comparison. Browser login alone does not
verify the Python client's access.

Prepare the dedicated environment once using the existing lockfile:

```bash
cd /home/rafael/projects/infq-tensorly-torch
/home/rafael/.local/bin/uv sync --locked --group experiments
```

This installs the default development group and experiment dependencies without
notebook or plotting groups. Keep the copied raw images under
`experiments/imagenet-compression/data/raw/imagenetv2/` and the unchanged four
prepared files under `data/processed/imagenetv2/` in that experiment directory.
Use `$HOME/.cache/tltorch-model-compression/torch` for the matching cached weight
file. Transfer and hash-check these existing inputs; do not regenerate selections.
`TORCH_HOME` is PyTorch's weight-cache setting and remains separate from MLflow;
its behavior and the export below are unchanged.

For a readiness check, run only:

```bash
cd /home/rafael/projects/infq-tensorly-torch
export TORCH_HOME="$HOME/.cache/tltorch-model-compression/torch"
/home/rafael/.local/bin/uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration.yaml --dry-run
```

After checking available GPU capacity, manually start the comparison by removing
`--dry-run` from that last command. The YAML selects `cuda:0`, evaluation batches
of 16, four CPU threads, and the original 1,000 exploration images. Compression
and timing settings are unchanged. Cheetah has two RTX 6000 Ada GPUs; device
visibility can change their logical numbering, so verify the physical GPU before
execution and do not change the chosen device silently.

Each execution prints its workspace, experiment ID, run ID, and output directory.
Find the run in workspace `quirc`, experiment `model-compression`, with run name
`resnet18-exploration-tucker-blocktt-075`. Local files appear under
`experiments/imagenet-compression/results/runs/<UTC timestamp>-<unique suffix>/`;
`run.json` links that directory to its tracking IDs. Readiness checks do not
establish artifact permissions, checkpoint download behavior, or scientific
results. The first real run will test creation/uploads; download verification
remains a separate check after a verified artifact exists.


## Notebooks and historical evidence

The three notebooks import the shared functions from `run_experiment.py` by
adding this directory to `sys.path`; the hyphenated directory is not a Python
package import. Open Jupyter from the repository root or this directory. The
opt_einsum and torch.compile helpers retain their separate benchmark sequences;
neither is part of normal configured execution. `timing_followup.py` also remains
separate and uses synthetic inputs. No benchmark was rerun during the move.

`torch_compile_comparison.ipynb` is a historical benchmark tied to its recorded
PyTorch `2.5.1+cpu` environment. Its exact-version assertion remains in place;
it has not been validated for execution under the current PyTorch 2.7.1 CUDA
environment. Retaining a notebook here does not mean it is ready to execute
in the shared environment. Consult its recorded setup before attempting a rerun.

Notebook outputs, execution counts, historical configurations, raw timings,
plots, diagnostics, and split audits retain their earlier contents. The notebooks'
source imports now use this layout, but their outputs were not regenerated.
Before rerunning a notebook or follow-up, explicitly choose an unused destination
under `results/runs/`. Existing-directory checks protect the historical folders.
The main notebook still accepts `IMAGENETV2_DIR`; point it at this experiment's
`data/raw/imagenetv2` to reuse the moved images, or at the existing external cache.
The configured runner uses prepared data by default and does not use this override.

[HISTORICAL-RUNS.md](HISTORICAL-RUNS.md) preserves the original example README
verbatim, including original commands, paths, environment versions, measurements,
and download checksums. Its old entry points describe past runs; use this README
for current commands. The original audit in `splits/imagenetv2_20261019/` is also
preserved. Recorded paths in those files may refer to the old layout. They are
provenance, not current path defaults. This refactor makes no new accuracy or
performance claim.
