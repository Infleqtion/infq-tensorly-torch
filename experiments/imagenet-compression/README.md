# ImageNet model compression

`run_experiment.py` evaluates exactly one configured ResNet-18 model per execution:
the dense pretrained model, or a compressed copy initialized from those weights.
It evaluates and times that model once, verifies its saved checkpoint and MLflow
package, and uploads the results to one run. Compare separate dense and compressed
runs in the same MLflow experiment using matching manifests and execution settings.
Normal execution does not train, fine-tune, search, compile, or generate splits.
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
  configs/resnet18_exploration.yaml       # Tucker/BlockTT
  configs/resnet18_exploration_tt.yaml    # TT/BlockTT
  configs/resnet18_exploration_dense.yaml # dense reference
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

One YAML describes one model evaluation. Reusing it creates a new MLflow run and
a new local timestamp/UUID directory on every execution. The original YAML and
effective configuration are saved for each run.

Omit `compression`, or set `compression: null`, to evaluate the dense pretrained
model. Both become `compression: null` in the effective configuration. A supplied
mapping must contain a valid implementation and nonempty layer selection; an
empty or malformed mapping is an error, not a request for dense execution.
[resnet18_exploration_dense.yaml](configs/resnet18_exploration_dense.yaml) uses the
same scientific and runtime settings as the compressed example, omits compression,
and uses run name `resnet18-exploration-dense`.

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
are never overwritten. Real execution snapshots its selected code and YAML in a
local commit before tracking; unrelated work may leave the checkout dirty.

The dry run checks safe YAML syntax, duplicate/unknown options, supported runner
settings, and path resolution. It prints the effective configuration. It does
not import execution modules, load models/weights, inspect image contents, create
outputs, or contact MLflow. File availability and hashes, layer compatibility,
CUDA availability, artifact access, and serialization are checked during execution.

After review and readiness checks, launch each desired configuration separately:

```bash
# Dense reference; this execution never converts a layer.
uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration_dense.yaml
# Tucker/BlockTT; evaluates only the converted model.
uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration.yaml
# TT/BlockTT; also evaluates only the converted model.
uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration_tt.yaml
```

These commands are independent. The runner never launches a dense reference,
looks up another run, or calculates cross-run accuracy differences.

Two optional CLI interfaces support a future external controller; neither adds
Optuna or changes the configured model, measurements, or tracking fields:

```bash
uv run --no-sync python experiments/imagenet-compression/run_experiment.py \
  --config experiments/imagenet-compression/configs/resnet18_exploration.yaml \
  --status-file /absolute/existing-directory/runner_status.json \
  --provenance-config experiments/imagenet-compression/configs/studies/example.yaml
```

The paths above illustrate the interface; supply an existing extra YAML and a
new receipt path before executing. `--status-file` requires an absolute filename
in an existing writable directory. Keep receipts outside source, for example in
an external controller directory. An existing destination, including a dangling
symlink, is rejected before snapshots or tracking. Complete JSON updates are
atomic; updates detect replacement or editing observed before publication. The
ownership check and replacement are not an atomic compare-and-swap. A supervisor
must reserve a unique attempt directory and allow only the runner to write its
receipt. This is a local receipt, not an additional MLflow artifact or tracking run.

Receipt schema version 1 records `started`, `running`, `succeeded`, or `failed`,
the current phase, selected configuration path and SHA-256, output directory,
PID/parent PID/hostname/start time, tracking URI/workspace, and experiment/run/model
IDs and model URI as available. Unknown identifiers are `null`. Failures record
exception and cause types without copying error messages or environment contents.
Argument and configuration validation precede receipt creation; validation errors
can therefore return nonzero without a receipt. Existing `failure.json` diagnostics
and tracebacks remain available when the runner creates them; a secondary receipt
error does not replace the original error.

Only after the runner returns through successful MLflow context exit does the CLI
write `succeeded`. It references measurements, results, run metadata, configurations,
source/input provenance, and verification files by path and SHA-256. A CSV or
initialized Model is not evidence of completion. A supervising process should
require both exit code zero and a matching succeeded receipt. If publishing that
final receipt fails, the CLI returns nonzero and preserves the already finished
run and its files. Abrupt termination may leave a started/running receipt; this
interface does not supervise timeouts or install signal handlers.

Repeat `--provenance-config` for a study definition and base template. These YAMLs
must be regular non-symlink files under this experiment's `configs/`, including
nested directories, and must not be Git-ignored. CLI filenames follow the caller's
working directory. Only explicitly named files join the existing snapshot,
committed-byte checks, source archive, and subsequent input verification. Repeated
references are deduplicated; naming the selected configuration again adds nothing.
Raw copies upload before model loading under `provenance_configs/` with their full
repository-relative paths, so equal basenames remain distinct. The source record
lists `provenance_configuration_paths` when extras are present. Unrelated YAMLs,
staging and working changes retain their existing treatment.

Both flags work with `--dry-run`: it validates destinations and extra YAML paths
without creating a receipt or outputs, committing, importing execution dependencies,
contacting tracking, or checking/downloading weights and images. Without either
flag, manual execution and the original snapshot scope are unchanged.

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

The **Runs** view shows execution settings as parameters: `device`, `dtype`,
`evaluation.batch_size`, `runtime.threads`, `seed`, `timing.input_shape`,
`timing.repeats`, and `timing.warmup`. Their names and encoding are unchanged.

The **Models** view shows one LoggedModel named `model` for either execution type.
Its model parameters always include `model` and `weights`. Compressed models also
record `implementation`, `compression.layers`, and every supplied
`compression.<layer>.<option>` field: rank, factorization, tensorization, fixed-rank
modes, and parameter caps. Decomposition controls use
`compression.<layer>.decomposition_kwargs.<control>`. The model-only parameter
`compression.weight_fraction` records a common floating-point target when all
selected layers use the same factorization and target. Mixed methods, differing
targets, integer/list ranks, and `same` requests omit it. This describes selected
weights, not achieved whole-model storage. Dense models record `1.0` as the
unconverted baseline's plotting coordinate; they receive no per-layer conversion
settings. Model settings are not duplicated as run parameters.
Integer rank `1` stays `1`, floating-point target `1.0` stays `1.0`, and lists use
deterministic JSON. Actual ranks, effective fitting settings, storage, and weight
reconstruction errors remain in the compression report when conversion occurs.

Both execution types log the same eight unprefixed metrics on the run and associate
them with that run's model ID:

| Measurement | Metric names |
| --- | --- |
| Accuracy and loss | `top1_percent`, `top5_percent`, `cross_entropy_mean` |
| Parameter storage | `parameter_count`, `parameter_bytes` |
| Latency | `latency_median_ms`, `latency_p25_ms`, `latency_p75_ms` |

Compressed runs additionally log `conversion_seconds` on the run only. Dense runs
omit it because no conversion was measured. Future results have no accuracy-delta
fields or automatically evaluated baseline. Compare models through their common
metric names in MLflow, keeping the evaluation partition and runtime settings fixed.
Detailed image counts, macro metrics, buffer bytes, and raw timing samples remain
in artifacts; replaced-layer diagnostics remain in compression reports. Metric
values retain full precision; CSV percentages use two decimals and latencies three.

`evaluation.workers` and `timing_seed` remain in configuration/provenance artifacts,
not searchable parameters. Tags `git.branch` and `git.commit` identify the recorded
checkout, and `tracking.workspace` identifies its workspace. Execution inputs are
committed and verified; repository-wide working-tree status remains in provenance
rather than a dashboard tag. No replacement hash tags are added.

With MLflow 3.17.2, ordinary model logging copies existing run parameters onto a
new model. The runner instead initializes one LoggedModel with its model settings
and passes that same ID to PyTorch model logging. Run parameters can therefore
remain available early for failed runs without leaking onto the model. Only one
run and one model are created; no additional baseline model or Registry entry is
added. The complete configuration is still uploaded before model loading, so
failed conversions retain all requested settings even before a Model exists.

MLflow also infers metric associations and copies existing step-zero metrics when
logging a model. The runner logs metrics after model packaging, clears the implicit
active-model default, and uses a synchronous client batch with explicit model IDs
on all eight absolute metrics. Conversion time has no model ID. Passing
`model_id=None` to fluent metric logging alone would not prevent unwanted
associations. The tested behavior uses
MLflow 3.17.2's LoggedModel initialization, existing-model logging, and metric-ID
APIs; the tracking server must support them. An isolated temporary file-store test
checks real SDK behavior without a server or added dependencies. The file-store
opt-in is confined to that test; configured execution still requires HTTP(S).
Older completed runs and models keep their original fields.

Parameters and artifacts use the fluent API, and metrics use the explicit client
batch described above. Uploads remain synchronous. The run context marks success
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
The maintained YAML selects the pretrained-weight cache directly:

```yaml
cache:
  torch_home: ~/.cache/tltorch-model-compression/torch
```

Choose another existing cache by changing `cache.torch_home` in your selected
configuration. This is the parent of `hub/`, so the expected file is
`<torch_home>/hub/checkpoints/resnet18-f37072fd.pth`. `~` expands for the current
user; relative cache paths resolve from this experiment directory, just like
other YAML paths. An explicit YAML cache overrides `TORCH_HOME` and an existing
PyTorch Hub override. With no `cache` section, PyTorch's existing resolution is
preserved, including `TORCH_HOME` or a caller's `torch.hub.set_dir()` setting.
Only real execution sets the Hub directory, before weight preflight and loading.
Imports, configuration loading, preparation, help, and dry-run do not change it.
The effective configuration saves the resolved explicit `torch_home`; provenance
records the actual `cache.hub_dir` and `weights.checkpoint_path` in every run.
Execution loads that file directly, checks its hash, and does not download missing weights. It validates every selected image's path,
numeric folder label, and SHA-256. Missing, mismatched, or unreadable images fail
the run rather than being skipped. The supplied manifest controls selection and
order, with no one-or-ten-images-per-class restriction. Its seed is recorded as
input provenance; selection is never repeated. Confirmation and final partitions
must still be used according to their intended purposes.

Preprocessing is the transform associated with
`ResNet18_Weights.IMAGENET1K_V1`. Runtime seed `2026` seeds Python, NumPy, and
PyTorch; timing input uses a separate local generator with seed `2027`. The selected
model executes eagerly in float32 with opt_einsum disabled. Compressed execution
checks that conversion and measurement preserve the pretrained source's parameters
and buffers, without evaluating or timing that source. Dense execution uses that
source directly and checks its state after measurement.

Accuracy is reported as top-1/top-5 percentages, without cross-run differences.
Cross-entropy is summed over images and divided by the total, so a smaller final
batch is weighted correctly. Macro precision, recall, and F1 are the unweighted
means across all output classes, including classes absent from the manifest.
Undefined class metrics contribute zero. These three metrics range from 0 to 1.

Timing uses a preallocated input on the selected device, evaluation/inference
mode, 20 warm-ups, and 100 measurements of batches of one. CUDA is synchronized
around each measurement. Median and inclusive quartiles are milliseconds per
batch; raw samples are retained. Loading, image transforms, transfers, and
conversion are excluded; conversion time is recorded separately only when it
occurs. Compressed storage comes from the public compression report. Dense storage
reuses the same library accounting helper directly, without a conversion or model
copy. Both count biases, buffers, frozen parameters, and shared tensor objects in
the same way. FLOPs remain unavailable (`null`, not zero): coverage of factorized contractions has not been validated.

Each output directory contains original/effective configuration, the exact
manifest, a one-row `results.csv`, a readable `report.md`, detailed measurements,
raw timings, one verified `model.pt`, and provenance. `measurements.json` contains
one `variant` (`dense` or `compressed`), `accuracy`, `latency`, and `storage` record;
only compressed measurements and CSVs include `conversion_seconds`.
`latency_samples.json` is that model's list of raw millisecond samples.
`compression_report.json` exists only after conversion and retains original versus
factorized weight/storage diagnostics, including replaced-layer totals. These
weight diagnostics do not require inference on the original model. No new result
contains an accuracy delta; historical artifact formats remain unchanged.
Provenance includes manifest and cached-weight hashes, preprocessing, seeds, versions, hardware, backends,
Git branch/commit/status, and a source archive. The archive contains the exact
committed execution inputs described below, with per-file SHA-256 hashes. It
excludes datasets, environments, historical runs, and unrelated personal files.

`original_config_file_sha256` and `effective_config_file_sha256` identify the exact
saved configuration files. They remain in `run.json` and `provenance.json`,
alongside manifest/source provenance, and are not searchable parameters. The effective file
contains resolved absolute paths: its hash is **not** a machine-independent ID
for equivalent experiments. Use the individual settings to compare experiments;
use these hashes to identify exact artifacts. `manifest_sha256` stays in provenance;
`source_archive_sha256` stays in `source.json` and its provenance copy. The Git
commit stays in those records and the existing `git.commit` tag. None of these
hashes or the old `git_commit` field are logged as run parameters or new tags.

`model.pt` contains the actual model architecture, parameters, and buffers, not
instructions to decompose again. The same full-model save/load path supports dense,
Tucker/BlockTT, and TT/BlockTT models. The runner loads an independent copy, compares every parameter and buffer exactly (including
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
from run_experiment import load_model

model = load_model("/path/to/run/model.pt", device="cpu")
```

The previous `load_compressed_model` helper remains available for trusted historical
`compressed_model.pt` files; it delegates to the same loader. Historical files do
not need renaming. Loading calls neither pretrained-weight loading nor decomposition.
The checkpoint uses pickle with `weights_only=False`; load only trusted artifacts. It depends on
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
independent of the caller's working directory. The archive and automatic commit
use the same explicit execution scope.

The runner also calls `mlflow.pytorch.log_model(..., name="model",
serialization_format="pickle")` in the same active run, after checkpoint
verification. MLflow 3.17.2 otherwise defaults to `.pt2` outside Databricks;
the explicit pickle choice retains the actual factorized modules and weights.
This logs a Model package and does not register a Model Registry entry. Its
input example and signature describe the fixed synthetic timing input and model
outputs. Exact installed requirements are recorded for MLflow skinny, pandas,
cloudpickle, PyTorch, TorchVision, TensorLy, NumPy, and SciPy. Only TensorLy-Torch
Python source is bundled under `code/tltorch/`; tests and caches are excluded.
Pandas is an experiment dependency because MLflow's PyTorch flavor imports it.

Packaging happens after accuracy and latency measurements. The measured model
stays on its selected device and precision in evaluation mode. An independent
MLflow reload must preserve parameters, buffers, factor types/ranks, and fixed-input
predictions (`rtol=1e-5`, `atol=1e-6`). The runner records evidence in
`mlflow_model_verification.json`, saves the returned URI and optional model ID in
`run.json` under `mlflow_model`, and prints the URI. Logging or reload failures
fail the run and retain the existing local/remote failure diagnostics. A remote
Model package may remain after a failed verification; the run and its verification
evidence determine whether it passed.

To load a trusted package in its recorded environment, use the URI from `run.json`:

```python
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("http://127.0.0.1:9361")
mlflow.set_workspace("quirc")
model = mlflow.pytorch.load_model(model_uri, map_location="cpu", weights_only=False)
# Use map_location="cuda:0" to load onto the chosen GPU.
```

Loading uses the packaged `tltorch` code and does not repeat decomposition.
Pickle packages require trusted artifacts and matching installed dependencies.
Offline tests exercise MLflow's real save/load code with dense, Tucker/BlockTT,
and TT/BlockTT models, including fresh processes. They do not establish server-side
Model permissions or promise portability across arbitrary package versions.

## Local execution commits

After configuration, input/hash, cached-weight, runtime/device, and packaging-import
checks, normal execution commits changed execution inputs. It does this before
selecting the tracking server/workspace/experiment or creating a run. If the scope
already matches HEAD, it reuses HEAD without an empty commit.

The commit and `source.zip` scope is:

- Trackable top-level Python files in `experiments/imagenet-compression/`, covering
  the runner and its preparation, tracking, and comparison helpers.
- Trackable `tltorch/**/*.py` files, excluding every `tests/` directory.
- The selected original YAML, which must be a trackable `.yaml` or `.yml` under
  `experiments/imagenet-compression/configs/`. Copy an external configuration there
  before real execution. Other configurations are left alone; dry-run still accepts
  an external YAML. This also preserves a separate TT configuration unless selected.
- `setup.py`, `README.rst` (read by setup.py), `pyproject.toml`, `uv.lock`, and
  `.python-version`.

Selection includes relevant paths from HEAD, the index, and trackable working
files, so staged and unstaged deletions and renames are included. The source
archive omits deleted paths and stores the committed renamed contents. Images, manifests,
weights, outputs, notebooks, tests, documentation other than the build README,
credentials, caches, environments, and unrelated files are excluded. Do not put
personal or generated code in the source directories; ignored files are excluded.
The commit message is `experiment: snapshot code and configuration for <run_name>`.
Git's path-limited commit preserves unrelated staged and unstaged changes, including
a staged file with further unstaged edits. It may update staging for files within
the execution scope. Hooks run normally. Resolve missing identity, failing hooks,
index locks, or conflicts manually if Git cannot commit; no tracking run or model
execution starts on that failure. A hook that changes the scoped content or adds
files outside scope causes verification to fail and requires manual review of any
commit already created. The runner never stashes, amends, switches branches,
configures identity, or pushes.

`source.json` records the commit, selected configuration path, source-file hashes,
and `execution_inputs_match_commit`. Its `working_tree_status` records the actual
repository-wide status, also retained in `provenance.json` alongside the environment
record's `working_tree_dirty` flag. Unrelated work can leave the checkout dirty even
when execution inputs match the commit. This status is not logged as an MLflow tag.
The source archive reads the recorded commit, and the saved original YAML must
match its committed bytes.
The exact effective configuration, installed environment, manifests, and weight
hashes remain necessary: a Git commit does not capture those external inputs.

The runner checks scoped contents again before packaging, after package verification,
and after final uploads. Changed or newly added execution inputs fail the run;
`failure.json` records the input error without hiding a concurrent original failure.
`execution_inputs_verification.json` records successful verification. These checks
compare on-disk content at those points; they are not a filesystem lock or proof
against a change that was made and reverted between checks. Avoid editing execution
inputs while a run is active. Commits stay local: push them later or transfer a Git
bundle so another machine can recover the recorded revision.

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
succeeded for `quirc`. This single-model refactor is validated offline; it has not
been deployed or exercised against the shared server. Keep existing completed
runs unchanged, and check client/server and artifact access before new execution.
Browser login alone does not verify the Python client's access.

After this implementation is reviewed and committed, transfer that exact commit
by a normal SSH push/fetch or a Git bundle and fast-forward the clean dedicated
checkout. This preparation step has not deployed it. If Cheetah still cannot fetch
GitHub over SSH, prepare a bundle locally after committing the reviewed changes:

```bash
cd /home/rafael_magaldi/Documents/infq-tensorly-torch-model-compression
git rev-parse rafael/exp  # Record this exact reviewed commit for the remote check.
git bundle create /tmp/imagenet-compression-update.bundle rafael/exp ^c4980d7a93ceab243329c2266359420e1a2f19f8
scp -o ClearAllForwardings=yes /tmp/imagenet-compression-update.bundle cheetah:/tmp/
```

On Cheetah, first confirm `rafael/exp` and a clean working tree. Stop and preserve
any unexpected work or divergence. Then verify the bundle and fast-forward:

```bash
cd /home/rafael/projects/infq-tensorly-torch
git branch --show-current
git status --short --untracked-files=all
git bundle verify /tmp/imagenet-compression-update.bundle
git fetch /tmp/imagenet-compression-update.bundle refs/heads/rafael/exp:refs/remotes/origin/rafael/exp
git merge --ff-only origin/rafael/exp
git rev-parse HEAD  # Must equal the reviewed local commit recorded above.
```

This preserves the GitHub SSH remote and upstream. The bundle requires the last
transferred commit `c4980d7…`, so it is for the existing dedicated checkout.
Keep existing prepared data and cached weights. Synchronize the environment
from the updated lockfile (pandas is required for Model packaging):

```bash
cd /home/rafael/projects/infq-tensorly-torch
/home/rafael/.local/bin/uv sync --locked --group experiments
```

This installs the default development group and experiment dependencies without
notebook or plotting groups. Keep the copied raw images under
`experiments/imagenet-compression/data/raw/imagenetv2/` and the unchanged four
prepared files under `data/processed/imagenetv2/` in that experiment directory.
The maintained YAML's `cache.torch_home` resolves to
`/home/rafael/.cache/tltorch-model-compression/torch` for Cheetah's `rafael` user.
Its checkpoint path remains
`/home/rafael/.cache/tltorch-model-compression/torch/hub/checkpoints/resnet18-f37072fd.pth`,
the previously prepared location. This is a path-consistency check, not a new
remote existence or hash check. No cache export is required. Change the YAML
setting to use another existing cache. Keep the existing weights and prepared
selections; this setup does not move or download them.

For a readiness check, run only:

```bash
cd /home/rafael/projects/infq-tensorly-torch
/home/rafael/.local/bin/uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration.yaml --dry-run
```

After checking available GPU capacity and reviewing pending source/configuration
changes, manually start the selected model evaluation with:

```bash
cd /home/rafael/projects/infq-tensorly-torch
/home/rafael/.local/bin/uv run --no-sync python experiments/imagenet-compression/run_experiment.py --config experiments/imagenet-compression/configs/resnet18_exploration.yaml
```

Use `resnet18_exploration_dense.yaml` for a separate dense run or
`resnet18_exploration_tt.yaml` for TT/BlockTT. Each command evaluates only its selected
model. No cache or tracking exports are required.

This command may create a local execution-input commit, then starts one tracked run.
A Git commit failure stops it before tracking or model execution. The YAML selects `cuda:0`, evaluation batches
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

## Serial Optuna pilot

The optional controller chooses configurations and supervises the existing runner.
Each child still evaluates one model and owns its MLflow run and LoggedModel. The
controller does not add tracking fields or a second evaluator. Install the locked
optional group alongside the experiment dependencies when preparing an environment:

```bash
uv sync --locked --inexact --group experiments --group optuna
uv run --no-sync python experiments/imagenet-compression/optuna_controller.py \
  --study experiments/imagenet-compression/configs/studies/resnet18_pilot.yaml --dry-run
```

The dry-run validates the definition and dense template. It does not create study
state, execution YAMLs, commits, tracking requests, or models, and does not check
or download dataset images or weights. Ordinary manual runner use does not import
or require Optuna. Real execution omits `--dry-run`; that is a separate, reviewed
launch, not part of preparing or validating this implementation.

The supplied pilot queues eight complete requests: Dense; Tucker and TT on
`layer4.0.conv2` at 0.75; BlockTT and Tucker on `fc` at 0.75; TT on
`layer4.1.conv2` at 0.75; then CP on `layer4.0.conv2` and `fc` at 0.50. Other
layers remain dense. Queue exhaustion stops execution; it never starts a joint
search. The enabled names and legal conditional choices are declared in the study
YAML. Dense has no rank parameter. Compressed choices use floating storage targets,
method-specific parameter names, Tucker-only spatial preservation, and the fixed
classifier tensorization and fitting controls recorded there. Generated YAMLs
preserve the template's scientific settings and order; all-Dense is explicit
`compression: null`.

The objectives are full-precision JSON `top1_percent` (maximize) and actual
whole-model `parameter_count` (minimize). Latency stays in the normal results but
is not an objective. The pilot uses Optuna 4.9.0's explicitly seeded TPE sampler;
its fixed queue checks execution rather than demonstrating optimization. Shape-only
rank matches produce local warnings, never automatic equivalence claims or reuse.

Study state defaults to `~/.local/state/tltorch-model-compression/optuna/<study>/`.
**Verify that this is persistent local disk before deployment.** If home is
network-mounted, choose a persistent local-SSD path in a new study definition.
The directory contains `study.sqlite3`, `study.json`, sampler checkpoint files,
`summary.json`, and unique `attempts/` directories with controller-owned
`attempt.json`, captured stdout/stderr, a configuration copy, and a runner-owned
receipt. Keep it private: sampler pickle is trusted executable data, not an
interchange format. Use only owned state from this environment. SQLite is serial;
exclusive study and checkout locks prevent competing controllers, including two
different studies sharing one checkout. Manual processes are not locked or stopped.

Generated configurations use `configs/optuna/<study>/trial_<number>_<unique>.yaml`
and are never overwritten. Each runner receives its study and base template as
explicit provenance YAMLs. Its existing automatic snapshot captures those inputs
and the controller source. The controller creates no pre-run commits. Unrelated
staged and working files remain permitted. Compatibility uses scientific source
and input hashes, package versions, paths, machine/device identity, fitting and
runtime seeds, and fixed settings, rather than requiring unchanged HEAD. The first
snapshot may capture those already-hashed pending sources; subsequent commits
must change only this study's generated configurations.

Resume uses the same command and state directory with `--resume`. Without that
flag, an existing study is protected against accidental reuse; `--resume` also
refuses to create a new study when its state is missing. It retains the original eight
asked-attempt budget and four-hour deadline, including downtime, and never
requeues the pilot. Each child has at most 1,200 seconds or the remaining deadline.
Two consecutive recognized candidate failures stop the pilot; unknown, shared,
or provenance failures stop immediately. The conversion-failure policy is unchanged.
In addition, a direct PyTorch out-of-memory error from a compressed candidate's
`evaluation` or `timing` computation can fail that candidate without objectives.
Setup, source checks, writing, verification, uploads, and finalization stay outside
these receipt phases. Continuing requires a matching failed receipt and selected
configuration, a durable nonzero exit from the owned child, no live owned process
group, confirmed terminal cleanup of the saved tracking run, and unchanged source,
input, and environment provenance. During evaluation or timing, dense-reference
OOMs, wrapped OOMs, ordinary `MemoryError`, timeouts, interruptions, and uncertain
cleanup still stop execution. The existing conversion-failure policy remains unchanged.
A memory failure describes this execution context; it does not establish that the
candidate cannot fit on other hardware or under different contention. No failed
request is automatically retried or retuned, and budgets are not extended.
The stopped first pilot remains historical evidence; a later launch needs a
separately reviewed new study context.

Sampler state is saved after suggestions and
after tell. Database and checkpoint writes are separate transactions: a mismatch
stops for reconciliation rather than silently reseeding or claiming uninterrupted
sampling. Missing child identity or durable exit evidence also requires review.

Only a recorded exit zero plus a matching succeeded receipt, intact hashes,
consistent provenance, both model verifications, source verification and completed
run finalization can supply objectives. A measured top-1 of zero is valid. Exact
requested duplicates reuse only intact successful evidence in the same compatible
study, linking its original trial, commit, run, and model. Reuse consumes an attempt
and is reported separately from newly executed measurements. Failed duplicates
receive no invented objectives and are not automatically retried.

During controller execution, SIGINT (Ctrl+C), SIGTERM, and SIGHUP request graceful
shutdown. SIGHUP therefore stops an ordinary terminal-launched controller on
terminal disconnection. Signals intentionally inherited as ignored remain ignored,
including SIGHUP under `nohup`. The previous handlers are restored when controller
execution ends; imports, help and dry-run do not install them. A handler only
records the request. Ordinary supervisor code checks it while waiting for the
runner, stops further launches, and retains interruption and child-exit evidence.
Repeated signals do not interrupt bounded cleanup. SIGKILL, power failure and
abrupt controller crashes cannot provide graceful cleanup; their existing
conservative resume/reconciliation checks still apply.

On interruption or timeout the controller signals only its owned process group,
escalates if necessary, and reaps the child. Shutdown is bounded by 120 seconds,
including a separately bounded tracking-cleanup subprocess. Cleanup can query only
the saved run ID at its saved URI/workspace: RUNNING becomes KILLED, terminal runs
remain untouched. Missing IDs, unconfirmed termination, or unavailable tracking
leave `needs_cleanup` and stop further launches. Review that evidence explicitly;
do not delete state or blindly restart RUNNING trials. No multi-objective pruning,
automatic retries, joint-study transition, or confirmation/final evaluation is
implemented here. The real pilot and Cheetah disk/GPU readiness remain untested.


## Uniform Tucker sweep

`configs/studies/resnet18_uniform_tucker.yaml` defines the separate
`uniform_tucker_sweep` mode. It queues exactly these five requests in order:

| Case | Selected layers | Requested weight fraction |
| --- | --- | ---: |
| Dense pretrained baseline | None; `compression: null` | 1.0 plotting coordinate |
| Uniform Tucker | All 21 Conv2d/Linear modules | 0.90 |
| Uniform Tucker | All 21 Conv2d/Linear modules | 0.75 |
| Uniform Tucker | All 21 Conv2d/Linear modules | 0.50 |
| Uniform Tucker | All 21 Conv2d/Linear modules | 0.25 |

The explicit YAML layer list follows torchvision's registered module order: the
initial convolution, 16 residual-block convolutions, three shortcut projections,
and classifier. An offline test checks this list against `resnet18(weights=None)`;
that check does not evaluate or compress the full model. Every compressed case
applies Tucker and one global floating-point fraction to that same list. It keeps
full convolution spatial ranks, the classifier's input `(8,8,8)` / output
`(10,10,10)` tensorization, and the existing Tucker fitting controls.

The dense template supplies unchanged preprocessing, exploration images, seeds,
float32 precision, device, batch size and timing settings. Each runner child loads
its own untouched pretrained baseline; no candidate starts from an earlier
compressed model, and no training or fine-tuning occurs. Requested fractions are
approximate weight-storage targets. Integer rank rounding and fixed spatial modes
can produce different retained fractions; actual parameter counts and compression
reports remain authoritative. Rank selection is unchanged.

Validate without creating study state or trial files:

```bash
uv run --no-sync python experiments/imagenet-compression/optuna_controller.py \
  --study experiments/imagenet-compression/configs/studies/resnet18_uniform_tucker.yaml --dry-run
```

A real launch requires separate review and omits `--dry-run`; resume adds
`--resume`. The distinct study name, state path and trial directory protect the
historical individual pilot. Five total attempts include the dense baseline, with
one worker, 1,200 seconds per trial, a four-hour persistent deadline, the existing
two-consecutive-candidate-failure limit and 120-second shutdown allowance. The
existing sampler is saved for the same resume protocol, but every choice is fixed
by the queue: `method` and, for Tucker, one `weight_fraction`. There are no 21
independent suggestions, adaptive follow-up trials, retries or extended budgets.
The existing individual-pilot mode, receipts, supervision and failure rules apply.

### Compare the five executions in MLflow

Open workspace `quirc`, experiment `model-compression`. In Runs, select the names
beginning `resnet18-uniform-tucker-exploration-v1-`; the suffixes identify `dense`
or `tucker-0.90`, `tucker-0.75`, `tucker-0.50`, `tucker-0.25`. A Runs search can use
`tags.mlflow.runName LIKE 'resnet18-uniform-tucker-exploration-v1-%'`. Include the
dense execution and use completed, verified runs for comparisons. A failed
candidate retains diagnostics but supplies no accuracy/size objectives.

Use the native run-comparison scatter chart with X = `parameter_count` and
Y = `top1_percent`. Both are existing unprefixed run metrics and also belong to
each execution's LoggedModel. This is the primary accuracy-versus-actual-size plot.

For the optional requested-fraction comparison, follow the selected runs' model
links to their five LoggedModels (all named `model`); saved `run.json` files also
record their exact model IDs. Include the dense model. Display the model parameter
`compression.weight_fraction` and `top1_percent`. Where the installed native model
comparison chart supports numeric model parameters on axes, choose X =
`compression.weight_fraction` and Y = `top1_percent`. The dense coordinate `1.0`
denotes no conversion, not a rank=1.0 request. Compressed coordinates are derived
from the actual requests, not a separately maintained study value.

The shared UI was not accessed during implementation. Exact chart controls and
support for numeric LoggedModel-parameter axes depend on the deployed server/UI;
the tested SDK is MLflow 3.17.2. If that axis is unavailable, retain the native
model comparison table for requested fractions and the primary metric scatter
plot. Do not duplicate the parameter onto runs, add metrics or install a plotting
package to work around the UI. MLflow documents
[run/model tracking](https://mlflow.org/docs/latest/ml/tracking/) and
[LoggedModel search and comparison](https://mlflow.org/docs/latest/ml/search/search-models/);
those pages do not verify this deployment's chart controls.
