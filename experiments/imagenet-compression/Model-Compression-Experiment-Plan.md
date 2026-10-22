# Model-compression experiment plan

ResNet-18: choosing layers, factorizations, and compression levels.

Updated: **22 October 2026**. Active branch: `rafael/exp`.
Repository: [Infleqtion/infq-tensorly-torch](https://github.com/Infleqtion/infq-tensorly-torch).
Base commit: `e323c465493c9a956b34efe81ca4d36cba356933`.

The prepared data splits, configured runner, MLflow workspace support, and local
CPU/CUDA checks now exist. The experiment preparation and directory refactor are
work on `rafael/exp` after the base commit above. Local execution checks use small
models and a recording tracking client. Cheetah hardware and authenticated
workspace listing have been checked; a real tracked comparison, artifact
upload/download, the feasibility pilot, and the wider study have not run.

This plan is separate from the unchanged
[compression feature specification](../../PR-Pipeline-Model-Compression.md).
Use the [experiment README](README.md) for setup and execution commands.

## What we want to learn

Which layers, factorizations, and compression levels give useful
accuracy–storage trade-offs? We want to understand how much storage each layer
can give up before task accuracy suffers, and which combinations remain useful.

The main comparison is task accuracy against actual model storage. We also want
trustworthy estimates of arithmetic work (FLOPs), inference latency, and runtime
memory. CPU latency has lower priority because the eventual edge device may
behave differently.

There may be several useful choices rather than one model that wins every
metric. Candidates for which improving one objective requires sacrificing
another form the *Pareto frontier*. Begin without fine-tuning to measure the
immediate effect of replacing trained weights with factorized approximations.

## Keep the baseline and execution settings fixed

Use TorchVision ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1`, ImageNetV2
matched-frequency, the weight enum's preprocessing, and numeric folder labels.
Start every candidate from the untouched pretrained baseline, never from an
already-compressed candidate. Measure that baseline on the same partition as
the compressed model; published ImageNet accuracy is not a substitute.

Use factorized execution, float32, and evaluation mode. Select CPU or CUDA
explicitly in the configuration; unavailable CUDA must fail rather than silently
switch to CPU. Both use the repository's shared uv environment. The locally
tested stack is Python 3.12.3, PyTorch `2.7.1+cu128`, TorchVision `0.22.1+cu128`,
and TensorLy 0.9.0. Prepare Cheetah from this same lockfile and verify its package
versions and configured CUDA device before execution. Compare timings
within the same hardware, device, batch size, thread count, and execution settings.

Keep PyTorch's opt_einsum integration disabled and TensorLy's separate plugin
inactive. Fine-tuning, quantization, compilation, opt_einsum, and Optuna are not
search dimensions in this study. Historical comparison notebooks remain
separate evidence; in particular, the compilation notebook requires its recorded
PyTorch `2.5.1+cpu` environment and has not been validated under the current
PyTorch 2.7.1 CUDA environment.

## Run the study in four stages

### A. Bounded feasibility pilot

First establish whether representative convolutional and linear configurations
can be fitted and evaluated at a reasonable cost. Exercise all applicable
methods on a small, agreed list. Check conversion, evaluation, and recording;
measure fitting and evaluation costs; identify requests that save little storage
or increase it; audit FLOP coverage; and establish suitable time and memory limits.
Use the results to propose a larger run list and estimate its cost.

Some fractional CP requests produce large ranks and expensive fits. Some Tucker
requests produce little saving or a larger representation. The pilot must expose
these cases before a full grid is launched. Bounded explicit ranks can test
feasibility, but their accuracy is not a fair storage-matched comparison with
methods using much larger representations.

The exact pilot cases and resource limits remain to be agreed. Automated batches
and timeout handling are not implemented. Stop after the pilot for review before
launching the wider study.

### B. Individual-layer comparisons

Compress one layer per candidate so its effect on task performance can be
attributed to that conversion. The proposed candidates are the eight 3-by-3
convolutions in `layer3` and `layer4`, plus the classifier:

- `layer3.0.conv1`, `layer3.0.conv2`, `layer3.1.conv1`, `layer3.1.conv2`.
- `layer4.0.conv1`, `layer4.0.conv2`, `layer4.1.conv1`, `layer4.1.conv2`.
- `fc`.

Keep the initial convolution, earlier stages, and shortcut convolutions unchanged.
Supported methods are CP, Tucker, and TT for `Conv2d`; CP, Tucker, and BlockTT for
`Linear`.

| Provisional weight-storage target | Approximate retained storage | Approximate saving |
| --- | ---: | ---: |
| `0.90` | 90% | 10% |
| `0.75` | 75% | 25% |
| `0.50` | 50% | 50% |

This gives up to 9 layers × 3 methods × 3 targets = 81 configurations before
repeats and dense-baseline evaluations. The pilot may revise the list for
practical reasons. These fractions target a layer's weight representation;
they are not enforced counts or promised savings including bias or across the
whole model. An integer rank and a floating-point storage target have different
meanings. Compare actual parameter counts and bytes from the compression report.

Keep classifier tensorization fixed: input modes `(8, 8, 8)` and output modes
`(10, 10, 10)`. Keep full spatial ranks for Tucker convolutions with
`fixed_rank_modes="spatial"`. Tensorization choices and spatial-rank reduction
can be studied separately later.

Record requested settings, effective fitting controls, actual ranks, and seeds.
Keep infeasible or failed requests visible with a reason. Once timeout handling
exists, retain those outcomes too. Do not silently lower ranks or describe
unequal actual storage as a matched comparison.

### C. A small set of combined configurations

Use the individual-layer results to construct three to six complete models:
a conservative choice that preserves accuracy, choices with more savings and
an acceptable trade-off, and a more aggressive choice that explores the limits.
Different layers may use different methods and ranks; leaving a layer dense
must remain an option.

Evaluate every combined model. Accuracy changes from individual-layer tests
are not generally additive. Repeat promising configurations with a few fitting
seeds where randomness matters, and report variation rather than selecting only
the most favorable fit. This is a focused follow-up, not a repeated full grid.

### D. Confirmation and final evaluation

Use confirmation images to check the shortlist and settle final configurations.
Then freeze those choices and evaluate on the reserved final partition. Do not
use final results to continue tuning while still calling that partition untouched.

Concentrate detailed latency and future peak-memory measurements on the shortlist.
Later, benchmark chosen candidates on the intended edge GPU with its actual
precision, batch size, and execution environment.

## Use the prepared data splits

The ImageNetV2 matched-frequency partitions have already been prepared. Each
contains the same 1,000 numeric class labels, with folder `10` meaning class 10.

| Selection | Images | Per class | Purpose | Recorded prior exposure |
| --- | ---: | ---: | --- | ---: |
| Exploration | 1,000 | 1 | Compare compression choices | 1,000 |
| Confirmation | 4,000 | 4 | Check a small shortlist | 0 |
| Final evaluation | 5,000 | 5 | Evaluate frozen choices | 0 |

Exploration preserves the original evaluated manifest. The recorded audit found
1,000 unique previously evaluated images, all in exploration; later notebook
rerun manifests used the same images. It found no recorded exposure outside that
selection. This cannot rule out undocumented evaluations. Final-set freshness
is relative to the audited records, not proof of universal non-use.

The prepared selections cover all 10,000 images and were checked for counts,
numeric labels, paths, hashes, and disjointness by path and file content. The
remaining images were selected from sorted paths using a local random generator
with seed `20261019`, without predictions or accuracy. No regeneration is needed.
Reading filenames, labels, and hashes does not evaluate a model on the final set.

Exploration and confirmation influence selection and therefore serve as
validation data. Keep every model in a comparison on identical examples and
preprocessing. Review any new evidence of prior use before final evaluation.
See [data preparation](data/README.md) and the
[retained exposure audit](splits/imagenetv2_20261019/AUDIT.md), including its
limitations and historical paths.

## Keep the workflow and its inputs together

The maintained workflow lives under `experiments/imagenet-compression/`:

| Location | Purpose |
| --- | --- |
| `run_experiment.py` | Explicit preparation, configuration validation, and tracked execution |
| `configs/` | Reusable YAML configurations |
| `data/raw/imagenetv2/` | Original images, ignored by Git |
| `data/processed/imagenetv2/` | Prepared manifests and metadata, ignored by Git |
| `splits/imagenetv2_20261019/` | Retained original manifests and audit |
| Notebooks and comparison helpers | Explanations and separate historical comparisons |
| `results/` and `HISTORICAL-RUNS.md` | Preserved historical evidence |
| `results/runs/` | New execution directories, ignored by Git |
| `tests/` | Offline workflow checks with small artificial inputs |

Shared `pyproject.toml`, `uv.lock`, `.python-version`, the uv environment, and
`tltorch/` library code stay at the repository root. There is no second uv project.
The [README](README.md#prepare-once-then-run-a-configuration) gives detailed commands.

Preparation is an explicit action. Normal execution consumes existing manifests
and cached weights; it does not generate selections or download inputs. One YAML
describes one compressed configuration and its measured dense baseline on the
same partition. It can be executed repeatedly: each execution creates a unique
run ID and local output directory, saving the exact original and effective
configurations. Historical outputs must not be overwritten. DVC is deferred and
is not part of this workflow.

## Separate implemented measurements from planned work

The current runner implements one comparison. The table describes what it saves,
not results of the proposed pilot or wider study.

| Measurement or artifact | Current status |
| --- | --- |
| Top-1/top-5 accuracy | Implemented for both models, including changes in percentage points from the measured dense baseline |
| Mean cross-entropy; macro precision, recall, F1 | Implemented values for both models; their changes from baseline are not separately saved |
| Actual parameter counts; parameter and buffer bytes | Implemented through compression reports, with layer, replacement-total, and whole-model counts |
| Weight reconstruction errors; actual ranks | Implemented per replaced layer, alongside requested and effective conversion settings |
| Conversion time | Implemented separately from inference timing |
| Inference latency | Implemented median, quartiles, and raw samples with recorded warm-up/repeat settings |
| Compressed checkpoint and provenance | Implemented save/reload verification, configurations, hashes, environment, and source archive |
| Per-image outcomes for paired comparisons | Planned; current aggregate results are insufficient for this analysis |
| Layer-output error on fixed real inputs | Planned |
| Peak inference memory | Planned |
| Audited FLOP estimates | Planned; complete totals currently remain unavailable (`null`) |
| Automated batches and timeouts | Planned |
| Study-wide summaries, plots, and shortlist selection | Planned |

Cross-entropy can reveal changes in confidence even when the predicted class
stays the same. Macro precision, recall, and F1 average class-level scores across
all output classes; undefined scores contribute zero. These saved values must
not be described as separately recorded baseline changes.

Storage counts include frozen parameters, biases, and registered buffers, and
count shared tensor objects once. Compare actual stored factors and baseline
totals. Tensor payload bytes are neither checkpoint file size nor peak runtime
memory. Smaller factors can still create large intermediate tensors.

Weight reconstruction error measures approximation of trained weights, not task
accuracy. The planned layer-output diagnostic will give original and compressed
layers the same real inputs. Define its normalization and aggregation before
comparing trials. Neither diagnostic replaces task evaluation.

Latency uses preallocated inputs, warm-up, repeated measurements, and CUDA
synchronization where needed. It excludes conversion, data loading, preprocessing,
and transfers. Synthetic inputs are appropriate for timing, not accuracy. Future
peak-memory measurements need a stated scope and matched input shape/batch size,
without unrelated resident models influencing the comparison.

FLOPs should describe operations actually executed at a fixed input shape,
including weight reconstruction when inference performs it. State how
multiply-adds are counted and audit unsupported operations. Never present a
partial estimate as a complete total. Keep measured, not measured, and unavailable
values distinct; missing measurements are not zero.

## Check the reliability of comparisons

Per-image outcomes remain a measurement gap. Saving them will allow paired
comparisons, where each model is judged on the same images. On 1,000 images,
0.2 percentage points is only two additional correct predictions; tiny differences
need confirmation on more examples.

Use a few fitting seeds for promising random decompositions. Keep fitting
variation separate from evaluation-sample uncertainty and timing variation.
Compare actual storage rather than assuming equal requested fractions imply
equal model sizes. Interpret FLOPs only when coverage supports the comparison.

## Track comparisons with the shared MLflow server

Use the existing shared MLflow server on Cheetah. Normal configured execution
requires an HTTP(S) tracking address and a nonempty `tracking.workspace`; the
existing configuration selects `quirc` before experiment lookup or run creation.
On Cheetah use `http://127.0.0.1:9361` and let the client read the existing
`/home/rafael/.mlflow/credentials` automatically. Never archive credentials.
Workspace access failures stop execution without falling back elsewhere. The
workspace is saved in configurations, `run.json`, and a run tag.
The MLflow client is an optional experiment
dependency, not a dependency of the compression library. There is no offline
logging mode: retaining local files does not remove the server requirement.

The server stores tracking results; it does not launch computation. Start the
experiment process separately on whichever machine should perform it. The runner
uses the tracking API, not a direct connection to the server's SQLite database.
If server storage is discussed, its directory is `/mnt/chicago-ml/mlflow/` on
Cheetah; its deployment and filesystem still need verification.

Use one MLflow experiment for the study. Each run records one compressed
configuration, its fitting settings/seed, one evaluation partition, and a dense
baseline measured on that same partition. Clear run names can distinguish pilot,
individual-layer, combined, confirmation, and final comparisons; automated stage
or batch orchestration is not implemented.

Selected layers and requested settings are logged as individual parameters for
comparison. Actual ranks, effective fitting controls, storage, and approximation
errors remain in the compression report. Exact configuration files, the manifest,
environment/input provenance, and source artifacts accompany each run. The
effective configuration resolves absolute paths; its SHA-256 identifies that
exact file, not a machine-independent definition of an equivalent experiment.

Current Git provenance records the actual branch, commit, working-tree status,
and hashes of an explicitly selected source archive, including uncommitted code.
A commit hash alone does not identify a dirty checkout. The runner permits dirty
work and does not create commits. This plan does not settle a future Git policy.

Configuration/source artifacts are uploaded before model loading, and
environment/input provenance before compression. Local measurements and failure
evidence are retained as execution progresses. After a run is created, a failure
record with its traceback is uploaded if possible, and marking the run failed
is attempted separately. Tracking failures do not replace the original exception.
A run finishes only after checkpoint verification and required uploads succeed;
an incomplete checkpoint is not uploaded as a verified model.

Authenticated workspace listing using MLflow 3.17.2 has succeeded on Cheetah,
including access to `quirc`. Browser login alone would not establish client
authentication. Experiment/run creation, artifact upload/download, and the real
comparison remain unverified; do not create a diagnostic run. Local recording-
client checks establish code behavior, not those server permissions. The next
comparison will be started manually using the [Cheetah instructions](README.md#manual-execution-on-cheetah).

## Make study results easy to review

Each completed comparison already produces a two-row CSV, readable report,
configurations, compression reports, manifest, measurements, raw timings,
verified checkpoint, and provenance. Failed executions retain available local
evidence. These files support review alongside MLflow; they are not offline runs.

Study-wide CSV summaries, per-image analyses, plots, and shortlist decisions are
future work. Plan an accuracy-versus-storage plot and, when coverage permits,
an accuracy-versus-FLOPs plot with the dense baseline and useful trade-offs shown.
A later explanatory notebook can follow inputs, baseline, a visible conversion,
the bounded batch, results, and shortlist. Reuse measurement helpers and keep
explanations in the main narrative. Historical notebooks and results are evidence,
not proof that this new study has run.

## Next milestones

1. Finish readiness checks for Cheetah's locked environment and `quirc` access;
   hardware inspection and authenticated workspace listing are already established.
2. Make the reviewed code and hash-verified prepared inputs available there.
3. Manually run one real tracked comparison to test experiment/run creation and
   uploads, then verify artifact download. Preparation must stop before execution.
4. Address the measurement gaps needed for the pilot and agree on its bounded
   run list and resource limits.
5. Execute the feasibility pilot, review its costs and outcomes, and decide
   whether to launch the wider study.

Keep the wider sweep, combined-model search, and final evaluation behind that
review. We are organizing experiments around existing decompositions, not adding
a compression algorithm or general optimization framework.

## References

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/).
- [ALDS: Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/2adcfc3929e7c03fac3100d3ad51da26-Abstract.html).
- [Distribution-Aware Tensor Decomposition for Compression of Convolutional Neural Networks (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/7f6901ebab786e43b21530328fc989ca-Paper-Conference.pdf).

The papers motivate layer-specific choices and data-dependent diagnostics. This
plan does not propose implementing their decomposition or optimization algorithms.
