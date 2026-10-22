# Audit of recorded ImageNetV2 exposure

The inspected records support the requested 1,000 / 4,000 / 5,000 partition.
They identify **1,000 unique previously evaluated image paths**, all in the
original exploration selection. Repeated evaluations do not add distinct images.
No inspected record identifies prior use outside exploration. This is an audit
of retained evidence, not proof that no undocumented run occurred.

Starting commit: `e323c465493c9a956b34efe81ca4d36cba356933`, branch
`rafael/model-compression`. The experiment plan was read; its experiments were
not started.

## Evaluation evidence

Paths below are relative to `examples/model_compression/results/`. For each
directory, the audit compared every manifest entry and read `configuration.json`
and the accuracy counts in `measurements.json`. A manifest alone proves selection,
not evaluation; these directories also contain completed configurations and
recorded measurements for all three model variants.

| Directory | Recorded completion (UTC) | Images per model | Outside exploration |
| --- | --- | ---: | ---: |
| `cpu_1000/` | 2026-10-15 21:51:20 | 1,000 | 0 |
| `notebook_run_001/` | 2026-10-16 21:21:46 | 1,000 | 0 |
| `notebook_run_002/` (untracked) | 2026-10-19 19:01:01 | 1,000 | 0 |
| `notebook_run_003/` (untracked) | 2026-10-19 19:09:52 | 1,000 | 0 |
| `notebook_run_004/` (untracked) | 2026-10-19 19:28:06 | 1,000 | 0 |

All five `selected_examples.json` files are byte-for-byte identical, including
order, labels, and image hashes. Their SHA-256 is
`c3c6dadb55007cf9258a1633a15537698a8005f244c1f96d83496bb023910a3a`.
The original is `cpu_1000/selected_examples.json`, with seed **2026**. Its selection
was reproduced from local files without model evaluation. All later notebook
configurations explicitly replay that original manifest.

`cpu_1000/validation_runs.json` records an earlier completed 1,000-image evaluation
at 2026-10-15 21:43:26 and says the repeat corrected CSV line endings while selected
images matched exactly. There is no separate earlier manifest: image identity
rests on that comparison and the same reproducible selection settings.
`cpu_1000/presentation.json` records CSV-only reformatting and the original manifest
hash. Neither record adds distinct exposure.

## Other checks and evidence limits

- `timing_followup/round_1.json` through `round_3.json`, checked against
  `timing_followup.py`, use seeded `torch.randn` inputs, including conversion
  diagnostics. No dataset images are loaded.
- The opt_einsum and compile runs' `configuration.json` and `validation.json`,
  the compile run's `setup.json`, and both executed comparison notebooks describe
  synthetic inputs for timing, capture, and output-agreement checks.
- The integration check in `tests/test_experiment.py` uses a tiny linear model
  and artificial tensors. Its dataset-selection tests create artificial files.
- Supplemental local records also add no exposure: the executed
  `/tmp/tltorch-uv-setup-z_vd8t_y/notebook-smoke.ipynb` uses synthetic tensors;
  plotting-recovery scripts and `validated-plot.ipynb` under
  `/tmp/tltorch-plot-recovery-89y6_6vp/` render saved run-003 measurements.
  These temporary files are supplemental evidence, not a portable archive.

No separately recorded real-image integration check beyond the listed evaluations
was found in this bounded inspection. Missing or undocumented checks cannot be
reconstructed from these records.

The current main notebook's validation metadata still names `notebook_run_001`,
although its code and saved output point to `notebook_run_004` and later interactive
execution counts. It does not prove a fresh-kernel execution of the current code.
For exposure, the separately saved manifests and measurement counts agree.

Per-image prediction/execution traces were not retained. Saved aggregates cannot
prove that every interactive rerun independently recomputed each value. The audit
conservatively treats every image in the evaluation manifests as already used.
No manifest was counted as evaluated solely because it existed. No specific
ambiguous record identifies additional images; audit any such evidence before
using the final set, and regenerate only into a new directory.

## Verified partitions

The local dataset contains exactly 1,000 numeric class folders with ten images
each. Folder `10` means label 10. All 10,000 files have distinct SHA-256 hashes.
This checks identical file bytes, not visual similarity or alternate encodings.

| Partition | Images | Classes | Per class | Previously used |
| --- | ---: | ---: | ---: | ---: |
| Exploration | 1,000 | 1,000 | 1 | 1,000 |
| Confirmation | 4,000 | 1,000 | 4 | 0 |
| Final evaluation | 5,000 | 1,000 | 5 | 0 |

Exploration is an unchanged copy. The remaining partitions use local seed
**20261019**, never predictions or accuracy. Additional audited exposure would
be reserved for confirmation; none was needed. Impossible allocations and
cross-partition content duplicates cause errors rather than changed sizes.

The saved manifests passed independent checks of counts, coverage, labels,
existence, hashes, disjointness, and exclusion of prior exposure. A temporary
repeat produced identical manifest bytes. Nine focused tests use artificial files.
`preparation.json` records the exact command, source hashes, inventory/helper hashes,
Python version, and counts; `README.md` explains purpose and reuse.

Reading filenames, labels, and hashes is not model evaluation. The final set is
unused **relative to the audited records**, not certified untouched by every
possible undocumented evaluation. All pre-existing files were preserved.
