# Experiment configurations

`resnet18_exploration.yaml` describes one compressed-model evaluation. See [the experiment README](../README.md) for execution, path rules,
tracking, and the exact artifacts saved on each run.

The optional `cache.torch_home` setting selects an existing PyTorch cache, as the
parent of `hub/`. The maintained configuration uses
`~/.cache/tltorch-model-compression/torch`; change that value to use another
location. Home-relative paths expand for the executing user, and relative paths
resolve from the experiment directory. Explicit YAML takes precedence over
`TORCH_HOME`. Omitting the section retains PyTorch's previous cache selection.
The runner applies the setting only during real execution; dry-run reports the
resolved configuration without changing the global Hub directory or checking weights.

For an explicitly named study definition or base template, use the repeatable
runner CLI option `--provenance-config`. These are additional provenance inputs,
not extra settings merged into the execution YAML. Only named regular YAML files
under this `configs/` directory are snapshotted; nested paths and repeated references
are supported. See the experiment README for `--status-file`, completion semantics,
and the side-effect-free `--dry-run` behavior of both options.

`studies/resnet18_pilot.yaml` is a controller definition, not an execution YAML.
It references the unchanged dense template and queues the reviewed eight individual
cases. The controller writes ordinary execution YAMLs under `configs/optuna/`,
preserving template settings. Use `uv run --no-sync python
experiments/imagenet-compression/optuna_controller.py --study
experiments/imagenet-compression/configs/studies/resnet18_pilot.yaml --dry-run`
from the repository root to inspect the plan without creating state. See the
experiment README for persistent state, strict completion, budgets and resume.
