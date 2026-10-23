# Experiment configurations

`resnet18_exploration.yaml` describes one reusable dense-versus-compressed
comparison. See [the experiment README](../README.md) for execution, path rules,
tracking, and the exact artifacts saved on each run.

The optional `cache.torch_home` setting selects an existing PyTorch cache, as the
parent of `hub/`. The maintained configuration uses
`~/.cache/tltorch-model-compression/torch`; change that value to use another
location. Home-relative paths expand for the executing user, and relative paths
resolve from the experiment directory. Explicit YAML takes precedence over
`TORCH_HOME`. Omitting the section retains PyTorch's previous cache selection.
The runner applies the setting only during real execution; dry-run reports the
resolved configuration without changing the global Hub directory or checking weights.
