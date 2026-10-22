# ImageNetV2 evaluation and selection partitions

- Exploration: 1,000 images, one per class; compare compression choices.
- Confirmation: 4,000 images, four per class; check the shortlist before fixing settings.
- Final evaluation: 5,000 images, five per class; evaluate only after settings are fixed.

These are evaluation partitions, not training or fine-tuning data. Exploration is a byte-for-byte copy of the original manifest. The other partitions use seed 20261019 unless overridden in the command below. Selection uses class membership and prior exposure, never predictions or accuracy.

The command validates and hashes files; it does not evaluate any model on the final set. Final-set freshness is relative to the audited exposure records, not proof about undocumented runs. `preparation.json` records source hashes, counts, and prior exposure.

Manifests use dataset-relative paths, integer labels, and SHA-256 hashes. They can be consumed by the existing `ManifestImages` / `make_loader` helpers. The old CLI's `select_examples` replay accepts only one or ten images per class; it has not been extended for these new partitions. No evaluation runner is added here.

Run from the repository root with the existing uv environment:

```bash
uv run --no-sync python -B -m examples.model_compression.data_splits --dataset /home/rafael_magaldi/.cache/tltorch-model-compression/data/imagenetv2-matched-frequency-format-val --exploration-manifest examples/model_compression/results/cpu_1000/selected_examples.json --used-manifest examples/model_compression/results/notebook_run_001/selected_examples.json --used-manifest examples/model_compression/results/notebook_run_002/selected_examples.json --used-manifest examples/model_compression/results/notebook_run_003/selected_examples.json --used-manifest examples/model_compression/results/notebook_run_004/selected_examples.json --seed 20261019 --output examples/model_compression/splits/imagenetv2_20261019
```

For another machine, change `--dataset` or set `IMAGENETV2_DIR` and omit that argument. Choose an unused `--output` directory. Review prior exposure before reusing the command after any further real-image evaluations.
