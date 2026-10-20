# Model compression from trained weights

We want someone with a trained PyTorch model to be able to say: “Replace these layers with smaller, factorized versions, using the weights my model has already learned.”

For example, a user might compress two convolutional layers and the final linear layer of an image classifier. They should receive a separate model they can evaluate or train, together with a report showing what changed. They should not have to write their own code for finding nested layers, copying biases, or counting the parameters in tensor factors.

The compressed weights will usually be approximations. Our job is to make conversion reliable and make its consequences visible. The experiment in this PR will show how much storage we save, what happens to accuracy, and whether inference becomes faster or slower.

This document describes the feature we want to build. It does not claim that the feature or experiment is already complete. It belongs to this new PR; `PR-Feature.md` describes the previous PR and must remain unchanged.

**We will reuse the conversion methods that already exist.**

TensorLy-Torch already has `FactorizedLinear.from_linear` and `FactorizedConv.from_conv`. The new utility should use them, with small fixes where needed to meet the behavior described below.

The inspiration is [torch-mpo's model-compression utility](https://github.com/krzysztofwos/torch-mpo/blob/0402b981ab9a9559691477a1057f22ed7747249c/src/torch_mpo/utils/compression.py): choose layers in a model and replace them with versions initialized from their trained weights. We are adopting that convenience, while using TensorLy-Torch's own representations and decompositions.

This work does not depend on the earlier `blocktt_MALS` PR. We are not adding a decomposition algorithm, searching for optimal ranks, or distributing a storage budget across the model.

**A user should be able to choose layers in one short call.**

Here is the proposed API, using an already-trained ResNet-18:

```python
from tltorch import compress_model

layers = {
    "layer4.0.conv2": {
        "factorization": "Tucker",
        "rank": 0.5,
    },
    "fc": {
        "factorization": "BlockTT",
        "rank": 0.5,
        "in_tensorized_features": (8, 8, 8),
        "out_tensorized_features": (10, 10, 10),
    },
}

compressed_model, report = compress_model(model, layers)
```

The dictionary keys are the layer names in the model. Here, `layer4.0.conv2` is a convolution inside a residual block, and `fc` is the final classifier. Only the named layers are selected; the other layers keep their existing weights. If several names refer to the same layer object, selecting any of them converts that shared layer everywhere it is used.

The classifier has 512 inputs and 1,000 outputs. Before applying BlockTT, we describe each of these dimensions as a product of smaller dimensions: 512 = 8 × 8 × 8 and 1,000 = 10 × 10 × 10.

The tuples provide those smaller dimensions, which are called “modes.” BlockTT pairs each input mode with an output mode, so both tuples must have the same number of entries. Their products must also match the layer’s input and output sizes.

Users can provide both tuples or omit both and let TensorLy-Torch’s existing automatic tensorization choose them.

Require a factorization and rank for each selected layer. Misspelled layer names, unknown options, and incompatible settings should produce clear errors. An empty selection returns an independent copy with a report showing no replacements.

The full signature is `compress_model(model, layers, *, implementation="factorized")`. By default, converted layers calculate their outputs using the stored factors directly. This is the `implementation="factorized"` option. Users may instead choose `implementation="reconstructed"`. In this mode, each forward pass first rebuilds an approximate dense weight from the factors, then uses that weight for the linear or convolutional operation. The model still stores the factor parameters, but the temporary dense weight requires additional working memory.

For the same factors, both modes should produce the same outputs within numerical precision. They differ in how the computation is performed, which can affect execution time and memory use. “Reconstructed” means rebuilding the weight represented by the factors. Any approximation error introduced during compression remains.

**The first PR supports three conversion choices.**

| Original layer | Replacement | What is decomposed |
| --- | --- | --- |
| `nn.Linear` | `FactorizedLinear`, BlockTT | A matrix with its input and output dimensions split into paired modes. |
| `nn.Conv2d` | `FactorizedConv`, Tucker | The convolution kernel. |
| `nn.Conv2d` | `FactorizedConv`, TT | The kernel ordered as input channels, height, width, output channels. |

BlockTT for a linear layer and TT for a convolution organize the weights differently.

For the linear layer, each factor connects one of the smaller input dimensions with one of the smaller output dimensions described above. This is the matrix representation also known as an MPO.

For the convolution, the factors correspond to the kernel’s four dimensions: input channels, height, width, and output channels.

These are different decompositions. Supporting BlockTT for linear layers does not automatically provide TT convolution support. Our convolution conversion will use TensorLy-Torch’s existing kernel decomposition; it does not need to reproduce the construction used by torch-mpo.

Initially support ordinary CPU and CUDA layers in float32 and float64. Convolutions must have `groups=1`, numeric padding, and zero padding mode. Preserve rectangular kernels and unequal stride, padding, and dilation tuples.

Leave other factorizations, other convolution dimensions, grouped/depthwise/transposed convolutions, and joint factorization for later. Selected custom subclasses, already-factorized layers, hooks, parametrizations, uninitialized lazy layers, sparse or quantized tensors, and unsupported devices/dtypes should be rejected. Custom code that reads a selected layer's dense weight directly, rather than calling the layer, is also outside this first version's contract.

For Tucker convolutions, users may set `fixed_rank_modes="spatial"`. This keeps the ranks for the kernel’s height and width equal to their original dimensions, while allowing compression along the channel dimensions. For example, a 3 × 3 kernel keeps spatial ranks of 3 and 3.

The spatial factor matrices still exist and count toward the replacement’s stored parameters. They can be trained unless the original weight was frozen.

For this first PR, accept the following per-layer settings:
- factorization and rank, required for every selected layer.
- max_parameters, an optional parameter limit for any selected layer.
- in_tensorized_features and out_tensorized_features, optional tuples for BlockTT linear layers. Users must provide both or omit both.
- fixed_rank_modes="spatial", an optional setting for Tucker convolutions.
Other per-layer options should raise a clear error. Additional decomposition options can be considered later when there is a concrete need for them.

**A requested rank is not a promise about the final model size.**

The requested compression is a target, so we must also report the actual result. For example, `rank=0.5` asks for approximately half as many weight parameters. The factor dimensions must be whole numbers, so the result may not match that fraction exactly. Some settings can even produce a larger representation than the original weight. This target applies separately to each selected layer’s weight parameters. It excludes the bias and is not a target for the whole model.

Similarly, `rank="same"` targets approximately the original number of weight parameters. It does not mean “preserve the weights exactly.” 

Users may also supply integer ranks or explicit rank sequences in the format accepted by the selected layer-conversion method. These specify decomposition ranks directly, rather than a target fraction of stored weight parameters.

Users who need a hard limit may also provide `max_parameters` for a selected layer. After conversion, we count everything stored as a parameter in that layer, including its bias. Frozen parameters count too: they still occupy memory even though training does not update them.

If the replacement exceeds the limit, conversion raises an error and leaves the original model unchanged. The user can then choose a smaller rank. The utility does not automatically try other ranks, and the limit applies to that layer rather than the whole model.

**Conversion should preserve everything we have not asked to approximate.**

Compression changes the selected weight matrices or convolution kernels. The other properties of the model should stay the same.

Start by making an independent copy of the model. This lets users keep the original as a baseline and compare it with the compressed version. Training or modifying the compressed model afterward must not change the original, and vice versa. The two models must not share the memory containing their parameters or buffers.

For each converted layer, preserve these properties:

- **Bias:** keep the original bias values. If the layer had no bias, do not introduce one.
- **Convolution settings:** keep the supported kernel size, stride, padding, and dilation. Compression should not accidentally change how the convolution moves across an image.
- **Device and numerical precision:** a layer on the GPU should remain on that GPU. A layer using float32 should continue to use float32, and likewise for float64.
- **Training settings:** preserve whether the layer is in training or evaluation mode. If its weight was frozen, the new factors must also be frozen. Handle the bias separately: a frozen weight does not necessarily mean a frozen bias.

Layers that were not selected should keep their existing parameters and stored information, including BatchNorm’s running statistics. Their outputs may still change because they receive different inputs from an earlier compressed layer. That is an expected consequence of approximation.

Conversion itself should not train the model or update its running statistics.

It should also leave PyTorch’s random-number generators unchanged, on both CPU and GPU. For example, calling conversion should not change the random numbers that a later training step would receive. This must remain true even if conversion fails.

If the surrounding code uses automatic mixed precision, conversion should still use the intended numerical precision. It must not quietly turn float32 weights into lower-precision factors or change the caller’s mixed-precision settings afterward.

For further training, users should create a new optimizer for the compressed model. Its factor parameters are new objects, so the original optimizer is still associated with the original model. The conversion utility does not transfer the previous optimizer’s history or accumulated gradients.

**Shared layers need special care.**

A model can use the same layer in more than one place. For example, `model.left` and `model.right` might be two names for the same linear layer.

If the user selects either name, convert that layer once and make both names refer to the same replacement. Otherwise, conversion would turn one shared layer into two independent layers and change how training works. List both names in the report. If the user gives them conflicting conversion settings, raise an error.

A different situation is two separate layer objects sharing only a weight or bias tensor. We will not support converting those tied parameters in the first PR. If a selected layer participates in such sharing, raise a clear error. More specialized arrangements where different tensors share underlying memory are also outside this first version’s scope.

**A failed conversion should leave the original model usable.**

Check layer names and settings before starting decompositions where possible. If copying, decomposition, measurement, or a requested parameter-cap check fails, raise an error instead of returning a compressed model. The original model and the caller’s random-number state must remain unchanged, even if some earlier layers were converted successfully.

The error should identify the affected layer and explain what went wrong. Invalid numerical values, such as NaN or infinity, should also cause failure.

Never hide a decomposition failure by substituting random weights or returning a partly initialized layer. Users must be able to trust that every returned replacement was initialized from the trained weights they supplied.

**The report should explain what actually happened.**

Return an ordinary dictionary describing the replacements and the model as a whole. Users should be able to see which layers changed, which factorizations and actual ranks were used, how much storage changed, and how closely the compressed weights approximate the originals.

For each replacement, show the original and new parameter counts. Also show the bytes occupied by parameters and, separately, by buffers. Provide the same measurements for the whole model, including layers that were not compressed.

Count the parameters actually stored in the factors, rather than the entries in the dense weight they represent. Include biases and frozen parameters: they still occupy memory. Count registered buffers too, such as BatchNorm’s running statistics.

Sharing must not inflate the totals. For example, if two names refer to the same layer, its parameters should appear only once in the whole-model count. Apply the same rule to shared parameter and buffer objects.

Calculate storage from each tensor’s number of elements and its element size. For example, 100 float32 values occupy 400 bytes. These measurements describe the stored parameters and buffers. They do not include activations, gradients, optimizer state, temporary calculations, or memory-management overhead, so they do not measure peak GPU memory or checkpoint file size.

For each replaced weight, reconstruct the compressed approximation and compare it with the original. Report two measurements:

- **Absolute error:** the size of the difference, measured using the Frobenius norm—the square root of the sum of squared differences between corresponding entries.
- **Relative error:** the absolute error divided by the original weight’s Frobenius norm. For example, an absolute error of 1 against an original norm of 10 gives a relative error of 0.1, or 10%.

An all-zero original weight needs special handling because its norm is zero. Report relative error as zero if the reconstruction is also exactly zero; otherwise report it as unavailable. The absolute error remains meaningful in either case.

Calculate these measurements without avoidable numerical overflow or loss of precision. Keep the measurements in the report, but do not retain the reconstructed dense weights.

Weight error describes the approximation made by a decomposition. It does not tell us whether the model still performs its task well; that requires evaluation on real examples.

**Tests should explain why we trust the conversion.**

Use small deterministic examples with understandable expected answers. In particular, separate these two questions:

- How well does the compressed weight approximate the original weight?
- Does the factorized layer correctly execute the operation represented by its compressed weight?

A lossy layer may differ from the original and still be implemented correctly. To test its forward pass, reconstruct its compressed weight and use an independent `F.linear` or `F.conv2d` calculation as the reference.

For each of the three supported conversions—BlockTT linear, Tucker convolution, and TT convolution—check the forward pass and gradient flow in both execution modes. Other preservation and failure tests can use whichever supported conversion makes the expected behavior easiest to understand.

| Test | How we know the result is correct |
| --- | --- |
| Select layers inside a small nested model. | The selected layers become factorized layers. Unselected layers retain their types, parameter values, and buffer values. Shared layers follow the sharing rule described above. |
| Check that the two models are independent. | Change the returned model and check that the original is unchanged. Then change the original and check that the returned model is unchanged. Include both parameters and buffers. |
| Preserve layer settings and caller state. | Give the source layers deliberately different settings, such as unequal convolution strides, a missing bias, evaluation mode, or a frozen weight with a trainable bias. Check that conversion preserves those choices, along with device, dtype, random-number state, and the surrounding autocast settings. |
| Recover weights with a known factorized structure. | Construct a small matrix using Kronecker products and a convolution kernel using separable factors. Their construction gives us ranks at which reconstruction should be accurate to numerical precision. Use unequal dimensions and nonsymmetric values so that swapping axes causes a visible error. |
| Execute a deliberately lossy approximation correctly. | Reconstruct the compressed weight and calculate the expected output with `F.linear` or `F.conv2d`. Both execution modes should agree with this reference, even when they differ from the original layer. |
| Train the compressed layer. | Backpropagation produces finite gradients for the trainable factors, and an optimizer step changes them. Frozen parameters remain unchanged. |
| Check counts and parameter limits. | Use a tiny example whose factor sizes and bias can be counted by hand. Check the reported counts and bytes, including buffers and model totals. A parameter cap equal to the replacement’s count passes; a cap one parameter smaller raises an error. |
| Preserve shared layers. | Two names for one source layer refer to one replacement. Both names appear in the report, and sharing does not double the totals. Conflicting settings and unsupported parameter ties raise clear errors. |
| Fail without damaging the source model. | Force a decomposition to fail after another layer has already converted successfully. The call raises an error, leaves the original model and random-number state unchanged, and returns no partial result. |

Include zero weights, invalid selections/settings, and the linear tensorization boundary described below. Ordinary tests must require no downloads, torchvision, dashboards, or benchmark hardware. CPU tests are required; CUDA checks run when available. Add tests alongside the behavior they establish.

**Additional notebook with experiment showing the trade-off on a real task.**

In addition to the implementation and checks, we should have a notebook demonstrating how to use the utility and measure its performance on a real task.. Use a pretrained torchvision ResNet-18 with `ResNet18_Weights.IMAGENET1K_V1`. This gives us a trained baseline without needing to train a classifier ourselves.

Evaluate it on ImageNetV2 matched-frequency. This dataset uses the same 1,000 classes as the pretrained classifier, so we can evaluate its predictions directly.

Running the experiment requires downloading approximately 45 MB of pretrained weights and 1.2 GiB of dataset files, plus space for extraction. Explain these downloads before starting them, and let users provide an existing local dataset instead. Document the dataset’s image-use terms and the additional packages needed to run the example.

Use the preprocessing supplied with the pretrained weights, including its image resizing, cropping, and normalization. Check the dataset labels before measuring accuracy. ImageNetV2’s numeric folder names are the actual class IDs: an image in folder `"10"` has label 10. Alphabetically sorting folder names and assigning new indices would produce incorrect labels.

Compare three versions of the model:

- The original pretrained model.
- A compressed model targeting approximately 75% of the original weight parameters in each selected layer.
- A compressed model targeting approximately 50% of the original weight parameters in each selected layer.

These percentages are requests to the rank-selection methods, not guaranteed final sizes. Report the actual counts produced.

For both compressed versions, select `layer4.0.conv2`, `layer4.1.conv2`, and `fc`. Use Tucker for the convolutions, keeping their spatial ranks at full size, and BlockTT for the classifier, using the dimension tuples shown earlier. This exercises both convolutional and linear conversion. TT convolution is covered separately by the correctness tests.

Create each compressed version from the untouched baseline. Use factorized execution and measure the initial effect of compression before any fine-tuning.

For a quick run, choose one image per class using a fixed random seed and save the selected filenames. This gives a 1,000-image evaluation that can be repeated exactly. Use all 10,000 images for a fuller comparison when practical. Every model must receive the same images with the same preprocessing, and the results must state how many images were evaluated.

Measure four things:

1. **Task performance.** Report top-1 accuracy, where the highest-scoring prediction must be correct, and top-5 accuracy, where the correct class must appear among the five highest-scoring predictions. Compare each compressed model with the dense baseline measured on those same images. Express the change in percentage points: a drop from 70% to 67% is a loss of 3 percentage points. Synthetic inputs cannot establish task accuracy.

2. **Storage.** Report actual parameter counts and the bytes occupied by parameters and buffers. Show both the whole-model totals and the contribution of the replaced layers, so readers can distinguish savings in selected layers from savings across the entire model.

3. **Inference latency.** Time all models on the same hardware, using the same dtype, batch size, and input shape. Put them in evaluation mode and run without gradient tracking. Start with batch size 1 and input shape `(1, 3, 224, 224)`, run 20 warm-up passes, and then take 100 timing measurements. On CUDA, ensure the GPU work has finished at the timing boundaries. Report the median and a measure of variation. Exclude image loading and transfers to the device. Synthetic inputs are suitable for this measurement.

4. **FLOPs, when the counting method supports the operations used.** Count the computation actually performed, including tensor contractions and any weight reconstruction. Explain whether a multiplication followed by an addition counts as one operation or two. Identify operations the counter misses. If coverage is incomplete, label any reported count as partial and leave the complete total unavailable.

Fewer parameters do not guarantee preserved accuracy or faster inference. The experiment should show what happens in this comparison; the utility does not promise a particular speedup or accuracy threshold.

Save a readable results CSV and the per-layer conversion details. Also save enough information to repeat the comparison: package versions, repository commit, pretrained weights, preprocessing, dataset version, selected image filenames, random seeds, conversion settings, hardware, and timing settings. Include relevant backend settings, such as CPU thread counts and CUDA precision settings.

Keep measurement separate from visualization. TensorBoard may display the saved measurements, but users should be able to produce and read the results without it.

The notebook should walk through the inputs, dense baseline, layer selection, conversion, and results. Keep the explanation short and put reusable measurement code in an example module. Include a saved results table so reviewers can understand the comparison without rerunning the experiment or opening a dashboard.

Dataset downloads, pretrained weights, optional visualization packages, and hardware benchmarks must remain outside the ordinary test suite.

**Implementation will proceed in small, reviewable steps.**

First make single-layer conversion reliable and test it. Then add model selection and copying, sharing and failure handling, and finally reports and parameter caps. Once those behaviors are established, build the measurement example and its notebook. Finish by reviewing the API, relevant existing tests, new tests, and documentation together.

Use clear names, direct control flow, and small helpers. Explain decisions and unfamiliar concepts; avoid comments that merely repeat the code. Each Codex step should report changed files, relevant decisions, validation performed, actual results, and unresolved issues, then stop for review.

**The earlier inspection gives us specific issues to check.**

The supplied Codex draft describes an inspection of commit `30a40c5`. It reports that the relevant conversion code matched commit `25c915d`. The findings below come from that draft; rewriting this document did not independently verify them.

The draft reported several problems that could affect this feature:

- Convolution conversion could discard the second value in an unequal stride tuple.
- A converted bias could share its underlying memory with the original bias.
- Conversion could lose evaluation mode or whether parameters were frozen.
- Some TT decomposition options were ignored.
- A layer’s recorded ranks could disagree with the ranks of its actual factors.

Check these findings against the checkout used for implementation. Make the small fixes needed for the agreed conversion behavior, and add tests that demonstrate those fixes.

One question remains open for BlockTT linear conversion. Automatic tensorization can sometimes produce just one input dimension and one output dimension, rather than splitting them into several pairs. In the inspected TensorLy 0.9.0 environment, fractional rank requests failed in this situation.

The draft proposed requiring at least two pairs. Before imposing that restriction, test which shapes and rank settings actually fail. If a small fix to the existing conversion path is sufficient, use it. Otherwise, reject the affected combinations with an error that explains what the user can change.

Adding dimensions of size 1 to the explicit tuples may provide a workaround. Check that behavior before recommending it, and do not suggest that adding those dimensions necessarily reduces storage.

The draft also reported 27 existing tests passing and one CP approximation test failing. Record that failure as part of the starting state. CP is outside this PR’s scope, so the failure does not by itself justify expanding this work to fix CP. Do not describe that run as a fully passing suite.

CUDA behavior and the real-model experiment had not been verified in that inspection. Their status must be reported separately as implementation proceeds.

Keep detailed commands and probe outputs in the implementation review notes. The acceptance criteria for this feature remain the behaviors and tests described in this document.

Reference sources: [linear conversion](tltorch/factorized_layers/factorized_linear.py), [convolution conversion](tltorch/factorized_layers/factorized_convolution.py), [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), and [ImageNetV2](https://github.com/modestyachts/ImageNetV2).
