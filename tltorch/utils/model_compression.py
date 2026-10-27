"""Convert selected layers in a model copy and measure storage and weight error."""

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real

import tensorly as tl
import torch
from torch import nn
from torch.nn.parameter import is_lazy
from torch.nn.utils.parametrize import is_parametrized

from ..factorized_layers import FactorizedConv, FactorizedLinear
from ..factorized_layers.factorized_convolution import (
    kernel_shape_to_factorization_shape, tensor_to_kernel,
)
from ..factorized_tensors.tensorized_matrices import validate_block_tt_rank
from .tensorize_shape import get_tensorized_shape


def _is_positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _normalize_rank(rank):
    if _is_positive_integer(rank):
        return int(rank)
    if isinstance(rank, Real) and not isinstance(rank, (Integral, bool)):
        if math.isfinite(rank) and rank > 0:
            return float(rank)
    if isinstance(rank, str) and rank == "same":
        return rank
    if isinstance(rank, (tuple, list)) and rank and all(_is_positive_integer(r) for r in rank):
        return tuple(int(r) for r in rank)
    raise ValueError("rank must be a positive integer, a finite positive float, 'same', "
                     "or a nonempty list/tuple of positive integers")


def _normalize_modes(modes, size, option):
    if not isinstance(modes, (tuple, list)) or not modes or not all(_is_positive_integer(m) for m in modes):
        raise ValueError(f"{option} must be a nonempty list/tuple of positive integers")
    product = math.prod(modes)
    if product != size:
        raise ValueError(f"{option} must have product {size}, got {product}")
    return tuple(int(m) for m in modes)


def _require_finite(tensor, description):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{description} contains NaN or infinity")


def _validate_layer(layer):
    if is_parametrized(layer):
        raise ValueError("selected layers with parametrizations are not supported")
    if any(is_lazy(p) for p in layer.parameters(recurse=False)):
        raise ValueError("uninitialized lazy layers are not supported")
    if isinstance(layer, (FactorizedLinear, FactorizedConv)):
        raise ValueError("already-factorized layers are not supported")
    if type(layer) not in (nn.Linear, nn.Conv2d):
        raise ValueError("select an ordinary nn.Linear or nn.Conv2d; "
                         f"custom subclasses and {type(layer).__name__} are not supported")

    # state_dict omits non-persistent buffers and children with no tensor state.
    extra_state = []
    extra_parameters = sorted(layer._parameters.keys() - {"weight", "bias"})
    if extra_parameters:
        extra_state.append(f"parameters {extra_parameters}")
    if layer._buffers:
        extra_state.append(f"buffers {sorted(layer._buffers)}")
    if layer._modules:
        extra_state.append(f"child modules {sorted(layer._modules)}")
    if extra_state:
        raise ValueError("conversion cannot preserve extra registered state in this version: "
                         + "; ".join(extra_state))

    # PyTorch has registration methods, but no public API to enumerate these hooks.
    hook_attributes = (
        "_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks",
        "_state_dict_pre_hooks", "_state_dict_hooks", "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    )
    if any(getattr(layer, attribute, None) for attribute in hook_attributes):
        raise ValueError("selected layers with hooks are not supported")
    for name in ("weight", "bias"):
        tensor = getattr(layer, name)
        if tensor is None:
            continue
        if type(tensor) not in (torch.Tensor, nn.Parameter) or tensor.is_nested:
            raise ValueError(f"{name} must be an ordinary dense Tensor or Parameter")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"{name} must be dense and non-quantized")
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must be on CPU or CUDA, got {tensor.device}")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} must use float32 or float64, got {tensor.dtype}")
        if getattr(tensor, "_backward_hooks", None) or getattr(tensor, "_post_accumulate_grad_hooks", None):
            raise ValueError(f"selected {name} has hooks, which are not supported")
        _require_finite(tensor, f"source {name}")

    if layer.bias is not None:
        if layer.bias.device != layer.weight.device or layer.bias.dtype != layer.weight.dtype:
            raise ValueError("weight and bias must use the same device and dtype")

    if type(layer) is nn.Conv2d:
        if layer.groups != 1:
            raise ValueError("convolutions must have groups=1")
        if layer.padding_mode != "zeros":
            raise ValueError("convolutions must use zero padding mode (padding_mode='zeros')")
        if (not isinstance(layer.padding, tuple) or len(layer.padding) != 2
                or any(not isinstance(p, Integral) or isinstance(p, bool) or p < 0 for p in layer.padding)):
            raise ValueError("padding must be two nonnegative integers; string padding is not supported")


def _normalize_decomposition_kwargs(factorization, settings):
    """Validate the supported fitting controls and fill in their effective defaults."""
    if not isinstance(settings, Mapping):
        raise ValueError("decomposition_kwargs must be a mapping")
    if factorization == "CP":
        options = {"init": "random", "n_iter_max": 100, "tol": 1e-8,
                   "random_state": 0, "l2_reg": 1e-5}
    else:
        options = {"init": "svd", "n_iter_max": 100, "tol": 1e-4, "random_state": None}
    unknown = settings.keys() - options.keys()
    if unknown:
        raise ValueError(f"unsupported decomposition_kwargs for {factorization}: "
                         + ", ".join(sorted(map(repr, unknown))))
    options.update(settings)
    init = options["init"]
    if factorization == "CP" and (not isinstance(init, str) or init != "random"):
        raise ValueError("CP requires init='random': the installed TensorLy SVD initialization "
                         "can allocate excessively large temporary tensors")
    if factorization == "Tucker" and (not isinstance(init, str) or init not in ("svd", "random")):
        raise ValueError("Tucker init must be 'svd' or 'random'")
    if not _is_positive_integer(options["n_iter_max"]):
        raise ValueError("n_iter_max must be a positive integer, excluding booleans")
    options["n_iter_max"] = int(options["n_iter_max"])
    for name in ("tol", "l2_reg") if factorization == "CP" else ("tol",):
        value = options[name]
        if (not isinstance(value, Real) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        options[name] = float(value)
    seed = options["random_state"]
    if seed is None and factorization == "Tucker":
        return options  # Preserve Tucker's existing unseeded default.
    if (not isinstance(seed, Integral) or isinstance(seed, bool)
            or not 0 <= seed < 2**32):
        raise ValueError("random_state must be an integer between 0 and 2**32 - 1 "
                         "(or None for Tucker); random-generator objects are not supported")
    options["random_state"] = int(seed)
    return options


def _prepare_conversion_options(layer, settings):
    _validate_layer(layer)
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a dictionary containing factorization and rank")
    missing = {"factorization", "rank"} - settings.keys()
    if missing:
        raise ValueError(f"missing required settings: {', '.join(sorted(missing))}")
    factorization = settings["factorization"]
    if not isinstance(factorization, str):
        raise ValueError("factorization must be CP/Tucker/BlockTT for Linear, or CP/Tucker/TT for Conv2d")

    factorization = factorization.lower()
    allowed = {"factorization", "rank", "max_parameters"}
    if type(layer) is nn.Linear:
        if factorization not in ("cp", "tucker", "blocktt"):
            raise ValueError("nn.Linear supports only factorization='CP', 'Tucker', or 'BlockTT'")
        factorization = {"cp": "CP", "tucker": "Tucker", "blocktt": "BlockTT"}[factorization]
        allowed.update(("in_tensorized_features", "out_tensorized_features"))
    else:
        if factorization not in ("cp", "tucker", "tt"):
            raise ValueError("nn.Conv2d supports only factorization='CP', 'Tucker', or 'TT'")
        factorization = {"cp": "CP", "tucker": "Tucker", "tt": "TT"}[factorization]
        if factorization == "Tucker":
            allowed.add("fixed_rank_modes")
    if factorization in ("CP", "Tucker"):
        allowed.add("decomposition_kwargs")
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported options for {factorization}: {', '.join(sorted(map(repr, unknown)))}")

    rank = _normalize_rank(settings["rank"])
    options = {"factorization": factorization, "rank": rank}
    if factorization in ("CP", "Tucker"):
        options["decomposition_kwargs"] = _normalize_decomposition_kwargs(
            factorization, settings.get("decomposition_kwargs", {}))
    shape = tuple(layer.weight.shape)
    if type(layer) is nn.Linear:
        has_input = "in_tensorized_features" in settings
        has_output = "out_tensorized_features" in settings
        if has_input != has_output:
            raise ValueError("provide both in_tensorized_features and out_tensorized_features, or omit both")
        if has_input:
            input_modes = _normalize_modes(settings["in_tensorized_features"], layer.in_features,
                                           "in_tensorized_features")
            output_modes = _normalize_modes(settings["out_tensorized_features"], layer.out_features,
                                            "out_tensorized_features")
            if len(input_modes) != len(output_modes):
                raise ValueError("input and output tensorized features must have the same number of modes")
        else:
            # Match from_linear's automatic tensorization without allocating factors.
            input_modes, output_modes = get_tensorized_shape(
                layer.in_features, layer.out_features, order=3, min_dim=2, verbose=False)
        # Resolve defaults so explicit and automatic choices compare by their actual modes.
        options.update(auto_tensorize=False, in_tensorized_features=tuple(input_modes),
                       out_tensorized_features=tuple(output_modes))
        shape = tuple(output_modes) + tuple(input_modes)
        if factorization == "BlockTT":
            validate_block_tt_rank((output_modes, input_modes), rank)

    if factorization == "CP":
        if isinstance(rank, tuple):
            raise ValueError("CP rank must be a positive integer, a float storage target, or 'same'; "
                             "rank sequences are not supported")
        resolved_rank = tl.cp_tensor.validate_cp_rank(shape, rank)
        if resolved_rank < 1:
            raise ValueError(f"CP rank target {rank!r} resolves to rank {resolved_rank}; "
                             "choose a target that gives at least rank 1")
    elif factorization == "Tucker":
        if isinstance(rank, tuple) and len(rank) != len(shape):
            if type(layer) is nn.Linear:
                raise ValueError(f"Tucker linear rank must have {len(shape)} entries: "
                                 "all output modes followed by all input modes")
            raise ValueError("Tucker rank must have four entries: output, input, height, width")
        fixed_modes = None
        if "fixed_rank_modes" in settings:
            if settings["fixed_rank_modes"] != "spatial":
                raise ValueError("fixed_rank_modes must be 'spatial'")
            expected = tuple(layer.kernel_size)
            if isinstance(rank, tuple) and rank[2:] != expected:
                raise ValueError(f"fixed_rank_modes='spatial' requires spatial ranks {expected}, "
                                 f"but rank={rank} requests {rank[2:]}")
            fixed_modes = [2, 3]
            options["fixed_rank_modes"] = "spatial"
        tl.tucker_tensor.validate_tucker_rank(shape, rank, fixed_modes=fixed_modes)
    elif factorization == "TT":
        shape = kernel_shape_to_factorization_shape("TT", shape)
        tl.tt_tensor.validate_tt_rank(shape, rank)
    return options


def _collect_requests(model, layers):
    # Keep every registered path, including descendants of shared containers.
    modules = dict(model.named_modules(remove_duplicate=False))
    aliases = {}
    tensor_owners = {}
    for name, module in modules.items():
        aliases.setdefault(id(module), []).append(name)
        for registrations in (module._parameters, module._buffers):
            for tensor_name, tensor in registrations.items():
                if tensor is not None:
                    path = f"{name}.{tensor_name}" if name else tensor_name
                    tensor_owners.setdefault(id(tensor), []).append((module, path))

    requests = {}
    for name, settings in layers.items():
        if not isinstance(name, str) or name not in modules:
            raise ValueError(f"Layer {name!r}: no registered module has this exact name")
        layer = modules[name]
        try:
            options = _prepare_conversion_options(layer, settings)
            max_parameters = None
            if "max_parameters" in settings:
                if not _is_positive_integer(settings["max_parameters"]):
                    raise ValueError("max_parameters must be a positive integer, excluding booleans")
                max_parameters = int(settings["max_parameters"])
            for parameter_name in ("weight", "bias"):
                parameter = getattr(layer, parameter_name)
                for owner, path in tensor_owners.get(id(parameter), []):
                    if owner is not layer:
                        raise ValueError(f"{parameter_name} is also registered as {path!r} "
                                         "on a different module; tied parameters are not supported")
        except (TypeError, ValueError, OverflowError, RuntimeError) as error:
            raise ValueError(f"Layer {name!r}: {error}") from error

        previous = requests.get(id(layer))
        if previous is not None:
            # Python equates 1 and 1.0; rank treats them as a size and a storage target.
            same_rank_kind = type(options["rank"]) is type(previous["options"]["rank"])
            if (options != previous["options"] or not same_rank_kind
                    or max_parameters != previous["max_parameters"]):
                raise ValueError(f"Layers {previous['name']!r} and {name!r}: "
                                 "conflicting settings for the same shared layer")
            continue
        requests[id(layer)] = {
            "name": name,
            "aliases": [alias for alias in aliases[id(layer)] if alias != name],
            "options": options,
            "requested_rank": list(options["rank"]) if isinstance(settings["rank"], list) else options["rank"],
            "max_parameters": max_parameters,
        }
    return list(requests.values())


@torch.no_grad()
def _measure_storage(modules):
    """Count registered tensor payloads once, with parameters taking precedence."""
    parameters, buffers = {}, {}
    for root in modules:
        for module_name, module in root.named_modules():
            for registrations, collected in ((module._parameters, parameters), (module._buffers, buffers)):
                for name, tensor in registrations.items():
                    if tensor is None:
                        continue
                    path = f"{module_name}.{name}" if module_name else name
                    if tensor.device.type == "meta":
                        raise ValueError(f"{path!r}: meta tensor storage is unsupported by this report; "
                                         "meta tensors have no materialized data")
                    if tensor.layout != torch.strided:
                        raise ValueError(f"{path!r}: {tensor.layout} tensor storage is unsupported by this report; "
                                         "only strided tensor storage can be measured")
                    collected[id(tensor)] = tensor

    # Collect both categories first: a buffer may also be a parameter elsewhere.
    return {
        "parameter_count": sum(tensor.numel() for tensor in parameters.values()),
        "parameter_bytes": sum(tensor.numel() * tensor.element_size() for tensor in parameters.values()),
        "buffer_bytes": sum(tensor.numel() * tensor.element_size()
                            for identity, tensor in buffers.items() if identity not in parameters),
    }


def _scaled_norm(tensor):
    """Return a scale and a normalized norm, avoiding squares of extreme values."""
    if tensor.numel() == 0:
        return 0.0, 0.0
    scale = tensor.abs().max()
    scale_value = scale.item()
    if scale_value == 0:
        return 0.0, 0.0
    return scale_value, torch.linalg.vector_norm(tensor / scale).item()


@torch.no_grad()
def _measure_weight_error(original, reconstructed):
    """Measure Frobenius errors without forming an overflowing source norm."""
    if original.shape != reconstructed.shape:
        raise ValueError(f"reconstructed weight shape {tuple(reconstructed.shape)} "
                         f"does not match source shape {tuple(original.shape)}")
    with torch.autocast(device_type=original.device.type, enabled=False):
        # Promotion must precede subtraction: opposite float32 values can overflow.
        original = original.to(dtype=torch.float64)
        difference = reconstructed.to(dtype=torch.float64) - original
        _require_finite(difference, "weight difference")
        error_scale, error_norm = _scaled_norm(difference)
        source_scale, source_norm = _scaled_norm(original)
        absolute = error_scale * error_norm
        if not math.isfinite(absolute):
            raise ValueError("absolute weight error cannot be represented as a finite number")
        if error_scale == 0:
            relative = 0.0
        elif source_scale == 0:
            relative = None
        else:
            # Separate powers of two so neither the source norm nor the ratio of
            # scales needs to fit in a float before the final result is computed.
            error_fraction, error_exponent = math.frexp(error_scale)
            source_fraction, source_exponent = math.frexp(source_scale)
            ratio = (error_fraction / source_fraction) * (error_norm / source_norm)
            try:
                relative = math.ldexp(ratio, error_exponent - source_exponent)
            except OverflowError as error:
                raise ValueError("relative weight error cannot be represented as a finite number") from error
            if relative == 0:
                raise ValueError("relative weight error cannot be represented as a nonzero finite number")
    return {"absolute": absolute, "relative": relative}


@torch.no_grad()
def _validate_replacement(replacement, factorization):
    for name, parameter in replacement.named_parameters():
        _require_finite(parameter, f"replacement parameter {name!r}")
    # Finite factors can overflow when multiplied. Keep this dense weight temporary.
    if isinstance(replacement, FactorizedLinear):
        weight = replacement.weight.to_matrix()
    else:
        # TT puts output channels last; use the existing kernel-layout conversion.
        weight = tensor_to_kernel(factorization, replacement.weight.to_tensor())
    _require_finite(weight, "reconstructed weight")
    return weight


def _convert_selected_layer(model, source_model, name, options, implementation, max_parameters):
    """Convert and measure a copied layer, enforcing its cap with layer context."""
    try:
        layer = model.get_submodule(name)
        source = source_model.get_submodule(name)
        with torch.autocast(device_type=layer.weight.device.type, enabled=False):
            if type(layer) is nn.Linear:
                replacement = FactorizedLinear.from_linear(layer, implementation=implementation, **options)
            else:
                replacement = FactorizedConv.from_conv(layer, implementation=implementation, **options)
            weight = _validate_replacement(replacement, options["factorization"])
            with torch.no_grad():
                original_storage = _measure_storage([source])
                compressed_storage = _measure_storage([replacement])
                if max_parameters is not None and compressed_storage["parameter_count"] > max_parameters:
                    raise ValueError(f"max_parameters={max_parameters} exceeded: "
                                     f"replacement stores {compressed_storage['parameter_count']} parameters")
                weight_error = _measure_weight_error(source.weight, weight)
    except Exception as error:
        raise RuntimeError(f"Layer {name!r}: {options['factorization']} conversion failed: {error}") from error
    return replacement, {"original": original_storage, "compressed": compressed_storage,
                         "weight_error": weight_error}


def _install_replacement(model, name, aliases, replacement):
    """Replace all registered names, returning the new root when needed."""
    for alias in [name, *aliases]:
        try:
            if alias:
                parent_name, _, child_name = alias.rpartition(".")
                setattr(model.get_submodule(parent_name), child_name, replacement)
            else:
                model = replacement
        except Exception as error:
            raise RuntimeError(f"Layer {name!r}: installing replacement at alias {alias!r} failed: "
                               f"{error}") from error
    return model


def compress_model(model, layers, *, implementation="factorized"):
    """Replace selected layers in an independent copy and report the actual result.

    ``layers`` maps exact registered module names to settings; the empty name
    selects the root. Each entry requires ``factorization`` and ``rank``. Linear
    supports CP, Tucker, or BlockTT, optionally with both ``in_tensorized_features``
    and ``out_tensorized_features``; omitting them uses the existing automatic
    tensorization. Both lists must have the same number of modes. This scope
    restriction avoids an existing Tucker factorized-forward problem with
    unequal lengths, although CP itself supports unequal lengths. Conv2d supports
    CP, Tucker, or TT; only Tucker accepts ``fixed_rank_modes="spatial"``.
    ``implementation`` is ``"factorized"`` (default) or ``"reconstructed"`` and
    controls how replacements execute their forward pass.

    An integer rank specifies a rank size; a float is an approximate storage target,
    and ``"same"`` requests a target of 1.0. CP has one rank and rejects sequences
    and targets resolving below one. Tucker linear rank sequences list all output
    modes followed by all input modes. Actual ranks can shrink during decomposition.
    Storage targets can produce more parameters than the dense weight, especially
    for Tucker; inspect the actual measurements or set ``max_parameters``.

    CP and Tucker accept a ``decomposition_kwargs`` mapping. Supported controls are
    ``init``, ``n_iter_max`` (positive integer), ``tol`` (finite, nonnegative), and
    ``random_state`` (integer from 0 through 2**32 - 1). CP also accepts finite,
    nonnegative ``l2_reg``. CP defaults are ``init="random"``, ``n_iter_max=100``,
    ``tol=1e-8``, ``random_state=0``, and ``l2_reg=1e-5``. Only random initialization
    is supported for CP: TensorLy's SVD path can allocate excessive temporary
    tensors. Random initialization starts fitting the original trained weights;
    it is never a fallback to unfitted random weights. Large CP ranks can still
    be expensive. The wrapper does not lower requested ranks, cap them, or retry
    fitting.

    Tucker keeps defaults ``init="svd"``, ``n_iter_max=100``, ``tol=1e-4``, and
    ``random_state=None``; it also accepts ``init="random"``. Integer seeds pass
    directly to decomposition without reseeding global NumPy state. Tucker's None
    keeps its existing unseeded behavior; choose an integer for reproducible random
    initialization. Other controls (including ``return_errors``) are rejected.
    TT and BlockTT do not accept ``decomposition_kwargs``.

    The optional ``max_parameters`` must be a positive integer, excluding booleans.
    It caps all stored replacement parameter elements, including bias and frozen
    parameters, but not buffers. Exceeding it raises an error without retrying ranks.
    A floating-point rank requests an approximate weight-storage target excluding
    bias; it is not an exact parameter cap. Neither option optimizes a model budget.

    Return ``(compressed_model, report)``. The report contains:

    - ``replacements``: one entry per unique converted layer, with ``name``,
      ``aliases``, ``factorization``, ``requested_rank``, ``actual_rank``,
      ``decomposition_kwargs`` (effective supported controls, or {} for TT/BlockTT),
      ``max_parameters`` (None when omitted), ``original``, ``compressed``, and
      ``weight_error``. CP's ``actual_rank`` is an integer; other factorizations use
      tuples describing the stored factors or core after decomposition.
    - ``replaced_layers`` and ``model``: each has ``original`` and ``compressed``
      storage measurements. The model totals include unchanged layers.

    Each storage measurement has ``parameter_count`` (elements, not objects),
    ``parameter_bytes``, and ``buffer_bytes``. Count actual factor parameters,
    biases, frozen parameters, and all registered buffers, including nonpersistent
    buffers. Bytes are ``numel() * element_size()``. Tensor object identity prevents
    double counting; an object registered as both a parameter and a buffer counts
    only as a parameter. Distinct tensors sharing storage are outside this contract.
    These are tensor payloads, excluding gradients, activations, optimizer state,
    temporary tensors, and allocator overhead, not peak memory or checkpoint size.
    Reporting requires strided tensors with materialized data. Sparse and other
    non-strided layouts need different accounting; meta tensors have no stored data.
    Their storage is unsupported by this report, even in unchanged layers or with
    an empty selection, so encountering them raises an error.

    ``weight_error`` contains Python floats ``absolute`` (Frobenius norm of the
    reconstructed minus original weight) and ``relative`` (absolute divided by
    the original norm). For zero source weights, relative is 0.0 for an exact zero
    reconstruction and None otherwise. Norms use scaling and float64 arithmetic;
    unrepresentable results raise errors. The report retains no tensors or graphs.
    Weight error does not establish task accuracy; evaluate the compressed model.
    An empty selection returns an independent copy, no replacements, zero replaced
    totals, and measured whole-model totals.

    Selected layers must be ordinary, initialized Linear or Conv2d modules on CPU
    or CUDA in float32 or float64. Convolutions require groups=1, numeric padding,
    and zero padding mode. Selected custom subclasses, hooks, parametrizations,
    sparse/quantized tensors, and already-factorized layers are unsupported. Selected
    layers may register only weight and bias parameters, with no buffers or children.
    These restrictions do not apply to unchanged layers, but their storage must be
    measurable. Custom code reading a selected layer's dense weight is unsupported.

    Shared layers are converted once and replaced at every registered path. Selected
    aliases must have compatible settings, including identical caps or all omitting
    the cap. Omitted decomposition controls and their explicit effective defaults
    compare equally; caller settings are not mutated. Weight or bias tensors tied
    to a different module are rejected.

    Validate all requests before copying or decomposing. Only the copy is modified;
    no forward passes or model-wide training-mode changes occur. CPU and CUDA RNG
    states are preserved through copying, conversion, measurements, and failures.
    Conversion and reconstruction disable autocast; measurements build no autograd
    graph. Preserve source dtype/device, bias, training mode, and trainability.
    Source weights/biases, replacement parameters, and reconstructed weights must
    be finite. Failures raise contextual errors instead of returning a partial model.
    Create a new optimizer for further training; optimizer state is not transferred.

    Example::

        from tltorch import compress_model

        compressed, report = compress_model(model, {
            "classifier": {"factorization": "BlockTT", "rank": 0.5,
                           "max_parameters": 10000},
        })
        print(report["model"]["compressed"]["parameter_bytes"])
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(layers, Mapping):
        raise TypeError("layers must be a dictionary mapping exact layer names to settings")
    if not isinstance(implementation, str) or implementation not in ("factorized", "reconstructed"):
        raise ValueError("implementation must be 'factorized' or 'reconstructed'")

    # Copying may run custom code that draws on any visible CUDA generator.
    # fork_rng also protects CPU state and works with an empty CUDA device list.
    with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
        requests = _collect_requests(model, layers)
        # Sparse tensors can fail during deepcopy; report the storage limitation first.
        try:
            original_model_storage = _measure_storage([model])
        except Exception as error:
            raise RuntimeError(f"Model storage measurement failed: {error}") from error
        try:
            compressed = deepcopy(model)
        except Exception as error:
            raise RuntimeError(f"Model copy failed: {error}") from error

        replacements = []
        for request in requests:
            name, options = request["name"], request["options"]
            replacement, measurements = _convert_selected_layer(
                compressed, model, name, options, implementation, request["max_parameters"])
            compressed = _install_replacement(compressed, name, request["aliases"], replacement)
            replacements.append({
                "name": name,
                "aliases": request["aliases"],
                "factorization": options["factorization"],
                "requested_rank": request["requested_rank"],
                "actual_rank": (int(replacement.rank) if options["factorization"] == "CP"
                                else tuple(int(rank) for rank in replacement.rank)),
                "decomposition_kwargs": dict(options.get("decomposition_kwargs", {})),
                "max_parameters": request["max_parameters"],
                **measurements,
            })

        try:
            report = {
                "replacements": replacements,
                "replaced_layers": {
                    "original": _measure_storage(model.get_submodule(r["name"]) for r in requests),
                    "compressed": _measure_storage(compressed.get_submodule(r["name"]) for r in requests),
                },
                "model": {"original": original_model_storage,
                          "compressed": _measure_storage([compressed])},
            }
        except Exception as error:
            raise RuntimeError(f"Model storage measurement failed: {error}") from error
    return compressed, report
