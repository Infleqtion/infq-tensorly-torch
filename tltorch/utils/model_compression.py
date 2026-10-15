"""Select and convert ordinary layers in an independent model copy.

Storage/error measurements and parameter caps remain separate work.
"""

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


def _prepare_conversion_options(layer, settings):
    _validate_layer(layer)
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a dictionary containing factorization and rank")
    missing = {"factorization", "rank"} - settings.keys()
    if missing:
        raise ValueError(f"missing required settings: {', '.join(sorted(missing))}")
    factorization = settings["factorization"]
    if not isinstance(factorization, str):
        raise ValueError("factorization must be BlockTT for Linear, or Tucker/TT for Conv2d")

    factorization = factorization.lower()
    allowed = {"factorization", "rank"}
    if type(layer) is nn.Linear:
        if factorization != "blocktt":
            raise ValueError("nn.Linear supports only factorization='BlockTT'")
        factorization = "BlockTT"
        allowed.update(("in_tensorized_features", "out_tensorized_features"))
    else:
        if factorization not in ("tucker", "tt"):
            raise ValueError("nn.Conv2d supports only factorization='Tucker' or 'TT'")
        factorization = "Tucker" if factorization == "tucker" else "TT"
        if factorization == "Tucker":
            allowed.add("fixed_rank_modes")
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported options for {factorization}: {', '.join(sorted(map(repr, unknown)))}")

    rank = _normalize_rank(settings["rank"])
    options = {"factorization": factorization, "rank": rank}
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
        validate_block_tt_rank((output_modes, input_modes), rank)
    elif factorization == "Tucker":
        if isinstance(rank, tuple) and len(rank) != 4:
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
        tl.tucker_tensor.validate_tucker_rank(tuple(layer.weight.shape), rank, fixed_modes=fixed_modes)
    else:
        shape = kernel_shape_to_factorization_shape("TT", tuple(layer.weight.shape))
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
            if options != previous["options"] or not same_rank_kind:
                raise ValueError(f"Layers {previous['name']!r} and {name!r}: "
                                 "conflicting settings for the same shared layer")
            continue
        requests[id(layer)] = {
            "name": name,
            "aliases": [alias for alias in aliases[id(layer)] if alias != name],
            "options": options,
            "requested_rank": deepcopy(settings["rank"]),
        }
    return list(requests.values())


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


def _convert_selected_layer(model, name, options, implementation):
    """Convert a layer in the copied model, preserving precision and error context."""
    try:
        layer = model.get_submodule(name)
        with torch.autocast(device_type=layer.weight.device.type, enabled=False):
            if type(layer) is nn.Linear:
                replacement = FactorizedLinear.from_linear(layer, implementation=implementation, **options)
            else:
                replacement = FactorizedConv.from_conv(layer, implementation=implementation, **options)
            _validate_replacement(replacement, options["factorization"])
    except Exception as error:
        raise RuntimeError(f"Layer {name!r}: {options['factorization']} conversion failed: {error}") from error
    return replacement


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
    """Replace named layers in an independent deep copy of ``model``.

    ``layers`` maps exact registered module names to settings. Each entry requires
    ``factorization`` and ``rank``. Linear supports BlockTT, optionally with both
    tensorized feature tuples. Conv2d supports Tucker or TT; only Tucker accepts
    ``fixed_rank_modes="spatial"``. The empty name selects the root module itself.

    Selected layers may register only ``weight`` and ``bias`` parameters, with
    no buffers or child modules: conversion cannot preserve extra registered state
    in this version. This restriction does not apply to unselected layers or model
    containers.

    Shared layers are converted once and replaced at every registered path.
    Selected aliases must have compatible settings. Weight or bias tensors tied
    to a different module are rejected; overlapping storage in distinct tensors
    is outside the supported scope.

    Return ``(compressed_model, {"replacements": [...]})``. Each replacement records
    the first selected name, its other registered names in ``aliases``, the
    factorization, requested rank, and actual rank. All requests are validated
    before copying or decomposing. Only the copy is modified; no forward passes
    or model-wide training-mode changes are performed.

    Copying and conversion preserve the caller's PyTorch CPU and CUDA RNG states,
    including on failure. Conversion and reconstruction run with autocast disabled
    for the layer's device, restoring the caller's context afterward. Source
    weights/biases, replacement parameters, and reconstructed weights must be
    finite. Failures raise an error instead of returning a partial model.

    Storage/error measurements, parameter caps, and the top-level public export
    remain deferred.
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
        try:
            compressed = deepcopy(model)
        except Exception as error:
            raise RuntimeError(f"Model copy failed: {error}") from error

        replacements = []
        for request in requests:
            name, options = request["name"], request["options"]
            replacement = _convert_selected_layer(compressed, name, options, implementation)
            compressed = _install_replacement(compressed, name, request["aliases"], replacement)
            replacements.append({
                "name": name,
                "aliases": request["aliases"],
                "factorization": options["factorization"],
                "requested_rank": request["requested_rank"],
                "actual_rank": tuple(replacement.rank),
            })
    return compressed, {"replacements": replacements}
