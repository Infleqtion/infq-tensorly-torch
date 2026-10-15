"""Select and convert ordinary layers in an independent model copy.

This module implements the first part of the model-compression contract. Shared
modules and tied parameters, RNG/autocast preservation, numerical safeguards,
storage/error measurements, and parameter caps remain separate work.
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
from ..factorized_layers.factorized_convolution import kernel_shape_to_factorization_shape
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
            options.update(auto_tensorize=False, in_tensorized_features=input_modes,
                           out_tensorized_features=output_modes)
        else:
            # Match from_linear's automatic tensorization without allocating factors.
            input_modes, output_modes = get_tensorized_shape(
                layer.in_features, layer.out_features, order=3, min_dim=2, verbose=False)
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

    Return ``(compressed_model, {"replacements": [...]})``. Each replacement records
    its name, factorization, requested rank, and actual rank. No storage or error
    measurements are included yet. All requests are validated before copying or
    decomposing, and only the copy is modified. No forward passes are performed.

    This incremental API is not exported at package level. Use models without
    shared-module aliases or tied parameters for now; handling those cases, RNG
    and autocast preservation, and complete numerical safeguards is deferred.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(layers, Mapping):
        raise TypeError("layers must be a dictionary mapping exact layer names to settings")
    if not isinstance(implementation, str) or implementation not in ("factorized", "reconstructed"):
        raise ValueError("implementation must be 'factorized' or 'reconstructed'")

    modules = dict(model.named_modules()) #build dict connecting each layer's name to its layer object
    requests = []
    for name, settings in layers.items():
        if not isinstance(name, str) or name not in modules:
            raise ValueError(f"Layer {name!r}: no registered module has this exact name")
        try:
            options = _prepare_conversion_options(modules[name], settings)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Layer {name!r}: {error}") from error
        requests.append((name, options, deepcopy(settings["rank"])))

    compressed = deepcopy(model)
    replacements = []
    for name, options, requested_rank in requests:
        layer = compressed.get_submodule(name)
        try:
            if type(layer) is nn.Linear:
                replacement = FactorizedLinear.from_linear(layer, implementation=implementation, **options)
            else:
                replacement = FactorizedConv.from_conv(layer, implementation=implementation, **options)
        except Exception as error:
            raise RuntimeError(f"Layer {name!r}: {options['factorization']} conversion failed: {error}") from error
        if name:
            parent_name, _, child_name = name.rpartition(".")
            setattr(compressed.get_submodule(parent_name), child_name, replacement)
        else:
            compressed = replacement
        replacements.append({
            "name": name,
            "factorization": options["factorization"],
            "requested_rank": requested_rank,
            "actual_rank": tuple(replacement.rank),
        })
    return compressed, {"replacements": replacements}
