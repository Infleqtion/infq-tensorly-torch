"""Model selection and copying tests; layer arithmetic is covered in Step 1."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from tltorch import FactorizedConv, FactorizedLinear
from tltorch.utils.model_compression import compress_model


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, (3, 2), stride=(2, 1), padding=(1, 2), dilation=(1, 2),
                      bias=False, dtype=torch.float64),
            nn.BatchNorm2d(4, dtype=torch.float64),
        )
        self.head = nn.Linear(4, 6, dtype=torch.float64)
        self.untouched = nn.Linear(6, 2, dtype=torch.float64)
        self.register_buffer("scale", torch.tensor([2., 3.], dtype=torch.float64))
        self.features.eval()
        self.head.weight.requires_grad_(False)
        with torch.no_grad():
            self.features[1].running_mean.copy_(torch.arange(4, dtype=torch.float64))
            self.features[1].running_var.fill_(2)
            self.features[1].num_batches_tracked.fill_(7)

    def forward(self, inputs):
        raise AssertionError("Conversion must not run the model")


def state_values(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def assert_state_values(model, expected):
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def assert_independent_models(source, converted):
    original = state_values(source)
    source_tensors = list(source.parameters()) + list(source.buffers())
    converted_tensors = list(converted.parameters()) + list(converted.buffers())
    source_storage = {tensor.untyped_storage().data_ptr() for tensor in source_tensors}
    assert all(tensor.untyped_storage().data_ptr() not in source_storage for tensor in converted_tensors)
    with torch.no_grad():
        for tensor in converted_tensors:
            tensor.add_(1)
    assert_state_values(source, original)
    converted_values = state_values(converted)
    with torch.no_grad():
        for tensor in source_tensors:
            tensor.add_(2)
    assert_state_values(converted, converted_values)


def forbid_decomposition(monkeypatch):
    calls = []

    def unexpected_conversion(*args, **kwargs):
        calls.append(True)
        raise AssertionError("All requests must be validated before conversion")

    monkeypatch.setattr(FactorizedLinear, "from_linear", unexpected_conversion)
    monkeypatch.setattr(FactorizedConv, "from_conv", unexpected_conversion)
    return calls


@pytest.mark.parametrize("factorization,implementation", [("Tucker", "factorized"), ("TT", "reconstructed")])
def test_nested_replacements_preserve_state_and_copy_independence(factorization, implementation):
    torch.manual_seed(11)
    source = SmallModel()
    original = state_values(source)
    convolution = {"factorization": factorization, "rank": 1}
    if factorization == "Tucker":
        convolution.update(rank=(2, 2, 3, 2), fixed_rank_modes="spatial")
    layers = {
        "features.0": convolution,
        "head": {"factorization": "BlockTT", "rank": 1,
                 "in_tensorized_features": (2, 2), "out_tensorized_features": (2, 3)},
    }
    original_settings = deepcopy(layers)
    converted, report = compress_model(source, layers, implementation=implementation)

    assert type(converted) is SmallModel
    assert isinstance(converted.features[0], FactorizedConv)
    assert isinstance(converted.head, FactorizedLinear)
    assert type(source.features[0]) is nn.Conv2d and type(source.head) is nn.Linear
    assert type(converted.untouched) is nn.Linear
    assert converted.features[0].implementation == converted.head.implementation == implementation
    assert converted.features[0].bias is None
    torch.testing.assert_close(converted.head.bias, source.head.bias, rtol=0, atol=0)
    assert all(not p.requires_grad for p in converted.head.weight.parameters())
    assert converted.head.bias.requires_grad
    assert all(p.requires_grad for p in converted.features[0].weight.parameters())
    assert all(p.dtype == torch.float64 and p.device.type == "cpu" for p in converted.parameters())
    assert converted.features[0].kernel_size == (3, 2)
    assert tuple(converted.features[0].stride) == (2, 1)
    assert tuple(converted.features[0].padding) == (1, 2)
    assert tuple(converted.features[0].dilation) == (1, 2)
    for name, module in source.named_modules():
        assert converted.get_submodule(name).training == module.training
    assert_state_values(converted.features[1], state_values(source.features[1]))
    assert_state_values(converted.untouched, state_values(source.untouched))
    torch.testing.assert_close(converted.scale, source.scale, rtol=0, atol=0)
    assert_state_values(source, original)
    assert layers == original_settings
    expected_conv_rank = (2, 2, 3, 2) if factorization == "Tucker" else (1, 1, 1, 1, 1)
    assert report == {"replacements": [
        {"name": "features.0", "factorization": factorization,
         "requested_rank": convolution["rank"], "actual_rank": expected_conv_rank},
        {"name": "head", "factorization": "BlockTT", "requested_rank": 1, "actual_rank": (1, 1, 1)},
    ]}
    assert_independent_models(source, converted)


def test_empty_selection_copies_parameters_buffers_and_modes():
    torch.manual_seed(12)
    source = SmallModel()
    converted, report = compress_model(source, {})
    assert report == {"replacements": []}
    assert_state_values(converted, state_values(source))
    for name, module in source.named_modules():
        assert converted.get_submodule(name) is not module
        assert converted.get_submodule(name).training == module.training
    assert_independent_models(source, converted)


@pytest.mark.parametrize("layers,message", [
    ({"features.9": {"factorization": "TT", "rank": 1}}, "features.9.*exact name"),
    ({7: {"factorization": "TT", "rank": 1}}, "Layer 7.*exact name"),
    ({"head": {"rank": 1}}, "head.*factorization"),
    ({"head": {"factorization": "BlockTT"}}, "head.*rank"),
    ({"head": []}, "head.*settings must be a dictionary"),
    ({"head": {"factorization": "Tucker", "rank": 1}}, "head.*only.*BlockTT"),
    ({"features.0": {"factorization": "BlockTT", "rank": 1}}, "features.0.*Tucker.*TT"),
    ({"head": {"factorization": "BlockTT", "rank": 1, "typo": True}}, "head.*typo"),
    ({"head": {"factorization": "BlockTT", "rank": 1, "max_parameters": 10}}, "head.*max_parameters"),
    ({"head": {"factorization": "BlockTT", "rank": 1, "decomposition_kwargs": {}}}, "head.*decomposition_kwargs"),
    ({"head": {"factorization": "BlockTT", "rank": 1, "auto_tensorize": False}}, "head.*auto_tensorize"),
    ({"head": {"factorization": "BlockTT", "rank": 1,
               "in_tensorized_features": (2, 2)}}, "head.*provide both"),
    ({"head": {"factorization": "BlockTT", "rank": 1,
               "in_tensorized_features": (3, 2), "out_tensorized_features": (2, 3)}}, "head.*product 4"),
    ({"head": {"factorization": "BlockTT", "rank": 1,
               "in_tensorized_features": (4,), "out_tensorized_features": (2, 3)}}, "head.*same number of modes"),
    ({"features.0": {"factorization": "TT", "rank": (1, 1, 1, 1)}}, "features.0.*number of ranks"),
    ({"features.0": {"factorization": "TT", "rank": (2, 1, 1, 1, 1)}}, "features.0.*boundary"),
    ({"features.0": {"factorization": "Tucker", "rank": (1, 1, 1)}}, "features.0.*four entries"),
    ({"features.0": {"factorization": "TT", "rank": 1,
                      "fixed_rank_modes": "spatial"}}, "features.0.*fixed_rank_modes"),
    ({"features.0": {"factorization": "Tucker", "rank": 1,
                      "fixed_rank_modes": "channels"}}, "features.0.*must be 'spatial'"),
])
def test_invalid_request_is_found_before_any_conversion(layers, message, monkeypatch):
    torch.manual_seed(13)
    source = SmallModel()
    before = state_values(source)
    calls = forbid_decomposition(monkeypatch)
    # The first request is valid. An invalid later request must prevent both conversions.
    requests = {"untouched": {"factorization": "BlockTT", "rank": 1}, **layers}
    with pytest.raises(ValueError, match=message):
        compress_model(source, requests)
    assert calls == []
    assert_state_values(source, before)


@pytest.mark.parametrize("layer_name,factorization", [("head", "BlockTT"), ("features.0", "Tucker")])
@pytest.mark.parametrize("kind,extra_name", [
    ("parameter", "gain"),
    ("buffer", "calibration"),
    ("nonpersistent_buffer", "scratch"),
    ("child", "observer"),
])
def test_extra_registered_state_prevents_all_conversion(layer_name, factorization, kind, extra_name, monkeypatch):
    source = SmallModel()
    layer = source.get_submodule(layer_name)
    if kind == "parameter":
        layer.register_parameter(extra_name, nn.Parameter(torch.ones(6, dtype=torch.float64)))
    elif kind == "buffer":
        layer.register_buffer(extra_name, torch.ones(6, dtype=torch.float64))
    elif kind == "nonpersistent_buffer":
        layer.register_buffer(extra_name, torch.ones(6, dtype=torch.float64), persistent=False)
    else:
        layer.add_module(extra_name, nn.Identity().eval())
    extra = getattr(layer, extra_name)
    extra_value = extra.detach().clone() if isinstance(extra, torch.Tensor) else None
    before = state_values(source)
    calls = forbid_decomposition(monkeypatch)

    # A valid first request must not start conversion before the later invalid one is checked.
    with pytest.raises(ValueError, match=f"{layer_name}.*cannot preserve.*this version.*{extra_name}"):
        compress_model(source, {
            "untouched": {"factorization": "BlockTT", "rank": 1},
            layer_name: {"factorization": factorization, "rank": 1},
        })
    assert calls == []
    assert_state_values(source, before)
    assert source.get_submodule(layer_name) is layer
    assert getattr(layer, extra_name) is extra
    if extra_value is not None:
        torch.testing.assert_close(extra, extra_value, rtol=0, atol=0)
    else:
        assert not extra.training


def test_unselected_extra_state_is_preserved_in_an_independent_copy():
    torch.manual_seed(22)
    source = SmallModel()
    source.register_parameter("container_gain", nn.Parameter(torch.ones(1, dtype=torch.float64)))
    source.untouched.register_parameter("gain", nn.Parameter(torch.ones(2, dtype=torch.float64)))
    source.untouched.register_buffer("calibration", torch.ones(2, dtype=torch.float64))
    source.untouched.register_buffer("scratch", torch.ones(2, dtype=torch.float64), persistent=False)
    source.untouched.add_module("observer", nn.Identity().eval())
    original = state_values(source)
    scratch = source.untouched.scratch.clone()

    converted, _ = compress_model(source, {"head": {"factorization": "BlockTT", "rank": 1}})
    assert isinstance(converted.head, FactorizedLinear)
    assert_state_values(source, original)
    assert_state_values(converted.untouched, state_values(source.untouched))
    torch.testing.assert_close(converted.container_gain, source.container_gain, rtol=0, atol=0)
    torch.testing.assert_close(converted.untouched.scratch, scratch, rtol=0, atol=0)
    assert "scratch" in converted.untouched._non_persistent_buffers_set
    assert type(converted.untouched.observer) is nn.Identity
    assert converted.untouched.observer is not source.untouched.observer
    assert not converted.untouched.observer.training

    # Non-persistent buffers are absent from the state_dict snapshots in the helper.
    converted.untouched.scratch.zero_()
    torch.testing.assert_close(source.untouched.scratch, scratch, rtol=0, atol=0)
    source.untouched.scratch.add_(2)
    assert torch.count_nonzero(converted.untouched.scratch) == 0
    assert_independent_models(source, converted)


@pytest.mark.parametrize("rank", [True, 0, -1, 0.0, float("nan"), float("inf"), "full", [], (1, 0, 1), (1, 1.5, 1)])
def test_invalid_rank_values_are_rejected(rank, monkeypatch):
    calls = forbid_decomposition(monkeypatch)
    with pytest.raises(ValueError, match="head.*rank must"):
        compress_model(nn.ModuleDict({"head": nn.Linear(4, 6)}),
                       {"head": {"factorization": "BlockTT", "rank": rank}})
    assert calls == []


@pytest.mark.parametrize("modes", [(), (0, 4), (True, 4), (2.0, 2)])
def test_tensorized_modes_require_positive_integers(modes):
    with pytest.raises(ValueError, match="head.*in_tensorized_features.*positive integers"):
        compress_model(nn.ModuleDict({"head": nn.Linear(4, 6)}), {
            "head": {"factorization": "BlockTT", "rank": 1,
                     "in_tensorized_features": modes, "out_tensorized_features": (2, 3)},
        })


def test_fractional_tucker_target_preserves_spatial_ranks_and_reports_actual_shapes(monkeypatch):
    torch.manual_seed(14)
    source = nn.Conv2d(4, 5, (3, 2), dtype=torch.float64)
    original_conversion = FactorizedConv.from_conv
    received_ranks = []

    def record_rank_and_convert(layer, **options):
        received_ranks.append(options["rank"])
        return original_conversion(layer, **options)

    monkeypatch.setattr(FactorizedConv, "from_conv", record_rank_and_convert)
    converted, report = compress_model(source, {
        "": {"factorization": "Tucker", "rank": 0.5, "fixed_rank_modes": "spatial"},
    })

    assert received_ranks == [0.5]
    assert type(received_ranks[0]) is float
    actual_rank = tuple(converted.weight.core.shape)
    assert actual_rank == tuple(factor.shape[1] for factor in converted.weight.factors)
    assert tuple(converted.rank) == tuple(converted.weight.rank) == actual_rank
    assert actual_rank[2:] == source.kernel_size
    assert converted.weight.factors[2].shape == (3, 3)
    assert converted.weight.factors[3].shape == (2, 2)
    replacement = report["replacements"][0]
    assert replacement["requested_rank"] == 0.5
    assert type(replacement["requested_rank"]) is float
    assert replacement["actual_rank"] == actual_rank


@pytest.mark.parametrize("rank", [2, (2, 2, 3, 2), [2, 2, 3, 2]])
def test_compatible_tucker_spatial_ranks(rank):
    torch.manual_seed(14)
    source = nn.Conv2d(4, 5, (3, 2), dtype=torch.float64)
    converted, report = compress_model(source, {
        "": {"factorization": "Tucker", "rank": rank, "fixed_rank_modes": "spatial"},
    })
    expected = (2, 2, 3, 2)
    assert tuple(converted.rank) == tuple(converted.weight.rank) == expected
    assert tuple(converted.weight.core.shape) == expected
    assert tuple(factor.shape[1] for factor in converted.weight.factors) == expected
    assert converted.weight.factors[2].shape == (3, 3)
    assert converted.weight.factors[3].shape == (2, 2)
    assert report["replacements"][0]["requested_rank"] == rank
    assert report["replacements"][0]["actual_rank"] == expected


@pytest.mark.parametrize("rank", [(2, 2, 1, 1), [2, 2, 1, 1]])
def test_tucker_spatial_rank_conflict_prevents_all_conversion(rank, monkeypatch):
    calls = forbid_decomposition(monkeypatch)
    source = nn.ModuleDict({"head": nn.Linear(4, 6), "conv": nn.Conv2d(4, 5, (3, 2))})
    with pytest.raises(ValueError, match=r"conv.*spatial ranks \(3, 2\).*(1, 1)"):
        compress_model(source, {
            "head": {"factorization": "BlockTT", "rank": 1},
            "conv": {"factorization": "Tucker", "rank": rank, "fixed_rank_modes": "spatial"},
        })
    assert calls == []


@pytest.mark.parametrize("rank,expected", [(1, (1, 1, 1)), (1.0, (1, 3, 1))])
def test_integer_rank_and_float_target_keep_distinct_meanings(rank, expected):
    torch.manual_seed(15)
    converted, report = compress_model(nn.Linear(12, 20), {
        "": {"factorization": "BlockTT", "rank": rank,
             "in_tensorized_features": (3, 4), "out_tensorized_features": (4, 5)},
    })
    assert tuple(converted.rank) == expected
    assert type(report["replacements"][0]["requested_rank"]) is type(rank)


def test_automatic_single_core_blocktt_keeps_step1_behavior(monkeypatch):
    torch.manual_seed(16)
    source = nn.Linear(5, 7, bias=False)
    converted, report = compress_model(source, {"": {"factorization": "BlockTT", "rank": "same"}})
    assert tuple(converted.rank) == (1, 1)
    assert converted.in_tensorized_features == (5,)
    assert converted.out_tensorized_features == (7,)
    assert converted.implementation == "factorized"
    assert converted.bias is None
    torch.testing.assert_close(converted.weight.to_matrix(), source.weight, rtol=0, atol=0)
    calls = forbid_decomposition(monkeypatch)
    with pytest.raises(ValueError, match="Layer ''.*single BlockTT core.*rank"):
        compress_model(source, {"": {"factorization": "BlockTT", "rank": 0.5}})
    assert calls == []


class CustomLinear(nn.Linear):
    pass


@pytest.mark.parametrize("factory,message", [
    (lambda: CustomLinear(4, 6), "custom subclasses"),
    (lambda: nn.LazyLinear(6), "uninitialized lazy"),
    (lambda: FactorizedLinear((2, 2), (2, 3), factorization="BlockTT", rank=1), "already-factorized"),
    (lambda: nn.Conv1d(2, 3, 3), "ordinary nn.Linear or nn.Conv2d"),
    (lambda: nn.Linear(4, 6, dtype=torch.float16), "float32 or float64"),
    (lambda: nn.Linear(4, 6, device="meta"), "CPU or CUDA"),
    (lambda: nn.Conv2d(4, 6, 3, groups=2), "groups=1"),
    (lambda: nn.Conv2d(4, 6, 3, padding="same"), "string padding"),
    (lambda: nn.Conv2d(4, 6, 3, padding=1, padding_mode="reflect"), "zero padding mode"),
])
def test_unsupported_selected_layers(factory, message):
    torch.manual_seed(17)
    source = nn.ModuleDict({"selected": factory()})
    factorization = "Tucker" if isinstance(source.selected, nn.Conv2d) else "BlockTT"
    with pytest.raises(ValueError, match="selected.*" + message):
        compress_model(source, {"selected": {"factorization": factorization, "rank": 1}})


@pytest.mark.parametrize("hook_kind", ["forward", "backward", "parameter"])
def test_selected_hooks_are_rejected(hook_kind):
    layer = nn.Linear(4, 6)
    if hook_kind == "forward":
        layer.register_forward_hook(lambda *args: None)
    elif hook_kind == "backward":
        layer.register_full_backward_hook(lambda *args: None)
    else:
        layer.weight.register_hook(lambda gradient: gradient)
    with pytest.raises(ValueError, match="selected.*hooks"):
        compress_model(nn.ModuleDict({"selected": layer}),
                       {"selected": {"factorization": "BlockTT", "rank": 1}})


def test_selected_parametrizations_are_rejected():
    layer = nn.Linear(4, 6)
    parametrize.register_parametrization(layer, "weight", nn.Identity())
    with pytest.raises(ValueError, match="selected.*parametrizations"):
        compress_model(nn.ModuleDict({"selected": layer}),
                       {"selected": {"factorization": "BlockTT", "rank": 1}})


class CustomTensor(torch.Tensor):
    pass


@pytest.mark.parametrize("kind,message", [
    ("sparse", "dense and non-quantized"),
    ("quantized", "dense and non-quantized"),
    ("subclass", "ordinary dense Tensor or Parameter"),
])
def test_unsupported_selected_tensors(kind, message):
    torch.manual_seed(18)
    layer = nn.Linear(4, 6)
    weight = layer.weight.detach()
    if kind == "sparse":
        layer.weight = nn.Parameter(weight.to_sparse())
    elif kind == "quantized":
        layer.weight = nn.Parameter(torch.quantize_per_tensor(weight, 0.1, 0, torch.qint8), requires_grad=False)
    else:
        layer.weight = nn.Parameter(weight.as_subclass(CustomTensor))
    with pytest.raises(ValueError, match="selected.*" + message):
        compress_model(nn.ModuleDict({"selected": layer}),
                       {"selected": {"factorization": "BlockTT", "rank": 1}})


def test_restrictions_do_not_apply_to_unselected_layers():
    torch.manual_seed(19)
    source = nn.ModuleDict({"selected": nn.Linear(4, 6), "custom": CustomLinear(4, 6, dtype=torch.float16)})
    source.custom.register_forward_hook(lambda *args: None)
    converted, _ = compress_model(source, {"selected": {"factorization": "BlockTT", "rank": 1}})
    assert isinstance(converted.selected, FactorizedLinear)
    assert type(converted.custom) is CustomLinear
    assert_state_values(converted.custom, state_values(source.custom))


@pytest.mark.parametrize("model,layers,implementation,error,message", [
    (None, {}, "factorized", TypeError, "model must be"),
    (nn.Module(), [], "factorized", TypeError, "layers must be"),
    (nn.Module(), {}, "dense", ValueError, "implementation must be"),
])
def test_invalid_top_level_arguments(model, layers, implementation, error, message):
    with pytest.raises(error, match=message):
        compress_model(model, layers, implementation=implementation)


def test_later_decomposition_failure_keeps_source_and_original_cause(monkeypatch):
    torch.manual_seed(20)
    source = SmallModel()
    before = state_values(source)
    modes = {name: module.training for name, module in source.named_modules()}
    failure = RuntimeError("injected decomposition failure")
    original_conversion = FactorizedLinear.from_linear
    calls = []

    def convert_or_fail(layer, **options):
        calls.append(layer)
        if len(calls) == 2:
            with torch.no_grad():
                layer.weight.zero_()
            raise failure
        return original_conversion(layer, **options)

    monkeypatch.setattr(FactorizedLinear, "from_linear", convert_or_fail)
    with pytest.raises(RuntimeError, match="untouched.*injected decomposition failure") as error:
        compress_model(source, {
            "head": {"factorization": "BlockTT", "rank": 1},
            "untouched": {"factorization": "BlockTT", "rank": 1},
        })
    assert error.value.__cause__ is failure
    assert len(calls) == 2
    assert calls[0] is not source.head and calls[1] is not source.untouched
    assert_state_values(source, before)
    assert {name: module.training for name, module in source.named_modules()} == modes
    assert type(source.head) is type(source.untouched) is nn.Linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_model_copy_and_replacement():
    torch.manual_seed(21)
    source = nn.Sequential(nn.Linear(4, 6, device="cuda", dtype=torch.float32))
    converted, _ = compress_model(source, {"0": {"factorization": "BlockTT", "rank": 1}})
    assert isinstance(converted[0], FactorizedLinear)
    assert all(p.device == source[0].weight.device and p.dtype == torch.float32 for p in converted.parameters())
    assert_independent_models(source, converted)
