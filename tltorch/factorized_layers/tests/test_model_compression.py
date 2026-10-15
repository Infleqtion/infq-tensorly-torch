"""Model selection and copying tests; layer arithmetic is covered in Step 1."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from tltorch import FactorizedConv, FactorizedLinear
from tltorch.utils import model_compression
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


def rng_states():
    return [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]


def assert_rng_states(expected):
    actual = rng_states()
    assert len(actual) == len(expected)
    assert all(torch.equal(current, before) for current, before in zip(actual, expected))


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
        {"name": "features.0", "aliases": [], "factorization": factorization,
         "requested_rank": convolution["rank"], "actual_rank": expected_conv_rank},
        {"name": "head", "aliases": [], "factorization": "BlockTT", "requested_rank": 1, "actual_rank": (1, 1, 1)},
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
    random_state = rng_states()

    def convert_or_fail(layer, **options):
        torch.rand(3)  # A failing decomposition may already have consumed randomness.
        calls.append(layer)
        if len(calls) == 2:
            with torch.no_grad():
                layer.weight.zero_()
            raise failure
        return original_conversion(layer, **options)

    monkeypatch.setattr(FactorizedLinear, "from_linear", convert_or_fail)
    result = None
    with pytest.raises(RuntimeError, match="untouched.*injected decomposition failure") as error:
        result = compress_model(source, {
            "head": {"factorization": "BlockTT", "rank": 1},
            "untouched": {"factorization": "BlockTT", "rank": 1},
        })
    assert result is None
    assert error.value.__cause__ is failure
    assert_rng_states(random_state)
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


@pytest.mark.parametrize("nested", [False, True], ids=["shared-layer", "shared-container"])
@pytest.mark.parametrize("select_both", [False, True], ids=["one-alias", "both-aliases"])
def test_shared_layer_is_converted_once_and_all_aliases_are_replaced(nested, select_both, monkeypatch):
    torch.manual_seed(23)
    source = nn.Module()
    shared = nn.Linear(8, 8).eval()
    source.left = source.right = nn.Sequential(shared) if nested else shared
    left, right = ("left.0", "right.0") if nested else ("left", "right")
    before = state_values(source)
    original_conversion = FactorizedLinear.from_linear
    calls = []

    def record_conversion(layer, **options):
        calls.append(layer)
        return original_conversion(layer, **options)

    monkeypatch.setattr(FactorizedLinear, "from_linear", record_conversion)
    # List/tuple spelling and explicit/default modes express the same request.
    layers = {right: {"factorization": "blocktt", "rank": [1, 1, 1, 1],
                      "in_tensorized_features": [2, 2, 2], "out_tensorized_features": [2, 2, 2]}}
    if select_both:
        layers[left] = {"factorization": "BlockTT", "rank": (1, 1, 1, 1)}
    settings = deepcopy(layers)
    converted, report = compress_model(source, layers)
    replacement = converted.get_submodule(right)
    assert isinstance(replacement, FactorizedLinear)
    assert converted.get_submodule(left) is replacement
    if nested:
        assert converted.left is converted.right
    assert source.get_submodule(left) is source.get_submodule(right) is shared
    assert len(calls) == 1 and calls[0] is not shared
    assert not replacement.training
    assert_state_values(source, before)
    assert layers == settings
    assert report == {"replacements": [{
        "name": right, "aliases": [left], "factorization": "BlockTT",
        "requested_rank": [1, 1, 1, 1], "actual_rank": tuple(replacement.rank),
    }]}
    assert_independent_models(source, converted)


@pytest.mark.parametrize("other_rank", [2, 1.0], ids=["different-rank", "integer-versus-fraction"])
def test_conflicting_alias_settings_prevent_all_conversion(other_rank, monkeypatch):
    source = nn.ModuleDict({"first": nn.Linear(8, 8), "left": nn.Linear(8, 8)})
    source.right = source.left
    before = state_values(source)
    calls = forbid_decomposition(monkeypatch)
    with pytest.raises(ValueError, match="left.*right.*conflicting settings"):
        compress_model(source, {
            "first": {"factorization": "BlockTT", "rank": 1},
            "left": {"factorization": "BlockTT", "rank": 1},
            "right": {"factorization": "BlockTT", "rank": other_rank},
        })
    assert calls == []
    assert source.left is source.right
    assert_state_values(source, before)


@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
@pytest.mark.parametrize("owner", ["selected-layer", "unselected-layer", "container"])
def test_parameter_ties_to_other_modules_are_rejected(parameter_name, owner, monkeypatch):
    source = nn.ModuleDict({"selected": nn.Linear(4, 4), "other": nn.Linear(4, 4)})
    parameter = getattr(source.selected, parameter_name)
    layers = {"selected": {"factorization": "BlockTT", "rank": 1}}
    if owner == "container":
        source.register_parameter("saved_parameter", parameter)
        other_name = "saved_parameter"
    else:
        setattr(source.other, parameter_name, parameter)
        other_name = "other." + parameter_name
        if owner == "selected-layer":
            layers["other"] = {"factorization": "BlockTT", "rank": 1}
    before = state_values(source)
    calls = forbid_decomposition(monkeypatch)
    with pytest.raises(ValueError, match=f"selected.*{parameter_name}.*{other_name}.*tied parameters"):
        compress_model(source, layers)
    assert calls == []
    assert getattr(source.selected, parameter_name) is parameter
    assert_state_values(source, before)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
@pytest.mark.parametrize("empty", [False, True], ids=["selected", "empty-selection"])
@pytest.mark.parametrize("fail", [False, True], ids=["copy-success", "copy-failure"])
def test_copying_and_conversion_preserve_rng_even_for_empty_selection(device, empty, fail, monkeypatch):
    torch.manual_seed(24)
    source = nn.Sequential(nn.Linear(4, 6, device=device))
    before = state_values(source)
    original_copy = model_compression.deepcopy
    failure = RuntimeError("injected copying failure")

    def copy_with_randomness(value):
        if isinstance(value, nn.Module):
            # Model copying may call custom code that draws random numbers.
            torch.rand(3)
            torch.rand(3, device=device)
            if fail:
                raise failure
        return original_copy(value)

    monkeypatch.setattr(model_compression, "deepcopy", copy_with_randomness)
    layers = {} if empty else {"0": {"factorization": "BlockTT", "rank": 1}}
    random_state = rng_states()
    if fail:
        with pytest.raises(RuntimeError, match="Model copy failed.*injected copying failure") as error:
            compress_model(source, layers)
        assert error.value.__cause__ is failure
    else:
        converted, report = compress_model(source, layers)
        assert converted is not source
        assert len(report["replacements"]) == (0 if empty else 1)
        if empty:
            assert_state_values(converted, before)
    assert_rng_states(random_state)
    assert_state_values(source, before)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
@pytest.mark.parametrize("factorization", ["BlockTT", "Tucker", "TT"])
@pytest.mark.parametrize("fail", [False, True], ids=["success", "conversion-failure"])
def test_autocast_is_disabled_for_conversion_and_restored(device, factorization, fail, monkeypatch):
    torch.manual_seed(25)
    if factorization == "BlockTT":
        source = nn.Linear(4, 6, device=device, dtype=torch.float32)
        converter, method = FactorizedLinear, "from_linear"
        rank = 1
    else:
        source = nn.Conv2d(3, 4, (3, 2), device=device, dtype=torch.float32)
        converter, method = FactorizedConv, "from_conv"
        rank = (1, 1, 1, 1) if factorization == "Tucker" else 1
    original_conversion = getattr(converter, method)
    before = state_values(source)
    failure = RuntimeError("injected conversion failure")

    def check_precision_and_convert(layer, **options):
        assert not torch.is_autocast_enabled(device)
        torch.rand(3)
        torch.rand(3, device=device)
        replacement = original_conversion(layer, **options)
        if fail:
            raise failure
        return replacement

    monkeypatch.setattr(converter, method, check_precision_and_convert)
    outer_state = (torch.is_autocast_enabled(device), torch.get_autocast_dtype(device),
                   torch.is_autocast_cache_enabled())
    random_state = rng_states()
    amp_dtype = torch.bfloat16 if device == "cpu" else torch.float16
    with torch.autocast(device, dtype=amp_dtype, cache_enabled=False):
        settings = {"": {"factorization": factorization, "rank": rank}}
        if fail:
            with pytest.raises(RuntimeError, match="Layer ''.*injected conversion failure") as error:
                compress_model(source, settings)
            assert error.value.__cause__ is failure
        else:
            converted, _ = compress_model(source, settings)
            assert all(p.dtype == torch.float32 and p.device == source.weight.device
                       for p in converted.parameters())
        assert torch.is_autocast_enabled(device)
        assert torch.get_autocast_dtype(device) == amp_dtype
        assert not torch.is_autocast_cache_enabled()
    assert (torch.is_autocast_enabled(device), torch.get_autocast_dtype(device),
            torch.is_autocast_cache_enabled()) == outer_state
    assert_rng_states(random_state)
    assert_state_values(source, before)


@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_source_prevents_all_conversion(parameter_name, value, monkeypatch):
    source = SmallModel()
    with torch.no_grad():
        getattr(source.head, parameter_name).flatten()[0] = value
    before = state_values(source)
    calls = forbid_decomposition(monkeypatch)
    random_state = rng_states()
    with pytest.raises(ValueError, match=f"head.*{parameter_name}.*NaN or infinity"):
        compress_model(source, {
            "untouched": {"factorization": "BlockTT", "rank": 1},
            "head": {"factorization": "BlockTT", "rank": 1},
        })
    assert calls == []
    assert_rng_states(random_state)
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("factorization", ["BlockTT", "Tucker", "TT"])
@pytest.mark.parametrize("invalid", ["factor", "bias", "overflow"])
def test_nonfinite_replacement_is_rejected_without_changing_source(factorization, invalid, monkeypatch):
    torch.manual_seed(26)
    if factorization == "BlockTT":
        source = nn.Linear(4, 6)
        converter, method = FactorizedLinear, "from_linear"
        rank = 1
    else:
        source = nn.Conv2d(3, 4, (3, 2))
        converter, method = FactorizedConv, "from_conv"
        rank = (1, 1, 1, 1) if factorization == "Tucker" else 1
    original_conversion = getattr(converter, method)
    before = state_values(source)

    def invalid_conversion(layer, **options):
        replacement = original_conversion(layer, **options)
        with torch.no_grad():
            if invalid == "factor":
                next(replacement.weight.parameters()).fill_(float("nan"))
            elif invalid == "bias":
                replacement.bias.fill_(float("inf"))
            else:
                # Every stored value is finite, but multiplying the factors overflows float32.
                for parameter in replacement.weight.parameters():
                    parameter.fill_(1e20)
                assert all(torch.isfinite(p).all() for p in replacement.parameters())
                dense = (replacement.weight.to_matrix() if factorization == "BlockTT"
                         else replacement.weight.to_tensor())
                assert not torch.isfinite(dense).all()
        return replacement

    monkeypatch.setattr(converter, method, invalid_conversion)
    message = "reconstructed weight" if invalid == "overflow" else "replacement parameter"
    random_state = rng_states()
    with pytest.raises(RuntimeError, match=f"Layer ''.*{message}.*NaN or infinity") as error:
        compress_model(source, {"": {"factorization": factorization, "rank": rank}})
    assert isinstance(error.value.__cause__, ValueError)
    assert_rng_states(random_state)
    assert_state_values(source, before)


@pytest.mark.parametrize("check,error_type", [
    ("source weight", ValueError), ("reconstructed weight", RuntimeError),
])
def test_numerical_check_exception_keeps_layer_context_and_caller_state(check, error_type, monkeypatch):
    torch.manual_seed(27)
    source = nn.Linear(4, 6)
    before = state_values(source)
    original_check = model_compression._require_finite
    failure = RuntimeError("injected numerical check failure")

    def fail_during_check(tensor, description):
        if description == check:
            torch.rand(3)
            raise failure
        return original_check(tensor, description)

    monkeypatch.setattr(model_compression, "_require_finite", fail_during_check)
    random_state = rng_states()
    with pytest.raises(error_type, match="Layer ''.*injected numerical check failure") as error:
        compress_model(source, {"": {"factorization": "BlockTT", "rank": 1}})
    assert error.value.__cause__ is failure
    assert_rng_states(random_state)
    assert_state_values(source, before)


def test_alias_installation_failure_preserves_source_rng_and_original_cause():
    torch.manual_seed(28)
    failure = TypeError("container rejects factorized replacement")
    attempts = []

    class RejectingContainer(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

        def __setattr__(self, name, value):
            if name == "child" and isinstance(value, FactorizedLinear):
                torch.rand(3)  # Custom installation code may consume randomness before failing.
                attempts.append(self)
                raise failure
            super().__setattr__(name, value)

    source = nn.Module()
    shared = nn.Linear(4, 6).eval()
    source.selected = shared
    source.container = RejectingContainer(shared)
    before = state_values(source)
    modes = {name: module.training for name, module in source.named_modules()}
    random_state = rng_states()
    result = None

    # The selected path is installed first; its alias in the custom container then fails.
    with pytest.raises(RuntimeError, match=r"Layer 'selected'.*alias 'container\.child'.*rejects") as error:
        result = compress_model(source, {"selected": {"factorization": "BlockTT", "rank": 1}})
    assert result is None
    assert error.value.__cause__ is failure
    assert len(attempts) == 1 and attempts[0] is not source.container
    assert source.selected is source.container.child is shared
    assert_state_values(source, before)
    assert {name: module.training for name, module in source.named_modules()} == modes
    assert_rng_states(random_state)
