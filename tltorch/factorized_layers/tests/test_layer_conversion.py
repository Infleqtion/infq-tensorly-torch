"""Conversion references come from known weights or dense execution of the result."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tltorch import BlockTT, FactorizedConv, FactorizedLinear, TTTensor, TuckerTensor


def reconstructed_weight(layer):
    if isinstance(layer, FactorizedLinear):
        return layer.weight.to_matrix()
    weight = layer.weight.to_tensor()
    if isinstance(layer.weight, TTTensor):
        # TT stores output channels last; PyTorch expects them first.
        weight = weight.movedim(-1, 0)
    return weight


def assert_independent(source, converted):
    source_values = {name: value.detach().clone() for name, value in source.state_dict().items()}
    source_storage = {p.untyped_storage().data_ptr() for p in source.parameters()}
    assert all(p.untyped_storage().data_ptr() not in source_storage for p in converted.parameters())
    with torch.no_grad():
        for parameter in converted.parameters():
            parameter.add_(0.25)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_values[name], rtol=0, atol=0)
    converted_values = {name: value.detach().clone() for name, value in converted.state_dict().items()}
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.5)
    for name, value in converted.state_dict().items():
        torch.testing.assert_close(value, converted_values[name], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_blocktt_conversion_recovers_kronecker_weight(device):
    # Unequal row/column modes expose a transposition or incorrect interleaving.
    left = torch.tensor([[1., -2., 0.5], [3., 0.25, -1.]], dtype=torch.float64, device=device)
    right = torch.tensor([[2., -1.], [0.5, 3.], [-2., 0.75], [1.5, 4.]], dtype=torch.float64, device=device)
    source = nn.Linear(6, 8, bias=False, dtype=torch.float64, device=device)
    with torch.no_grad():
        source.weight.copy_(torch.kron(left, right))
    expected = source.weight.detach().clone()
    converted = FactorizedLinear.from_linear(
        source, factorization="BlockTT", rank=1, auto_tensorize=False,
        in_tensorized_features=(3, 2), out_tensorized_features=(2, 4),
    )
    assert converted.bias is None
    torch.testing.assert_close(converted.weight.to_matrix(), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(source.weight, expected, rtol=0, atol=0)
    assert_independent(source, converted)


@pytest.mark.parametrize("factorization,dtype", [("Tucker", torch.float64), ("TT", torch.float32)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_conv_conversion_recovers_separable_kernel(factorization, dtype, device):
    vectors = [torch.tensor(values, dtype=dtype, device=device) for values in
               ([1., -2., 0.5, 3.], [0.5, 2., -1.], [1., -0.5, 2.], [-1., 0.25])]
    expected = torch.einsum("o,i,h,w->oihw", *vectors)
    source = nn.Conv2d(3, 4, (3, 2), bias=False, dtype=dtype, device=device)
    with torch.no_grad():
        source.weight.copy_(expected)
    rank = (1, 1, 1, 1) if factorization == "Tucker" else (1, 1, 1, 1, 1)
    converted = FactorizedConv.from_conv(source, factorization=factorization, rank=rank)
    assert converted.bias is None
    torch.testing.assert_close(reconstructed_weight(converted), expected)
    torch.testing.assert_close(source.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("factorization", ["BlockTT", "Tucker", "TT"])
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
def test_lossy_conversion_forward_and_optimizer_step(factorization, implementation):
    torch.manual_seed(17)
    dtype = torch.float32 if factorization == "TT" else torch.float64
    if factorization == "BlockTT":
        source = nn.Linear(6, 8, dtype=dtype)
        source.bias.requires_grad_(False)
        converted = FactorizedLinear.from_linear(
            source, factorization=factorization, rank=1, implementation=implementation,
            auto_tensorize=False, in_tensorized_features=(2, 3), out_tensorized_features=(2, 4),
        )
        inputs = torch.randn(2, 3, 6, dtype=dtype, requires_grad=True)
    else:
        source = nn.Conv2d(3, 4, (3, 2), stride=(2, 1), padding=(1, 2),
                           dilation=(1, 2), dtype=dtype)
        source.bias.requires_grad_(False)
        rank = (1, 1, 1, 1) if factorization == "Tucker" else 1
        converted = FactorizedConv.from_conv(
            source, factorization=factorization, rank=rank, implementation=implementation,
        )
        inputs = torch.randn(2, 3, 9, 10, dtype=dtype, requires_grad=True)

    weight = reconstructed_weight(converted).detach()
    assert torch.linalg.vector_norm(weight - source.weight) > 1e-3
    reference_inputs = inputs.detach().clone().requires_grad_()
    if factorization == "BlockTT":
        expected = F.linear(reference_inputs, weight, source.bias.detach())
    else:
        expected = F.conv2d(reference_inputs, weight, source.bias.detach(),
                            stride=source.stride, padding=source.padding, dilation=source.dilation)
    actual = converted(inputs)
    torch.testing.assert_close(actual, expected)
    target = torch.linspace(-0.3, 0.7, actual.numel(), dtype=dtype).reshape_as(actual)
    loss = (actual - target).square().mean()
    reference_loss = (expected - target).square().mean()
    optimizer = torch.optim.SGD(converted.parameters(), lr=0.01)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)

    factors_before = [p.detach().clone() for p in converted.weight.parameters()]
    for parameter in converted.weight.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert converted.bias.grad is None
    bias_before = converted.bias.detach().clone()
    source_before = {name: p.detach().clone() for name, p in source.named_parameters()}
    optimizer.step()
    assert all(not torch.equal(before, after) for before, after in
               zip(factors_before, converted.weight.parameters()))
    torch.testing.assert_close(converted.bias, bias_before, rtol=0, atol=0)
    for name, parameter in source.named_parameters():
        torch.testing.assert_close(parameter, source_before[name], rtol=0, atol=0)
        assert parameter.grad is None


@pytest.mark.parametrize("factorization", ["BlockTT", "Tucker", "TT"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_conversion_preserves_state_and_independent_bias(factorization, device):
    torch.manual_seed(23)
    if factorization == "BlockTT":
        source = nn.Linear(6, 8, dtype=torch.float64, device=device)
    else:
        source = nn.Conv2d(3, 4, (3, 2), stride=(2, 1), padding=(1, 2),
                           dilation=(1, 2), dtype=torch.float64, device=device)
    source.eval()
    source.weight.requires_grad_(False)
    source.bias.requires_grad_(True)
    before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    if factorization == "BlockTT":
        converted = FactorizedLinear.from_linear(source, factorization=factorization, rank=1)
        inputs = torch.randn(2, 6, dtype=torch.float64, device=device)
    else:
        converted = FactorizedConv.from_conv(source, factorization=factorization, rank=0.5)
        inputs = torch.randn(2, 3, 9, 10, dtype=torch.float64, device=device)
        assert converted.kernel_size == source.kernel_size
        assert tuple(converted.stride) == source.stride
        assert tuple(converted.padding) == source.padding
        assert tuple(converted.dilation) == source.dilation
    assert all(not module.training for module in converted.modules())
    assert not source.training and not source.weight.requires_grad and source.bias.requires_grad
    assert all(not p.requires_grad for p in converted.weight.parameters())
    assert converted.bias.requires_grad
    assert all(p.dtype == source.weight.dtype and p.device == source.weight.device
               for p in converted.parameters())
    torch.testing.assert_close(converted.bias, before["bias"], rtol=0, atol=0)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    frozen_before = [p.detach().clone() for p in converted.weight.parameters()]
    optimizer = torch.optim.SGD(converted.parameters(), lr=0.01)
    converted(inputs).square().mean().backward()
    assert converted.bias.grad is not None and torch.isfinite(converted.bias.grad).all()
    optimizer.step()
    assert not torch.equal(converted.bias, before["bias"])
    for old, parameter in zip(frozen_before, converted.weight.parameters()):
        assert parameter.grad is None
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)
    assert_independent(source, converted)


@pytest.mark.parametrize("factorization,expected_rank", [
    ("BlockTT", (1, 4, 1)), ("Tucker", (4, 2, 3, 2)), ("TT", (1, 2, 6, 4, 1)),
])
def test_conversion_records_ranks_after_svd_shape_limits(factorization, expected_rank):
    torch.manual_seed(31)
    if factorization == "BlockTT":
        source = nn.Linear(6, 8, dtype=torch.float64)
        converted = FactorizedLinear.from_linear(
            source, factorization=factorization, rank=20, implementation="factorized",
            auto_tensorize=False, in_tensorized_features=(2, 3), out_tensorized_features=(2, 4),
        )
        inputs = torch.randn(2, 6, dtype=torch.float64)
    else:
        source = nn.Conv2d(2, 4, (3, 2), dtype=torch.float64)
        converted = FactorizedConv.from_conv(
            source, factorization=factorization, rank=20, implementation="factorized",
        )
        inputs = torch.randn(2, 2, 7, 8, dtype=torch.float64)
    # These are the full unfolding limits for the small shapes above.
    assert tuple(converted.rank) == tuple(converted.weight.rank) == expected_rank
    assert converted.input_rank == 20
    torch.testing.assert_close(converted(inputs), source(inputs))


def test_tucker_conversion_keeps_spatial_ranks():
    torch.manual_seed(41)
    source = nn.Conv2d(4, 5, (3, 2), dtype=torch.float64)
    converted = FactorizedConv.from_conv(
        source, factorization="Tucker", rank=0.5, fixed_rank_modes="spatial",
        implementation="factorized",
    )
    assert tuple(converted.rank[2:]) == (3, 2)
    assert converted.weight.factors[2].shape == (3, 3)
    assert converted.weight.factors[3].shape == (2, 2)
    inputs = torch.randn(2, 4, 7, 8, dtype=torch.float64)
    expected = F.conv2d(inputs, reconstructed_weight(converted), source.bias)
    torch.testing.assert_close(converted(inputs), expected)


@pytest.mark.parametrize("in_features,out_features,rank,automatic", [
    (5, 7, "same", True), (1, 4, 1.0, True), (1, 1, (1, 1), True), (6, 8, 3, False),
])
def test_single_pair_blocktt_exact_conversion(in_features, out_features, rank, automatic):
    source = nn.Linear(in_features, out_features, bias=False, dtype=torch.float64)
    options = {} if automatic else dict(
        auto_tensorize=False, in_tensorized_features=(in_features,),
        out_tensorized_features=(out_features,),
    )
    converted = FactorizedLinear.from_linear(source, factorization="BlockTT", rank=rank, **options)
    assert converted.in_tensorized_features == (in_features,)
    assert converted.out_tensorized_features == (out_features,)
    assert tuple(converted.rank) == (1, 1)
    assert converted.input_rank == rank
    assert converted.bias is None
    assert sum(p.numel() for p in converted.parameters()) == source.weight.numel()
    torch.testing.assert_close(converted.weight.to_matrix(), source.weight, rtol=0, atol=0)
    inputs = torch.arange(2 * in_features, dtype=torch.float64).reshape(2, in_features)
    for implementation in ("factorized", "reconstructed"):
        converted.implementation = implementation
        torch.testing.assert_close(converted(inputs), source(inputs))
    assert_independent(source, converted)


@pytest.mark.parametrize("automatic", [True, False])
def test_single_pair_blocktt_rejects_fractional_target(automatic):
    source = nn.Linear(5, 7, bias=False)
    options = {} if automatic else dict(
        auto_tensorize=False, in_tensorized_features=(5,), out_tensorized_features=(7,),
    )
    with pytest.raises(ValueError, match="single.*core.*rank"):
        FactorizedLinear.from_linear(source, factorization="BlockTT", rank=0.5, **options)


def test_single_pair_blocktt_checks_explicit_boundary_ranks():
    with pytest.raises(ValueError, match="boundary"):
        FactorizedLinear.from_linear(nn.Linear(5, 7), factorization="BlockTT", rank=(1, 2))


def test_explicit_singleton_modes_are_not_automatically_compression():
    source = nn.Linear(5, 7, bias=False, dtype=torch.float64)
    # Even rank one costs 36 weights, above floor(0.5 * 35) = 17.
    with pytest.raises(ValueError, match="budget 17.*rank-one.*36"):
        FactorizedLinear.from_linear(
            source, factorization="BlockTT", rank=0.5, auto_tensorize=False,
            in_tensorized_features=(5, 1), out_tensorized_features=(7, 1),
        )
    converted = FactorizedLinear.from_linear(
        source, factorization="BlockTT", rank=1, auto_tensorize=False,
        in_tensorized_features=(5, 1), out_tensorized_features=(7, 1),
    )
    torch.testing.assert_close(converted.weight.to_matrix(), source.weight)
    # A 7-by-5 core and a scalar core occupy one more entry than the matrix.
    assert sum(p.numel() for p in converted.parameters()) == 36
    assert source.weight.numel() == 35


@pytest.mark.parametrize("factorization", ["BlockTT", "Tucker", "TT"])
def test_decomposition_exception_preserves_source(factorization, monkeypatch):
    tensor_type = {"BlockTT": BlockTT, "Tucker": TuckerTensor, "TT": TTTensor}[factorization]
    failure = RuntimeError("injected decomposition failure")

    def fail_after_touching_input(self, tensor, **kwargs):
        tensor.zero_()
        raise failure

    monkeypatch.setattr(tensor_type, "init_from_tensor", fail_after_touching_input)
    source = nn.Linear(6, 8) if factorization == "BlockTT" else nn.Conv2d(2, 4, (3, 2))
    before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    with pytest.raises(RuntimeError, match="injected decomposition failure") as error:
        if factorization == "BlockTT":
            FactorizedLinear.from_linear(source, factorization=factorization, rank=1)
        else:
            FactorizedConv.from_conv(source, factorization=factorization, rank=1)
    assert error.value is failure
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_linear_explicit_bias_overrides_remain_available():
    biased = nn.Linear(6, 8)
    biasless = nn.Linear(6, 8, bias=False)
    assert FactorizedLinear.from_linear(biased, factorization="BlockTT", rank=1, bias=False).bias is None
    assert FactorizedLinear.from_linear(biasless, factorization="BlockTT", rank=1, bias=True).bias is not None


def test_conv_random_initialization_remains_explicit(monkeypatch):
    def unexpected_decomposition(*args, **kwargs):
        raise AssertionError("Explicit random initialization must not decompose weights")

    monkeypatch.setattr(TTTensor, "init_from_tensor", unexpected_decomposition)
    source = nn.Conv1d(2, 3, 3, dtype=torch.float64).eval()
    converted = FactorizedConv.from_conv(source, factorization="TT", rank=2, decompose_weights=False)
    assert not converted.training
    assert all(p.dtype == torch.float64 for p in converted.parameters())
    assert torch.isfinite(converted(torch.ones(1, 2, 7, dtype=torch.float64))).all()


@pytest.mark.parametrize("dtype", [None, torch.float32])
def test_conv_dtype_override_and_one_dimensional_conversion(dtype):
    torch.manual_seed(47)
    source = nn.Conv1d(2, 3, 3, dtype=torch.float64)
    converted = FactorizedConv.from_conv(source, factorization="TT", rank=2, dtype=dtype, device=None)
    expected_dtype = source.weight.dtype if dtype is None else dtype
    assert all(p.dtype == expected_dtype for p in converted.parameters())
    inputs = torch.randn(1, 2, 7, dtype=expected_dtype)
    expected = F.conv1d(inputs, reconstructed_weight(converted), source.bias.to(expected_dtype))
    torch.testing.assert_close(converted(inputs), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("factorization", ["BlockTT", "Tucker", "TT"])
def test_conversion_on_cuda(factorization):
    torch.manual_seed(51)
    dtype = torch.float64 if factorization == "TT" else torch.float32
    if factorization == "BlockTT":
        source = nn.Linear(6, 8, device="cuda", dtype=dtype)
        converted = FactorizedLinear.from_linear(source, factorization=factorization, rank=1,
                                                  implementation="factorized")
        inputs = torch.randn(2, 6, device="cuda", dtype=dtype)
        expected = F.linear(inputs, reconstructed_weight(converted), source.bias)
    else:
        source = nn.Conv2d(3, 4, (3, 2), stride=(2, 1), padding=(1, 2), dilation=(1, 2),
                           device="cuda", dtype=dtype)
        converted = FactorizedConv.from_conv(source, factorization=factorization, rank=0.5,
                                              implementation="factorized")
        inputs = torch.randn(2, 3, 9, 10, device="cuda", dtype=dtype)
        expected = F.conv2d(inputs, reconstructed_weight(converted), source.bias,
                            stride=source.stride, padding=source.padding, dilation=source.dilation)
    assert all(p.device == source.weight.device and p.dtype == dtype for p in converted.parameters())
    torch.testing.assert_close(converted(inputs), expected)
    converted(inputs).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in converted.weight.parameters())
