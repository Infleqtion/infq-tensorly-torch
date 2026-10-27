"""Contraction correctness and workspace bounds, independent of decomposition."""
import json
from math import prod, sqrt

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
import torch.backends.opt_einsum as opt_einsum
import tensorly as tl
from tensorly import plugins

from tltorch.factorized_tensors.tensorized_matrices import TuckerTensorized
from tltorch.factorized_tensors.complex_tensorized_matrices import ComplexTuckerTensorized
from tltorch.functional.factorized_linear import linear_tucker
from tltorch.functional.linear import factorized_linear

SEED = 20261027
DOUBLE_CLOSE = dict(rtol=1e-10, atol=1e-12)
SINGLE_CLOSE = dict(rtol=1e-5, atol=1e-6)
GRAD_CLOSE = dict(rtol=1e-9, atol=1e-11)


@pytest.fixture(autouse=True)
def controlled_arithmetic():
    """Control only this test's arithmetic, restoring every setting afterwards."""
    previous = (opt_einsum.enabled, tl.backend.current_backend().einsum,
                plugins.PREVIOUS_EINSUM, torch.get_float32_matmul_precision(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        opt_einsum.enabled = False
        plugins.use_default_einsum()
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        with torch.autocast('cpu', enabled=False), torch.autocast('cuda', enabled=False):
            yield
    finally:
        opt_einsum.enabled = previous[0]
        tl.backend.BackendManager.register_backend_method('einsum', previous[1])
        plugins.PREVIOUS_EINSUM = previous[2]
        torch.set_float32_matmul_precision(previous[3])
        torch.backends.cuda.matmul.allow_tf32 = previous[4]
        torch.backends.cudnn.allow_tf32 = previous[5]
        torch.use_deterministic_algorithms(previous[6], warn_only=previous[7])


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    return request.param


def make_weight(output_shape, input_shape, ranks, dtype, device):
    generator = torch.Generator().manual_seed(SEED)
    core = nn.Parameter((torch.randn(ranks, generator=generator, dtype=dtype) / 2).to(device))
    factors = [nn.Parameter((torch.randn(s, r, generator=generator, dtype=dtype) / sqrt(s)).to(device))
               for s, r in zip(output_shape + input_shape, ranks)]
    cls = ComplexTuckerTensorized if dtype.is_complex else TuckerTensorized
    return cls(core, factors, (output_shape, input_shape), rank=ranks)


def matrix_reference(core, factors, output_modes):
    """Reconstruct independently using Kronecker products and matrix products."""
    def kronecker(group):
        result = group[0]
        for factor in group[1:]:
            # Explicit Kronecker product also handles singleton-column QR strides.
            result = (result[:, None, :, None] * factor[None, :, None, :]).reshape(
                result.shape[0] * factor.shape[0], result.shape[1] * factor.shape[1])
        return result
    left = kronecker(factors[:output_modes])
    right = kronecker(factors[output_modes:])
    return left @ core.reshape(left.shape[1], right.shape[1]) @ right.T


@pytest.mark.parametrize('geometry', [
    ((3, 2), (2, 3, 2), (3, 2, 2, 3, 2)),
    ((2, 3, 2), (3, 2), (2, 2, 1, 2, 1)),
    ((1, 3), (2,), (1, 1, 1)),
], ids=['full-unequal', 'reduced-unequal', 'singleton'])
@pytest.mark.parametrize('transpose', [True, False], ids=['transpose', 'direct'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128], ids=['float64', 'complex128'])
def test_tucker_raw_forward(device, geometry, transpose, dtype):
    output_shape, input_shape, ranks = geometry
    weight = make_weight(output_shape, input_shape, ranks, dtype, device)
    matrix = matrix_reference(weight.core, list(weight.factors), len(output_shape))
    consumed, remaining = (input_shape, output_shape) if transpose else (output_shape, input_shape)
    generator = torch.Generator().manual_seed(SEED + 1)
    for leading in [(), (2,), (2, 3)]:
        inputs = torch.randn(*leading, prod(consumed), generator=generator, dtype=dtype).to(device)
        actual = linear_tucker(inputs, weight, transpose=transpose)
        expected = F.linear(inputs, matrix if transpose else matrix.T)
        assert actual.shape == (prod(leading), *remaining)
        torch.testing.assert_close(actual.reshape(expected.shape), expected, **DOUBLE_CLOSE)


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.complex128, torch.complex64])
def test_tucker_public_shape_bias_and_parameter_preservation(device, dtype):
    weight = make_weight((3, 2), (2, 2, 2), (2, 2, 2, 1, 2), dtype, device)
    matrix = matrix_reference(weight.core, list(weight.factors), 2)
    generator = torch.Generator().manual_seed(SEED + 2)
    inputs = torch.randn(3, 2, 8, generator=generator, dtype=dtype).to(device).transpose(0, 1)
    assert not inputs.is_contiguous()
    bias = torch.randn(6, generator=generator, dtype=dtype).to(device)
    parameters = list(weight.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    input_before = inputs.clone()
    tolerance = SINGLE_CLOSE if dtype in (torch.float32, torch.complex64) else DOUBLE_CLOSE
    for data in [inputs[0, 0], inputs]:
        for optional_bias in [None, bias]:
            actual = factorized_linear(data, weight, bias=optional_bias)
            expected = F.linear(data, matrix, optional_bias)
            assert actual.shape == (*data.shape[:-1], 6)
            assert actual.dtype == dtype and actual.device == data.device
            torch.testing.assert_close(actual, expected, **tolerance)
    assert all(a is b for a, b in zip(parameters, weight.parameters()))
    for parameter, original in zip(parameters, before):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    torch.testing.assert_close(inputs, input_before, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_tucker_gradients(device, dtype):
    weight = make_weight((2, 3), (2, 2), (2, 2, 2, 1), dtype, device)
    generator = torch.Generator().manual_seed(SEED + 3)
    inputs = torch.randn(2, 4, generator=generator, dtype=dtype).to(device).requires_grad_()
    bias = torch.randn(6, generator=generator, dtype=dtype).to(device).requires_grad_()
    parameters = [inputs, *weight.parameters(), bias]
    actual = factorized_linear(inputs, weight, bias=bias)
    expected = F.linear(inputs, matrix_reference(weight.core, list(weight.factors), 2), bias)
    actual_grad = torch.autograd.grad(actual.abs().square().sum(), parameters)
    expected_grad = torch.autograd.grad(expected.abs().square().sum(), parameters)
    for observed, reference in zip(actual_grad, expected_grad):
        assert torch.isfinite(observed).all() and observed.abs().sum() > 0
        torch.testing.assert_close(observed, reference, **GRAD_CLOSE)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_tucker_tuple_indexed_shared_gradients(device, dtype):
    generator = torch.Generator().manual_seed(SEED + 4)
    core = nn.Parameter(torch.randn(2, 2, 2, 2, 1, generator=generator, dtype=dtype).to(device))
    factors = [nn.Parameter(torch.randn(s, r, generator=generator, dtype=dtype).to(device))
               for s, r in zip((2, 2, 3, 2, 2), core.shape)]
    joint = TuckerTensorized(core, factors, (2, (2, 3), (2, 2)))
    inputs = torch.randn(2, 4, generator=generator, dtype=dtype).to(device).requires_grad_()
    bias = torch.randn(6, generator=generator, dtype=dtype).to(device).requires_grad_()
    actual, expected = 0, 0
    for index in [0, 1]:
        selected = joint[(index, slice(None), slice(None))]
        assert selected.core.grad_fn is not None
        assert all(a is b for a, b in zip(selected.factors, factors[1:]))
        actual = actual + factorized_linear(inputs, selected, bias=bias)
        # Collapse the joint mode directly, without using TuckerTensorized indexing.
        selected_core = torch.tensordot(factors[0][index], core, dims=([0], [0]))
        expected = expected + F.linear(inputs, matrix_reference(selected_core, factors[1:], 2), bias)
    parameters = [inputs, *joint.parameters(), bias]
    torch.testing.assert_close(actual, expected, **DOUBLE_CLOSE)
    gradients = torch.autograd.grad(actual.abs().square().sum(), parameters)
    references = torch.autograd.grad(expected.abs().square().sum(), parameters)
    for gradient, reference in zip(gradients, references):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, reference, **GRAD_CLOSE)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('transpose', [True, False])
def test_tucker_gradcheck(dtype, transpose):
    generator = torch.Generator().manual_seed(SEED + 5)
    consumed = 2 if transpose else 4
    inputs = torch.randn(1, consumed, generator=generator, dtype=dtype).requires_grad_()
    core = torch.randn(2, 1, 2, generator=generator, dtype=dtype).requires_grad_()
    factors = [torch.randn(s, r, generator=generator, dtype=dtype).requires_grad_()
               for s, r in zip((2, 2, 2), core.shape)]

    def forward(data, center, *matrices):
        # Buffers retain these non-Parameter tensors' graphs during finite differences.
        weight = TuckerTensorized(center, matrices, ((2, 2), (2,)))
        return linear_tucker(data, weight, transpose=transpose)

    assert torch.autograd.gradcheck(forward, (inputs, core, *factors),
                                    eps=1e-6, rtol=1e-3, atol=1e-5)


class TensorSizeLimit(TorchDispatchMode):
    """Reject oversized operation results, independent of the chosen operator names."""
    def __init__(self, limit):
        self.limit = limit
        self.maximum = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        values = (result,) if isinstance(result, torch.Tensor) else result
        if isinstance(values, (tuple, list)):
            for value in values:
                if isinstance(value, torch.Tensor):
                    self.maximum = max(self.maximum, value.numel())
                    assert value.numel() <= self.limit, 'input/core outer product exceeded workspace bound'
        return result


def test_tucker_bounded_intermediates(device, monkeypatch, record_property):
    weight = make_weight((3, 2), (2, 2), (3, 2, 2, 2), torch.float64, device)
    inputs = torch.ones(2, 4, dtype=torch.float64, device=device)
    expected = F.linear(inputs, matrix_reference(weight.core, list(weight.factors), 2))

    def reconstruction_forbidden(*args, **kwargs):
        raise AssertionError('forward must not reconstruct weights')

    monkeypatch.setattr(TuckerTensorized, 'to_matrix', reconstruction_forbidden)
    monkeypatch.setattr(TuckerTensorized, 'to_tensor', reconstruction_forbidden)
    # Core=24, input=8, output=12; the old outer product has 192 elements.
    with TensorSizeLimit(48) as observed:
        actual = factorized_linear(inputs, weight)
    torch.testing.assert_close(actual, expected, **DOUBLE_CLOSE)
    record_property('largest_operation_result_elements', observed.maximum)


def numerical_case(name, scale, device):
    """Make each operand once in float32, including representative unit-scale factors."""
    generator = torch.Generator().manual_seed(SEED + 6)
    if name == 'cancellation':
        core = torch.ones(2, 2, 2, dtype=torch.float32)
        factors = [torch.eye(2, dtype=torch.float32) for _ in range(3)]
        inputs = torch.tensor([[2.0**24, 1, -2.0**24, -1]], dtype=torch.float32) * scale
        bias = torch.zeros(2, dtype=torch.float32)
        output_shape, input_shape = (2,), (2, 2)
    else:
        output_shape, input_shape = (3, 2), (2, 3)
        ranks = (3, 2, 2, 3) if name == 'full' else (2, 2, 1, 2)
        factors = [torch.linalg.qr(torch.randn(s, r, generator=generator, dtype=torch.float32),
                                   mode='reduced')[0].contiguous()
                   for s, r in zip(output_shape + input_shape, ranks)]
        core = torch.randn(ranks, generator=generator, dtype=torch.float32)
        inputs = torch.randn(3, 6, generator=generator, dtype=torch.float32) * scale
        bias = torch.randn(6, generator=generator, dtype=torch.float32) * scale
    weight = TuckerTensorized(core.to(device), [f.to(device) for f in factors],
                               (output_shape, input_shape))
    return inputs.to(device), weight, bias.to(device)


def roundoff_bounds(inputs, weight, bias):
    """Use operation depth and absolute terms, never an observed error, to set bounds.

    gamma(K,u)=K*u/(1-K*u). K allows separate product/addition rounding.
    The float64 envelope is inflated by its own reconstruction/evaluation error.
    """
    output_modes = len(weight.tensorized_shape[0])
    core = weight.core.double()
    factors = [factor.double() for factor in weight.factors]
    data, offset = inputs.double(), bias.double()
    output_ranks = core.shape[:output_modes]
    input_ranks = core.shape[output_modes:]
    input_shape = weight.tensorized_shape[1]
    staged_k = 2 * (sum(input_shape) + prod(input_ranks) + sum(output_ranks)) + 1
    reference_k = 2 * (prod(output_ranks) + prod(input_ranks) + prod(input_shape)) + len(factors) - 1

    def gamma(k, u):
        return k * u / (1 - k * u)

    reference_rounding = gamma(reference_k, 2.0**-53)
    absolute_weight = matrix_reference(core.abs(), [factor.abs() for factor in factors], output_modes)
    envelope = F.linear(data.abs(), absolute_weight, offset.abs()) / (1 - reference_rounding)
    envelope = torch.nextafter(envelope, torch.full_like(envelope, float('inf')))
    reference = F.linear(data, matrix_reference(core, factors, output_modes), offset)
    bounds = [(gamma(k, 2.0**-24) + reference_rounding) * envelope
              for k in (staged_k, reference_k)]
    return reference, bounds, (staged_k, reference_k)


@pytest.mark.parametrize('name', ['full', 'reduced', 'cancellation'])
@pytest.mark.parametrize('scale', [2.0**-8, 1.0, 2.0**8])
def test_tucker_float32_roundoff(device, name, scale, record_property):
    inputs, weight, bias = numerical_case(name, scale, device)
    reference, bounds, depths = roundoff_bounds(inputs, weight, bias)
    output_modes = len(weight.tensorized_shape[0])
    staged64 = factorized_linear(inputs.double(), TuckerTensorized(
        weight.core.double(), [factor.double() for factor in weight.factors], weight.tensorized_shape),
        bias=bias.double())
    torch.testing.assert_close(staged64, reference, **DOUBLE_CLOSE)
    routes = [factorized_linear(inputs, weight, bias=bias),
              F.linear(inputs, matrix_reference(weight.core, list(weight.factors), output_modes), bias)]
    measurements = []
    for result, bound, depth in zip(routes, bounds, depths):
        error = (result.double() - reference).abs()
        assert torch.all(error <= bound), (error.max().item(), bound.max().item())
        measurements.append(dict(max_absolute_error=error.max().item(),
                                 max_error_bound=bound.max().item(), reduction_depth=depth))
        if name != 'cancellation':
            torch.testing.assert_close(result.double(), reference, **SINGLE_CLOSE)
    record_property('roundoff', json.dumps(dict(seed=SEED + 6, scale=scale,
                                               case=name, routes=measurements)))
