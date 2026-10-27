"""Storage budgets select feasible ranks before allocation or decomposition."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tltorch import FactorizedConv, FactorizedLinear
from tltorch.factorized_tensors import FactorizedTensor, TTTensor
from tltorch.factorized_tensors.tensorized_matrices import BlockTT
from tltorch.factorized_tensors.tt_rank import resolve_tt_rank, tt_parameter_count
from tltorch.factorized_layers.factorized_convolution import tensor_to_kernel


def assert_feasible(shape, ranks):
    assert ranks[0] == ranks[-1] == 1
    for i in range(1, len(shape)):
        assert 1 <= ranks[i] <= min(math.prod(shape[:i]), math.prod(shape[i:]))
    for i, size in enumerate(shape):
        assert ranks[i + 1] <= ranks[i] * size
        assert ranks[i] <= size * ranks[i + 1]


@pytest.mark.parametrize('shape', [(512, 3, 3, 512), (4, 3, 2, 5),
                                   (64, 3, 2, 128), (16, 5, 2, 32), (1, 4, 1, 5), (4, 5)])
def test_deterministic_feasible_budget(shape):
    rng = torch.get_rng_state().clone()
    ranks = resolve_tt_rank(shape, .75)
    assert ranks == resolve_tt_rank(shape, .75)
    assert torch.equal(rng, torch.get_rng_state())
    assert_feasible(shape, ranks)
    assert tt_parameter_count(shape, ranks) <= math.floor(.75 * math.prod(shape))


def test_known_counts_and_redistribution():
    shape = (4, 3, 2, 5)
    assert tt_parameter_count(shape, (1, 3, 3, 3, 1)) == 72
    assert resolve_tt_rank(shape, .75) == (1, 4, 4, 2, 1)
    assert tt_parameter_count(shape, resolve_tt_rank(shape, .75)) == 16 + 48 + 16 + 10 == 90
    large = (512, 3, 3, 512)
    ranks = resolve_tt_rank(large, .75)
    assert ranks == (1, 464, 464, 465, 1)
    assert tt_parameter_count(large, ranks) == 1768816 <= 1769472


def test_rounding_same_saturation_and_single_core():
    # A 4x5 matrix costs 9*r. floor(.74*20)=14 permits only rank one.
    assert resolve_tt_rank((4, 5), .74) == (1, 1, 1)
    assert resolve_tt_rank((4, 5), 'same') == resolve_tt_rank((4, 5), 1.0) == (1, 2, 1)
    assert resolve_tt_rank((4, 5), 100.0) == (1, 4, 1)
    assert resolve_tt_rank((4, 5), 1e308) == (1, 4, 1)
    assert resolve_tt_rank((7,), 'same') == resolve_tt_rank((7,), 2.0) == (1, 1)
    vector = torch.arange(7, dtype=torch.float64)
    direct = TTTensor.from_tensor(vector, 'same')
    allocated = TTTensor.new((7,), 'same', dtype=vector.dtype).init_from_tensor(vector)
    for factor in [direct, allocated]:
        torch.testing.assert_close(factor.to_tensor(), vector)
    with pytest.raises(ValueError, match='budget 5.*rank-one.*7'):
        resolve_tt_rank((7,), .75)
    with pytest.raises(ValueError, match='budget 3.*rank-one.*6'):
        resolve_tt_rank((2, 1, 1, 2), .75)


@pytest.mark.parametrize('method', ['from_tensor', 'init_from_tensor'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('contiguous', [True, False], ids=['contiguous', 'strided-view'])
@pytest.mark.parametrize('mutate', ['core', 'source'])
def test_single_core_storage_independence(method, dtype, contiguous, mutate):
    from tltorch.factorized_tensors.complex_factorized_tensors import ComplexTTTensor

    values = torch.arange(12, dtype=torch.float64).to(dtype)
    if values.is_complex():
        values = values + 0.25j * values
    source = values[:6] if contiguous else values[::2]
    assert source.is_contiguous() == contiguous
    original = source.clone()
    tensor_type = ComplexTTTensor if source.is_complex() else TTTensor
    if method == 'from_tensor':
        factor = tensor_type.from_tensor(source, 'same')
    else:
        factor = tensor_type.new(source.shape, 'same', dtype=dtype, device=source.device)
        factor.init_from_tensor(source)

    core = factor.factors[0]
    assert tuple(factor.rank) == (1, 1)
    assert tuple(factor.shape) == tuple(source.shape)
    assert tuple(core.shape) == (1, 6, 1)
    assert core.dtype == dtype and core.device == source.device
    assert torch.equal(factor.to_tensor(), original)

    # Separate cases exercise both mutation directions even before the fix.
    with torch.no_grad():
        if mutate == 'core':
            core.add_(2)
        else:
            source.add_(2)
    if mutate == 'core':
        assert torch.equal(source, original), 'Stored core shares source storage'
        assert torch.equal(factor.to_tensor(), original + 2)
    else:
        assert torch.equal(factor.to_tensor(), original), 'Source shares stored core storage'
        assert torch.equal(source, original + 2)


@pytest.mark.parametrize('target', [0.0, -0.5, float('inf'), float('-inf'), float('nan')])
def test_invalid_targets(target):
    with pytest.raises(ValueError, match='finite positive float'):
        resolve_tt_rank((4, 5), target)


def test_explicit_inputs_keep_tensorly_rank_cap_semantics():
    ranks = [1, 12, 12, 1]
    tensor = torch.randn(4, 3, 5)
    factor = TTTensor.from_tensor(tensor, ranks)
    assert ranks == [1, 12, 12, 1]
    assert tuple(factor.rank) == (1, 4, 5, 1)
    assert resolve_tt_rank(tensor.shape, 12) == tuple(ranks)
    assert resolve_tt_rank(tensor.shape, tuple(ranks)) == tuple(ranks)


def test_tt_entry_points_agree_and_survive_svd():
    torch.manual_seed(4)
    tensor = torch.randn(4, 3, 2, 5, dtype=torch.float64)
    expected = resolve_tt_rank(tensor.shape, .75)
    allocated = TTTensor.new(tensor.shape, .75, dtype=tensor.dtype)
    decomposed = TTTensor.from_tensor(tensor, .75)
    allocated.init_from_tensor(tensor)
    direct = TTTensor(list(allocated.factors), shape=tensor.shape, rank=.75).init_from_tensor(tensor)
    for factor in [allocated, decomposed, direct]:
        assert tuple(factor.rank) == expected
        assert sum(core.numel() for core in factor.factors) == 90
        assert tuple(core.shape[-1] for core in factor.factors) == expected[1:]


def test_blocktt_paths_and_batched_weight_budget():
    torch.manual_seed(5)
    tensorized = ((2, 3), (2, 5))  # merged modes (4,15), original 60 elements
    tensor = torch.randn(2, 3, 2, 5, dtype=torch.float64)
    expected = resolve_tt_rank((4, 15), .75)
    allocated = BlockTT.new(tensorized, .75, dtype=tensor.dtype)
    decomposed = BlockTT.from_tensor(tensor, tensorized, .75)
    allocated.init_from_tensor(tensor)
    from tltorch.factorized_tensors.core import TensorizedTensor
    matrix_factor = TensorizedTensor.from_matrix(tensor.reshape(6, 10), (2, 3), (2, 5),
                                                 rank=.75, factorization='BlockTT')
    direct = BlockTT(list(allocated.factors), tensorized, rank=.75).init_from_tensor(tensor)
    for factor in [allocated, decomposed, matrix_factor, direct]:
        assert tuple(factor.rank) == expected
        assert sum(p.numel() for p in factor.parameters()) == 38
    explicit = [1, 20, 1]
    factor = BlockTT.from_tensor(tensor, tensorized, explicit)
    assert explicit == [1, 20, 1] and tuple(factor.rank) == (1, 4, 1)
    batched = BlockTT.new((4, (4, 3, 2), (5, 3, 2)), .5)
    assert sum(p.numel() for p in batched.parameters()) <= .5 * (4 * 24 * 30)
    single = BlockTT.new(((3,), (4,)), 'same')
    assert tuple(single.rank) == (1, 1) and sum(p.numel() for p in single.parameters()) == 12
    direct_single = BlockTT.from_tensor(torch.randn(3, 4), ((3,), (4,)), 'same')
    assert tuple(direct_single.rank) == (1, 1)
    with pytest.raises(ValueError, match='single.*core.*rank'):
        BlockTT.new(((3,), (4,)), .5)


@pytest.mark.parametrize('dtype,rtol,atol', [(torch.float64, 1e-10, 1e-12),
                                           (torch.float32, 1e-5, 1e-6)])
def test_public_conversions_execute_reconstructed_weight(dtype, rtol, atol):
    # Small contractions justify these fixed tolerances; this does not override
    # the previously failed 512-channel float32 diagnostic or its tolerance.
    torch.manual_seed(6)
    conv = nn.Conv2d(4, 5, (3, 2), bias=False, padding=(1, 0)).to(dtype)
    fact = FactorizedConv.from_conv(conv, rank=.75, factorization='TT', implementation='factorized')
    expected_rank = resolve_tt_rank((4, 3, 2, 5), .75)
    assert tuple(fact.weight.rank) == tuple(fact.rank) == expected_rank
    assert sum(p.numel() for p in fact.weight.parameters()) == 90
    x = torch.randn(2, 4, 5, 6, dtype=dtype)
    kernel = tensor_to_kernel('TT', fact.weight.to_tensor())
    expected = F.conv2d(x, kernel, padding=conv.padding, stride=conv.stride, dilation=conv.dilation)
    torch.testing.assert_close(fact(x), expected, rtol=rtol, atol=atol)
    linear = nn.Linear(6, 10, bias=False).to(dtype)
    fact_linear = FactorizedLinear.from_linear(linear, rank=.75, factorization='BlockTT',
        auto_tensorize=False, in_tensorized_features=(2, 3), out_tensorized_features=(2, 5),
        bias=False, implementation='factorized')
    assert tuple(fact_linear.rank) == tuple(fact_linear.weight.rank) == (1, 2, 1)
    x = torch.randn(3, 6, dtype=dtype)
    torch.testing.assert_close(fact_linear(x), F.linear(x, fact_linear.weight.to_matrix()), rtol=rtol, atol=atol)


@pytest.mark.parametrize('factorization', ['ComplexTT'])
def test_complex_inherits_shared_policy(factorization):
    tensor = FactorizedTensor.new((4, 3, 2, 5), rank=.75, factorization=factorization)
    tensor.normal_()
    assert tuple(tensor.rank) == (1, 4, 4, 2, 1)
    assert tensor.to_tensor().dtype == torch.complex64
    # Complex handlers store real/imaginary components separately; ranks/counts
    # describe complex elements, rather than doubling the weight-budget units.
    assert sum(math.prod(core.shape) for core in tensor.factors) == 90


def test_complex_blocktt_entry_points():
    from tltorch.factorized_tensors.complex_tensorized_matrices import ComplexBlockTT
    tensor = torch.randn(2, 3, 2, 5, dtype=torch.complex128)
    factor = ComplexBlockTT.from_tensor(tensor, ((2, 3), (2, 5)), .75)
    allocated = ComplexBlockTT.new(((2, 3), (2, 5)), .75, dtype=tensor.dtype)
    allocated.init_from_tensor(tensor)
    assert tuple(factor.rank) == tuple(allocated.rank) == (1, 2, 1)
    torch.testing.assert_close(factor.to_tensor(), allocated.to_tensor())
