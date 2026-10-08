"""Extreme-scale measurements and state isolation for MPO compression."""

import math

import pytest
import torch

from .. import mpo_compression as mpo


MODES = (2, 2, 2, 2)


def target_matrix(dtype=torch.float32, scale=1.):
    return torch.randn(16, 16, dtype=dtype,
                       generator=torch.Generator().manual_seed(123)) * scale


def reference_norm(tensor):
    # An independent accumulation that remains safe when naive squares fail.
    return math.hypot(*tensor.detach().double().reshape(-1).tolist())


@pytest.mark.parametrize('dtype,scale', [
    (torch.float32, 1.), (torch.float32, 1e-25), (torch.float32, 1e20),
    (torch.float64, 1e-150), (torch.float64, 1e150),
])
@pytest.mark.parametrize('budget', [48, 96])
def test_budget_errors_match_independent_reference_across_scales(dtype, scale, budget):
    matrix = target_matrix(dtype, scale)
    original = matrix.clone()
    weight, info = mpo.blocktt_MALS(
        matrix, MODES, MODES, budget, max_sweeps_per_refinement=2,
    )
    residual = matrix.double() - weight.to_matrix().detach().double()
    absolute_error = reference_norm(residual)
    relative_error = absolute_error / reference_norm(matrix)
    assert math.isfinite(info['final_relative_weight_error'])
    assert info['final_relative_weight_error'] == pytest.approx(relative_error, rel=2e-14)
    assert info['final_squared_frobenius_error'] == pytest.approx(
        absolute_error * absolute_error, rel=3e-14, abs=0,
    )
    assert 0 < info['final_relative_weight_error'] < info['initial_relative_weight_error']
    assert info['final_squared_frobenius_error'] > 0
    assert info['termination_reason'] == 'parameter_budget_blocks_growth'
    if budget == 96:
        assert info['number_of_accepted_growth_steps'] > 0
    else:
        assert info['number_of_accepted_growth_steps'] == 0
    assert all(c.dtype == dtype for c in weight.factors)
    assert torch.equal(matrix, original)
    assert all(math.isfinite(row['score']) for row in info['history'][1:])


def test_operands_are_promoted_before_subtraction():
    positive = torch.tensor([[3e38]], dtype=torch.float32)
    negative = -positive
    assert torch.isinf(positive - negative).all()
    measured = mpo._measure_matrix(positive, negative)
    assert measured['relative_weight_error'] == pytest.approx(2.)
    assert measured['maximum_absolute_error'] > torch.finfo(torch.float32).max
    assert math.isfinite(measured['absolute_frobenius_error'])


@pytest.mark.parametrize('scale', [1e-300, 1e-200, 1e200, 1e300])
def test_robust_norms_when_intermediate_squares_are_unrepresentable(scale):
    values = target_matrix(torch.float64, scale)
    assert mpo._frobenius_norm(values) == pytest.approx(reference_norm(values), rel=3e-15, abs=0)


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_unrepresentable_required_squared_diagnostics_raise(scale):
    before = torch.random.get_rng_state().clone()
    with pytest.raises(FloatingPointError, match='Squared norm.*representable float64'):
        mpo.blocktt_MALS(target_matrix(torch.float64, scale), MODES, MODES, 96)
    assert torch.equal(before, torch.random.get_rng_state())


def test_tiny_discarded_tail_with_representable_global_error():
    matrix = torch.zeros(4, 2, dtype=torch.float64)
    matrix[0, 0], matrix[3, 0], matrix[3, 1] = 10, 1e-200, 1
    weight, info = mpo.blocktt_MALS(
        matrix, (1, 1, 2), (2, 2, 1), 6, initial_ranks=1,
    )
    stored = sum(core.numel() for core in weight.factors)
    assert stored == info['final_mpo_parameters'] == 6
    residual = matrix.double() - weight.to_matrix().detach().double()
    absolute_error = reference_norm(residual)
    assert absolute_error == pytest.approx(1., rel=1e-14)
    assert info['final_squared_frobenius_error'] == pytest.approx(
        absolute_error * absolute_error, rel=3e-14, abs=0,
    )
    assert info['final_relative_weight_error'] == pytest.approx(
        absolute_error / reference_norm(matrix), rel=2e-14,
    )


@pytest.mark.parametrize('max_sweeps', [1, None], ids=['one_sweep', 'default_sweeps'])
def test_compression_preserves_precision_inside_cpu_autocast(max_sweeps):
    matrix = torch.randn(6, 4, dtype=torch.float32,
                         generator=torch.Generator().manual_seed(42))
    # Two cores reduce to rank-2 SVD of this independently formed paired unfolding.
    unfolding = matrix.reshape(3, 2, 2, 2).permute(0, 2, 1, 3).reshape(6, 4)
    u, s, vh = torch.linalg.svd(unfolding, full_matrices=False)
    reference = (u[:, :2] * s[:2]) @ vh[:2]
    reference = reference.reshape(3, 2, 2, 2).permute(0, 2, 1, 3).reshape(6, 4)
    options = {} if max_sweeps is None else {'max_sweeps_per_refinement': max_sweeps}
    previous_autocast = torch.is_autocast_enabled('cpu')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        weight, info = mpo.blocktt_MALS(
            matrix, (2, 2), (3, 2), 20, initial_ranks=1, **options,
        )
        assert torch.is_autocast_enabled('cpu')
        assert torch.get_autocast_dtype('cpu') == torch.bfloat16
    assert torch.is_autocast_enabled('cpu') == previous_autocast
    assert all(core.dtype == matrix.dtype for core in weight.factors)
    reconstruction = weight.to_matrix().detach()
    torch.testing.assert_close(reconstruction, reference, rtol=2e-5, atol=2e-6)
    residual = matrix.double() - reconstruction.double()
    absolute_error = reference_norm(residual)
    assert info['final_squared_frobenius_error'] == pytest.approx(
        absolute_error * absolute_error, rel=3e-14, abs=0,
    )
    assert info['final_relative_weight_error'] == pytest.approx(
        absolute_error / reference_norm(matrix), rel=2e-14,
    )


def test_norm_ratio_and_score_range_checks():
    tiny = torch.tensor([math.ulp(0.)], dtype=torch.float64)
    assert mpo._frobenius_norm(tiny) == tiny.item() != 0
    with pytest.raises(FloatingPointError, match='Frobenius norm'):
        mpo._frobenius_norm(torch.full((4,), 1e308, dtype=torch.float64))
    for residual, reference in ((math.nan, 1.), (math.inf, math.inf),
                                (1e-300, 1e300), (1e300, 1e-300)):
        with pytest.raises(FloatingPointError):
            mpo._relative_error(residual, reference)
    with pytest.raises(FloatingPointError, match='measurement residual'):
        mpo._measurement_residual(torch.tensor([1e308], dtype=torch.float64),
                                  torch.tensor([-1e308], dtype=torch.float64))
    # The score is representable although separately squaring these is not.
    assert 0 < mpo._growth_score(1e160, math.nextafter(1e160, 0.), 1e10) < math.inf
    for before, after in ((1e-300, 0.), (math.nan, .1), (1e308, -1e308)):
        with pytest.raises(FloatingPointError):
            mpo._growth_score(before, after, 1)
    assert mpo._relative_error(0., 0.) == 0
    assert mpo._relative_error(1e-300, 0.) == math.inf


@pytest.mark.parametrize('seed', [11, 913])
def test_rng_state_future_draws_and_input_graph_are_preserved(seed):
    matrix = target_matrix().requires_grad_()
    original = matrix.detach().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        before = torch.random.get_rng_state().clone()
        control = torch.Generator().set_state(before)
        expected = torch.randn(20, generator=control)
        weight, _ = mpo.blocktt_MALS(
            matrix, MODES, MODES, 96, max_sweeps_per_refinement=1,
        )
        assert torch.equal(before, torch.random.get_rng_state())
        assert torch.equal(torch.randn(20), expected)
    assert torch.equal(matrix, original)
    assert all(c.is_leaf and c.requires_grad and c.data_ptr() != matrix.data_ptr() for c in weight.factors)
    weight.to_matrix().square().sum().backward()
    assert all(c.grad is not None for c in weight.factors)
    assert matrix.grad is None


def test_rng_and_input_restored_after_decomposition_failure(monkeypatch):
    matrix = target_matrix().requires_grad_()
    original = matrix.detach().clone()

    def failing_decomposition(tensor, **kwargs):
        torch.randn(20)
        tensor.add_(10)  # Even an upstream mutation must touch only our copy.
        raise RuntimeError('deliberate decomposition failure')

    monkeypatch.setattr(mpo, 'tensor_train_matrix', failing_decomposition)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='deliberate decomposition failure'):
        mpo.blocktt_MALS(matrix, MODES, MODES, 96)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(matrix, original) and matrix.grad is None


def test_nonfinite_core_arithmetic_raises(monkeypatch):
    split = mpo._split_fixed_rank_pair

    def nonfinite_split(*args):
        first, second = split(*args)
        return first * math.inf, second

    monkeypatch.setattr(mpo, '_split_fixed_rank_pair', nonfinite_split)
    with pytest.raises(FloatingPointError, match='Nonfinite measurement'):
        mpo.blocktt_MALS(target_matrix(), MODES, MODES, 48)
