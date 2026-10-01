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
    weight, info = mpo.compress_matrix_to_mpo_budget(
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
        mpo.compress_matrix_to_mpo_budget(target_matrix(torch.float64, scale), MODES, MODES, 96)
    assert torch.equal(before, torch.random.get_rng_state())


def test_discarded_local_energy_retains_its_range_check():
    projected = torch.diag(torch.tensor([1., 1e-200], dtype=torch.float64)).reshape(1, 2, 2, 1)
    with pytest.raises(FloatingPointError, match='Squared norm'):
        mpo._split_fixed_rank_pair(projected, 1, 'left_to_right')


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
        weight, _ = mpo.compress_matrix_to_mpo_budget(
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
        mpo.compress_matrix_to_mpo_budget(matrix, MODES, MODES, 96)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(matrix, original) and matrix.grad is None


def test_nonfinite_core_arithmetic_raises(monkeypatch):
    split = mpo._split_fixed_rank_pair

    def nonfinite_split(*args):
        first, second, discarded = split(*args)
        return first * math.inf, second, discarded

    monkeypatch.setattr(mpo, '_split_fixed_rank_pair', nonfinite_split)
    with pytest.raises(FloatingPointError, match='Nonfinite measurement'):
        mpo.compress_matrix_to_mpo_budget(target_matrix(), MODES, MODES, 48)
