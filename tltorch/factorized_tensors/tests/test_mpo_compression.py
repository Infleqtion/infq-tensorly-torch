"""Mathematical and public-contract checks for budget-driven MPO compression."""

import math

import pytest
import torch

from ... import BlockTT, FactorizedLinear, compress_matrix_to_mpo_budget
from .. import mpo_compression as mpo


MODES = (2, 2, 2, 2)


def random_matrix(rows=16, columns=16, dtype=torch.float64, seed=42):
    return torch.randn(rows, columns, dtype=dtype,
                       generator=torch.Generator().manual_seed(seed))


def compress(matrix, budget, **kwargs):
    return compress_matrix_to_mpo_budget(matrix, MODES, MODES, budget, **kwargs)


@pytest.mark.parametrize('dtype,atol', [(torch.float64, 1e-12), (torch.float32, 2e-5)])
@pytest.mark.parametrize('pair', [0, 1, 2])
def test_canonicalization_preserves_tensor_and_orthonormal_environments(dtype, atol, pair):
    cores = mpo._initialize_cores(random_matrix(dtype=dtype), MODES, MODES, 3)
    reference = mpo._blocktt_from_cores(cores, MODES, MODES).to_matrix().detach()
    # An invertible diagonal gauge leaves the tensor unchanged but destroys
    # canonical form. Check the resulting exterior bases independently.
    gauge = torch.tensor([.1, 2., 5.], dtype=dtype)
    cores[1] = cores[1] * gauge
    cores[2] = cores[2] / gauge[:, None, None]
    shapes = [core.shape for core in cores]
    mpo._canonicalize_around_pair(cores, pair)
    torch.testing.assert_close(mpo._paired_cores_to_matrix(cores, MODES, MODES),
                               reference, rtol=atol, atol=atol)
    assert [core.shape for core in cores] == shapes
    left, right = mpo._pair_environments(cores, pair)
    assert left.shape == (4 ** pair, cores[pair].shape[0])
    assert right.shape == (cores[pair + 1].shape[-1], 4 ** (2 - pair))
    torch.testing.assert_close(left.T @ left, torch.eye(left.shape[1], dtype=dtype),
                               rtol=atol, atol=atol)
    torch.testing.assert_close(right @ right.T, torch.eye(right.shape[0], dtype=dtype),
                               rtol=atol, atol=atol)


def test_projection_orthogonality_and_excess_loss_identity():
    matrix = random_matrix(seed=913)
    cores = mpo._initialize_cores(random_matrix(seed=914), MODES, MODES, 2)
    mpo._canonicalize_around_pair(cores, 1)
    left, right = mpo._pair_environments(cores, 1)
    target = matrix.reshape((2,) * 8).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(4, 4, 4, 4)
    projected = mpo._project_target_pair(target, 1, left, right)
    reconstructed = torch.einsum('la,apqb,bt->lpqt', left, projected, right)
    residual = target - reconstructed
    normal_residual = torch.einsum('la,lpqt,bt->apqb', left, residual, right)
    torch.testing.assert_close(normal_residual, torch.zeros_like(normal_residual),
                               rtol=0, atol=1e-12)
    perturbation = random_matrix(8, 8, seed=915).reshape(projected.shape)
    changed = torch.einsum('la,apqb,bt->lpqt', left, projected + perturbation, right)
    excess_loss = (target - changed).square().sum() - residual.square().sum()
    torch.testing.assert_close(excess_loss, perturbation.square().sum(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('direction', ['left_to_right', 'right_to_left'])
def test_two_core_refinement_equals_truncated_svd(direction):
    matrix = random_matrix(6, 4)
    inputs, outputs = (2, 2), (3, 2)
    # This paired unfolding is not the original matrix's ordinary SVD.
    unfolding = matrix.reshape(3, 2, 2, 2).permute(0, 2, 1, 3).reshape(6, 4)
    u, s, vh = torch.linalg.svd(unfolding, full_matrices=False)
    reference = (u[:, :2] * s[:2]) @ vh[:2]
    first, second, discarded = mpo._split_fixed_rank_pair(
        unfolding.reshape(1, 6, 4, 1), 2, direction,
    )
    torch.testing.assert_close(torch.einsum('apc,cqb->apqb', first, second).reshape(6, 4),
                               reference, rtol=1e-12, atol=1e-12)
    assert discarded == pytest.approx(s[2:].square().sum().item())
    # Starting from a different target detects merely merging/splitting cores
    # instead of projecting the original matrix.
    initial = mpo._initialize_cores(random_matrix(6, 4, seed=812), inputs, outputs, 2)
    refined, _ = mpo._refine_two_site(matrix, initial, inputs, outputs, 2, 1e-10)
    dense_reference = reference.reshape(3, 2, 2, 2).permute(0, 2, 1, 3).reshape(6, 4)
    torch.testing.assert_close(mpo._paired_cores_to_matrix(refined, inputs, outputs),
                               dense_reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('rank', [2, 3])
def test_two_core_budget_growth_matches_svd_optimum(rank):
    matrix = random_matrix(6, 4)
    unfolding = matrix.reshape(3, 2, 2, 2).permute(0, 2, 1, 3).reshape(6, 4)
    u, s, vh = torch.linalg.svd(unfolding, full_matrices=False)
    reference = ((u[:, :rank] * s[:rank]) @ vh[:rank])
    reference = reference.reshape(3, 2, 2, 2).permute(0, 2, 1, 3).reshape(6, 4)
    weight, info = compress_matrix_to_mpo_budget(
        matrix, (2, 2), (3, 2), 10 * rank, initial_ranks=1,
    )
    torch.testing.assert_close(weight.to_matrix(), reference, rtol=1e-12, atol=1e-12)
    assert weight.rank == (1, rank, 1)
    assert info['final_squared_frobenius_error'] == pytest.approx(s[rank:].square().sum().item())
    assert info['history'][-1]['error_after_refinement'] is not None


def test_full_outer_bonds_match_middle_unfolding_optimum():
    matrix = random_matrix(seed=19)
    _, info = compress(matrix, 224, initial_ranks=(1, 4, 6, 4, 1))
    middle = matrix.reshape((2,) * 8).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(16, 16)
    tail = torch.linalg.svdvals(middle)[6:]
    optimum = (tail.norm() / matrix.norm()).item()
    assert info['initial_relative_weight_error'] == pytest.approx(optimum, abs=1e-14)
    assert info['final_relative_weight_error'] == pytest.approx(optimum, abs=1e-14)


@pytest.mark.parametrize('budget,initial_ranks', [(16, 1), (24, 1), (48, 2), (59, 2),
                                                (60, 2), (96, 2), (224, 2), (544, 16)])
def test_storage_errors_histories_and_work_counts(budget, initial_ranks):
    matrix = random_matrix()
    weight, info = compress(matrix, budget, initial_ranks=initial_ranks,
                            max_sweeps_per_refinement=1, sweep_tolerance=0)
    assert isinstance(weight, BlockTT)
    assert compress_matrix_to_mpo_budget is mpo.compress_matrix_to_mpo_budget
    ranks = (weight.factors[0].shape[0],) + tuple(c.shape[-1] for c in weight.factors)
    cost = sum(core.numel() for core in weight.factors)
    assert weight.rank == info['final_rank_profile'] == ranks
    assert cost == sum(ranks[k] * 4 * ranks[k + 1] for k in range(4))
    assert cost == info['final_mpo_parameters'] <= budget
    assert info['unused_budget'] == budget - cost
    assert info['dense_parameters'] == weight.numel() == 256
    residual = weight.to_matrix().detach() - matrix
    assert info['final_squared_frobenius_error'] == pytest.approx(residual.square().sum().item())
    assert info['final_relative_weight_error'] == pytest.approx((residual.norm() / matrix.norm()).item())
    assert info['final_squared_frobenius_error'] <= info['initial_squared_frobenius_error']
    assert info['number_of_accepted_growth_steps'] == len(info['history']) - 1
    assert info['number_of_refinement_calls'] == len(info['refinement_history'])
    assert info['refinement_pair_svd_count'] == 6 * info['completed_refinement_sweeps']
    assert info['growth_pair_svd_count'] == info['number_of_candidate_evaluations']
    assert info['full_target_decomposition_count'] == 1
    assert 'candidate_history' not in info and 'refinement_update_history' not in info
    for previous, state in zip(info['history'], info['history'][1:]):
        bond = state['changed_bond']
        expected = list(previous['rank_profile'])
        expected[bond] += 1
        assert state['rank_profile'] == tuple(expected)
        # Both adjacent core slices contribute to actual added storage.
        added = previous['rank_profile'][bond - 1] * 4 + 4 * previous['rank_profile'][bond + 1]
        assert state['added_parameters'] == added
        assert state['mpo_parameters'] - previous['mpo_parameters'] == added
        assert state['error_before_growth'] == previous['relative_weight_error']
        assert state['error_after_growth'] == state['error_before_refinement']
        assert state['relative_weight_error'] == state['error_after_refinement']
        assert state['score'] == pytest.approx(
            (state['error_before_growth'] ** 2 - state['error_after_growth'] ** 2) / added,
        )
    assert info['history'][-1]['relative_weight_error'] == info['final_relative_weight_error']
    for state, call in zip(info['history'], info['refinement_history']):
        assert call['growth_step'] == state['growth_step']
        assert call['refinement_call'] == state['refinement_call']
        assert call['error_before'] == state['error_before_refinement']
        assert call['error_after'] == state['error_after_refinement']
        assert call['error_after'] <= call['error_before']
    if initial_ranks == 16:
        assert info['initial_rank_profile'] == (1, 4, 16, 4, 1)
        assert cost == 544 != weight.numel()


def test_independent_candidates_selection_and_single_initialization(monkeypatch):
    matrix = random_matrix()
    initialize = mpo.tensor_train_matrix
    evaluate = mpo._evaluate_growth
    initializations = []
    evaluations = []

    def counted_initialize(*args, **kwargs):
        initializations.append(1)
        return initialize(*args, **kwargs)

    def audited_evaluate(target, current, paired, inputs, outputs, pair, allowance):
        before = [core.clone() for core in current]
        result = evaluate(target, current, paired, inputs, outputs, pair, allowance)
        candidate, metrics, threshold = result
        assert sum(core.numel() for core in candidate) <= 96
        for old, source, new in zip(before, current, candidate):
            torch.testing.assert_close(old, source, rtol=0, atol=0)
            assert source.data_ptr() != new.data_ptr()
        evaluations.append((current, pair, metrics, threshold))
        return result

    monkeypatch.setattr(mpo, 'tensor_train_matrix', counted_initialize)
    monkeypatch.setattr(mpo, '_evaluate_growth', audited_evaluate)
    _, info = compress(matrix, 96, max_sweeps_per_refinement=1)
    assert len(initializations) == 1
    assert len(evaluations) == info['number_of_candidate_evaluations']
    current = evaluations[0][0]
    first_round = [row for row in evaluations if row[0] is current]
    assert [row[1] for row in first_round] == [0, 1, 2]
    paired = mpo._matrix_to_paired_tensor(matrix, MODES, MODES)
    # Reversing evaluation order must not change any candidate's result.
    for state, pair, metrics, threshold in reversed(first_round):
        _, repeated, _ = evaluate(matrix, state, paired, MODES, MODES, pair, info['roundoff_factor'])
        assert repeated['relative_weight_error'] == pytest.approx(metrics['relative_weight_error'], abs=1e-14)
    for accepted in info['history'][1:]:
        state = next(row[0] for row in evaluations
                     if mpo._rank_profile(row[0]) == info['history'][accepted['growth_step'] - 1]['rank_profile'])
        before = accepted['error_before_growth']
        cost = sum(core.numel() for core in state)
        eligible = [row for row in evaluations if row[0] is state
                    and before - row[2]['relative_weight_error'] > row[3]]
        winner = max(eligible, key=lambda row: (
            (before ** 2 - row[2]['relative_weight_error'] ** 2) / (row[2]['mpo_parameters'] - cost),
            -row[2]['relative_weight_error'], -row[2]['mpo_parameters'], -row[1],
        ))
        assert accepted['changed_bond'] == winner[1] + 1


def test_candidate_tie_order():
    # Explicit contractual ties, independent of SVD roundoff in actual searches.
    candidates = [dict(score=1., relative_weight_error=.5, actual_mpo_parameters=20, bond=2)]
    for changes in ({'bond': 1}, {'actual_mpo_parameters': 19},
                    {'relative_weight_error': .4}, {'score': 2.}):
        candidates.append({**candidates[-1], **changes})
    assert sorted(reversed(candidates), key=mpo._candidate_order) == candidates


def test_budget_blocked_state_is_refined_and_caps_do_not_end_growth():
    for budget in (48, 59, 60, 96):
        _, info = compress(random_matrix(), budget, max_sweeps_per_refinement=1, sweep_tolerance=0)
        assert info['termination_reason'] == 'parameter_budget_blocks_growth'
        assert info['refinement_calls_at_sweep_limit'] > 0
        assert len(info['refinement_history']) == len(info['history'])
        assert info['final_squared_frobenius_error'] < info['initial_squared_frobenius_error']
        if budget < 60:
            assert info['number_of_candidate_evaluations'] == 0
            assert info['number_of_refinement_calls'] == 1
        else:
            assert info['number_of_accepted_growth_steps'] > 0


def test_refinement_keeps_initial_best_despite_allowed_roundoff(monkeypatch):
    matrix = torch.zeros(16, 16, dtype=torch.float64)
    matrix[0, 0] = 1
    initial = mpo._initialize_cores(matrix, MODES, MODES, 1)
    originals = [core.clone() for core in initial]
    split = mpo._split_fixed_rank_pair

    def slightly_worse(*args):
        first, second, discarded = split(*args)
        return first * (1 + 1e-14), second, discarded

    monkeypatch.setattr(mpo, '_split_fixed_rank_pair', slightly_worse)
    refined, info = mpo._refine_two_site(matrix, initial, MODES, MODES, 2, 0)
    torch.testing.assert_close(mpo._paired_cores_to_matrix(refined, MODES, MODES), matrix, rtol=0, atol=0)
    for before, source, returned in zip(originals, initial, refined):
        torch.testing.assert_close(before, source, rtol=0, atol=0)
        assert source.data_ptr() != returned.data_ptr()
    assert info['termination_reason'] == 'sweep_improvement_small'


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rectangular_noncontiguous_trainable_output_and_wrapper(dtype):
    matrix = random_matrix(8, 12, dtype=dtype).T.requires_grad_()
    assert not matrix.is_contiguous()
    original = matrix.detach().clone()
    inputs, outputs = (2, 2, 2), (3, 2, 2)
    kwargs = dict(initial_ranks=1, max_sweeps_per_refinement=1)
    first, info = compress_matrix_to_mpo_budget(matrix, inputs, outputs, 40, **kwargs)
    second, repeated = compress_matrix_to_mpo_budget(matrix, inputs, outputs, 40, **kwargs)
    assert first.rank == second.rank and info['termination_reason'] == repeated['termination_reason']
    torch.testing.assert_close(first.to_matrix(), second.to_matrix())
    torch.testing.assert_close(matrix, original, rtol=0, atol=0)
    assert first.to_matrix().shape == matrix.shape
    assert info['minimum_mpo_parameters'] == 14
    assert list(first.parameters()) == list(first.factors)
    for a, b in zip(first.factors, second.factors):
        assert isinstance(a, torch.nn.Parameter) and a.is_leaf and a.requires_grad
        assert a.dtype == dtype and a.device.type == 'cpu'
        assert a.data_ptr() != b.data_ptr() and a.data_ptr() != matrix.data_ptr()
    state = torch.random.get_rng_state().clone()
    layer = FactorizedLinear(inputs, outputs, factorization=first, bias=False)
    assert layer.weight is first and torch.equal(state, torch.random.get_rng_state())
    batch = random_matrix(5, 8, dtype=dtype, seed=8)
    torch.testing.assert_close(layer(batch), batch @ first.to_matrix().T)
    layer(batch).square().sum().backward()
    assert matrix.grad is None
    assert all(c.grad is not None and torch.isfinite(c.grad).all() for c in first.factors)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('nonzero', [False, True])
def test_zero_and_exact_initialization_skip_refinement_and_growth(dtype, nonzero, monkeypatch):
    matrix = torch.zeros(16, 16, dtype=dtype)
    if nonzero:
        matrix[0, 0] = 1e-25

    def forbidden(*args, **kwargs):
        raise AssertionError('Exact initialization requires no refinement or growth.')

    monkeypatch.setattr(mpo, '_refine_two_site', forbidden)
    monkeypatch.setattr(mpo, '_choose_growth', forbidden)
    weight, info = compress(matrix, 16, initial_ranks=1)
    torch.testing.assert_close(weight.to_matrix(), matrix, rtol=0, atol=0)
    assert info['termination_reason'] == 'exact_reconstruction'
    assert info['final_relative_weight_error'] == info['final_squared_frobenius_error'] == 0
    assert info['number_of_refinement_calls'] == info['number_of_candidate_evaluations'] == 0
    assert info['history'][0]['error_after_refinement'] is None


def test_invalid_initial_storage_and_dimensional_redundancy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid initialization must not be refined.')

    monkeypatch.setattr(mpo, '_refine_two_site', forbidden)
    with pytest.raises(ValueError, match=r'48.*budget 47.*smaller initial_ranks'):
        compress(random_matrix(), 47)
    with pytest.raises(ValueError, match='dimensional redundancy'):
        compress(random_matrix(), 500, initial_ranks=(1, 4, 16, 1, 1))
    monkeypatch.setattr(mpo, 'tensor_train_matrix', forbidden)
    with pytest.raises(ValueError, match='minimum stored-core cost 16'):
        compress(random_matrix(), 15, initial_ranks=1)


def test_growth_only_stagnation_does_not_prove_infeasibility():
    matrix = torch.zeros(16, 16, dtype=torch.float64)
    matrix[0, 0], matrix[-1, -1] = 1., .5
    _, stuck = compress(matrix, 544, initial_ranks=1)
    _, better = compress(matrix, 48, initial_ranks=2)
    assert stuck['termination_reason'] == 'no_numerically_improving_candidate'
    assert stuck['final_relative_weight_error'] > .1
    assert better['final_relative_weight_error'] < 1e-12


def test_roundoff_threshold_is_not_an_accuracy_target():
    paired = torch.diag(torch.tensor([1., 2e-6, 0., 0.], dtype=torch.float32))
    matrix = paired.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(4, 4)
    _, info = compress_matrix_to_mpo_budget(matrix, (2, 2), (2, 2), 16, initial_ranks=1)
    assert info['termination_reason'] == 'no_numerically_improving_candidate'
    assert info['final_relative_weight_error'] > 0
    assert info['number_of_candidate_evaluations'] == 1
    assert info['number_of_accepted_growth_steps'] == 0
    assert 'requested_relative_tolerance' not in info
    _, saturated = compress(random_matrix(dtype=torch.float32), 544, initial_ranks=16)
    assert saturated['termination_reason'] == 'local_rank_limits_reached'
    assert saturated['final_relative_weight_error'] > 0


@pytest.mark.parametrize('corruption', ['canonicalization', 'refinement', 'growth'])
def test_material_numerical_errors_are_not_silently_accepted(corruption, monkeypatch):
    if corruption == 'canonicalization':
        def corrupt(cores, pair):
            cores[0] = cores[0] * 2
        monkeypatch.setattr(mpo, '_canonicalize_around_pair', corrupt)
        message = 'Canonicalization changed the tensor'
    else:
        split = mpo._split_fixed_rank_pair
        def corrupt(projected, rank, direction):
            first, second, energy = split(projected, rank, direction)
            if corruption == 'refinement' or rank == 3:
                first = first * 100
            return first, second, energy
        monkeypatch.setattr(mpo, '_split_fixed_rank_pair', corrupt)
        message = 'error increased'
    with pytest.raises(RuntimeError, match=message):
        compress(random_matrix(), 96, max_sweeps_per_refinement=1)


@pytest.mark.parametrize('kwargs,error', [
    ({'max_parameters': True}, TypeError), ({'max_parameters': 48.5}, TypeError),
    ({'max_parameters': 0}, ValueError), ({'max_parameters': -1}, ValueError),
    ({'initial_ranks': True}, TypeError), ({'initial_ranks': 0}, ValueError),
    ({'initial_ranks': 2.5}, TypeError), ({'initial_ranks': 'same'}, TypeError),
    ({'initial_ranks': (1, 2, 1)}, ValueError),
    ({'initial_ranks': (2, 2, 2, 2, 1)}, ValueError),
    ({'initial_ranks': (1, 2, 2, 2, 2)}, ValueError),
    ({'initial_ranks': (1, True, 2, 2, 1)}, TypeError),
    ({'initial_ranks': (1, 0, 2, 2, 1)}, ValueError),
    ({'max_sweeps_per_refinement': True}, TypeError),
    ({'max_sweeps_per_refinement': 1.5}, TypeError),
    ({'max_sweeps_per_refinement': 0}, ValueError),
    ({'sweep_tolerance': True}, TypeError), ({'sweep_tolerance': '0'}, TypeError),
    ({'sweep_tolerance': -1}, ValueError), ({'sweep_tolerance': math.nan}, ValueError),
    ({'sweep_tolerance': math.inf}, ValueError),
])
def test_invalid_controls_even_for_zero_matrix(kwargs, error):
    with pytest.raises(error):
        compress_matrix_to_mpo_budget(torch.zeros(16, 16), MODES, MODES,
                                      **{'max_parameters': 48, **kwargs})


@pytest.mark.parametrize('inputs,outputs,error', [
    ((4, 4), (2, 2, 2, 2), ValueError), ((16,), (16,), ValueError),
    ((2, 2, 2), (2, 2, 4), ValueError), ((2, 2, 4), (2, 2, 2), ValueError),
    ((2, 0, 2, 2), MODES, ValueError), ((2, True, 2, 2), MODES, TypeError),
    ((2, 2., 2, 2), MODES, TypeError), ('2222', MODES, TypeError),
])
def test_invalid_tensorizations(inputs, outputs, error):
    with pytest.raises(error):
        compress_matrix_to_mpo_budget(random_matrix(), inputs, outputs, 48)


@pytest.mark.parametrize('matrix,error', [
    ([[1.]], TypeError), (torch.zeros(0, 16), ValueError),
    (torch.zeros(2, 2, 2), ValueError), (torch.zeros(16, 16, dtype=torch.int64), TypeError),
    (torch.zeros(16, 16, dtype=torch.float16), TypeError),
    (torch.zeros(16, 16, dtype=torch.bfloat16), TypeError),
    (torch.zeros(16, 16, dtype=torch.complex128), TypeError),
    (torch.full((16, 16), math.nan), ValueError),
    (torch.full((16, 16), math.inf), ValueError),
    (torch.empty(16, 16, device='meta'), ValueError),
    (torch.eye(16).to_sparse(), ValueError),
])
def test_unsupported_matrices(matrix, error):
    with pytest.raises(error):
        compress(matrix, 48)
