# Budget-driven MPO matrix compression

[Issue #1](https://github.com/Infleqtion/infq-tensorly-torch/issues/1) asks for a
rank-adaptive matrix product operator (MPO) compression. This PR attempts to implement
that by approximating a dense matrix while respecting an upper bound on stored core
entries. The motivation behind this is to apply this method on a trained
neural-network weight matrix. The returned `BlockTT` can then be used in a tensorized
layer and fine-tuned; however, this feature itself performs only the matrix
approximation and does not attempt to train or evaluate a complete model.

## How it should be used:

The user supplies a matrix, with `input_modes` for its columns, `output_modes` for its
rows, and an integer `max_parameters` budget to `tltorch.blocktt_MALS`. The function
returns `(weight, diagnostics)`: a trainable `BlockTT` with independent cores, and
information about actual ranks, storage, reconstruction errors, refinement, and
stopping.

Reconstruction can be done with `weight.to_matrix()`. The reconstruction error is reported
in `diagnostics` and can also be checked by comparing the reconstructed matrix with the
original. Actual MPO storage is `sum(core.numel() for core in weight.factors)`, which
counts all stored entries, including zeros. In contrast, `weight.numel()` reports the
dense matrix size.

## How it should work:

First, use TT-SVD to turn the dense matrix into an initial MPO with the requested starting
ranks. Then alternate between improving the current cores and deciding which bond to enlarge:

1. Improve the cores without changing their ranks. Sweep through neighboring pairs of cores.
For each pair, use the original matrix and the surrounding cores to compute an improved
two-site tensor. Split it back into two cores using SVD, keeping the existing bond dimension.
Sweep in both directions until the improvement becomes small or the sweep limit is reached,
retaining the best approximation found.

2. Try increasing each internal bond by one. Consider only increases that fit the storage budget
and the local dimensions. Each trial starts from the same refined MPO and updates the two
cores adjacent to that bond. These are separate candidates; their changes are not accumulated.

3. Choose the most useful increase. Measure the full matrix reconstruction error for each
candidate. Score it by the reduction in squared error divided by the number of additional
stored entries, counting both affected cores. Accept the candidate with the highest score,
provided its improvement exceeds the numerical threshold.

After accepting a candidate, refine its cores again and repeat. Stop when the reconstruction is
exact or no allowed bond increase produces a meaningful improvement. The budget can therefore
remain partly unused.

## Checks:

The acceptance checks below distinguish independent mathematical references from
checks of state, reported information, and failure handling. Test names link to
their source files.

| Required behavior and evidence | Test functions |
| --- | --- |
| **Reconstruction and physical axes — independent references.** A nonsymmetric Kronecker target checks nonuniform axis ordering, ranks, and storage in both dtypes. A separately formed paired unfolding supplies the two-core SVD reference. | [test_nonuniform_kronecker_target_preserves_physical_axis_order][compression]; [test_two_core_budget_growth_matches_svd_optimum][compression] |
| **Two-site mathematics — independent identities and SVD reference.** Projection satisfies residual orthogonality and the excess-loss identity; both split directions and refinement agree with truncated SVD, including its residual energy. Exterior bases are obtained from implementation helpers. | [test_projection_orthogonality_and_excess_loss_identity][compression]; [test_two_core_refinement_equals_truncated_svd][compression] |
| **Budget and initialization — direct counts and consistency.** Actual ranks and stored entries respect the budget; measured residuals agree with diagnostics, and histories are internally consistent. Oversized, below-minimum, and dimensionally redundant initializations are rejected. | [test_storage_errors_histories_and_work_counts][compression]; [test_invalid_initial_storage_and_dimensional_redundancy][compression] |
| **Competing bonds — independent analytic answer.** Gains of 4 at cost 5 and 5 at cost 7 select the first bond. The returned matrix retains entries 10 and 2, with storage 14, squared residual 5, two candidates, one growth, and the expected normalized score. | [test_competing_bonds_choose_improvement_per_added_parameter][compression] |
| **Isolation, refinement, and stopping — instrumentation and fault injection.** Candidates preserve their source and evaluation-order independence; a slight worsening retains initialization; material corruption raises. Budget-blocked states are refined, and sweep caps do not prevent affordable growth. | [test_independent_candidates_selection_and_single_initialization][compression]; [test_refinement_keeps_initial_best_despite_allowed_roundoff][compression]; [test_material_numerical_errors_are_not_silently_accepted][compression]; [test_budget_blocked_state_is_refined_and_caps_do_not_end_growth][compression] |
| **Input domain — validation checks.** Invalid controls, tensorizations, and unsupported matrices raise appropriate exceptions. Valid rectangular, noncontiguous float32/float64 inputs are covered by the trainable-output check below. | [test_invalid_controls_even_for_zero_matrix][compression]; [test_invalid_tensorizations][compression]; [test_unsupported_matrices][compression] |
| **Numerics — independent residuals and range checks.** Robust residual references check extreme scales and a tiny discarded tail. CPU autocast preserves core precision, SVD agreement, diagnostics, and caller context. Unrepresentable required squared diagnostics still raise. | [test_budget_errors_match_independent_reference_across_scales][numerics]; [test_tiny_discarded_tail_with_representable_global_error][numerics]; [test_compression_preserves_precision_inside_cpu_autocast][numerics]; [test_unrepresentable_required_squared_diagnostics_raise][numerics] |
| **Caller state and trainable output — state and integration checks.** Input values and future RNG draws are preserved, including an injected decomposition failure. Independent leaf cores receive gradients without differentiating through the input; a wrapping `FactorizedLinear` agrees with dense application of the returned weight. | [test_rng_state_future_draws_and_input_graph_are_preserved][numerics]; [test_rng_and_input_restored_after_decomposition_failure][numerics]; [test_rectangular_noncontiguous_trainable_output_and_wrapper][compression] |

Inputs must be finite, nonempty, dense CPU float32/float64 matrices; noncontiguous
inputs are supported. Mode sequences have equal length of at least two, with
products matching the matrix dimensions. Initialization must fit the budget and
local dimensions; the default rank cap is not automatically lowered for a small
budget. CPU autocast is disabled throughout compression and the caller's context
is restored afterward. Subsequent `BlockTT` inference follows the caller's
autocast context. Nonfinite arithmetic or unrepresentable required measurements
and scores can still raise numerical exceptions.

This is a growth-only local heuristic: it does not shrink or reallocate ranks.
Stopping establishes neither global optimality nor infeasibility, and a sweep
cap does not prove convergence. Separate runs with larger budgets need not
improve error. Dense targets, environments, and reconstructions limit practical
problem size. These checks establish behavior on their tested cases; they do
not establish model accuracy preservation or faster inference.

[compression]: tltorch/factorized_tensors/tests/test_mpo_compression.py
[numerics]: tltorch/factorized_tensors/tests/test_mpo_numerics.py
