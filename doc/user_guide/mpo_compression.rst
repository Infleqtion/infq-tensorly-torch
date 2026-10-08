Budget-driven MPO matrix compression
====================================

``tltorch.blocktt_MALS`` approximates a dense matrix ``A``
with a :class:`~tltorch.BlockTT` matrix product operator (MPO). It reduces

.. math::

   \|\widehat A-A\|_F^2
   \quad\text{subject to}\quad
   P = \sum_{k=1}^{L} r_{k-1}\,o_k\,i_k\,r_k \le B.

``input_modes`` tensorizes columns and ``output_modes`` tensorizes rows.
Both have the same length, at least two. Core ``k`` has axes
``(r[k], output_modes[k], input_modes[k], r[k+1])`` with unit boundary ranks.
Reconstruction groups all output modes before all input modes.

The budget counts **all stored core entries**, even if frozen or zero. Use
``sum(core.numel() for core in weight.factors)``; ``weight.numel()`` instead
reports the dense matrix size. A percentage ``p`` denotes dense entries
**retained**, not saved: ``B = math.floor(p / 100 * A.numel())``. Feasible
storage counts are discrete, so the budget is an upper bound.

A small example
---------------

Each call below independently compresses the same 16-by-16 matrix. The table
shows requested budget, actual core storage, and relative Frobenius error.
A private generator makes the target reproducible without changing the global
CPU random stream.

.. code-block:: python

   import math
   import torch
   import tltorch

   generator = torch.Generator().manual_seed(42)
   matrix = torch.randn(16, 16, dtype=torch.float64, generator=generator)
   modes = (2, 2, 2, 2)
   print("budget  stored  relative_error")
   for retained_percentage in (18.75, 37.5, 62.5, 87.5):
       budget = math.floor(retained_percentage / 100 * matrix.numel())
       weight, info = tltorch.blocktt_MALS(
           matrix, modes, modes, budget,
       )
       stored = sum(core.numel() for core in weight.factors)
       assert stored == info["final_mpo_parameters"] <= budget
       print(f'{budget:6d}  {stored:6d}  {info["final_relative_weight_error"]:.6f}')

   layer = tltorch.FactorizedLinear(
       modes, modes, factorization=weight,
       bias=False, implementation="factorized",
   )
   assert layer.weight is weight
   inputs = torch.randn(2, 16, dtype=matrix.dtype, generator=generator)
   torch.testing.assert_close(layer(inputs), inputs @ weight.to_matrix().T)

Wrapping shares the returned weight object; training the layer changes that
weight. The returned cores are independent trainable leaf parameters.
Decomposition does not differentiate through or modify ``matrix``. CPU RNG
state is preserved on both successful return and exceptions.

Algorithm and initialization
----------------------------

One TensorLy TT-SVD initializes small rank caps. A full refinement sweep visits
adjacent pairs left-to-right, then right-to-left. Reduced QR puts the exterior
cores into orthonormal form. Projecting the **original target** through these
exterior bases solves local least squares; a truncated pair SVD keeps the
current shared rank. Refinement retains its best measured iterate, including
initialization.

Each growth candidate starts from an independent copy of the same current
cores and increases one shared bond by one. Local dimensional limits and the
cost of **both** adjacent cores are checked before evaluating it. Improving
candidates are scored by squared relative-error reduction per added entry.
For a fixed nonzero target this orders candidates identically to squared
absolute-error reduction. Ties prefer lower error, fewer entries, then lower
1-based bond index. An accepted state is refined before checking growth again,
even if its storage has exhausted the budget, unless its residual is exactly zero.

``initial_ranks=2`` is a uniform cap; a full ``(1, r1, ..., 1)`` profile is also
accepted. Unfolding dimensions can clip those caps. Initialization must fit the
budget: for the example, rank two costs 48 entries and rank one costs 16.
For a smaller budget, explicitly pass ``initial_ranks=1``. A budget below
``sum(o * i for o, i in zip(output_modes, input_modes))`` cannot represent the
supported positive-rank cores and is rejected before decomposition.

Stored ranks must permit shape-preserving QR in both directions: for a paired
core ``(left, physical, right)``, require ``right <= left * physical`` and
``left <= physical * right``. TT-SVD can leave some nonuniform profiles that
violate these restrictions; they are rejected rather than silently shrunk.
For example, with four paired dimensions of size four, caps
``(1, 4, 16, 1, 1)`` leave a dimensionally redundant third core.

``max_sweeps_per_refinement`` caps full sweeps per refinement call.
``sweep_tolerance`` tests the decrease in relative error over a full sweep;
it is **not an approximation-error target**. The separate numerical-improvement
threshold uses ``50 * number_of_cores * finfo(dtype).eps``, scaled by target,
reconstruction and residual norms. This is a practical roundoff allowance,
not a rigorous convergence bound.

Diagnostics and stopping
------------------------

The returned dictionary contains the following keys:

* ``max_parameters``, ``minimum_mpo_parameters``, ``dense_parameters``,
  ``unused_budget``;
* ``initial_rank_profile``, ``final_rank_profile``,
  ``initial_mpo_parameters``, ``final_mpo_parameters``;
* ``initial_squared_frobenius_error``, ``final_squared_frobenius_error``,
  ``initial_relative_weight_error``, ``final_relative_weight_error``;
* ``max_sweeps_per_refinement``, ``sweep_tolerance``, ``roundoff_factor``,
  ``termination_reason``;
* ``number_of_accepted_growth_steps``, ``number_of_candidate_evaluations``,
  ``number_of_refinement_calls``, ``completed_refinement_sweeps``,
  ``refinement_calls_at_sweep_limit``, ``growth_pair_svd_count``,
  ``refinement_pair_svd_count``, ``full_target_decomposition_count``;
* ``history`` and ``refinement_history``.

``history`` has one row for initialization and each accepted growth state.
``growth_step=0`` is initialization; ``changed_bond`` uses 1-based indices.
``rank_profile``, ``mpo_parameters``, ``added_parameters`` and ``score`` describe
that accepted state and its growth move. The relative-error fields
``error_before_growth`` and ``error_after_growth`` describe the move immediately,
while ``error_before_refinement`` and ``error_after_refinement`` describe its
subsequent refinement. ``relative_weight_error`` and ``squared_frobenius_error``
describe the state after refinement if one occurred, otherwise its arrival.
Inapplicable fields are ``None``. ``refinement_call`` links to a 1-based call ID.

``refinement_history`` records that call ID, ``growth_step``, ``rank_profile``,
``mpo_parameters``, relative ``error_before``/``error_after``,
``squared_frobenius_error_before``/``squared_frobenius_error_after``,
``completed_sweeps``, ``pair_svd_count`` and ``termination_reason``.
Its reason is ``sweep_improvement_small`` or ``max_sweeps_reached``. A sweep cap
can coexist with further accepted growth and is not treated as convergence.
Rejected candidates and individual pair updates are not public traces.

The outer stopping reasons are:

* ``exact_reconstruction``: the measured residual is actually zero;
* ``parameter_budget_blocks_growth``: no dimensionally permitted increment fits;
* ``local_rank_limits_reached``: every single-bond increment exceeds local dimensions;
* ``no_numerically_improving_candidate``: evaluated candidates fail the improvement threshold.

Limits
------

Inputs must be finite, nonempty, dense strided CPU float32/float64 matrices;
noncontiguous inputs work. GPU, complex, integer, half-precision and sparse
inputs are rejected. Modes, ranks and budgets require integers, excluding
booleans. CPU autocast is disabled throughout compression so core contractions,
QR and SVD retain the input dtype; the caller's autocast state is restored on
exit. Subsequent use of the returned ``BlockTT`` follows the caller's autocast
context. Measurements promote operands to float64 **before subtraction** and
use robust scaled norms.
Nonfinite measured arithmetic or unrepresentable required norms, squared errors
or scores raise ``FloatingPointError`` rather than reporting false zero or NaN.
Even a representable norm can have an unrepresentable square. Rescale explicitly
when necessary. A zero target reconstructed exactly has relative error zero.

Ranks only grow: there is no shrinking, reallocation or backtracking. Search
can stop despite a better feasible approximation, and independent runs at
larger budgets need not have lower errors. Stopping proves neither global
optimality nor infeasibility; sweep caps do not prove convergence.

The solver uses dense targets, explicit exterior environments and intermediate
dense reconstructions. These limit scalability. Small examples do not establish
large-model compression quality, and no inference speedup has been established.
