"""Budget-driven MPO approximation of small dense CPU matrices.

Working cores have axes (left rank, output mode * input mode, right rank).
Measurements use float64; decompositions and contractions keep the input dtype.
"""

import math
from collections.abc import Sequence
from functools import wraps
from numbers import Integral, Real

import torch
from torch import nn
import tensorly as tl
from tensorly.decomposition import tensor_train_matrix

from .tensorized_matrices import BlockTT

tl.set_backend('pytorch')


def _preserve_cpu_rng(function):
    """Preserve the caller's CPU stream on both normal return and exceptions."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        # No seed is set, and no CUDA generator is initialized or inspected.
        # fork_rng restores CPU state on both normal return and exceptions.
        with torch.random.fork_rng(devices=[]):
            return function(*args, **kwargs)
    return wrapped


def _checked_product(first, second, quantity):
    """Reject nonfinite or underflowed scalar results, including diagnostics."""
    result = first * second
    if not math.isfinite(result) or (result == 0.0 and first != 0.0 and second != 0.0):
        raise FloatingPointError(f"{quantity} is outside the representable float64 range; "
                                 "rescale the problem explicitly.")
    return result


def _squared_norm(norm):
    """A nonzero residual must never get a falsely zero squared-error diagnostic."""
    return _checked_product(norm, norm, "Squared norm")


def _frobenius_norm(tensor):
    """Measure any real tensor in float64, scaling only at extreme magnitudes.

    The maximum entry determines actual zero, not a potentially underflowed
    sum of squares. The ordinary-range path retains the existing float64
    reduction order. Scaling the extreme-range path keeps squares near one.
    Even a scaled norm may exceed float64's final scalar range: fail explicitly.
    """
    values = tensor.detach().to(torch.float64)
    if not torch.isfinite(values).all().item():
        raise FloatingPointError("Nonfinite measurement values: core/forward arithmetic or "
                                 "a residual exceeded its dtype's numerical range.")
    if values.numel() == 0:
        return 0.0
    largest = values.abs().max().item()
    if largest == 0.0:
        return 0.0
    limits = torch.finfo(torch.float64)
    if math.sqrt(limits.tiny) <= largest <= math.sqrt(limits.max / values.numel()):
        norm = torch.linalg.vector_norm(values).item()
        if math.isfinite(norm) and norm != 0.0:
            return norm
    scaled_norm = torch.linalg.vector_norm(values / largest).item()
    return _checked_product(largest, scaled_norm, "Frobenius norm")


def _measurement_residual(first, second):
    """Promote BEFORE subtraction; casting an overflowed float32 residual is too late."""
    residual = first.detach().to(torch.float64) - second.detach().to(torch.float64)
    if not torch.isfinite(residual).all().item():
        raise FloatingPointError("Nonfinite measurement residual: operands or their difference "
                                 "exceeded the representable numerical range.")
    return residual


def _validate_matrix(matrix, name):
    """Check the supported tensor shape, storage, device, dtype, and values."""
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor.")
    if matrix.ndim != 2 or matrix.numel() == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional tensor.")
    if matrix.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU; got {matrix.device}.")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64; got {matrix.dtype}.")
    if matrix.layout != torch.strided:
        raise ValueError(f"{name} must use dense strided storage.")
    if not torch.isfinite(matrix).all().item():
        raise ValueError(f"{name} must contain only finite values.")


def _positive_integer_tuple(values, name):
    """Validate mode dimensions or bond dimensions without rounding them."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers.")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain only positive integers; got {value!r}.")
        if value <= 0:
            raise ValueError(f"{name} entries must be positive; got {value}.")
    return tuple(int(value) for value in values)


def _validate_tensorization(matrix, input_modes, output_modes):
    """Validate the matrix and return its matching input/output mode tuples."""
    _validate_matrix(matrix, "matrix")
    input_modes = _positive_integer_tuple(input_modes, "input_modes")
    output_modes = _positive_integer_tuple(output_modes, "output_modes")
    if len(input_modes) != len(output_modes) or len(input_modes) < 2:
        raise ValueError("input_modes and output_modes must have the same length, at least 2.")
    if math.prod(input_modes) != matrix.shape[1]:
        raise ValueError("The input_modes product must equal the matrix column count.")
    if math.prod(output_modes) != matrix.shape[0]:
        raise ValueError("The output_modes product must equal the matrix row count.")
    return input_modes, output_modes


def _relative_error(residual_norm, reference_norm):
    """Use robust norms: zero means actual zero; no denominator epsilon is added."""
    if not math.isfinite(residual_norm) or not math.isfinite(reference_norm):
        raise FloatingPointError("Relative error requires finite measured norms.")
    if reference_norm == 0.0:
        return 0.0 if residual_norm == 0.0 else math.inf
    relative = residual_norm / reference_norm
    if not math.isfinite(relative) or (relative == 0.0 and residual_norm != 0.0):
        raise FloatingPointError("Relative error is outside the representable float64 range.")
    return relative


def _growth_score(before, after, added_parameters):
    """Equivalent squared-error reduction, without subtracting two large squares."""
    if not math.isfinite(before) or not math.isfinite(after):
        raise FloatingPointError("Growth scoring requires finite relative errors.")
    if before == after:
        return 0.0
    return _checked_product(before - after, _relative_error(before + after, added_parameters),
                            "Squared-error reduction per added parameter")


def _rank_profile(cores):
    return (int(cores[0].shape[0]),) + tuple(int(core.shape[-1]) for core in cores)


def _initialize_cores(matrix, input_modes, output_modes, initial_ranks):
    """Initialize once with TensorLy TT-SVD, without temporary linear layers."""
    if isinstance(initial_ranks, Integral) and not isinstance(initial_ranks, bool):
        if initial_ranks <= 0:
            raise ValueError("initial_ranks must be positive.")
        initial_ranks = int(initial_ranks)
    else:
        initial_ranks = _positive_integer_tuple(initial_ranks, "initial_ranks")
        if len(initial_ranks) != len(input_modes) + 1:
            raise ValueError("initial_ranks must have number_of_cores + 1 entries.")
        if initial_ranks[0] != 1 or initial_ranks[-1] != 1:
            raise ValueError("initial_ranks must have boundary ranks equal to 1.")

    # TensorLy accepts grouped row modes followed by grouped column modes;
    # it interleaves physical indices internally. Use actual factor shapes
    # because unfolding dimensions can clip requested rank caps.
    factors = tensor_train_matrix(
        matrix.reshape(output_modes + input_modes), rank=initial_ranks,
    )
    return [factor.detach().contiguous().clone().reshape(
        factor.shape[0], output_mode * input_mode, factor.shape[-1]
    ) for factor, output_mode, input_mode in zip(factors, output_modes, input_modes)]


def _validate_core_shapes(cores, physical_sizes):
    """Require the stored dimensions needed for shape-preserving QR and SVD."""
    ranks = _rank_profile(cores)
    if len(cores) != len(physical_sizes) or ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("MPO cores must match the modes and have unit boundary ranks.")
    for index, (core, physical) in enumerate(zip(cores, physical_sizes)):
        left, right = ranks[index:index + 2]
        if tuple(core.shape) != (left, physical, right):
            raise ValueError(f"Core {index} has inconsistent physical or bond dimensions.")
        if not torch.isfinite(core).all().item():
            raise FloatingPointError(f"Core {index} contains nonfinite values.")
        if right > left * physical or left > physical * right:
            raise ValueError(
                f"Core {index}, shape {tuple(core.shape)}: dimensional redundancy "
                "prevents shape-preserving canonicalization/splitting; "
                "supply a feasible initial_ranks profile."
            )


def _matrix_to_paired_tensor(matrix, input_modes, output_modes):
    number_of_cores = len(input_modes)
    # Grouped output/input axes become adjacent physical pairs.
    axes = [axis for k in range(number_of_cores) for axis in (k, number_of_cores + k)]
    paired_modes = tuple(j * i for j, i in zip(output_modes, input_modes))
    return matrix.reshape(output_modes + input_modes).permute(axes).reshape(paired_modes)


def _paired_cores_to_matrix(cores, input_modes, output_modes):
    """Direct contraction using BlockTT's grouped-output/grouped-input ordering."""
    expanded = [
        core.reshape(core.shape[0], j, i, core.shape[-1])
        for core, j, i in zip(cores, output_modes, input_modes)
    ]
    contracted = expanded[0]
    for core in expanded[1:]:
        shape = (contracted.shape[0], contracted.shape[1] * core.shape[1],
                 contracted.shape[2] * core.shape[2], core.shape[-1])
        contracted = torch.einsum("aijb,bklc->aikjlc", contracted, core).reshape(shape)
    return contracted.reshape(math.prod(output_modes), math.prod(input_modes))


def _measure_matrix(matrix, reconstruction):
    residual = _measurement_residual(matrix, reconstruction)
    absolute_error = _frobenius_norm(residual)
    return {
        "absolute_frobenius_error": absolute_error,
        "relative_weight_error": _relative_error(absolute_error, _frobenius_norm(matrix)),
        "maximum_absolute_error": residual.abs().max().item(),
    }


def _measure_cores(matrix, cores, input_modes, output_modes):
    ranks = _rank_profile(cores)
    parameter_count = sum(core.numel() for core in cores)
    analytic_count = sum(ranks[k] * j * i * ranks[k + 1]
                         for k, (j, i) in enumerate(zip(output_modes, input_modes)))
    if parameter_count != analytic_count:
        raise RuntimeError("Stored core count does not match the rank/mode formula.")
    return {
        "actual_rank_profile": ranks,
        "mpo_parameters": parameter_count,
        **_measure_matrix(matrix, _paired_cores_to_matrix(cores, input_modes, output_modes)),
    }


def _blocktt_from_cores(cores, input_modes, output_modes):
    # FactorList registers plain tensors as buffers. Explicit Parameters make
    # every returned core a trainable leaf independent of all working storage.
    factors = [nn.Parameter(core.detach().reshape(
        core.shape[0], j, i, core.shape[-1]
    ).contiguous().clone()) for core, j, i in zip(cores, output_modes, input_modes)]
    return BlockTT(factors, tensorized_shape=(output_modes, input_modes),
                   rank=_rank_profile(cores))


def _canonicalize_around_pair(cores, pair_index):
    """Modify detached cores in place, leaving the active pair noncanonical.

    Left QR: (r_left*d, r_right) = Q H; keep Q and absorb H to the right.
    Right QR: (r_left, d*r_right).T = Q H; keep Q.T and absorb H.T left.
    Reduced QR must preserve every stored shape, including for zero cores.
    """
    for k in range(pair_index):
        left, physical, right = cores[k].shape
        if right > left * physical:
            raise ValueError(f"Core {k}: dimensional redundancy prevents shape-preserving left QR.")
        q, remainder = torch.linalg.qr(cores[k].reshape(left * physical, right), mode="reduced")
        cores[k] = q.reshape(left, physical, right)
        cores[k + 1] = torch.einsum("ab,bpc->apc", remainder, cores[k + 1])
    for k in range(len(cores) - 1, pair_index + 1, -1):
        left, physical, right = cores[k].shape
        if left > physical * right:
            raise ValueError(f"Core {k}: dimensional redundancy prevents shape-preserving right QR.")
        q, remainder = torch.linalg.qr(cores[k].reshape(left, physical * right).T, mode="reduced")
        cores[k] = q.T.reshape(left, physical, right)
        cores[k - 1] = torch.einsum("apb,bc->apc", cores[k - 1], remainder.T)


def _pair_environments(cores, pair_index):
    """Explicit small left-column and right-row orthonormal environment matrices."""
    left_environment = cores[0].new_ones((1, 1))
    for core in cores[:pair_index]:
        left_environment = torch.einsum(
            "la,apb->lpb", left_environment, core
        ).reshape(-1, core.shape[-1])
    right_environment = cores[0].new_ones((1, 1))
    for core in reversed(cores[pair_index + 2:]):
        right_environment = torch.einsum(
            "apb,bt->apt", core, right_environment
        ).reshape(core.shape[0], -1)
    return left_environment, right_environment


def _project_target_pair(paired_target, pair_index, left_environment, right_environment):
    target_view = paired_target.reshape(
        left_environment.shape[0], paired_target.shape[pair_index],
        paired_target.shape[pair_index + 1], right_environment.shape[1],
    )
    # The exterior bases are isometries: their transposes solve local least squares.
    return torch.einsum("la,lpqt,bt->apqb", left_environment, target_view, right_environment)


def _split_fixed_rank_pair(projected_pair, shared_rank, direction):
    left, first_mode, second_mode, right = projected_pair.shape
    unfolding = projected_pair.reshape(left * first_mode, second_mode * right)
    if shared_rank > min(unfolding.shape):
        raise ValueError("Dimensional redundancy prevents a shape-preserving fixed-rank SVD split.")
    u, s, vh = torch.linalg.svd(unfolding, full_matrices=False)
    # Preserve the range check on local discarded energy, even though it is
    # not exposed as a global error bound or an exhaustive pair-update trace.
    discarded_squared_norm = _squared_norm(_frobenius_norm(s[shared_rank:]))
    if direction == "left_to_right":
        first = u[:, :shared_rank]
        second = s[:shared_rank, None] * vh[:shared_rank]
    elif direction == "right_to_left":
        first = u[:, :shared_rank] * s[None, :shared_rank]
        second = vh[:shared_rank]
    else:
        raise ValueError("Unknown two-site sweep direction.")
    return (
        first.reshape(left, first_mode, shared_rank),
        second.reshape(shared_rank, second_mode, right),
        discarded_squared_norm,
    )


def _refine_two_site(matrix, initial_cores, input_modes, output_modes,
                     max_sweeps, sweep_tolerance):
    """Optimize core entries at fixed stored ranks by two-site least squares.

    One full sweep is left-to-right followed by right-to-left. Recompute mixed
    canonical form with reduced QR around each pair, project the original target
    through orthonormal exterior environments, and split at the existing rank.
    No full-target decomposition, rank adaptation, or gradient descent occurs.

    Stop when a full sweep decreases measured relative error by <= sweep_tolerance
    (an absolute change, NOT requested approximation accuracy). Retain the best
    measured pair iterate, including initialization. Inputs are never modified.

    Roundoff factor is 50 * number_of_cores * finfo(dtype).eps. Canonicalization
    drift and local error increases are checked in absolute Frobenius norm,
    scaled by max(target norm, pre-update reconstruction norm, residual norm).
    This is a practical check for small dense problems, not a rigorous bound.
    Local discarded energies are diagnostics, NOT an accumulated global bound.
    """
    cores = [core.detach().clone() for core in initial_cores]
    rank_profile = _rank_profile(cores)

    initial_metrics = _measure_cores(matrix, initial_cores, input_modes, output_modes)
    initial_error = initial_metrics["relative_weight_error"]
    parameter_count = initial_metrics["mpo_parameters"]
    paired_target = _matrix_to_paired_tensor(matrix.detach(), input_modes, output_modes)
    target_norm = _frobenius_norm(matrix)
    roundoff_factor = 50 * len(cores) * torch.finfo(matrix.dtype).eps
    best_error = initial_error
    best_cores = [core.clone() for core in cores]
    pair_svd_count = 0
    previous_sweep_error = initial_error
    termination_reason = "max_sweeps_reached"

    for full_sweep in range(1, int(max_sweeps) + 1):
        for direction, pair_indices in (
            ("left_to_right", range(len(cores) - 1)),
            ("right_to_left", range(len(cores) - 2, -1, -1)),
        ):
            for pair_index in pair_indices:
                context = f"sweep={full_sweep}, direction={direction}, bond={pair_index + 1}"
                before = _paired_cores_to_matrix(cores, input_modes, output_modes)
                before_absolute_error = _frobenius_norm(_measurement_residual(matrix, before))
                before_error = _relative_error(before_absolute_error, target_norm)
                scale = max(target_norm, _frobenius_norm(before), before_absolute_error)
                absolute_allowance = _checked_product(roundoff_factor, scale, "Roundoff allowance")

                _canonicalize_around_pair(cores, pair_index)
                canonical = _paired_cores_to_matrix(cores, input_modes, output_modes)
                canonical_drift = _frobenius_norm(_measurement_residual(canonical, before))
                if not math.isfinite(canonical_drift) or canonical_drift > absolute_allowance:
                    raise RuntimeError(f"Canonicalization changed the tensor: {context}, "
                                       f"drift={canonical_drift}, allowance={absolute_allowance}.")
                left_environment, right_environment = _pair_environments(cores, pair_index)
                left_gram = left_environment.T @ left_environment
                right_gram = right_environment @ right_environment.T
                for name, gram in (("left", left_gram), ("right", right_gram)):
                    defect = _frobenius_norm(_measurement_residual(
                        gram, torch.eye(gram.shape[0], dtype=matrix.dtype, device=matrix.device)
                    ))
                    if not math.isfinite(defect) or defect > roundoff_factor * gram.shape[0]:
                        raise RuntimeError(f"Nonorthonormal {name} environment: {context}, defect={defect}.")

                projected_pair = _project_target_pair(
                    paired_target, pair_index, left_environment, right_environment
                )
                first, second, _ = _split_fixed_rank_pair(
                    projected_pair, rank_profile[pair_index + 1], direction
                )
                cores[pair_index], cores[pair_index + 1] = first, second
                actual_profile = (int(cores[0].shape[0]),) + tuple(int(c.shape[-1]) for c in cores)
                actual_count = sum(c.numel() for c in cores)
                if actual_profile != rank_profile or actual_count != parameter_count:
                    raise RuntimeError(f"Fixed-rank update changed storage: {context}.")
                after = _paired_cores_to_matrix(cores, input_modes, output_modes)
                after_absolute_error = _frobenius_norm(_measurement_residual(matrix, after))
                after_error = _relative_error(after_absolute_error, target_norm)
                if (not math.isfinite(after_absolute_error)
                        or after_absolute_error > before_absolute_error + absolute_allowance):
                    raise RuntimeError(
                        f"Two-site error increased: {context}, before={before_error}, "
                        f"after={after_error}, absolute_allowance={absolute_allowance}."
                    )
                pair_svd_count += 1
                # Keep the relative allowance range check even without a trace.
                _relative_error(absolute_allowance, target_norm)
                if after_error < best_error:
                    best_error = after_error
                    best_cores = [core.clone() for core in cores]

        # For zero-reference errors, equal infinities mean no measured progress.
        decrease = (0.0 if previous_sweep_error == after_error
                    else previous_sweep_error - after_error)
        if decrease <= sweep_tolerance:
            termination_reason = "sweep_improvement_small"
            break
        previous_sweep_error = after_error

    # Accepted states use the same contiguous layout as stored BlockTT cores.
    best_cores = [core.contiguous().clone() for core in best_cores]
    final_metrics = _measure_cores(matrix, best_cores, input_modes, output_modes)
    if final_metrics["relative_weight_error"] > initial_error:
        raise RuntimeError("Returned best iterate is worse than initialization.")
    return best_cores, {
        "completed_sweeps": full_sweep,
        "termination_reason": termination_reason,
        "pair_svd_count": pair_svd_count,
    }


def _evaluate_growth(matrix, current_cores, paired_target, input_modes, output_modes,
                     pair_index, roundoff_factor):
    """Evaluate one local rank increment on independent copies of the current cores.

    This helper is called only after local-dimension and budget prechecks. All
    candidate errors use the public BlockTT reconstruction/measurement convention.
    The SVD initializes the added direction from the projected ORIGINAL target.
    """
    ranks = _rank_profile(current_cores)
    cores = [core.detach().clone() for core in current_cores]
    before = _paired_cores_to_matrix(current_cores, input_modes, output_modes)
    before_metrics = _measure_cores(matrix, current_cores, input_modes, output_modes)
    target_norm = _frobenius_norm(matrix)
    scale = max(target_norm, _frobenius_norm(before),
                before_metrics["absolute_frobenius_error"])
    absolute_allowance = _checked_product(roundoff_factor, scale, "Roundoff allowance")
    context = f"growth bond={pair_index + 1}"

    _canonicalize_around_pair(cores, pair_index)
    canonical = _paired_cores_to_matrix(cores, input_modes, output_modes)
    drift = _frobenius_norm(_measurement_residual(canonical, before))
    if not math.isfinite(drift) or drift > absolute_allowance:
        raise RuntimeError(f"Canonicalization changed the tensor: {context}, "
                           f"drift={drift}, allowance={absolute_allowance}.")
    left_environment, right_environment = _pair_environments(cores, pair_index)
    for name, gram in (
        ("left", left_environment.T @ left_environment),
        ("right", right_environment @ right_environment.T),
    ):
        identity = torch.eye(gram.shape[0], dtype=matrix.dtype, device=matrix.device)
        defect = _frobenius_norm(_measurement_residual(gram, identity))
        if not math.isfinite(defect) or defect > roundoff_factor * gram.shape[0]:
            raise RuntimeError(f"Nonorthonormal {name} environment: {context}, defect={defect}.")

    projected_pair = _project_target_pair(
        paired_target, pair_index, left_environment, right_environment
    )
    first, second, _ = _split_fixed_rank_pair(
        projected_pair, ranks[pair_index + 1] + 1, "left_to_right"
    )
    cores[pair_index], cores[pair_index + 1] = first, second
    cores = [core.contiguous().clone() for core in cores]
    metrics = _measure_cores(matrix, cores, input_modes, output_modes)
    after_absolute_error = metrics["absolute_frobenius_error"]
    before_absolute_error = before_metrics["absolute_frobenius_error"]
    if (not math.isfinite(after_absolute_error)
            or after_absolute_error > before_absolute_error + absolute_allowance):
        raise RuntimeError(
            f"Two-site growth error increased: {context}, before={before_absolute_error}, "
            f"after={after_absolute_error}, absolute_allowance={absolute_allowance}."
        )
    return cores, metrics, _relative_error(absolute_allowance, target_norm)


def _candidate_order(candidate):
    """Maximize score, then prefer lower error, storage and 1-based bond index."""
    return (candidate["score"], -candidate["relative_weight_error"],
            -candidate["actual_mpo_parameters"], -candidate["bond"])


def _choose_growth(matrix, current_cores, current_metrics, paired_target,
                   input_modes, output_modes, max_parameters, roundoff_factor):
    """Compare affordable local moves without changing the current state."""
    current_profile = current_metrics["actual_rank_profile"]
    current_cost = current_metrics["mpo_parameters"]
    current_error = current_metrics["relative_weight_error"]
    physical_sizes = tuple(j * i for j, i in zip(output_modes, input_modes))
    best_candidate = best_cores = best_metrics = None
    dimensional_candidates = evaluated_candidates = 0
    for bond in range(1, len(physical_sizes)):
        left_rank, right_rank = current_profile[bond - 1], current_profile[bond + 1]
        left_size, right_size = physical_sizes[bond - 1:bond + 1]
        local_limit = min(left_rank * left_size, right_size * right_rank)
        requested_profile = list(current_profile)
        requested_profile[bond] += 1
        if requested_profile[bond] > local_limit:
            continue
        dimensional_candidates += 1
        # Increasing r_bond adds a slice to BOTH adjacent cores.
        added_parameters = left_rank * left_size + right_size * right_rank
        proposed_cost = current_cost + added_parameters
        if proposed_cost > max_parameters:
            continue

        # Every candidate clones this SAME current state, including its gauges.
        candidate_cores, metrics, relative_allowance = _evaluate_growth(
            matrix, current_cores, paired_target, input_modes, output_modes,
            bond - 1, roundoff_factor,
        )
        evaluated_candidates += 1
        actual_cost = metrics["mpo_parameters"]
        if (metrics["actual_rank_profile"] != tuple(requested_profile)
                or actual_cost != proposed_cost or actual_cost > max_parameters):
            raise RuntimeError(f"Growth bond {bond}: unexpected ranks or storage.")
        candidate_error = metrics["relative_weight_error"]
        # For a fixed nonzero target, squared relative and absolute errors differ
        # by a constant, so either gives the same ordering of growth candidates.
        candidate = {
            "bond": bond, "actual_mpo_parameters": actual_cost,
            "added_parameters": actual_cost - current_cost,
            "relative_weight_error": candidate_error,
            "score": _growth_score(current_error, candidate_error, actual_cost - current_cost),
        }
        if current_error - candidate_error <= relative_allowance:
            continue
        if best_candidate is None or _candidate_order(candidate) > _candidate_order(best_candidate):
            best_candidate, best_cores, best_metrics = candidate, candidate_cores, metrics

    termination_reason = None
    if best_candidate is None:
        if dimensional_candidates == 0:
            termination_reason = "local_rank_limits_reached"
        elif evaluated_candidates == 0:
            termination_reason = "parameter_budget_blocks_growth"
        else:
            termination_reason = "no_numerically_improving_candidate"
    return best_candidate, best_cores, best_metrics, evaluated_candidates, termination_reason


@torch.no_grad()
@_preserve_cpu_rng
def compress_matrix_to_mpo_budget(
    matrix,
    input_modes,
    output_modes,
    max_parameters,
    *,
    initial_ranks=2,
    max_sweeps_per_refinement=10,
    sweep_tolerance=1e-10,
):
    """Approximate a dense matrix with an MPO under a stored-core budget.

    Reduce ``||weight.to_matrix() - matrix||_F**2`` using fixed-rank two-site
    least-squares sweeps and independent one-bond rank increases. This is a
    growth-only local heuristic, with no global optimality guarantee.

    Parameters
    ----------
    matrix : torch.Tensor
        Finite, nonempty, two-dimensional CPU float32 or float64 tensor with
        dense strided storage. Noncontiguous inputs are supported.
    input_modes, output_modes : sequence of positive int
        Column and row tensorizations, respectively. Their products must match
        the matrix dimensions; both sequences need the same length, at least 2.
        Core k has axes ``(r[k], output_modes[k], input_modes[k], r[k+1])``.
    max_parameters : int
        Upper bound on the sum of all stored core entries. For a percentage p
        of dense entries retained, pass ``math.floor(p / 100 * matrix.numel())``.
        Counts are discrete, so some budget may remain unused. The minimum
        supported cost is ``sum(o * i for o, i in zip(output_modes, input_modes))``.
        ``BlockTT.numel()`` reports dense size, not core storage.
    initial_ranks : int or sequence of int, optional
        TT-SVD rank cap (default 2), or a full profile with unit boundary ranks.
        Unfolding dimensions may clip the caps. Actual initialization must fit
        the budget and admit shape-preserving left and right QR at every core;
        some nonuniform profiles fail this condition. Use smaller ranks when
        initialization exceeds the budget. Ranks never shrink during search.
    max_sweeps_per_refinement : int, optional
        Positive cap on full left-to-right/right-to-left sweeps per refinement,
        default 10. Reaching this cap does not establish convergence.
    sweep_tolerance : float, optional
        Finite nonnegative threshold on the decrease in relative error over a
        full refinement sweep, default 1e-10. This is a progress threshold,
        not a requested matrix approximation error.

    Returns
    -------
    weight : BlockTT
        Independent, trainable leaf parameters with actual rank metadata.
        Reconstruction and subsequent training retain gradients to the cores;
        decomposition does not differentiate through the input matrix.
    diagnostics : dict
        Initial/final rank profiles, stored counts, squared Frobenius errors
        and relative errors; budget, unused budget, refinement controls,
        roundoff factor, work counts, and sweep-cap information. ``history``
        records initialization and accepted growth states, with separate
        before/after growth and refinement relative errors. Its unqualified
        error fields describe the state after refinement, if one occurred.
        ``refinement_history`` contains per-call errors, sweep counts and
        reasons: ``sweep_improvement_small`` or ``max_sweeps_reached``.
        Outer ``termination_reason`` is ``exact_reconstruction``,
        ``parameter_budget_blocks_growth``, ``local_rank_limits_reached``, or
        ``no_numerically_improving_candidate``. No rejected-candidate or
        pair-update traces are returned. See the MPO compression user guide
        for diagnostic keys and history conventions.

    Raises
    ------
    TypeError
        If input types or dtypes are unsupported, including boolean controls.
    ValueError
        If shapes, values, controls, initialization or budget are invalid.
    FloatingPointError
        If measured values or required norms, squared errors or scores cannot
        be represented. Explicit rescaling may be necessary.
    RuntimeError
        If an update violates storage, canonicalization or error checks.

    Notes
    -----
    TT-SVD initializes once. Each candidate projects the original target
    through orthonormal exterior environments and increases one bond by one.
    Meaningful candidates maximize squared relative-error reduction per added
    core entry; ties prefer lower error, fewer entries, then lower bond index.
    Refinement retains its best measured iterate, including its input.
    The roundoff factor is ``50 * len(input_modes) * finfo(matrix.dtype).eps``;
    checks scale this by target, reconstruction and residual norms.

    Input values and CPU RNG state are preserved, including on exceptions.
    Core arithmetic keeps the input dtype; measurement operands are promoted
    to float64 before subtraction and norms use scaling when needed. Exact
    reconstruction means a measured zero residual, not a small-error cutoff.
    Search may stagnate despite better feasible approximations; it neither
    reallocates ranks nor proves infeasibility. Dense targets, explicit
    environments and intermediate dense reconstructions limit scalability.
    No inference speedup has been established.
    """
    input_modes, output_modes = _validate_tensorization(matrix, input_modes, output_modes)
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, Integral):
        raise TypeError("max_parameters must be an integer, not a boolean.")
    max_parameters = int(max_parameters)
    physical_sizes = tuple(j * i for j, i in zip(output_modes, input_modes))
    minimum_parameters = sum(physical_sizes)
    if max_parameters < minimum_parameters:
        raise ValueError(
            f"Budget {max_parameters} is below the minimum stored-core cost "
            f"{minimum_parameters} for this tensorization (all ranks one)."
        )
    if (isinstance(max_sweeps_per_refinement, bool)
            or not isinstance(max_sweeps_per_refinement, Integral)):
        raise TypeError("max_sweeps_per_refinement must be a positive integer, not a boolean.")
    if max_sweeps_per_refinement <= 0:
        raise ValueError("max_sweeps_per_refinement must be a positive integer.")
    if isinstance(sweep_tolerance, bool) or not isinstance(sweep_tolerance, Real):
        raise TypeError("sweep_tolerance must be a finite nonnegative real number, not a boolean.")
    sweep_tolerance = float(sweep_tolerance)
    if not math.isfinite(sweep_tolerance) or sweep_tolerance < 0:
        raise ValueError("sweep_tolerance must be finite and nonnegative.")

    # Own a detached target copy, including for noncontiguous inputs.
    matrix = matrix.detach().clone()
    current_cores = _initialize_cores(matrix, input_modes, output_modes, initial_ranks)
    initial_metrics = _measure_cores(matrix, current_cores, input_modes, output_modes)
    initial_cost = initial_metrics["mpo_parameters"]
    if initial_cost > max_parameters:
        raise ValueError(
            f"Initial MPO uses {initial_cost} stored parameters, exceeding budget "
            f"{max_parameters}; smaller initial_ranks (for example 1) may be needed."
        )
    # Validate local dimensional feasibility even if initialization is exact.
    _validate_core_shapes(current_cores, physical_sizes)
    current_metrics = initial_metrics
    paired_target = _matrix_to_paired_tensor(matrix.detach(), input_modes, output_modes)
    roundoff_factor = 50 * len(physical_sizes) * torch.finfo(matrix.dtype).eps
    history = [{
        "growth_step": 0, "changed_bond": None,
        "rank_profile": initial_metrics["actual_rank_profile"],
        "mpo_parameters": initial_cost, "added_parameters": 0, "score": None,
        "error_before_growth": None, "error_after_growth": None,
        "error_before_refinement": None, "error_after_refinement": None,
        "squared_frobenius_error": _squared_norm(initial_metrics["absolute_frobenius_error"]),
        "relative_weight_error": initial_metrics["relative_weight_error"],
        "refinement_call": None,
    }]
    candidate_evaluations = 0
    refinement_history = []

    # Zero targets have exactly zero initial residual and relative error zero
    # under the explicit zero-norm convention. No growth
    # score or division by the target norm is evaluated in that case.
    while current_metrics["maximum_absolute_error"] != 0.0:
        before_refinement = current_metrics
        refined_cores, refinement = _refine_two_site(
            matrix, current_cores, input_modes, output_modes, max_sweeps=int(max_sweeps_per_refinement),
            sweep_tolerance=sweep_tolerance,
        )
        refined_metrics = _measure_cores(matrix, refined_cores, input_modes, output_modes)
        if (refined_metrics["actual_rank_profile"] != before_refinement["actual_rank_profile"]
                or refined_metrics["mpo_parameters"] != before_refinement["mpo_parameters"]):
            raise RuntimeError("Fixed-rank refinement changed ranks or storage.")
        if refined_metrics["absolute_frobenius_error"] > before_refinement["absolute_frobenius_error"]:
            raise RuntimeError("Fixed-rank refinement failed to retain its best accepted iterate.")
        call_id = len(refinement_history) + 1
        growth_step = len(history) - 1
        refinement_history.append({
            "refinement_call": call_id, "growth_step": growth_step,
            "rank_profile": refined_metrics["actual_rank_profile"],
            "mpo_parameters": refined_metrics["mpo_parameters"],
            "error_before": before_refinement["relative_weight_error"],
            "error_after": refined_metrics["relative_weight_error"],
            "squared_frobenius_error_before": _squared_norm(before_refinement["absolute_frobenius_error"]),
            "squared_frobenius_error_after": _squared_norm(refined_metrics["absolute_frobenius_error"]),
            "completed_sweeps": refinement["completed_sweeps"],
            "termination_reason": refinement["termination_reason"],
            "pair_svd_count": refinement["pair_svd_count"],
        })
        current_cores, current_metrics = refined_cores, refined_metrics
        history[-1].update({
            "refinement_call": call_id,
            "error_before_refinement": before_refinement["relative_weight_error"],
            "error_after_refinement": current_metrics["relative_weight_error"],
            "relative_weight_error": current_metrics["relative_weight_error"],
            "squared_frobenius_error": _squared_norm(current_metrics["absolute_frobenius_error"]),
        })
        if current_metrics["maximum_absolute_error"] == 0.0:
            termination_reason = "exact_reconstruction"
            break

        # Refine BEFORE checking whether any rank growth is affordable. If
        # blocked, this already-refined state is returned without duplicate work.
        best_candidate, best_cores, best_metrics, evaluated, blocked_reason = _choose_growth(
            matrix, current_cores, current_metrics, paired_target, input_modes, output_modes,
            max_parameters, roundoff_factor,
        )
        candidate_evaluations += evaluated
        if best_candidate is None:
            termination_reason = blocked_reason
            break
        history.append({
            "growth_step": len(history), "changed_bond": best_candidate["bond"],
            "rank_profile": best_metrics["actual_rank_profile"],
            "mpo_parameters": best_metrics["mpo_parameters"],
            "added_parameters": best_candidate["added_parameters"], "score": best_candidate["score"],
            "error_before_growth": current_metrics["relative_weight_error"],
            "error_after_growth": best_metrics["relative_weight_error"],
            "error_before_refinement": None, "error_after_refinement": None,
            "squared_frobenius_error": _squared_norm(best_metrics["absolute_frobenius_error"]),
            "relative_weight_error": best_metrics["relative_weight_error"], "refinement_call": None,
        })
        current_cores, current_metrics = best_cores, best_metrics
    else:
        termination_reason = "exact_reconstruction"

    weight = _blocktt_from_cores(current_cores, input_modes, output_modes)
    # Measure the public reconstruction, rather than a projected local loss.
    final_metrics = _measure_cores(matrix, current_cores, input_modes, output_modes)
    final_metrics.update(_measure_matrix(matrix, weight.to_matrix()))
    final_cost = final_metrics["mpo_parameters"]
    if (final_cost > max_parameters
            or final_metrics["absolute_frobenius_error"] > initial_metrics["absolute_frobenius_error"]):
        raise RuntimeError("Returned iterate violates the budget or best-iterate guarantee.")
    refinement_pair_svds = sum(row["pair_svd_count"] for row in refinement_history)
    capped_calls = [row["refinement_call"] for row in refinement_history
                    if row["termination_reason"] == "max_sweeps_reached"]
    return weight, {
        "max_parameters": max_parameters, "minimum_mpo_parameters": minimum_parameters,
        "max_sweeps_per_refinement": int(max_sweeps_per_refinement), "sweep_tolerance": sweep_tolerance,
        "initial_rank_profile": initial_metrics["actual_rank_profile"],
        "final_rank_profile": final_metrics["actual_rank_profile"],
        "initial_squared_frobenius_error": _squared_norm(initial_metrics["absolute_frobenius_error"]),
        "final_squared_frobenius_error": _squared_norm(final_metrics["absolute_frobenius_error"]),
        "initial_relative_weight_error": initial_metrics["relative_weight_error"],
        "final_relative_weight_error": final_metrics["relative_weight_error"],
        "initial_mpo_parameters": initial_cost, "final_mpo_parameters": final_cost,
        "unused_budget": max_parameters - final_cost, "dense_parameters": matrix.numel(),
        "termination_reason": termination_reason, "roundoff_factor": roundoff_factor,
        "number_of_accepted_growth_steps": len(history) - 1,
        "number_of_candidate_evaluations": candidate_evaluations,
        "number_of_refinement_calls": len(refinement_history),
        "completed_refinement_sweeps": sum(row["completed_sweeps"] for row in refinement_history),
        "refinement_calls_at_sweep_limit": len(capped_calls),
        "growth_pair_svd_count": candidate_evaluations,
        "refinement_pair_svd_count": refinement_pair_svds,
        "full_target_decomposition_count": 1,
        "history": history,
        "refinement_history": refinement_history,
    }
