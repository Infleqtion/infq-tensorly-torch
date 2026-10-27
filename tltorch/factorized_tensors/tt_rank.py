"""Shape-only, bounded TT rank selection for weight-parameter budgets."""

import math
from numbers import Integral, Real

from tensorly.tt_tensor import validate_tt_rank


_REFINEMENT_PASSES = 8


def tt_parameter_count(shape, ranks):
    """Count core elements, excluding bias and anything outside this weight."""
    return sum(left * size * right for left, size, right in zip(ranks, shape, ranks[1:]))


def resolve_tt_rank(shape, rank, *, tensor_numel=None):
    """Resolve a fractional weight budget to feasible TT-SVD bond dimensions.

    ``'same'`` means 1.0. The budget is floor(target * prod(shape)); ranks are
    positive and respect unfolding and neighboring-rank limits. Budgets below
    the rank-one count raise ValueError. Targets above 1.0 are supported and
    saturate at the largest feasible profile when it fits below the budget.

    Start at the largest fitting capped common-rank profile (binary search).
    Up to eight refinement passes consider single-bond increases and transfers
    that decrease one bond by one while increasing another. Each increase goes
    directly to its largest feasible, affordable value. Accept only strict count
    improvements. Ties prefer least L1 change from the initial profile, then
    lexicographic order. This bounded heuristic does not optimize approximation
    quality or guarantee the best possible use of every budget. Unused budget
    is a search limitation, not necessarily a mathematical impossibility.

    Explicit integer/list/tuple ranks retain TensorLy semantics, including
    clipping at decomposition time. Caller-owned lists are never modified.
    ``tensor_numel`` supplies the original element count for batched BlockTT,
    whose batch dimensions repeat across physical core modes. Otherwise it is
    prod(shape). Only fractional requests change policy; no factors or RNG are used here.
    """
    if rank == 'same':
        rank = 1.0
    if not isinstance(rank, Real) or isinstance(rank, Integral):
        return tuple(validate_tt_rank(shape, rank))
    target = float(rank)
    if not math.isfinite(target) or target <= 0:
        raise ValueError('TT storage target must be a finite positive float')
    if not shape or any(not isinstance(n, Integral) or isinstance(n, bool) or n <= 0 for n in shape):
        raise ValueError('TT shape must contain positive integer dimensions')
    shape = tuple(int(n) for n in shape)
    order = len(shape)
    # BlockTT batch modes occur in every core, but only once in the source tensor.
    dense_count = math.prod(shape) if tensor_numel is None else int(tensor_numel)
    caps = [min(math.prod(shape[:i]), math.prod(shape[i:])) for i in range(1, order)]
    full = (1, *caps, 1)
    full_count = tt_parameter_count(shape, full)
    try:
        budget = math.floor(target * dense_count)
    except OverflowError:
        # A finite large target may overflow only the intermediate float product.
        numerator, denominator = target.as_integer_ratio()
        budget = numerator * dense_count // denominator
    if budget >= full_count:
        return full
    minimum = sum(shape)
    if budget < minimum:
        raise ValueError(f'TT weight budget {budget} is below the rank-one representation '
                         f'of {minimum} parameters for shape {shape}')

    def common_profile(value):
        return (1, *(min(value, cap) for cap in caps), 1)

    low, high = 1, max(caps)
    while low < high:
        middle = (low + high + 1) // 2
        if tt_parameter_count(shape, common_profile(middle)) <= budget:
            low = middle
        else:
            high = middle - 1
    initial = common_profile(low)
    current = initial

    def feasible(ranks):
        return (all(1 <= ranks[i] <= caps[i - 1] for i in range(1, order))
                and all(ranks[i + 1] <= ranks[i] * shape[i]
                        and ranks[i] <= shape[i] * ranks[i + 1] for i in range(order)))

    def increase(base, bond):
        # Only the two adjacent cores depend on this bond, so its cost is linear.
        left, right = base[bond - 1], base[bond + 1]
        coefficient = left * shape[bond - 1] + shape[bond] * right
        other_cost = tt_parameter_count(shape, base) - base[bond] * coefficient
        upper = min(caps[bond - 1], left * shape[bond - 1], shape[bond] * right,
                    (budget - other_cost) // coefficient)
        lower = max(current[bond] + 1,
                    (left + shape[bond - 1] - 1) // shape[bond - 1],
                    (right + shape[bond] - 1) // shape[bond])
        if upper < lower:
            return None
        candidate = list(base)
        candidate[bond] = upper
        candidate = tuple(candidate)
        return candidate if feasible(candidate) else None

    def preference(ranks):
        return (-tt_parameter_count(shape, ranks),
                sum(abs(a - b) for a, b in zip(ranks, initial)), ranks)

    for _ in range(_REFINEMENT_PASSES):
        best = current
        for bond in range(1, order):
            candidate = increase(current, bond)
            if candidate is not None and preference(candidate) < preference(best):
                best = candidate
            for donor in range(1, order):
                if donor == bond or current[donor] == 1:
                    continue
                base = list(current)
                base[donor] -= 1
                # A decrease may temporarily violate a neighboring constraint;
                # validate the final redistributed tuple, not this intermediate.
                candidate = increase(base, bond)
                if candidate is not None and preference(candidate) < preference(best):
                    best = candidate
        if tt_parameter_count(shape, best) <= tt_parameter_count(shape, current):
            break
        current = best
    return current
