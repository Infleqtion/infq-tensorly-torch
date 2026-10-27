from .factorized_tensordot import tensor_dot_tucker, tensor_dot_cp
import tensorly as tl
tl.set_backend('pytorch')

# Author: Jean Kossaifi

def linear_tucker(tensor, tucker_matrix, transpose=True, channels_first=True):
    """Contract Tucker linear weights without forming an input/core outer product."""
    output_shape, input_shape = tucker_matrix.tensorized_shape
    output_modes = list(range(len(output_shape)))
    input_modes = list(range(len(output_shape), len(output_shape) + len(input_shape)))
    if transpose:
        contracted_modes, remaining_modes = input_modes, output_modes
    else:
        input_shape = output_shape
        contracted_modes, remaining_modes = output_modes, input_modes
    tensor = tensor.reshape(-1, *input_shape)

    # Consume physical input modes before introducing any core output modes.
    for tensor_mode, weight_mode in enumerate(contracted_modes, start=1):
        factor = tucker_matrix.factors[weight_mode]
        # Transpose explicitly: mode_dot(transpose=True) would conjugate complex factors.
        tensor = tl.tenalg.mode_dot(tensor, tl.transpose(factor), tensor_mode)

    tensor = tl.tenalg.tensordot(
        tensor, tucker_matrix.core,
        modes=(list(range(1, tensor.ndim)), contracted_modes))
    for tensor_mode, weight_mode in enumerate(remaining_modes, start=1):
        tensor = tl.tenalg.mode_dot(tensor, tucker_matrix.factors[weight_mode], tensor_mode)
    return tensor

def linear_cp(tensor, cp_matrix, transpose=True):
    if transpose:
        out_features, in_features = len(cp_matrix.tensorized_shape[0]), len(cp_matrix.tensorized_shape[1])
        in_shape = cp_matrix.tensorized_shape[1]
        modes_cp = list(range(out_features, cp_matrix.order))
    else:
        in_features, out_features = len(cp_matrix.tensorized_shape[0]), len(cp_matrix.tensorized_shape[1])
        in_shape = cp_matrix.tensorized_shape[0]
        modes_cp = list(range(in_features))
    tensor = tensor.reshape(-1, *in_shape)

    modes_tensor = list(range(1, tensor.ndim))

    return tensor_dot_cp(tensor, cp_matrix, (modes_tensor, modes_cp))


def linear_blocktt(tensor, tt_matrix, transpose=True):
    if transpose:
        contraction_axis = 1
    else:
        contraction_axis = 0
    ndim = len(tt_matrix.tensorized_shape[contraction_axis])
    tensor = tensor.reshape(-1, *tt_matrix.tensorized_shape[contraction_axis])

    bs = 'a'
    start = ord(bs) + 1
    in_idx = bs + ''.join(chr(i) for i in [start+i for i in range(ndim)])
    factors_idx = []
    for i in range(ndim):
        if transpose:
            idx = [start+ndim*2+i, start+ndim+i, start+i, start+ndim*2+i+1]
        else:
            idx = [start+ndim*2+i, start+i, start+ndim+i, start+ndim*2+i+1]
        factors_idx.append(''.join(chr(j) for j in idx))
    out_idx = bs + ''.join(chr(i) for i in [start + ndim + i for i in range(ndim)])
    eq = in_idx + ',' + ','.join(i for i in factors_idx) + '->' + out_idx
    res = tl.einsum(eq, tensor, *tt_matrix.factors)
    return tl.reshape(res, (tl.shape(res)[0], -1))

