import warnings

try:
    from cuquantum import cutensornet
    cutn_handle_cache = {}
except ImportError:
    cutensornet = None

import cupy
from cupy._core import _accelerator


def _get_einsum_operands(args):
    """Parse & retrieve einsum operands, assuming ``args`` is in either
    "subscript" or "interleaved" format.
    """

    if len(args) == 0:
        raise ValueError(
            'must specify the einstein sum subscripts string and at least one '
            'operand, or at least one operand and its corresponding '
            'subscripts list')

    if isinstance(args[0], str):
        expr = args[0]
        operands = list(args[1:])
        return expr, operands
    else:
        args = list(args)
        operands = []
        inputs = []
        output = None
        while len(args) >= 2:
            operands.append(args.pop(0))
            inputs.append(args.pop(0))
        if len(args) == 1:
            output = args.pop(0)
        assert not args
        return inputs, operands, output


def _try_use_cutensornet(*args, **kwargs):
    if cutensornet is None:
        return None

    if (_accelerator.ACCELERATOR_CUTENSORNET not in
            _accelerator.get_routine_accelerators()):
        return None

    if cupy.cuda.runtime.is_hip:
        return None

    # cannot pop as we might still need kwargs later
    dtype = kwargs.get('dtype', None)
    path = kwargs.get('optimize', None)

    # we do very lightweight pre-processing here just to inspect the
    # operands; the actual input verification is deferred to cuTensorNet
    # which can generate far better diagonostic messages
    args = _get_einsum_operands(args)
    operands = [cupy.asarray(op) for op in args[1]]

    if len(operands) == 1:
        # As of cuTENSOR 1.5.0 it still chokes with some common operations
        # like trace ("ii->") so it's easier to just skip all single-operand
        # cases instead of whitelisting what could be done explicitly
        return None

    if (any(op.size == 0 for op in operands) or
            any(len(op.shape) == 0 for op in operands)):
        # To cuTensorNet the shape is invalid
        return None

    # all input dtypes must be identical (to a numerical dtype)
    result_dtype = cupy.result_type(*operands) if dtype is None else dtype
    if result_dtype not in (
            cupy.float32, cupy.float64, cupy.complex64, cupy.complex128):
        return None
    operands = [op.astype(result_dtype, copy=False) for op in operands]

    # TODO(leofang): support a new optimize option to turn on cutn pathfinder
    if path is False:
        # following the same convention (contracting from the right) as would
        # be produced by _iter_path_pairs(), but converting to a list of pairs
        # due to cuTensorNet's requirement
        path = [(i-1, i-2) for i in range(len(operands), 1, -1)]
    elif (path is not None) and (path is not True) and (
            path[0] == 'einsum_path'):
        # let cuTensorNet check if the format is correct
        path = path[1:]
    else:
        path = None
        warnings.warn(
            'the cuTensorNet backend ignores the "optimize" option '
            'except when an explicit contraction path is provided '
            'or when optimize=False (disable optimization)',
            UserWarning, stacklevel=3)

    device = cupy.cuda.runtime.getDevice()
    handle = cutn_handle_cache.get(
        device, cutensornet.create())
    cutn_options = {'device_id': device, 'handle': handle,
                    'memory_limit': 4**31}  # TODO(leofang): fix?
    cutn_optimizer = {'path': path} if path else None

    if len(args) == 2:
        out = cutensornet.contract(
            args[0], *operands, options=cutn_options, optimize=cutn_optimizer)
    elif len(args) == 3:
        inputs = [i for pair in zip(operands, args[0]) for i in pair]
        if args[2] is not None:
            inputs.append(args[2])
        out = cutensornet.contract(
            *inputs, options=cutn_options, optimize=cutn_optimizer)
    else:
        assert False

    return out
