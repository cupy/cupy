try:
    from cuquantum import cutensornet
    cutn_handle_cache = {}
except ImportError:
    cutensornet = None

import cupy
from cupy._core import _accelerator


def _get_einsum_operands(args):
    """Parse einsum operands.

    Parameters
    ----------
    args : tuple
        The non-keyword arguments to einsum

    Returns
    -------
    operands : list of array_like
        The operands to use in the contraction
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

    dtype = kwargs.get('dtype', None)
    optimize = kwargs.get('optimize', None)
    # TODO(leofang): raise warning: we don't care optmizie for now?

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

    device = cupy.cuda.runtime.getDevice()
    handle = cutn_handle_cache.get(
        device, cutensornet.create())
    cutn_options = {'device_id': device, 'handle': handle,
                    'memory_limit': 2**31}  # TODO(kataoka): fix?
    if len(args) == 2:
        out = cutensornet.contract(args[0], *operands, options=cutn_options)
    elif len(args) == 3:
        inputs = [i for pair in zip(operands, args[0]) for i in pair]
        if args[2] is not None:
            inputs.append(args[2])
        out = cutensornet.contract(*inputs, options=cutn_options)
    else:
        assert False

    return out
