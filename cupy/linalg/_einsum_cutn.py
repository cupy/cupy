try:
    from cuquantum import cutensornet
    cutn_handle_cache = {}
except ImportError:
    cutensornet = None

import cupy


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

    return inputs, operands, output


def _try_use_cutensornet(*args, **kwargs):
    if cutensornet is None:
        return None

    if cupy.cuda.runtime.is_hip:
        return None

    dtype = kwargs.get('dtype', None)
    optimize = kwargs.get('optimize', None)
    # raise warning: we don't care optmizie for now?

    args = _get_einsum_operands(args)
    operands = [cupy.asarray(op) for op in args[1]]
    #print("operands dtypes:", [op.dtype for op in operands])

    if (any(op.size == 0 for op in operands) or
            any(len(op.shape) == 0 for op in operands)):
        # To cuTensorNet the shape is invalid
        return None

    result_dtype = cupy.result_type(*operands) if dtype is None else dtype
    if result_dtype not in (
            cupy.float32, cupy.float64, cupy.complex64, cupy.complex128):
        return None
    operands = [op.astype(result_dtype, copy=False) for op in operands]

    device = cupy.cuda.runtime.getDevice()
    handle = cutn_handle_cache.get(
        device, cutensornet.create())
    cutn_options = {'device_id': device, 'handle': handle,
                    'memory_limit': 2**31}
    #print("using cutn... operands dtypes:", [op.dtype for op in operands])
    #assert all(operands[0].dtype == op.dtype for op in operands[1:])
    if len(args) == 2:
        out = cutensornet.contract(args[0], *operands, options=cutn_options)
    elif len(args) == 3:
        out = cutensornet.contract(
            *([i for pair in zip(operands, args[0]) for i in pair] + [args[2]]),
            options=cutn_options)
    else:
        assert False
    #except:
    #    print([op.shape for op in operands])
    #    raise

    return out
