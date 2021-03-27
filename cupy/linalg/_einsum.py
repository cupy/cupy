import copy
import itertools
import string
import warnings

import cupy
from cupy._core import _accelerator
from cupy import _util
from cupy.linalg._einsum_opt import _greedy_path
from cupy.linalg._einsum_opt import _optimal_path


try:
    import cupy_backends.cuda.libs.cutensor  # NOQA
    from cupy import cutensor
except ImportError:
    cutensor = None


options = {
    'sum_ellipsis': False,
    'broadcast_diagonal': False,
}


einsum_symbols = string.ascii_uppercase + string.ascii_lowercase


def _transpose_ex(a, axeses):
    """Transpose and diagonal

    Args:
        a
        axeses (sequence of sequences of ints)

    Returns:
        p: a with its axes permutated. A writeable view is returned whenever
            possible.
    """

    shape = []
    strides = []
    for axes in axeses:
        shape.append(a.shape[axes[0]] if axes else 1)
        stride = sum(a.strides[axis] for axis in axes)
        strides.append(stride)
    a = a.view()
    # TODO(niboshi): Confirm update_x_contiguity flags
    a._set_shape_and_strides(shape, strides, True, True)
    return a


def _parse_int_subscript(list_subscript):
    str_subscript = ''
    for s in list_subscript:
        if s is Ellipsis:
            str_subscript += '@'
        elif isinstance(s, int):
            str_subscript += einsum_symbols[s]
        else:
            raise TypeError(
                'each subscript must be either an integer or an ellipsis'
                ' to provide subscripts strings as lists')
    return str_subscript


def _parse_einsum_input(args):
    """Parse einsum operands.

    This function is based on `numpy.core.einsumfunc._parse_einsum_input`
    function in NumPy 1.14.

    Parameters
    ----------
    args : tuple
        The non-keyword arguments to einsum

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    (['@a, @a'], 'xz', [a, b])

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    (['@a, @a'], 'xz', [a, b])
    """

    if len(args) == 0:
        raise ValueError(
            'must specify the einstein sum subscripts string and at least one '
            'operand, or at least one operand and its corresponding '
            'subscripts list')

    if isinstance(args[0], str):
        subscripts = args[0]
        operands = list(args[1:])

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,-> ':
                continue
            if s not in einsum_symbols:
                raise ValueError(
                    'invalid subscript \'%s\' in einstein sum subscripts '
                    'string, subscripts must be letters' % s)

        # Parse '...'
        subscripts = subscripts.replace('...', '@')
        if '.' in subscripts:
            raise ValueError(
                'einstein sum subscripts string contains a \'.\' that is not '
                'part of an ellipsis (\'...\')')

        # Parse '->'
        if ('-' in subscripts) or ('>' in subscripts):
            # Check for proper '->'
            invalid = subscripts.count('-') > 1 or subscripts.count('>') > 1
            subscripts = subscripts.split('->')
            if invalid or len(subscripts) != 2:
                raise ValueError(
                    'einstein sum subscript string does not contain proper '
                    '\'->\' output specified')
            input_subscripts, output_subscript = subscripts
            output_subscript = output_subscript.replace(' ', '')

        else:
            input_subscripts = subscripts
            output_subscript = None

        input_subscripts = input_subscripts.replace(' ', '').split(',')
        if len(input_subscripts) != len(operands):
            msg = 'more' if len(operands) > len(input_subscripts) else 'fewer'
            raise ValueError(
                msg + ' operands provided to einstein sum function than '
                'specified in the subscripts string')

    else:
        args = list(args)
        operands = []
        input_subscripts = []
        while len(args) >= 2:
            operands.append(args.pop(0))
            input_subscripts.append(_parse_int_subscript(args.pop(0)))
        if args:
            output_subscript = _parse_int_subscript(args[0])
        else:
            output_subscript = None

    return input_subscripts, output_subscript, operands


def _chr(label):
    if label < 0:
        return '...[%d]' % label
    else:
        return chr(label)


def _parse_ellipsis_subscript(subscript, idx, ndim=None, ellipsis_len=None):
    """Parse a subscript that may contain ellipsis

    Args:
        subscript (str): An einsum subscript of an operand or an output. '...'
            should be replaced by '@'.
        idx (int or None): For error messages, give int idx for the idx-th
            operand or None for the output.
        ndim (int, optional): ndim of the operand
        ellipsis_len (int, optional): number of broadcast dimensions of the
            output.

    Returns:
        list of ints: The parsed subscript

    """
    subs = subscript.split('@')
    if len(subs) == 1:
        sub, = subs
        if ndim is not None and len(sub) != ndim:
            if len(sub) > ndim:
                raise ValueError(
                    'einstein sum subscripts string %s contains too many '
                    'subscripts for operand %d' % (sub, idx))
            raise ValueError(
                'operand %d has more dimensions than subscripts string %s '
                'given in einstein sum, but no \'...\' ellipsis provided to '
                'broadcast the extra dimensions.' % (idx, sub))
        return [ord(label) for label in sub]
    elif len(subs) == 2:
        left_sub, right_sub = subs
        if ndim is not None:
            ellipsis_len = ndim - (len(left_sub) + len(right_sub))
        if ellipsis_len < 0:
            raise ValueError(
                'einstein sum subscripts string %s...%s contains too many '
                'subscripts for operand %d' % (left_sub, right_sub, idx))
        ret = []
        ret.extend(ord(label) for label in left_sub)
        ret.extend(range(-ellipsis_len, 0))
        ret.extend(ord(label) for label in right_sub)
        return ret
    else:
        # >= 2 ellipses for an operand
        raise ValueError(
            'einstein sum subscripts string contains a \'.\' that is not '
            'part of an ellipsis (\'...\') ' +
            ('in the output' if idx is None else 'for operand %d' % idx))


def _einsum_diagonals(input_subscripts, operands):
    """Compute diagonal for each operand

    This function mutates args.
    """
    for idx in range(len(input_subscripts)):
        sub = input_subscripts[idx]
        arr = operands[idx]

        if len(set(sub)) < len(sub):
            axeses = {}
            for axis, label in enumerate(sub):
                axeses.setdefault(label, []).append(axis)

            axeses = list(axeses.items())

            for label, axes in axeses:
                if options['broadcast_diagonal']:
                    axes = [axis for axis in axes if arr.shape[axis] != 1]
                dims = {arr.shape[axis] for axis in axes}
                if len(dims) >= 2:
                    dim0 = dims.pop()
                    dim1 = dims.pop()
                    raise ValueError(
                        'dimensions in operand %d'
                        ' for collapsing index \'%s\' don\'t match (%d != %d)'
                        % (idx, _chr(label), dim0, dim1)
                    )

            sub, axeses = zip(*axeses)  # axeses is not empty
            input_subscripts[idx] = list(sub)
            operands[idx] = _transpose_ex(arr, axeses)


def _iter_path_pairs(path):
    """Decompose path into binary path

    Args:
        path (sequence of tuples of ints)

    Yields:
        tuple of ints: pair (idx0, idx1) that represents the operation
            {pop(idx0); pop(idx1); append();}
    """

    for indices in path:
        assert all(idx >= 0 for idx in indices)
        # [3, 1, 4, 9] -> [(9, 4), (-1, 3), (-1, 1)]
        if len(indices) >= 2:
            indices = sorted(indices, reverse=True)
            yield indices[0], indices[1]
            for idx in indices[2:]:
                yield -1, idx


def _flatten_transpose(a, axeses):
    """Transpose and flatten each

    Args:
        a
        axeses (sequence of sequences of ints)

    Returns:
        aT: a with its axes permutated and flatten
        shapes: flattened shapes
    """

    transpose_axes = []
    shapes = []
    for axes in axeses:
        transpose_axes.extend(axes)
        shapes.append([a.shape[axis] for axis in axes])
    return (
        a.transpose(transpose_axes).reshape(
            tuple([cupy._core.internal.prod(shape) for shape in shapes])),
        shapes
    )


def _use_cutensor(dtype0, sub0, dtype1, sub1, batch_dims, contract_dims):
    if dtype0 != dtype1:
        return False
    if dtype0 not in (cupy.float32, cupy.float64,
                      cupy.complex64, cupy.complex128):
        return False
    if (len(contract_dims) >= 1 and (sub0[-1] in batch_dims or
                                     sub1[-1] in batch_dims)):
        return False
    return True


def _get_out_shape(shape0, sub0, shape1, sub1, sub_out):
    extent = {}
    for size, i in zip(shape0 + shape1, sub0 + sub1):
        extent[i] = size
    out_shape = [extent[i] for i in sub_out]
    return out_shape


def _expand_dims_transpose(arr, mode, mode_out):
    """Return a reshaped and transposed array.

    The input array ``arr`` having ``mode`` as its modes is reshaped and
    transposed so that modes of the output becomes ``mode_out``.

    Example
        >>> import cupy
        >>> a = cupy.zeros((10, 20))
        >>> mode_a = ('A', 'B')
        >>> mode_out = ('B', 'C', 'A')
        >>> out = cupy.linalg.einsum._expand_dims_transpose(a, mode_a,
        ...                                                 mode_out)
        >>> out.shape
        (20, 1, 10)

    Args:
        arr (cupy.ndarray):
        mode (tuple or list): The modes of input array.
        mode_out (tuple or list): The modes of output array.

    Returns:
        cupy.ndarray: The reshaped and transposed array.

    """
    mode = list(mode)
    shape = list(arr.shape)
    axes = []
    for i in mode_out:
        if i not in mode:
            mode.append(i)
            shape.append(1)
        axes.append(mode.index(i))
    return cupy.transpose(arr.reshape(shape), axes)


def reduced_binary_einsum(arr0, sub0, arr1, sub1, sub_others):
    set0 = set(sub0)
    set1 = set(sub1)
    assert len(set0) == len(sub0), 'operand 0 should be reduced: diagonal'
    assert len(set1) == len(sub1), 'operand 1 should be reduced: diagonal'

    if len(sub0) == 0 or len(sub1) == 0:
        return arr0 * arr1, sub0 + sub1

    set_others = set(sub_others)
    shared = set0 & set1
    batch_dims = shared & set_others
    contract_dims = shared - batch_dims

    bs0, cs0, ts0 = _make_transpose_axes(sub0, batch_dims, contract_dims)
    bs1, cs1, ts1 = _make_transpose_axes(sub1, batch_dims, contract_dims)

    sub_b = [sub0[axis] for axis in bs0]
    assert sub_b == [sub1[axis] for axis in bs1]
    sub_l = [sub0[axis] for axis in ts0]
    sub_r = [sub1[axis] for axis in ts1]

    sub_out = sub_b + sub_l + sub_r
    assert set(sub_out) <= set_others, 'operands should be reduced: unary sum'

    if len(contract_dims) == 0:
        # Use element-wise multiply when no contraction is needed
        if len(sub_out) == len(sub_others):
            # to assure final output of einsum is C-contiguous
            sub_out = sub_others
        arr0 = _expand_dims_transpose(arr0, sub0, sub_out)
        arr1 = _expand_dims_transpose(arr1, sub1, sub_out)
        return arr0 * arr1, sub_out

    for accelerator in _accelerator.get_routine_accelerators():
        if accelerator == _accelerator.ACCELERATOR_CUTENSOR:
            if _use_cutensor(arr0.dtype, sub0, arr1.dtype, sub1,
                             batch_dims, contract_dims):
                if len(sub_out) == len(sub_others):
                    # to assure final output of einsum is C-contiguous
                    sub_out = sub_others
                out_shape = _get_out_shape(
                    arr0.shape, sub0, arr1.shape, sub1, sub_out)
                arr_out = cupy.empty(out_shape, arr0.dtype)
                arr0 = cupy.ascontiguousarray(arr0)
                arr1 = cupy.ascontiguousarray(arr1)
                desc_0 = cutensor.create_tensor_descriptor(arr0)
                desc_1 = cutensor.create_tensor_descriptor(arr1)
                desc_out = cutensor.create_tensor_descriptor(arr_out)
                arr_out = cutensor.contraction(
                    1.0,
                    arr0, desc_0, sub0,
                    arr1, desc_1, sub1,
                    0.0,
                    arr_out, desc_out, sub_out)
                return arr_out, sub_out

    tmp0, shapes0 = _flatten_transpose(arr0, [bs0, ts0, cs0])
    tmp1, shapes1 = _flatten_transpose(arr1, [bs1, cs1, ts1])
    shapes_out = shapes0[0] + shapes0[1] + shapes1[2]
    assert shapes0[0] == shapes1[0]
    arr_out = cupy.matmul(tmp0, tmp1).reshape(shapes_out)
    return arr_out, sub_out


def _make_transpose_axes(sub, b_dims, c_dims):
    bs = []
    cs = []
    ts = []
    for axis, label in enumerate(sub):
        if label in b_dims:
            bs.append((label, axis))
        elif label in c_dims:
            cs.append((label, axis))
        else:
            ts.append((label, axis))
    return (
        _tuple_sorted_by_0(bs),
        _tuple_sorted_by_0(cs),
        _tuple_sorted_by_0(ts),
    )


def _tuple_sorted_by_0(zs):
    return tuple(i for _, i in sorted(zs))


def einsum(*operands, **kwargs):
    """einsum(subscripts, *operands, dtype=False)

    Evaluates the Einstein summation convention on the operands.
    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion. This function
    provides a way to compute such summations.

    .. note::
       Memory contiguity of calculation result is not always compatible with
       `numpy.einsum`.
       ``out``, ``order``, and ``casting`` options are not supported.

    Args:
        subscripts (str): Specifies the subscripts for summation.
        operands (sequence of arrays): These are the arrays for the operation.

    Returns:
        cupy.ndarray:
            The calculation based on the Einstein summation convention.

    .. seealso:: :func:`numpy.einsum`

    """

    input_subscripts, output_subscript, operands = \
        _parse_einsum_input(operands)
    assert isinstance(input_subscripts, list)
    assert isinstance(operands, list)

    dtype = kwargs.pop('dtype', None)

    # casting = kwargs.pop('casting', 'safe')
    casting_kwargs = {}  # casting is not supported yet in astype

    optimize = kwargs.pop('optimize', False)
    if optimize is True:
        optimize = 'greedy'
    if kwargs:
        raise TypeError('Did not understand the following kwargs: %s'
                        % list(kwargs.keys))

    result_dtype = cupy.result_type(*operands) if dtype is None else dtype
    operands = [
        cupy.asanyarray(arr)
        for arr in operands
    ]

    input_subscripts = [
        _parse_ellipsis_subscript(sub, idx, ndim=arr.ndim)
        for idx, (sub, arr) in enumerate(zip(input_subscripts, operands))
    ]

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    for idx, sub in enumerate(input_subscripts):
        sh = operands[idx].shape
        for axis, label in enumerate(sub):
            dim = sh[axis]
            if label in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[label] == 1:
                    dimension_dict[label] = dim
                elif dim not in (1, dimension_dict[label]):
                    dim_old = dimension_dict[label]
                    raise ValueError(
                        'Size of label \'%s\' for operand %d (%d) '
                        'does not match previous terms (%d).'
                        % (_chr(label), idx, dim, dim_old))
            else:
                dimension_dict[label] = dim

    if output_subscript is None:
        # Build output subscripts
        tmp_subscripts = list(itertools.chain.from_iterable(input_subscripts))
        output_subscript = [
            label
            for label in sorted(set(tmp_subscripts))
            if label < 0 or tmp_subscripts.count(label) == 1
        ]
    else:
        if not options['sum_ellipsis']:
            if '@' not in output_subscript and -1 in dimension_dict:
                raise ValueError(
                    'output has more dimensions than subscripts '
                    'given in einstein sum, but no \'...\' ellipsis '
                    'provided to broadcast the extra dimensions.')
        output_subscript = _parse_ellipsis_subscript(
            output_subscript, None,
            ellipsis_len=sum(label < 0 for label in dimension_dict.keys())
        )

        # Make sure output subscripts are in the input
        tmp_subscripts = set(itertools.chain.from_iterable(input_subscripts))
        for label in output_subscript:
            if label not in tmp_subscripts:
                raise ValueError(
                    'einstein sum subscripts string included output subscript '
                    '\'%s\' which never appeared in an input' % _chr(label))
        if len(output_subscript) != len(set(output_subscript)):
            for label in output_subscript:
                if output_subscript.count(label) >= 2:
                    raise ValueError(
                        'einstein sum subscripts string includes output '
                        'subscript \'%s\' multiple times' % _chr(label))

    _einsum_diagonals(input_subscripts, operands)

    # no more raises

    if len(operands) >= 2:
        if any(arr.size == 0 for arr in operands):
            return cupy.zeros(
                tuple(dimension_dict[label] for label in output_subscript),
                dtype=result_dtype
            )

        # Don't squeeze if unary, because this affects later (in trivial sum)
        # whether the return is a writeable view.
        for idx in range(len(operands)):
            arr = operands[idx]
            if 1 in arr.shape:
                squeeze_indices = []
                sub = []
                for axis, label in enumerate(input_subscripts[idx]):
                    if arr.shape[axis] == 1:
                        squeeze_indices.append(axis)
                    else:
                        sub.append(label)
                input_subscripts[idx] = sub
                operands[idx] = cupy.squeeze(arr, axis=tuple(squeeze_indices))
                assert operands[idx].ndim == len(input_subscripts[idx])
            del arr

    # unary einsum without summation should return a (writeable) view
    returns_view = len(operands) == 1

    # unary sum
    for idx, sub in enumerate(input_subscripts):
        other_subscripts = copy.copy(input_subscripts)
        other_subscripts[idx] = output_subscript
        other_subscripts = set(itertools.chain.from_iterable(other_subscripts))
        sum_axes = tuple(
            axis
            for axis, label in enumerate(sub)
            if label not in other_subscripts
        )
        if sum_axes:
            returns_view = False
            input_subscripts[idx] = [
                label
                for axis, label in enumerate(sub)
                if axis not in sum_axes
            ]

            operands[idx] = operands[idx].sum(
                axis=sum_axes, dtype=result_dtype)

    if returns_view:
        operands = [a.view() for a in operands]
    else:
        operands = [
            a.astype(result_dtype, copy=False, **casting_kwargs)
            for a in operands
        ]

    # no more casts

    optimize_algorithms = {
        'greedy': _greedy_path,
        'optimal': _optimal_path,
    }
    if optimize is False:
        path = [tuple(range(len(operands)))]
    elif len(optimize) and (optimize[0] == 'einsum_path'):
        path = optimize[1:]
    else:
        try:
            if len(optimize) == 2 and isinstance(optimize[1], (int, float)):
                algo = optimize_algorithms[optimize[0]]
                memory_limit = int(optimize[1])
            else:
                algo = optimize_algorithms[optimize]
                memory_limit = 2 ** 31  # TODO(kataoka): fix?
        except (TypeError, KeyError):  # unhashable type or not found
            raise TypeError('Did not understand the path (optimize): %s'
                            % str(optimize))
        input_sets = [set(sub) for sub in input_subscripts]
        output_set = set(output_subscript)
        path = algo(input_sets, output_set, dimension_dict, memory_limit)
        if any(len(indices) > 2 for indices in path):
            warnings.warn(
                'memory efficient einsum is not supported yet',
                _util.PerformanceWarning)

    for idx0, idx1 in _iter_path_pairs(path):
        # "reduced" binary einsum
        arr0 = operands.pop(idx0)
        sub0 = input_subscripts.pop(idx0)
        arr1 = operands.pop(idx1)
        sub1 = input_subscripts.pop(idx1)
        sub_others = list(itertools.chain(
            output_subscript,
            itertools.chain.from_iterable(input_subscripts)))
        arr_out, sub_out = reduced_binary_einsum(
            arr0, sub0, arr1, sub1, sub_others)
        operands.append(arr_out)
        input_subscripts.append(sub_out)
        del arr0, arr1

    # unary einsum at last
    arr0, = operands
    sub0, = input_subscripts

    transpose_axes = []
    for label in output_subscript:
        if label in sub0:
            transpose_axes.append(sub0.index(label))

    arr_out = arr0.transpose(transpose_axes).reshape([
        dimension_dict[label]
        for label in output_subscript
    ])
    assert returns_view or arr_out.dtype == result_dtype
    return arr_out
