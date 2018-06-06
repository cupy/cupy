import functools
import operator
import string
import warnings

import six.moves

import cupy
from cupy.linalg.einsum_opt import _greedy_path
from cupy.linalg.einsum_opt import _optimal_path


options = {
    'sum_ellipsis': False,
    'broadcast_diagonal': False,
}


einsum_symbols = string.ascii_uppercase + string.ascii_lowercase


def _concat(lists):
    return sum(lists, [])


def _prod(xs):
    return functools.reduce(operator.mul, xs, 1)


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
    a._set_shape_and_strides(shape, strides)
    return a


def _parse_int_subscript(list_subscript):
    str_subscript = ''
    for s in list_subscript:
        if s is Ellipsis:
            str_subscript += '@'
        elif isinstance(s, int):
            str_subscript += einsum_symbols[s]
        else:
            raise ValueError(
                "each subscript must be either an integer or an ellipsis"
                " to provide subscripts strings as lists")
    return str_subscript


def _parse_einsum_input(operands, parse_ellipsis=True):
    """Parse einsum operands.

    This function is based on `numpy.core.einsumfunc._parse_einsum_input`
    function in NumPy 1.14.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    (['@a, @a'], 'xz', [a, b])

    >>> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    (['@a, @a'], 'xz', [a, b])
    """

    if len(operands) == 0:
        raise ValueError(
            "must specify the einstein sum subscripts string and at least one "
            "operand, or at least one operand and its corresponding "
            "subscripts list")

    if isinstance(operands[0], str):
        subscripts = operands[0]
        operands = list(operands[1:])

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,-> ':
                continue
            if s not in einsum_symbols:
                raise ValueError(
                    "invalid subscript '%s' in einstein sum subscripts string,"
                    " subscripts must be letters" % s)

        # Parse '...'
        subscripts = subscripts.replace('...', '@')
        if '.' in subscripts:
            raise ValueError(
                "einstein sum subscripts string contains a '.' that is not "
                "part of an ellipsis ('...')")

        # Parse '->'
        if ('-' in subscripts) or ('>' in subscripts):
            # Check for proper '->'
            invalid = subscripts.count('-') > 1 or subscripts.count('>') > 1
            subscripts = subscripts.split('->')
            if invalid or len(subscripts) != 2:
                raise ValueError(
                    "einstein sum subscript string does not contain proper "
                    "'->' output specified")
            input_subscripts, output_subscript = subscripts
            output_subscript = output_subscript.replace(' ', '')

        else:
            input_subscripts = subscripts
            output_subscript = None

        input_subscripts = input_subscripts.replace(' ', '').split(',')
        if len(input_subscripts) != len(operands):
            raise ValueError(
                ("more" if len(operands) > len(input_subscripts) else "fewer")
                +
                " operands provided to einstein sum function than specified "
                "in the subscripts string")

    else:
        tmp_operands = list(operands)
        operands = []
        input_subscripts = []
        while len(tmp_operands) >= 2:
            operands.append(tmp_operands.pop(0))
            input_subscripts.append(_parse_int_subscript(
                tmp_operands.pop(0)))
        if tmp_operands:
            output_subscript = _parse_int_subscript(tmp_operands[0])
        else:
            output_subscript = None

    return input_subscripts, output_subscript, operands


def _chr(label):
    if label < 0:
        return '...[%d]' % label
    else:
        return chr(label)


def _parse_ellipsis_subscript(subscript, k, ndim=None, ellipsis_len=None):
    """Parse a subscript that may contain ellipsis

    Args:
        subscript (str): An einsum subscript of an operand or an output. '...'
            should be replaced by '@'.
        k (int or None): For error messages, give int k for the k-th operand
            or None for the output.
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
                    "einstein sum subscripts string %s contains too many "
                    "subscripts for operand %d" % (sub, k))
            raise ValueError(
                "operand %d has more dimensions than subscripts string %s "
                "given in einstein sum, but no '...' ellipsis provided to "
                "broadcast the extra dimensions." % (k, sub))
        return [ord(label) for label in sub]
    elif len(subs) == 2:
        left_sub, right_sub = subs
        if ndim is not None:
            ellipsis_len = ndim - (len(left_sub) + len(right_sub))
        if ellipsis_len < 0:
            raise ValueError(
                "einstein sum subscripts string %s...%s contains too many "
                "subscripts for operand %d" % (left_sub, right_sub, k))
        ret = []
        ret.extend([ord(label) for label in left_sub])
        ret.extend(list(range(-ellipsis_len, 0)))
        ret.extend([ord(label) for label in right_sub])
        return ret
    else:
        # >= 2 ellipses for an operand
        raise ValueError(
            "einstein sum subscripts string contains a '.' that is not "
            "part of an ellipsis ('...') " +
            ("in the output" if k is None else "for operand %d" % k))


def _einsum_diagonals(input_subscripts, operands):
    """Compute diagonal for each operand

    This function mutates args.
    """
    for num in six.moves.range(len(input_subscripts)):
        sub = input_subscripts[num]
        arr = operands[num]

        if len(set(sub)) < len(sub):
            axes = {}
            for i, label in enumerate(sub):
                axes.setdefault(label, []).append(i)

            axes = list(axes.items())

            for label, indices in axes:
                if options['broadcast_diagonal']:
                    indices = [j for j in indices if arr.shape[j] != 1]
                dims = {arr.shape[j] for j in indices}
                if len(dims) >= 2:
                    dim0 = dims.pop()
                    dim1 = dims.pop()
                    raise ValueError(
                        "dimensions in operand %d"
                        " for collapsing index '%s' don't match (%d != %d)"
                        % (num, _chr(label), dim0, dim1)
                    )

            sub, axes = zip(*axes)
            input_subscripts[num] = list(sub)
            operands[num] = _transpose_ex(arr, list(axes))


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
            indices = list(sorted(indices, reverse=True))
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

    shapes = [
        [a.shape[axis] for axis in axes]
        for axes in axeses
    ]
    return (
        a.transpose(sum(axeses, ())).reshape(
            tuple(_prod(shape) for shape in shapes)),
        shapes
    )


def reduced_binary_einsum(arr0, sub0, arr1, sub1, sub_others):
    set0 = set(sub0)
    set1 = set(sub1)
    assert len(set0) == len(sub0), "operand 0 should be reduced: diagonal"
    assert len(set1) == len(sub1), "operand 1 should be reduced: diagonal"

    set_others = set(sub_others)
    shared = set0 & set1
    batch_dims = shared & set_others
    contract_dims = shared - batch_dims

    bs0, cs0, ts0 = _make_transpose_axes(sub0, batch_dims, contract_dims)
    bs1, cs1, ts1 = _make_transpose_axes(sub1, batch_dims, contract_dims)

    tmp0, shapes0 = _flatten_transpose(arr0, [bs0, ts0, cs0])
    tmp1, shapes1 = _flatten_transpose(arr1, [bs1, cs1, ts1])
    shapes_out = shapes0[0] + shapes0[1] + shapes1[2]
    assert shapes0[0] == shapes1[0]
    arr_out = cupy.matmul(tmp0, tmp1).reshape(shapes_out)

    sub_b = [sub0[i] for i in bs0]
    assert sub_b == [sub1[i] for i in bs1]
    sub_l = [sub0[i] for i in ts0]
    sub_r = [sub1[i] for i in ts1]

    sub_out = sub_b + sub_l + sub_r
    assert set(sub_out) <= set_others, "operands should be reduced: unary sum"

    return arr_out, sub_out


def _make_transpose_axes(sub, b_dims, c_dims):
    bs = []
    cs = []
    ts = []
    for i, label in enumerate(sub):
        if label in b_dims:
            bs.append((label, i))
        elif label in c_dims:
            cs.append((label, i))
        else:
            ts.append((label, i))
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
        raise TypeError("Did not understand the following kwargs: %s"
                        % list(kwargs.keys))

    result_dtype = cupy.result_type(*operands) if dtype is None else dtype
    operands = [
        cupy.asanyarray(arr)
        for arr in operands
    ]

    input_subscripts = [
        _parse_ellipsis_subscript(sub, k, ndim=arr.ndim)
        for k, (sub, arr) in enumerate(zip(input_subscripts, operands))
    ]

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    for tnum, term in enumerate(input_subscripts):
        sh = operands[tnum].shape
        for cnum, label in enumerate(term):
            dim = sh[cnum]
            if label in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[label] == 1:
                    dimension_dict[label] = dim
                elif dim not in (1, dimension_dict[label]):
                    dim_old = dimension_dict[label]
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                     "does not match previous terms (%d)."
                                     % (_chr(label), tnum, dim, dim_old))
            else:
                dimension_dict[label] = dim

    if output_subscript is None:
        # Build output subscripts
        tmp_subscripts = _concat(input_subscripts)
        output_subscript = [
            label
            for label in sorted(set(tmp_subscripts))
            if label < 0 or tmp_subscripts.count(label) == 1
        ]
    else:
        if not options['sum_ellipsis']:
            if '@' not in output_subscript and -1 in dimension_dict:
                raise ValueError(
                    "output has more dimensions than subscripts "
                    "given in einstein sum, but no '...' ellipsis "
                    "provided to broadcast the extra dimensions.")
        output_subscript = _parse_ellipsis_subscript(
            output_subscript, None,
            ellipsis_len=len(
                [label for label in dimension_dict.keys() if label < 0])
        )

        # Make sure output subscripts are in the input
        tmp_subscripts = set(_concat(input_subscripts))
        for label in output_subscript:
            if label not in tmp_subscripts:
                raise ValueError(
                    "einstein sum subscripts string included output subscript "
                    "'%s' which never appeared in an input" % _chr(label))
        if len(output_subscript) != len(set(output_subscript)):
            for label in output_subscript:
                if output_subscript.count(label) >= 2:
                    raise ValueError(
                        "einstein sum subscripts string includes output "
                        "subscript '%s' multiple times" % _chr(label))

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
        for num in six.moves.range(len(operands)):
            arr = operands[num]
            if 1 in arr.shape:
                squeeze_indices = []
                sub = []
                for i, label in enumerate(input_subscripts[num]):
                    if arr.shape[i] == 1:
                        squeeze_indices.append(i)
                    else:
                        sub.append(label)
                input_subscripts[num] = sub
                operands[num] = cupy.squeeze(arr, axis=tuple(squeeze_indices))
                assert len(operands[num].shape) == len(input_subscripts[num])

    # unary einsum without summation should return a (writeable) view
    returns_view = len(operands) == 1

    # unary sum
    for num, sub in enumerate(input_subscripts):
        other_subscripts = list(input_subscripts)
        other_subscripts[num] = output_subscript
        other_subscripts = _concat(other_subscripts)
        sum_axes = tuple(
            i
            for i, label in enumerate(sub)
            if label not in other_subscripts
        )
        if sum_axes:
            returns_view = False
            input_subscripts[num] = [
                label
                for i, label in enumerate(sub)
                if i not in sum_axes
            ]

            # Cannot do the following in cupy (bug?)
            # operands[num] = operands[num].sum(
            #     axis=sum_axes, dtype=result_dtype)

            operands[num] = (
                operands[num]
                .astype(result_dtype, copy=False, **casting_kwargs)
                .sum(axis=sum_axes)
                # .sum uses platform integer types by default
                .astype(result_dtype, copy=False)
            )

    if returns_view:
        operands = [arr.view() for arr in operands]
    else:
        operands = [
            arr.astype(result_dtype, copy=False, **casting_kwargs)
            for arr in operands
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
            raise TypeError("Did not understand the path (optimize): %s"
                            % str(optimize))
        input_sets = [set(sub) for sub in input_subscripts]
        output_set = set(output_subscript)
        path = algo(input_sets, output_set, dimension_dict, memory_limit)
        if any(len(indices) > 2 for indices in path):
            warnings.warn(RuntimeWarning(
                "memory efficient einsum is not supported yet"))

    for idx0, idx1 in _iter_path_pairs(path):
        # "reduced" binary einsum
        arr0 = operands.pop(idx0)
        sub0 = input_subscripts.pop(idx0)
        arr1 = operands.pop(idx1)
        sub1 = input_subscripts.pop(idx1)
        sub_others = _concat([output_subscript] + input_subscripts)
        arr_out, sub_out = reduced_binary_einsum(
            arr0, sub0, arr1, sub1, sub_others)
        operands.append(arr_out)
        input_subscripts.append(sub_out)

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
