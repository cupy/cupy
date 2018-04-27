import functools
import itertools
import operator

import xp

from einsum_opt import _greedy_path, _optimal_path


options = {
    'sum_ellipsis': False,
    'broadcast_diagonal': False,
}


einsum_symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def _concat(lists):
    return sum(lists, [])


def _prod(xs):
    return functools.reduce(operator.mul, xs, 1)


def _transpose_ex(a, axeses):
    """Transpose and diagonal

    Args:
        a
        axeses (list of list of ints)

    Returns:
        p: a with its axes permutated. A writeable view is returned whenever
            possible.
    """

    shape = []
    strides = []
    for axes in axeses:
        dims = [a.shape[axis] for axis in axes]
        dim = max(dims)  # TODO(kataoka): fix to dim=0
        stride = sum(
            0 if d == 1 else a.strides[axis]
            for axis, d in zip(axes, dims)
        )
        shape.append(dim)
        strides.append(stride)
    return xp.set_shape_and_strides(a.view(), shape, strides)


def _parse_int_subscript(sub):
    subscripts = ""
    for s in sub:
        if s is Ellipsis:
            subscripts += "@"
        elif isinstance(s, int):
            subscripts += einsum_symbols[s]
        else:
            raise TypeError("For this input type lists must contain "
                            "either int or Ellipsis")
    return subscripts


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
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = list(operands[1:])

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

        # Parse "..."
        subscripts = subscripts.replace("...", "@")
        if "." in subscripts:
            raise ValueError("Invalid Ellipses.")

        # Parse "->"
        if ("-" in subscripts) or (">" in subscripts):
            # Check for proper "->"
            invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
            subscripts = subscripts.split("->")
            if invalid or len(subscripts) != 2:
                raise ValueError("Subscripts can only contain one '->'.")
            input_subscripts, output_subscript = subscripts

        else:
            input_subscripts = subscripts
            output_subscript = None

        input_subscripts = input_subscripts.split(",")
        if len(input_subscripts) != len(operands):
            raise ValueError("Number of einsum subscripts must be equal to the "
                             "number of operands.")

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


def _chr(char):
    if char < 0:
        return "...[%d]" % char
    else:
        return chr(char)


def _parse_ellipsis_subscript(subscript, ndim=None, ellipsis_len=None):
    subs = subscript.split('@')
    if len(subs) == 1:
        sub, = subs
        if ndim is not None and len(sub) != ndim:
            # raise ValueError later
            return "Einstein sum subscript %s does not contain the correct number of indices " % subs
        return list(map(ord, sub))
    elif len(subs) == 2:
        left_sub, right_sub = subs
        if ndim is not None:
            ellipsis_len = ndim - (len(left_sub) + len(right_sub))
        if ellipsis_len < 0:
            # raise ValueError later
            return "Einstein sum subscript %s...%s does not contain the correct number of indices " % (left_sub, right_sub)
        return list(itertools.chain(
            map(ord, left_sub),
            range(-ellipsis_len, 0),
            map(ord, right_sub),
        ))
    else:
        # >= 2 ellipses for an operand
        raise ValueError("Invalid Ellipses.")


def _einsum_diagonals(input_subscripts, operands):
    """Compute diagonal for each operand

    This function mutates args.
    """

    for num, sub in enumerate(input_subscripts):
        if len(set(sub)) < len(sub):
            op = operands[num]

            axes = {}
            for i, s in enumerate(sub):
                axes.setdefault(s, []).append(i)

            axes = list(axes.items())
            input_subscripts[num] = [
                s
                for s, _ in axes
            ]

            if not options['broadcast_diagonal']:
                for s, indices in axes:
                    dims = list({op.shape[j] for j in indices})
                    if len(dims) >= 2:
                        raise ValueError(
                            "dimensions in operand %d"
                            " for collapsing index '%s' don't match (%d != %d)"
                            % (num, _chr(s), dims[0], dims[1])
                        )

            axes = [
                indices
                for _, indices in axes
            ]
            operands[num] = _transpose_ex(
                op, axes
            )


def einsum(*operands, **kwargs):
    input_subscripts, output_subscript, operands = _parse_einsum_input(operands)
    assert isinstance(input_subscripts, list)
    assert isinstance(operands, list)

    dtype = kwargs.pop('dtype', None)

    optimize = kwargs.pop('optimize', False)
    # assert optimize is False, "optimize: sorry"
    if optimize is True:
        optimize = 'greedy'
    if kwargs:
        raise TypeError("Did not understand the following kwargs: %s"
                        % list(kwargs.keys))

    operands = [
        xp.asanyarray(arr)
        for arr in operands
    ]
    result_dtype = dtype or xp.result_type(*operands)

    input_subscripts = [
        _parse_ellipsis_subscript(sub, ndim=arr.ndim)
        for sub, arr in zip(input_subscripts, operands)
    ]
    for i, sub_or_err in enumerate(input_subscripts):
        if isinstance(sub_or_err, str):
            raise ValueError(sub_or_err + "for operand %d." % i)

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    for tnum, term in enumerate(input_subscripts):
        sh = operands[tnum].shape
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    dim_old = dimension_dict[char]
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                     "does not match previous terms (%d)."
                                     % (_chr(char), tnum, dim, dim_old))
            else:
                dimension_dict[char] = dim

    if output_subscript is None:
        # Build output subscripts
        tmp_subscripts = _concat(input_subscripts)
        output_subscript = [
            s
            for s in sorted(set(tmp_subscripts))
            if s < 0 or tmp_subscripts.count(s) == 1
        ]
    else:
        if not options['sum_ellipsis']:
            if '@' not in output_subscript and -1 in dimension_dict:
                raise ValueError("output had too few broadcast dimensions")
        output_subscript = _parse_ellipsis_subscript(
            output_subscript,
            ellipsis_len=len(list(s for s in dimension_dict.keys() if s < 0))
        )

        # Make sure output subscripts are in the input
        tmp_subscripts = set(_concat(input_subscripts))
        for char in output_subscript:
            if char not in tmp_subscripts:
                raise ValueError(
                    "Output character %s did not appear in the input" % _chr(char))

    _einsum_diagonals(input_subscripts, operands)

    # no raise after this

    if any(op.size == 0 for op in operands):
        return xp.zeros(
            tuple(dimension_dict[s] for s in output_subscript),
            dtype=result_dtype
        )

    if len(operands) >= 2:
        # Don't squeeze if unary, because this affects later (in trivial sum)
        # whether the return is a writeable view.
        for num in range(len(operands)):
            op = operands[num]
            if 1 in op.shape:
                squeeze_indices = []
                sub = []
                for i, s in enumerate(input_subscripts[num]):
                    if op.shape[i] == 1:
                        squeeze_indices.append(i)
                    else:
                        sub.append(s)
                input_subscripts[num] = sub
                operands[num] = xp.squeeze(op, axis=tuple(squeeze_indices))
                assert len(operands[num].shape) == len(input_subscripts[num])

    # unary einsum without summation should return a (writeable) view
    returns_view = len(operands) == 1

    # unary sum
    for num, sub in enumerate(input_subscripts):
        other_subscripts = input_subscripts.copy()
        other_subscripts[num] = output_subscript
        other_subscripts = _concat(other_subscripts)
        sum_axes = tuple(
            i
            for i, s in enumerate(sub)
            if s not in other_subscripts
        )
        if sum_axes:
            returns_view = False
            input_subscripts[num] = [
                s
                for i, s in enumerate(sub)
                if i not in sum_axes
            ]
            op = operands[num]

            # numpy.sum uses platform integer types by default
            operands[num] = op.sum(axis=sum_axes, dtype=dtype or op.dtype)

    """
    count_dict = {k: 0 for k in dimension_dict}
    for sub in input_subscripts:
        for s in sub:
            count_dict[s] += 1
    """

    optimize_algorithms = {
        'greedy': _greedy_path,
        'optimal': _optimal_path,
    }
    if optimize is False:
        path = [(0, 1)] * (len(operands) - 1)  # TODO(kataoka): fix
    elif isinstance(optimize, str) and optimize in optimize_algorithms.keys():
        input_sets = [set(sub) for sub in input_subscripts]
        output_set = set(output_subscript)
        memory_arg = 1e99
        algo = optimize_algorithms[optimize]
        path = algo(input_sets, output_set, dimension_dict, memory_arg)
    elif len(optimize) and (optimize[0] == 'einsum_path'):
        path = optimize[1:]
    else:
        raise TypeError("Did not understand the path (optimize): %s" % str(optimize))

    for idx0, idx1 in path:
        # repeat binary einsum
        assert idx0 < idx1
        sub1 = input_subscripts.pop(idx1)
        op1 = operands.pop(idx1)
        sub0 = input_subscripts.pop(idx0)
        op0 = operands.pop(idx0)

        """
        # This does not work because 0-dim array here might have been >=1-dim
        # einsum.einsum(',i->', 3, np.array([1, 2], np.int16))
        if op0.ndim == 0 and op1.ndim != 0:
            op0 = op0.astype(op1.dtype)
        elif op1.ndim == 0 and op0.ndim != 0:
            op1 = op1.astype(op0.dtype)
        """

        set0 = set(sub0)
        set1 = set(sub1)
        assert len(set0) == len(sub0)
        assert len(set1) == len(sub1)

        set_out = set(_concat([output_subscript] + input_subscripts))
        shared = set0 & set1
        batch_dims = shared & set_out
        contract_dims = shared - batch_dims

        bs0, cs0, ts0 = _make_transpose_axes(sub0, batch_dims, contract_dims)
        bs1, cs1, ts1 = _make_transpose_axes(sub1, batch_dims, contract_dims)

        batch_size = _prod([dimension_dict[s] for s in batch_dims])
        contract_size = _prod([dimension_dict[s] for s in contract_dims])

        tmp0 = op0.transpose(bs0 + ts0 + cs0).reshape(batch_size, -1, contract_size)
        tmp1 = op1.transpose(bs1 + cs1 + ts1).reshape(batch_size, contract_size, -1)
        if dtype and xp.result_type(tmp0, tmp1) != dtype:
            tmp0 = tmp0.astype(dtype)
            tmp1 = tmp1.astype(dtype)
        tmp_out = xp.matmul(tmp0, tmp1)

        sub_b = [sub0[i] for i in bs0]
        assert sub_b == [sub1[i] for i in bs1]
        sub_l = [sub0[i] for i in ts0]
        sub_r = [sub1[i] for i in ts1]

        sub_out = sub_b + sub_l + sub_r
        op_out = tmp_out.reshape([dimension_dict[s] for s in sub_out])

        input_subscripts.append(sub_out)
        operands.append(op_out)

    # unary einsum at last
    op0, = operands
    sub0, = input_subscripts

    transpose_axes = []
    for s in output_subscript:
        try:
            transpose_axes.append(sub0.index(s))
        except ValueError:
            pass

    op_out = op0.transpose(transpose_axes).reshape([
        dimension_dict[s]
        for s in output_subscript
    ])
    if optimize is False:
        if not returns_view and op_out.dtype != result_dtype:
            # assert False  # TODO(kataoka)
            op_out = op_out.astype(result_dtype)
    return op_out


def _tuple_sorted_by_0(zs):
    return tuple(i for _, i in sorted(zs))


def _make_transpose_axes(sub, b_dims, c_dims):
    bs = []
    cs = []
    ts = []
    for i, s in enumerate(sub):
        if s in b_dims:
            bs.append((s, i))
        elif s in c_dims:
            cs.append((s, i))
        else:
            ts.append((s, i))
    return (
        _tuple_sorted_by_0(bs),
        _tuple_sorted_by_0(cs),
        _tuple_sorted_by_0(ts),
    )
"""
    if position == 0:
        it = itertools.chain(sorted(bs), sorted(ts), sorted(cs))
    else:
        it = itertools.chain(sorted(bs), sorted(cs), sorted(ts))
    return tuple(i for _, i in it)
"""
