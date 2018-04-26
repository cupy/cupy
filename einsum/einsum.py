import functools
import itertools
import operator

import xp


einsum_symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
einsum_symbols_set = set(einsum_symbols)


def _concat(lists):
    return sum(lists, [])


def _prod(xs):
    return functools.reduce(operator.mul, xs, 1)


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
    >>> __parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b])

    >>> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b])
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


def einsum(*operands):
    input_subscripts, output_subscript, operands = _parse_einsum_input(operands)

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

    path = [(0, 1)] * (len(operands) - 1)  # TODO(kataoka): optimize

    # diagonal
    for num, sub in enumerate(input_subscripts):
        i = 0
        while i < len(sub):
            s = sub[i]
            if sub.count(s) > 1:
                indices = []
                sub = []
                for j, t in enumerate(input_subscripts[num]):
                    if j == i:
                        indices.append(j)
                        sub.append(t)
                    elif t == s:
                        indices.append(j)
                    else:
                        sub.append(t)
                input_subscripts[num] = sub

                diag_ndim = len(indices)
                op = operands[num]
                dim = op.shape[i]
                op = xp.moveaxis(
                    op,
                    tuple(indices), tuple(range(diag_ndim))
                )
                operands[num] = xp.moveaxis(
                    op[(xp.arange(dim),) * diag_ndim],
                    0, i
                )
            del s
            i += 1
        del i

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
            input_subscripts[num] = [
                s
                for i, s in enumerate(sub)
                if i not in sum_axes
            ]
            op = operands[num]
            operands[num] = op.sum(axis=sum_axes).astype(op.dtype)

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

    """
    count_dict = {k: 0 for k in dimension_dict}
    for sub in input_subscripts:
        for s in sub:
            count_dict[s] += 1
    """

    for idx0, idx1 in path:
        # repeat binary einsum
        assert idx0 < idx1
        sub1 = input_subscripts.pop(idx1)
        op1 = operands.pop(idx1)
        sub0 = input_subscripts.pop(idx0)
        op0 = operands.pop(idx0)

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

    return op0.transpose(transpose_axes).reshape([
        dimension_dict[s]
        for s in output_subscript
    ])


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
