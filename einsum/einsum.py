einsum_symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
einsum_symbols_set = set(einsum_symbols)


def _concat(lists):
    return sum(lists, [])


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
        operands = operands[1:]

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
            operands.append(tmp_operands.popleft())
            input_subscripts.append(_parse_int_subscript(
                tmp_operands.popleft()))
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
        if ndim is not None:
            left_sub, right_sub = subs
            ellipsis_len = ndim - (len(left_sub) + len(right_sub))
        if ellipsis_len < 0:
            # raise ValueError later
            return "Einstein sum subscript %s...%s does not contain the correct number of indices " % (left_sub, right_sub)
        return list(map(ord, left_sub)) + list(range(-ellipsis_len, 0)) + list(map(ord, right_sub))
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

    if len(operands) >= 2:
        # We don't have to return a view
    for num in range(len(operands)):
        op = operands[num]
        squeeze_indices = [
            i
            for i, n in enumerate(op.shape)
            if n == 1
        ]
        if squeeze_indices:


    count_dict = {k: 0 for k in dimension_dict}
    for sub in input_subscripts:
        for s in sub:
            count_dict[s] += 1

    for idx0, idx1 in path:

    input_subscripts = [sub.split("@") for sub in input_subscripts]
    if any(len(sub) > 2 for sub in input_subscripts):
        # Each subscript
        raise ValueError("Invalid Ellipses.")

