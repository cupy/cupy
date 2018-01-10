import collections

import numpy
import six

import cupy


einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)


def calc_single_view(ioperand, subscript):
    """Calculates 'ii->i' by cupy.diagonal if needed.

    Args:
        ioperand (cupy.ndarray): Array to be calculated diagonal.
        subscript (str):
            Specifies the subscripts. If the same label appears
            more than once, calculate diagonal for those axes.
    """

    if '@' in subscript:
        assert subscript.count('@') == 1
        assert ioperand.ndim >= len(subscript) - 1
    else:
        assert ioperand.ndim == len(subscript)

    subscripts_excluded_at = subscript.replace('@', '')
    labels = set(subscripts_excluded_at)
    label_to_axis = collections.defaultdict(list)
    for i, label in enumerate(subscript):
        label_to_axis[label].append(i)

    result = ioperand
    count_dict = collections.Counter(subscript)
    ellipsis_pos = subscript.find('@')

    for label in labels:
        if count_dict[label] == 1:
            continue
        axes_to_diag = []
        for i, char in enumerate(subscripts_excluded_at):
            if char == label:
                if ellipsis_pos == -1 or i < ellipsis_pos:
                    axes_to_diag.append(i)
                else:
                    axes_to_diag.append(i - len(subscripts_excluded_at))
        axes_to_diag = cupy.core.normalize_axis_tuple(axes_to_diag,
                                                      result.ndim)
        for axis in reversed(axes_to_diag[1:]):
            shape_a = result.shape[axis]
            shape_b = result.shape[axes_to_diag[0]]
            if shape_a != shape_b:
                raise ValueError('dimensions in operand 0 for collapsing'
                                 ' index \'{0}\' don\'t match'
                                 ' ({1} != {2})'.format(label, shape_a,
                                                        shape_b))
            result = result.diagonal(0, axis, axes_to_diag[0])
            result = cupy.rollaxis(result, -1, axes_to_diag[0])
            if ellipsis_pos != -1 and axis > ellipsis_pos:
                axis -= result.ndim - len(subscript) + 1
            subscript = subscript[:axis] + subscript[axis + 1:]
            subscripts_excluded_at = subscript.replace('@', '')
    return result, subscript


def calc_summed_view(ioperand, input_subscript, output_subscript):
    """Calculates 'i->' by cupy.sum if needed.

    Args:
        ioperand (cupy.ndarray): Array to be summed.
        input_subscript (str): Specifies the subscripts for input array.
        output_subscript (str):
            Specifies the subscripts for output array. If one label exists in
            input_subscript but not in output_subscript, this label will be
            summed.
    """

    assert len(set(input_subscript)) == len(input_subscript)
    assert len(set(output_subscript)) == len(output_subscript)
    assert set(output_subscript).issubset(set(input_subscript))

    input_subscript_excluded_at = input_subscript.replace('@', '')

    label_to_summed = set(input_subscript_excluded_at) - set(output_subscript)
    axes_to_summed = []
    ellipsis_pos = input_subscript.find('@')
    for i, label in enumerate(input_subscript_excluded_at):
        if label in label_to_summed:
            if ellipsis_pos == -1 or i < ellipsis_pos:
                axes_to_summed.append(i)
            else:
                axes_to_summed.append(i - len(input_subscript_excluded_at))

    if axes_to_summed:
        result = ioperand.sum(axis=tuple(axes_to_summed)). \
            astype(ioperand.dtype)
    else:
        result = ioperand
    for label in label_to_summed:
        input_subscript = input_subscript.replace(label, '')

    return result, input_subscript


def calc_transposed_view(ioperand, input_subscript, output_subscript):
    """Calculates 'ij->ji' by cupy.transpose if needed.

    Args:
        ioperand (cupy.ndarray): Array to be transpose.
        input_subscript (str): Specifies the subscripts for input arrays.
        output_subscript (str):
            Specifies the subscripts for output arrays. If input does not
            match output, ``operand`` is transposed so that it matches.
    """

    assert len(set(output_subscript)) == len(output_subscript)
    assert set(input_subscript) == set(output_subscript)

    if input_subscript == output_subscript:
        return ioperand

    source_axes = []
    destination_axes = []
    ellipsis_pos_input = input_subscript.find('@')
    ellipsis_pos_output = output_subscript.find('@')

    for label_pos_output, label in enumerate(output_subscript):
        if label == '@':
            continue
        if ellipsis_pos_input == -1 or label_pos_output < ellipsis_pos_output:
            destination_axes.append(label_pos_output)
        else:
            destination_axes.append(label_pos_output - len(output_subscript))
        label_pos_input = input_subscript.find(label)
        if ellipsis_pos_input == -1 or label_pos_input < ellipsis_pos_input:
            source_axes.append(label_pos_input)
        else:
            source_axes.append(label_pos_input - len(input_subscript))

    return cupy.moveaxis(ioperand, source_axes, destination_axes)


def move_broadcast_axes_to_front(ioperands, subscripts):
    broadcasted_operands = []
    broadcasted_subscripts = []
    for operand, subscript in six.moves.zip(ioperands, subscripts):
        if '@' in subscript:
            ellipsis_pos = subscript.find('@')
            source_axes = list(six.moves.range(ellipsis_pos))
            destination_axes = [i - ellipsis_pos for i in source_axes]
            operand = cupy.moveaxis(operand, source_axes, destination_axes)
            subscript = subscript[ellipsis_pos:] + subscript[:ellipsis_pos]
        broadcasted_operands.append(operand)
        broadcasted_subscripts.append(subscript)
    return broadcasted_operands, broadcasted_subscripts


def calc_combined_view(ioperands, subscripts):
    """Calculates 'i,j->ij' by cupy.tensordot.

    Args:
        ioperands (sequence of arrays): Arrays to be combined.
        subscripts (sequence of str): Specifies the subscripts.
    """
    if len(ioperands) == 1:
        return ioperands[0], subscripts[0]

    a_shape_stack = []
    b_shape_stack = []
    is_first_operand = True
    for operand, subscript in six.moves.zip(ioperands, subscripts):
        if subscript and '@' == subscript[0]:
            broadcasted_dims = operand.ndim - len(subscript) + 1
            a_shape = numpy.prod(operand.shape[:broadcasted_dims],
                                 dtype=numpy.uint32)
            if len(operand.shape[:broadcasted_dims]) > len(a_shape_stack):
                a_shape_stack = list(operand.shape[:broadcasted_dims])
            b_shape = numpy.prod(operand.shape[broadcasted_dims:],
                                 dtype=numpy.uint32)
            b_shape_stack += operand.shape[broadcasted_dims:]
            operand = operand.reshape(a_shape, 1, b_shape)
        else:
            b_shape_stack += operand.shape
            operand = operand.reshape(1, 1, operand.size)
        if is_first_operand:
            result = operand
            is_first_operand = False
        else:
            result = cupy.matmul(result, operand)
        result = result.reshape(result.shape[0],
                                result.shape[1] * result.shape[2], 1)

    subscript = ''.join(subscripts)
    if '@' in subscript:
        subscript = '@' + subscript.replace('@', '')
    return result.reshape(a_shape_stack + b_shape_stack), subscript


def parse_einsum_input(operands):
    """Parses einsum operands.

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
    >>> parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b])

    >>> parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
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

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = operand_list
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain "
                                    "either int or Ellipsis")
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                elif isinstance(s, int):
                    subscripts += einsum_symbols[s]
                else:
                    raise TypeError("For this input type lists must contain "
                                    "either int or Ellipsis")
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input"
                             % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    return (input_subscripts, output_subscript, operands)


def einsum(*operands):
    """einsum(subscripts, *operands)

    Evaluates the Einstein summation convention on the operands.
    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion. This function
    provides a way to compute such summations.

    .. note::
       Memory contiguity of calculation result is not always compatible with
       `numpy.einsum`.
       ``out``, ``order``, ``dtype``, ``casting`` and ``optimize`` options
       are not supported.

    Args:
        subscripts (str): Specifies the subscripts for summation.
        operands (sequence of arrays): These are the arrays for the operation.

    Returns:
        cupy.ndarray:
            The calculation based on the Einstein summation convention.

    .. seealso:: :func:`numpy.einsum`

    """

    # TODO(fukatani): Support 'out', 'order', 'dtype', 'casting', 'optimize'

    input_subscripts, output_subscript, ioperands = \
        parse_einsum_input(operands)
    subscripts = '{}->{}'.format(input_subscripts, output_subscript)

    converted_inputs = []
    dtype = numpy.result_type(*ioperands)
    for a in ioperands:
        if isinstance(a, cupy.ndarray):
            converted_inputs.append(a.astype(dtype))
        else:
            converted_inputs.append(cupy.asarray(a, dtype=dtype))

    if output_subscript:
        count_dict = collections.Counter(output_subscript)
        for key in count_dict:
            if count_dict[key] == 1:
                continue
            raise ValueError('einstein sum subscripts string includes output '
                             'subscript \'{}\' multiple times'.format(key))

    input_subscripts_list = input_subscripts.split(',')
    single_views = []
    for i in six.moves.range(len(input_subscripts_list)):
        subscript = input_subscripts_list[i]
        ioperand = converted_inputs[i]
        if len(subscript) > ioperand.ndim:
            raise ValueError('einstein sum subscripts string contains too '
                             'many subscripts for operand {}'.format(i))
        if len(subscript) < ioperand.ndim:
            raise ValueError('operand {} has more dimensions than subscripts'
                             ' given in einstein sum'.format(i))
        result, subscript = calc_single_view(ioperand, subscript)
        single_views.append((result, subscript))

    if len(converted_inputs) >= 2:
        results = [view[0] for view in single_views]
        subscripts = [view[1] for view in single_views]
        results, subscripts = move_broadcast_axes_to_front(results, subscripts)
        result, subscript = calc_combined_view(results, subscripts)
        result, subscript = calc_single_view(result, subscript)
    else:
        result, subscript = single_views[0]

    result, subscript = calc_summed_view(result, subscript, output_subscript)
    result = calc_transposed_view(result, subscript, output_subscript)
    return result
