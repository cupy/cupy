import collections
import re
import string

import numpy
import six

import cupy


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


def get_dummy_labels(label_list):
    dummy_label_set = set()
    count_dict = collections.Counter(label_list)
    for label, count in six.iteritems(count_dict):
        if label != '@' and count >= 2:
            dummy_label_set.add(label)
    return dummy_label_set


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

    if not operands:
        raise ValueError('must specify the einstein sum subscripts string and '
                         'at least one operand, or at least one operand and '
                         'its corresponding subscripts list')

    subscripts = operands[0]
    ioperands = operands[1:]

    if not isinstance(subscripts, str):
        raise TypeError('Current cupy einsum support only string subscripts')

    subscripts = subscripts.replace(' ', '')
    irregular_chars = set(subscripts) - set(string.ascii_letters) - set('->,.')
    if irregular_chars:
        pickup = list(irregular_chars)[0]
        raise ValueError('invalid subscript \'{}\' in einstein sum subscripts '
                         'string, subscripts must be letters'.format(pickup))

    converted_inputs = []
    dtype = numpy.result_type(*ioperands)
    for a in ioperands:
        if isinstance(a, cupy.ndarray):
            converted_inputs.append(a.astype(dtype))
        else:
            converted_inputs.append(cupy.asarray(a, dtype=dtype))

    # For simplicity of implementation of subscripts interpretation,
    # All '...' is replaced to '@'.
    subscripts = subscripts.replace('...', '@')
    if '.' in subscripts:
        raise ValueError('einstein sum subscripts string contains a \'.\' that'
                         'is not part of an ellipsis (\'...\')')

    match = re.match('^([a-zA-Z@,]+)(->[a-zA-Z@]*)?$', subscripts)
    if not match:
        raise ValueError('einstein sum subscript string does not contain '
                         'proper \'->\' output specified')

    input_subscripts = match.group(1)
    if match.group(2):
        output_subscript = match.group(2)[2:]

        if output_subscript.count('@') >= 2:
            raise ValueError(
                'Two or more \'...\' ellipsis can\'t be used for'
                'output subscript')

        # For compatibility with numpy.
        # numpy.einsum arrows inputs like 'i->i...'.
        # In this case, `...` does not affect the einsum results.
        if '@' not in input_subscripts and '@' in subscripts:
            output_subscript = output_subscript.replace('@', '')

        irregular_chars = set(output_subscript) - set(input_subscripts)
        if irregular_chars:
            pickup = list(irregular_chars)[0]
            raise ValueError('einstein sum subscripts string included output '
                             'subscript \'{}\' which never appeared in an '
                             'input'.format(pickup))

        count_dict = collections.Counter(output_subscript)
        for key in count_dict:
            if count_dict[key] == 1:
                continue
            raise ValueError('einstein sum subscripts string includes output '
                             'subscript \'{}\' multiple times'.format(key))
    else:
        label_list = list(input_subscripts.replace(',', ''))
        out_label_set = set(label_list) - get_dummy_labels(label_list)
        out_label_list = sorted(list(out_label_set))
        output_subscript = ''.join(out_label_list)

    input_subscripts_list = input_subscripts.split(',')
    if len(input_subscripts_list) < len(converted_inputs):
        raise ValueError('fewer operands provided to einstein sum function '
                         'than specified in the subscripts string')
    if len(input_subscripts_list) > len(converted_inputs):
        raise ValueError('more operands provided to einstein sum function '
                         'than specified in the subscripts string')

    single_views = []
    for i in six.moves.range(len(input_subscripts_list)):
        subscript = input_subscripts_list[i]
        ioperand = converted_inputs[i]
        if len(subscript) - subscript.count('@') > ioperand.ndim:
            raise ValueError('einstein sum subscripts string contains too '
                             'many subscripts for operand {}'.format(i))
        if '@' not in subscript and len(subscript) < ioperand.ndim:
            raise ValueError('operand has more dimensions than subscripts'
                             ' given in einstein sum, but no \'...\' ellipsis'
                             ' provided to broadcast the extra dimensions.')
        if subscript.count('@') >= 2:
            raise ValueError('Two or more \'...\' ellipsis can\'t be used for '
                             'one operand')
        if '@' in input_subscripts and '@' not in output_subscript:
            if len(subscript) <= ioperand.ndim:
                raise ValueError('output had too few broadcast dimensions')
            subscript = subscript.replace('@', '')

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
