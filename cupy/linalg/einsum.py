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

    assert ioperand.ndim == len(subscript)

    labels = set(subscript)
    label_to_axis = collections.defaultdict(list)
    for i, label in enumerate(subscript):
        label_to_axis[label].append(i)

    result = ioperand
    count_dict = collections.Counter(subscript)
    for label in labels:
        if count_dict[label] == 1:
            continue
        axes_to_diag = []
        for i, char in enumerate(subscript):
            if char == label:
                axes_to_diag.append(i)
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
            subscript = subscript[:axis] + subscript[axis + 1:]
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

    subscript = input_subscript
    label_to_summed = set(input_subscript) - set(output_subscript)
    axes_to_summed = []
    for i, label in enumerate(input_subscript):
        if label in label_to_summed:
            axes_to_summed.append(i)

    if axes_to_summed:
        result = ioperand.sum(axis=tuple(axes_to_summed)). \
            astype(ioperand)
    else:
        result = ioperand
    for label in label_to_summed:
        subscript = subscript.replace(label, '')

    return result, subscript


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

    transpose_orders = []
    for label in output_subscript:
        transpose_orders.append(input_subscript.find(label))
    if transpose_orders == sorted(transpose_orders):
        return ioperand
    else:
        return ioperand.transpose(transpose_orders)


def calc_combined_view(ioperands, subscripts):
    """Calculates 'i,j->ij' by cupy.tensordot.

    Args:
        ioperands (sequence of arrays): Arrays to be combined.
        subscripts (sequence of str): Specifies the subscripts.
    """

    result = ioperands[0]
    for ioperand in ioperands[1:]:
        # TODO(fukatani): add up at here if enable.
        result = cupy.tensordot(result, ioperand, axes=0)
    return result, ''.join(subscripts)


def get_dummy_labels(label_list):
    dummy_label_set = set()
    count_dict = collections.Counter(label_list)
    for label, count in six.iteritems(count_dict):
        if count >= 2:
            dummy_label_set.add(label)
    return dummy_label_set


def einsum(*operands):
    """einsum(subscripts, *operands)

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion. This function
    provides a way to compute such summations.

    .. note::

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

    # TODO(fukatani): Support '...'
    if '.' in subscripts:
        raise TypeError('Current cupy einsum does not support \'...\' '
                        'ellipsis')

    subscripts = subscripts.replace(' ', '')
    irregular_chars = set(subscripts) - set(string.ascii_letters) - set('->,')
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

    match = re.match('^([a-zA-Z,]+)(->[a-zA-Z]*)?$', subscripts)
    if not match:
        raise ValueError('einstein sum subscript string does not contain '
                         'proper \'->\' output specified')

    input_subscripts = match.group(1)
    if match.group(2):
        output_subscript = match.group(2)[2:]

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
        output_subscript = ''.join(sorted(list(out_label_set)))

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
        if len(subscript) > ioperand.ndim:
            raise ValueError('einstein sum subscripts string contains too '
                             'many subscripts for operand {}'.format(i))
        if len(subscript) < ioperand.ndim:
            raise ValueError('operand has more dimensions than subscripts'
                             ' given in einstein sum, but no \'...\' ellipsis'
                             ' provided to broadcast the extra dimensions.')

        result, subscript = calc_single_view(ioperand, subscript)
        single_views.append((result, subscript))

    if len(converted_inputs) >= 2:
        results = [view[0] for view in single_views]
        subscripts = [view[1] for view in single_views]
        result, subscript = calc_combined_view(results, subscripts)
        result, subscript = calc_single_view(result, subscript)
    else:
        result, subscript = single_views[0]

    result, subscript = calc_summed_view(result, subscript, output_subscript)
    return calc_transposed_view(result, subscript, output_subscript)
