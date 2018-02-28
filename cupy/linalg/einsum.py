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


def _compute_size_by_dict(indices, idx_dict, broadcasted_dims, operand_idx=()):
    """Compute computation cost.

    If '@' exists in indices we evaluate it as max dims of broadcasted dims in
    operand_idx.
    Args:
        indices (str): Indexes which will be contracted.
        idx_dict (dict): Dimension length for each index.
        broadcasted_dims (dict): Dimension length of broadcasted dims for each
            operand sets.
        operand_idx (tuple of ints): Indices of operands will be computated.
            If (), all operands for einsum is considered.
    Return:
        int: computation cost for indices.
    """
    ret = 1
    for i in indices:
        if i == '@':
            if operand_idx:
                ret *= max([broadcasted_dims[oi] for oi in operand_idx])
            else:
                ret *= max(broadcasted_dims)
        else:
            ret *= idx_dict[i]
    return ret


def _find_contraction(positions, input_sets, output_set):
    """Find which indexes contracted or remained.

    Args:
        positions (pair of ints): Indexes which input set will be contracted.
        input_sets : before contraction.
    Return:
        new_result (set of chars): New input set produced by contraction.
        remaining (list of sets of chars): Remained inpus sets.
        idx_removed (set of chars): Removed indexes after calculation.
        idx_contract: Contracted indexes.
    """

    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)

    return new_result, remaining, idx_removed, idx_contract


def _greedy_path(input_sets, output_set, dim_dict, broadcasted_dims,
                 memory_limit):
    """Finds the best path from all possible combinations.

    """
    if len(input_sets) == 1:
        return [(0,)]

    path = []
    for iteration in six.moves.range(len(input_sets) - 1):
        iteration_results = []
        comb_iter = []

        # Compute all unique pairs
        for x in six.moves.range(len(input_sets)):
            for y in six.moves.range(x + 1, len(input_sets)):
                comb_iter.append((x, y))

        for positions in comb_iter:
            # Find the contraction
            contract = _find_contraction(positions, input_sets, output_set)
            idx_result, new_input_sets, idx_removed, idx_contract = contract

            # Sieve the results based on memory_limit
            if _compute_size_by_dict(idx_result, dim_dict, broadcasted_dims,
                                     comb_iter) > memory_limit:
                continue

            # Build sort tuple
            removed_size = _compute_size_by_dict(idx_removed, dim_dict,
                                                 broadcasted_dims, comb_iter)
            cost = _compute_size_by_dict(idx_contract, dim_dict,
                                         broadcasted_dims, comb_iter)
            sort = (-removed_size, cost)

            # Add contraction to possible choices
            iteration_results.append([sort, positions, new_input_sets])

        # If we did not find a new contraction contract remaining
        if not iteration_results:
            path.append(tuple(six.moves.range(len(input_sets))))
            break

        # Sort based on first index
        best = min(iteration_results, key=lambda x: x[0])
        path.append(best[1])
        input_sets = best[2]

    return path


def _parse_einsum_input(operands):
    if not operands:
        raise ValueError('must specify the einstein sum subscripts string and '
                         'at least one operand, or at least one operand and '
                         'its corresponding subscripts list')

    subscripts = operands[0]
    ioperands = list(operands[1:])

    if not isinstance(subscripts, str):
        raise TypeError('Current cupy einsum support only string subscripts')

    num_input_subscripts = len(subscripts.split(','))
    if num_input_subscripts < len(ioperands):
        raise ValueError('fewer operands provided to einstein sum function '
                         'than specified in the subscripts string')
    if num_input_subscripts > len(ioperands):
        raise ValueError('more operands provided to einstein sum function '
                         'than specified in the subscripts string')

    subscripts = subscripts.replace(' ', '')
    irregular_chars = set(subscripts) - set(string.ascii_letters) - set('->,.')
    if irregular_chars:
        pickup = list(irregular_chars)[0]
        raise ValueError('invalid subscript \'{}\' in einstein sum subscripts '
                         'string, subscripts must be letters'.format(pickup))

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

    converted_ioperands = []
    dtype = numpy.result_type(*ioperands)
    for a in ioperands:
        if isinstance(a, cupy.ndarray):
            converted_ioperands.append(a.astype(dtype))
        else:
            converted_ioperands.append(cupy.asarray(a, dtype=dtype))

    return input_subscripts, output_subscript, converted_ioperands


def input_subscript_sanity_check(subscript, ioperand, index):
    if len(subscript) - subscript.count('@') > ioperand.ndim:
        raise ValueError('einstein sum subscripts string contains too '
                         'many subscripts for operand {}'.format(index))
    if '@' not in subscript and len(subscript) < ioperand.ndim:
        raise ValueError('operand has more dimensions than subscripts'
                         ' given in einstein sum, but no \'...\' ellipsis'
                         ' provided to broadcast the extra dimensions.')
    if subscript.count('@') >= 2:
        raise ValueError('Two or more \'...\' ellipsis can\'t be used for '
                         'one operand')


def einsum_path(*operands, **kwargs):
    """einsum_path(subscripts, *operands)

    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    .. seealso:: :func:`numpy.einsum_path`

    .. note:: 'optimal' option is not supported. 'einsum_call' argument is also
     not supported.

    Args:
        subscripts (str):
            Specifies the subscripts for summation.
        operands (sequence of arrays):
            These are the arrays for the operation.
        optimize (bool or list or tuple or 'greedy'):
            Choose the type of path. If a tuple is provided, the second
            argument is assumed to be the maximum intermediate size created. If
            only a single argument is provided the largest input or output
            array size is used as a maximum intermediate size.
            * if a list is given that starts with ``einsum_path``, uses this as
              the contraction path
            * if False no optimization is taken
            * if True defaults to the 'greedy' algorithm
            * 'greedy' An algorithm that chooses the best pair contraction
              at each step. Effectively, this algorithm searches the largest
              inner, Hadamard, and then outer products at each step. Scales
              cubically with the number of terms in the contraction. Equivalent
              to the 'optimal' path for most contractions.
            Default is 'greedy'.

    Returns:
        opearands:
            Type unified operands.
        contraction_list:
            Contraction list including all contraction which choosed by this
            function. Each element of list has ``contract_inds``,
            ``idx_removed``, ``einsum_str`` and ``out_subscript``.
            ``contract_inds`` indicates which ioperands will be contracted.
            ``idx_removed`` is the set of the indexes which will be removed by
            this contraction.
            ``subscript`` equals input subscript.
            ``out_subscript`` is subscript of contraction result.
            First element is first contraction, second element is second
            contraction, and so on.

    """
    # Make sure all keywords are valid
    valid_contract_kwargs = ['optimize']
    unknown_kwargs = set(kwargs.keys()).difference(valid_contract_kwargs)
    if unknown_kwargs:
        raise TypeError('Did not understand the following kwargs:'
                        ' %s' % unknown_kwargs)

    # Figure out what the path really is
    path_type = kwargs.pop('optimize', True)
    if path_type is True:
        path_type = 'greedy'
    if path_type is None:
        path_type = False

    memory_limit = None

    # No optimization or a named path algorithm
    if (path_type is False) or isinstance(path_type, str):
        pass

    # Path tuple with memory limit
    elif isinstance(path_type, collections.Iterable) and \
            (len(path_type) == 2) and isinstance(path_type[0], str) and \
            isinstance(path_type[1], (int, float)):
        memory_limit = int(path_type[1])
        path_type = path_type[0]

    else:
        raise TypeError('Did not understand the path: %s' % str(path_type))

    # Python side parsing
    input_subscripts, output_subscript, operands = \
        _parse_einsum_input(operands)

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and ensure all dimensions are correct
    dim_dict = {}
    broadcasted_dims = []
    for i, subscript in enumerate(input_list):
        ioperand = operands[i]
        input_subscript_sanity_check(subscript, ioperand, i)
        ellipsis_pos = subscript.find('@')
        for cnum, char in enumerate(subscript):
            if cnum == ellipsis_pos:
                continue
            if ellipsis_pos != -1 and cnum > ellipsis_pos:
                cnum -= len(subscript)
            dim = ioperand.shape[cnum]
            if char in dim_dict:
                if dim_dict[char] != dim:
                    raise ValueError('Size of label \'%s\' for operand %d does'
                                     ' not match previous terms.', char, i)
            else:
                dim_dict[char] = dim
        if ellipsis_pos != -1:
            dim = 1
            upper = ellipsis_pos + ioperand.ndim - len(subscript) + 1
            for j in six.moves.range(ellipsis_pos, upper):
                dim *= ioperand.shape[j]
            broadcasted_dims.append(dim)
    if broadcasted_dims:
        dim_dict['@'] = max(broadcasted_dims)

    # Compute size of each input array plus the output array
    size_list = []
    for subscript in input_list + [output_subscript]:
        size_list.append(_compute_size_by_dict(subscript, dim_dict,
                                               broadcasted_dims))
    max_size = max(size_list)

    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit

    # Compute the path
    if not path_type or (len(input_list) in [1, 2]) or (indices == output_set):
        # Nothing to be optimized, leave it to einsum
        path = [tuple(six.moves.range(len(input_list)))]
    elif path_type == 'greedy':
        # Maximum memory should be at most out_size for this algorithm
        memory_arg = min(memory_arg, max_size)
        path = _greedy_path(input_sets, output_set, dim_dict, broadcasted_dims,
                            memory_arg)
    elif path_type[0] == 'einsum_path':
        path = path_type[1:]
    else:
        raise KeyError('Path name %s not found', path_type)

    cost_list, scale_list, size_list, contraction_list = [], [], [], []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        cost = _compute_size_by_dict(idx_contract, dim_dict, broadcasted_dims)
        if idx_removed:
            cost *= 2
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dim_dict,
                                               broadcasted_dims))

        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))

        # Last contraction
        if cnum == len(path) - 1:
            idx_result = output_subscript
        else:
            sort_result = [(dim_dict[ind], ind) for ind in out_inds]
            idx_result = ''.join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        einsum_str = ','.join(tmp_inputs) + '->' + idx_result

        contraction = (contract_inds, idx_removed, einsum_str, input_list[:])
        contraction_list.append(contraction)

    return operands, contraction_list


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
