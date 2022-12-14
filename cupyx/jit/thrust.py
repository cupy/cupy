from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import Data as _Data


def _wrap_thrust_func(headers):
    def wrapper(func):
        class FuncWrapper(_internal_types.BuiltinFunc):
            def call(self, env, *args, **kwargs):
                for header in headers:
                    env.generated.add_code(f'#include <{header}>')
                env.generated.add_code('#include <thrust/execution_policy.h>')
                env.generated.add_code('#include <thrust/functional.h>')
                env.generated.backend = 'nvcc'
                data_args = [_Data.init(a, env) for a in args]
                data_kwargs = {k: _Data.init(kwargs[k], env) for k in kwargs}
                return func(env, *data_args, **data_kwargs)
        return FuncWrapper()
    return wrapper


class _ExecPolicyType(_cuda_types.TypeBase):
    pass


def _assert_exec_policy_type(exec_policy: _Data):
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise TypeError(f'{exec_policy.code} must be execution policy type')


def _assert_pointer_type(a: _Data) -> None:
    # TODO(asi1024): Typecheck for EqualityComparable.
    if not isinstance(a.ctype, _cuda_types.PointerBase):
        raise TypeError(f'`{a.code}` must be of pointer type: `{a.ctype}`')


def _assert_same_type(a: _Data, b: _Data) -> None:
    if a.ctype != b.ctype:
        raise TypeError(
            f'`{a.code}` and `{b.code}` must be of the same type: '
            f'`{a.ctype}` != `{b.ctype}`')


def _assert_same_pointer_type(a: _Data, b: _Data) -> None:
    # TODO(asi1024): Typecheck for EqualityComparable.
    _assert_pointer_type(a)
    _assert_pointer_type(b)
    if a.ctype.child_type != b.ctype.child_type:
        raise TypeError(
            f'`{a.code}` and `{b.code}` must be of the same pointer type: '
            f'`{a.ctype.child_type}` != `{b.type.child_type}`')


def _assert_pointer_of(a: _Data, b: _Data) -> None:
    _assert_pointer_type(a)
    if a.ctype.child_type != b.ctype:
        raise TypeError(
            f'`*{a.code}` and `{b.code}` must be of the same type: '
            f'`{a.ctype.child_type}` != `{b.ctype}`')


host = _Data('thrust::host', _ExecPolicyType())
device = _Data('thrust::device', _ExecPolicyType())
seq = _Data('thrust::seq', _ExecPolicyType())


@_wrap_thrust_func(['thrust/adjacent_difference.h'])
def adjacent_difference(env, exec_policy, first, last, result, binary_op=None):
    """Computes the differences of adjacent elements.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    _assert_same_pointer_type(first, result)
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first, last, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::adjacent_difference({params})', result.ctype)


# TODO(asi1024): Support all_of
# TODO(asi1024): Support any_of


@_wrap_thrust_func(['thrust/binary_search.h'])
def binary_search(env, exec_policy, first, last, *args):
    """Attempts to find the element value with binary search.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)

    if 1 <= len(args) <= 2:
        value = args[0]
        comp = args[1] if len(args) == 2 else None
        _assert_pointer_of(first, value)
        result_ctype = _cuda_types.bool_
    elif 3 <= len(args) <= 4:
        value_first = args[0]
        value_last = args[1]
        result = args[2]
        comp = args[3] if len(args) == 4 else None
        _assert_same_pointer_type(first, value_first)
        _assert_same_type(value_first, value_last)
        result_ctype = result.ctype
    else:
        raise TypeError('Invalid number of inputs of thrust.binary_search')

    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last, *args]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::binary_search({params})', result_ctype)


@_wrap_thrust_func(['thrust/copy.h'])
def copy(env, exec_policy, first, last, result):
    """Copies the elements.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    _assert_same_pointer_type(first, result)
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::copy({params})', result.ctype)


# TODO(asi1024): Add copy_if
# TODO(asi1024): Add copy_n


@_wrap_thrust_func(['thrust/count.h'])
def count(env, exec_policy, first, last, value):
    """Counts the number of elements in [first, last) that equals to ``value``.
    """
    _assert_exec_policy_type(exec_policy)
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::count({params})', _cuda_types.int32)


# TODO(asi1024): Add count_if


@_wrap_thrust_func(['thrust/equal.h'])
def equal(env, exec_policy, first1, last1, first2, binary_pred=None):
    """Returns true if the two ranges are identical.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_pointer_type(first1, first2)
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    args = [exec_policy, first1, last1, first2]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::equal({params})', _cuda_types.bool_)


@_wrap_thrust_func(['thrust/binary_search.h'])
def equal_range(env, exec_policy, first, last, value, comp=None):
    """Attempts to find the element value in an ordered range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::equal_range({params})',
        _cuda_types.Tuple([first.ctype, first.ctype]))


@_wrap_thrust_func(['thrust/scan.h'])
def exclusive_scan(
        env, exec_policy, first, last, result, init=None, binary_op=None):
    """Computes an exclusive prefix sum operation.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    _assert_same_pointer_type(first, result)
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first, last, result]
    if init is not None:
        args.append(init)
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::exclusive_scan({params})', result.ctype)


@_wrap_thrust_func(['thrust/scan.h'])
def exclusive_scan_by_key(
        env, exec_policy, first1, last1, first2, result,
        init=None, binary_pred=None, binary_op=None):
    """Computes an exclusive prefix sum operation by key.
    _assert_exec_policy_type(exec_policy)
    """
    _assert_pointer_type(first1)
    _assert_same_type(first1, last1)
    _assert_same_pointer_type(first2, result)
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first1, last1, first2, result]
    if init is not None:
        args.append(init)
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::exclusive_scan_by_key({params})', result.ctype)


@_wrap_thrust_func(['thrust/fill.h'])
def fill(env, exec_policy, first, last, value):
    """Assigns the value to every element in the range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::fill({params})', _cuda_types.void)


# TODO(asi1024): Add fill_n


@_wrap_thrust_func(['thrust/find.h'])
def find(env, exec_policy, first, last, value):
    """Finds the first iterator whose value equals to ``value``.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::find({params})', first.ctype)


# TODO(asi1024): Add find_if
# TODO(asi1024): Add find_if_not
# TODO(asi1024): Add for_each
# TODO(asi1024): Add for_each_n


@_wrap_thrust_func(['thrust/gather.h'])
def gather(env, exec_policy, map_first, map_last, input_first, result):
    """Copies elements from source into destination  according to a map.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(map_first)
    _assert_same_type(map_first, map_last)
    _assert_same_pointer_type(input_first, result)
    args = [exec_policy, map_first, map_last, input_first, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::gather({params})', result.ctype)


# TODO(asi1024): Add gather_if
# TODO(asi1024): Add generate_n


@_wrap_thrust_func(['thrust/scan.h'])
def inclusive_scan(
        env, exec_policy, first, last, result, binary_op=None):
    """Computes an inclusive prefix sum operation.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first, last)
    _assert_same_pointer_type(first, result)
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first, last, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::inclusive_scan({params})', result.ctype)


@_wrap_thrust_func(['thrust/scan.h'])
def inclusive_scan_by_key(
        env, exec_policy, first1, last1, first2, result,
        binary_pred=None, binary_op=None):
    """Computes an inclusive prefix sum operation by key.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first1)
    _assert_same_type(first1, last1)
    _assert_same_pointer_type(first2, result)
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    args = [exec_policy, first1, last1, first2, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::inclusive_scan_by_key({params})', result.ctype)


@_wrap_thrust_func(['thrust/inner_product.h'])
def inner_product(
        env, exec_policy, first1, last1, first2, init,
        binary_op1=None, binary_op2=None):
    """Calculates an inner product of the ranges.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_pointer_type(first1, first2)
    if binary_op1 is not None:
        raise NotImplementedError('binary_op1 option is not supported')
    if binary_op2 is not None:
        raise NotImplementedError('binary_op2 option is not supported')
    args = [exec_policy, first1, last1, first2, init]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::inner_product({params})', init.ctype)


# TODO(asi1024): Add is_partitioned


@_wrap_thrust_func(['thrust/sort.h'])
def is_sorted(env, exec_policy, first, last, comp=None):
    """Returns true if the range is sorted in ascending order.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::is_sorted({params})', _cuda_types.bool_)


@_wrap_thrust_func(['thrust/sort.h'])
def is_sorted_until(env, exec_policy, first, last, comp=None):
    """Returns the last iterator for which the range is sorted.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::is_sorted_until({params})', first.ctype)


@_wrap_thrust_func(['thrust/binary_search.h'])
def lower_bound(env, exec_policy, first, last, *args):
    """Attempts to find the element value with binary search.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)

    if 1 <= len(args) <= 2:
        value = args[0]
        comp = args[1] if len(args) == 2 else None
        _assert_pointer_of(first, value)
        result_ctype = first.ctype
    elif 3 <= len(args) <= 4:
        value_first = args[0]
        value_last = args[1]
        result = args[2]
        comp = args[3] if len(args) == 4 else None
        _assert_same_pointer_type(first, value_first)
        _assert_same_type(value_first, value_last)
        result_ctype = result.ctype
    else:
        raise TypeError('Invalid number of inputs of thrust.lower_bound')

    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last, *args]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::lower_bound({params})', result_ctype)


class _ConstantIterator(_cuda_types.PointerBase):

    def __str__(self) -> str:
        value_type = self.child_type
        return f'thrust::constant_iterator<{value_type}>'


@_wrap_thrust_func(['thrust/iterator/constant_iterator.h'])
def make_constant_iterator(env, x, i=None):
    """Finds the first positions whose values differ.
    """
    if i is not None:
        raise NotImplementedError('index_type is not supported')
    args = [x]
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::make_constant_iterator({params})',
        _ConstantIterator(x.ctype)
    )


class _CountingIterator(_cuda_types.PointerBase):

    def __str__(self) -> str:
        value_type = self.child_type
        return f'thrust::counting_iterator<{value_type}>'


@_wrap_thrust_func(['thrust/iterator/counting_iterator.h'])
def make_counting_iterator(env, x, i=None):
    """Finds the first positions whose values differ.
    """
    if i is not None:
        raise NotImplementedError('index_type is not supported')
    args = [x]
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::make_counting_iterator({params})',
        _CountingIterator(x.ctype)
    )


# TODO(asi1024): Add make_discard_iterator
# TODO(asi1024): Add make_index_sequence


@_wrap_thrust_func(['thrust/mismatch.h'])
def mismatch(env, exec_policy, first1, last1, first2, pred=None):
    """Finds the first positions whose values differ.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_pointer_type(first1, first2)
    # TODO(asi1024): Typecheck for EqualityComparable.
    if pred is not None:
        raise NotImplementedError('pred option is not supported')
    args = [exec_policy, first1, last1, first2]
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::mismatch({params})',
        _cuda_types.Tuple([first1.ctype, first2.ctype])
    )


# TODO(asi1024): Add none_of
# TODO(asi1024): Add partition
# TODO(asi1024): Add partition_copy
# TODO(asi1024): Add partition_point


@_wrap_thrust_func(['thrust/reduce.h'])
def reduce(env, exec_policy, first, last, init=None, binary_op=None):
    """Generalization of summation.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    args = [exec_policy, first, last]
    if init is not None:
        # TODO(asi1024): Typecheck for init.
        args.append(init)
        return_type = init.ctype
    else:
        return_type = first.ctype.child_type
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::reduce({params})', return_type)


@_wrap_thrust_func(['thrust/reduce.h'])
def reduce_by_key(
        env, exec_policy, keys_first, keys_last, values_first,
        keys_output, values_output, binary_pred=None, binary_op=None):
    """Generalization of reduce to key-value pairs.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(keys_first)
    _assert_same_type(keys_first, keys_last)
    _assert_pointer_type(values_first)
    _assert_pointer_type(keys_output)
    _assert_pointer_type(values_output)
    args = [
        exec_policy, keys_first, keys_last, values_first,
        keys_output, values_output
    ]
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    if binary_op is not None:
        raise NotImplementedError('binary_op option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::reduce_by_key({params})',
        _cuda_types.Tuple([keys_output.ctype, values_output.ctype])
    )


@_wrap_thrust_func(['thrust/remove.h'])
def remove(env, exec_policy, first, last, value):
    """Removes from the range all elements that are rqual to value.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::remove({params})', first.ctype)


@_wrap_thrust_func(['thrust/remove.h'])
def remove_copy(env, exec_policy, first, last, result, value):
    """Removes from the range all elements that are rqual to value.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_pointer_type(result)
    args = [exec_policy, first, last, result, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::remove_copy({params})', result.ctype)


# TODO(asi1024): Add remove_copy_if
# TODO(asi1024): Add remove_if


@_wrap_thrust_func(['thrust/replace.h'])
def replace(env, exec_policy, first, last, old_value, new_value):
    """Replaces every element in the range equal to old_value with new_value.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_same_type(old_value, new_value)
    args = [exec_policy, first, last, old_value, new_value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::replace({params})', _cuda_types.void)


@_wrap_thrust_func(['thrust/replace.h'])
def replace_copy(env, exec_policy, first, last, result, old_value, new_value):
    """Replaces every element in the range equal to old_value with new_value.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_pointer_type(result)
    _assert_same_type(old_value, new_value)
    args = [exec_policy, first, last, result, old_value, new_value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::replace_copy({params})', result.ctype)


# TODO(asi1024): Add replace_copy_if
# TODO(asi1024): Add replace_if


@_wrap_thrust_func(['thrust/reverse.h'])
def reverse(env, exec_policy, first, last):
    """Reverses a range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    args = [exec_policy, first, last]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::reverse({params})', _cuda_types.void)


@_wrap_thrust_func(['thrust/reverse.h'])
def reverse_copy(env, exec_policy, first, last, result):
    """Reverses a range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_pointer_type(result)
    args = [exec_policy, first, last, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::reverse_copy({params})', result.ctype)


@_wrap_thrust_func(['thrust/scatter.h'])
def scatter(env, exec_policy, first, last, map, result):
    """Copies elements from source range into an output range according to map.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    _assert_pointer_type(map)
    _assert_pointer_type(result)
    args = [exec_policy, first, last, map, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::scatter({params})', _cuda_types.void)


# TODO(asi1024): Add scatter_if


@_wrap_thrust_func(['thrust/sequence.h'])
def sequence(env, exec_policy, first, last, init=None, step=None):
    """Fills the range with a sequence of numbers.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    args = [exec_policy, first, last]
    if init is not None:
        args.append(init)
    if step is not None:
        _assert_same_type(init, step)
        args.append(step)
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::sequence({params})', _cuda_types.void)


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_difference(
        env, exec_policy, first1, last1, first2, last2, result, comp=None):
    """Constructs a sorted range that is the set difference of sorted inputs.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_type(first2, last2)
    _assert_pointer_type(result)
    args = [exec_policy, first1, last1, first2, last2, result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::set_difference({params})', result.ctype)


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_difference_by_key(
        env, exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
        values_first1, values_first2, keys_result, values_result, comp=None):
    """Constructs the key-value set difference of sorted inputs.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(keys_first1, keys_last1)
    _assert_same_type(keys_first2, keys_last2)
    _assert_pointer_type(values_first1)
    _assert_pointer_type(values_first2)
    _assert_pointer_type(keys_result)
    _assert_pointer_type(values_result)
    args = [exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
            values_first1, values_first2, keys_result, values_result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::set_difference_by_key({params})',
        _cuda_types.Tuple([keys_result.ctype, values_result.ctype])
    )


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_intersection(
        env, exec_policy, first1, last1, first2, last2, result, comp=None):
    """Constructs a sorted range that is the set intersection of sorted inputs.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_type(first2, last2)
    _assert_pointer_type(result)
    args = [exec_policy, first1, last1, first2, last2, result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::set_intersection({params})', result.ctype)


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_intersection_by_key(
        env, exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
        values_first1, keys_result, values_result, comp=None):
    """Constructs the key-value set intersection of sorted inputs.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(keys_first1, keys_last1)
    _assert_same_type(keys_first2, keys_last2)
    _assert_pointer_type(values_first1)
    _assert_pointer_type(keys_result)
    _assert_pointer_type(values_result)
    args = [exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
            values_first1, keys_result, values_result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::set_intersection_by_key({params})',
        _cuda_types.Tuple([keys_result.ctype, values_result.ctype])
    )


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_symmetric_difference(
        env, exec_policy, first1, last1, first2, last2, result, comp=None):
    """Constructs a sorted range that is the symmetric difference.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_type(first2, last2)
    _assert_pointer_type(result)
    args = [exec_policy, first1, last1, first2, last2, result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::set_symmetric_difference({params})', result.ctype)


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_symmetric_difference_by_key(
        env, exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
        values_first1, values_first2, keys_result, values_result, comp=None):
    """Constructs the key-value symmetric difference of sorted inputs.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(keys_first1, keys_last1)
    _assert_same_type(keys_first2, keys_last2)
    _assert_pointer_type(values_first1)
    _assert_pointer_type(values_first2)
    _assert_pointer_type(keys_result)
    _assert_pointer_type(values_result)
    args = [exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
            values_first1, values_first2, keys_result, values_result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::set_symmetric_difference_by_key({params})',
        _cuda_types.Tuple([keys_result.ctype, values_result.ctype])
    )


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_union(
        env, exec_policy, first1, last1, first2, last2, result, comp=None):
    """Constructs a sorted range that is the set union of sorted inputs.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(first1, last1)
    _assert_same_type(first2, last2)
    _assert_pointer_type(result)
    args = [exec_policy, first1, last1, first2, last2, result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::set_union({params})', result.ctype)


@_wrap_thrust_func(['thrust/set_operations.h'])
def set_union_by_key(
        env, exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
        values_first1, values_first2, keys_result, values_result, comp=None):
    """Constructs the key-value union of sorted inputs.
    """
    # TODO(asi1024): Typecheck for Comparable.
    _assert_exec_policy_type(exec_policy)
    _assert_same_type(keys_first1, keys_last1)
    _assert_same_type(keys_first2, keys_last2)
    _assert_pointer_type(values_first1)
    _assert_pointer_type(values_first2)
    _assert_pointer_type(keys_result)
    _assert_pointer_type(values_result)
    args = [exec_policy, keys_first1, keys_last1, keys_first2, keys_last2,
            values_first1, values_first2, keys_result, values_result]
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::set_union_by_key({params})',
        _cuda_types.Tuple([keys_result.ctype, values_result.ctype])
    )


# TODO(asi1024): Add shuffle
# TODO(asi1024): Add shuffle_copy


@_wrap_thrust_func(['thrust/sort.h'])
def sort(env, exec_policy, first, last, comp=None):
    """Sorts the elements in [first, last) into ascending order.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, first, last]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::sort({params})', _cuda_types.void)


@_wrap_thrust_func(['thrust/sort.h'])
def sort_by_key(
        env, exec_policy, keys_first, keys_last, values_first, comp=None):
    """Performs key-value sort.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(keys_first)
    _assert_same_type(keys_first, keys_last)
    _assert_pointer_type(values_first)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, keys_first, keys_last, values_first]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::sort_by_key({params})', _cuda_types.void)


# TODO(asi1024): Add stable_partition
# TODO(asi1024): Add stable_partition_copy


@_wrap_thrust_func(['thrust/sort.h'])
def stable_sort(env, exec_policy, first, last, comp=None):
    """Sorts the elements in [first, last) into ascending order.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, first, last]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::stable_sort({params})', _cuda_types.void)


@_wrap_thrust_func(['thrust/sort.h'])
def stable_sort_by_key(
        env, exec_policy, keys_first, keys_last, values_first, comp=None):
    """Performs key-value sort.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(keys_first)
    _assert_same_type(keys_first, keys_last)
    _assert_pointer_type(values_first)
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, keys_first, keys_last, values_first]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::stable_sort_by_key({params})', _cuda_types.void)


@_wrap_thrust_func(['thrust/swap.h'])
def swap_ranges(env, exec_policy, first1, last1, first2):
    """Swaps each of the elements in the range.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first1)
    _assert_same_type(first1, last1)
    _assert_pointer_type(first2)
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, first1, last1, first2]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::swap_ranges({params})', first2.ctype)


# TODO(asi1024): Add tabulate
# TODO(asi1024): Add transform
# TODO(asi1024): Add transform_exclusive_scan
# TODO(asi1024): Add transform_if
# TODO(asi1024): Add transform_inclusive_scan
# TODO(asi1024): Add transform_reduce
# TODO(asi1024): Add uninitialized_copy
# TODO(asi1024): Add uninitialized_copy_n
# TODO(asi1024): Add uninitialized_fill
# TODO(asi1024): Add uninitialized_fill_n


@_wrap_thrust_func(['thrust/unique.h'])
def unique(env, exec_policy, first, last, binary_pred=None):
    """Removes all but the first element of the group.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    args = [exec_policy, first, last]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::unique({params})', first.ctype)


@_wrap_thrust_func(['thrust/unique.h'])
def unique_by_key(
        env, exec_policy, keys_first, keys_last, values_first,
        binary_pred=None):
    """Uniques key-value pairs.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(keys_first)
    _assert_same_type(keys_first, keys_last)
    _assert_pointer_type(values_first)
    args = [exec_policy, keys_first, keys_last, values_first]
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::unique_by_key({params})',
        _cuda_types.Tuple([keys_first.ctype, values_first.ctype]),
    )


# TODO(asi1024): Add unique_by_key_copy
# TODO(asi1024): Add unique_by_key
# TODO(asi1024): Add unique_count


@_wrap_thrust_func(['thrust/binary_search.h'])
def upper_bound(env, exec_policy, first, last, *args):
    """Attempts to find the element value with binary search.
    """
    _assert_exec_policy_type(exec_policy)
    _assert_pointer_type(first)
    _assert_same_type(first, last)

    if 1 <= len(args) <= 2:
        value = args[0]
        comp = args[1] if len(args) == 2 else None
        _assert_pointer_of(first, value)
        result_ctype = first.ctype
    elif 3 <= len(args) <= 4:
        value_first = args[0]
        value_last = args[1]
        result = args[2]
        comp = args[3] if len(args) == 4 else None
        _assert_same_pointer_type(first, value_first)
        _assert_same_type(value_first, value_last)
        result_ctype = result.ctype
    else:
        raise TypeError('Invalid number of inputs of thrust.upper_bound')

    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    args = [exec_policy, first, last, *args]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::upper_bound({params})', result_ctype)
