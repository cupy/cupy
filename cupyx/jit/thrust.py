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


host = _Data('thrust::host', _ExecPolicyType())
device = _Data('thrust::device', _ExecPolicyType())
seq = _Data('thrust::seq', _ExecPolicyType())


@_wrap_thrust_func(['thrust/adjacent_difference.h'])
def adjacent_difference(env, exec_policy, first, last, result, binary_op=None):
    """Computes the differences of adjacent elements.
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    if first.ctype.child_type != result.ctype.child_type:
        raise TypeError('`first` and `result` must be of the same type')
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
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')

    if 1 <= len(args) <= 2:
        value = args[0]
        comp = args[1] if len(args) == 2 else None
        if first.ctype.child_type != value.ctype:
            raise TypeError('`first` and `result` must be of the same type')
        if comp is not None:
            raise NotImplementedError('comp option is not supported')
        result_ctype = _cuda_types.bool_
    elif 3 <= len(args) <= 4:
        value_first = args[0]
        value_last = args[1]
        result = args[2]
        comp = args[3] if len(args) == 4 else None
        if first.ctype.child_type != value_first.ctype.child_type:
            raise TypeError(
                '`first` and `value_first` must be of the same type')
        if value_first.ctype != value_last.ctype:
            raise TypeError(
                '`value_first` and `value_last` must be of the same type')
        if comp is not None:
            raise NotImplementedError('comp option is not supported')
        result_ctype = result.ctype
    else:
        raise TypeError('Invalid number of inputs of thrust.binary_search')

    args = [exec_policy, first, last, *args]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::binary_search({params})', result_ctype)


@_wrap_thrust_func(['thrust/copy.h'])
def copy(env, exec_policy, first, last, result):
    """Copies the elements.
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    if first.ctype.child_type != result.ctype.child_type:
        raise TypeError('`first` and `result` must be of the same type')
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
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
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
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first1.ctype, _cuda_types.PointerBase):
        raise TypeError('`first1` must be of pointer type')
    if not isinstance(first2.ctype, _cuda_types.PointerBase):
        raise TypeError('`first2` must be of pointer type')
    if first1.ctype != last1.ctype:
        raise TypeError('`first1` and `last1` must be of the same type')
    if first1.ctype.child_type != first2.ctype.child_type:
        raise TypeError('`first1` and `first2` must be of the same type')
    if binary_pred is not None:
        raise NotImplementedError('binary_pred option is not supported')
    args = [exec_policy, first1, last1, first2]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::equal({params})', _cuda_types.bool_)


@_wrap_thrust_func(['thrust/binary_search.h'])
def equal_range(env, exec_policy, first, last, value, comp=None):
    """Attempts to find the element value in an ordered range.
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
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
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if not isinstance(result.ctype, _cuda_types.PointerBase):
        raise TypeError('`result` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    if first.ctype.child_type != result.ctype.child_type:
        raise TypeError('`first` and `last` must be of the same type')
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
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first1.ctype, _cuda_types.PointerBase):
        raise TypeError('`first1` must be of pointer type')
    if not isinstance(first2.ctype, _cuda_types.PointerBase):
        raise TypeError('`first2` must be of pointer type')
    if not isinstance(result.ctype, _cuda_types.PointerBase):
        raise TypeError('`result` must be of pointer type')
    if first1.ctype != last1.ctype:
        raise TypeError('`first1` and `last1` must be of the same type')
    if first2.ctype.child_type != result.ctype.child_type:
        raise TypeError('`first2` and `result` must be of the same type')
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
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    if first.ctype.child_type != value.ctype:
        raise TypeError('`*first` and `value` must be of the same type')
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::fill({params})', _cuda_types.void)


# TODO(asi1024): Add fill_n


@_wrap_thrust_func(['thrust/find.h'])
def find(env, exec_policy, first, last, value):
    """Finds the first iterator whose value equals to ``value``.
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, value]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::find({params})', first.ctype)


@_wrap_thrust_func(['thrust/mismatch.h'])
def mismatch(env, exec_policy, first1, last1, first2, pred=None):
    """Finds the first positions whose values differ.
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first1.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if first1.ctype != last1.ctype:
        raise TypeError('`first1` and `last1` must be of the same type')
    if first1.ctype.child_type != first2.ctype.child_type:
        raise TypeError('`first1` and `first2` must be of the same type')
    if pred is not None:
        raise NotImplementedError('pred option is not supported')
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first1, last1, first2]
    params = ', '.join([a.code for a in args])
    return _Data(
        f'thrust::mismatch({params})',
        _cuda_types.Tuple([first1.ctype, first2.ctype])
    )


# TODO(asi1024): Add find_if
# TODO(asi1024): Add find_if_not
# TODO(asi1024): Add for_each
# TODO(asi1024): Add for_each_n


@_wrap_thrust_func(['thrust/gather.h'])
def gather(env, exec_policy, map_first, map_last, input_first, result):
    """Copies elements from source into destination  according to a map.
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(map_first.ctype, _cuda_types.PointerBase):
        raise TypeError('`map_first` must be of pointer type')
    if not isinstance(input_first.ctype, _cuda_types.PointerBase):
        raise TypeError('`input_first` must be of pointer type')
    if not isinstance(result.ctype, _cuda_types.PointerBase):
        raise TypeError('`result_first` must be of pointer type')
    if map_first.ctype != map_last.ctype:
        raise TypeError('`map_first` and `map_last` must be of the same type')
    if input_first.ctype.child_type != result.ctype.child_type:
        raise TypeError(
            '`*input_first` and `*result` must be of the same type')
    args = [exec_policy, map_first, map_last, input_first, result]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::gather({params})', result.ctype)


# TODO(asi1024): Add gather_if
# TODO(asi1024): Add generate_n
# TODO(asi1024): Add inclusive_scan
# TODO(asi1024): Add inclusive_scan_by_key
# TODO(asi1024): Add inner_product
# TODO(asi1024): Add is_partitioned
# TODO(asi1024): Add is_sorted
# TODO(asi1024): Add is_sorted_until
# TODO(asi1024): Add lower_bound
# TODO(asi1024): Add make_constant_iterator
# TODO(asi1024): Add make_counting_iterator
# TODO(asi1024): Add make_discard_iterator
# TODO(asi1024): Add make_index_sequence


@_wrap_thrust_func(['thrust/sort.h'])
def sort(env, exec_policy, first, last, comp=None):
    """Sorts the elements in [first, last) into ascending order.
    """
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
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
    if not isinstance(exec_policy.ctype, _ExecPolicyType):
        raise ValueError('The first argument must be execution policy type')
    if not isinstance(keys_first.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if keys_first.ctype != keys_last.ctype:
        raise TypeError(
            '`keys_first` and `keys_last` must be of the same type')
    if not isinstance(values_first.ctype, _cuda_types.PointerBase):
        raise TypeError('`values_first` must be of pointer type')
    if comp is not None:
        raise NotImplementedError('comp option is not supported')
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, keys_first, keys_last, values_first]
    params = ', '.join([a.code for a in args])
    return _Data(f'thrust::sort_by_key({params})', _cuda_types.void)
