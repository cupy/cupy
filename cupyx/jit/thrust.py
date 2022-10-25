from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types


def _wrap_thrust_func(headers):
    def wrapper(func):
        class FuncWrapper(_internal_types.BuiltinFunc):
            def call(self, env, *args, **kwargs):
                for header in headers:
                    env.generated.add_code(f'#include <{header}>')
                env.generated.add_code('#include <thrust/functional.h>')
                env.generated.backend = 'nvcc'
                return func(env, *args, **kwargs)
        return FuncWrapper()
    return wrapper


device = _internal_types.Data('thrust::device', _cuda_types.Unknown())


@_wrap_thrust_func(['thrust/copy.h', 'thrust/execution_policy.h'])
def copy(env, exec_policy, first, last, result):
    """Copies the elements.
    """
    if exec_policy.code != 'thrust::device':
        raise ValueError('`exec_policy` must be `cupyx.jit.thrust.device`')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    if first.ctype.child_type != result.ctype.child_type:
        raise TypeError('`first` and `result` must be of the same type')
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, result]
    params = ', '.join([_internal_types.Data.init(a, env).code for a in args])
    return _internal_types.Data(f'thrust::copy({params})', result.ctype)


@_wrap_thrust_func(['thrust/count.h', 'thrust/execution_policy.h'])
def count(env, exec_policy, first, last, value):
    """Counts the number of elements in [first, last) that equals to ``value``.
    """
    if exec_policy.code != 'thrust::device':
        raise ValueError('`exec_policy` must be `cupyx.jit.thrust.device`')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, value]
    params = ', '.join([_internal_types.Data.init(a, env).code for a in args])
    return _internal_types.Data(f'thrust::count({params})', _cuda_types.int32)


@_wrap_thrust_func(['thrust/find.h', 'thrust/execution_policy.h'])
def find(env, exec_policy, first, last, value):
    """Finds the first iterator whose value equals to ``value``.
    """
    if exec_policy.code != 'thrust::device':
        raise ValueError('`exec_policy` must be `cupyx.jit.thrust.device`')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first, last, value]
    params = ', '.join([_internal_types.Data.init(a, env).code for a in args])
    return _internal_types.Data(f'thrust::find({params})', first.ctype)


@_wrap_thrust_func(['thrust/mismatch.h', 'thrust/execution_policy.h'])
def mismatch(env, exec_policy, first1, last1, first2):
    """Finds the first positions whose values differ.
    """
    if exec_policy.code != 'thrust::device':
        raise ValueError('`exec_policy` must be `cupyx.jit.thrust.device`')
    if not isinstance(first1.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if first1.ctype != last1.ctype:
        raise TypeError('`first1` and `last1` must be of the same type')
    if first1.ctype.child_type != first2.ctype.child_type:
        raise TypeError('`first1` and `first2` must be of the same type')
    # TODO(asi1024): Typecheck for EqualityComparable.
    args = [exec_policy, first1, last1, first2]
    params = ', '.join([_internal_types.Data.init(a, env).code for a in args])
    return _internal_types.Data(
        f'thrust::mismatch({params})',
        _cuda_types.Tuple([first1.ctype, first2.ctype])
    )


@_wrap_thrust_func(['thrust/sort.h', 'thrust/execution_policy.h'])
def sort(env, exec_policy, first, last):
    """Sorts the elements in [first, last) into ascending order.
    """
    if exec_policy.code != 'thrust::device':
        raise ValueError('`exec_policy` must be `cupyx.jit.thrust.device`')
    if not isinstance(first.ctype, _cuda_types.PointerBase):
        raise TypeError('`first` must be of pointer type')
    if first.ctype != last.ctype:
        raise TypeError('`first` and `last` must be of the same type')
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, first, last]
    params = ', '.join([_internal_types.Data.init(a, env).code for a in args])
    return _internal_types.Data(f'thrust::sort({params})', _cuda_types.void)


@_wrap_thrust_func(['thrust/sort.h', 'thrust/execution_policy.h'])
def sort_by_key(env, exec_policy, keys_first, keys_last, values_first):
    """Performs key-value sort.
    """
    if exec_policy.code != 'thrust::device':
        raise ValueError('`exec_policy` must be `cupyx.jit.thrust.device`')
    if not isinstance(keys_first.ctype, _cuda_types.PointerBase):
        raise TypeError('`keys_first` must be of pointer type')
    if keys_first.ctype != keys_last.ctype:
        raise TypeError(
            '`keys_first` and `keys_last` must be of the same type')
    if not isinstance(values_first.ctype, _cuda_types.PointerBase):
        raise TypeError('`values_first` must be of pointer type')
    # TODO(asi1024): Typecheck for Comparable.
    args = [exec_policy, keys_first, keys_last, values_first]
    params = ', '.join([_internal_types.Data.init(a, env).code for a in args])
    return _internal_types.Data(
        f'thrust::sort_by_key({params})', _cuda_types.void)
