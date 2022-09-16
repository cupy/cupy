from cupyx.jit import _internal_types
from cupyx.jit import _cuda_types


def _wrap_thrust_func(header):
    def wrapper(func):
        class FuncWrapper(_internal_types.BuiltinFunc):
            def call(self, env, *args, **kwargs):
                env.generated.add_code(f'#include <{header}>')
                env.generated.add_code('#include <thrust/functional.h>')
                env.generated.backend = 'nvcc'
                return func(env, *args, **kwargs)
        return FuncWrapper()
    return wrapper


device = _internal_types.Data('thrust::device', _cuda_types.Unknown())


@_wrap_thrust_func('thrust/count.h')
def count(env, exec_policy, first, last, value):
    """Count the number of elements in [first, last) that equals to ``value``.
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
