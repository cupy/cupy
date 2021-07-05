from cupy_backends.cuda.api import runtime
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit._internal_types import Range
from cupyx.jit import _compile


class RangeFunc(BuiltinFunc):

    def call(self, env, *args, **kwargs):
        if len(args) == 0:
            raise TypeError('range expected at least 1 argument, got 0')
        elif len(args) == 1:
            start, stop, step = Constant(0), args[0], Constant(1)
        elif len(args) == 2:
            start, stop, step = args[0], args[1], Constant(1)
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise TypeError(
                f'range expected at most 3 argument, got {len(args)}')

        stop = Data.init(stop, env)
        start = Data.init(start, env)
        step = Data.init(step, env)

        if start.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if stop.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if step.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')

        if isinstance(step, Constant):
            step_is_positive = step.obj >= 0
        elif step.ctype.dtype.kind == 'u':
            step_is_positive = True
        else:
            step_is_positive = None

        if env.mode == 'numpy':
            ctype = _cuda_types.Scalar(int)
        elif env.mode == 'cuda':
            ctype = stop.ctype
        else:
            assert False

        return Range(start, stop, step, ctype, step_is_positive)


class SyncThreads(BuiltinFunc):

    def __call__(self):
        """Calls ``__syncthreads()``
        """
        super().__call__()

    def call_const(self, env):
        return Data('__syncthreads()', _cuda_types.void)


class SharedMemory(BuiltinFunc):

    def __call__(self, dtype, size):
        """Allocates shared memory and returns the 1-dim array.

        Args:
            dtype (dtype):
                The dtype of the returned array.
            size (int or None):
                If ``int`` type, the size of static shared memory.
                If ``None``, declares the shared memory with extern specifier.
        """
        super().__call__()

    def call_const(self, env, dtype, size):
        name = env.get_fresh_variable_name(prefix='_smem')
        child_type = _cuda_types.Scalar(dtype)
        while env[name] is not None:
            name = env.get_fresh_variable_name(prefix='_smem')  # retry
        env[name] = Data(name, _cuda_types.SharedMem(child_type, size))
        return Data(name, _cuda_types.Ptr(child_type))


class AtomicOp(BuiltinFunc):

    def __init__(self, op, dtypes):
        self._op = op
        self._name = 'atomic' + op
        self._dtypes = dtypes
        super().__init__()

    def call(self, env, array, index, value):
        array = Data.init(array, env)
        if not isinstance(array.ctype, (_cuda_types.CArray, _cuda_types.Ptr)):
            raise TypeError('The first argument must be of array type.')
        target = _compile._indexing(array, index, env)
        ctype = target.ctype
        value = _compile._astype_scalar(value, ctype, 'same_kind', env)
        name = self._name
        value = Data.init(value, env)
        if ctype.dtype.char not in self._dtypes:
            raise TypeError(f'`{name}` does not support {ctype.dtype} input.')
        if ctype.dtype.char == 'e' and runtime.runtimeGetVersion() < 10000:
            raise RuntimeError(
                'float16 atomic operation is not supported this CUDA version.')
        return Data(f'{name}(&{target.code}, {value.code})', ctype)


builtin_functions_dict = {
    range: RangeFunc(),
}

syncthreads = SyncThreads()
shared_memory = SharedMemory()

# TODO: Add more atomic functions.
atomic_add = AtomicOp(
    'Add', 'iILQfd' if runtime.is_hip else 'iILQefd')
