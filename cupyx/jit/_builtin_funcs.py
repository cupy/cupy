import warnings

import cupy

from cupy_backends.cuda.api import runtime
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit._internal_types import Range
from cupyx.jit import _compile

from functools import reduce


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

        if isinstance(step, Constant):
            step_is_positive = step.obj >= 0
        elif step.ctype.dtype.kind == 'u':
            step_is_positive = True
        else:
            step_is_positive = None

        stop = Data.init(stop, env)
        start = Data.init(start, env)
        step = Data.init(step, env)

        if start.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if stop.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')
        if step.ctype.dtype.kind not in 'iu':
            raise TypeError('range supports only for integer type.')

        if env.mode == 'numpy':
            ctype = _cuda_types.Scalar(int)
        elif env.mode == 'cuda':
            ctype = stop.ctype
        else:
            assert False

        return Range(start, stop, step, ctype, step_is_positive)


class LenFunc(BuiltinFunc):
    def call(self, env, *args, **kwds):
        if len(args) != 1:
            raise TypeError(f'len() expects only 1 argument, got {len(args)}')
        if kwds:
            raise TypeError('keyword arguments are not supported')
        arg = args[0]
        if not isinstance(arg.ctype, _cuda_types.CArray):
            raise TypeError('len() supports only array type')
        if not arg.ctype.ndim:
            raise TypeError('len() of unsized array')
        return Data(f'static_cast<long long>({arg.code}.shape()[0])',
                    _cuda_types.Scalar('q'))


class MinFunc(BuiltinFunc):
    def call(self, env, *args, **kwds):
        if len(args) < 2:
            raise TypeError(
                f'min() expects at least 2 arguments, got {len(args)}')
        if kwds:
            raise TypeError('keyword arguments are not supported')
        return reduce(lambda a, b: _compile._call_ufunc(
            cupy.minimum, (a, b), None, env), args)


class MaxFunc(BuiltinFunc):
    def call(self, env, *args, **kwds):
        if len(args) < 2:
            raise TypeError(
                f'max() expects at least 2 arguments, got {len(args)}')
        if kwds:
            raise TypeError('keyword arguments are not supported')
        return reduce(lambda a, b: _compile._call_ufunc(
            cupy.maximum, (a, b), None, env), args)


class SyncThreads(BuiltinFunc):

    def __call__(self):
        """Calls ``__syncthreads()``.

        .. seealso:: `Synchronization functions`_

        .. _Synchronization functions:
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions
        """
        super.__call__(self)

    def call_const(self, env):
        return Data('__syncthreads()', _cuda_types.void)


class SyncWarp(BuiltinFunc):

    def __call__(self, *, mask=0xffffffff):
        """Calls ``__syncwarp()``.

        Args:
            mask (int): Active threads in a warp. Default is 0xffffffff.

        .. seealso:: `Synchronization functions`_

        .. _Synchronization functions:
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions
        """
        super().__call__()

    def call(self, env, *, mask=None):
        if runtime.is_hip:
            warnings.warn(f'mask {mask} is ignored on HIP', RuntimeWarning)
            mask = None

        if mask:
            if isinstance(mask, Constant):
                if not (0x0 <= mask.obj <= 0xffffffff):
                    raise ValueError('mask is out of range')
            mask = _compile._astype_scalar(
                mask, _cuda_types.int32, 'same_kind', env)
            mask = Data.init(mask, env)
            code = f'__syncwarp({mask.code})'
        else:
            code = '__syncwarp()'
        return Data(code, _cuda_types.void)


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
        super.__call__(self)

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


class Grid(BuiltinFunc):

    def __call__(self, ndim):
        """Compute the thread index in the grid.

        Computation of the first integer is as follows::

            jit.threadIdx.x + jit.blockIdx.x * jit.blockDim.x

        and for the other two integers the ``y`` and ``z`` attributes are used.

        Args:
            ndim (int): The dimension of the grid. Only 1, 2, or 3 is allowed.

        Returns:
            int or tuple:
                If ``ndim`` is 1, an integer is returned, otherwise a tuple.

        .. note::
            This function follows the convention of Numba's `numba.cuda.grid`_.

        .. _numba.cuda.grid:
            https://numba.readthedocs.io/en/stable/cuda/kernels.html#absolute-positions

        """
        super.__call__(self)

    def call_const(self, env, ndim):
        if not isinstance(ndim, int):
            raise TypeError('ndim must be an integer')

        # Numba convention: for 1D we return a single variable,
        # otherwise a tuple
        code = 'threadIdx.{n} + blockIdx.{n} * blockDim.{n}'
        if ndim == 1:
            return Data(code.format(n='x'), _cuda_types.uint32)
        elif ndim == 2:
            dims = ('x', 'y')
        elif ndim == 3:
            dims = ('x', 'y', 'z')
        else:
            raise ValueError('Only ndim=1,2,3 are supported')

        elts_code = ', '.join(code.format(n=n) for n in dims)
        ctype = _cuda_types.Tuple([_cuda_types.uint32]*ndim)
        return Data(f'thrust::make_tuple({elts_code})', ctype)


class WarpShuffleOp(BuiltinFunc):

    def __init__(self, op, dtypes):
        self._op = op
        self._name = '__shfl_' + (op + '_' if op else '') + 'sync'
        self._dtypes = dtypes
        doc = f"""Call the {self._name} function. Please refer to
        `Warp Shuffle Functions`_ for detail explanation.

        .. _Warp Shuffle Functions:
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
        """
        self.__doc__ = doc

    def __call__(self, mask, var, val_id, *, width=32):
        super().__call__()

    def call(self, env, mask, var, val_id, *, width=None):
        name = self._name

        var = Data.init(var, env)
        ctype = var.ctype
        if ctype.dtype.char not in self._dtypes:
            raise TypeError(f'`{name}` does not support {ctype.dtype} input.')

        try:
            mask = mask.obj
        except Exception:
            raise TypeError('mask must be an integer')
        if runtime.is_hip:
            warnings.warn(f'mask {mask} is ignored on HIP', RuntimeWarning)
        elif not (0x0 <= mask <= 0xffffffff):
            raise ValueError('mask is out of range')

        # val_id refers to "delta" for shfl_{up, down}, "srcLane" for shfl, and
        # "laneMask" for shfl_xor
        if self._op in ('up', 'down'):
            val_id_t = _cuda_types.uint32
        else:
            val_id_t = _cuda_types.int32
        val_id = _compile._astype_scalar(val_id, val_id_t, 'same_kind', env)
        val_id = Data.init(val_id, env)

        if width:
            if isinstance(width, Constant):
                if width.obj not in (2, 4, 8, 16, 32):
                    raise ValueError('width needs to be power of 2')
        else:
            width = Constant(64) if runtime.is_hip else Constant(32)
        width = _compile._astype_scalar(
            width, _cuda_types.int32, 'same_kind', env)
        width = Data.init(width, env)

        code = f'{name}({hex(mask)}, {var.code}, {val_id.code}'
        code += f', {width.code})'
        return Data(code, ctype)


builtin_functions_dict = {
    range: RangeFunc(),
    len: LenFunc(),
    min: MinFunc(),
    max: MaxFunc(),
}

syncthreads = SyncThreads()
syncwarp = SyncWarp()
shared_memory = SharedMemory()
grid = Grid()

# TODO: Add more atomic functions.
atomic_add = AtomicOp('Add', 'iILQefd')

# warp-shuffle functions
shfl_sync = WarpShuffleOp('', 'iIlqfd' if runtime.is_hip else 'iIlLqQefd')
shfl_up_sync = WarpShuffleOp('up', 'iIlqfd' if runtime.is_hip else 'iIlLqQefd')
shfl_down_sync = WarpShuffleOp('down',
                               'iIlqfd' if runtime.is_hip else 'iIlLqQefd')
shfl_xor_sync = WarpShuffleOp('xor',
                              'iIlqfd' if runtime.is_hip else 'iIlLqQefd')
