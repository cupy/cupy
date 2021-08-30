import warnings

import cupy

from cupy_backends.cuda.api import runtime
from cupy.cuda import device
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
        super().__call__()

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
            if mask is not None:
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
        doc = f"""Calls the ``{self._name}`` function to operate atomically on
        ``array[index]``. Please refer to `Atomic Functions`_ for detailed
        explanation.

        Args:
            array: A :class:`cupy.ndarray` to index over.
            index: A valid index such that the address to the corresponding
                array element ``array[index]`` can be computed.
            value: Represent the value to use for the specified operation. For
                the case of :obj:`atomic_cas`, this is the value for
                ``array[index]`` to compare with.
            alt_value: Only used in :obj:`atomic_cas` to represent the value
                to swap to.

        .. seealso:: `Numba's corresponding atomic functions`_

        .. _Atomic Functions:
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions

        .. _Numba's corresponding atomic functions:
            https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html#synchronization-and-atomic-operations
        """
        self.__doc__ = doc

    def __call__(self, array, index, value, alt_value=None):
        super().__call__()

    def call(self, env, array, index, value, value2=None):
        name = self._name
        op = self._op
        array = Data.init(array, env)
        if not isinstance(array.ctype, (_cuda_types.CArray, _cuda_types.Ptr)):
            raise TypeError('The first argument must be of array type.')
        target = _compile._indexing(array, index, env)
        ctype = target.ctype
        if ctype.dtype.name not in self._dtypes:
            raise TypeError(f'`{name}` does not support {ctype.dtype} input.')
        # On HIP, 'e' is not supported and we will never reach here
        if (op == 'Add' and ctype.dtype.char == 'e'
                and runtime.runtimeGetVersion() < 10000):
            raise RuntimeError(
                'float16 atomic operation is not supported before CUDA 10.0.')
        value = _compile._astype_scalar(value, ctype, 'same_kind', env)
        value = Data.init(value, env)
        if op == 'CAS':
            assert value2 is not None
            # On HIP, 'H' is not supported and we will never reach here
            if ctype.dtype.char == 'H':
                if runtime.runtimeGetVersion() < 10010:
                    raise RuntimeError(
                        'uint16 atomic operation is not supported before '
                        'CUDA 10.1')
                if int(device.get_compute_capability()) < 70:
                    raise RuntimeError(
                        'uint16 atomic operation is not supported before '
                        'sm_70')
            value2 = _compile._astype_scalar(value2, ctype, 'same_kind', env)
            value2 = Data.init(value2, env)
            code = f'{name}(&{target.code}, {value.code}, {value2.code})'
        else:
            assert value2 is None
            code = f'{name}(&{target.code}, {value.code})'
        return Data(code, ctype)


class GridFunc(BuiltinFunc):

    def __init__(self, mode):
        if mode == 'grid':
            self._desc = 'Compute the thread index in the grid.'
            self._eq = 'jit.threadIdx.x + jit.blockIdx.x * jit.blockDim.x'
            self._link = 'numba.cuda.grid'
            self._code = 'threadIdx.{n} + blockIdx.{n} * blockDim.{n}'
        elif mode == 'gridsize':
            self._desc = 'Compute the grid size.'
            self._eq = 'jit.blockDim.x * jit.gridDim.x'
            self._link = 'numba.cuda.gridsize'
            self._code = 'blockDim.{n} * gridDim.{n}'
        else:
            raise ValueError('unsupported function')

        doc = f"""        {self._desc}

        Computation of the first integer is as follows::

            {self._eq}

        and for the other two integers the ``y`` and ``z`` attributes are used.

        Args:
            ndim (int): The dimension of the grid. Only 1, 2, or 3 is allowed.

        Returns:
            int or tuple:
                If ``ndim`` is 1, an integer is returned, otherwise a tuple.

        .. note::
            This function follows the convention of Numba's `{self._link}`_.

        .. _{self._link}:
            https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html#{self._link}

        """
        self.__doc__ = doc

    def __call__(self, ndim):
        super().__call__()

    def call_const(self, env, ndim):
        if not isinstance(ndim, int):
            raise TypeError('ndim must be an integer')

        # Numba convention: for 1D we return a single variable,
        # otherwise a tuple
        if ndim == 1:
            return Data(self._code.format(n='x'), _cuda_types.uint32)
        elif ndim == 2:
            dims = ('x', 'y')
        elif ndim == 3:
            dims = ('x', 'y', 'z')
        else:
            raise ValueError('Only ndim=1,2,3 are supported')

        elts_code = ', '.join(self._code.format(n=n) for n in dims)
        ctype = _cuda_types.Tuple([_cuda_types.uint32]*ndim)
        return Data(f'thrust::make_tuple({elts_code})', ctype)


class WarpShuffleOp(BuiltinFunc):

    def __init__(self, op, dtypes):
        self._op = op
        self._name = '__shfl_' + (op + '_' if op else '') + 'sync'
        self._dtypes = dtypes
        doc = f"""Calls the ``{self._name}`` function. Please refer to
        `Warp Shuffle Functions`_ for detailed explanation.

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
        if ctype.dtype.name not in self._dtypes:
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


class LaneID(BuiltinFunc):
    def __call__(self):
        """Returns the lane ID of the calling thread, ranging in
        ``[0, jit.warpsize)``.

        .. note::
            Unlike `numba.cuda.laneid`_, this is a callable function instead
            of a property.

        .. _numba.cuda.laneid:
            https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html#numba.cuda.laneid
        """
        super().__call__()

    def _get_preamble(self):
        preamble = '__device__ __forceinline__ unsigned int LaneId() {'
        if not runtime.is_hip:
            # see https://github.com/NVIDIA/cub/blob/main/cub/util_ptx.cuh#L419
            preamble += """
                unsigned int ret;
                asm ("mov.u32 %0, %%laneid;" : "=r"(ret) );
                return ret; }
            """
        else:
            # defined in hip/hcc_detail/device_functions.h
            preamble += """
                return __lane_id(); }
            """
        return preamble

    def call_const(self, env):
        env.preambles.add(self._get_preamble())
        return Data('LaneId()', _cuda_types.uint32)


builtin_functions_dict = {
    range: RangeFunc(),
    len: LenFunc(),
    min: MinFunc(),
    max: MaxFunc(),
}

syncthreads = SyncThreads()
syncwarp = SyncWarp()
shared_memory = SharedMemory()
grid = GridFunc('grid')
gridsize = GridFunc('gridsize')
laneid = LaneID()

# atomic functions
atomic_add = AtomicOp(
    'Add',
    ('int32', 'uint32', 'uint64', 'float32', 'float64')
    + (() if runtime.is_hip else ('float16',)))
atomic_sub = AtomicOp(
    'Sub', ('int32', 'uint32'))
atomic_exch = AtomicOp(
    'Exch', ('int32', 'uint32', 'uint64', 'float32'))
atomic_min = AtomicOp(
    'Min', ('int32', 'uint32', 'uint64'))
atomic_max = AtomicOp(
    'Max', ('int32', 'uint32', 'uint64'))
atomic_inc = AtomicOp(
    'Inc', ('uint32',))
atomic_dec = AtomicOp(
    'Dec', ('uint32',))
atomic_cas = AtomicOp(
    'CAS',
    ('int32', 'uint32', 'uint64')
    + (() if runtime.is_hip else ('uint16',)))
atomic_and = AtomicOp(
    'And', ('int32', 'uint32', 'uint64'))
atomic_or = AtomicOp(
    'Or', ('int32', 'uint32', 'uint64'))
atomic_xor = AtomicOp(
    'Xor', ('int32', 'uint32', 'uint64'))

# warp-shuffle functions
_shfl_dtypes = (
    ('int32', 'uint32', 'int64', 'float32', 'float64')
    + (() if runtime.is_hip else ('uint64', 'float16')))
shfl_sync = WarpShuffleOp('', _shfl_dtypes)
shfl_up_sync = WarpShuffleOp('up', _shfl_dtypes)
shfl_down_sync = WarpShuffleOp('down', _shfl_dtypes)
shfl_xor_sync = WarpShuffleOp('xor', _shfl_dtypes)
