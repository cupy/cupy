import functools
import warnings

import numpy

import cupy
from cupyx.jit import _compile
from cupyx.jit import _typerules
from cupyx.jit import _types


class _CudaFunction:
    """JIT cupy function object
    """

    def __init__(self, func, mode, device=False, inline=False):
        self.attributes = []

        if device:
            self.attributes.append('__device__')
        else:
            self.attributes.append('__global__')

        if inline:
            self.attributes.append('inline')

        self.name = getattr(func, 'name', func.__name__)
        self.func = func
        self.mode = mode

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _emit_code_from_types(self, in_types, ret_type=None):
        return _compile.transpile(
            self.func, self.attributes, self.mode, in_types, ret_type)


class _JitRawKernel:
    """JIT CUDA kernel object.

    The decorator :func:``cupyx.jit.rawkernel`` converts the target function
    to an object of this class. This class is not inteded to be instantiated
    by users.
    """

    def __init__(self, func, mode):
        self._func = func
        self._mode = mode
        self._cache = {}
        self._cached_codes = {}

    def __call__(
            self, grid, block, args, shared_mem=0,
            stream=None, enable_cooperative_groups=False):
        """Calls the CUDA kernel.

        The compilation will be deferred until the first function call.
        CuPy's JIT compiler infers the types of arguments at the call
        time, and will cache the compiled kernels for speeding up any
        subsequent calls.

        Args:
            grid (tuple of int): Size of grid in blocks.
            block (tuple of int): Dimensions of each thread block.
            args (tuple):
                Arguments of the kernel. The type of all elements must be
                ``bool``, ``int``, ``float``, ``complex``, NumPy scalar or
                ``cupy.ndarray``.
            shared_mem (int):
                Dynamic shared-memory size per thread block in bytes.
            stream (cupy.cuda.Stream): CUDA stream.

        .. seealso:: :ref:`jit_kernel_definition`
        """
        in_types = []
        for x in args:
            if isinstance(x, cupy.ndarray):
                t = _types.CArray.from_ndarray(x)
            elif numpy.isscalar(x):
                t = _typerules.get_ctype_from_scalar(self._mode, x)
            else:
                raise TypeError(f'{type(x)} is not supported for RawKernel')
            in_types.append(t)
        in_types = tuple(in_types)

        kern = self._cache.get(in_types)
        if kern is None:
            result = _compile.transpile(
                self._func,
                ['extern "C"', '__global__'],
                self._mode,
                in_types,
                _types.Void(),
            )
            fname = result.func_name
            module = cupy._core.core.compile_with_cache(
                source=result.code,
                options=('-D CUPY_JIT_MODE',))
            kern = module.get_function(fname)
            self._cache[in_types] = kern
            self._cached_codes[in_types] = result.code

        kern(grid, block, args, shared_mem, stream, enable_cooperative_groups)

    def __getitem__(self, grid_and_block):
        """Numba-style kernel call.

        .. seealso:: :ref:`jit_kernel_definition`
        """
        grid, block = grid_and_block
        if not isinstance(grid, tuple):
            grid = (grid, 1, 1)
        if not isinstance(block, tuple):
            block = (block, 1, 1)
        return lambda *args, **kwargs: self(grid, block, args, **kwargs)

    @property
    def cached_codes(self):
        """Returns a dict that has input types as keys and codes values.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
        if len(self._cached_codes) == 0:
            warnings.warn(
                'No codes are cached because compilation is deferred until '
                'the first function call.')
        return self._cached_codes

    @property
    def cached_code(self):
        """Returns `next(iter(self.cached_codes.values()))`.

        This proprety method is for debugging purpose.
        The return value is not guaranteed to keep backward compatibility.
        """
        codes = self.cached_codes
        if len(codes) > 1:
            warnings.warn(
                'The input types of the kernel could not be inferred. '
                'Please use `.cached_codes` instead.')
        return next(iter(codes.values()))


def rawkernel(mode='cuda'):
    """A decorator compiles a Python function into CUDA kernel.
    """
    cupy._util.experimental('cupyx.jit.rawkernel')

    def wrapper(func):
        return functools.update_wrapper(_JitRawKernel(func, mode), func)
    return wrapper


class _Dim3:
    def __init__(self, name):
        self.x = _compile.CudaObject(f'{name}.x', _types.uint32)
        self.y = _compile.CudaObject(f'{name}.y', _types.uint32)
        self.z = _compile.CudaObject(f'{name}.z', _types.uint32)
        self.__doc__ = f"""dim3 {name}

        A namedtuple of three integers represents {name}.

        Attributes:
            x (uint32): {name}.x
            y (uint32): {name}.y
            z (uint32): {name}.z
        """


threadIdx = _Dim3('threadIdx')
blockDim = _Dim3('blockDim')
blockIdx = _Dim3('blockIdx')
gridDim = _Dim3('gridDim')

syncthreads = _compile.SyncThreads()
shared_memory = _compile.SharedMemory()
