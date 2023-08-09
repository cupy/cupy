import functools
import warnings

import numpy

import cupy
from cupy._core import core
from cupyx.jit import _compile
from cupyx.jit import _cuda_typerules
from cupyx.jit import _cuda_types
from cupyx.jit import _internal_types
from cupyx.jit._cuda_types import Scalar


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

    def __init__(self, func, mode, device):
        self._func = func
        self._mode = mode
        self._device = device
        self._cache = {}
        self._cached_codes = {}

    def __call__(
            self, grid, block, args, shared_mem=0, stream=None):
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
                t = _cuda_types.CArray.from_ndarray(x)
            elif numpy.isscalar(x):
                t = _cuda_typerules.get_ctype_from_scalar(self._mode, x)
            else:
                raise TypeError(f'{type(x)} is not supported for RawKernel')
            in_types.append(t)
        in_types = tuple(in_types)
        device_id = cupy.cuda.get_device_id()

        kern, enable_cg = self._cache.get((in_types, device_id), (None, None))
        if kern is None:
            result = self._cached_codes.get(in_types)
            if result is None:
                result = _compile.transpile(
                    self._func,
                    ['extern "C"', '__global__'],
                    self._mode,
                    in_types,
                    _cuda_types.void,
                )
                self._cached_codes[in_types] = result

            fname = result.func_name
            enable_cg = result.enable_cooperative_groups
            options = ('-DCUPY_JIT_MODE', '--std=c++14')
            backend = result.backend
            if backend == 'nvcc':
                options += ('-DCUPY_JIT_NVCC',)
            module = core.compile_with_cache(
                source=result.code,
                options=options,
                backend=backend)
            kern = module.get_function(fname)
            self._cache[(in_types, device_id)] = (kern, enable_cg)

        new_args = []
        for a, t in zip(args, in_types):
            if isinstance(t, Scalar):
                if t.dtype.char == 'e':
                    a = numpy.float32(a)
                else:
                    a = t.dtype.type(a)
            new_args.append(a)

        kern(grid, block, tuple(new_args), shared_mem, stream, enable_cg)

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
        return dict([(k, v.code) for k, v in self._cached_codes.items()])

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


def rawkernel(*, mode='cuda', device=False):
    """A decorator compiles a Python function into CUDA kernel.
    """
    cupy._util.experimental('cupyx.jit.rawkernel')

    def wrapper(func):
        return functools.update_wrapper(
            _JitRawKernel(func, mode, device), func)
    return wrapper


threadIdx = _internal_types.Data('threadIdx', _cuda_types.dim3)
blockDim = _internal_types.Data('blockDim', _cuda_types.dim3)
blockIdx = _internal_types.Data('blockIdx', _cuda_types.dim3)
gridDim = _internal_types.Data('gridDim', _cuda_types.dim3)

warpsize = _internal_types.Data('warpSize', _cuda_types.int32)
warpsize.__doc__ = r"""Returns the number of threads in a warp.

.. seealso:: :obj:`numba.cuda.warpsize`
"""
