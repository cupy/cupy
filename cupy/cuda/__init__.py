import contextlib
import warnings

import cupy as _cupy
from cupy._environment import (
    get_cuda_path,
    get_hipcc_path,
    get_nvcc_path,
    get_rocm_path,
)
from cupy.cuda import (
    compiler,
    device,
    function,
    memory,
    memory_hook,
    memory_hooks,
    pinned_memory,
    profiler,
    stream,
    texture,
)
from cupy_backends.cuda.api import driver, runtime
from cupy_backends.cuda.libs import nvrtc

_available = None


class _UnavailableModule():
    available = False

    def __init__(self, name):
        self.__name__ = name


from cupy.cuda import cub

try:
    from cupy_backends.cuda.libs import nvtx
except ImportError:
    nvtx = _UnavailableModule('cupy.cuda.nvtx')

try:
    from cupy.cuda import thrust
except ImportError:
    thrust = _UnavailableModule('cupy.cuda.thrust')


def __getattr__(key):
    if key == 'cusolver':
        from cupy_backends.cuda.libs import cusolver
        _cupy.cuda.cusolver = cusolver
        return cusolver
    elif key == 'cusparse':
        from cupy_backends.cuda.libs import cusparse
        _cupy.cuda.cusparse = cusparse
        return cusparse
    elif key == 'curand':
        from cupy_backends.cuda.libs import curand
        _cupy.cuda.curand = curand
        return curand
    elif key == 'cublas':
        from cupy_backends.cuda.libs import cublas
        _cupy.cuda.cublas = cublas
        return cublas
    elif key == 'jitify':
        if not runtime.is_hip and driver.get_build_version() > 0:
            import cupy.cuda.jitify as jitify
        else:
            jitify = _UnavailableModule('cupy.cuda.jitify')
        _cupy.cuda.jitify = jitify
        return jitify

    # `nvtx_enabled` flags are kept for backward compatibility with Chainer.
    # Note: module-level getattr only runs on Python 3.7+.
    for mod in [nvtx]:
        flag = '{}_enabled'.format(mod.__name__.split('.')[-1])
        if key == flag:
            warnings.warn('''
cupy.cuda.{} has been deprecated in CuPy v8 and will be removed in the future release.
Use {}.available instead.
                '''.format(flag, mod.__name__), DeprecationWarning)
            return not isinstance(mod, _UnavailableModule)

    raise AttributeError(
        "module '{}' has no attribute '{}'".format(__name__, key))


def is_available():
    global _available
    if _available is None:
        _available = False
        try:
            _available = runtime.getDeviceCount() > 0
        except Exception as e:
            if (not runtime.is_hip and e.args[0] !=
                    'cudaErrorNoDevice: no CUDA-capable device is detected'):
                raise
            elif runtime.is_hip and 'hipErrorNoDevice' not in e.args[0]:
                raise
    return _available


def get_local_runtime_version() -> int:
    """
    Returns the version of the CUDA Runtime installed in the environment.

    Unlike :func:`cupy.cuda.runtime.runtimeGetVersion`, which returns the
    CUDA Runtime version statically linked to CuPy, this function returns the
    version retrieved from the shared library installed on the host.
    Use this method to probe the CUDA Runtime version installed in the
    environment.
    """
    return runtime._getLocalRuntimeVersion()


# import class and function
from cupy.cuda.device import Device, get_cublas_handle, get_device_id
from cupy.cuda.function import Function, Module
from cupy.cuda.graph import Graph
from cupy.cuda.memory import (
    BaseMemory,
    CFunctionAllocator,
    ManagedMemory,
    Memory,
    MemoryAsync,
    MemoryAsyncPool,
    MemoryPointer,
    MemoryPool,
    PythonFunctionAllocator,
    UnownedMemory,
    alloc,
    get_allocator,
    malloc_async,
    malloc_managed,
    set_allocator,
)
from cupy.cuda.memory_hook import MemoryHook
from cupy.cuda.pinned_memory import (
    PinnedMemory,
    PinnedMemoryPointer,
    PinnedMemoryPool,
    alloc_pinned_memory,
    set_pinned_memory_allocator,
)
from cupy.cuda.stream import (
    Event,
    ExternalStream,
    Stream,
    get_current_stream,
    get_elapsed_time,
)


@contextlib.contextmanager
def using_allocator(allocator=None):
    """Sets a thread-local allocator for GPU memory inside
       context manager

    Args:
        allocator (function): CuPy memory allocator. It must have the same
            interface as the :func:`cupy.cuda.alloc` function, which takes the
            buffer size as an argument and returns the device buffer of that
            size. When ``None`` is specified, raw memory allocator will be
            used (i.e., memory pool is disabled).
    """
    # Note: cupy/memory.pyx would be the better place to implement this
    # function but `contextmanager` decoration doesn't behave well in Cython.
    if allocator is None:
        allocator = memory._malloc
    previous_allocator = memory._get_thread_local_allocator()
    memory._set_thread_local_allocator(allocator)
    try:
        yield
    finally:
        memory._set_thread_local_allocator(previous_allocator)


@contextlib.contextmanager
def profile():
    """Enable CUDA profiling during with statement.

    This function enables profiling on entering a with statement, and disables
    profiling on leaving the statement.

    >>> with cupy.cuda.profile():
    ...    # do something you want to measure
    ...    pass

    .. note::
        When starting ``nvprof`` from the command line, manually setting
        ``--profile-from-start off`` may be required for the desired behavior.

    .. warning:: This context manager is deprecated. Please use
        :class:`cupyx.profiler.profile` instead.
    """
    warnings.warn(
        'cupy.cuda.profile has been deprecated since CuPy v10 '
        'and will be removed in the future. Use cupyx.profiler.profile '
        'instead.')

    profiler.start()
    try:
        yield
    finally:
        profiler.stop()
