import contextlib
import warnings

import cupy as _cupy
from cupy._environment import get_cuda_path  # NOQA
from cupy._environment import get_nvcc_path  # NOQA
from cupy._environment import get_rocm_path  # NOQA
from cupy._environment import get_hipcc_path  # NOQA
from cupy.cuda import compiler  # NOQA
from cupy.cuda import device  # NOQA
from cupy.cuda import function  # NOQA
from cupy.cuda import memory  # NOQA
from cupy.cuda import memory_hook  # NOQA
from cupy.cuda import memory_hooks  # NOQA
from cupy.cuda import pinned_memory  # NOQA
from cupy.cuda import profiler  # NOQA
from cupy.cuda import stream  # NOQA
from cupy.cuda import texture  # NOQA
from cupy_backends.cuda.api import driver  # NOQA
from cupy_backends.cuda.api import runtime  # NOQA
from cupy_backends.cuda.libs import nvrtc  # NOQA


_available = None


class _UnavailableModule():
    available = False

    def __init__(self, name):
        self.__name__ = name


from cupy.cuda import cub  # NOQA


try:
    from cupy_backends.cuda.libs import nvtx  # NOQA
except ImportError:
    nvtx = _UnavailableModule('cupy.cuda.nvtx')

try:
    from cupy.cuda import thrust  # NOQA
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
                '''.format(flag, mod.__name__), DeprecationWarning)  # NOQA
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
from cupy.cuda.device import Device  # NOQA
from cupy.cuda.device import get_cublas_handle  # NOQA
from cupy.cuda.device import get_device_id  # NOQA
from cupy.cuda.function import Function  # NOQA
from cupy.cuda.function import Module  # NOQA
from cupy.cuda.memory import alloc  # NOQA
from cupy.cuda.memory import BaseMemory  # NOQA
from cupy.cuda.memory import malloc_managed  # NOQA
from cupy.cuda.memory import malloc_async  # NOQA
from cupy.cuda.memory import ManagedMemory  # NOQA
from cupy.cuda.memory import Memory  # NOQA
from cupy.cuda.memory import MemoryAsync  # NOQA
from cupy.cuda.memory import MemoryPointer  # NOQA
from cupy.cuda.memory import MemoryPool  # NOQA
from cupy.cuda.memory import MemoryAsyncPool  # NOQA
from cupy.cuda.memory import PythonFunctionAllocator  # NOQA
from cupy.cuda.memory import CFunctionAllocator  # NOQA
from cupy.cuda.memory import set_allocator  # NOQA
from cupy.cuda.memory import get_allocator  # NOQA
from cupy.cuda.memory import UnownedMemory  # NOQA
from cupy.cuda.memory_hook import MemoryHook  # NOQA
from cupy.cuda.pinned_memory import alloc_pinned_memory  # NOQA
from cupy.cuda.pinned_memory import PinnedMemory  # NOQA
from cupy.cuda.pinned_memory import PinnedMemoryPointer  # NOQA
from cupy.cuda.pinned_memory import PinnedMemoryPool  # NOQA
from cupy.cuda.pinned_memory import set_pinned_memory_allocator  # NOQA
from cupy.cuda.stream import Event  # NOQA
from cupy.cuda.stream import get_current_stream  # NOQA
from cupy.cuda.stream import get_elapsed_time  # NOQA
from cupy.cuda.stream import Stream  # NOQA
from cupy.cuda.stream import ExternalStream  # NOQA
from cupy.cuda.graph import Graph  # NOQA


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
