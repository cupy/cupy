import contextlib
import warnings

from cupy._environment import get_cuda_path, get_nvcc_path  # NOQA
from cupy.cuda import compiler  # NOQA
from cupy.cuda import device  # NOQA
from cupy.cuda import function  # NOQA
from cupy.cuda import memory  # NOQA
from cupy.cuda import memory_hook  # NOQA
from cupy.cuda import memory_hooks  # NOQA
from cupy.cuda import pinned_memory  # NOQA
from cupy.cuda import stream  # NOQA
from cupy.cuda import texture  # NOQA
from cupy_backends.cuda.api import driver  # NOQA
from cupy_backends.cuda.api import runtime  # NOQA
from cupy_backends.cuda.libs import cublas  # NOQA
from cupy_backends.cuda.libs import curand  # NOQA
from cupy_backends.cuda.libs import cusolver  # NOQA
from cupy_backends.cuda.libs import cusparse  # NOQA
from cupy_backends.cuda.libs import nvrtc  # NOQA
from cupy_backends.cuda.libs import profiler  # NOQA


_available = None


class _UnavailableModule():
    available = False

    def __init__(self, name):
        self.__name__ = name


# TODO(leofang): always import cub (but not enable it) when hipCUB is supported
if not runtime.is_hip:
    from cupy.cuda import cub  # NOQA
else:
    cub = _UnavailableModule('cupy.cuda.cub')

try:
    from cupy.cuda import nvtx  # NOQA
except ImportError:
    nvtx = _UnavailableModule('cupy.cuda.nvtx')

try:
    from cupy.cuda import thrust  # NOQA
except ImportError:
    thrust = _UnavailableModule('cupy.cuda.thrust')

try:
    from cupy.cuda import nccl  # NOQA
except ImportError:
    nccl = _UnavailableModule('cupy.cuda.nccl')

try:
    from cupy_backends.cuda.libs import cutensor
except ImportError:
    cutensor = _UnavailableModule('cupy.cuda.cutensor')


def __getattr__(key):
    # `*_enabled` flags are kept for backward compatibility.
    # Note: module-level getattr only runs on Python 3.7+.
    if key == 'cusolver_enabled':
        # cuSOLVER is always available in CUDA 8.0+.
        warnings.warn('''
cupy.cuda.cusolver_enabled has been deprecated in CuPy v8 and will be removed in the future release.
This flag always returns True as cuSOLVER is always available in CUDA 8.0 or later.
            ''', DeprecationWarning)  # NOQA
        return True

    for mod in [nvtx, nccl, thrust, cub, cutensor]:
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
            if (e.args[0] !=
                    'cudaErrorNoDevice: no CUDA-capable device is detected'):
                raise
    return _available


# import class and function
from cupy.cuda.compiler import compile_with_cache  # NOQA
from cupy.cuda.device import Device  # NOQA
from cupy.cuda.device import get_cublas_handle  # NOQA
from cupy.cuda.device import get_device_id  # NOQA
from cupy.cuda.function import Function  # NOQA
from cupy.cuda.function import Module  # NOQA
from cupy.cuda.memory import alloc  # NOQA
from cupy.cuda.memory import BaseMemory  # NOQA
from cupy.cuda.memory import malloc_managed  # NOQA
from cupy.cuda.memory import ManagedMemory  # NOQA
from cupy.cuda.memory import Memory  # NOQA
from cupy.cuda.memory import MemoryPointer  # NOQA
from cupy.cuda.memory import MemoryPool  # NOQA
from cupy.cuda.memory import PythonFunctionAllocator  # NOQA
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

    """
    profiler.start()
    try:
        yield
    finally:
        profiler.stop()
