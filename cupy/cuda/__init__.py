import contextlib
import os

from cupy._environment import get_cuda_path  # NOQA
from cupy.cuda import compiler  # NOQA
from cupy.cuda import device  # NOQA
from cupy.cuda import driver  # NOQA
from cupy.cuda import function  # NOQA
from cupy.cuda import memory  # NOQA
from cupy.cuda import memory_hook  # NOQA
from cupy.cuda import memory_hooks  # NOQA
from cupy.cuda import pinned_memory  # NOQA
from cupy.cuda import profiler  # NOQA
from cupy.cuda import runtime  # NOQA
from cupy.cuda import stream  # NOQA
from cupy.cuda import texture  # NOQA


_available = None
_cub_disabled = None


from cupy.cuda import cusolver  # NOQA
# This flag is kept for backward compatibility.
# It is always True as cuSOLVER library is always available in CUDA 8.0+.
cusolver_enabled = True

try:
    from cupy.cuda import nvtx  # NOQA
    nvtx_enabled = True
except ImportError:
    nvtx_enabled = False

try:
    from cupy.cuda import thrust  # NOQA
    thrust_enabled = True
except ImportError:
    thrust_enabled = False

cub_enabled = False
if int(os.getenv('CUB_DISABLED', 0)) == 0:
    try:
        from cupy.cuda import cub  # NOQA
        cub_enabled = True
    except ImportError:
        pass

try:
    from cupy.cuda import nccl  # NOQA
    nccl_enabled = True
except ImportError:
    nccl_enabled = False

try:
    from cupy.cuda import cutensor  # NOQA
    cutensor_enabled = True
except ImportError:
    cutensor_enabled = False


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
