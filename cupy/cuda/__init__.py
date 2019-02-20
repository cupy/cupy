import contextlib
import os

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


_available = None
_cuda_path = None


from cupy.cuda import cusolver  # NOQA
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

try:
    from cupy.cuda import nccl  # NOQA
    nccl_enabled = True
except ImportError:
    nccl_enabled = False


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


def get_cuda_path():
    global _cuda_path
    if _cuda_path is None:
        _cuda_path = os.getenv('CUDA_PATH', None)
        if _cuda_path is not None:
            return _cuda_path

        for p in os.getenv('PATH', '').split(os.pathsep):
            for cmd in ('nvcc', 'nvcc.exe'):
                nvcc_path = os.path.join(p, cmd)
                if not os.path.exists(nvcc_path):
                    continue
                nvcc_dir = os.path.dirname(os.path.abspath(nvcc_path))
                _cuda_path = os.path.normpath(os.path.join(nvcc_dir, '..'))
                return _cuda_path

        if os.path.exists('/usr/local/cuda'):
            _cuda_path = '/usr/local/cuda'

    return _cuda_path


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
