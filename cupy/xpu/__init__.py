from __future__ import annotations

import contextlib
import warnings

import cupy as _cupy
from cupy._environment import get_cuda_path  # NOQA
from cupy._environment import get_nvcc_path  # NOQA
from cupy._environment import get_rocm_path  # NOQA
from cupy._environment import get_hipcc_path  # NOQA
from cupy._environment import get_cann_path # NOQA

#from backends.backend.runtime import is_ascend as _is_ascend

_is_ascend = True # TODO: ASCEND get from env?
if not _is_ascend:
    from cupy.xpu import compiler  # NOQA
    from cupy.xpu import function  # NOQA
    from cupy.xpu import texture  # NOQA
from cupy.xpu import device  # NOQA
from cupy.xpu import memory  # NOQA
from cupy.xpu import memory_hook  # NOQA
from cupy.xpu import memory_hooks  # NOQA
from cupy.xpu import pinned_memory  # NOQA
from cupy.xpu import profiler  # NOQA
from cupy.xpu import stream  # NOQA

from backends.backend.api import driver  # NOQA
from backends.backend.api import runtime  # NOQA

# import class and function
from cupy.xpu.device import Device  # NOQA
from cupy.xpu.device import get_cublas_handle  # NOQA
from cupy.xpu.device import get_device_id  # NOQA
if not _is_ascend:
    from cupy.xpu.function import Function  # NOQA
    from cupy.xpu.function import Module  # NOQA
from cupy.xpu.memory import alloc  # NOQA
from cupy.xpu.memory import BaseMemory  # NOQA
from cupy.xpu.memory import malloc_managed  # NOQA
from cupy.xpu.memory import malloc_async  # NOQA
from cupy.xpu.memory import ManagedMemory  # NOQA
from cupy.xpu.memory import Memory  # NOQA
from cupy.xpu.memory import MemoryAsync  # NOQA
from cupy.xpu.memory import MemoryPointer  # NOQA
from cupy.xpu.memory import MemoryPool  # NOQA
if not _is_ascend:
    from cupy.xpu.memory import MemoryAsyncPool  # NOQA
from cupy.xpu.memory import PythonFunctionAllocator  # NOQA
from cupy.xpu.memory import CFunctionAllocator  # NOQA
from cupy.xpu.memory import set_allocator  # NOQA
from cupy.xpu.memory import get_allocator  # NOQA
from cupy.xpu.memory import UnownedMemory  # NOQA
from cupy.xpu.memory_hook import MemoryHook  # NOQA

from cupy.xpu.pinned_memory import alloc_pinned_memory  # NOQA
from cupy.xpu.pinned_memory import PinnedMemory  # NOQA
from cupy.xpu.pinned_memory import PinnedMemoryPointer  # NOQA
from cupy.xpu.pinned_memory import PinnedMemoryPool  # NOQA
from cupy.xpu.pinned_memory import set_pinned_memory_allocator  # NOQA

from cupy.xpu.stream import Event  # NOQA
from cupy.xpu.stream import get_current_stream  # NOQA
from cupy.xpu.stream import get_elapsed_time  # NOQA
from cupy.xpu.stream import Stream  # NOQA
from cupy.xpu.stream import ExternalStream  # NOQA
if not _is_ascend:
    from cupy.xpu.graph import Graph  # NOQA


@contextlib.contextmanager
def using_allocator(allocator=None):
    """Sets a thread-local allocator for GPU memory inside
       context manager

    Args:
        allocator (function): CuPy memory allocator. It must have the same
            interface as the :func:`cupy.xpu.alloc` function, which takes the
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

    >>> with cupy.xpu.profile():
    ...    # do something you want to measure
    ...    pass

    .. note::
        When starting ``nvprof`` from the command line, manually setting
        ``--profile-from-start off`` may be required for the desired behavior.

    .. warning:: This context manager is deprecated. Please use
        :class:`cupyx.profiler.profile` instead.
    """
    warnings.warn(
        'cupy.xpu.profile has been deprecated since CuPy v10 '
        'and will be removed in the future. Use cupyx.profiler.profile '
        'instead.')

    profiler.start()
    try:
        yield
    finally:
        profiler.stop()