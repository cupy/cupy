from __future__ import annotations

import os
import warnings
from multiprocessing import get_start_method

from cupy.cuda.memory import UnownedMemory, MemoryPointer

from cupy_backends.cuda.api.runtime import (
    ipcGetMemHandle, ipcOpenMemHandle, ipcCloseMemHandle)

from cupy import _util


class IPCMemoryHandle:
    """
    A serializable class representing an Inter-process Communication (IPC)
    handle for sharing CuPy arrays between processes.
    """

    def __init__(self, ndarray_instance):
        """
        Initialize an IPCMemoryHandle object.

        Args:
            ndarray_instance (cupy.ndarray): CuPy array instance to create
            the handle.
        """
        from cupy._core.core import ndarray

        if type(ndarray_instance) is not ndarray:
            if isinstance(ndarray_instance, ndarray):
                warnings.warn("The provided object is an instance of a "
                              "subclass of cupy.ndarray, not cupy.ndarray "
                              "itself. The .open() method will always return "
                              "a cupy.ndarray object.")
            else:
                raise TypeError(
                    "The provided object is neither an instance of "
                    "cupy.ndarray, nor an instance of a subclass of "
                    "cupy.ndarray.")

        self.dtype = ndarray_instance.dtype
        self.shape = ndarray_instance.shape
        self.size = ndarray_instance.size
        self.strides = ndarray_instance.strides
        self.handle = ipcGetMemHandle(ndarray_instance.data.ptr)
        self.opened = False
        self.source_pid = os.getpid()
        self.arr_ptr = -1

    def __del__(self, is_shutting_down=_util.is_shutting_down):
        """
        Closes the handle when object is deleted or goes out of scope.
        """
        if is_shutting_down():
            return
        self.close()

    def close(self):
        """
        Closes the IPC handle.
        """
        if self.arr_ptr != -1:
            ipcCloseMemHandle(self.arr_ptr)
            self.arr_ptr = -1

    def open(self):
        """
        Retrieve the CuPy array associated with this IPC handle.

        Returns:
            cupy.ndarray: The shared CuPy array.

        Raises:
            RuntimeError: If open() is called on the source process or if
            open() has already been called, or if the process was created
            using the 'fork' method.
        """
        if os.getpid() == self.source_pid:
            raise RuntimeError(
                "The handle cannot be opened on the source process.")

        if self.opened:
            raise RuntimeError(
                "The handle was already opened for this process.")

        if get_start_method() == "fork":
            warnings.warn(
                "This method was called on a forked process. "
                "The handle cannot be opened on a direct forked descendant "
                "of the process on which the handle was created.")

        from cupy._core.core import ndarray

        try:
            self.arr_ptr = ipcOpenMemHandle(self.handle)
        except Exception as e:
            print("Error:", str(e))
            print("Possible reason: An attempt was made to open the handle "
                  "in a process that is a direct forked descendant of the "
                  "process on which the handle was created.")

        mem = UnownedMemory(self.arr_ptr, self.size, owner=self)
        memptr = MemoryPointer(mem, offset=0)

        arr = ndarray(shape=self.shape, dtype=self.dtype,
                      memptr=memptr, strides=self.strides)

        self.opened = True

        return arr
