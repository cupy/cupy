import os
from cupy.cuda.memory import UnownedMemory, MemoryPointer
from cupy.cuda.runtime import ipcOpenMemHandle, ipcCloseMemHandle
from multiprocessing import get_start_method


class IPCHandle:
    """
    A serializable class representing an Inter-process Communication (IPC)
    handle for sharing Cupy arrays between processes.
    """

    def __init__(self, shape, size, dtype, strides, handle):
        """
        Initialize an IPCHandle object.

        Args:
            shape (tuple of ints): The shape of the array.
            size (int): The size of the array in bytes.
            dtype: The data type of the array.
            strides (tuple of ints): The strides of the array.
            handle: Pointer to the GPU memory.
        """
        self.dtype = dtype
        self.shape = shape
        self.size = size
        self.strides = strides
        self.handle = handle
        self.opened = False
        self.source_pid = os.getpid()
        self.arr_ptr = -1

    def __del__(self):
        """
        Closes the handle when object is deleted or goes out of scope.
        """
        if self.arr_ptr != -1:
            ipcCloseMemHandle(self.arr_ptr)

    def close(self):
        """
        Closes the IPC handle.
        """
        if self.arr_ptr != -1:
            ipcCloseMemHandle(self.arr_ptr)
            self.arr_ptr = -1

    def get(self):
        """
        Retrieve the Cupy array associated with this IPC handle.

        Returns:
            Cupy array: The shared Cupy array.

        Raises:
            RuntimeError: If get() is called on the source process or if
            get() has already been called, or if the process was created
            using the 'fork' method.
        """
        if get_start_method() == "fork":
            raise RuntimeError(
                "The handle cannot be spawned on a forked process.")

        if os.getpid() == self.source_pid:
            raise RuntimeError(
                "The handle cannot be opened on the source process.")

        if self.opened:
            raise RuntimeError(
                "The handle was already opened for this process.")

        from cupy._core.core import ndarray

        self.arr_ptr = ipcOpenMemHandle(self.handle)
        mem = UnownedMemory(self.arr_ptr, self.size, owner=self)
        memptr = MemoryPointer(mem, offset=0)
        arr = ndarray(shape=self.shape, dtype=self.dtype,
                      memptr=memptr, strides=self.strides)

        self.opened = True

        return arr
