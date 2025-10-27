import cupy
from cupy.cuda import runtime

class IPCHandle:
    """Helper class for sharing GPU memory between processes using IPC."""

    def __init__(self, array):
        self._shape = array.shape
        self._strides = array.strides
        self._dtype = array.dtype
        self._handle = runtime.ipcGetMemHandle(array.data.ptr)
        self._closed = False

    def get(self):
        """Open the IPC handle and return a CuPy array."""
        if self._closed:
            raise RuntimeError("IPC handle has already been used or closed.")

        ptr = runtime.ipcOpenMemHandle(self._handle)
        a = cupy.ndarray(self._shape, self._dtype, cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(ptr, 0, self), 0))
        self._closed = True
        return a

    def __del__(self):
        if not self._closed:
            runtime.ipcCloseMemHandle(self._handle)
            self._closed = True


def get_ipc_handle(array):
    """Return an IPCHandle for a given CuPy array."""
    return IPCHandle(array)
