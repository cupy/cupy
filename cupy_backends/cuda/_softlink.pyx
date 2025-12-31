import ctypes
import warnings

from libc.stdint cimport intptr_t
cimport cython


def _log(msg: str) -> None:
    import cupy
    cupy._environment._log(msg)


cdef class SoftLink:
    def __init__(
            self, str libname, str prefix, *, bint mandatory=False, handle=0):
        self.error = None
        self.prefix = prefix
        self._libname = libname
        self._cdll = None
        if libname is None and handle == 0:
            # Stub build or CUDA/HIP only library.
            self.error = RuntimeError(
                'The library is unavailable in the current platform.')
        elif libname is None and handle > 0:
            self._cdll = ctypes.CDLL(None, handle=handle)
            # Note: It seems CDLL does not honor handle starting Python 3.13,
            # so we overwrite it to be safe. (xref: python/cpython#143304)
            self._cdll._handle = handle
            _log(f'Library already loaded (prefix={prefix}, handle={handle})')
        else:
            # Ignore handle.
            try:
                self._cdll = ctypes.CDLL(libname)
                _log(f'Library "{libname}" loaded')
            except Exception as e:
                self.error = e
                msg = (
                    f'CuPy failed to load {libname}: {type(e).__name__}: {e}')
                if mandatory:
                    raise RuntimeError(msg) from e
                warnings.warn(msg)

    cdef func_ptr get(self, str name):
        """
        Returns a function pointer for the API.
        """
        cdef str funcname = f'{self.prefix}{name}'
        if self._cdll is None:
            _log(f'[NOTICE] {self._libname} ({funcname}): library not loaded')
            return <func_ptr>_fail_unsupported
        cdef object func = getattr(self._cdll, funcname, None)
        if func is None:
            _log(f'[NOTICE] {self._libname} ({funcname}): function not found')
            return <func_ptr>_fail_not_found
        cdef intptr_t ptr = ctypes.addressof(func)
        _log(f'{self._libname} ({funcname}): function loaded')
        return cython.operator.dereference(<func_ptr*>ptr)


cdef int _fail_unsupported() except -1 nogil:
    with gil:
        raise AssertionError('''
*** The requested function is not supported in the current version of
*** the toolkit installed in your environment.
***
*** This is likely a bug in CuPy. Please report this issue to:
***   https://github.com/cupy/cupy/issues
''')

cdef int _fail_not_found() except -1 nogil:
    with gil:
        raise AssertionError('''
*** The requested function could not be found in the library.
***
*** This is likely a bug in CuPy. Please report this issue to:
***   https://github.com/cupy/cupy/issues
''')
