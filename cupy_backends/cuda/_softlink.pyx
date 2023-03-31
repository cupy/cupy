import ctypes
import warnings

from libc.stdint cimport intptr_t
cimport cython


cdef class SoftLink:
    def __init__(self, object libname, str prefix):
        self._cdll = None
        if libname is not None:
            try:
                self._cdll = ctypes.CDLL(libname)
            except Exception as e:
                warnings.warn(
                    f'Warning: CuPy failed to load "{libname}": '
                    f'({type(e).__name__}: {e})')
        self._prefix = prefix

    cdef func_ptr get(self, str name):
        """
        Returns a function pointer for the API.
        """
        if self._cdll is None:
            return <func_ptr>_fail_unsupported
        cdef str funcname = f'{self._prefix}{name}'
        cdef object func = getattr(self._cdll, funcname, None)
        if func is None:
            return <func_ptr>_fail_not_found
        cdef intptr_t ptr = ctypes.addressof(func)
        return cython.operator.dereference(<func_ptr*>ptr)


cdef int _fail_unsupported() nogil except -1:
    with gil:
        raise AssertionError('''
*** The requested function is not supported in the current version of
*** the toolkit installed in your environment.
***
*** This is likely a bug in CuPy. Please report this issue to:
***   https://github.com/cupy/cupy/issues
''')

cdef int _fail_not_found() nogil except -1:
    with gil:
        raise AssertionError('''
*** The requested function could not be found in the library.
***
*** This is likely a bug in CuPy. Please report this issue to:
***   https://github.com/cupy/cupy/issues
''')
