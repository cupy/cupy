cimport cython
from cpython.pythread cimport (
    PyThread_tss_create, PyThread_tss_get, PyThread_tss_set)

import threading
import weakref


@cython.no_gc
cdef class _ThreadLocalBase:
    """Helper to organize thread-local storage that we need quite often
    in CuPy.
    In some cases it may be that contextvars are a better solution
    (one problem is that it is hard to support context managers that can be
    entered multiple times with them).

    The usage pattern is the following::

        cdef Py_tss_t _thread_local_key
        if PyThread_tss_create(&_thread_local_key) != 0:
            raise MemoryError()

        @cython.no_gc
        class _ThreadLocal(_ThreadLocalBase):
            # define any local objects and use an __init__ if needed.

            @staticmethod
            cpdef _ThreadLocal get():
                return <_ThreadLocal>_ThreadLocal.get(
                    _ThreadLocal, _thread_local_key)

    It is necessary to define the ``_thread_local_key`` module level.
    So unfortunately, this convoluted ``_get()`` helper call is necessary.
    """
    cdef object _wref

    @staticmethod
    cdef _get(cls, Py_tss_t& _thread_local_key)
