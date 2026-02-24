cimport cython
from cpython.pythread cimport PyThread_tss_get, PyThread_tss_set

import gc
import threading
import weakref


@cython.no_gc  # Must have no GC or the `_wref` trick needs strengthening
cdef class _ThreadLocalBase:
    """Helper for thread-local storage, see .pxd file for details.
    """
    def _cleanup(self, ref):
        # Use a Python method for cleanup, this way we store a bound
        # method, which keeps alive the ``_ThreadLocal`` instance ``self``.
        # We clean up the weakref (just in case it doesn't clean up it's
        # __callback__ function -- i.e. this method)
        self._wref = None

    @staticmethod
    cdef _get(cls, Py_tss_t& _thread_local_key):
        cdef _ThreadLocalBase new
        cdef void *tls = PyThread_tss_get(&_thread_local_key)
        if tls != NULL:
            return <object>tls

        new = cls()
        if gc.is_tracked(new):
            raise RuntimeError(
                "internal error: _ThreadLocal classes are assumed to be "
                "marked `@cython.no_gc` (may be needed for _wref trick).")
        tls = <void *>new
        if PyThread_tss_set(&_thread_local_key, tls) != 0:
            raise MemoryError()

        # Tie the lifetime of `new` to the thread (hope it is cleaned up).
        # This is how threading.local() works as well (2025-11).
        new._wref = weakref.ref(threading.current_thread(), new._cleanup)
        return new
