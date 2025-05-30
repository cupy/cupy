# cython: language_level=3
# cython: binding=True

"""
fastrlock equivalent using threading.RLock
"""

import threading


cdef create_fastrlock():
    return threading.RLock()


cdef bint lock_fastrlock(rlock, long current_thread, bint blocking) except -1:
    # The 'current_thread' argument is deprecated and ignored.
    # Pass -1 for backwards compatibility.
    return rlock.acquire(blocking=blocking)


cdef int unlock_fastrlock(rlock) except -1:
    rlock.release()
    return 0
