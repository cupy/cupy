# cython: language_level=3

cdef create_fastrlock()
cdef bint lock_fastrlock(rlock, long current_thread, bint blocking) except -1
cdef int unlock_fastrlock(rlock) except -1
