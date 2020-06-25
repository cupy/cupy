import os

cdef list _reduction_backends = []
cdef list _routine_backends = []


def _get_backend(str s):
    if s == 'cub':
        return BACKEND_CUB
    # if s == 'cutensor':
    #     return BACKEND_CUTENSOR
    raise ValueError('Unknown backend: {}'.format(s))


def set_reduction_backends(backends):
    global _reduction_backends
    _reduction_backends = [_get_backend(b) for b in backends]


def set_routine_backends(backends):
    global _routine_backends
    _routine_backends = [_get_backend(b) for b in backends]


def _get_routine_backends():
    return _routine_backends


cdef _set_default_backends():
    cdef str b, backend_names = os.getenv('CUPY_BACKENDS', '')
    cdef list backends = [b for b in backend_names.split(',') if b]
    set_reduction_backends(backends)
    set_routine_backends(backends)


_set_default_backends()
