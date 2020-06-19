cdef list _reduction_backends = []
cdef list _routine_backends = []


def _get_backend(str s):
    if s == 'cub':
        return BACKEND_CUB
    # if s == 'cutensor':
    #     return BACKEND_CUTENSOR
    raise ValueError('Unknown backend: {}'.format(s))


def set_reduction_backends(str s):
    global _reduction_backends
    _reduction_backends = [_get_backend(t) for t in s.split(',')]


def set_routine_backends(str s):
    global _routine_backends
    _routine_backends = [_get_backend(t) for t in s.split(',')]
