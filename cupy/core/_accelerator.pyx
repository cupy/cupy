import os

cdef list _reduction_accelerators = []
cdef list _routine_accelerators = []


cdef int _get_accelerator(accelerator) except -1:
    if isinstance(accelerator, int):
        return accelerator
    if accelerator == 'cub':
        return ACCELERATOR_CUB
    if accelerator == 'cutensor':
        return ACCELERATOR_CUTENSOR
    raise ValueError('Unknown accelerator: {}'.format(accelerator))


def set_reduction_accelerators(accelerators):
    global _reduction_accelerators
    _reduction_accelerators = [_get_accelerator(b) for b in accelerators]


def set_routine_accelerators(accelerators):
    global _routine_accelerators
    _routine_accelerators = [_get_accelerator(b) for b in accelerators]


def get_reduction_accelerators():
    return _reduction_accelerators


def get_routine_accelerators():
    return _routine_accelerators


cdef _set_default_accelerators():
    cdef str b, accelerator_names = os.getenv('CUPY_ACCELERATORS', '')
    cdef list accelerators = [b for b in accelerator_names.split(',') if b]
    set_reduction_accelerators(accelerators)
    set_routine_accelerators(accelerators)


_set_default_accelerators()
