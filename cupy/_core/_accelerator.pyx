import os

from cupy_backends.cuda.api cimport runtime


cdef list _elementwise_accelerators = []
cdef list _reduction_accelerators = []
cdef list _routine_accelerators = []


cdef int _get_accelerator(accelerator) except -1:
    if isinstance(accelerator, int):
        return accelerator
    if accelerator == 'cub':
        return ACCELERATOR_CUB
    if accelerator == 'cutensor':
        return ACCELERATOR_CUTENSOR
    if accelerator == 'cutensornet':
        return ACCELERATOR_CUTENSORNET
    raise ValueError('Unknown accelerator: {}'.format(accelerator))


def set_elementwise_accelerators(accelerators):
    global _elementwise_accelerators
    _elementwise_accelerators = [_get_accelerator(b) for b in accelerators]


def set_reduction_accelerators(accelerators):
    global _reduction_accelerators
    _reduction_accelerators = [_get_accelerator(b) for b in accelerators]


def set_routine_accelerators(accelerators):
    global _routine_accelerators
    _routine_accelerators = [_get_accelerator(b) for b in accelerators]


def get_elementwise_accelerators():
    return _elementwise_accelerators


def get_reduction_accelerators():
    return _reduction_accelerators


def get_routine_accelerators():
    return _routine_accelerators


cdef _set_default_accelerators():
    cdef str b, accelerator_names = os.getenv(
        'CUPY_ACCELERATORS', '' if runtime._is_hip_environment else 'cub')
    cdef list accelerators = [b for b in accelerator_names.split(',') if b]
    set_elementwise_accelerators(accelerators)
    set_reduction_accelerators(accelerators)
    set_routine_accelerators(accelerators)


_set_default_accelerators()
