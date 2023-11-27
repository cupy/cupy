import functools
from typing import Callable, Final, Optional

import numpy

import cupy


@functools.lru_cache
def _min_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(numpy.iinfo(dtype).min)
    elif dtype.kind in 'f':
        return dtype.type(-numpy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


@functools.lru_cache
def _max_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(numpy.iinfo(dtype).max)
    elif dtype.kind in 'f':
        return dtype.type(numpy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


@functools.lru_cache
def _zero_of(dtype):
    return dtype.type(0)


@functools.lru_cache
def _one_of(dtype):
    return dtype.type(1)


class _OpMode:
    func: cupy._core._kernel.ufunc
    numpy_func: numpy.ufunc
    idempotent: bool
    identity_of: Callable

    def __init__(
        self, func_name: str, idempotent: bool, identity_of: Callable,
    ) -> None:
        try:
            self.func = getattr(cupy, func_name)
            self.numpy_func = getattr(numpy, func_name)
        except AttributeError:
            raise RuntimeError('No such function exists')

        self.idempotent = idempotent
        self.identity_of = identity_of


Mode = Optional[_OpMode]


REPLICA: Final[None] = None
MIN = _OpMode('minimum',  True,  _max_value_of)
MAX = _OpMode('maximum',  True,  _min_value_of)
SUM = _OpMode('add', False, _zero_of)
PROD = _OpMode('multiply', False, _one_of)
