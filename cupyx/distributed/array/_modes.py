import dataclasses
import typing
from typing import Any, Callable, Final, Iterable, Optional, TypeVar, Union
from typing_extensions import TypeGuard


import numpy
import cupy
from cupy import _core

from numpy.typing import ArrayLike

from cupyx.distributed._nccl_comm import _get_nccl_dtype_and_count
from cupyx.distributed.array import _linalg
from cupyx.distributed.array import _index_arith


def _min_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(cupy.iinfo(dtype).min)
    elif dtype.kind in 'f':
        return dtype.type(-cupy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


def _max_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(cupy.iinfo(dtype).max)
    elif dtype.kind in 'f':
        return dtype.type(cupy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


def _zero_value_of(dtype):
    return dtype.type(0)


def _one_value_of(dtype):
    return dtype.type(1)


class OpMode:
    func: cupy.ufunc
    numpy_func: numpy.ufunc
    idempotent: bool
    identity_of: Callable

    _T = TypeVar('_T', bound=numpy.generic)
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


Mode = Optional[OpMode]


REPLICA_MODE: Final[Mode] = None


def is_op_mode(mode: Mode) -> TypeGuard[OpMode]:
    return mode is not REPLICA_MODE


MODES: Final[dict[str, Mode]] = {
    'replica': REPLICA_MODE,
    'min':  OpMode('minimum',  True,  _max_value_of),
    'max':  OpMode('maximum',  True,  _min_value_of),
    'sum':  OpMode('add',      False, _zero_value_of),
    'prod': OpMode('multiply', False, _one_value_of),
}
