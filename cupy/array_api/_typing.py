"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

from cupy.cuda import Device

__all__ = [
    "Array",
    "Device",
    "Dtype",
    "SupportsDLPack",
    "SupportsBufferProtocol",
    "PyCapsule",
]

import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from numpy import (
    dtype,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

from ._array_object import Array

_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...


if TYPE_CHECKING or sys.version_info >= (3, 9):
    Dtype = dtype[Union[
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
    ]]
else:
    Dtype = dtype

SupportsBufferProtocol = Any
PyCapsule = Any


class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule: ...
