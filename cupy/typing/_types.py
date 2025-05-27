from __future__ import annotations

from numpy.typing import ArrayLike, DTypeLike, NBitBase  # noqa: F401
from typing import TypeVar
import numpy
from cupy.typing._internal import _ScalarT

from cupy._core import core

NDArray = core.ndarray[numpy.dtype[_ScalarT]]

_BoolOrIntArrayT = TypeVar(
    "_BoolOrIntArrayT", bound=NDArray[numpy.integer | numpy.bool]
)
