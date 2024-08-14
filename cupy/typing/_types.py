from collections.abc import Sequence
from typing import Any, Literal, SupportsIndex, TypeVar, Union

import numpy
from numpy.typing import (
    ArrayLike,  # NOQA
    DTypeLike,  # NOQA
    NBitBase,  # NOQA
)

from cupy._core import core

# Shapes
_Shape = tuple[int, ...]
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

_OrderKACF = Literal[None, "K", "A", "C", "F"]
_OrderCF = Literal[None, "C", "F"]
_ScalarType_co = TypeVar("_ScalarType_co", bound=numpy.generic, covariant=True)
NDArray = core.ndarray[Any, numpy.dtype[_ScalarType_co]]
