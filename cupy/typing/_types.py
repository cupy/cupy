from __future__ import annotations

from collections.abc import Sequence

from typing import Any, Literal, SupportsIndex, TypeVar

import numpy
from cupy._core import core

from numpy.typing import ArrayLike  # NOQA
from numpy.typing import DTypeLike  # NOQA
from numpy.typing import NBitBase  # NOQA


# Shapes
_Shape = tuple[int, ...]
_ShapeLike = SupportsIndex | Sequence[SupportsIndex]

_OrderKACF = Literal["K", "A", "C", "F"] | None
_OrderCF = Literal["C", "F"] | None
_ScalarType_co = TypeVar("_ScalarType_co", bound=numpy.generic, covariant=True)
NDArray = core.ndarray[Any, numpy.dtype[_ScalarType_co]]
