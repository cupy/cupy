from typing import Any, Generic, TypeVar

import numpy

_ShapeT_co = TypeVar(
    "_ShapeT_co", bound=tuple[int, ...], default=Any, covariant=True
)
_DTypeT_co = TypeVar(
    "_DTypeT_co", bound=numpy.dtype, default=numpy.dtype, covariant=True
)

class ndarray(Generic[_ShapeT_co, _DTypeT_co]): ...
