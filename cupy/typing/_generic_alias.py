from typing import Any, TypeVar

import numpy
from cupy._core import core

from numpy.typing import ArrayLike  # NOQA
from numpy.typing import DTypeLike  # NOQA
from numpy.typing import NBitBase  # NOQA
from numpy._typing import _DTypeLike  # NOQA
from numpy._typing import _ShapeLike  # NOQA


_ScalarType_co = TypeVar("_ScalarType_co", bound=numpy.generic, covariant=True)
NDArray = core.ndarray[Any, numpy.dtype[_ScalarType_co]]
