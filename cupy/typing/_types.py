from __future__ import annotations

from typing import Any, TypeVar

import numpy
from numpy.typing import ArrayLike, DTypeLike, NBitBase  # noqa: F401

from cupy._core import core

_ScalarType = TypeVar("_ScalarType", bound=numpy.generic)
NDArray = core.ndarray[Any, numpy.dtype[_ScalarType]]
