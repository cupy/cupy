from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike, DTypeLike, NBitBase  # noqa: F401
import numpy
from cupy.typing._internal import _ScalarT

from cupy._core import core

NDArray = core.ndarray[Any, numpy.dtype[_ScalarT]]
