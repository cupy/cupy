"""Type utilities dependent on cupy."""

from __future__ import annotations

import typing
from typing import Any, Protocol

from cupy._core import core
from cupy.typing._standalone import _T, _DTypeT, _DTypeT_co, _NestedSequence


@typing.runtime_checkable
class _SupportsArray(Protocol[_DTypeT_co]):
    # Only covers default signature (no optional args provided)
    def __array__(self) -> core.ndarray[Any, _DTypeT_co]: ...


_DualArrayLike = (
    _SupportsArray[_DTypeT]
    | _NestedSequence[_SupportsArray[_DTypeT]]
    | _T
    | _NestedSequence[_T]
)

# Anything castable to ndarray with specified dtype (work in progress)
_ArrayLikeInt = Any
