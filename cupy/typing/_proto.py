from __future__ import annotations

import sys
import typing
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex, TypeVar

from cupy._core import core
from cupy.typing._internal import _DTypeT_co

if TYPE_CHECKING:
    from _typeshed import SupportsFlush
else:
    SupportsFlush = object

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as _Buffer
else:

    @typing.runtime_checkable
    class _Buffer(Protocol):
        def __buffer__(self, flags: int, /) -> memoryview: ...


class _SupportsFileMethods(SupportsFlush, Protocol):
    def fileno(self) -> SupportsIndex: ...
    def tell(self) -> SupportsIndex: ...
    def seek(self, offset: int, whence: int, /) -> object: ...


@typing.runtime_checkable
class _SupportsArray(Protocol[_DTypeT_co]):
    # Only covers default signature (no optional args provided)
    def __array__(self) -> core.ndarray[Any, _DTypeT_co]: ...


_T_co = TypeVar("_T_co", covariant=True)


@typing.runtime_checkable
class _NestedSequence(Protocol[_T_co]):
    # Required methods to be collections.abc.Sequence
    def __getitem__(self, index: int, /) -> _T_co | _NestedSequence[_T_co]:
        raise NotImplementedError

    def __len__(self, /) -> int:
        raise NotImplementedError

    # Methods can be derived by collections.abc.Sequence
    def __contains__(self, x: object, /) -> bool:
        raise NotImplementedError

    def __iter__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]:
        raise NotImplementedError

    def __reversed__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]:
        raise NotImplementedError

    def count(self, value: Any, /) -> int:
        raise NotImplementedError

    def index(self, value: Any, /) -> int:
        raise NotImplementedError
