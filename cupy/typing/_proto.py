from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, SupportsIndex

if TYPE_CHECKING:
    from _typeshed import SupportsFlush
else:
    SupportsFlush = object

class _SupportsFileMethods(SupportsFlush, Protocol):
    def fileno(self) -> SupportsIndex: ...
    def tell(self) -> SupportsIndex: ...
    def seek(self, offset: int, whence: int, /) -> object: ...
