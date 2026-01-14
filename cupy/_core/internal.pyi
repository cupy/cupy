# TODO: Add entries when necessary

from typing import Any

from numpy import dtype  # MEMO: mypy complains if imported from cupy

from cupy._core.core import NDArray

def _get_strides_for_order_K(
    x: NDArray[Any], dtype: dtype, shape: list[int] | None = ...
) -> list[int]: ...
def _update_order_char(
    is_c_contiguous: bool, is_f_contiguous: bool, order_char: int
) -> int: ...
