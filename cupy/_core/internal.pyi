# TODO: Add entries when necessary

from typing import Any

import cupy
from cupy.typing import NDArray

def _get_strides_for_order_K(
    x: NDArray[Any], dtype: cupy.dtype, shape: list[int] | None = ...
) -> list[int]: ...
def _update_order_char(
    is_c_contiguous: bool, is_f_contiguous: bool, order_char: int
) -> int: ...
