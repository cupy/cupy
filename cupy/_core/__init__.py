# mypy: ignore-errors

from cupy._core import core, fusion, internal

# internal APIs for testing and development
from cupy._core._accelerator import (
    get_elementwise_accelerators,
    get_reduction_accelerators,
    get_routine_accelerators,
    set_elementwise_accelerators,
    set_reduction_accelerators,
    set_routine_accelerators,
)

# import class and function
from cupy._core._kernel import (
    ElementwiseKernel,
    _get_warpsize,
    create_ufunc,
    ufunc,
)
from cupy._core._reduction import ReductionKernel, create_reduction_func
from cupy._core._routines_binary import (
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    invert,
    left_shift,
    right_shift,
)
from cupy._core._routines_linalg import (
    _mat_ptrs,
    dot,
    get_compute_type,
    matmul,
    set_compute_type,
    tensordot_core,
)
from cupy._core._routines_logic import (
    create_comparison,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)
from cupy._core._routines_manipulation import (
    array_split,
    broadcast,
    broadcast_to,
    concatenate_method,
    moveaxis,
    rollaxis,
    size,
)
from cupy._core._routines_math import (
    absolute,
    add,
    angle,
    angle_deg,
    conjugate,
    divide,
    floor_divide,
    multiply,
    negative,
    positive,
    power,
    remainder,
    sqrt,
    subtract,
    true_divide,
)
from cupy._core._routines_statistics import nanmax, nanmin
from cupy._core.core import (
    _internal_ascontiguousarray,
    _internal_asfortranarray,
    array,
    ascontiguousarray,
    asfortranarray,
    divmod,
    elementwise_copy,
    ndarray,
)
from cupy._core.dlpack import from_dlpack, fromDlpack
from cupy._core.internal import complete_slice, get_size
from cupy._core.raw import RawKernel, RawModule
