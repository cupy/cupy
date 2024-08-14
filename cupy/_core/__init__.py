# mypy: ignore-errors

from cupy._core import core
from cupy._core import fusion
from cupy._core import internal


# internal APIs for testing and development
from cupy._core._accelerator import set_elementwise_accelerators
from cupy._core._accelerator import set_reduction_accelerators
from cupy._core._accelerator import set_routine_accelerators
from cupy._core._accelerator import get_elementwise_accelerators
from cupy._core._accelerator import get_reduction_accelerators
from cupy._core._accelerator import get_routine_accelerators


# import class and function
from cupy._core._kernel import create_ufunc
from cupy._core._kernel import ElementwiseKernel
from cupy._core._kernel import ufunc
from cupy._core._kernel import _get_warpsize
from cupy._core._reduction import create_reduction_func
from cupy._core._reduction import ReductionKernel
from cupy._core._routines_binary import bitwise_and
from cupy._core._routines_binary import bitwise_or
from cupy._core._routines_binary import bitwise_xor
from cupy._core._routines_binary import invert
from cupy._core._routines_binary import left_shift
from cupy._core._routines_binary import right_shift
from cupy._core._routines_linalg import _mat_ptrs
from cupy._core._routines_linalg import dot
from cupy._core._routines_linalg import get_compute_type
from cupy._core._routines_linalg import matmul
from cupy._core._routines_linalg import set_compute_type
from cupy._core._routines_linalg import tensordot_core
from cupy._core._routines_logic import create_comparison
from cupy._core._routines_logic import equal
from cupy._core._routines_logic import greater
from cupy._core._routines_logic import greater_equal
from cupy._core._routines_logic import less
from cupy._core._routines_logic import less_equal
from cupy._core._routines_logic import not_equal
from cupy._core._routines_manipulation import array_split
from cupy._core._routines_manipulation import broadcast
from cupy._core._routines_manipulation import broadcast_to
from cupy._core._routines_manipulation import concatenate_method
from cupy._core._routines_manipulation import moveaxis
from cupy._core._routines_manipulation import rollaxis
from cupy._core._routines_manipulation import size
from cupy._core._routines_math import absolute
from cupy._core._routines_math import add
from cupy._core._routines_math import angle, angle_deg
from cupy._core._routines_math import conjugate
from cupy._core._routines_math import divide
from cupy._core._routines_math import floor_divide
from cupy._core._routines_math import multiply
from cupy._core._routines_math import negative
from cupy._core._routines_math import positive
from cupy._core._routines_math import power
from cupy._core._routines_math import remainder
from cupy._core._routines_math import sqrt
from cupy._core._routines_math import subtract
from cupy._core._routines_math import true_divide
from cupy._core._routines_statistics import nanmax
from cupy._core._routines_statistics import nanmin
from cupy._core.core import _internal_ascontiguousarray
from cupy._core.core import _internal_asfortranarray
from cupy._core.core import array
from cupy._core.core import ascontiguousarray
from cupy._core.core import asfortranarray
from cupy._core.core import divmod
from cupy._core.core import elementwise_copy
from cupy._core.core import ndarray
from cupy._core.dlpack import fromDlpack
from cupy._core.dlpack import from_dlpack
from cupy._core.internal import complete_slice
from cupy._core.internal import get_size
from cupy._core.raw import RawKernel
from cupy._core.raw import RawModule
