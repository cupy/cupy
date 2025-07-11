# mypy: ignore-errors
from __future__ import annotations


from cupy._core import core  # NOQA
from cupy._core import fusion  # NOQA
from cupy._core import internal  # NOQA


# internal APIs for testing and development
from cupy._core._accelerator import set_elementwise_accelerators  # NOQA
from cupy._core._accelerator import set_reduction_accelerators  # NOQA
from cupy._core._accelerator import set_routine_accelerators  # NOQA
from cupy._core._accelerator import get_elementwise_accelerators  # NOQA
from cupy._core._accelerator import get_reduction_accelerators  # NOQA
from cupy._core._accelerator import get_routine_accelerators  # NOQA


# import class and function
from cupy._core._kernel import create_ufunc  # NOQA
from cupy._core._kernel import ElementwiseKernel  # NOQA
from cupy._core._kernel import ufunc  # NOQA
from cupy._core._kernel import _get_warpsize  # NOQA
from cupy._core._reduction import create_reduction_func  # NOQA
from cupy._core._reduction import ReductionKernel  # NOQA
from cupy._core._routines_binary import bitwise_and  # NOQA
from cupy._core._routines_binary import bitwise_or  # NOQA
from cupy._core._routines_binary import bitwise_xor  # NOQA
from cupy._core._routines_binary import invert  # NOQA
from cupy._core._routines_binary import left_shift  # NOQA
from cupy._core._routines_binary import right_shift  # NOQA
from cupy._core._routines_linalg import _mat_ptrs  # NOQA
from cupy._core._routines_linalg import dot  # NOQA
from cupy._core._routines_linalg import get_compute_type  # NOQA
from cupy._core._routines_linalg import matmul  # NOQA
from cupy._core._routines_linalg import set_compute_type  # NOQA
from cupy._core._routines_linalg import tensordot_core  # NOQA
from cupy._core._routines_logic import create_comparison  # NOQA
from cupy._core._routines_logic import equal  # NOQA
from cupy._core._routines_logic import greater  # NOQA
from cupy._core._routines_logic import greater_equal  # NOQA
from cupy._core._routines_logic import less  # NOQA
from cupy._core._routines_logic import less_equal  # NOQA
from cupy._core._routines_logic import not_equal  # NOQA
from cupy._core._routines_manipulation import array_split  # NOQA
from cupy._core._routines_manipulation import broadcast  # NOQA
from cupy._core._routines_manipulation import broadcast_to  # NOQA
from cupy._core._routines_manipulation import concatenate_method  # NOQA
from cupy._core._routines_manipulation import moveaxis  # NOQA
from cupy._core._routines_manipulation import rollaxis  # NOQA
from cupy._core._routines_manipulation import size  # NOQA
from cupy._core._routines_math import absolute  # NOQA
from cupy._core._routines_math import add  # NOQA
from cupy._core._routines_math import angle, angle_deg  # NOQA
from cupy._core._routines_math import conjugate  # NOQA
from cupy._core._routines_math import divide  # NOQA
from cupy._core._routines_math import floor_divide  # NOQA
from cupy._core._routines_math import multiply  # NOQA
from cupy._core._routines_math import negative  # NOQA
from cupy._core._routines_math import positive  # NOQA
from cupy._core._routines_math import power  # NOQA
from cupy._core._routines_math import remainder  # NOQA
from cupy._core._routines_math import sqrt  # NOQA
from cupy._core._routines_math import subtract  # NOQA
from cupy._core._routines_math import true_divide  # NOQA
from cupy._core._routines_statistics import nanmax  # NOQA
from cupy._core._routines_statistics import nanmin  # NOQA
from cupy._core.core import _internal_ascontiguousarray  # NOQA
from cupy._core.core import _internal_asfortranarray  # NOQA
from cupy._core.core import array  # NOQA
from cupy._core.core import ascontiguousarray  # NOQA
from cupy._core.core import asfortranarray  # NOQA
from cupy._core.core import divmod  # NOQA
from cupy._core.core import elementwise_copy  # NOQA
from cupy._core.core import ndarray  # NOQA
from cupy._core.dlpack import fromDlpack  # NOQA
from cupy._core.dlpack import from_dlpack  # NOQA
from cupy._core.internal import complete_slice  # NOQA
from cupy._core.internal import get_size  # NOQA
from cupy._core.raw import RawKernel  # NOQA
from cupy._core.raw import RawModule  # NOQA
