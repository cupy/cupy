from cupy.core import core  # NOQA
from cupy.core import internal  # NOQA


# import class and function
from cupy.core._kernel import create_ufunc  # NOQA
from cupy.core._kernel import ElementwiseKernel  # NOQA
from cupy.core._kernel import ufunc  # NOQA
from cupy.core._reduction import create_reduction_func  # NOQA
from cupy.core._reduction import ReductionKernel  # NOQA
from cupy.core._routines_manipulation import array_split  # NOQA
from cupy.core._routines_manipulation import broadcast  # NOQA
from cupy.core._routines_manipulation import broadcast_to  # NOQA
from cupy.core._routines_manipulation import concatenate_method  # NOQA
from cupy.core._routines_manipulation import moveaxis  # NOQA
from cupy.core._routines_manipulation import rollaxis  # NOQA
from cupy.core._routines_manipulation import size  # NOQA'
from cupy.core._routines_math import absolute  # NOQA
from cupy.core._routines_math import add  # NOQA
from cupy.core._routines_math import angle  # NOQA
from cupy.core._routines_math import conjugate  # NOQA
from cupy.core._routines_math import divide  # NOQA
from cupy.core._routines_math import floor_divide  # NOQA
from cupy.core._routines_math import imag  # NOQA
from cupy.core._routines_math import multiply  # NOQA
from cupy.core._routines_math import negative  # NOQA
from cupy.core._routines_math import power  # NOQA
from cupy.core._routines_math import real  # NOQA
from cupy.core._routines_math import remainder  # NOQA
from cupy.core._routines_math import sqrt  # NOQA
from cupy.core._routines_math import subtract  # NOQA
from cupy.core._routines_math import true_divide  # NOQA
from cupy.core._routines_statistics import nanmax  # NOQA
from cupy.core._routines_statistics import nanmin  # NOQA
from cupy.core.core import _internal_ascontiguousarray  # NOQA
from cupy.core.core import _internal_asfortranarray  # NOQA
from cupy.core.core import array  # NOQA
from cupy.core.core import ascontiguousarray  # NOQA
from cupy.core.core import asfortranarray  # NOQA
from cupy.core.core import bitwise_and  # NOQA
from cupy.core.core import bitwise_or  # NOQA
from cupy.core.core import bitwise_xor  # NOQA
from cupy.core.core import create_comparison  # NOQA
from cupy.core.core import divmod  # NOQA
from cupy.core.core import dot  # NOQA
from cupy.core.core import elementwise_copy  # NOQA
from cupy.core.core import elementwise_copy_where  # NOQA
from cupy.core.core import equal  # NOQA
from cupy.core.core import greater  # NOQA
from cupy.core.core import greater_equal  # NOQA
from cupy.core.core import invert  # NOQA
from cupy.core.core import left_shift  # NOQA
from cupy.core.core import less  # NOQA
from cupy.core.core import less_equal  # NOQA
from cupy.core.core import matmul  # NOQA
from cupy.core.core import ndarray  # NOQA
from cupy.core.core import not_equal  # NOQA
from cupy.core.core import right_shift  # NOQA
from cupy.core.core import tensordot_core  # NOQA
from cupy.core.dlpack import fromDlpack  # NOQA
from cupy.core.internal import complete_slice  # NOQA
from cupy.core.internal import get_size  # NOQA
from cupy.core.raw import RawKernel  # NOQA
from cupy.core.raw import RawModule  # NOQA


# Whether to use reduction kernels based on cub::BlockReduce
import os
cub_block_reduction_enabled = False
if int(os.getenv('CUPY_CUB_BLOCK_REDUCTION_DISABLED', 1)) == 0:
    cub_block_reduction_enabled = True
del os
