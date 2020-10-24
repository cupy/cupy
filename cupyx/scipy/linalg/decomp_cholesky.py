import numpy
from numpy import linalg

import cupy
from cupy import core
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy.cuda import device
from cupy.linalg import decomposition
from cupy.linalg import util
from cupy.cublas import batched_gesv, get_batched_gesv_limit


__all__ = ['cholesky', 'cho_factor', 'cho_solve']

def _cholesky(a, lower=False, overwrite_a=False, clean=True,
              check_finite=True):
    """Common code for cholesky() and cho_factor()."""

    util._assert_cupy_array(a)
    util._assert_nd_squareness(a)
    a = cupy.atleast_2d(a)

    if a.ndim > 2:
        return _potrf_batched(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd':
        dtype = a.dtype.char
    else:
        dtype = numpy.promote_types(a.dtype.char, 'f').char

    x = a.astype(dtype, order='C', copy=True)
    n = len(a)
    handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)

    if dtype == 'f':
        potrf = cusolver.spotrf
        potrf_bufferSize = cusolver.spotrf_bufferSize
    elif dtype == 'd':
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    elif dtype == 'F':
        potrf = cusolver.cpotrf
        potrf_bufferSize = cusolver.cpotrf_bufferSize
    else:  # dtype == 'D':
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize

    buffersize = potrf_bufferSize(
        handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=dtype)
    potrf(
        handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
        workspace.data.ptr, buffersize, dev_info.data.ptr)
    cupy.linalg.util._check_cusolver_dev_info_if_synchronization_allowed(
        potrf, dev_info)

    util._tril(x, k=0)
    return x
