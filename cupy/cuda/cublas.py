"""Thin wrapper of CUBLAS."""
import ctypes
import sys

from cupy.cuda import driver
from cupy.cuda import internal


if 'win32' == sys.platform:
    _cublas = internal.load_library(
        internal.get_windows_cuda_library_names('cublas'))
else:
    _cublas = internal.load_library('cublas')


_I = ctypes.c_int
_P = ctypes.c_void_p
_F = ctypes.c_float
_D = ctypes.c_double
_IP = ctypes.POINTER(_I)
_FP = ctypes.POINTER(_F)
_DP = ctypes.POINTER(_D)

Handle = _P

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

CUBLAS_POINTER_MODE_HOST = 0
CUBLAS_POINTER_MODE_DEVICE = 1

###############################################################################
# Error handling
###############################################################################

STATUS = {
    0: 'CUBLAS_STATUS_SUCCESS',
    1: 'CUBLAS_STATUS_NOT_INITIALIZED',
    3: 'CUBLAS_STATUS_ALLOC_FAILED',
    7: 'CUBLAS_STATUS_INVALID_VALUE',
    8: 'CUBLAS_STATUS_ARCH_MISMATCH',
    11: 'CUBLAS_STATUS_MAPPING_ERROR',
    13: 'CUBLAS_STATUS_EXECUTION_FAILED',
    14: 'CUBLAS_STATUS_INTERNAL_ERROR',
    15: 'CUBLAS_STATUS_NOT_SUPPORTED',
    16: 'CUBLAS_STATUS_LICENSE_ERROR',
}


class CUBLASError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CUBLASError, self).__init__(STATUS[status])


def check_status(status):
    if status != 0:
        raise CUBLASError(status)


###############################################################################
# Context
###############################################################################

_cublas.cublasCreate_v2.argtypes = (_P,)


def create():
    handle = Handle()
    status = _cublas.cublasCreate_v2(ctypes.byref(handle))
    check_status(status)
    return handle


_cublas.cublasDestroy_v2.argtypes = (Handle,)


def destroy(handle):
    status = _cublas.cublasDestroy_v2(handle)
    check_status(status)


_cublas.cublasGetVersion_v2.argtypes = (Handle, _IP)


def getVersion(handle):
    version = ctypes.c_int()
    status = _cublas.cublasGetVersion_v2(handle, ctypes.byref(version))
    check_status(status)
    return version


_cublas.cublasGetPointerMode_v2.argtypes = (Handle, _IP)


def getPointerMode(handle):
    mode = ctypes.c_int()
    status = _cublas.cublasGetPointerMode_v2(handle, ctypes.byref(mode))
    check_status(status)
    return mode.value


_cublas.cublasSetPointerMode_v2.argtypes = (Handle, _I)


def setPointerMode(handle, mode):
    status = _cublas.cublasSetPointerMode_v2(handle, mode)
    check_status(status)


###############################################################################
# Stream
###############################################################################

_cublas.cublasSetStream_v2.argtypes = (Handle, driver.Stream)


def setStream(handle, stream):
    status = _cublas.cublasSetStream_v2(handle, stream)
    check_status(status)


_cublas.cublasGetStream_v2.argtypes = (Handle, _P)


def getStream(handle):
    stream = driver.Stream()
    status = _cublas.cublasGetStream_v2(handle, ctypes.byref(stream))
    check_status(status)
    return stream

###############################################################################
# BLAS Level 1
###############################################################################

_cublas.cublasIsamax_v2.argtypes = (Handle, _I, _P, _I, _IP)


def isamax(handle, n, x, incx):
    result = ctypes.c_int()
    status = _cublas.cublasIsamax_v2(
        handle, n, x, incx, ctypes.byref(result))
    check_status(status)
    return result.value


_cublas.cublasIsamin_v2.argtypes = (Handle, _I, _P, _I, _IP)


def isamin(handle, n, x, incx):
    result = ctypes.c_int()
    status = _cublas.cublasIsamin_v2(
        handle, n, x, incx, ctypes.byref(result))
    check_status(status)
    return result.value


_cublas.cublasSasum_v2.argtypes = (Handle, _I, _P, _I, _FP)


def sasum(handle, n, x, incx):
    result = ctypes.c_float()
    status = _cublas.cublasSasum_v2(
        handle, n, x, incx, ctypes.byref(result))
    check_status(status)
    return result.value


_cublas.cublasSaxpy_v2.argtypes = (Handle, _I, _FP, _P, _I, _P, _I)


def saxpy(handle, n, alpha, x, incx, y, incy):
    status = _cublas.cublasSaxpy_v2(
        handle, n, ctypes.byref(_F(alpha)), x, incx, y, incy)
    check_status(status)


_cublas.cublasDaxpy_v2.argtypes = (Handle, _I, _DP, _P, _I, _P, _I)


def daxpy(handle, n, alpha, x, incx, y, incy):
    status = _cublas.cublasDaxpy_v2(
        handle, n, ctypes.byref(_D(alpha)), x, incx, y, incy)
    check_status(status)


_cublas.cublasSdot_v2.argtypes = (Handle, _I, _P, _I, _P, _I, _P)


def sdot(handle, n, x, incx, y, incy, result):
    status = _cublas.cublasSdot_v2(
        handle, n, x, incx, y, incy, result)
    check_status(status)


_cublas.cublasDdot_v2.argtypes = (Handle, _I, _P, _I, _P, _I, _P)


def ddot(handle, n, x, incx, y, incy, result):
    status = _cublas.cublasDdot_v2(
        handle, n, x, incx, y, incy, result)
    check_status(status)


_cublas.cublasSnrm2_v2.argtypes = (Handle, _I, _P, _I, _FP)


def snrm2(handle, n, x, incx):
    result = ctypes.c_float()
    status = _cublas.cublasSnrm2_v2(
        handle, n, x, incx, ctypes.byref(result))
    check_status(status)
    return result.value


_cublas.cublasSscal_v2.argtypes = (Handle, _I, _FP, _P, _I)


def sscal(handle, n, alpha, x, incx):
    status = _cublas.cublasSscal_v2(
        handle, n, ctypes.byref(_F(alpha)), x, incx)
    check_status(status)

###############################################################################
# BLAS Level 2
###############################################################################

_cublas.cublasSgemv_v2.argtypes = (Handle, _I, _I, _I, _FP, _P, _I, _P, _I,
                                   _FP, _P, _I)


def sgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    status = _cublas.cublasSgemv_v2(
        handle, trans, m, n, ctypes.byref(_F(alpha)),
        A, lda, x, incx, ctypes.byref(_F(beta)), y, incy)
    check_status(status)


_cublas.cublasDgemv_v2.argtypes = (Handle, _I, _I, _I, _DP, _P, _I, _P, _I,
                                   _DP, _P, _I)


def dgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    status = _cublas.cublasDgemv_v2(
        handle, trans, m, n, ctypes.byref(_D(alpha)),
        A, lda, x, incx, ctypes.byref(_D(beta)), y, incy)
    check_status(status)


_cublas.cublasSger_v2.argtypes = (
    Handle, _I, _I, _FP, _P, _I, _P, _I, _P, _I)


def sger(handle, m, n, alpha, x, incx, y, incy, A, lda):
    status = _cublas.cublasSger_v2(
        handle, m, n, ctypes.byref(_F(alpha)), x, incx, y, incy, A, lda)
    check_status(status)

_cublas.cublasDger_v2.argtypes = (
    Handle, _I, _I, _DP, _P, _I, _P, _I, _P, _I)


def dger(handle, m, n, alpha, x, incx, y, incy, A, lda):
    status = _cublas.cublasDger_v2(
        handle, m, n, ctypes.byref(_D(alpha)), x, incx, y, incy, A, lda)
    check_status(status)

###############################################################################
# BLAS Level 3
###############################################################################

_cublas.cublasSgemm_v2.argtypes = (Handle, _I, _I, _I, _I, _I, _FP, _P, _I,
                                   _P, _I, _FP, _P, _I)


def sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
          ldc):
    status = _cublas.cublasSgemm_v2(
        handle, transa, transb, m, n, k, ctypes.byref(_F(alpha)),
        A, lda, B, ldb, ctypes.byref(_F(beta)), C, ldc)
    check_status(status)


_cublas.cublasDgemm_v2.argtypes = (Handle, _I, _I, _I, _I, _I, _DP, _P, _I,
                                   _P, _I, _DP, _P, _I)


def dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C,
          ldc):
    status = _cublas.cublasDgemm_v2(
        handle, transa, transb, m, n, k, ctypes.byref(_D(alpha)),
        A, lda, B, ldb, ctypes.byref(_D(beta)), C, ldc)
    check_status(status)


_cublas.cublasSgemmBatched.argtypes = (
    Handle, _I, _I, _I, _I, _I, _FP, _P, _I, _P, _I, _FP, _P, _I, _I)


def sgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray,
                 ldb, beta, Carray, ldc, batchCount):
    status = _cublas.cublasSgemmBatched(
        handle, transa, transb, m, n, k, ctypes.byref(_F(alpha)),
        Aarray, lda, Barray, ldb, ctypes.byref(_F(beta)),
        Carray, ldc, batchCount)
    check_status(status)

###############################################################################
# BLAS extension
###############################################################################

CUBLAS_SIDE_LEFT = 0
CUBLAS_SIDE_RIGHT = 1

_cublas.cublasSdgmm.argtypes = (Handle, _I, _I, _I, _P, _I, _P, _I, _P, _I)


def sdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc):
    status = _cublas.cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    check_status(status)
