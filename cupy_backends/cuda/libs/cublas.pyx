# distutils: language = c++

"""Thin wrapper of CUBLAS."""

cimport cython  # NOQA

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex':
        float x, y

    ctypedef struct cuDoubleComplex 'cuDoubleComplex':
        double x, y

cdef extern from '../../cupy_blas.h' nogil:
    ctypedef void* Stream 'cudaStream_t'
    ctypedef int DataType 'cudaDataType'

    # Context
    int cublasCreate(Handle* handle)
    int cublasDestroy(Handle handle)
    int cublasGetVersion(Handle handle, int* version)
    int cublasGetPointerMode(Handle handle, PointerMode* mode)
    int cublasSetPointerMode(Handle handle, PointerMode mode)

    # Stream
    int cublasSetStream(Handle handle, Stream streamId)
    int cublasGetStream(Handle handle, Stream* streamId)

    # Math Mode
    int cublasSetMathMode(Handle handle, Math mode)
    int cublasGetMathMode(Handle handle, Math* mode)

    # BLAS Level 1
    int cublasIsamaxBit(Handle handle, intBit n, float* x, intBit incx,
                        intBit* result)
    int cublasIdamaxBit(Handle handle, intBit n, double* x, intBit incx,
                        intBit* result)
    int cublasIcamaxBit(Handle handle, intBit n, cuComplex* x, intBit incx,
                        intBit* result)
    int cublasIzamaxBit(Handle handle, intBit n, cuDoubleComplex* x,
                        intBit incx, intBit* result)
    int cublasIsaminBit(Handle handle, intBit n, float* x, intBit incx,
                        intBit* result)
    int cublasIdaminBit(Handle handle, intBit n, double* x, intBit incx,
                        intBit* result)
    int cublasIcaminBit(Handle handle, intBit n, cuComplex* x, intBit incx,
                        intBit* result)
    int cublasIzaminBit(Handle handle, intBit n, cuDoubleComplex* x,
                        intBit incx, intBit* result)
    int cublasSasumBit(Handle handle, intBit n, float* x, intBit incx,
                       float* result)
    int cublasDasumBit(Handle handle, intBit n, double* x, intBit incx,
                       double* result)
    int cublasScasumBit(Handle handle, intBit n, cuComplex* x, intBit incx,
                        float* result)
    int cublasDzasumBit(Handle handle, intBit n, cuDoubleComplex* x,
                        intBit incx, double* result)
    int cublasSaxpyBit(Handle handle, intBit n, float* alpha, float* x,
                       intBit incx, float* y, intBit incy)
    int cublasDaxpyBit(Handle handle, intBit n, double* alpha, double* x,
                       intBit incx, double* y, intBit incy)
    int cublasCaxpyBit(Handle handle, intBit n, cuComplex* alpha,
                       cuComplex* x, intBit incx, cuComplex* y, intBit incy)
    int cublasZaxpyBit(Handle handle, intBit n, cuDoubleComplex* alpha,
                       cuDoubleComplex* x, intBit incx, cuDoubleComplex* y,
                       intBit incy)
    int cublasSdotBit(Handle handle, intBit n, float* x, intBit incx, float* y,
                      intBit incy, float* result)
    int cublasDdotBit(Handle handle, intBit n, double* x, intBit incx,
                      double* y, intBit incy, double* result)
    int cublasCdotuBit(Handle handle, intBit n, cuComplex* x, intBit incx,
                       cuComplex* y, intBit incy, cuComplex* result)
    int cublasCdotcBit(Handle handle, intBit n, cuComplex* x, intBit incx,
                       cuComplex* y, intBit incy, cuComplex* result)
    int cublasZdotuBit(Handle handle, intBit n, cuDoubleComplex* x,
                       intBit incx, cuDoubleComplex* y, intBit incy,
                       cuDoubleComplex* result)
    int cublasZdotcBit(Handle handle, intBit n, cuDoubleComplex* x,
                       intBit incx, cuDoubleComplex* y, intBit incy,
                       cuDoubleComplex* result)
    int cublasSnrm2Bit(Handle handle, intBit n, float* x, intBit incx,
                       float* result)
    int cublasDnrm2Bit(Handle handle, intBit n, double* x, intBit incx,
                       double* result)
    int cublasScnrm2Bit(Handle handle, intBit n, cuComplex* x, intBit incx,
                        float* result)
    int cublasDznrm2Bit(Handle handle, intBit n, cuDoubleComplex* x,
                        intBit incx, double* result)
    int cublasSscalBit(Handle handle, intBit n, float* alpha, float* x,
                       intBit incx)
    int cublasDscalBit(Handle handle, intBit n, double* alpha, double* x,
                       intBit incx)
    int cublasCscalBit(Handle handle, intBit n, cuComplex* alpha, cuComplex* x,
                       intBit incx)
    int cublasCsscalBit(Handle handle, intBit n, float* alpha, cuComplex* x,
                        intBit incx)
    int cublasZscalBit(Handle handle, intBit n, cuDoubleComplex* alpha,
                       cuDoubleComplex* x, intBit incx)
    int cublasZdscalBit(Handle handle, intBit n, double* alpha,
                        cuDoubleComplex* x, intBit incx)

    # BLAS Level 2
    int cublasSgemvBit(
        Handle handle, Operation trans, intBit m, intBit n, float* alpha,
        float* A, intBit lda, float* x, intBit incx, float* beta, float* y,
        intBit incy)
    int cublasDgemvBit(
        Handle handle, Operation trans, intBit m, intBit n, double* alpha,
        double* A, intBit lda, double* x, intBit incx, double* beta, double* y,
        intBit incy)
    int cublasCgemvBit(
        Handle handle, Operation trans, intBit m, intBit n, cuComplex* alpha,
        cuComplex* A, intBit lda, cuComplex* x, intBit incx, cuComplex* beta,
        cuComplex* y, intBit incy)
    int cublasZgemvBit(
        Handle handle, Operation trans, intBit m, intBit n,
        cuDoubleComplex* alpha, cuDoubleComplex* A, intBit lda,
        cuDoubleComplex* x, intBit incx, cuDoubleComplex* beta,
        cuDoubleComplex* y, intBit incy)
    int cublasSgerBit(
        Handle handle, intBit m, intBit n, float* alpha, float* x, intBit incx,
        float* y, intBit incy, float* A, intBit lda)
    int cublasDgerBit(
        Handle handle, intBit m, intBit n, double* alpha, double* x,
        intBit incx, double* y, intBit incy, double* A, intBit lda)
    int cublasCgeruBit(
        Handle handle, intBit m, intBit n, cuComplex* alpha, cuComplex* x,
        intBit incx, cuComplex* y, intBit incy, cuComplex* A, intBit lda)
    int cublasCgercBit(
        Handle handle, intBit m, intBit n, cuComplex* alpha, cuComplex* x,
        intBit incx, cuComplex* y, intBit incy, cuComplex* A, intBit lda)
    int cublasZgeruBit(
        Handle handle, intBit m, intBit n, cuDoubleComplex* alpha,
        cuDoubleComplex* x, intBit incx, cuDoubleComplex* y, intBit incy,
        cuDoubleComplex* A, intBit lda)
    int cublasZgercBit(
        Handle handle, intBit m, intBit n, cuDoubleComplex* alpha,
        cuDoubleComplex* x, intBit incx, cuDoubleComplex* y, intBit incy,
        cuDoubleComplex* A, intBit lda)
    int cublasSsbmvBit(
        Handle handle, FillMode uplo, intBit n, intBit k, const float* alpha,
        const float* A, intBit lda, const float* x, intBit incx,
        const float* beta, float* y, intBit incy)
    int cublasDsbmvBit(
        Handle handle, FillMode uplo, intBit n, intBit k, const double* alpha,
        const double* A, intBit lda, const double* x, intBit incx,
        const double* beta, double* y, intBit incy)

    # BLAS Level 3
    int cublasSgemmBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, float* alpha,
        float* A, intBit lda, float* B, intBit ldb,
        float* beta, float* C, intBit ldc)
    int cublasDgemmBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, double* alpha,
        double* A, intBit lda, double* B, intBit ldb,
        double* beta, double* C, intBit ldc)
    int cublasCgemmBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, cuComplex* alpha,
        cuComplex* A, intBit lda, cuComplex* B, intBit ldb,
        cuComplex* beta, cuComplex* C, intBit ldc)
    int cublasZgemmBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, cuDoubleComplex* alpha,
        cuDoubleComplex* A, intBit lda, cuDoubleComplex* B, intBit ldb,
        cuDoubleComplex* beta, cuDoubleComplex* C, intBit ldc)
    int cublasSgemmBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const float* alpha,
        const float** Aarray, intBit lda, const float** Barray, intBit ldb,
        const float* beta, float** Carray, intBit ldc, intBit batchCount)
    int cublasDgemmBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const double* alpha,
        const double** Aarray, intBit lda, const double** Barray, intBit ldb,
        const double* beta, double** Carray, intBit ldc, intBit batchCount)
    int cublasCgemmBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const cuComplex* alpha,
        const cuComplex** Aarray, intBit lda,
        const cuComplex** Barray, intBit ldb,
        const cuComplex* beta, cuComplex** Carray, intBit ldc,
        intBit batchCount)
    int cublasZgemmBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const cuDoubleComplex* alpha,
        const cuDoubleComplex** Aarray, intBit lda,
        const cuDoubleComplex** Barray, intBit ldb,
        const cuDoubleComplex* beta, cuDoubleComplex** Carray, intBit ldc,
        intBit batchCount)
    int cublasSgemmStridedBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const float* alpha,
        const float* A, intBit lda, long long strideA,
        const float* B, intBit ldb, long long strideB,
        const float* beta, float* C, intBit ldc, long long strideC,
        intBit batchCount)
    int cublasDgemmStridedBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const double* alpha,
        const double* A, intBit lda, long long strideA,
        const double* B, intBit ldb, long long strideB,
        const double* beta, double* C, intBit ldc, long long strideC,
        intBit batchCount)
    int cublasCgemmStridedBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const cuComplex* alpha,
        const cuComplex* A, intBit lda, long long strideA,
        const cuComplex* B, intBit ldb, long long strideB,
        const cuComplex* beta, cuComplex* C, intBit ldc, long long strideC,
        intBit batchCount)
    int cublasZgemmStridedBatchedBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k, const cuDoubleComplex* alpha,
        const cuDoubleComplex* A, intBit lda, long long strideA,
        const cuDoubleComplex* B, intBit ldb, long long strideB,
        const cuDoubleComplex* beta,
        cuDoubleComplex* C, intBit ldc, long long strideC, intBit batchCount)
    int cublasStrsmBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const float* alpha,
        const float* A, intBit lda, float* B, intBit ldb)
    int cublasDtrsmBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const double* alpha,
        const double* A, intBit lda, double* B, intBit ldb)
    int cublasCtrsmBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const cuComplex* alpha,
        const cuComplex* A, intBit lda, cuComplex* B, intBit ldb)
    int cublasZtrsmBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const cuDoubleComplex* alpha,
        const cuDoubleComplex* A, intBit lda, cuDoubleComplex* B, intBit ldb)
    int cublasStrsmBatchedBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const float* alpha,
        const float* const* A, intBit lda, float* const* B, intBit ldb,
        intBit batchCount)
    int cublasDtrsmBatchedBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const double* alpha,
        const double* const* A, intBit lda, double* const* B, intBit ldb,
        intBit batchCount)
    int cublasCtrsmBatchedBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const cuComplex* alpha,
        const cuComplex* const* A, intBit lda, cuComplex* const* B,
        intBit ldb, intBit batchCount)
    int cublasZtrsmBatchedBit(
        Handle handle, SideMode size, FillMode uplo, Operation trans,
        DiagType diag, intBit m, intBit n, const cuDoubleComplex* alpha,
        const cuDoubleComplex* const* A, intBit lda,
        cuDoubleComplex* const* B, intBit ldb, intBit batchCount)
    int cublasSsyrkBit(
        Handle handle, FillMode uplo, Operation trans, intBit n, intBit k,
        float* alpha, float* A, intBit lda,
        float* beta, float* C, intBit ldc)
    int cublasDsyrkBit(
        Handle handle, FillMode uplo, Operation trans, intBit n, intBit k,
        double* alpha, double* A, intBit lda,
        double* beta, double* C, intBit ldc)
    int cublasCsyrkBit(
        Handle handle, FillMode uplo, Operation trans, intBit n, intBit k,
        cuComplex* alpha, cuComplex* A, intBit lda,
        cuComplex* beta, cuComplex* C, intBit ldc)
    int cublasZsyrkBit(
        Handle handle, FillMode uplo, Operation trans, intBit n, intBit k,
        cuDoubleComplex* alpha, cuDoubleComplex* A, intBit lda,
        cuDoubleComplex* beta, cuDoubleComplex* C, intBit ldc)

    # BLAS extension
    int cublasSgeamBit(
        Handle handle, Operation transa, Operation transb, intBit m,
        intBit n, const float* alpha, const float* A, intBit lda,
        const float* beta, const float* B, intBit ldb,
        float* C, intBit ldc)
    int cublasDgeamBit(
        Handle handle, Operation transa, Operation transb, intBit m,
        intBit n, const double* alpha, const double* A, intBit lda,
        const double* beta, const double* B, intBit ldb,
        double* C, intBit ldc)
    int cublasCgeamBit(
        Handle handle, Operation transa, Operation transb, intBit m,
        intBit n, const cuComplex* alpha, const cuComplex* A, intBit lda,
        const cuComplex* beta, const cuComplex* B, intBit ldb,
        cuComplex* C, intBit ldc)
    int cublasZgeamBit(
        Handle handle, Operation transa, Operation transb, intBit m,
        intBit n, const cuDoubleComplex* alpha, const cuDoubleComplex* A,
        intBit lda, const cuDoubleComplex* beta, const cuDoubleComplex* B,
        intBit ldb, cuDoubleComplex* C, intBit ldc)
    int cublasSdgmmBit(
        Handle handle, SideMode mode, intBit m, intBit n, const float* A,
        intBit lda, const float* x, intBit incx, float* C, intBit ldc)
    int cublasDdgmmBit(
        Handle handle, SideMode mode, intBit m, intBit n, const double* A,
        intBit lda, const double* x, intBit incx, double* C, intBit ldc)
    int cublasCdgmmBit(
        Handle handle, SideMode mode, intBit m, intBit n, const cuComplex* A,
        intBit lda, const cuComplex* x, intBit incx, cuComplex* C,
        intBit ldc)
    int cublasZdgmmBit(
        Handle handle, SideMode mode, intBit m, intBit n,
        const cuDoubleComplex* A, intBit lda, const cuDoubleComplex* x,
        intBit incx, cuDoubleComplex* C, intBit ldc)
    int cublasSgemmExBit(
        Handle handle, Operation transa,
        Operation transb, intBit m, intBit n, intBit k,
        const float *alpha, const void *A, DataType Atype,
        intBit lda, const void *B, DataType Btype, intBit ldb,
        const float *beta, void *C, DataType Ctype, intBit ldc)
    int cublasSgetrfBatched(
        Handle handle, int n, float **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int cublasDgetrfBatched(
        Handle handle, int n, double **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int cublasCgetrfBatched(
        Handle handle, int n, cuComplex **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)
    int cublasZgetrfBatched(
        Handle handle, int n, cuDoubleComplex **Aarray, int lda,
        int *PivotArray, int *infoArray, int batchSize)

    int cublasSgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const float **Aarray, int lda, const int *devIpiv,
        float **Barray, int ldb, int *info, int batchSize)
    int cublasDgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const double **Aarray, int lda, const int *devIpiv,
        double **Barray, int ldb, int *info, int batchSize)
    int cublasCgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const cuComplex **Aarray, int lda, const int *devIpiv,
        cuComplex **Barray, int ldb, int *info, int batchSize)
    int cublasZgetrsBatched(
        Handle handle, Operation trans, int n, int nrhs,
        const cuDoubleComplex **Aarray, int lda, const int *devIpiv,
        cuDoubleComplex **Barray, int ldb, int *info, int batchSize)

    int cublasSgetriBatched(
        Handle handle, int n, const float **Aarray, int lda,
        int *PivotArray, float *Carray[], int ldc, int *infoArray,
        int batchSize)
    int cublasDgetriBatched(
        Handle handle, int n, const double **Aarray, int lda,
        int *PivotArray, double *Carray[], int ldc, int *infoArray,
        int batchSize)
    int cublasCgetriBatched(
        Handle handle, int n, const cuComplex **Aarray, int lda,
        int *PivotArray, cuComplex *Carray[], int ldc, int *infoArray,
        int batchSize)
    int cublasZgetriBatched(
        Handle handle, int n, const cuDoubleComplex **Aarray, int lda,
        int *PivotArray, cuDoubleComplex *Carray[], int ldc, int *infoArray,
        int batchSize)
    int cublasGemmExBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k,
        const void *alpha,
        const void *A, DataType Atype, intBit lda,
        const void *B, DataType Btype, intBit ldb,
        const void *beta,
        void *C, DataType Ctype, intBit ldc,
        DataType computetype, GemmAlgo algo)
    int cublasGemmExBit_v11(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k,
        const void *alpha,
        const void *A, DataType Atype, intBit lda,
        const void *B, DataType Btype, intBit ldb,
        const void *beta,
        void *C, DataType Ctype, intBit ldc,
        ComputeType computetype, GemmAlgo algo)
    int cublasGemmStridedBatchedExBit(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k,
        const void *alpha,
        const void *A, DataType Atype, intBit lda, long long strideA,
        const void *B, DataType Btype, intBit ldb, long long strideB,
        const void *beta,
        void *C, DataType Ctype, intBit ldc, long long strideC,
        intBit batchCount, DataType computetype, GemmAlgo algo)
    int cublasGemmStridedBatchedExBit_v11(
        Handle handle, Operation transa, Operation transb,
        intBit m, intBit n, intBit k,
        const void *alpha,
        const void *A, DataType Atype, intBit lda, long long strideA,
        const void *B, DataType Btype, intBit ldb, long long strideB,
        const void *beta,
        void *C, DataType Ctype, intBit ldc, long long strideC,
        intBit batchCount, ComputeType computetype, GemmAlgo algo)
    int cublasStpttr(
        Handle handle, FillMode uplo, int n, const float *AP, float *A,
        int lda)
    int cublasDtpttr(
        Handle handle, FillMode uplo, int n, const double *AP, double *A,
        int lda)
    int cublasStrttp(
        Handle handle, FillMode uplo, int n, const float *A, int lda,
        float *AP)
    int cublasDtrttp(
        Handle handle, FillMode uplo, int n, const double *A, int lda,
        double *AP)


###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
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


cdef dict HIP_STATUS = {
    0: 'HIPBLAS_STATUS_SUCCESS',
    1: 'HIPBLAS_STATUS_NOT_INITIALIZED',
    2: 'HIPBLAS_STATUS_ALLOC_FAILED',
    3: 'HIPBLAS_STATUS_INVALID_VALUE',
    4: 'HIPBLAS_STATUS_MAPPING_ERROR',
    5: 'HIPBLAS_STATUS_EXECUTION_FAILED',
    6: 'HIPBLAS_STATUS_INTERNAL_ERROR',
    7: 'HIPBLAS_STATUS_NOT_SUPPORTED',
    8: 'HIPBLAS_STATUS_ARCH_MISMATCH',
    9: 'HIPBLAS_STATUS_HANDLE_IS_NULLPTR',
}


class CUBLASError(RuntimeError):

    def __init__(self, status):
        self.status = status
        cdef str err
        if runtime._is_hip_environment:
            err = HIP_STATUS[status]
        else:
            err = STATUS[status]
        super(CUBLASError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUBLASError(status)


###############################################################################
# Context
###############################################################################

cpdef intptr_t create() except? 0:
    cdef Handle handle
    with nogil:
        status = cublasCreate(&handle)
    check_status(status)
    return <intptr_t>handle


cpdef destroy(intptr_t handle):
    with nogil:
        status = cublasDestroy(<Handle>handle)
    check_status(status)


cpdef int getVersion(intptr_t handle) except? -1:
    cdef int version
    with nogil:
        status = cublasGetVersion(<Handle>handle, &version)
    check_status(status)
    return version


cpdef int getPointerMode(intptr_t handle) except? -1:
    cdef PointerMode mode
    with nogil:
        status = cublasGetPointerMode(<Handle>handle, &mode)
    check_status(status)
    return mode


cpdef setPointerMode(intptr_t handle, int mode):
    with nogil:
        status = cublasSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)


###############################################################################
# Stream
###############################################################################

cpdef setStream(intptr_t handle, size_t stream):
    # TODO(leofang): It seems most of cuBLAS APIs support stream capture (as of
    # CUDA 11.5) under certain conditions, see
    # https://docs.nvidia.com/cuda/cublas/index.html#CUDA-graphs
    # Before we come up with a robust strategy to test the support conditions,
    # we disable this functionality.
    if not runtime._is_hip_environment and runtime.streamIsCapturing(stream):
        raise NotImplementedError(
            'calling cuBLAS API during stream capture is currently '
            'unsupported')

    with nogil:
        status = cublasSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? 0:
    cdef Stream stream
    with nogil:
        status = cublasGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream


cdef _setStream(intptr_t handle):
    """Set current stream"""
    setStream(handle, stream_module.get_current_stream_ptr())

###############################################################################
# Math Mode
###############################################################################

cpdef setMathMode(intptr_t handle, int mode):
    with nogil:
        status = cublasSetMathMode(<Handle>handle, <Math>mode)
    check_status(status)


cpdef int getMathMode(intptr_t handle) except? -1:
    cdef Math mode
    with nogil:
        status = cublasGetMathMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


###############################################################################
# BLAS Level 1
###############################################################################

cpdef isamax(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIsamaxBit(
            <Handle>handle, n, <float*>x, incx, <intBit*>result)
    check_status(status)

cpdef idamax(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIdamaxBit(
            <Handle>handle, n, <double*>x, incx, <intBit*>result)
    check_status(status)

cpdef icamax(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIcamaxBit(
            <Handle>handle, n, <cuComplex*>x, incx, <intBit*>result)
    check_status(status)

cpdef izamax(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIzamaxBit(
            <Handle>handle, n, <cuDoubleComplex*>x, incx, <intBit*>result)
    check_status(status)


cpdef isamin(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIsaminBit(
            <Handle>handle, n, <float*>x, incx, <intBit*>result)
    check_status(status)

cpdef idamin(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIdaminBit(
            <Handle>handle, n, <double*>x, incx, <intBit*>result)
    check_status(status)

cpdef icamin(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIcaminBit(
            <Handle>handle, n, <cuComplex*>x, incx, <intBit*>result)
    check_status(status)

cpdef izamin(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasIzaminBit(
            <Handle>handle, n, <cuDoubleComplex*>x, incx, <intBit*>result)
    check_status(status)


cpdef sasum(intptr_t handle, intBit n, size_t x, intBit incx,
            size_t result):
    _setStream(handle)
    with nogil:
        status = cublasSasumBit(
            <Handle>handle, n, <float*>x, incx, <float*>result)
    check_status(status)

cpdef dasum(intptr_t handle, intBit n, size_t x, intBit incx,
            size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDasumBit(
            <Handle>handle, n, <double*>x, incx, <double*>result)
    check_status(status)

cpdef scasum(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasScasumBit(
            <Handle>handle, n, <cuComplex*>x, incx, <float*>result)
    check_status(status)

cpdef dzasum(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDzasumBit(
            <Handle>handle, n, <cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef saxpy(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasSaxpyBit(
            <Handle>handle, n, <float*>alpha, <float*>x, incx, <float*>y, incy)
    check_status(status)

cpdef daxpy(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasDaxpyBit(
            <Handle>handle, n, <double*>alpha, <double*>x, incx, <double*>y,
            incy)
    check_status(status)

cpdef caxpy(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasCaxpyBit(
            <Handle>handle, n, <cuComplex*>alpha, <cuComplex*>x, incx,
            <cuComplex*>y, incy)
    check_status(status)

cpdef zaxpy(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasZaxpyBit(
            <Handle>handle, n, <cuDoubleComplex*>alpha, <cuDoubleComplex*>x,
            incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sdot(intptr_t handle, intBit n, size_t x, intBit incx, size_t y,
           intBit incy, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasSdotBit(
            <Handle>handle, n, <float*>x, incx, <float*>y, incy,
            <float*>result)
    check_status(status)

cpdef ddot(intptr_t handle, intBit n, size_t x, intBit incx, size_t y,
           intBit incy, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDdotBit(
            <Handle>handle, n, <double*>x, incx, <double*>y, incy,
            <double*>result)
    check_status(status)

cpdef cdotu(intptr_t handle, intBit n, size_t x, intBit incx, size_t y,
            intBit incy, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasCdotuBit(
            <Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy,
            <cuComplex*>result)
    check_status(status)

cpdef cdotc(intptr_t handle, intBit n, size_t x, intBit incx, size_t y,
            intBit incy, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasCdotcBit(
            <Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy,
            <cuComplex*>result)
    check_status(status)

cpdef zdotu(intptr_t handle, intBit n, size_t x, intBit incx, size_t y,
            intBit incy, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasZdotuBit(
            <Handle>handle, n, <cuDoubleComplex*>x, incx,
            <cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)

cpdef zdotc(intptr_t handle, intBit n, size_t x, intBit incx, size_t y,
            intBit incy, size_t result):
    with nogil:
        status = cublasZdotcBit(
            <Handle>handle, n, <cuDoubleComplex*>x, incx,
            <cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef snrm2(intptr_t handle, intBit n, size_t x, intBit incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasSnrm2Bit(<Handle>handle, n, <float*>x, incx,
                                <float*>result)
    check_status(status)

cpdef dnrm2(intptr_t handle, intBit n, size_t x, intBit incx, size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDnrm2Bit(<Handle>handle, n, <double*>x, incx,
                                <double*>result)
    check_status(status)

cpdef scnrm2(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasScnrm2Bit(<Handle>handle, n, <cuComplex*>x, incx,
                                 <float*>result)
    check_status(status)

cpdef dznrm2(intptr_t handle, intBit n, size_t x, intBit incx,
             size_t result):
    _setStream(handle)
    with nogil:
        status = cublasDznrm2Bit(<Handle>handle, n, <cuDoubleComplex*>x, incx,
                                 <double*>result)
    check_status(status)


cpdef sscal(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx):
    _setStream(handle)
    with nogil:
        status = cublasSscalBit(<Handle>handle, n, <float*>alpha,
                                <float*>x, incx)
    check_status(status)

cpdef dscal(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx):
    _setStream(handle)
    with nogil:
        status = cublasDscalBit(<Handle>handle, n, <double*>alpha,
                                <double*>x, incx)
    check_status(status)

cpdef cscal(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx):
    _setStream(handle)
    with nogil:
        status = cublasCscalBit(<Handle>handle, n, <cuComplex*>alpha,
                                <cuComplex*>x, incx)
    check_status(status)

cpdef csscal(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx):
    _setStream(handle)
    with nogil:
        status = cublasCsscalBit(<Handle>handle, n, <float*>alpha,
                                 <cuComplex*>x, incx)
    check_status(status)

cpdef zscal(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx):
    _setStream(handle)
    with nogil:
        status = cublasZscalBit(<Handle>handle, n, <cuDoubleComplex*>alpha,
                                <cuDoubleComplex*>x, incx)
    check_status(status)

cpdef zdscal(intptr_t handle, intBit n, size_t alpha, size_t x, intBit incx):
    _setStream(handle)
    with nogil:
        status = cublasZdscalBit(<Handle>handle, n, <double*>alpha,
                                 <cuDoubleComplex*>x, incx)
    check_status(status)


###############################################################################
# BLAS Level 2
###############################################################################

cpdef sgemv(intptr_t handle, int trans, intBit m, intBit n, size_t alpha,
            size_t A, intBit lda, size_t x, intBit incx, size_t beta,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasSgemvBit(
            <Handle>handle, <Operation>trans, m, n, <float*>alpha,
            <float*>A, lda, <float*>x, incx, <float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgemv(intptr_t handle, int trans, intBit m, intBit n, size_t alpha,
            size_t A, intBit lda, size_t x, intBit incx, size_t beta,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasDgemvBit(
            <Handle>handle, <Operation>trans, m, n, <double*>alpha,
            <double*>A, lda, <double*>x, incx, <double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgemv(intptr_t handle, int trans, intBit m, intBit n, size_t alpha,
            size_t A, intBit lda, size_t x, intBit incx, size_t beta,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasCgemvBit(
            <Handle>handle, <Operation>trans, m, n, <cuComplex*>alpha,
            <cuComplex*>A, lda, <cuComplex*>x, incx, <cuComplex*>beta,
            <cuComplex*>y, incy)
    check_status(status)


cpdef zgemv(intptr_t handle, int trans, intBit m, intBit n, size_t alpha,
            size_t A, intBit lda, size_t x, intBit incx, size_t beta,
            size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasZgemvBit(
            <Handle>handle, <Operation>trans, m, n, <cuDoubleComplex*>alpha,
            <cuDoubleComplex*>A, lda, <cuDoubleComplex*>x, incx,
            <cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sger(intptr_t handle, intBit m, intBit n, size_t alpha, size_t x,
           intBit incx, size_t y, intBit incy, size_t A, intBit lda):
    _setStream(handle)
    with nogil:
        status = cublasSgerBit(
            <Handle>handle, m, n, <float*>alpha, <float*>x, incx, <float*>y,
            incy, <float*>A, lda)
    check_status(status)


cpdef dger(intptr_t handle, intBit m, intBit n, size_t alpha, size_t x,
           intBit incx, size_t y, intBit incy, size_t A, intBit lda):
    _setStream(handle)
    with nogil:
        status = cublasDgerBit(
            <Handle>handle, m, n, <double*>alpha, <double*>x, incx, <double*>y,
            incy, <double*>A, lda)
    check_status(status)


cpdef cgeru(intptr_t handle, intBit m, intBit n, size_t alpha, size_t x,
            intBit incx, size_t y, intBit incy, size_t A, intBit lda):
    _setStream(handle)
    with nogil:
        status = cublasCgeruBit(
            <Handle>handle, m, n, <cuComplex*>alpha, <cuComplex*>x, incx,
            <cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef cgerc(intptr_t handle, intBit m, intBit n, size_t alpha, size_t x,
            intBit incx, size_t y, intBit incy, size_t A, intBit lda):
    _setStream(handle)
    with nogil:
        status = cublasCgercBit(
            <Handle>handle, m, n, <cuComplex*>alpha, <cuComplex*>x, incx,
            <cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef zgeru(intptr_t handle, intBit m, intBit n, size_t alpha, size_t x,
            intBit incx, size_t y, intBit incy, size_t A, intBit lda):
    _setStream(handle)
    with nogil:
        status = cublasZgeruBit(
            <Handle>handle, m, n, <cuDoubleComplex*>alpha,
            <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy,
            <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef zgerc(intptr_t handle, intBit m, intBit n, size_t alpha, size_t x,
            intBit incx, size_t y, intBit incy, size_t A, intBit lda):
    _setStream(handle)
    with nogil:
        status = cublasZgercBit(
            <Handle>handle, m, n, <cuDoubleComplex*>alpha,
            <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy,
            <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef ssbmv(intptr_t handle, int uplo, intBit n, intBit k,
            size_t alpha, size_t A, intBit lda,
            size_t x, intBit incx, size_t beta, size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasSsbmvBit(
            <Handle>handle, <FillMode>uplo, n, k,
            <float*>alpha, <float*>A, lda,
            <float*>x, incx, <float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsbmv(intptr_t handle, int uplo, intBit n, intBit k,
            size_t alpha, size_t A, intBit lda,
            size_t x, intBit incx, size_t beta, size_t y, intBit incy):
    _setStream(handle)
    with nogil:
        status = cublasDsbmvBit(
            <Handle>handle, <FillMode>uplo, n, k,
            <double*>alpha, <double*>A, lda,
            <double*>x, incx, <double*>beta, <double*>y, incy)
    check_status(status)


###############################################################################
# BLAS Level 3
###############################################################################

cpdef sgemm(intptr_t handle, int transa, int transb,
            intBit m, intBit n, intBit k, size_t alpha, size_t A,
            intBit lda, size_t B, intBit ldb, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasSgemmBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <float*>alpha, <float*>A, lda, <float*>B, ldb, <float*>beta,
            <float*>C, ldc)
    check_status(status)


cpdef dgemm(intptr_t handle, int transa, int transb,
            intBit m, intBit n, intBit k, size_t alpha, size_t A,
            intBit lda, size_t B, intBit ldb, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasDgemmBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <double*>alpha, <double*>A, lda, <double*>B, ldb, <double*>beta,
            <double*>C, ldc)
    check_status(status)


cpdef cgemm(intptr_t handle, int transa, int transb,
            intBit m, intBit n, intBit k, size_t alpha, size_t A,
            intBit lda, size_t B, intBit ldb, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasCgemmBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuComplex*>alpha, <cuComplex*>A, lda, <cuComplex*>B, ldb,
            <cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zgemm(intptr_t handle, int transa, int transb,
            intBit m, intBit n, intBit k, size_t alpha, size_t A,
            intBit lda, size_t B, intBit ldb, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasZgemmBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuDoubleComplex*>alpha, <cuDoubleComplex*>A, lda,
            <cuDoubleComplex*>B, ldb, <cuDoubleComplex*>beta,
            <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sgemmBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t Aarray, intBit lda, size_t Barray,
        intBit ldb, size_t beta, size_t Carray, intBit ldc,
        intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasSgemmBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <float*>alpha, <const float**>Aarray, lda, <const float**>Barray,
            ldb, <float*>beta, <float**>Carray, ldc, batchCount)
    check_status(status)


cpdef dgemmBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t Aarray, intBit lda, size_t Barray,
        intBit ldb, size_t beta, size_t Carray, intBit ldc,
        intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasDgemmBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <double*>alpha, <const double**>Aarray, lda,
            <const double**>Barray, ldb, <double*>beta,
            <double**>Carray, ldc, batchCount)
    check_status(status)


cpdef cgemmBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t Aarray, intBit lda, size_t Barray,
        intBit ldb, size_t beta, size_t Carray, intBit ldc,
        intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasCgemmBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuComplex*>alpha, <const cuComplex**>Aarray, lda,
            <const cuComplex**>Barray, ldb, <cuComplex*>beta,
            <cuComplex**>Carray, ldc, batchCount)
    check_status(status)


cpdef zgemmBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t Aarray, intBit lda, size_t Barray,
        intBit ldb, size_t beta, size_t Carray, intBit ldc,
        intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasZgemmBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <cuDoubleComplex*>alpha, <const cuDoubleComplex**>Aarray, lda,
            <const cuDoubleComplex**>Barray, ldb, <cuDoubleComplex*>beta,
            <cuDoubleComplex**>Carray, ldc, batchCount)
    check_status(status)


cpdef sgemmStridedBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t A, intBit lda, long long strideA,
        size_t B, intBit ldb, long long strideB, size_t beta, size_t C,
        intBit ldc, long long strideC, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasSgemmStridedBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const float*>alpha,
            <const float*>A, lda, <long long>strideA,
            <const float*>B, ldb, <long long>strideB,
            <const float*>beta,
            <float*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef dgemmStridedBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t A, intBit lda, long long strideA,
        size_t B, intBit ldb, long long strideB, size_t beta, size_t C,
        intBit ldc, long long strideC, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasDgemmStridedBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const double*>alpha,
            <const double*>A, lda, <long long>strideA,
            <const double*>B, ldb, <long long>strideB,
            <const double*>beta,
            <double*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef cgemmStridedBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t A, intBit lda, long long strideA,
        size_t B, intBit ldb, long long strideB, size_t beta, size_t C,
        intBit ldc, long long strideC, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasCgemmStridedBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const cuComplex*>alpha,
            <const cuComplex*>A, lda, <long long>strideA,
            <const cuComplex*>B, ldb, <long long>strideB,
            <const cuComplex*>beta,
            <cuComplex*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef zgemmStridedBatched(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t A, intBit lda, long long strideA,
        size_t B, intBit ldb, long long strideB, size_t beta, size_t C,
        intBit ldc, long long strideC, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasZgemmStridedBatchedBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const cuDoubleComplex*>alpha,
            <const cuDoubleComplex*>A, lda, <long long>strideA,
            <const cuDoubleComplex*>B, ldb, <long long>strideB,
            <const cuDoubleComplex*>beta,
            <cuDoubleComplex*>C, ldc, <long long>strideC,
            batchCount)
    check_status(status)


cpdef strsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb):
    _setStream(handle)
    with nogil:
        status = cublasStrsmBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const float*>alpha, <const float*>Aarray,
            lda, <float*>Barray, ldb)
    check_status(status)


cpdef dtrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb):
    _setStream(handle)
    with nogil:
        status = cublasDtrsmBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const double*>alpha, <const double*>Aarray,
            lda, <double*>Barray, ldb)
    check_status(status)


cpdef ctrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb):
    _setStream(handle)
    with nogil:
        status = cublasCtrsmBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const cuComplex*>alpha,
            <const cuComplex*>Aarray, lda, <cuComplex*>Barray, ldb)
    check_status(status)


cpdef ztrsm(
        intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb):
    _setStream(handle)
    with nogil:
        status = cublasZtrsmBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const cuDoubleComplex*>alpha,
            <const cuDoubleComplex*>Aarray, lda, <cuDoubleComplex*>Barray, ldb)
    check_status(status)


cpdef strsmBatched(
    intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasStrsmBatchedBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const float*>alpha,
            <const float* const*>Aarray, lda, <float* const*>Barray, ldb,
            batchCount)
    check_status(status)


cpdef dtrsmBatched(
    intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasDtrsmBatchedBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const double*>alpha,
            <const double* const*>Aarray, lda, <double* const*>Barray, ldb,
            batchCount)
    check_status(status)


cpdef ctrsmBatched(
    intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasCtrsmBatchedBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const cuComplex*>alpha,
            <const cuComplex* const*>Aarray, lda, <cuComplex* const*>Barray,
            ldb, batchCount)
    check_status(status)


cpdef ztrsmBatched(
    intptr_t handle, int side, int uplo, int trans, int diag,
        intBit m, intBit n, size_t alpha, size_t Aarray, intBit lda,
        size_t Barray, intBit ldb, intBit batchCount):
    _setStream(handle)
    with nogil:
        status = cublasZtrsmBatchedBit(
            <Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans,
            <DiagType>diag, m, n, <const cuDoubleComplex*>alpha,
            <const cuDoubleComplex* const*>Aarray, lda,
            <cuDoubleComplex* const*>Barray, ldb, batchCount)
    check_status(status)


cpdef ssyrk(intptr_t handle, int uplo, int trans, intBit n, intBit k,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasSsyrkBit(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const float*>alpha, <const float*>A, lda,
            <const float*>beta, <float*>C, ldc)
    check_status(status)


cpdef dsyrk(intptr_t handle, int uplo, int trans, intBit n, intBit k,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasDsyrkBit(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const double*>alpha, <const double*>A, lda,
            <const double*>beta, <double*>C, ldc)
    check_status(status)


cpdef csyrk(intptr_t handle, int uplo, int trans, intBit n, intBit k,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasCsyrkBit(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const cuComplex*>alpha, <const cuComplex*>A, lda,
            <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zsyrk(intptr_t handle, int uplo, int trans, intBit n, intBit k,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t C,
            intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasZsyrkBit(
            <Handle>handle, <FillMode>uplo, <Operation>trans, n, k,
            <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


###############################################################################
# BLAS extension
###############################################################################

cpdef sgeam(intptr_t handle, int transa, int transb, intBit m, intBit n,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t B,
            intBit ldb, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasSgeamBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const float*>alpha, <const float*>A, lda, <const float*>beta,
            <const float*>B, ldb, <float*>C, ldc)
    check_status(status)

cpdef dgeam(intptr_t handle, int transa, int transb, intBit m, intBit n,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t B,
            intBit ldb, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasDgeamBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const double*>alpha, <const double*>A, lda, <const double*>beta,
            <const double*>B, ldb, <double*>C, ldc)
    check_status(status)

cpdef cgeam(intptr_t handle, int transa, int transb, intBit m, intBit n,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t B,
            intBit ldb, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasCgeamBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const cuComplex*>alpha, <const cuComplex*>A, lda,
            <const cuComplex*>beta, <const cuComplex*>B, ldb,
            <cuComplex*>C, ldc)
    check_status(status)

cpdef zgeam(intptr_t handle, int transa, int transb, intBit m, intBit n,
            size_t alpha, size_t A, intBit lda, size_t beta, size_t B,
            intBit ldb, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasZgeamBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n,
            <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda,
            <const cuDoubleComplex*>beta, <const cuDoubleComplex*>B, ldb,
            <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sdgmm(intptr_t handle, int mode, intBit m, intBit n, size_t A,
            intBit lda, size_t x, intBit incx, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasSdgmmBit(
            <Handle>handle, <SideMode>mode, m, n, <const float*>A, lda,
            <const float*>x, incx, <float*>C, ldc)
    check_status(status)

cpdef ddgmm(intptr_t handle, int mode, intBit m, intBit n, size_t A,
            intBit lda, size_t x, intBit incx, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasDdgmmBit(
            <Handle>handle, <SideMode>mode, m, n, <const double*>A, lda,
            <const double*>x, incx, <double*>C, ldc)
    check_status(status)

cpdef cdgmm(intptr_t handle, int mode, intBit m, intBit n, size_t A,
            intBit lda, size_t x, intBit incx, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasCdgmmBit(
            <Handle>handle, <SideMode>mode, m, n, <const cuComplex*>A, lda,
            <const cuComplex*>x, incx, <cuComplex*>C, ldc)
    check_status(status)

cpdef zdgmm(intptr_t handle, int mode, intBit m, intBit n, size_t A,
            intBit lda, size_t x, intBit incx, size_t C, intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasZdgmmBit(
            <Handle>handle, <SideMode>mode, m, n, <const cuDoubleComplex*>A,
            lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sgemmEx(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t A, int Atype, intBit lda, size_t B,
        int Btype, intBit ldb, size_t beta, size_t C, int Ctype,
        intBit ldc):
    _setStream(handle)
    with nogil:
        status = cublasSgemmExBit(
            <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
            <const float*>alpha, <const void*>A, <DataType>Atype, lda,
            <const void*>B, <DataType>Btype, ldb, <const float*>beta,
            <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef sgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasSgetrfBatched(
            <Handle>handle, n, <float**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef dgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasDgetrfBatched(
            <Handle>handle, n, <double**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef cgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasCgetrfBatched(
            <Handle>handle, n, <cuComplex**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef zgetrfBatched(intptr_t handle, int n, size_t Aarray, int lda,
                    size_t PivotArray, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasZgetrfBatched(
            <Handle>handle, n, <cuDoubleComplex**>Aarray, lda,
            <int*>PivotArray, <int*>infoArray, batchSize)
    check_status(status)


cpdef int sgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasSgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const float**>Aarray, lda, <const int*>devIpiv,
            <float**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int dgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasDgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const double**>Aarray, lda, <const int*>devIpiv,
            <double**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int cgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasCgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuComplex**>Aarray, lda, <const int*>devIpiv,
            <cuComplex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)

cpdef int zgetrsBatched(intptr_t handle, int trans, int n, int nrhs,
                        size_t Aarray, int lda, size_t devIpiv,
                        size_t Barray, int ldb, size_t info, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasZgetrsBatched(
            <Handle>handle, <Operation>trans, n, nrhs,
            <const cuDoubleComplex**>Aarray, lda, <const int*>devIpiv,
            <cuDoubleComplex**>Barray, ldb, <int*>info, batchSize)
    check_status(status)


cpdef sgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasSgetriBatched(
            <Handle>handle, n, <const float**>Aarray, lda, <int*>PivotArray,
            <float**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef dgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasDgetriBatched(
            <Handle>handle, n, <const double**>Aarray, lda, <int*>PivotArray,
            <double**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef cgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasCgetriBatched(
            <Handle>handle, n, <const cuComplex**>Aarray, lda,
            <int*>PivotArray,
            <cuComplex**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef zgetriBatched(
        intptr_t handle, int n, size_t Aarray, int lda, size_t PivotArray,
        size_t Carray, int ldc, size_t infoArray, int batchSize):
    _setStream(handle)
    with nogil:
        status = cublasZgetriBatched(
            <Handle>handle, n, <const cuDoubleComplex**>Aarray, lda,
            <int*>PivotArray,
            <cuDoubleComplex**>Carray, ldc, <int*>infoArray, batchSize)
    check_status(status)


cpdef gemmEx(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha, size_t A, int Atype, intBit lda, size_t B,
        int Btype, intBit ldb, size_t beta, size_t C, int Ctype,
        intBit ldc, int computeType, int algo):
    _setStream(handle)
    with nogil:
        if (
            not runtime._is_hip_environment and
            computeType >= CUBLAS_COMPUTE_16F
        ):
            status = cublasGemmExBit_v11(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda,
                <const void*>B, <DataType>Btype, ldb,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc,
                <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = cublasGemmExBit(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda,
                <const void*>B, <DataType>Btype, ldb,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc,
                <ComputeType>computeType, <GemmAlgo>algo)
    check_status(status)


cpdef gemmStridedBatchedEx(
        intptr_t handle, int transa, int transb, intBit m, intBit n,
        intBit k, size_t alpha,
        size_t A, int Atype, intBit lda, long long strideA,
        size_t B, int Btype, intBit ldb, long long strideB,
        size_t beta,
        size_t C, int Ctype, intBit ldc, long long strideC,
        intBit batchCount, int computeType, int algo):
    _setStream(handle)
    with nogil:
        if computeType >= CUBLAS_COMPUTE_16F:
            status = cublasGemmStridedBatchedExBit_v11(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda, <long long>strideA,
                <const void*>B, <DataType>Btype, ldb, <long long>strideB,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc, <long long>strideC,
                batchCount, <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = cublasGemmStridedBatchedExBit(
                <Handle>handle, <Operation>transa, <Operation>transb, m, n, k,
                <const void*>alpha,
                <const void*>A, <DataType>Atype, lda, <long long>strideA,
                <const void*>B, <DataType>Btype, ldb, <long long>strideB,
                <const void*>beta,
                <void*>C, <DataType>Ctype, ldc, <long long>strideC,
                batchCount, <ComputeType>computeType, <GemmAlgo>algo)
    check_status(status)


cpdef stpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasStpttr(<Handle>handle, <FillMode>uplo, n,
                              <const float*>AP, <float*>A, lda)
    check_status(status)


cpdef dtpttr(intptr_t handle, int uplo, int n, size_t AP, size_t A, int lda):
    _setStream(handle)
    with nogil:
        status = cublasDtpttr(<Handle>handle, <FillMode>uplo, n,
                              <const double*>AP, <double*>A, lda)
    check_status(status)


cpdef strttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP):
    _setStream(handle)
    with nogil:
        status = cublasStrttp(<Handle>handle, <FillMode>uplo, n,
                              <const float*>A, lda, <float*>AP)
    check_status(status)


cpdef dtrttp(intptr_t handle, int uplo, int n, size_t A, int lda, size_t AP):
    _setStream(handle)
    with nogil:
        status = cublasDtrttp(<Handle>handle, <FillMode>uplo, n,
                              <const double*>A, lda, <double*>AP)
    check_status(status)
