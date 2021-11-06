# distutils: language = c++

# This code was automatically generated. Do not modify it directly.

cimport cython  # NOQA

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module


###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_complex.h':
    ctypedef struct cuComplex 'cuComplex'
    ctypedef struct cuDoubleComplex 'cuDoubleComplex'


cdef extern from *:
    ctypedef void* Stream 'cudaStream_t'
    ctypedef int DataType 'cudaDataType'
    ctypedef int LibraryPropertyType 'libraryPropertyType_t'


cdef extern from '../../cupy_blas.h' nogil:
    Status cublasCreate(Handle* handle)
    Status cublasDestroy(Handle handle)
    Status cublasGetVersion(Handle handle, int* version)
    Status cublasGetProperty(LibraryPropertyType type, int* value)
    size_t cublasGetCudartVersion()
    Status cublasSetStream(Handle handle, Stream streamId)
    Status cublasGetStream(Handle handle, Stream* streamId)
    Status cublasGetPointerMode(Handle handle, PointerMode* mode)
    Status cublasSetPointerMode(Handle handle, PointerMode mode)
    Status cublasGetAtomicsMode(Handle handle, AtomicsMode* mode)
    Status cublasSetAtomicsMode(Handle handle, AtomicsMode mode)
    Status cublasGetMathMode(Handle handle, Math* mode)
    Status cublasSetMathMode(Handle handle, Math mode)
    Status cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy)
    Status cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
    Status cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
    Status cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
    Status cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, Stream stream)
    Status cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, Stream stream)
    Status cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, Stream stream)
    Status cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, Stream stream)
    Status cublasNrm2Ex(Handle handle, int n, const void* x, DataType xType, int incx, void* result, DataType resultType, DataType executionType)
    Status cublasSnrm2(Handle handle, int n, const float* x, int incx, float* result)
    Status cublasDnrm2(Handle handle, int n, const double* x, int incx, double* result)
    Status cublasScnrm2(Handle handle, int n, const cuComplex* x, int incx, float* result)
    Status cublasDznrm2(Handle handle, int n, const cuDoubleComplex* x, int incx, double* result)
    Status cublasDotEx(Handle handle, int n, const void* x, DataType xType, int incx, const void* y, DataType yType, int incy, void* result, DataType resultType, DataType executionType)
    Status cublasDotcEx(Handle handle, int n, const void* x, DataType xType, int incx, const void* y, DataType yType, int incy, void* result, DataType resultType, DataType executionType)
    Status cublasSdot(Handle handle, int n, const float* x, int incx, const float* y, int incy, float* result)
    Status cublasDdot(Handle handle, int n, const double* x, int incx, const double* y, int incy, double* result)
    Status cublasCdotu(Handle handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result)
    Status cublasCdotc(Handle handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result)
    Status cublasZdotu(Handle handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result)
    Status cublasZdotc(Handle handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result)
    Status cublasScalEx(Handle handle, int n, const void* alpha, DataType alphaType, void* x, DataType xType, int incx, DataType executionType)
    Status cublasSscal(Handle handle, int n, const float* alpha, float* x, int incx)
    Status cublasDscal(Handle handle, int n, const double* alpha, double* x, int incx)
    Status cublasCscal(Handle handle, int n, const cuComplex* alpha, cuComplex* x, int incx)
    Status cublasCsscal(Handle handle, int n, const float* alpha, cuComplex* x, int incx)
    Status cublasZscal(Handle handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx)
    Status cublasZdscal(Handle handle, int n, const double* alpha, cuDoubleComplex* x, int incx)
    Status cublasAxpyEx(Handle handle, int n, const void* alpha, DataType alphaType, const void* x, DataType xType, int incx, void* y, DataType yType, int incy, DataType executiontype)
    Status cublasSaxpy(Handle handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
    Status cublasDaxpy(Handle handle, int n, const double* alpha, const double* x, int incx, double* y, int incy)
    Status cublasCaxpy(Handle handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy)
    Status cublasZaxpy(Handle handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
    Status cublasCopyEx(Handle handle, int n, const void* x, DataType xType, int incx, void* y, DataType yType, int incy)
    Status cublasScopy(Handle handle, int n, const float* x, int incx, float* y, int incy)
    Status cublasDcopy(Handle handle, int n, const double* x, int incx, double* y, int incy)
    Status cublasCcopy(Handle handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy)
    Status cublasZcopy(Handle handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
    Status cublasSswap(Handle handle, int n, float* x, int incx, float* y, int incy)
    Status cublasDswap(Handle handle, int n, double* x, int incx, double* y, int incy)
    Status cublasCswap(Handle handle, int n, cuComplex* x, int incx, cuComplex* y, int incy)
    Status cublasZswap(Handle handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
    Status cublasSwapEx(Handle handle, int n, void* x, DataType xType, int incx, void* y, DataType yType, int incy)
    Status cublasIsamax(Handle handle, int n, const float* x, int incx, int* result)
    Status cublasIdamax(Handle handle, int n, const double* x, int incx, int* result)
    Status cublasIcamax(Handle handle, int n, const cuComplex* x, int incx, int* result)
    Status cublasIzamax(Handle handle, int n, const cuDoubleComplex* x, int incx, int* result)
    Status cublasIamaxEx(Handle handle, int n, const void* x, DataType xType, int incx, int* result)
    Status cublasIsamin(Handle handle, int n, const float* x, int incx, int* result)
    Status cublasIdamin(Handle handle, int n, const double* x, int incx, int* result)
    Status cublasIcamin(Handle handle, int n, const cuComplex* x, int incx, int* result)
    Status cublasIzamin(Handle handle, int n, const cuDoubleComplex* x, int incx, int* result)
    Status cublasIaminEx(Handle handle, int n, const void* x, DataType xType, int incx, int* result)
    Status cublasAsumEx(Handle handle, int n, const void* x, DataType xType, int incx, void* result, DataType resultType, DataType executiontype)
    Status cublasSasum(Handle handle, int n, const float* x, int incx, float* result)
    Status cublasDasum(Handle handle, int n, const double* x, int incx, double* result)
    Status cublasScasum(Handle handle, int n, const cuComplex* x, int incx, float* result)
    Status cublasDzasum(Handle handle, int n, const cuDoubleComplex* x, int incx, double* result)
    Status cublasSrot(Handle handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s)
    Status cublasDrot(Handle handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s)
    Status cublasCrot(Handle handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s)
    Status cublasCsrot(Handle handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s)
    Status cublasZrot(Handle handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s)
    Status cublasZdrot(Handle handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s)
    Status cublasRotEx(Handle handle, int n, void* x, DataType xType, int incx, void* y, DataType yType, int incy, const void* c, const void* s, DataType csType, DataType executiontype)
    Status cublasSrotg(Handle handle, float* a, float* b, float* c, float* s)
    Status cublasDrotg(Handle handle, double* a, double* b, double* c, double* s)
    Status cublasCrotg(Handle handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s)
    Status cublasZrotg(Handle handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s)
    Status cublasRotgEx(Handle handle, void* a, void* b, DataType abType, void* c, void* s, DataType csType, DataType executiontype)
    Status cublasSrotm(Handle handle, int n, float* x, int incx, float* y, int incy, const float* param)
    Status cublasDrotm(Handle handle, int n, double* x, int incx, double* y, int incy, const double* param)
    Status cublasRotmEx(Handle handle, int n, void* x, DataType xType, int incx, void* y, DataType yType, int incy, const void* param, DataType paramType, DataType executiontype)
    Status cublasSrotmg(Handle handle, float* d1, float* d2, float* x1, const float* y1, float* param)
    Status cublasDrotmg(Handle handle, double* d1, double* d2, double* x1, const double* y1, double* param)
    Status cublasRotmgEx(Handle handle, void* d1, DataType d1Type, void* d2, DataType d2Type, void* x1, DataType x1Type, const void* y1, DataType y1Type, void* param, DataType paramType, DataType executiontype)
    Status cublasSgemv(Handle handle, Operation trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
    Status cublasDgemv(Handle handle, Operation trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy)
    Status cublasCgemv(Handle handle, Operation trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy)
    Status cublasZgemv(Handle handle, Operation trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    Status cublasSgbmv(Handle handle, Operation trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
    Status cublasDgbmv(Handle handle, Operation trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy)
    Status cublasCgbmv(Handle handle, Operation trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy)
    Status cublasZgbmv(Handle handle, Operation trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    Status cublasStrmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const float* A, int lda, float* x, int incx)
    Status cublasDtrmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const double* A, int lda, double* x, int incx)
    Status cublasCtrmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx)
    Status cublasZtrmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx)
    Status cublasStbmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const float* A, int lda, float* x, int incx)
    Status cublasDtbmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const double* A, int lda, double* x, int incx)
    Status cublasCtbmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx)
    Status cublasZtbmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx)
    Status cublasStpmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const float* AP, float* x, int incx)
    Status cublasDtpmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const double* AP, double* x, int incx)
    Status cublasCtpmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuComplex* AP, cuComplex* x, int incx)
    Status cublasZtpmv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx)
    Status cublasStrsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const float* A, int lda, float* x, int incx)
    Status cublasDtrsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const double* A, int lda, double* x, int incx)
    Status cublasCtrsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx)
    Status cublasZtrsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx)
    Status cublasStpsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const float* AP, float* x, int incx)
    Status cublasDtpsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const double* AP, double* x, int incx)
    Status cublasCtpsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuComplex* AP, cuComplex* x, int incx)
    Status cublasZtpsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx)
    Status cublasStbsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const float* A, int lda, float* x, int incx)
    Status cublasDtbsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const double* A, int lda, double* x, int incx)
    Status cublasCtbsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx)
    Status cublasZtbsv(Handle handle, FillMode uplo, Operation trans, DiagType diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx)
    Status cublasSsymv(Handle handle, FillMode uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
    Status cublasDsymv(Handle handle, FillMode uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy)
    Status cublasCsymv(Handle handle, FillMode uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy)
    Status cublasZsymv(Handle handle, FillMode uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    Status cublasChemv(Handle handle, FillMode uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy)
    Status cublasZhemv(Handle handle, FillMode uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    Status cublasSsbmv(Handle handle, FillMode uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
    Status cublasDsbmv(Handle handle, FillMode uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy)
    Status cublasChbmv(Handle handle, FillMode uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy)
    Status cublasZhbmv(Handle handle, FillMode uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    Status cublasSspmv(Handle handle, FillMode uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy)
    Status cublasDspmv(Handle handle, FillMode uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy)
    Status cublasChpmv(Handle handle, FillMode uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy)
    Status cublasZhpmv(Handle handle, FillMode uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
    Status cublasSger(Handle handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda)
    Status cublasDger(Handle handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda)
    Status cublasCgeru(Handle handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda)
    Status cublasCgerc(Handle handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda)
    Status cublasZgeru(Handle handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)
    Status cublasZgerc(Handle handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)
    Status cublasSsyr(Handle handle, FillMode uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda)
    Status cublasDsyr(Handle handle, FillMode uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda)
    Status cublasCsyr(Handle handle, FillMode uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda)
    Status cublasZsyr(Handle handle, FillMode uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda)
    Status cublasCher(Handle handle, FillMode uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda)
    Status cublasZher(Handle handle, FillMode uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda)
    Status cublasSspr(Handle handle, FillMode uplo, int n, const float* alpha, const float* x, int incx, float* AP)
    Status cublasDspr(Handle handle, FillMode uplo, int n, const double* alpha, const double* x, int incx, double* AP)
    Status cublasChpr(Handle handle, FillMode uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP)
    Status cublasZhpr(Handle handle, FillMode uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP)
    Status cublasSsyr2(Handle handle, FillMode uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda)
    Status cublasDsyr2(Handle handle, FillMode uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda)
    Status cublasCsyr2(Handle handle, FillMode uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda)
    Status cublasZsyr2(Handle handle, FillMode uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)
    Status cublasCher2(Handle handle, FillMode uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda)
    Status cublasZher2(Handle handle, FillMode uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)
    Status cublasSspr2(Handle handle, FillMode uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP)
    Status cublasDspr2(Handle handle, FillMode uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP)
    Status cublasChpr2(Handle handle, FillMode uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP)
    Status cublasZhpr2(Handle handle, FillMode uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP)
    Status cublasSgemm(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
    Status cublasDgemm(Handle handle, Operation transa, Operation transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
    Status cublasCgemm(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasCgemm3m(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasZgemm(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasZgemm3m(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasSgemmEx(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const void* A, DataType Atype, int lda, const void* B, DataType Btype, int ldb, const float* beta, void* C, DataType Ctype, int ldc)
    Status cublasCgemmEx(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const void* A, DataType Atype, int lda, const void* B, DataType Btype, int ldb, const cuComplex* beta, void* C, DataType Ctype, int ldc)
    Status cublasSsyrk(Handle handle, FillMode uplo, Operation trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc)
    Status cublasDsyrk(Handle handle, FillMode uplo, Operation trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc)
    Status cublasCsyrk(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasZsyrk(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasCsyrkEx(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuComplex* alpha, const void* A, DataType Atype, int lda, const cuComplex* beta, void* C, DataType Ctype, int ldc)
    Status cublasCsyrk3mEx(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuComplex* alpha, const void* A, DataType Atype, int lda, const cuComplex* beta, void* C, DataType Ctype, int ldc)
    Status cublasCherk(Handle handle, FillMode uplo, Operation trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc)
    Status cublasZherk(Handle handle, FillMode uplo, Operation trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc)
    Status cublasCherkEx(Handle handle, FillMode uplo, Operation trans, int n, int k, const float* alpha, const void* A, DataType Atype, int lda, const float* beta, void* C, DataType Ctype, int ldc)
    Status cublasCherk3mEx(Handle handle, FillMode uplo, Operation trans, int n, int k, const float* alpha, const void* A, DataType Atype, int lda, const float* beta, void* C, DataType Ctype, int ldc)
    Status cublasSsyr2k(Handle handle, FillMode uplo, Operation trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
    Status cublasDsyr2k(Handle handle, FillMode uplo, Operation trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
    Status cublasCsyr2k(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasZsyr2k(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasCher2k(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc)
    Status cublasZher2k(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc)
    Status cublasSsyrkx(Handle handle, FillMode uplo, Operation trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
    Status cublasDsyrkx(Handle handle, FillMode uplo, Operation trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
    Status cublasCsyrkx(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasZsyrkx(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasCherkx(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc)
    Status cublasZherkx(Handle handle, FillMode uplo, Operation trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc)
    Status cublasSsymm(Handle handle, SideMode side, FillMode uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
    Status cublasDsymm(Handle handle, SideMode side, FillMode uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
    Status cublasCsymm(Handle handle, SideMode side, FillMode uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasZsymm(Handle handle, SideMode side, FillMode uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasChemm(Handle handle, SideMode side, FillMode uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc)
    Status cublasZhemm(Handle handle, SideMode side, FillMode uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
    Status cublasStrsm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb)
    Status cublasDtrsm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb)
    Status cublasCtrsm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb)
    Status cublasZtrsm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb)
    Status cublasStrmm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc)
    Status cublasDtrmm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc)
    Status cublasCtrmm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc)
    Status cublasZtrmm(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc)
    Status cublasSgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount)
    Status cublasDgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount)
    Status cublasCgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount)
    Status cublasZgemmBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount)
    Status cublasSgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount)
    Status cublasDgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount)
    Status cublasCgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount)
    Status cublasCgemm3mStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount)
    Status cublasZgemmStridedBatched(Handle handle, Operation transa, Operation transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount)
    Status cublasSgeam(Handle handle, Operation transa, Operation transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc)
    Status cublasDgeam(Handle handle, Operation transa, Operation transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc)
    Status cublasCgeam(Handle handle, Operation transa, Operation transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc)
    Status cublasZgeam(Handle handle, Operation transa, Operation transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc)
    Status cublasSgetrfBatched(Handle handle, int n, float* const A[], int lda, int* P, int* info, int batchSize)
    Status cublasDgetrfBatched(Handle handle, int n, double* const A[], int lda, int* P, int* info, int batchSize)
    Status cublasCgetrfBatched(Handle handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize)
    Status cublasZgetrfBatched(Handle handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize)
    Status cublasSgetriBatched(Handle handle, int n, const float* const A[], int lda, const int* P, float* const C[], int ldc, int* info, int batchSize)
    Status cublasDgetriBatched(Handle handle, int n, const double* const A[], int lda, const int* P, double* const C[], int ldc, int* info, int batchSize)
    Status cublasCgetriBatched(Handle handle, int n, const cuComplex* const A[], int lda, const int* P, cuComplex* const C[], int ldc, int* info, int batchSize)
    Status cublasZgetriBatched(Handle handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize)
    Status cublasSgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize)
    Status cublasDgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize)
    Status cublasCgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize)
    Status cublasZgetrsBatched(Handle handle, Operation trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize)
    Status cublasStrsmBatched(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount)
    Status cublasDtrsmBatched(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount)
    Status cublasCtrsmBatched(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount)
    Status cublasZtrsmBatched(Handle handle, SideMode side, FillMode uplo, Operation trans, DiagType diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount)
    Status cublasSmatinvBatched(Handle handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info, int batchSize)
    Status cublasDmatinvBatched(Handle handle, int n, const double* const A[], int lda, double* const Ainv[], int lda_inv, int* info, int batchSize)
    Status cublasCmatinvBatched(Handle handle, int n, const cuComplex* const A[], int lda, cuComplex* const Ainv[], int lda_inv, int* info, int batchSize)
    Status cublasZmatinvBatched(Handle handle, int n, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const Ainv[], int lda_inv, int* info, int batchSize)
    Status cublasSgeqrfBatched(Handle handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize)
    Status cublasDgeqrfBatched(Handle handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize)
    Status cublasCgeqrfBatched(Handle handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize)
    Status cublasZgeqrfBatched(Handle handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize)
    Status cublasSgelsBatched(Handle handle, Operation trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize)
    Status cublasDgelsBatched(Handle handle, Operation trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize)
    Status cublasCgelsBatched(Handle handle, Operation trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize)
    Status cublasZgelsBatched(Handle handle, Operation trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize)
    Status cublasSdgmm(Handle handle, SideMode mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc)
    Status cublasDdgmm(Handle handle, SideMode mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc)
    Status cublasCdgmm(Handle handle, SideMode mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc)
    Status cublasZdgmm(Handle handle, SideMode mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc)
    Status cublasStpttr(Handle handle, FillMode uplo, int n, const float* AP, float* A, int lda)
    Status cublasDtpttr(Handle handle, FillMode uplo, int n, const double* AP, double* A, int lda)
    Status cublasCtpttr(Handle handle, FillMode uplo, int n, const cuComplex* AP, cuComplex* A, int lda)
    Status cublasZtpttr(Handle handle, FillMode uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda)
    Status cublasStrttp(Handle handle, FillMode uplo, int n, const float* A, int lda, float* AP)
    Status cublasDtrttp(Handle handle, FillMode uplo, int n, const double* A, int lda, double* AP)
    Status cublasCtrttp(Handle handle, FillMode uplo, int n, const cuComplex* A, int lda, cuComplex* AP)
    Status cublasZtrttp(Handle handle, FillMode uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP)
    Status cublasSetWorkspace(Handle handle, void* workspace, size_t workspaceSizeInBytes)

    # Define by hand for backward compatibility
    Status cublasGemmEx(Handle handle, Operation transa, Operation transb, int m, int n, int k, const void* alpha, const void* A, DataType Atype, int lda, const void* B, DataType Btype, int ldb, const void* beta, void* C, DataType Ctype, int ldc, DataType computeType, GemmAlgo algo)
    Status cublasGemmEx_v11(Handle handle, Operation transa, Operation transb, int m, int n, int k, const void* alpha, const void* A, DataType Atype, int lda, const void* B, DataType Btype, int ldb, const void* beta, void* C, DataType Ctype, int ldc, ComputeType computeType, GemmAlgo algo)
    Status cublasGemmBatchedEx(Handle handle, Operation transa, Operation transb, int m, int n, int k, const void* alpha, const void* const Aarray[], DataType Atype, int lda, const void* const Barray[], DataType Btype, int ldb, const void* beta, void* const Carray[], DataType Ctype, int ldc, int batchCount, DataType computeType, GemmAlgo algo)
    Status cublasGemmBatchedEx_v11(Handle handle, Operation transa, Operation transb, int m, int n, int k, const void* alpha, const void* const Aarray[], DataType Atype, int lda, const void* const Barray[], DataType Btype, int ldb, const void* beta, void* const Carray[], DataType Ctype, int ldc, int batchCount, ComputeType computeType, GemmAlgo algo)
    Status cublasGemmStridedBatchedEx(Handle handle, Operation transa, Operation transb, int m, int n, int k, const void* alpha, const void* A, DataType Atype, int lda, long long int strideA, const void* B, DataType Btype, int ldb, long long int strideB, const void* beta, void* C, DataType Ctype, int ldc, long long int strideC, int batchCount, DataType computeType, GemmAlgo algo)
    Status cublasGemmStridedBatchedEx_v11(Handle handle, Operation transa, Operation transb, int m, int n, int k, const void* alpha, const void* A, DataType Atype, int lda, long long int strideA, const void* B, DataType Btype, int ldb, long long int strideB, const void* beta, void* C, DataType Ctype, int ldc, long long int strideC, int batchCount, ComputeType computeType, GemmAlgo algo)


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
# Wrapper functions
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


cpdef int getProperty(int type) except? -1:
    cdef int value
    with nogil:
        status = cublasGetProperty(<LibraryPropertyType>type, &value)
    check_status(status)
    return value


cpdef size_t getCudartVersion():
    return cublasGetCudartVersion()


cpdef setStream(intptr_t handle, size_t streamId):
    with nogil:
        status = cublasSetStream(<Handle>handle, <Stream>streamId)
    check_status(status)


cpdef size_t getStream(intptr_t handle) except? 0:
    cdef Stream streamId
    with nogil:
        status = cublasGetStream(<Handle>handle, &streamId)
    check_status(status)
    return <size_t>streamId


cpdef int getPointerMode(intptr_t handle) except? -1:
    cdef PointerMode mode
    with nogil:
        status = cublasGetPointerMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef setPointerMode(intptr_t handle, int mode):
    with nogil:
        status = cublasSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)


cpdef int getAtomicsMode(intptr_t handle) except? -1:
    cdef AtomicsMode mode
    with nogil:
        status = cublasGetAtomicsMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef setAtomicsMode(intptr_t handle, int mode):
    with nogil:
        status = cublasSetAtomicsMode(<Handle>handle, <AtomicsMode>mode)
    check_status(status)


cpdef int getMathMode(intptr_t handle) except? -1:
    cdef Math mode
    with nogil:
        status = cublasGetMathMode(<Handle>handle, &mode)
    check_status(status)
    return <int>mode


cpdef setMathMode(intptr_t handle, int mode):
    with nogil:
        status = cublasSetMathMode(<Handle>handle, <Math>mode)
    check_status(status)


cpdef setVector(int n, int elemSize, intptr_t x, int incx, intptr_t devicePtr, int incy):
    with nogil:
        status = cublasSetVector(n, elemSize, <const void*>x, incx, <void*>devicePtr, incy)
    check_status(status)


cpdef getVector(int n, int elemSize, intptr_t x, int incx, intptr_t y, int incy):
    with nogil:
        status = cublasGetVector(n, elemSize, <const void*>x, incx, <void*>y, incy)
    check_status(status)


cpdef setMatrix(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb):
    with nogil:
        status = cublasSetMatrix(rows, cols, elemSize, <const void*>A, lda, <void*>B, ldb)
    check_status(status)


cpdef getMatrix(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb):
    with nogil:
        status = cublasGetMatrix(rows, cols, elemSize, <const void*>A, lda, <void*>B, ldb)
    check_status(status)


cpdef setVectorAsync(int n, int elemSize, intptr_t hostPtr, int incx, intptr_t devicePtr, int incy):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status = cublasSetVectorAsync(n, elemSize, <const void*>hostPtr, incx, <void*>devicePtr, incy, <Stream>stream)
    check_status(status)


cpdef getVectorAsync(int n, int elemSize, intptr_t devicePtr, int incx, intptr_t hostPtr, int incy):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status = cublasGetVectorAsync(n, elemSize, <const void*>devicePtr, incx, <void*>hostPtr, incy, <Stream>stream)
    check_status(status)


cpdef setMatrixAsync(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status = cublasSetMatrixAsync(rows, cols, elemSize, <const void*>A, lda, <void*>B, ldb, <Stream>stream)
    check_status(status)


cpdef getMatrixAsync(int rows, int cols, int elemSize, intptr_t A, int lda, intptr_t B, int ldb):
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    with nogil:
        status = cublasGetMatrixAsync(rows, cols, elemSize, <const void*>A, lda, <void*>B, ldb, <Stream>stream)
    check_status(status)


cpdef nrm2Ex(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result, size_t resultType, size_t executionType):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasNrm2Ex(<Handle>handle, n, <const void*>x, <DataType>xType, incx, <void*>result, <DataType>resultType, <DataType>executionType)
    check_status(status)


cpdef snrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSnrm2(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)


cpdef dnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDnrm2(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)


cpdef scnrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasScnrm2(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)


cpdef dznrm2(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDznrm2(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef dotEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t result, size_t resultType, size_t executionType):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDotEx(<Handle>handle, n, <const void*>x, <DataType>xType, incx, <const void*>y, <DataType>yType, incy, <void*>result, <DataType>resultType, <DataType>executionType)
    check_status(status)


cpdef dotcEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t result, size_t resultType, size_t executionType):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDotcEx(<Handle>handle, n, <const void*>x, <DataType>xType, incx, <const void*>y, <DataType>yType, incy, <void*>result, <DataType>resultType, <DataType>executionType)
    check_status(status)


cpdef sdot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSdot(<Handle>handle, n, <const float*>x, incx, <const float*>y, incy, <float*>result)
    check_status(status)


cpdef ddot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDdot(<Handle>handle, n, <const double*>x, incx, <const double*>y, incy, <double*>result)
    check_status(status)


cpdef cdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCdotu(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)


cpdef cdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCdotc(<Handle>handle, n, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>result)
    check_status(status)


cpdef zdotu(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZdotu(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef zdotc(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZdotc(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>result)
    check_status(status)


cpdef scalEx(intptr_t handle, int n, intptr_t alpha, size_t alphaType, intptr_t x, size_t xType, int incx, size_t executionType):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasScalEx(<Handle>handle, n, <const void*>alpha, <DataType>alphaType, <void*>x, <DataType>xType, incx, <DataType>executionType)
    check_status(status)


cpdef sscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSscal(<Handle>handle, n, <const float*>alpha, <float*>x, incx)
    check_status(status)


cpdef dscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDscal(<Handle>handle, n, <const double*>alpha, <double*>x, incx)
    check_status(status)


cpdef cscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCscal(<Handle>handle, n, <const cuComplex*>alpha, <cuComplex*>x, incx)
    check_status(status)


cpdef csscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsscal(<Handle>handle, n, <const float*>alpha, <cuComplex*>x, incx)
    check_status(status)


cpdef zscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZscal(<Handle>handle, n, <const cuDoubleComplex*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef zdscal(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZdscal(<Handle>handle, n, <const double*>alpha, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef axpyEx(intptr_t handle, int n, intptr_t alpha, size_t alphaType, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, size_t executiontype):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasAxpyEx(<Handle>handle, n, <const void*>alpha, <DataType>alphaType, <const void*>x, <DataType>xType, incx, <void*>y, <DataType>yType, incy, <DataType>executiontype)
    check_status(status)


cpdef saxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSaxpy(<Handle>handle, n, <const float*>alpha, <const float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef daxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDaxpy(<Handle>handle, n, <const double*>alpha, <const double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef caxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCaxpy(<Handle>handle, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zaxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZaxpy(<Handle>handle, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef copyEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCopyEx(<Handle>handle, n, <const void*>x, <DataType>xType, incx, <void*>y, <DataType>yType, incy)
    check_status(status)


cpdef scopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasScopy(<Handle>handle, n, <const float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef dcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDcopy(<Handle>handle, n, <const double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef ccopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCcopy(<Handle>handle, n, <const cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zcopy(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZcopy(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSswap(<Handle>handle, n, <float*>x, incx, <float*>y, incy)
    check_status(status)


cpdef dswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDswap(<Handle>handle, n, <double*>x, incx, <double*>y, incy)
    check_status(status)


cpdef cswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCswap(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy)
    check_status(status)


cpdef zswap(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZswap(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef swapEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSwapEx(<Handle>handle, n, <void*>x, <DataType>xType, incx, <void*>y, <DataType>yType, incy)
    check_status(status)


cpdef isamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIsamax(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(status)


cpdef idamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIdamax(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(status)


cpdef icamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIcamax(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(status)


cpdef izamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIzamax(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)


cpdef iamaxEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIamaxEx(<Handle>handle, n, <const void*>x, <DataType>xType, incx, <int*>result)
    check_status(status)


cpdef isamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIsamin(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(status)


cpdef idamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIdamin(<Handle>handle, n, <const double*>x, incx, <int*>result)
    check_status(status)


cpdef icamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIcamin(<Handle>handle, n, <const cuComplex*>x, incx, <int*>result)
    check_status(status)


cpdef izamin(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIzamin(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <int*>result)
    check_status(status)


cpdef iaminEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasIaminEx(<Handle>handle, n, <const void*>x, <DataType>xType, incx, <int*>result)
    check_status(status)


cpdef asumEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t result, size_t resultType, size_t executiontype):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasAsumEx(<Handle>handle, n, <const void*>x, <DataType>xType, incx, <void*>result, <DataType>resultType, <DataType>executiontype)
    check_status(status)


cpdef sasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSasum(<Handle>handle, n, <const float*>x, incx, <float*>result)
    check_status(status)


cpdef dasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDasum(<Handle>handle, n, <const double*>x, incx, <double*>result)
    check_status(status)


cpdef scasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasScasum(<Handle>handle, n, <const cuComplex*>x, incx, <float*>result)
    check_status(status)


cpdef dzasum(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDzasum(<Handle>handle, n, <const cuDoubleComplex*>x, incx, <double*>result)
    check_status(status)


cpdef srot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSrot(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>c, <const float*>s)
    check_status(status)


cpdef drot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDrot(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>c, <const double*>s)
    check_status(status)


cpdef crot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCrot(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const cuComplex*>s)
    check_status(status)


cpdef csrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsrot(<Handle>handle, n, <cuComplex*>x, incx, <cuComplex*>y, incy, <const float*>c, <const float*>s)
    check_status(status)


cpdef zrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZrot(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const cuDoubleComplex*>s)
    check_status(status)


cpdef zdrot(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZdrot(<Handle>handle, n, <cuDoubleComplex*>x, incx, <cuDoubleComplex*>y, incy, <const double*>c, <const double*>s)
    check_status(status)


cpdef rotEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t c, intptr_t s, size_t csType, size_t executiontype):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasRotEx(<Handle>handle, n, <void*>x, <DataType>xType, incx, <void*>y, <DataType>yType, incy, <const void*>c, <const void*>s, <DataType>csType, <DataType>executiontype)
    check_status(status)


cpdef srotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSrotg(<Handle>handle, <float*>a, <float*>b, <float*>c, <float*>s)
    check_status(status)


cpdef drotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDrotg(<Handle>handle, <double*>a, <double*>b, <double*>c, <double*>s)
    check_status(status)


cpdef crotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCrotg(<Handle>handle, <cuComplex*>a, <cuComplex*>b, <float*>c, <cuComplex*>s)
    check_status(status)


cpdef zrotg(intptr_t handle, intptr_t a, intptr_t b, intptr_t c, intptr_t s):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZrotg(<Handle>handle, <cuDoubleComplex*>a, <cuDoubleComplex*>b, <double*>c, <cuDoubleComplex*>s)
    check_status(status)


cpdef rotgEx(intptr_t handle, intptr_t a, intptr_t b, size_t abType, intptr_t c, intptr_t s, size_t csType, size_t executiontype):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasRotgEx(<Handle>handle, <void*>a, <void*>b, <DataType>abType, <void*>c, <void*>s, <DataType>csType, <DataType>executiontype)
    check_status(status)


cpdef srotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSrotm(<Handle>handle, n, <float*>x, incx, <float*>y, incy, <const float*>param)
    check_status(status)


cpdef drotm(intptr_t handle, int n, intptr_t x, int incx, intptr_t y, int incy, intptr_t param):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDrotm(<Handle>handle, n, <double*>x, incx, <double*>y, incy, <const double*>param)
    check_status(status)


cpdef rotmEx(intptr_t handle, int n, intptr_t x, size_t xType, int incx, intptr_t y, size_t yType, int incy, intptr_t param, size_t paramType, size_t executiontype):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasRotmEx(<Handle>handle, n, <void*>x, <DataType>xType, incx, <void*>y, <DataType>yType, incy, <const void*>param, <DataType>paramType, <DataType>executiontype)
    check_status(status)


cpdef srotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSrotmg(<Handle>handle, <float*>d1, <float*>d2, <float*>x1, <const float*>y1, <float*>param)
    check_status(status)


cpdef drotmg(intptr_t handle, intptr_t d1, intptr_t d2, intptr_t x1, intptr_t y1, intptr_t param):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDrotmg(<Handle>handle, <double*>d1, <double*>d2, <double*>x1, <const double*>y1, <double*>param)
    check_status(status)


cpdef rotmgEx(intptr_t handle, intptr_t d1, size_t d1Type, intptr_t d2, size_t d2Type, intptr_t x1, size_t x1Type, intptr_t y1, size_t y1Type, intptr_t param, size_t paramType, size_t executiontype):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasRotmgEx(<Handle>handle, <void*>d1, <DataType>d1Type, <void*>d2, <DataType>d2Type, <void*>x1, <DataType>x1Type, <const void*>y1, <DataType>y1Type, <void*>param, <DataType>paramType, <DataType>executiontype)
    check_status(status)


cpdef sgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgemv(<Handle>handle, <Operation>trans, m, n, <const float*>alpha, <const float*>A, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgemv(<Handle>handle, <Operation>trans, m, n, <const double*>alpha, <const double*>A, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgemv(<Handle>handle, <Operation>trans, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zgemv(intptr_t handle, int trans, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgemv(<Handle>handle, <Operation>trans, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgbmv(<Handle>handle, <Operation>trans, m, n, kl, ku, <const float*>alpha, <const float*>A, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgbmv(<Handle>handle, <Operation>trans, m, n, kl, ku, <const double*>alpha, <const double*>A, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef cgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgbmv(<Handle>handle, <Operation>trans, m, n, kl, ku, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zgbmv(intptr_t handle, int trans, int m, int n, int kl, int ku, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgbmv(<Handle>handle, <Operation>trans, m, n, kl, ku, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef strmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStrmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const float*>A, lda, <float*>x, incx)
    check_status(status)


cpdef dtrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtrmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const double*>A, lda, <double*>x, incx)
    check_status(status)


cpdef ctrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtrmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuComplex*>A, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztrmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtrmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStbmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const float*>A, lda, <float*>x, incx)
    check_status(status)


cpdef dtbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtbmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const double*>A, lda, <double*>x, incx)
    check_status(status)


cpdef ctbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtbmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const cuComplex*>A, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztbmv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtbmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStpmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const float*>AP, <float*>x, incx)
    check_status(status)


cpdef dtpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtpmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const double*>AP, <double*>x, incx)
    check_status(status)


cpdef ctpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtpmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuComplex*>AP, <cuComplex*>x, incx)
    check_status(status)


cpdef ztpmv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtpmv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuDoubleComplex*>AP, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef strsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStrsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const float*>A, lda, <float*>x, incx)
    check_status(status)


cpdef dtrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtrsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const double*>A, lda, <double*>x, incx)
    check_status(status)


cpdef ctrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtrsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuComplex*>A, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztrsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtrsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStpsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const float*>AP, <float*>x, incx)
    check_status(status)


cpdef dtpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtpsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const double*>AP, <double*>x, incx)
    check_status(status)


cpdef ctpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtpsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuComplex*>AP, <cuComplex*>x, incx)
    check_status(status)


cpdef ztpsv(intptr_t handle, int uplo, int trans, int diag, int n, intptr_t AP, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtpsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, <const cuDoubleComplex*>AP, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef stbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStbsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const float*>A, lda, <float*>x, incx)
    check_status(status)


cpdef dtbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtbsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const double*>A, lda, <double*>x, incx)
    check_status(status)


cpdef ctbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtbsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const cuComplex*>A, lda, <cuComplex*>x, incx)
    check_status(status)


cpdef ztbsv(intptr_t handle, int uplo, int trans, int diag, int n, int k, intptr_t A, int lda, intptr_t x, int incx):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtbsv(<Handle>handle, <FillMode>uplo, <Operation>trans, <DiagType>diag, n, k, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>x, incx)
    check_status(status)


cpdef ssymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsymv(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const float*>A, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsymv(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const double*>A, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef csymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsymv(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zsymv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZsymv(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef chemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasChemv(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhemv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZhemv(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef ssbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsbmv(<Handle>handle, <FillMode>uplo, n, k, <const float*>alpha, <const float*>A, lda, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dsbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsbmv(<Handle>handle, <FillMode>uplo, n, k, <const double*>alpha, <const double*>A, lda, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef chbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasChbmv(<Handle>handle, <FillMode>uplo, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhbmv(intptr_t handle, int uplo, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZhbmv(<Handle>handle, <FillMode>uplo, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSspmv(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const float*>AP, <const float*>x, incx, <const float*>beta, <float*>y, incy)
    check_status(status)


cpdef dspmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDspmv(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const double*>AP, <const double*>x, incx, <const double*>beta, <double*>y, incy)
    check_status(status)


cpdef chpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasChpmv(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>AP, <const cuComplex*>x, incx, <const cuComplex*>beta, <cuComplex*>y, incy)
    check_status(status)


cpdef zhpmv(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t AP, intptr_t x, int incx, intptr_t beta, intptr_t y, int incy):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZhpmv(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>AP, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>beta, <cuDoubleComplex*>y, incy)
    check_status(status)


cpdef sger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSger(<Handle>handle, m, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>A, lda)
    check_status(status)


cpdef dger(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDger(<Handle>handle, m, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>A, lda)
    check_status(status)


cpdef cgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgeru(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef cgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgerc(<Handle>handle, m, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef zgeru(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgeru(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef zgerc(intptr_t handle, int m, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgerc(<Handle>handle, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef ssyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsyr(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>A, lda)
    check_status(status)


cpdef dsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsyr(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>A, lda)
    check_status(status)


cpdef csyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsyr(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <cuComplex*>A, lda)
    check_status(status)


cpdef zsyr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZsyr(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef cher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCher(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>A, lda)
    check_status(status)


cpdef zher(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZher(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef sspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSspr(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <float*>AP)
    check_status(status)


cpdef dspr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDspr(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <double*>AP)
    check_status(status)


cpdef chpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasChpr(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const cuComplex*>x, incx, <cuComplex*>AP)
    check_status(status)


cpdef zhpr(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZhpr(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>AP)
    check_status(status)


cpdef ssyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsyr2(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>A, lda)
    check_status(status)


cpdef dsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsyr2(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>A, lda)
    check_status(status)


cpdef csyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsyr2(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef zsyr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZsyr2(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef cher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCher2(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>A, lda)
    check_status(status)


cpdef zher2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZher2(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef sspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSspr2(<Handle>handle, <FillMode>uplo, n, <const float*>alpha, <const float*>x, incx, <const float*>y, incy, <float*>AP)
    check_status(status)


cpdef dspr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDspr2(<Handle>handle, <FillMode>uplo, n, <const double*>alpha, <const double*>x, incx, <const double*>y, incy, <double*>AP)
    check_status(status)


cpdef chpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasChpr2(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>alpha, <const cuComplex*>x, incx, <const cuComplex*>y, incy, <cuComplex*>AP)
    check_status(status)


cpdef zhpr2(intptr_t handle, int uplo, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZhpr2(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>x, incx, <const cuDoubleComplex*>y, incy, <cuDoubleComplex*>AP)
    check_status(status)


cpdef sgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgemm(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const float*>A, lda, <const float*>B, ldb, <const float*>beta, <float*>C, ldc)
    check_status(status)


cpdef dgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgemm(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const double*>alpha, <const double*>A, lda, <const double*>B, ldb, <const double*>beta, <double*>C, ldc)
    check_status(status)


cpdef cgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgemm(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef cgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgemm3m(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zgemm(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgemm(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef zgemm3m(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgemm3m(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgemmEx(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const void*>A, <DataType>Atype, lda, <const void*>B, <DataType>Btype, ldb, <const float*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef cgemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgemmEx(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const void*>A, <DataType>Atype, lda, <const void*>B, <DataType>Btype, ldb, <const cuComplex*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef ssyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsyrk(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const float*>alpha, <const float*>A, lda, <const float*>beta, <float*>C, ldc)
    check_status(status)


cpdef dsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsyrk(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const double*>alpha, <const double*>A, lda, <const double*>beta, <double*>C, ldc)
    check_status(status)


cpdef csyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsyrk(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zsyrk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZsyrk(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef csyrkEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsyrkEx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuComplex*>alpha, <const void*>A, <DataType>Atype, lda, <const cuComplex*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef csyrk3mEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsyrk3mEx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuComplex*>alpha, <const void*>A, <DataType>Atype, lda, <const cuComplex*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef cherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCherk(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const float*>alpha, <const cuComplex*>A, lda, <const float*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zherk(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZherk(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const double*>alpha, <const cuDoubleComplex*>A, lda, <const double*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef cherkEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCherkEx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const float*>alpha, <const void*>A, <DataType>Atype, lda, <const float*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef cherk3mEx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t beta, intptr_t C, size_t Ctype, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCherk3mEx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const float*>alpha, <const void*>A, <DataType>Atype, lda, <const float*>beta, <void*>C, <DataType>Ctype, ldc)
    check_status(status)


cpdef ssyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsyr2k(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const float*>alpha, <const float*>A, lda, <const float*>B, ldb, <const float*>beta, <float*>C, ldc)
    check_status(status)


cpdef dsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsyr2k(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const double*>alpha, <const double*>A, lda, <const double*>B, ldb, <const double*>beta, <double*>C, ldc)
    check_status(status)


cpdef csyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsyr2k(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zsyr2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZsyr2k(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef cher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCher2k(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const float*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zher2k(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZher2k(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const double*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef ssyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsyrkx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const float*>alpha, <const float*>A, lda, <const float*>B, ldb, <const float*>beta, <float*>C, ldc)
    check_status(status)


cpdef dsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsyrkx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const double*>alpha, <const double*>A, lda, <const double*>B, ldb, <const double*>beta, <double*>C, ldc)
    check_status(status)


cpdef csyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsyrkx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zsyrkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZsyrkx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef cherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCherkx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const float*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zherkx(intptr_t handle, int uplo, int trans, int n, int k, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZherkx(<Handle>handle, <FillMode>uplo, <Operation>trans, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const double*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef ssymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSsymm(<Handle>handle, <SideMode>side, <FillMode>uplo, m, n, <const float*>alpha, <const float*>A, lda, <const float*>B, ldb, <const float*>beta, <float*>C, ldc)
    check_status(status)


cpdef dsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDsymm(<Handle>handle, <SideMode>side, <FillMode>uplo, m, n, <const double*>alpha, <const double*>A, lda, <const double*>B, ldb, <const double*>beta, <double*>C, ldc)
    check_status(status)


cpdef csymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCsymm(<Handle>handle, <SideMode>side, <FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zsymm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZsymm(<Handle>handle, <SideMode>side, <FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef chemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasChemm(<Handle>handle, <SideMode>side, <FillMode>uplo, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <const cuComplex*>beta, <cuComplex*>C, ldc)
    check_status(status)


cpdef zhemm(intptr_t handle, int side, int uplo, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t beta, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZhemm(<Handle>handle, <SideMode>side, <FillMode>uplo, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef strsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStrsm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const float*>alpha, <const float*>A, lda, <float*>B, ldb)
    check_status(status)


cpdef dtrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtrsm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const double*>alpha, <const double*>A, lda, <double*>B, ldb)
    check_status(status)


cpdef ctrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtrsm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <cuComplex*>B, ldb)
    check_status(status)


cpdef ztrsm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtrsm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>B, ldb)
    check_status(status)


cpdef strmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStrmm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const float*>alpha, <const float*>A, lda, <const float*>B, ldb, <float*>C, ldc)
    check_status(status)


cpdef dtrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtrmm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const double*>alpha, <const double*>A, lda, <const double*>B, ldb, <double*>C, ldc)
    check_status(status)


cpdef ctrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtrmm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>B, ldb, <cuComplex*>C, ldc)
    check_status(status)


cpdef ztrmm(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtrmm(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>B, ldb, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const float* const*>Aarray, lda, <const float* const*>Barray, ldb, <const float*>beta, <float* const*>Carray, ldc, batchCount)
    check_status(status)


cpdef dgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const double*>alpha, <const double* const*>Aarray, lda, <const double* const*>Barray, ldb, <const double*>beta, <double* const*>Carray, ldc, batchCount)
    check_status(status)


cpdef cgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex* const*>Aarray, lda, <const cuComplex* const*>Barray, ldb, <const cuComplex*>beta, <cuComplex* const*>Carray, ldc, batchCount)
    check_status(status)


cpdef zgemmBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, int lda, intptr_t Barray, int ldb, intptr_t beta, intptr_t Carray, int ldc, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgemmBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>Aarray, lda, <const cuDoubleComplex* const*>Barray, ldb, <const cuDoubleComplex*>beta, <cuDoubleComplex* const*>Carray, ldc, batchCount)
    check_status(status)


cpdef sgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const float*>alpha, <const float*>A, lda, strideA, <const float*>B, ldb, strideB, <const float*>beta, <float*>C, ldc, strideC, batchCount)
    check_status(status)


cpdef dgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const double*>alpha, <const double*>A, lda, strideA, <const double*>B, ldb, strideB, <const double*>beta, <double*>C, ldc, strideC, batchCount)
    check_status(status)


cpdef cgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, strideA, <const cuComplex*>B, ldb, strideB, <const cuComplex*>beta, <cuComplex*>C, ldc, strideC, batchCount)
    check_status(status)


cpdef cgemm3mStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgemm3mStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuComplex*>alpha, <const cuComplex*>A, lda, strideA, <const cuComplex*>B, ldb, strideB, <const cuComplex*>beta, <cuComplex*>C, ldc, strideC, batchCount)
    check_status(status)


cpdef zgemmStridedBatched(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, int lda, long long int strideA, intptr_t B, int ldb, long long int strideB, intptr_t beta, intptr_t C, int ldc, long long int strideC, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgemmStridedBatched(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, strideA, <const cuDoubleComplex*>B, ldb, strideB, <const cuDoubleComplex*>beta, <cuDoubleComplex*>C, ldc, strideC, batchCount)
    check_status(status)


cpdef sgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const float*>alpha, <const float*>A, lda, <const float*>beta, <const float*>B, ldb, <float*>C, ldc)
    check_status(status)


cpdef dgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const double*>alpha, <const double*>A, lda, <const double*>beta, <const double*>B, ldb, <double*>C, ldc)
    check_status(status)


cpdef cgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const cuComplex*>alpha, <const cuComplex*>A, lda, <const cuComplex*>beta, <const cuComplex*>B, ldb, <cuComplex*>C, ldc)
    check_status(status)


cpdef zgeam(intptr_t handle, int transa, int transb, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t beta, intptr_t B, int ldb, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgeam(<Handle>handle, <Operation>transa, <Operation>transb, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>beta, <const cuDoubleComplex*>B, ldb, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef sgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgetrfBatched(<Handle>handle, n, <float* const*>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)


cpdef dgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgetrfBatched(<Handle>handle, n, <double* const*>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)


cpdef cgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgetrfBatched(<Handle>handle, n, <cuComplex* const*>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)


cpdef zgetrfBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgetrfBatched(<Handle>handle, n, <cuDoubleComplex* const*>A, lda, <int*>P, <int*>info, batchSize)
    check_status(status)


cpdef sgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgetriBatched(<Handle>handle, n, <const float* const*>A, lda, <const int*>P, <float* const*>C, ldc, <int*>info, batchSize)
    check_status(status)


cpdef dgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgetriBatched(<Handle>handle, n, <const double* const*>A, lda, <const int*>P, <double* const*>C, ldc, <int*>info, batchSize)
    check_status(status)


cpdef cgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgetriBatched(<Handle>handle, n, <const cuComplex* const*>A, lda, <const int*>P, <cuComplex* const*>C, ldc, <int*>info, batchSize)
    check_status(status)


cpdef zgetriBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t P, intptr_t C, int ldc, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgetriBatched(<Handle>handle, n, <const cuDoubleComplex* const*>A, lda, <const int*>P, <cuDoubleComplex* const*>C, ldc, <int*>info, batchSize)
    check_status(status)


cpdef sgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const float* const*>Aarray, lda, <const int*>devIpiv, <float* const*>Barray, ldb, <int*>info, batchSize)
    check_status(status)


cpdef dgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const double* const*>Aarray, lda, <const int*>devIpiv, <double* const*>Barray, ldb, <int*>info, batchSize)
    check_status(status)


cpdef cgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const cuComplex* const*>Aarray, lda, <const int*>devIpiv, <cuComplex* const*>Barray, ldb, <int*>info, batchSize)
    check_status(status)


cpdef zgetrsBatched(intptr_t handle, int trans, int n, int nrhs, intptr_t Aarray, int lda, intptr_t devIpiv, intptr_t Barray, int ldb, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgetrsBatched(<Handle>handle, <Operation>trans, n, nrhs, <const cuDoubleComplex* const*>Aarray, lda, <const int*>devIpiv, <cuDoubleComplex* const*>Barray, ldb, <int*>info, batchSize)
    check_status(status)


cpdef strsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStrsmBatched(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const float*>alpha, <const float* const*>A, lda, <float* const*>B, ldb, batchCount)
    check_status(status)


cpdef dtrsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtrsmBatched(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const double*>alpha, <const double* const*>A, lda, <double* const*>B, ldb, batchCount)
    check_status(status)


cpdef ctrsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtrsmBatched(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuComplex*>alpha, <const cuComplex* const*>A, lda, <cuComplex* const*>B, ldb, batchCount)
    check_status(status)


cpdef ztrsmBatched(intptr_t handle, int side, int uplo, int trans, int diag, int m, int n, intptr_t alpha, intptr_t A, int lda, intptr_t B, int ldb, int batchCount):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtrsmBatched(<Handle>handle, <SideMode>side, <FillMode>uplo, <Operation>trans, <DiagType>diag, m, n, <const cuDoubleComplex*>alpha, <const cuDoubleComplex* const*>A, lda, <cuDoubleComplex* const*>B, ldb, batchCount)
    check_status(status)


cpdef smatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSmatinvBatched(<Handle>handle, n, <const float* const*>A, lda, <float* const*>Ainv, lda_inv, <int*>info, batchSize)
    check_status(status)


cpdef dmatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDmatinvBatched(<Handle>handle, n, <const double* const*>A, lda, <double* const*>Ainv, lda_inv, <int*>info, batchSize)
    check_status(status)


cpdef cmatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCmatinvBatched(<Handle>handle, n, <const cuComplex* const*>A, lda, <cuComplex* const*>Ainv, lda_inv, <int*>info, batchSize)
    check_status(status)


cpdef zmatinvBatched(intptr_t handle, int n, intptr_t A, int lda, intptr_t Ainv, int lda_inv, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZmatinvBatched(<Handle>handle, n, <const cuDoubleComplex* const*>A, lda, <cuDoubleComplex* const*>Ainv, lda_inv, <int*>info, batchSize)
    check_status(status)


cpdef sgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgeqrfBatched(<Handle>handle, m, n, <float* const*>Aarray, lda, <float* const*>TauArray, <int*>info, batchSize)
    check_status(status)


cpdef dgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgeqrfBatched(<Handle>handle, m, n, <double* const*>Aarray, lda, <double* const*>TauArray, <int*>info, batchSize)
    check_status(status)


cpdef cgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgeqrfBatched(<Handle>handle, m, n, <cuComplex* const*>Aarray, lda, <cuComplex* const*>TauArray, <int*>info, batchSize)
    check_status(status)


cpdef zgeqrfBatched(intptr_t handle, int m, int n, intptr_t Aarray, int lda, intptr_t TauArray, intptr_t info, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgeqrfBatched(<Handle>handle, m, n, <cuDoubleComplex* const*>Aarray, lda, <cuDoubleComplex* const*>TauArray, <int*>info, batchSize)
    check_status(status)


cpdef sgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSgelsBatched(<Handle>handle, <Operation>trans, m, n, nrhs, <float* const*>Aarray, lda, <float* const*>Carray, ldc, <int*>info, <int*>devInfoArray, batchSize)
    check_status(status)


cpdef dgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDgelsBatched(<Handle>handle, <Operation>trans, m, n, nrhs, <double* const*>Aarray, lda, <double* const*>Carray, ldc, <int*>info, <int*>devInfoArray, batchSize)
    check_status(status)


cpdef cgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCgelsBatched(<Handle>handle, <Operation>trans, m, n, nrhs, <cuComplex* const*>Aarray, lda, <cuComplex* const*>Carray, ldc, <int*>info, <int*>devInfoArray, batchSize)
    check_status(status)


cpdef zgelsBatched(intptr_t handle, int trans, int m, int n, int nrhs, intptr_t Aarray, int lda, intptr_t Carray, int ldc, intptr_t info, intptr_t devInfoArray, int batchSize):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZgelsBatched(<Handle>handle, <Operation>trans, m, n, nrhs, <cuDoubleComplex* const*>Aarray, lda, <cuDoubleComplex* const*>Carray, ldc, <int*>info, <int*>devInfoArray, batchSize)
    check_status(status)


cpdef sdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasSdgmm(<Handle>handle, <SideMode>mode, m, n, <const float*>A, lda, <const float*>x, incx, <float*>C, ldc)
    check_status(status)


cpdef ddgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDdgmm(<Handle>handle, <SideMode>mode, m, n, <const double*>A, lda, <const double*>x, incx, <double*>C, ldc)
    check_status(status)


cpdef cdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCdgmm(<Handle>handle, <SideMode>mode, m, n, <const cuComplex*>A, lda, <const cuComplex*>x, incx, <cuComplex*>C, ldc)
    check_status(status)


cpdef zdgmm(intptr_t handle, int mode, int m, int n, intptr_t A, int lda, intptr_t x, int incx, intptr_t C, int ldc):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZdgmm(<Handle>handle, <SideMode>mode, m, n, <const cuDoubleComplex*>A, lda, <const cuDoubleComplex*>x, incx, <cuDoubleComplex*>C, ldc)
    check_status(status)


cpdef stpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStpttr(<Handle>handle, <FillMode>uplo, n, <const float*>AP, <float*>A, lda)
    check_status(status)


cpdef dtpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtpttr(<Handle>handle, <FillMode>uplo, n, <const double*>AP, <double*>A, lda)
    check_status(status)


cpdef ctpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtpttr(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>AP, <cuComplex*>A, lda)
    check_status(status)


cpdef ztpttr(intptr_t handle, int uplo, int n, intptr_t AP, intptr_t A, int lda):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtpttr(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>AP, <cuDoubleComplex*>A, lda)
    check_status(status)


cpdef strttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasStrttp(<Handle>handle, <FillMode>uplo, n, <const float*>A, lda, <float*>AP)
    check_status(status)


cpdef dtrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasDtrttp(<Handle>handle, <FillMode>uplo, n, <const double*>A, lda, <double*>AP)
    check_status(status)


cpdef ctrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasCtrttp(<Handle>handle, <FillMode>uplo, n, <const cuComplex*>A, lda, <cuComplex*>AP)
    check_status(status)


cpdef ztrttp(intptr_t handle, int uplo, int n, intptr_t A, int lda, intptr_t AP):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        status = cublasZtrttp(<Handle>handle, <FillMode>uplo, n, <const cuDoubleComplex*>A, lda, <cuDoubleComplex*>AP)
    check_status(status)


cpdef setWorkspace(intptr_t handle, intptr_t workspace, size_t workspaceSizeInBytes):
    with nogil:
        status = cublasSetWorkspace(<Handle>handle, <void*>workspace, workspaceSizeInBytes)
    check_status(status)


# Define by hand for backward compatibility
cpdef gemmEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, intptr_t B, size_t Btype, int ldb, intptr_t beta, intptr_t C, size_t Ctype, int ldc, int computeType, int algo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        if computeType >= CUBLAS_COMPUTE_16F:
            status = cublasGemmEx_v11(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const void*>alpha, <const void*>A, <DataType>Atype, lda, <const void*>B, <DataType>Btype, ldb, <const void*>beta, <void*>C, <DataType>Ctype, ldc, <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = cublasGemmEx(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const void*>alpha, <const void*>A, <DataType>Atype, lda, <const void*>B, <DataType>Btype, ldb, <const void*>beta, <void*>C, <DataType>Ctype, ldc, <DataType>computeType, <GemmAlgo>algo)
    check_status(status)


# Define by hand for backward compatibility
cpdef gemmBatchedEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t Aarray, size_t Atype, int lda, intptr_t Barray, size_t Btype, int ldb, intptr_t beta, intptr_t Carray, size_t Ctype, int ldc, int batchCount, int computeType, int algo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        if computeType >= CUBLAS_COMPUTE_16F:
            status = cublasGemmBatchedEx_v11(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const void*>alpha, <const void* const*>Aarray, <DataType>Atype, lda, <const void* const*>Barray, <DataType>Btype, ldb, <const void*>beta, <void* const*>Carray, <DataType>Ctype, ldc, batchCount, <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = cublasGemmBatchedEx(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const void*>alpha, <const void* const*>Aarray, <DataType>Atype, lda, <const void* const*>Barray, <DataType>Btype, ldb, <const void*>beta, <void* const*>Carray, <DataType>Ctype, ldc, batchCount, <DataType>computeType, <GemmAlgo>algo)
    check_status(status)


# Define by hand for backward compatibility
cpdef gemmStridedBatchedEx(intptr_t handle, int transa, int transb, int m, int n, int k, intptr_t alpha, intptr_t A, size_t Atype, int lda, long long int strideA, intptr_t B, size_t Btype, int ldb, long long int strideB, intptr_t beta, intptr_t C, size_t Ctype, int ldc, long long int strideC, int batchCount, int computeType, int algo):
    setStream(handle, stream_module.get_current_stream_ptr())
    with nogil:
        if computeType >= CUBLAS_COMPUTE_16F:
            status = cublasGemmStridedBatchedEx_v11(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const void*>alpha, <const void*>A, <DataType>Atype, lda, strideA, <const void*>B, <DataType>Btype, ldb, strideB, <const void*>beta, <void*>C, <DataType>Ctype, ldc, strideC, batchCount, <ComputeType>computeType, <GemmAlgo>algo)
        else:
            status = cublasGemmStridedBatchedEx(<Handle>handle, <Operation>transa, <Operation>transb, m, n, k, <const void*>alpha, <const void*>A, <DataType>Atype, lda, strideA, <const void*>B, <DataType>Btype, ldb, strideB, <const void*>beta, <void*>C, <DataType>Ctype, ldc, strideC, batchCount, <DataType>computeType, <GemmAlgo>algo)
    check_status(status)
