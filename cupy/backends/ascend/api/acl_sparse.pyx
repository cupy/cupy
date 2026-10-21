# distutils: language = c++
"""aclsparse (CANN ops-sparse) binding exposing a cuSPARSE-compatible surface.

This module is the Ascend counterpart of :mod:`cupy.backends.cuda.libs.cusparse`.
It wraps the CANN sparse operator library (``ops-sparse``, ``libops_sparse.so``,
public header ``cann_ops_sparse.h`` -- vendored next to this file, identical to
``~/repos/ops-sparse/include/cann_ops_sparse.h``) behind the **names, signatures
and constants used by ``cupyx/cusparse.py``**, so that the upstream facade code
needs no per-op changes: on Ascend
:func:`cupyx._cusparse_backend.get_cusparse` simply returns this module.

Deliberate design points (see ``docs/ascend/ascend_sparse.md``):

* **Handle cache** -- one ``aclsparseHandle_t`` per ``(device, current stream)``,
  created with ``aclsparseSetPointerMode(ACL_SPARSE_POINTER_MODE_HOST)`` so that
  the host-side ``alpha``/``beta``/``nnzTotal`` pointers passed by ``cupyx``
  (``numpy.array(...).ctypes.data``) are read directly.  This mirrors the
  reference ``torch_extension``'s ``AclSparseHandleCache`` and upstream
  ``cupy.cuda.device.get_cusparse_handle()``.
* **Device pointers pass straight through** -- unlike ops-fft (which stages
  through host buffers), aclsparse consumes device pointers, so
  ``array.data.ptr`` is forwarded verbatim.
* **dtype handling** -- ``cupy._core._dtype.to_cuda_dtype`` returns an
  ``aclDataType`` on Ascend (it maps to ``numpy_dtype_to_acl_dtype``), so the
  ``*Type`` arguments received here are already aclnn dtypes and are forwarded
  unchanged.  The cuSPARSE dtype *prefixes* (``s``/``d``/``c``/``z``) used by
  ``cupyx.cusparse._call_cusparse`` coincide with the aclsparse ones
  (``S``/``D``/``C``/``Z``) for fp32/fp64/complex64.
* **Legacy vs generic** -- ops-sparse implements only a subset of the legacy
  cuSPARSE API (``Scsrgeam2``, sorts, format conversions).  ``cupyx`` probes
  availability through :data:`_available_aclsparse` (consulted by
  ``cupyx.cusparse.check_availability`` on Ascend) and falls back to the
  generic (descriptor-based) API, which ops-sparse implements 1:1
  (``SpMV``/``SpMM``/``SpGEMM``/``SpSM``/``SparseToDense``/``DenseToSparse``).
* **Not implemented by ops-sparse** (raise a clear error; the corresponding
  ``check_availability`` entries are ``False`` so ``cupyx`` degrades):
  ``csrilu02`` (ILU(0)), ``csr2csr_compress``/``nnz_compress``,
  ``csrgemm``/``csrgemm2`` legacy (``cupyx`` uses ``spgemm`` instead),
  ``csrsm2`` legacy (``cupyx`` uses ``spsm``), fp64 ``csrgeam2`` (ops-sparse
  only ships the FP32 ``aclsparseScsrgeam2``), fp64/complex128 ``SpGEMM``.

Conventions verified against ``~/repos/ops-sparse/test/*`` (the same semantics
as cuSPARSE, e.g. SpMV: ``x`` has ``op(A) == T ? m : n`` elements, ``y`` has
``op(A) == T ? n : m``).

Verification boundary: without a 910B only import/registration (L3) is
testable here; every numerical claim requires hardware (L4).
"""
from __future__ import annotations

import threading

import numpy

from libc.stdint cimport int32_t, int64_t, intptr_t

from cupy.backends.backend.api cimport runtime
from cupy.xpu cimport stream as stream_module


# ---------------------------------------------------------------------------
# Extern declarations (vendored header: cann_ops_sparse.h, next to this file)
# ---------------------------------------------------------------------------
cdef extern from "cann_ops_sparse.h" nogil:
    # Opaque descriptor types (never dereferenced, only cast to/from ints)
    ctypedef struct aclsparseContext
    ctypedef aclsparseContext* aclsparseHandle_t
    ctypedef struct aclsparseSpMatDescr
    ctypedef aclsparseSpMatDescr* aclsparseSpMatDescr_t
    ctypedef const aclsparseSpMatDescr* aclsparseConstSpMatDescr_t
    ctypedef struct aclsparseDnVecDescr
    ctypedef aclsparseDnVecDescr* aclsparseDnVecDescr_t
    ctypedef const aclsparseDnVecDescr* aclsparseConstDnVecDescr_t
    ctypedef struct aclsparseDnMatDescr
    ctypedef aclsparseDnMatDescr* aclsparseDnMatDescr_t
    ctypedef const aclsparseDnMatDescr* aclsparseConstDnMatDescr_t
    ctypedef struct aclsparseSpVecDescr
    ctypedef aclsparseSpVecDescr* aclsparseSpVecDescr_t
    ctypedef const aclsparseSpVecDescr* aclsparseConstSpVecDescr_t
    ctypedef struct aclsparseMatDescr
    ctypedef aclsparseMatDescr* aclsparseMatDescr_t
    ctypedef struct aclsparseSpSMDescr
    ctypedef aclsparseSpSMDescr* aclsparseSpSMDescr_t
    ctypedef struct aclsparseSpGEMMDescr
    ctypedef aclsparseSpGEMMDescr* aclsparseSpGEMMDescr_t

    # Enums: int aliases so Cython emits the header typedef names verbatim
    # (same trick as aclfft.pyx); the numeric values are defined by the header.
    ctypedef int aclsparseStatus_t
    ctypedef int aclsparseOperation_t
    ctypedef int aclsparseOrder_t
    ctypedef int aclsparseMatrixType_t
    ctypedef int aclsparseDiagType_t
    ctypedef int aclsparseFillMode_t
    ctypedef int aclsparseIndexType_t
    ctypedef int aclsparseFormat_t
    ctypedef int aclsparseIndexBase_t
    ctypedef int aclsparseDirection_t
    ctypedef int aclsparseAction_t
    ctypedef int aclsparseCsr2CscAlg_t
    ctypedef int aclsparseSpMVAlg_t
    ctypedef int aclsparseSpMMAlg_t
    ctypedef int aclsparseSpSMAlg_t
    ctypedef int aclsparseSpGEMMAlg_t
    ctypedef int aclsparsePointerMode_t
    ctypedef int aclsparseDenseToSparseAlg_t
    ctypedef int aclsparseSparseToDenseAlg_t
    ctypedef int aclsparseSpMatAttribute_t
    ctypedef int aclDataType          # from acl/acl_base_rt.h
    ctypedef int aclError             # from acl/acl_base_rt.h
    ctypedef void* aclrtStream        # from acl/acl_base_rt.h

    # --- enum constants used below -----------------------------------------
    int ACL_SPARSE_STATUS_SUCCESS
    int ACL_SPARSE_OP_NON_TRANSPOSE
    int ACL_SPARSE_OP_TRANSPOSE
    int ACL_SPARSE_OP_CONJUGATE_TRANSPOSE
    aclsparseOrder_t ACL_SPARSE_ORDER_ROW
    aclsparseOrder_t ACL_SPARSE_ORDER_COL
    int ACL_SPARSE_MATRIX_TYPE_GENERAL
    int ACL_SPARSE_FILL_MODE_LOWER
    int ACL_SPARSE_FILL_MODE_UPPER
    int ACL_SPARSE_DIAG_TYPE_NON_UNIT
    int ACL_SPARSE_DIAG_TYPE_UNIT
    aclsparseIndexType_t ACL_SPARSE_INDEX_32I
    aclsparseIndexType_t ACL_SPARSE_INDEX_64I
    int ACL_SPARSE_INDEX_BASE_ZERO
    int ACL_SPARSE_DIRECTION_ROW
    int ACL_SPARSE_DIRECTION_COLUMN
    int ACL_SPARSE_ACTION_SYMBOLIC
    int ACL_SPARSE_ACTION_NUMERIC
    int ACL_SPARSE_CSR2CSC_ALG_DEFAULT
    int ACL_SPARSE_CSR2CSC_ALG1
    int ACL_SPARSE_SPMV_ALG_DEFAULT
    int ACL_SPARSE_SPMM_ALG_DEFAULT
    int ACL_SPARSE_SPSM_ALG_DEFAULT
    int ACL_SPARSE_SPGEMM_DEFAULT
    int ACL_SPARSE_SPGEMM_ALG2
    int ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT
    int ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT
    int ACL_SPARSE_SPMAT_FILL_MODE
    int ACL_SPARSE_SPMAT_DIAG_TYPE
    int ACL_SPARSE_POINTER_MODE_HOST
    int ACL_SPARSE_POINTER_MODE_DEVICE
    int ACL_MEMCPY_HOST_TO_DEVICE     # from acl/acl_rt.h (via acl_base_rt.h)

    # --- handle / stream / pointer mode -------------------------------------
    aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle)
    aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle)
    aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle,
                                         aclrtStream stream)
    aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle,
                                         aclrtStream *stream)
    aclsparseStatus_t aclsparseSetPointerMode(aclsparseHandle_t handle,
                                              aclsparsePointerMode_t mode)
    aclsparseStatus_t aclsparseGetPointerMode(aclsparseHandle_t handle,
                                              aclsparsePointerMode_t *mode)
    aclsparseStatus_t aclsparseGetVersion(aclsparseHandle_t handle,
                                          int *version)

    # --- legacy matrix descriptor -------------------------------------------
    aclsparseStatus_t aclsparseCreateMatDescr(aclsparseMatDescr_t *descr)
    aclsparseStatus_t aclsparseDestroyMatDescr(aclsparseMatDescr_t descr)
    void aclsparseSetMatType(aclsparseMatDescr_t descr,
                             aclsparseMatrixType_t type)
    void aclsparseSetMatIndexBase(aclsparseMatDescr_t descr,
                                  aclsparseIndexBase_t base)
    void aclsparseSetMatFillMode(aclsparseMatDescr_t descr,
                                 aclsparseFillMode_t fillMode)
    void aclsparseSetMatDiagType(aclsparseMatDescr_t descr,
                                 aclsparseDiagType_t diagType)

    # --- sparse matrix descriptors ------------------------------------------
    aclsparseStatus_t aclsparseCreateCsr(
        aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
        int64_t nnz, void *csrRowOffsets, void *csrColInd, void *csrValues,
        aclsparseIndexType_t csrRowOffsetsType,
        aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase,
        aclDataType valueType)
    aclsparseStatus_t aclsparseCreateCoo(
        aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
        int64_t nnz, void *cooRowInd, void *cooColInd, void *cooValues,
        aclsparseIndexType_t cooIdxType, aclsparseIndexBase_t idxBase,
        aclDataType valueType)
    aclsparseStatus_t aclsparseCreateCsc(
        aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols,
        int64_t nnz, void *cscColOffsets, void *cscRowInd, void *cscValues,
        aclsparseIndexType_t cscColOffsetsType,
        aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase,
        aclDataType valueType)
    aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t descr)
    aclsparseStatus_t aclsparseCsrSetPointers(aclsparseSpMatDescr_t descr,
                                              void *rowOffsets,
                                              void *colIndices,
                                              void *values)
    aclsparseStatus_t aclsparseSpMatGetSize(aclsparseConstSpMatDescr_t descr,
                                            int64_t *rows, int64_t *cols,
                                            int64_t *nnz)
    aclsparseStatus_t aclsparseSpMatSetAttribute(
        aclsparseSpMatDescr_t descr, aclsparseSpMatAttribute_t attribute,
        const void *data, size_t dataSize)
    aclsparseStatus_t aclsparseSpMatGetAttribute(
        aclsparseConstSpMatDescr_t descr, aclsparseSpMatAttribute_t attribute,
        void *data, size_t dataSize)
    aclsparseStatus_t aclsparseSpMatGetFormat(
        aclsparseConstSpMatDescr_t descr, int *format)
    aclsparseStatus_t aclsparseSpMatGetValues(aclsparseSpMatDescr_t descr,
                                              void **values)
    aclsparseStatus_t aclsparseSpMatSetValues(aclsparseSpMatDescr_t descr,
                                              void *values)
    aclsparseStatus_t aclsparseCooGet(
        aclsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
        int64_t *nnz, void **cooRowInd, void **cooColInd, void **cooValues,
        aclsparseIndexType_t *cooIdxType, aclsparseIndexBase_t *idxBase,
        aclDataType *valueType)

    # --- sparse / dense vector & dense matrix descriptors -------------------
    aclsparseStatus_t aclsparseCreateSpVec(
        aclsparseSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz,
        void *indices, void *values, aclsparseIndexType_t idxType,
        aclsparseIndexBase_t idxBase, aclDataType valueType)
    aclsparseStatus_t aclsparseDestroySpVec(aclsparseConstSpVecDescr_t descr)
    aclsparseStatus_t aclsparseSpVecGet(
        aclsparseSpVecDescr_t spVecDescr, int64_t *size, int64_t *nnz,
        void **indices, void **values, aclsparseIndexType_t *idxType,
        aclsparseIndexBase_t *idxBase, aclDataType *valueType)
    aclsparseStatus_t aclsparseCreateDnVec(aclsparseDnVecDescr_t *dnVecDescr,
                                           int64_t size, void *values,
                                           aclDataType valueType)
    aclsparseStatus_t aclsparseDestroyDnVec(aclsparseConstDnVecDescr_t descr)
    aclsparseStatus_t aclsparseDnVecGet(aclsparseDnVecDescr_t dnVecDescr,
                                        int64_t *size, void **values,
                                        aclDataType *valueType)
    aclsparseStatus_t aclsparseCreateDnMat(
        aclsparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols,
        int64_t ld, void *values, aclDataType valueType,
        aclsparseOrder_t order)
    aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t descr)
    aclsparseStatus_t aclsparseDnMatGet(aclsparseDnMatDescr_t dnMatDescr,
                                        int64_t *rows, int64_t *cols,
                                        int64_t *ld, void **values,
                                        aclDataType *valueType,
                                        aclsparseOrder_t *order)

    # --- generic ops ---------------------------------------------------------
    aclsparseStatus_t aclsparseSpMVGetBufferSize(
        aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
        const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
        aclsparseSpMVAlg_t alg, size_t *bufferSize)
    aclsparseStatus_t aclsparseSpMV(
        aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
        const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
        aclsparseSpMVAlg_t alg, void *externalBuffer)
    aclsparseStatus_t aclsparseSpMMGetBufferSize(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
        const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
        aclsparseSpMMAlg_t alg, size_t *size)
    aclsparseStatus_t aclsparseSpMM(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
        const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
        aclsparseSpMMAlg_t alg, void *buffer)
    aclsparseStatus_t aclsparseSparseToDense_bufferSize(
        aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
        aclsparseDnMatDescr_t matB, aclsparseSparseToDenseAlg_t alg,
        size_t *bufferSize)
    aclsparseStatus_t aclsparseSparseToDense(
        aclsparseHandle_t handle, aclsparseConstSpMatDescr_t matA,
        aclsparseDnMatDescr_t matB, aclsparseSparseToDenseAlg_t alg,
        void *buffer)
    aclsparseStatus_t aclsparseDenseToSparseGetBufferSize(
        aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
        aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
        size_t *bufferSize)
    aclsparseStatus_t aclsparseDenseToSparseAnalysis(
        aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
        aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
        void *buffer)
    aclsparseStatus_t aclsparseDenseToSparseConvert(
        aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
        aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
        void *buffer)
    aclsparseStatus_t aclsparseGather(aclsparseHandle_t handle,
                                      aclsparseConstDnVecDescr_t vecY,
                                      aclsparseSpVecDescr_t vecX)
    aclsparseStatus_t aclsparseCsr2cscEx2_bufferSize(
        aclsparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
        const int *csrRowPtr, const int *csrColInd, void *cscVal,
        int *cscColPtr, int *cscRowInd, aclDataType valType,
        aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
        aclsparseCsr2CscAlg_t alg, size_t *bufferSize)
    aclsparseStatus_t aclsparseCsr2cscEx2(
        aclsparseHandle_t handle, int m, int n, int nnz, const void *csrVal,
        const int *csrRowPtr, const int *csrColInd, void *cscVal,
        int *cscColPtr, int *cscRowInd, aclDataType valType,
        aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
        aclsparseCsr2CscAlg_t alg, void *buffer)
    aclsparseStatus_t aclsparseXcsrsort_bufferSizeExt(
        aclsparseHandle_t handle, int m, int n, int nnz, const int *csrRowPtr,
        const int *csrColInd, size_t *pBufferSizeInBytes)
    aclsparseStatus_t aclsparseXcsrsort(
        aclsparseHandle_t handle, int m, int n, int nnz,
        const aclsparseMatDescr_t descrA, const int *csrRowPtr,
        int *csrColInd, int *P, void *pBuffer)
    aclsparseStatus_t aclsparseXcscsort_bufferSizeExt(
        aclsparseHandle_t handle, int m, int n, int nnz, const int *cscColPtr,
        const int *cscRowInd, size_t *pBufferSizeInBytes)
    aclsparseStatus_t aclsparseXcscsort(
        aclsparseHandle_t handle, int m, int n, int nnz,
        const aclsparseMatDescr_t descrA, const int *cscColPtr,
        int *cscRowInd, int *P, void *pBuffer)
    aclsparseStatus_t aclsparseXcoosort_bufferSizeExt(
        aclsparseHandle_t handle, int m, int n, int nnz, const int *cooRowsA,
        const int *cooColsA, size_t *pBufferSizeInBytes)
    aclsparseStatus_t aclsparseXcoosortByRow(
        aclsparseHandle_t handle, int m, int n, int nnz, int *cooRowsA,
        int *cooColsA, int *P, void *pBuffer)
    aclsparseStatus_t aclsparseXcoosortByColumn(
        aclsparseHandle_t handle, int m, int n, int nnz, int *cooRowsA,
        int *cooColsA, int *P, void *pBuffer)
    aclsparseStatus_t aclsparseXcsr2coo(aclsparseHandle_t handle,
                                        const int32_t *csrRowPtr,
                                        int64_t nnz, int64_t m,
                                        int32_t *cooRowInd,
                                        aclsparseIndexBase_t idxBase)
    aclsparseStatus_t aclsparseXcoo2csr(aclsparseHandle_t handle,
                                        const int *cooRowInd, int nnz, int m,
                                        int *csrRowPtr,
                                        aclsparseIndexBase_t idxBase)
    aclsparseStatus_t aclsparseXcsrgeam2Nnz(
        aclsparseHandle_t handle, int m, int n,
        const aclsparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, const aclsparseMatDescr_t descrB,
        int nnzB, const int *csrSortedRowPtrB, const int *csrSortedColIndB,
        const aclsparseMatDescr_t descrC, int *csrSortedRowPtrC,
        int *nnzTotalDevHostPtr, void *workspace)
    aclsparseStatus_t aclsparseScsrgeam2_bufferSizeExt(
        aclsparseHandle_t handle, int m, int n, const float *alpha,
        const aclsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const float *beta, const aclsparseMatDescr_t descrB, int nnzB,
        const float *csrSortedValB, const int *csrSortedRowPtrB,
        const int *csrSortedColIndB, const aclsparseMatDescr_t descrC,
        const float *csrSortedValC, const int *csrSortedRowPtrC,
        const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
    aclsparseStatus_t aclsparseScsrgeam2(
        aclsparseHandle_t handle, int m, int n, const float *alpha,
        const aclsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const float *beta, const aclsparseMatDescr_t descrB, int nnzB,
        const float *csrSortedValB, const int *csrSortedRowPtrB,
        const int *csrSortedColIndB, const aclsparseMatDescr_t descrC,
        float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC,
        void *pBuffer)
    aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr)
    aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr)
    aclsparseStatus_t aclsparseSpSMBufferSize(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
        aclsparseDnMatDescr_t matC, aclDataType computeType,
        aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t spsmDescr,
        size_t *bufferSize)
    aclsparseStatus_t aclsparseSpSMAnalysis(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
        aclsparseDnMatDescr_t matC, aclDataType computeType,
        aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t spsmDescr, void *buffer)
    aclsparseStatus_t aclsparseSpSM(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
        aclsparseDnMatDescr_t matC, aclDataType computeType,
        aclsparseSpSMAlg_t alg, aclsparseSpSMDescr_t spsmDescr)
    aclsparseStatus_t aclsparseSpGEMMCreateDescr(
        aclsparseSpGEMMDescr_t *descr)
    aclsparseStatus_t aclsparseSpGEMMDestroyDescr(
        aclsparseSpGEMMDescr_t descr)
    aclsparseStatus_t aclsparseSpGEMMSetCInValid(
        aclsparseSpGEMMDescr_t spgemmDescr, int32_t cInDataValid)
    aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
        const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
        aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t spgemmDescr,
        size_t *bufferSize1, void *externalBuffer1)
    aclsparseStatus_t aclsparseSpGEMMCompute(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
        const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
        aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t spgemmDescr,
        size_t *bufferSize2, void *externalBuffer2)
    aclsparseStatus_t aclsparseSpGEMMCopy(
        aclsparseHandle_t handle, aclsparseOperation_t opA,
        aclsparseOperation_t opB, const void *alpha,
        aclsparseConstSpMatDescr_t matA, aclsparseConstSpMatDescr_t matB,
        const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
        aclsparseSpGEMMAlg_t alg, aclsparseSpGEMMDescr_t spgemmDescr)


# ---------------------------------------------------------------------------
# Optional entry points
# ---------------------------------------------------------------------------
# ops-sparse implements a different operator set per SoC (``arch22`` = 910B,
# ``arch35`` = 950) and can be built with a subset of operators
# (``build.sh --ops=``), so a library that loads fine can still miss whole
# operator families.  These declarations are made *weak* on purpose: a missing
# implementation resolves to NULL instead of aborting ``import`` with
# "undefined symbol", and `_probe_optional_ops()` (see below) then disables the
# affected entries in `_available_aclsparse`.
# ---------------------------------------------------------------------------
cdef extern from *:
    """
    #include <cstring>
    #include "cann_ops_sparse.h"

    #pragma weak aclsparseCooGet
    #pragma weak aclsparseCsr2cscEx2
    #pragma weak aclsparseCsr2cscEx2_bufferSize
    #pragma weak aclsparseScsrgeam2
    #pragma weak aclsparseScsrgeam2_bufferSizeExt
    #pragma weak aclsparseSpGEMMSetCInValid
    #pragma weak aclsparseSparseToDense
    #pragma weak aclsparseSparseToDense_bufferSize
    #pragma weak aclsparseXcoo2csr
    #pragma weak aclsparseXcoosortByColumn
    #pragma weak aclsparseXcoosortByRow
    #pragma weak aclsparseXcoosort_bufferSizeExt
    #pragma weak aclsparseXcsr2coo
    #pragma weak aclsparseXcsrgeam2Nnz
    #pragma weak aclsparseXcsrsort
    #pragma weak aclsparseXcsrsort_bufferSizeExt

    // 1 = available, 0 = missing from libops_sparse.so, -1 = unknown group.
    static int _aclsparse_optional_available(const char *op)
    {
        if (strcmp(op, "cooGet") == 0) {
            return aclsparseCooGet != 0;
        }
        if (strcmp(op, "csr2cscEx2") == 0) {
            return aclsparseCsr2cscEx2 != 0
                && aclsparseCsr2cscEx2_bufferSize != 0;
        }
        if (strcmp(op, "csrsort") == 0) {
            return aclsparseXcsrsort != 0
                && aclsparseXcsrsort_bufferSizeExt != 0;
        }
        if (strcmp(op, "coosort") == 0) {
            return aclsparseXcoosortByRow != 0
                && aclsparseXcoosortByColumn != 0
                && aclsparseXcoosort_bufferSizeExt != 0;
        }
        if (strcmp(op, "csr2coo") == 0) {
            return aclsparseXcsr2coo != 0;
        }
        if (strcmp(op, "coo2csr") == 0) {
            return aclsparseXcoo2csr != 0;
        }
        if (strcmp(op, "csrgeam2") == 0) {
            return aclsparseScsrgeam2 != 0
                && aclsparseScsrgeam2_bufferSizeExt != 0
                && aclsparseXcsrgeam2Nnz != 0;
        }
        if (strcmp(op, "sparseToDense") == 0) {
            return aclsparseSparseToDense != 0
                && aclsparseSparseToDense_bufferSize != 0;
        }
        if (strcmp(op, "spgemmSetCInValid") == 0) {
            return aclsparseSpGEMMSetCInValid != 0;
        }
        return -1;
    }
    """
    int _aclsparse_optional_available(const char *op)


# ``aclrtMemcpy`` lives in ``acl/acl_rt.h``, which the vendored sparse header
# does not pull in (it only includes ``acl/acl_base_rt.h``), so declare it here
# instead of touching the vendored copy.
cdef extern from "acl/acl_rt.h" nogil:
    ctypedef int aclrtMemcpyKind
    aclError aclrtMemcpy(void *dst, size_t destMax, const void *src,
                         size_t count, aclrtMemcpyKind kind)
    int ACL_MEMCPY_HOST_TO_DEVICE


# ---------------------------------------------------------------------------
# cuSPARSE-compatible constants (values identical to the CUDA ones, which is
# what ``cupyx/cusparse.py`` relies on for its ``1 - flag`` arithmetic).
# ---------------------------------------------------------------------------
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1
CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

CUSPARSE_POINTER_MODE_HOST = 0
CUSPARSE_POINTER_MODE_DEVICE = 1

CUSPARSE_MATRIX_TYPE_GENERAL = 0
CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

CUSPARSE_FILL_MODE_LOWER = 0
CUSPARSE_FILL_MODE_UPPER = 1

CUSPARSE_DIAG_TYPE_NON_UNIT = 0
CUSPARSE_DIAG_TYPE_UNIT = 1

CUSPARSE_INDEX_BASE_ZERO = 0
CUSPARSE_INDEX_BASE_ONE = 1

CUSPARSE_INDEX_16U = 1
CUSPARSE_INDEX_32I = 2
CUSPARSE_INDEX_64I = 3

CUSPARSE_DIRECTION_ROW = 0
CUSPARSE_DIRECTION_COLUMN = 1

CUSPARSE_ORDER_ROW = 1
CUSPARSE_ORDER_COL = 0

CUSPARSE_ACTION_SYMBOLIC = 0
CUSPARSE_ACTION_NUMERIC = 1

CUSPARSE_CSR2CSC_ALG_DEFAULT = 0
CUSPARSE_CSR2CSC_ALG1 = 1

CUSPARSE_MV_ALG_DEFAULT = 0
CUSPARSE_MM_ALG_DEFAULT = 0
CUSPARSE_SPSM_ALG_DEFAULT = 0
CUSPARSE_SPGEMM_DEFAULT = 0
CUSPARSE_SPGEMM_ALG2 = 2

CUSPARSE_SPMAT_FILL_MODE = 0
CUSPARSE_SPMAT_DIAG_TYPE = 1

CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0
CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1

CUSPARSE_ALG_NAIVE = 0
CUSPARSE_ALG_MERGE_PATH = 1

CUSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0
CUSPARSE_SPARSETODENSE_ALG_DEFAULT = 0

#: aclDataType code -> name (for error messages only)
_ACL_DT_NAMES = {
    0: 'ACL_FLOAT', 1: 'ACL_FLOAT16', 2: 'ACL_INT8', 3: 'ACL_INT32',
    4: 'ACL_UINT8', 6: 'ACL_INT16', 7: 'ACL_UINT16', 8: 'ACL_UINT32',
    9: 'ACL_INT64', 10: 'ACL_UINT64', 11: 'ACL_DOUBLE', 12: 'ACL_BOOL',
    16: 'ACL_COMPLEX64', 17: 'ACL_COMPLEX128', 27: 'ACL_BF16',
    33: 'ACL_COMPLEX32',
}

_ACL_SPARSE_STATUS_NAMES = {
    0: 'SUCCESS', 1: 'NOT_INITIALIZED', 2: 'ALLOC_FAILED', 3: 'INVALID_VALUE',
    4: 'ARCH_MISMATCH', 5: 'EXECUTION_FAILED', 6: 'INTERNAL_ERROR',
    7: 'MATRIX_TYPE_NOT_SUPPORTED', 8: 'NOT_SUPPORTED',
    9: 'INSUFFICIENT_RESOURCES', 10: 'HANDLE_IS_NULLPTR',
}


class CuSparseError(RuntimeError):
    """Raised when an aclsparse call returns a non-success status.

    Named after the cuSPARSE exception because ``cupyx/cusparse.py`` catches
    ``_cusparse.CuSparseError`` in its ``spgemm`` fallback path.
    """

    def __init__(self, int status, str op_name):
        self.status = status
        self.errstr = _ACL_SPARSE_STATUS_NAMES.get(status, 'unknown')
        super().__init__(
            f'aclsparse error in {op_name}: {self.errstr} ({status})')


#: alias used by code that prefers the backend-neutral name
SparseError = CuSparseError


cdef inline int check_status(int status, str op_name) except -1:
    if status != ACL_SPARSE_STATUS_SUCCESS:
        raise CuSparseError(status, op_name)
    return 0


cdef inline aclsparseIndexType_t _to_acl_index_type(int index_type) except? (<aclsparseIndexType_t>255):
    """cuSPARSE ``CUSPARSE_INDEX_*`` -> ``ACL_SPARSE_INDEX_*``."""
    if index_type == CUSPARSE_INDEX_32I:
        return ACL_SPARSE_INDEX_32I
    elif index_type == CUSPARSE_INDEX_64I:
        return ACL_SPARSE_INDEX_64I
    else:
        raise NotImplementedError(
            'ops-sparse only supports 32/64-bit CSR/CSC/COO index arrays '
            f'(got CUSPARSE_INDEX_{index_type})')


cdef inline aclsparseOrder_t _to_acl_order(int order) except? (<aclsparseOrder_t>255):
    """cuSPARSE ``CUSPARSE_ORDER_*`` -> ``ACL_SPARSE_ORDER_*``."""
    if order == CUSPARSE_ORDER_COL:
        return ACL_SPARSE_ORDER_COL
    elif order == CUSPARSE_ORDER_ROW:
        return ACL_SPARSE_ORDER_ROW
    else:
        raise ValueError(f'unknown matrix order: {order}')


cdef inline void* _p(intptr_t ptr) noexcept nogil:
    return <void*>ptr


# ---------------------------------------------------------------------------
# Handle management: one aclsparseHandle_t per (device, current stream), with
# HOST pointer mode so that cupyx's host-side alpha/beta/nnz pointers work.
# ---------------------------------------------------------------------------
cdef dict _handle_cache = {}
cdef object _handle_lock = threading.Lock()


cdef intptr_t _create_handle(intptr_t stream) except? 0:
    cdef aclsparseHandle_t handle = NULL
    cdef int status
    with nogil:
        status = aclsparseCreate(&handle)
    if status != ACL_SPARSE_STATUS_SUCCESS:
        raise CuSparseError(status, 'aclsparseCreate')
    if handle == NULL:
        raise CuSparseError(1, 'aclsparseCreate (null handle)')
    # Bind the handle to the current stream; alpha/beta live on the host.
    # NOTE: aclsparseSetStream(handle, NULL) succeeds but leaves the handle's
    # stream null,breaking every following ops
    # skip the aclsparseSetStream() call for default stream (ptr == 0)
    # aclsparseCreate already binds a usable default stream
    if stream != 0:
        with nogil:
            status = aclsparseSetStream(handle, <aclrtStream>stream)
        if status != ACL_SPARSE_STATUS_SUCCESS:
            with nogil:
                aclsparseDestroy(handle)
            raise CuSparseError(status, 'aclsparseSetStream')
    with nogil:
        status = aclsparseSetPointerMode(handle,
                                         ACL_SPARSE_POINTER_MODE_HOST)
    if status != ACL_SPARSE_STATUS_SUCCESS:
        with nogil:
            aclsparseDestroy(handle)
        raise CuSparseError(status, 'aclsparseSetPointerMode')
    return <intptr_t>handle


def get_handle():
    """Return the aclsparse handle for the current (device, stream).

    Mirrors ``cupy.cuda.device.get_cusparse_handle()``: the same handle is
    reused for the same device *and* stream, because aclsparse (like
    cuSPARSE) records the stream inside the handle.
    """
    cdef int dev_id = runtime.getDevice()
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    key = (dev_id, stream)
    handle = _handle_cache.get(key)
    if handle is None:
        with _handle_lock:
            handle = _handle_cache.get(key)
            if handle is None:
                handle = _create_handle(stream)
                _handle_cache[key] = handle
    return handle


def destroy(intptr_t handle):
    """Destroy one cached handle (no-op for an unknown/null pointer)."""
    cdef int status
    if handle == 0:
        return
    with nogil:
        status = aclsparseDestroy(<aclsparseHandle_t>handle)
    check_status(status, 'aclsparseDestroy')
    for key, value in list(_handle_cache.items()):
        if value == handle:
            del _handle_cache[key]


def release_all():
    """Destroy every cached handle (called when cupy tears a device down)."""
    for key, handle in list(_handle_cache.items()):
        if handle:
            aclsparseDestroy(<aclsparseHandle_t>handle)
    _handle_cache.clear()


cdef void _bind_stream(intptr_t handle) except *:
    """Re-bind a handle to the current stream (mirrors cupy's ``_setStream``)."""
    cdef int status
    cdef intptr_t stream = stream_module.get_current_stream_ptr()
    # NOTE: aclsparseSetStream(hanle, NULL) will leave the handle's stream null
    if stream == 0:
        return # keep the handle's existing (default) stream
    with nogil:
        status = aclsparseSetStream(<aclsparseHandle_t>handle,
                                    <aclrtStream>stream)
    check_status(status, 'aclsparseSetStream')


# ---------------------------------------------------------------------------
# Version / availability
# ---------------------------------------------------------------------------
cpdef int getVersion(intptr_t handle) except? -1:
    """ops-sparse version, encoded as MAJOR * 10000 + MINOR * 100 + PATCH."""
    cdef int version = 0
    cdef int status
    with nogil:
        status = aclsparseGetVersion(<aclsparseHandle_t>handle, &version)
    check_status(status, 'aclsparseGetVersion')
    return version


def get_build_version():
    """ops-sparse build version (``NULL`` handle is allowed by the library).

    Encoded as ``MAJOR * 10000 + MINOR * 100 + PATCH`` (e.g. 10000 == 1.0.0),
    i.e. the same numeric scale ``cupyx.cusparse.check_availability`` compares
    against.
    """
    cdef int version = 0
    cdef int status
    with nogil:
        status = aclsparseGetVersion(NULL, &version)
    if status != ACL_SPARSE_STATUS_SUCCESS or version == 0:
        # Older ops-sparse builds may not implement the query; report the
        # minimum supported version rather than failing the whole module.
        return 10000
    return version


#: Which cuSPARSE entry points ops-sparse can serve.  ``cupyx/cusparse.py``
#: consults this through :func:`cupyx.cusparse.check_availability` on Ascend;
#: every ``False`` entry has a documented fallback in ``cupyx`` (usually the
#: generic descriptor API, which ops-sparse implements 1:1).  See
#: ``docs/ascend/ascend_sparse.md`` for the per-op rationale.
_available_aclsparse = {
    # --- legacy API: replaced by the generic paths in cupyx -----------------
    'csrmv': False,          # -> spmv
    'csrmm': False,          # -> spmm
    'csrmm2': False,         # -> spmm
    'csrgeam': False,        # -> csrgeam2
    'csrgemm': False,        # -> spgemm
    'csrgemm2': False,       # -> spgemm
    'csr2dense': False,      # -> sparseToDense
    'csc2dense': False,      # -> sparseToDense
    'dense2csr': False,      # -> denseToSparse
    'dense2csc': False,      # -> denseToSparse
    'csr2csc': False,        # -> csr2cscEx2
    'csc2csr': False,        # -> csc2csrEx2
    'gthr': False,           # -> gather (SpVec/DnVec descriptors)
    'csrsm2': False,         # -> spsm
    'csrilu02': False,       # not in ops-sparse (ILU(0))
    'csr2csr_compress': False,   # nnz_compress not in ops-sparse
    # --- generic API: implemented over aclsparse ----------------------------
    'csrmvEx': True,         # routed through aclsparseSpMV
    'spmv': True,
    'spmm': True,
    'spgemm': True,
    'spsm': True,
    'csrgeam2': True,        # aclsparseScsrgeam2, FP32 only
    'csrsort': True,
    'cscsort': True,
    'coosort': True,
    'coo2csr': True,
    'csr2coo': True,
    'csr2cscEx2': True,
    'csc2csrEx2': True,
    'denseToSparse': True,
    'sparseToDense': True,
}


#: Op groups whose symbols may be absent from a given libops_sparse.so build
#: (SoC-dependent operator set, or an explicit `build.sh --ops=` subset).
_OPTIONAL_GROUPS = (
    'cooGet', 'csr2cscEx2', 'csrsort', 'coosort', 'csr2coo', 'coo2csr',
    'csrgeam2', 'sparseToDense', 'spgemmSetCInValid',
)

#: Groups that share their implementation with another group.
_OPTIONAL_ALIASES = {
    # csc2csrEx2 is implemented on top of csr2cscEx2.
    'csr2cscEx2': ('csc2csrEx2',),
}

#: Groups whose implementation is missing from the loaded library.
_missing_optional = set()


def _probe_optional_ops():
    """Disable the ops whose symbols are absent from ``libops_sparse.so``.

    Called once at import time; relies on the weak declarations above, which
    make missing implementations NULL instead of an import-time
    "undefined symbol" error.  ``cupyx`` then falls back to its own
    implementations (see ``has_op``/``check_availability``).
    """
    for op in _OPTIONAL_GROUPS:
        if _aclsparse_optional_available(op.encode()) != 1:
            _missing_optional.add(op)
            for name in (op,) + _OPTIONAL_ALIASES.get(op, ()):
                if name in _available_aclsparse:
                    _available_aclsparse[name] = False


def has_op(name):
    """Whether the named cuSPARSE entry point is served by ops-sparse."""
    try:
        return _available_aclsparse[name]
    except KeyError:
        raise ValueError(
            f'No availability information specified for {name}') from None


cdef inline void _require_op(str op) except *:
    """Guard an entry point whose symbols may be missing from the library."""
    if op in _missing_optional:
        raise NotImplementedError(
            f'{op} is not provided by the installed libops_sparse.so '
            '(this ops-sparse build targets a different SoC or was built with '
            'a subset of operators, e.g. "build.sh --ops=..."). '
            'Use the non-cuSPARSE fallback instead.')


_probe_optional_ops()


# ---------------------------------------------------------------------------
# Legacy matrix descriptor
# ---------------------------------------------------------------------------
cpdef size_t createMatDescr() except? -1:
    cdef aclsparseMatDescr_t descr = NULL
    cdef int status
    with nogil:
        status = aclsparseCreateMatDescr(&descr)
    check_status(status, 'aclsparseCreateMatDescr')
    return <size_t>descr


cpdef void destroyMatDescr(size_t descr) except *:
    cdef int status
    with nogil:
        status = aclsparseDestroyMatDescr(<aclsparseMatDescr_t>descr)
    check_status(status, 'aclsparseDestroyMatDescr')


cpdef void setMatType(size_t descr, int typ) except *:
    aclsparseSetMatType(<aclsparseMatDescr_t>descr,
                        <aclsparseMatrixType_t>typ)


cpdef void setMatIndexBase(size_t descr, int base) except *:
    aclsparseSetMatIndexBase(<aclsparseMatDescr_t>descr,
                             <aclsparseIndexBase_t>base)


cpdef void setMatFillMode(size_t descr, int fill_mode) except *:
    aclsparseSetMatFillMode(<aclsparseMatDescr_t>descr,
                            <aclsparseFillMode_t>fill_mode)


cpdef void setMatDiagType(size_t descr, int diag_type) except *:
    aclsparseSetMatDiagType(<aclsparseMatDescr_t>descr,
                            <aclsparseDiagType_t>diag_type)


# ---------------------------------------------------------------------------
# Sparse matrix descriptors
# ---------------------------------------------------------------------------
cpdef size_t createCsr(int64_t rows, int64_t cols, int64_t nnz,
                       intptr_t csrRowOffsets, intptr_t csrColInd,
                       intptr_t csrValues, int csrRowOffsetsType,
                       int csrColIndType, int idxBase,
                       int valueType) except? -1:
    cdef aclsparseSpMatDescr_t desc = NULL
    cdef int status
    cdef aclsparseIndexType_t ptr_type = _to_acl_index_type(
        csrRowOffsetsType)
    cdef aclsparseIndexType_t idx_type = _to_acl_index_type(csrColIndType)
    with nogil:
        status = aclsparseCreateCsr(
            &desc, rows, cols, nnz, _p(csrRowOffsets), _p(csrColInd),
            _p(csrValues), ptr_type, idx_type,
            <aclsparseIndexBase_t>idxBase, <aclDataType>valueType)
    check_status(status, 'aclsparseCreateCsr')
    return <size_t>desc


cpdef size_t createCoo(int64_t rows, int64_t cols, int64_t nnz,
                       intptr_t cooRowInd, intptr_t cooColInd,
                       intptr_t cooValues, int cooIdxType, int idxBase,
                       int valueType) except? -1:
    cdef aclsparseSpMatDescr_t desc = NULL
    cdef int status
    cdef aclsparseIndexType_t idx_type = _to_acl_index_type(cooIdxType)
    with nogil:
        status = aclsparseCreateCoo(
            &desc, rows, cols, nnz, _p(cooRowInd), _p(cooColInd),
            _p(cooValues), idx_type, <aclsparseIndexBase_t>idxBase,
            <aclDataType>valueType)
    check_status(status, 'aclsparseCreateCoo')
    return <size_t>desc


cpdef size_t createCsc(int64_t rows, int64_t cols, int64_t nnz,
                       intptr_t cscColOffsets, intptr_t cscRowInd,
                       intptr_t cscValues, int cscColOffsetsType,
                       int cscRowIndType, int idxBase,
                       int valueType) except? -1:
    cdef aclsparseSpMatDescr_t desc = NULL
    cdef int status
    cdef aclsparseIndexType_t ptr_type = _to_acl_index_type(
        cscColOffsetsType)
    cdef aclsparseIndexType_t idx_type = _to_acl_index_type(cscRowIndType)
    with nogil:
        status = aclsparseCreateCsc(
            &desc, rows, cols, nnz, _p(cscColOffsets), _p(cscRowInd),
            _p(cscValues), ptr_type, idx_type,
            <aclsparseIndexBase_t>idxBase, <aclDataType>valueType)
    check_status(status, 'aclsparseCreateCsc')
    return <size_t>desc


cpdef void destroySpMat(size_t desc) except *:
    cdef int status
    with nogil:
        status = aclsparseDestroySpMat(<aclsparseConstSpMatDescr_t>desc)
    check_status(status, 'aclsparseDestroySpMat')


cpdef void csrSetPointers(size_t desc, intptr_t csrRowOffsets,
                          intptr_t csrColInd, intptr_t csrValues) except *:
    cdef int status
    with nogil:
        status = aclsparseCsrSetPointers(
            <aclsparseSpMatDescr_t>desc, _p(csrRowOffsets), _p(csrColInd),
            _p(csrValues))
    check_status(status, 'aclsparseCsrSetPointers')


cpdef void spMatGetSize(size_t desc, intptr_t rows, intptr_t cols,
                        intptr_t nnz) except *:
    cdef int status
    with nogil:
        status = aclsparseSpMatGetSize(
            <aclsparseConstSpMatDescr_t>desc, <int64_t*>rows, <int64_t*>cols,
            <int64_t*>nnz)
    check_status(status, 'aclsparseSpMatGetSize')


cpdef void spMatSetAttribute(size_t desc, int attribute, int data) except *:
    # cupyx passes the raw enum value (an int); aclsparse takes a pointer to
    # it plus its byte size, exactly like cusparseSpMatSetAttribute.
    cdef int status
    cdef int value = data
    with nogil:
        status = aclsparseSpMatSetAttribute(
            <aclsparseSpMatDescr_t>desc, <aclsparseSpMatAttribute_t>attribute,
            <const void*>&value, sizeof(int))
    check_status(status, 'aclsparseSpMatSetAttribute')


cpdef int spMatGetFormat(size_t desc) except? -1:
    cdef int status
    cdef aclsparseFormat_t fmt = <aclsparseFormat_t>0
    with nogil:
        status = aclsparseSpMatGetFormat(<aclsparseConstSpMatDescr_t>desc,
                                         &fmt)
    check_status(status, 'aclsparseSpMatGetFormat')
    return fmt


cpdef intptr_t spMatGetValues(size_t desc) except? 0:
    cdef int status
    cdef void* values = NULL
    with nogil:
        status = aclsparseSpMatGetValues(<aclsparseSpMatDescr_t>desc, &values)
    check_status(status, 'aclsparseSpMatGetValues')
    return <intptr_t>values


cpdef void spMatSetValues(size_t desc, intptr_t values) except *:
    cdef int status
    with nogil:
        status = aclsparseSpMatSetValues(<aclsparseSpMatDescr_t>desc,
                                         _p(values))
    check_status(status, 'aclsparseSpMatSetValues')


# --- descriptor getters (namedtuples mirror cupy's cusparse binding) --------
from collections import namedtuple as _namedtuple  # noqa: E402

CsrAttributes = _namedtuple(
    'CsrAttributes',
    ['rows', 'cols', 'nnz', 'rowOffsets', 'colInd', 'values',
     'rowOffsetsType', 'colIndType', 'idxBase', 'valueType'])
CooAttributes = _namedtuple(
    'CooAttributes',
    ['rows', 'cols', 'nnz', 'rowInd', 'colInd', 'values', 'idxType', 'idxBase',
     'valueType'])
SpVecAttributes = _namedtuple(
    'SpVecAttributes',
    ['size', 'nnz', 'indices', 'values', 'idxType', 'idxBase', 'valueType'])
DnVecAttributes = _namedtuple(
    'DnVecAttributes', ['size', 'values', 'valueType'])
DnMatAttributes = _namedtuple(
    'DnMatAttributes', ['rows', 'cols', 'ld', 'values', 'valueType', 'order'])


cpdef object cooGet(size_t desc):
    _require_op('cooGet')
    cdef int status
    cdef int64_t rows = 0, cols = 0, nnz = 0
    cdef void* rowInd = NULL
    cdef void* colInd = NULL
    cdef void* values = NULL
    cdef aclsparseIndexType_t idxType = <aclsparseIndexType_t>0
    cdef aclsparseIndexBase_t idxBase = <aclsparseIndexBase_t>0
    cdef aclDataType valueType = <aclDataType>0
    with nogil:
        status = aclsparseCooGet(<aclsparseSpMatDescr_t>desc, &rows, &cols,
                                 &nnz, &rowInd, &colInd, &values, &idxType,
                                 &idxBase, &valueType)
    check_status(status, 'aclsparseCooGet')
    return CooAttributes(rows, cols, nnz, <intptr_t>rowInd, <intptr_t>colInd,
                         <intptr_t>values, idxType, idxBase, valueType)


cpdef object csrGet(size_t desc):
    # ops-sparse has no aclsparseCsrGet; derive the fields from
    # SpMatGetSize plus the values pointer (the three CSR arrays are fixed by
    # createCsr, and cupyx only reads `.desc` attributes defensively).
    cdef int status
    cdef int64_t rows = 0, cols = 0, nnz = 0
    cdef void* values = NULL
    with nogil:
        status = aclsparseSpMatGetSize(<aclsparseConstSpMatDescr_t>desc,
                                       &rows, &cols, &nnz)
    check_status(status, 'aclsparseSpMatGetSize')
    with nogil:
        status = aclsparseSpMatGetValues(<aclsparseSpMatDescr_t>desc, &values)
    check_status(status, 'aclsparseSpMatGetValues')
    return CsrAttributes(rows, cols, nnz, 0, 0, <intptr_t>values,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, 0)


# ---------------------------------------------------------------------------
# Sparse / dense vector and dense matrix descriptors
# ---------------------------------------------------------------------------
cpdef size_t createSpVec(int64_t size, int64_t nnz, intptr_t indices,
                         intptr_t values, int idxType, int idxBase,
                         int valueType) except? -1:
    cdef aclsparseSpVecDescr_t desc = NULL
    cdef int status
    cdef aclsparseIndexType_t acl_idx_type = _to_acl_index_type(idxType)
    with nogil:
        status = aclsparseCreateSpVec(
            &desc, size, nnz, _p(indices), _p(values), acl_idx_type,
            <aclsparseIndexBase_t>idxBase, <aclDataType>valueType)
    check_status(status, 'aclsparseCreateSpVec')
    return <size_t>desc


cpdef void destroySpVec(size_t desc) except *:
    cdef int status
    with nogil:
        status = aclsparseDestroySpVec(<aclsparseConstSpVecDescr_t>desc)
    check_status(status, 'aclsparseDestroySpVec')


cpdef object spVecGet(size_t desc):
    cdef int status
    cdef int64_t size = 0, nnz = 0
    cdef void* indices = NULL
    cdef void* values = NULL
    cdef aclsparseIndexType_t idxType = <aclsparseIndexType_t>0
    cdef aclsparseIndexBase_t idxBase = <aclsparseIndexBase_t>0
    cdef aclDataType valueType = <aclDataType>0
    with nogil:
        status = aclsparseSpVecGet(<aclsparseSpVecDescr_t>desc, &size, &nnz,
                                   &indices, &values, &idxType, &idxBase,
                                   &valueType)
    check_status(status, 'aclsparseSpVecGet')
    return SpVecAttributes(size, nnz, <intptr_t>indices, <intptr_t>values,
                           idxType, idxBase, valueType)


cpdef size_t createDnVec(int64_t size, intptr_t values,
                         int valueType) except? -1:
    cdef aclsparseDnVecDescr_t desc = NULL
    cdef int status
    with nogil:
        status = aclsparseCreateDnVec(&desc, size, _p(values),
                                      <aclDataType>valueType)
    check_status(status, 'aclsparseCreateDnVec')
    return <size_t>desc


cpdef void destroyDnVec(size_t desc) except *:
    cdef int status
    with nogil:
        status = aclsparseDestroyDnVec(<aclsparseConstDnVecDescr_t>desc)
    check_status(status, 'aclsparseDestroyDnVec')


cpdef object dnVecGet(size_t desc):
    cdef int status
    cdef int64_t size = 0
    cdef void* values = NULL
    cdef aclDataType valueType = <aclDataType>0
    with nogil:
        status = aclsparseDnVecGet(<aclsparseDnVecDescr_t>desc, &size,
                                   &values, &valueType)
    check_status(status, 'aclsparseDnVecGet')
    return DnVecAttributes(size, <intptr_t>values, valueType)


cpdef size_t createDnMat(int64_t rows, int64_t cols, int64_t ld,
                         intptr_t values, int valueType,
                         int order) except? -1:
    cdef aclsparseDnMatDescr_t desc = NULL
    cdef int status
    cdef aclsparseOrder_t acl_order = _to_acl_order(order)
    with nogil:
        status = aclsparseCreateDnMat(&desc, rows, cols, ld, _p(values),
                                      <aclDataType>valueType, acl_order)
    check_status(status, 'aclsparseCreateDnMat')
    return <size_t>desc


cpdef void destroyDnMat(size_t desc) except *:
    cdef int status
    with nogil:
        status = aclsparseDestroyDnMat(<aclsparseConstDnMatDescr_t>desc)
    check_status(status, 'aclsparseDestroyDnMat')


cpdef object dnMatGet(size_t desc):
    cdef int status
    cdef int64_t rows = 0, cols = 0, ld = 0
    cdef void* values = NULL
    cdef aclDataType valueType = <aclDataType>0
    cdef aclsparseOrder_t order = <aclsparseOrder_t>0
    with nogil:
        status = aclsparseDnMatGet(<aclsparseDnMatDescr_t>desc, &rows, &cols,
                                   &ld, &values, &valueType, &order)
    check_status(status, 'aclsparseDnMatGet')
    # aclsparse ORDER_ROW == 0 / ORDER_COL == 1, cuSPARSE is the other way
    # round; translate back so callers see cuSPARSE values.
    cdef int cusparse_order = (
        CUSPARSE_ORDER_ROW if order == ACL_SPARSE_ORDER_ROW
        else CUSPARSE_ORDER_COL)
    return DnMatAttributes(rows, cols, ld, <intptr_t>values, valueType,
                           cusparse_order)


# ---------------------------------------------------------------------------
# Pointer mode / stream helpers used by cupyx
# ---------------------------------------------------------------------------
cpdef void setPointerMode(intptr_t handle, int mode) except *:
    cdef int status
    with nogil:
        status = aclsparseSetPointerMode(
            <aclsparseHandle_t>handle, <aclsparsePointerMode_t>mode)
    check_status(status, 'aclsparseSetPointerMode')


cpdef void setStream(intptr_t handle, intptr_t stream) except *:
    cdef int status
    with nogil:
        status = aclsparseSetStream(<aclsparseHandle_t>handle,
                                    <aclrtStream>stream)
    check_status(status, 'aclsparseSetStream')


# ---------------------------------------------------------------------------
# Generic ops: SpMV / SpMM
# ---------------------------------------------------------------------------
cpdef size_t spMV_bufferSize(intptr_t handle, int opA, intptr_t alpha,
                             size_t matA, size_t vecX, intptr_t beta,
                             size_t vecY, int computeType,
                             int alg) except? -1:
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpMVGetBufferSize(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <const void*>alpha, <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstDnVecDescr_t>vecX, <const void*>beta,
            <aclsparseDnVecDescr_t>vecY, <aclDataType>computeType,
            <aclsparseSpMVAlg_t>alg, &bufferSize)
    check_status(status, 'aclsparseSpMVGetBufferSize')
    return bufferSize


cpdef void spMV(intptr_t handle, int opA, intptr_t alpha, size_t matA,
                size_t vecX, intptr_t beta, size_t vecY, int computeType,
                int alg, intptr_t externalBuffer) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpMV(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <const void*>alpha, <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstDnVecDescr_t>vecX, <const void*>beta,
            <aclsparseDnVecDescr_t>vecY, <aclDataType>computeType,
            <aclsparseSpMVAlg_t>alg, _p(externalBuffer))
    check_status(status, 'aclsparseSpMV')


cpdef size_t spMM_bufferSize(intptr_t handle, int opA, int opB,
                             intptr_t alpha, size_t matA, size_t matB,
                             intptr_t beta, size_t matC, int computeType,
                             int alg) except? -1:
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpMMGetBufferSize(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstDnMatDescr_t>matB, <const void*>beta,
            <aclsparseDnMatDescr_t>matC, <aclDataType>computeType,
            <aclsparseSpMMAlg_t>alg, &bufferSize)
    check_status(status, 'aclsparseSpMMGetBufferSize')
    return bufferSize


cpdef void spMM(intptr_t handle, int opA, int opB, intptr_t alpha,
                size_t matA, size_t matB, intptr_t beta, size_t matC,
                int computeType, int alg, intptr_t externalBuffer) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpMM(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstDnMatDescr_t>matB, <const void*>beta,
            <aclsparseDnMatDescr_t>matC, <aclDataType>computeType,
            <aclsparseSpMMAlg_t>alg, _p(externalBuffer))
    check_status(status, 'aclsparseSpMM')


# ---------------------------------------------------------------------------
# Generic ops: SpSM (sparse triangular solve, matrix RHS)
# ---------------------------------------------------------------------------
cpdef size_t spSM_createDescr() except? -1:
    cdef aclsparseSpSMDescr_t descr = NULL
    cdef int status
    with nogil:
        status = aclsparseSpSMCreateDescr(&descr)
    check_status(status, 'aclsparseSpSMCreateDescr')
    return <size_t>descr


cpdef void spSM_destroyDescr(size_t descr) except *:
    cdef int status
    with nogil:
        status = aclsparseSpSMDestroyDescr(<aclsparseSpSMDescr_t>descr)
    check_status(status, 'aclsparseSpSMDestroyDescr')


cpdef size_t spSM_bufferSize(intptr_t handle, int opA, int opB,
                             intptr_t alpha, size_t matA, size_t matB,
                             size_t matC, int computeType, int alg,
                             size_t spsmDescr) except? -1:
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpSMBufferSize(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstDnMatDescr_t>matB, <aclsparseDnMatDescr_t>matC,
            <aclDataType>computeType, <aclsparseSpSMAlg_t>alg,
            <aclsparseSpSMDescr_t>spsmDescr, &bufferSize)
    check_status(status, 'aclsparseSpSMBufferSize')
    return bufferSize


cpdef void spSM_analysis(intptr_t handle, int opA, int opB, intptr_t alpha,
                         size_t matA, size_t matB, size_t matC,
                         int computeType, int alg, size_t spsmDescr,
                         intptr_t externalBuffer) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpSMAnalysis(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstDnMatDescr_t>matB, <aclsparseDnMatDescr_t>matC,
            <aclDataType>computeType, <aclsparseSpSMAlg_t>alg,
            <aclsparseSpSMDescr_t>spsmDescr, _p(externalBuffer))
    check_status(status, 'aclsparseSpSMAnalysis')


cpdef void spSM_solve(intptr_t handle, int opA, int opB, intptr_t alpha,
                      size_t matA, size_t matB, size_t matC, int computeType,
                      int alg, size_t spsmDescr,
                      intptr_t externalBuffer) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpSM(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstDnMatDescr_t>matB, <aclsparseDnMatDescr_t>matC,
            <aclDataType>computeType, <aclsparseSpSMAlg_t>alg,
            <aclsparseSpSMDescr_t>spsmDescr)
    check_status(status, 'aclsparseSpSM')


# ---------------------------------------------------------------------------
# Generic ops: SpGEMM (multi-stage, mirrors cusparseSpGEMM*)
# ---------------------------------------------------------------------------
cpdef size_t spGEMM_createDescr() except? -1:
    cdef aclsparseSpGEMMDescr_t descr = NULL
    cdef int status
    with nogil:
        status = aclsparseSpGEMMCreateDescr(&descr)
    check_status(status, 'aclsparseSpGEMMCreateDescr')
    # cupyx always builds C from scratch (beta == 0, empty matC), so tell the
    # library the C input is not valid data.  The entry point is optional:
    # some ops-sparse builds do not export it (weak symbol -> NULL).
    if _aclsparse_optional_available(b'spgemmSetCInValid') == 1:
        with nogil:
            status = aclsparseSpGEMMSetCInValid(descr, 0)
        check_status(status, 'aclsparseSpGEMMSetCInValid')
    return <size_t>descr


cpdef void spGEMM_destroyDescr(size_t descr) except *:
    cdef int status
    with nogil:
        status = aclsparseSpGEMMDestroyDescr(<aclsparseSpGEMMDescr_t>descr)
    check_status(status, 'aclsparseSpGEMMDestroyDescr')


cpdef size_t spGEMM_workEstimation(
        intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA,
        size_t matB, intptr_t beta, size_t matC, int computeType, int alg,
        size_t spgemmDescr, size_t bufferSize,
        intptr_t externalBuffer1) except? -1:
    cdef size_t bufferSize1 = bufferSize
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpGEMMWorkEstimation(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstSpMatDescr_t>matB, <const void*>beta,
            <aclsparseSpMatDescr_t>matC, <aclDataType>computeType,
            <aclsparseSpGEMMAlg_t>alg, <aclsparseSpGEMMDescr_t>spgemmDescr,
            &bufferSize1, _p(externalBuffer1))
    check_status(status, 'aclsparseSpGEMMWorkEstimation')
    return bufferSize1


cpdef size_t spGEMM_compute(
        intptr_t handle, int opA, int opB, intptr_t alpha, size_t matA,
        size_t matB, intptr_t beta, size_t matC, int computeType, int alg,
        size_t spgemmDescr, size_t bufferSize,
        intptr_t externalBuffer2) except? -1:
    cdef size_t bufferSize2 = bufferSize
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpGEMMCompute(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstSpMatDescr_t>matB, <const void*>beta,
            <aclsparseSpMatDescr_t>matC, <aclDataType>computeType,
            <aclsparseSpGEMMAlg_t>alg, <aclsparseSpGEMMDescr_t>spgemmDescr,
            &bufferSize2, _p(externalBuffer2))
    check_status(status, 'aclsparseSpGEMMCompute')
    return bufferSize2


cpdef void spGEMM_copy(intptr_t handle, int opA, int opB, intptr_t alpha,
                       size_t matA, size_t matB, intptr_t beta, size_t matC,
                       int computeType, int alg,
                       size_t spgemmDescr) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSpGEMMCopy(
            <aclsparseHandle_t>handle, <aclsparseOperation_t>opA,
            <aclsparseOperation_t>opB, <const void*>alpha,
            <aclsparseConstSpMatDescr_t>matA,
            <aclsparseConstSpMatDescr_t>matB, <const void*>beta,
            <aclsparseSpMatDescr_t>matC, <aclDataType>computeType,
            <aclsparseSpGEMMAlg_t>alg, <aclsparseSpGEMMDescr_t>spgemmDescr)
    check_status(status, 'aclsparseSpGEMMCopy')


# ---------------------------------------------------------------------------
# Generic ops: dense <-> sparse
# ---------------------------------------------------------------------------
cpdef size_t sparseToDense_bufferSize(intptr_t handle, size_t matA,
                                      size_t matB, int alg) except? -1:
    _require_op('sparseToDense')
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSparseToDense_bufferSize(
            <aclsparseHandle_t>handle, <aclsparseConstSpMatDescr_t>matA,
            <aclsparseDnMatDescr_t>matB,
            <aclsparseSparseToDenseAlg_t>alg, &bufferSize)
    check_status(status, 'aclsparseSparseToDense_bufferSize')
    return bufferSize


cpdef void sparseToDense(intptr_t handle, size_t matA, size_t matB, int alg,
                         intptr_t buffer) except *:
    _require_op('sparseToDense')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseSparseToDense(
            <aclsparseHandle_t>handle, <aclsparseConstSpMatDescr_t>matA,
            <aclsparseDnMatDescr_t>matB,
            <aclsparseSparseToDenseAlg_t>alg, _p(buffer))
    check_status(status, 'aclsparseSparseToDense')


cpdef size_t denseToSparse_bufferSize(intptr_t handle, size_t matA,
                                      size_t matB, int alg) except? -1:
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseDenseToSparseGetBufferSize(
            <aclsparseHandle_t>handle, <aclsparseConstDnMatDescr_t>matA,
            <aclsparseSpMatDescr_t>matB,
            <aclsparseDenseToSparseAlg_t>alg, &bufferSize)
    check_status(status, 'aclsparseDenseToSparseGetBufferSize')
    return bufferSize


cpdef void denseToSparse_analysis(intptr_t handle, size_t matA, size_t matB,
                                  int alg, intptr_t buffer) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseDenseToSparseAnalysis(
            <aclsparseHandle_t>handle, <aclsparseConstDnMatDescr_t>matA,
            <aclsparseSpMatDescr_t>matB,
            <aclsparseDenseToSparseAlg_t>alg, _p(buffer))
    check_status(status, 'aclsparseDenseToSparseAnalysis')


cpdef void denseToSparse_convert(intptr_t handle, size_t matA, size_t matB,
                                 int alg, intptr_t buffer) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseDenseToSparseConvert(
            <aclsparseHandle_t>handle, <aclsparseConstDnMatDescr_t>matA,
            <aclsparseSpMatDescr_t>matB,
            <aclsparseDenseToSparseAlg_t>alg, _p(buffer))
    check_status(status, 'aclsparseDenseToSparseConvert')


cpdef void gather(intptr_t handle, size_t vecY, size_t vecX) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseGather(<aclsparseHandle_t>handle,
                                 <aclsparseConstDnVecDescr_t>vecY,
                                 <aclsparseSpVecDescr_t>vecX)
    check_status(status, 'aclsparseGather')


# ---------------------------------------------------------------------------
# Format conversion: CSR <-> CSC (Ex2)
# ---------------------------------------------------------------------------
cpdef size_t csr2cscEx2_bufferSize(
        intptr_t handle, int m, int n, int nnz, intptr_t csrVal,
        intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal,
        intptr_t cscColPtr, intptr_t cscRowInd, int valType, int copyValues,
        int idxBase, int alg) except? -1:
    _require_op('csr2cscEx2')
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseCsr2cscEx2_bufferSize(
            <aclsparseHandle_t>handle, m, n, nnz, <const void*>csrVal,
            <const int*>csrRowPtr, <const int*>csrColInd, _p(cscVal),
            <int*>cscColPtr, <int*>cscRowInd, <aclDataType>valType,
            <aclsparseAction_t>copyValues, <aclsparseIndexBase_t>idxBase,
            <aclsparseCsr2CscAlg_t>alg, &bufferSize)
    check_status(status, 'aclsparseCsr2cscEx2_bufferSize')
    return bufferSize


cpdef void csr2cscEx2(
        intptr_t handle, int m, int n, int nnz, intptr_t csrVal,
        intptr_t csrRowPtr, intptr_t csrColInd, intptr_t cscVal,
        intptr_t cscColPtr, intptr_t cscRowInd, int valType, int copyValues,
        int idxBase, int alg, intptr_t buffer) except *:
    _require_op('csr2cscEx2')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseCsr2cscEx2(
            <aclsparseHandle_t>handle, m, n, nnz, <const void*>csrVal,
            <const int*>csrRowPtr, <const int*>csrColInd, _p(cscVal),
            <int*>cscColPtr, <int*>cscRowInd, <aclDataType>valType,
            <aclsparseAction_t>copyValues, <aclsparseIndexBase_t>idxBase,
            <aclsparseCsr2CscAlg_t>alg, _p(buffer))
    check_status(status, 'aclsparseCsr2cscEx2')


# ---------------------------------------------------------------------------
# Sorting (in-place, permutation returned in P)
# ---------------------------------------------------------------------------
cpdef size_t xcsrsort_bufferSizeExt(intptr_t handle, int m, int n, int nnz,
                                    intptr_t csrRowPtr,
                                    intptr_t csrColInd) except? -1:
    _require_op('csrsort')
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcsrsort_bufferSizeExt(
            <aclsparseHandle_t>handle, m, n, nnz, <const int*>csrRowPtr,
            <const int*>csrColInd, &bufferSize)
    check_status(status, 'aclsparseXcsrsort_bufferSizeExt')
    return bufferSize


cpdef void xcsrsort(intptr_t handle, int m, int n, int nnz, size_t descrA,
                    intptr_t csrRowPtr, intptr_t csrColInd, intptr_t P,
                    intptr_t pBuffer) except *:
    _require_op('csrsort')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcsrsort(
            <aclsparseHandle_t>handle, m, n, nnz,
            <const aclsparseMatDescr_t>descrA, <const int*>csrRowPtr,
            <int*>csrColInd, <int*>P, _p(pBuffer))
    check_status(status, 'aclsparseXcsrsort')


cpdef size_t xcscsort_bufferSizeExt(intptr_t handle, int m, int n, int nnz,
                                    intptr_t cscColPtr,
                                    intptr_t cscRowInd) except? -1:
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcscsort_bufferSizeExt(
            <aclsparseHandle_t>handle, m, n, nnz, <const int*>cscColPtr,
            <const int*>cscRowInd, &bufferSize)
    check_status(status, 'aclsparseXcscsort_bufferSizeExt')
    return bufferSize


cpdef void xcscsort(intptr_t handle, int m, int n, int nnz, size_t descrA,
                    intptr_t cscColPtr, intptr_t cscRowInd, intptr_t P,
                    intptr_t pBuffer) except *:
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcscsort(
            <aclsparseHandle_t>handle, m, n, nnz,
            <const aclsparseMatDescr_t>descrA, <const int*>cscColPtr,
            <int*>cscRowInd, <int*>P, _p(pBuffer))
    check_status(status, 'aclsparseXcscsort')


cpdef size_t xcoosort_bufferSizeExt(intptr_t handle, int m, int n, int nnz,
                                    intptr_t cooRows,
                                    intptr_t cooCols) except? -1:
    _require_op('coosort')
    cdef size_t bufferSize = 0
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcoosort_bufferSizeExt(
            <aclsparseHandle_t>handle, m, n, nnz, <const int*>cooRows,
            <const int*>cooCols, &bufferSize)
    check_status(status, 'aclsparseXcoosort_bufferSizeExt')
    return bufferSize


cpdef void xcoosortByRow(intptr_t handle, int m, int n, int nnz,
                         intptr_t cooRows, intptr_t cooCols, intptr_t P,
                         intptr_t pBuffer) except *:
    _require_op('coosort')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcoosortByRow(
            <aclsparseHandle_t>handle, m, n, nnz, <int*>cooRows,
            <int*>cooCols, <int*>P, _p(pBuffer))
    check_status(status, 'aclsparseXcoosortByRow')


cpdef void xcoosortByColumn(intptr_t handle, int m, int n, int nnz,
                            intptr_t cooRows, intptr_t cooCols, intptr_t P,
                            intptr_t pBuffer) except *:
    _require_op('coosort')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcoosortByColumn(
            <aclsparseHandle_t>handle, m, n, nnz, <int*>cooRows,
            <int*>cooCols, <int*>P, _p(pBuffer))
    check_status(status, 'aclsparseXcoosortByColumn')


# ---------------------------------------------------------------------------
# Format conversion: CSR <-> COO (int32 indices only)
# ---------------------------------------------------------------------------
cpdef void xcsr2coo(intptr_t handle, intptr_t csrRowPtr, int nnz, int m,
                    intptr_t cooRowInd, int idxBase) except *:
    _require_op('csr2coo')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcsr2coo(
            <aclsparseHandle_t>handle, <const int32_t*>csrRowPtr,
            <int64_t>nnz, <int64_t>m, <int32_t*>cooRowInd,
            <aclsparseIndexBase_t>idxBase)
    check_status(status, 'aclsparseXcsr2coo')


cpdef void xcoo2csr(intptr_t handle, intptr_t cooRowInd, int nnz, int m,
                    intptr_t csrRowPtr, int idxBase) except *:
    _require_op('coo2csr')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcoo2csr(
            <aclsparseHandle_t>handle, <const int*>cooRowInd, nnz, m,
            <int*>csrRowPtr, <aclsparseIndexBase_t>idxBase)
    check_status(status, 'aclsparseXcoo2csr')


cpdef void createIdentityPermutation(intptr_t handle, int n,
                                     intptr_t p) except *:
    """Fill ``p[0..n-1]`` with ``0, 1, ..., n-1`` on the device.

    ops-sparse has no ``cusparseCreateIdentityPermutation`` equivalent, but the
    identity permutation is a host constant, so a single H2D ``aclrtMemcpy``
    of a cached ``numpy.arange`` is exact and avoids a custom kernel.
    """
    cdef int status
    cdef intptr_t src
    if n <= 0:
        return
    ident = numpy.arange(n, dtype=numpy.int32)
    src = ident.ctypes.data
    with nogil:
        status = aclrtMemcpy(
            _p(p), <size_t>(n * 4), <const void*>src, <size_t>(n * 4),
            <aclrtMemcpyKind>ACL_MEMCPY_HOST_TO_DEVICE)
    if status != 0:
        raise CuSparseError(status, 'aclrtMemcpy(createIdentityPermutation)')


# ---------------------------------------------------------------------------
# Generic ops: csrgeam2 (C = alpha*A + beta*B).  ops-sparse implements only
# the FP32 variant (``aclsparseScsrgeam2``); the nnz stage is type-agnostic.
# ---------------------------------------------------------------------------
cpdef void xcsrgeam2Nnz(intptr_t handle, int m, int n, size_t descrA,
                        int nnzA, intptr_t csrRowPtrA, intptr_t csrColIndA,
                        size_t descrB, int nnzB, intptr_t csrRowPtrB,
                        intptr_t csrColIndB, size_t descrC,
                        intptr_t csrRowPtrC, intptr_t nnzTotalDevHostPtr,
                        intptr_t workspace) except *:
    _require_op('csrgeam2')
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcsrgeam2Nnz(
            <aclsparseHandle_t>handle, m, n,
            <const aclsparseMatDescr_t>descrA, nnzA,
            <const int*>csrRowPtrA, <const int*>csrColIndA,
            <const aclsparseMatDescr_t>descrB, nnzB,
            <const int*>csrRowPtrB, <const int*>csrColIndB,
            <const aclsparseMatDescr_t>descrC, <int*>csrRowPtrC,
            <int*>nnzTotalDevHostPtr, _p(workspace))
    check_status(status, 'aclsparseXcsrgeam2Nnz')


cpdef void xcsrgeamNnz(intptr_t handle, int m, int n, size_t descrA,
                       int nnzA, intptr_t csrRowPtrA, intptr_t csrColIndA,
                       size_t descrB, int nnzB, intptr_t csrRowPtrB,
                       intptr_t csrColIndB, size_t descrC,
                       intptr_t csrRowPtrC,
                       intptr_t nnzTotalDevHostPtr) except *:
    # Legacy name (no workspace); identical semantics on ops-sparse.
    cdef int status
    _bind_stream(handle)
    with nogil:
        status = aclsparseXcsrgeam2Nnz(
            <aclsparseHandle_t>handle, m, n,
            <const aclsparseMatDescr_t>descrA, nnzA,
            <const int*>csrRowPtrA, <const int*>csrColIndA,
            <const aclsparseMatDescr_t>descrB, nnzB,
            <const int*>csrRowPtrB, <const int*>csrColIndB,
            <const aclsparseMatDescr_t>descrC, <int*>csrRowPtrC,
            <int*>nnzTotalDevHostPtr, NULL)
    check_status(status, 'aclsparseXcsrgeam2Nnz')


cdef size_t _scsrgeam2_common(intptr_t handle, int m, int n, intptr_t alpha,
                              size_t descrA, int nnzA, intptr_t valA,
                              intptr_t ptrA, intptr_t indA, intptr_t beta,
                              size_t descrB, int nnzB, intptr_t valB,
                              intptr_t ptrB, intptr_t indB, size_t descrC,
                              intptr_t valC, intptr_t ptrC, intptr_t indC,
                              intptr_t buffer, str op_name,
                              bint query_only) except? -1:
    cdef int status
    cdef size_t bufferSize = 0
    _bind_stream(handle)
    if query_only:
        with nogil:
            status = aclsparseScsrgeam2_bufferSizeExt(
                <aclsparseHandle_t>handle, m, n, <const float*>alpha,
                <const aclsparseMatDescr_t>descrA, nnzA, <const float*>valA,
                <const int*>ptrA, <const int*>indA, <const float*>beta,
                <const aclsparseMatDescr_t>descrB, nnzB, <const float*>valB,
                <const int*>ptrB, <const int*>indB,
                <const aclsparseMatDescr_t>descrC, <const float*>valC,
                <const int*>ptrC, <const int*>indC, &bufferSize)
        check_status(status, op_name)
        return bufferSize
    else:
        with nogil:
            status = aclsparseScsrgeam2(
                <aclsparseHandle_t>handle, m, n, <const float*>alpha,
                <const aclsparseMatDescr_t>descrA, nnzA, <const float*>valA,
                <const int*>ptrA, <const int*>indA, <const float*>beta,
                <const aclsparseMatDescr_t>descrB, nnzB, <const float*>valB,
                <const int*>ptrB, <const int*>indB,
                <const aclsparseMatDescr_t>descrC, <float*>valC, <int*>ptrC,
                <int*>indC, _p(buffer))
        check_status(status, op_name)
        return 0


cpdef size_t scsrgeam2_bufferSizeExt(
        intptr_t handle, int m, int n, intptr_t alpha, size_t descrA,
        int nnzA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA,
        intptr_t beta, size_t descrB, int nnzB, intptr_t csrValB,
        intptr_t csrRowPtrB, intptr_t csrColIndB, size_t descrC,
        intptr_t csrValC, intptr_t csrRowPtrC,
        intptr_t csrColIndC) except? -1:
    _require_op('csrgeam2')
    return _scsrgeam2_common(
        handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
        beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC,
        csrRowPtrC, csrColIndC, 0, 'aclsparseScsrgeam2_bufferSizeExt', True)


cpdef void scsrgeam2(
        intptr_t handle, int m, int n, intptr_t alpha, size_t descrA,
        int nnzA, intptr_t csrValA, intptr_t csrRowPtrA, intptr_t csrColIndA,
        intptr_t beta, size_t descrB, int nnzB, intptr_t csrValB,
        intptr_t csrRowPtrB, intptr_t csrColIndB, size_t descrC,
        intptr_t csrValC, intptr_t csrRowPtrC, intptr_t csrColIndC,
        intptr_t buffer) except *:
    _require_op('csrgeam2')
    _scsrgeam2_common(
        handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
        beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC,
        csrRowPtrC, csrColIndC, buffer, 'aclsparseScsrgeam2', False)


# ---------------------------------------------------------------------------
# csrmvEx (legacy name, generic implementation): routed through aclsparseSpMV.
# The dtype arguments are aclDataType values on Ascend (to_cuda_dtype maps to
# numpy_dtype_to_acl_dtype), so they are forwarded verbatim.
# ---------------------------------------------------------------------------
cdef size_t _csrmvEx_common(intptr_t handle, int alg, int transA, int m,
                            int n, int nnz, intptr_t alpha, int alphatype,
                            size_t descrA, intptr_t csrValA, int csrValAtype,
                            intptr_t csrRowPtrA, intptr_t csrColIndA,
                            intptr_t x, int xtype, intptr_t beta,
                            int betatype, intptr_t y, int ytype,
                            int executiontype, intptr_t buffer, str op_name,
                            bint query_only) except? -1:
    cdef int status
    cdef size_t bufferSize = 0
    cdef aclsparseSpMatDescr_t matA = NULL
    cdef aclsparseDnVecDescr_t vecX = NULL
    cdef aclsparseDnVecDescr_t vecY = NULL
    # ops-sparse SpMV sizing (verified against test/spmv): with op(A) == T the
    # input vector has m elements and the output n, otherwise the reverse.
    cdef int64_t x_size = m if transA == CUSPARSE_OPERATION_TRANSPOSE else n
    cdef int64_t y_size = n if transA == CUSPARSE_OPERATION_TRANSPOSE else m

    _bind_stream(handle)
    status = aclsparseCreateCsr(
        &matA, m, n, nnz, _p(csrRowPtrA), _p(csrColIndA), _p(csrValA),
        ACL_SPARSE_INDEX_32I, ACL_SPARSE_INDEX_32I,
        ACL_SPARSE_INDEX_BASE_ZERO, <aclDataType>csrValAtype)
    if status != ACL_SPARSE_STATUS_SUCCESS:
        raise CuSparseError(status, op_name + ':aclsparseCreateCsr')
    try:
        status = aclsparseCreateDnVec(&vecX, x_size, _p(x), <aclDataType>xtype)
        if status != ACL_SPARSE_STATUS_SUCCESS:
            raise CuSparseError(status, op_name + ':aclsparseCreateDnVec(x)')
        status = aclsparseCreateDnVec(&vecY, y_size, _p(y), <aclDataType>ytype)
        if status != ACL_SPARSE_STATUS_SUCCESS:
            raise CuSparseError(status, op_name + ':aclsparseCreateDnVec(y)')
        try:
            if query_only:
                status = aclsparseSpMVGetBufferSize(
                    <aclsparseHandle_t>handle,
                    <aclsparseOperation_t>transA, <const void*>alpha, matA,
                    vecX, <const void*>beta, vecY,
                    <aclDataType>executiontype,
                    ACL_SPARSE_SPMV_ALG_DEFAULT, &bufferSize)
                check_status(status, op_name + ':aclsparseSpMVGetBufferSize')
            else:
                status = aclsparseSpMV(
                    <aclsparseHandle_t>handle,
                    <aclsparseOperation_t>transA, <const void*>alpha, matA,
                    vecX, <const void*>beta, vecY,
                    <aclDataType>executiontype, ACL_SPARSE_SPMV_ALG_DEFAULT,
                    _p(buffer))
                check_status(status, op_name + ':aclsparseSpMV')
        finally:
            if vecY != NULL:
                aclsparseDestroyDnVec(vecY)
            if vecX != NULL:
                aclsparseDestroyDnVec(vecX)
    finally:
        if matA != NULL:
            aclsparseDestroySpMat(matA)
    return bufferSize


cpdef size_t csrmvEx_bufferSize(
        intptr_t handle, int alg, int transA, int m, int n, int nnz,
        intptr_t alpha, int alphatype, size_t descrA, intptr_t csrValA,
        int csrValAtype, intptr_t csrRowPtrA, intptr_t csrColIndA,
        intptr_t x, int xtype, intptr_t beta, int betatype, intptr_t y,
        int ytype, int executiontype) except? -1:
    return _csrmvEx_common(
        handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA,
        csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y,
        ytype, executiontype, 0, 'csrmvEx_bufferSize', True)


cpdef void csrmvEx(
        intptr_t handle, int alg, int transA, int m, int n, int nnz,
        intptr_t alpha, int alphatype, size_t descrA, intptr_t csrValA,
        int csrValAtype, intptr_t csrRowPtrA, intptr_t csrColIndA,
        intptr_t x, int xtype, intptr_t beta, int betatype, intptr_t y,
        int ytype, int executiontype, intptr_t buffer) except *:
    _csrmvEx_common(
        handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA,
        csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y,
        ytype, executiontype, buffer, 'csrmvEx', False)


# ---------------------------------------------------------------------------
# Info-object stubs.  ops-sparse manages this state internally (its own
# descriptors), so the legacy info objects are inert placeholders; the legacy
# entry points that would consume them are reported unavailable.
# ---------------------------------------------------------------------------
cpdef size_t createCsrsm2Info() except? -1:
    return 0


cpdef void destroyCsrsm2Info(size_t info) except *:
    pass


cpdef size_t createCsrgemm2Info() except? -1:
    return 0


cpdef void destroyCsrgemm2Info(size_t info) except *:
    pass


cpdef size_t createCsrilu02Info() except? -1:
    return 0


cpdef void destroyCsrilu02Info(size_t info) except *:
    pass


# ---------------------------------------------------------------------------
# Legacy entry points ops-sparse does not implement.  They are registered as
# unavailable in :data:`_available_aclsparse` (so ``cupyx`` degrades to the
# generic API before ever calling them); defining them anyway turns an
# ``AttributeError`` into an actionable message if some code path calls them
# directly.
# ---------------------------------------------------------------------------
cdef class _UnavailableOp:
    """Callable that raises a clear error for an ops-sparse gap.

    A cdef class is used instead of a closure because ``__name__`` of a
    Cython-compiled function is read-only, so the usual
    ``func.__name__ = name`` trick does not work here.
    """

    cdef readonly object _op_name
    cdef readonly object _hint

    def __init__(self, name, hint):
        self._op_name = name
        self._hint = hint

    @property
    def __name__(self):
        return self._op_name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f'{self._op_name} is not available on the Ascend backend: '
            f'ops-sparse ({__name__}) does not implement it. {self._hint}')


def _make_unavailable(name, hint):
    return _UnavailableOp(name, hint)


_UNAVAILABLE_OPS = {
    'scsrmv': 'Use cusparse.spmv (generic API) instead.',
    'dcsrmv': 'Use cusparse.spmv (generic API) instead.',
    'ccsrmv': 'Use cusparse.spmv (generic API) instead.',
    'zcsrmv': 'Use cusparse.spmv (generic API) instead.',
    'scsrmm': 'Use cusparse.spmm (generic API) instead.',
    'dcsrmm': 'Use cusparse.spmm (generic API) instead.',
    'ccsrmm': 'Use cusparse.spmm (generic API) instead.',
    'zcsrmm': 'Use cusparse.spmm (generic API) instead.',
    'scsrmm2': 'Use cusparse.spmm (generic API) instead.',
    'dcsrmm2': 'Use cusparse.spmm (generic API) instead.',
    'ccsrmm2': 'Use cusparse.spmm (generic API) instead.',
    'zcsrmm2': 'Use cusparse.spmm (generic API) instead.',
    'scsrgeam': 'Use cusparse.csrgeam2 (FP32 only) instead.',
    'dcsrgeam': 'Use cusparse.csrgeam2 (FP32 only) instead.',
    'ccsrgeam': 'Use cusparse.csrgeam2 (FP32 only) instead.',
    'zcsrgeam': 'Use cusparse.csrgeam2 (FP32 only) instead.',
    'dcsrgeam2_bufferSizeExt': 'ops-sparse only implements the FP32 variant '
                               '(aclsparseScsrgeam2).',
    'ccsrgeam2_bufferSizeExt': 'ops-sparse only implements the FP32 variant '
                               '(aclsparseScsrgeam2).',
    'zcsrgeam2_bufferSizeExt': 'ops-sparse only implements the FP32 variant '
                               '(aclsparseScsrgeam2).',
    'dcsrgeam2': 'ops-sparse only implements the FP32 variant '
                 '(aclsparseScsrgeam2).',
    'ccsrgeam2': 'ops-sparse only implements the FP32 variant '
                 '(aclsparseScsrgeam2).',
    'zcsrgeam2': 'ops-sparse only implements the FP32 variant '
                 '(aclsparseScsrgeam2).',
    'scsrgemm': 'Use cusparse.spgemm (generic API) instead.',
    'dcsrgemm': 'Use cusparse.spgemm (generic API) instead.',
    'ccsrgemm': 'Use cusparse.spgemm (generic API) instead.',
    'zcsrgemm': 'Use cusparse.spgemm (generic API) instead.',
    'scsrgemm2': 'Use cusparse.spgemm (generic API) instead.',
    'dcsrgemm2': 'Use cusparse.spgemm (generic API) instead.',
    'ccsrgemm2': 'Use cusparse.spgemm (generic API) instead.',
    'zcsrgemm2': 'Use cusparse.spgemm (generic API) instead.',
    'scsrgemm2_bufferSizeExt': 'Use cusparse.spgemm (generic API) instead.',
    'dcsrgemm2_bufferSizeExt': 'Use cusparse.spgemm (generic API) instead.',
    'ccsrgemm2_bufferSizeExt': 'Use cusparse.spgemm (generic API) instead.',
    'zcsrgemm2_bufferSizeExt': 'Use cusparse.spgemm (generic API) instead.',
    'xcsrgemmNnz': 'Use cusparse.spgemm (generic API) instead.',
    'xcsrgemm2Nnz': 'Use cusparse.spgemm (generic API) instead.',
    'scsr2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'dcsr2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'ccsr2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'zcsr2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'scsc2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'dcsc2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'ccsc2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'zcsc2dense': 'Use cusparse.sparseToDense (generic API) instead.',
    'sdense2csr': 'Use cusparse.denseToSparse (generic API) instead.',
    'ddense2csr': 'Use cusparse.denseToSparse (generic API) instead.',
    'cdense2csr': 'Use cusparse.denseToSparse (generic API) instead.',
    'zdense2csr': 'Use cusparse.denseToSparse (generic API) instead.',
    'sdense2csc': 'Use cusparse.denseToSparse (generic API) instead.',
    'ddense2csc': 'Use cusparse.denseToSparse (generic API) instead.',
    'cdense2csc': 'Use cusparse.denseToSparse (generic API) instead.',
    'zdense2csc': 'Use cusparse.denseToSparse (generic API) instead.',
    'snnz': 'Use cusparse.denseToSparse (generic API) instead.',
    'dnnz': 'Use cusparse.denseToSparse (generic API) instead.',
    'cnnz': 'Use cusparse.denseToSparse (generic API) instead.',
    'znnz': 'Use cusparse.denseToSparse (generic API) instead.',
    'scsr2csc': 'Use cusparse.csr2cscEx2 (generic API) instead.',
    'dcsr2csc': 'Use cusparse.csr2cscEx2 (generic API) instead.',
    'ccsr2csc': 'Use cusparse.csr2cscEx2 (generic API) instead.',
    'zcsr2csc': 'Use cusparse.csr2cscEx2 (generic API) instead.',
    'sgthr': 'Use cusparse.gather with SpVec/DnVec descriptors instead.',
    'dgthr': 'Use cusparse.gather with SpVec/DnVec descriptors instead.',
    'cgthr': 'Use cusparse.gather with SpVec/DnVec descriptors instead.',
    'zgthr': 'Use cusparse.gather with SpVec/DnVec descriptors instead.',
    'scsr2csr_compress': 'nnz_compress is not implemented by ops-sparse.',
    'dcsr2csr_compress': 'nnz_compress is not implemented by ops-sparse.',
    'ccsr2csr_compress': 'nnz_compress is not implemented by ops-sparse.',
    'zcsr2csr_compress': 'nnz_compress is not implemented by ops-sparse.',
    'snnz_compress': 'nnz_compress is not implemented by ops-sparse.',
    'dnnz_compress': 'nnz_compress is not implemented by ops-sparse.',
    'cnnz_compress': 'nnz_compress is not implemented by ops-sparse.',
    'znnz_compress': 'nnz_compress is not implemented by ops-sparse.',
    'scsrilu02': 'ILU(0) is not implemented by ops-sparse; use '
                 'spilu(fill_factor != 1) (host scipy fallback).',
    'dcsrilu02': 'ILU(0) is not implemented by ops-sparse; use '
                 'spilu(fill_factor != 1) (host scipy fallback).',
    'ccsrilu02': 'ILU(0) is not implemented by ops-sparse; use '
                 'spilu(fill_factor != 1) (host scipy fallback).',
    'zcsrilu02': 'ILU(0) is not implemented by ops-sparse; use '
                 'spilu(fill_factor != 1) (host scipy fallback).',
    'scsrsm2_bufferSizeExt': 'Use cusparse.spsm (generic API) instead.',
    'dcsrsm2_bufferSizeExt': 'Use cusparse.spsm (generic API) instead.',
    'ccsrsm2_bufferSizeExt': 'Use cusparse.spsm (generic API) instead.',
    'zcsrsm2_bufferSizeExt': 'Use cusparse.spsm (generic API) instead.',
    'scsrsm2_analysis': 'Use cusparse.spsm (generic API) instead.',
    'dcsrsm2_analysis': 'Use cusparse.spsm (generic API) instead.',
    'ccsrsm2_analysis': 'Use cusparse.spsm (generic API) instead.',
    'zcsrsm2_analysis': 'Use cusparse.spsm (generic API) instead.',
    'scsrsm2_solve': 'Use cusparse.spsm (generic API) instead.',
    'dcsrsm2_solve': 'Use cusparse.spsm (generic API) instead.',
    'ccsrsm2_solve': 'Use cusparse.spsm (generic API) instead.',
    'zcsrsm2_solve': 'Use cusparse.spsm (generic API) instead.',
}

for _name, _hint in _UNAVAILABLE_OPS.items():
    globals()[_name] = _make_unavailable(_name, _hint)
