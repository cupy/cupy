/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef CANN_OPS_SPARSE_H_
#define CANN_OPS_SPARSE_H_

#include <unistd.h>
#include <stdint.h>
#include <acl/acl_base_rt.h>

#ifdef __cplusplus
extern "C" {
#endif
// This type indicates which operations is applied to the related input (e.g. sparse matrix, or vector).
typedef enum aclsparseOperation_t {
    ACL_SPARSE_OP_NON_TRANSPOSE = 0,   // The non-transpose operation is selected.
    ACL_SPARSE_OP_TRANSPOSE,           // The transpose operation is selected.
    ACL_SPARSE_OP_CONJUGATE_TRANSPOSE  // The conjugate transpose operation is selected.
} aclsparseOperation_t;

struct aclsparseContext;
struct aclsparseSpMatDescr;
struct aclsparseDnVecDescr;
struct aclsparseDnMatDescr;
struct aclsparseCubeSpmmMatDescr;
struct aclsparseSpVecDescr;
struct aclsparseSpSMDescr;
struct aclsparseSpGEMMDescr;

typedef struct aclsparseContext* aclsparseHandle_t;
typedef struct aclsparseSpMatDescr* aclsparseSpMatDescr_t;
typedef struct aclsparseDnVecDescr* aclsparseDnVecDescr_t;
typedef struct aclsparseDnMatDescr* aclsparseDnMatDescr_t;
typedef struct aclsparseCubeSpmmMatDescr* aclsparseCubeSpmmMatDescr_t;
typedef struct aclsparseSpVecDescr* aclsparseSpVecDescr_t;
typedef struct aclsparseSpSMDescr* aclsparseSpSMDescr_t;
typedef struct aclsparseSpGEMMDescr* aclsparseSpGEMMDescr_t;

typedef struct aclsparseSpMatDescr const* aclsparseConstSpMatDescr_t;
typedef struct aclsparseDnVecDescr const* aclsparseConstDnVecDescr_t;
typedef struct aclsparseDnMatDescr const* aclsparseConstDnMatDescr_t;
typedef struct aclsparseCubeSpmmMatDescr const* aclsparseConstCubeSpmmMatDescr_t;
typedef struct aclsparseSpVecDescr const* aclsparseConstSpVecDescr_t;

// Dense matrix data layout (used by B / C of SpMM).
typedef enum aclsparseOrder_t {
    ACL_SPARSE_ORDER_ROW = 0,   // Row-major.
    ACL_SPARSE_ORDER_COL = 1,   // Column-major.
} aclsparseOrder_t;

// SpMVOp 算法枚举（对标 Generic API 中 SpMVOp 的算法选项）
typedef enum aclsparseSpMVOpAlg_t {
    // Default algorithm; routes to ALG1. No workspace required.
    // Supports updating matA's values without rebuilding descr/plan.
    ACL_SPARSE_SPMVOP_ALG_DEFAULT = 0,
    // Deterministic algorithm; static row-level slicing, lightweight createDescr.
    // No workspace required. Supports updating matA's values (csrValues)
    // without rebuilding descr/plan.
    ACL_SPARSE_SPMVOP_ALG1,
    // Deterministic algorithm; nnz-balanced load distribution via device-side
    // reorder in createDescr. Requires workspace (see bufferSize).
    // WARNING: in-place value updates (modifying csrValues after createDescr)
    // invalidate the reorder/bin_edge tables; use ALG1 if in-place updates needed.
    // NOTE: ALG2 不提供运行时检测，用户需自行保证不在 ALG2 descr 上修改 csrValues。
    ACL_SPARSE_SPMVOP_ALG2
} aclsparseSpMVOpAlg_t;

// SpMVOp opaque types
struct aclsparseSpMVOpDescr;
struct aclsparseSpMVOpPlan;

typedef struct aclsparseSpMVOpDescr* aclsparseSpMVOpDescr_t;
typedef struct aclsparseSpMVOpDescr const* aclsparseConstSpMVOpDescr_t;
typedef struct aclsparseSpMVOpPlan* aclsparseSpMVOpPlan_t;

// Algorithm enum for SpMM.
typedef enum aclsparseSpMMAlg_t {
    ACL_SPARSE_SPMM_ALG_DEFAULT = 0,
    // Recommended entry for any sparse matrix format. V1 routes CSR to the same
    // SIMT/AIV implementation as ACL_SPARSE_SPMM_CSR_ALG1.
    ACL_SPARSE_SPMM_CSR_ALG1,
    // Explicit CSR algorithm 1. V1 uses the same SIMT/AIV implementation as
    // ACL_SPARSE_SPMM_ALG_DEFAULT.
    ACL_SPARSE_SPMM_CSR_FP32_HIGH_PRECISION_ALG,
    // fp32 high-precision algorithm. Same SIMT path as the default, but the fp32
    // accumulation uses Kahan compensated summation to suppress rounding /
    // cancellation error on long rows. Trades throughput for accuracy and is only
    // meaningful for fp32 (silently ignored for fp16 / int8). The other algorithms
    // remain the high-performance choice with standard (lower) fp32 precision.
    // ACL_SPARSE_SPMM_COO_ALG1,  // Reserved: COO support (V2).
} aclsparseSpMMAlg_t;

// Algorithm selection for the multi-stage sparse matrix-matrix product.
// DEFAULT and ALG1 use the deterministic full-product workspace path. ALG2
// and ALG3 retain the same public staging contract and accept a chunkFraction
// through aclsparseSpGEMMEstimateMemory.
typedef enum aclsparseSpGEMMAlg_t {
    ACL_SPARSE_SPGEMM_DEFAULT = 0,
    ACL_SPARSE_SPGEMM_ALG_DEFAULT = ACL_SPARSE_SPGEMM_DEFAULT,
    ACL_SPARSE_SPGEMM_ALG1 = 1,
    ACL_SPARSE_SPGEMM_ALG2 = 2,
    ACL_SPARSE_SPGEMM_ALG3 = 3
} aclsparseSpGEMMAlg_t;

// This data type represents the status returned by the library functions and it can have the following values:
typedef enum aclsparseStatus_t {
    ACL_SPARSE_STATUS_SUCCESS = 0,
    // The operation completed successfully
    ACL_SPARSE_STATUS_NOT_INITIALIZED,
    // The aclSPARSE library was not initialized. This is usually caused by the
    // lack of a prior call, an error in the Ascend Runtime API called by the
    // aclSPARSE routine, or an error in the hardware setup To correct: call
    // aclsparseCreate() prior to the function call; and check that the
    // hardware, an appropriate version of the driver, and the aclSPARSE library
    // are correctly installed The error also applies to generic APIs (aclSPARSE
    // Generic APIs) for indicating a matrix/vector descriptor not initialized
    ACL_SPARSE_STATUS_ALLOC_FAILED,
    // Resource allocation failed inside the aclSPARSE library. This is usually
    // caused by a device memory allocation (aclrtMalloc()) or by a host memory
    // allocation failure To correct: prior to the function call, deallocate
    // previously allocated memory as much as possible
    ACL_SPARSE_STATUS_INVALID_VALUE,
    // An unsupported value or parameter was passed to the function (a negative
    // vector size, for example) To correct: ensure that all the parameters being
    // passed have valid values
    ACL_SPARSE_STATUS_ARCH_MISMATCH,
    // The function requires a feature absent from the device architecture
    // To correct: compile and run the application on a device with appropriate
    // compute capability
    ACL_SPARSE_STATUS_EXECUTION_FAILED,
    // The NPU program failed to execute. This is often caused by a launch
    // failure of the kernel on the NPU, which can be caused by multiple
    // reasons To correct: check that the hardware, an appropriate version of
    // the driver, and the aclSPARSE library are correctly installed
    ACL_SPARSE_STATUS_INTERNAL_ERROR,
    // An internal aclSPARSE operation failed
    // To correct: check that the hardware, an appropriate version of the driver,
    // and the aclSPARSE library are correctly installed. Also, check that the
    // memory passed as a parameter to the routine is not being deallocated prior
    // to the routine completion
    ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
    // The matrix type is not supported by this function. This is
    // usually caused by passing an invalid matrix descriptor to the
    // function To correct: check that the fields in
    // aclsparseSpMatDescr_t descrA were set correctly
    ACL_SPARSE_STATUS_NOT_SUPPORTED,
    // The operation or data type combination is currently not supported by the
    // function
    ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES,
    // The resources for the computation, such as NPU global or shared
    // memory, are not sufficient to complete the operation. The error can
    // also indicate that the current computation mode (e.g. bit size of
    // sparse matrix indices) does not allow to handle the given input
    ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR
    // The handle is a null pointer
} aclsparseStatus_t;

typedef enum aclsparseSpMVAlg_t {
    ACL_SPARSE_SPMV_ALG_DEFAULT,
    // Default algorithm for any sparse matrix format.
    ACL_SPARSE_SPMV_COO_ALG1,
    // Default algorithm for COO sparse matrix format. May produce slightly different results
    // during different runs with the same input parameters.
    ACL_SPARSE_SPMV_COO_ALG2,
    // Provides deterministic (bit-wise) results for each run. If opA !=
    // ACL_SPARSE_OPERATION_NON_TRANSPOSE, it is identical to ACL_SPARSE_SPMV_COO_ALG1.
    ACL_SPARSE_SPMV_CSR_ALG1,
    // Default algorithm for CSR/CSC sparse matrix format. May produce slightly different
    // results during different runs with the same input parameters.
    ACL_SPARSE_SPMV_CSR_ALG2,
    // Provides deterministic (bit-wise) results for each run. If opA !=
    // ACL_SPARSE_OPERATION_NON_TRANSPOSE, it is identical to ACL_SPARSE_SPMV_CSR_ALG1.
    ACL_SPARSE_SPMV_SELL_ALG1,
    // Default algorithm for Sliced Ellpack sparse matrix format. Provides deterministic
    // (bit-wise) results for each run.
    ACL_SPARSE_SPMV_CSR_ALG_CUBE,
    // Cube (matrix unit) based segmented-sum algorithm for CSR sparse matrix format.
    // Requires opA == ACL_SPARSE_OP_NON_TRANSPOSE, a float32 compute type and a float32
    // output vector; the matrix values and the input vector are either float32 or
    // float16 (float16 inputs are accumulated in float32 by the cube unit). May produce
    // slightly different results than the vector-unit algorithms because the reduction
    // order differs. Reference: Algorithm 4.4 in https://arxiv.org/pdf/2506.23906
} aclsparseSpMVAlg_t;

// Complex types (aligned with cuComplex / cuDoubleComplex).
// Layout: { real, imag }, binary-compatible with float2 / double2.
typedef struct aclsparseComplex {
    float x;
    float y;
} aclsparseComplex;

typedef struct aclsparseDoubleComplex {
    double x;
    double y;
} aclsparseDoubleComplex;

// Computation direction (used by Legacy API, e.g. aclsparseSnnz).
typedef enum aclsparseDirection_t {
    ACL_SPARSE_DIRECTION_ROW = 0,
    ACL_SPARSE_DIRECTION_COLUMN = 1
} aclsparseDirection_t;

// Forward declaration for Legacy matrix descriptor.
struct aclsparseMatDescr;
typedef struct aclsparseMatDescr *aclsparseMatDescr_t;

// Matrix type for Legacy MatDescr.
typedef enum aclsparseMatrixType_t {
    ACL_SPARSE_MATRIX_TYPE_GENERAL = 0,
    ACL_SPARSE_MATRIX_TYPE_SYMMETRIC,
    ACL_SPARSE_MATRIX_TYPE_HERMITIAN,
    ACL_SPARSE_MATRIX_TYPE_TRIANGULAR
} aclsparseMatrixType_t;

// Diagonal type for Legacy MatDescr.
typedef enum aclsparseDiagType_t {
    ACL_SPARSE_DIAG_TYPE_NON_UNIT = 0,
    ACL_SPARSE_DIAG_TYPE_UNIT
} aclsparseDiagType_t;

// Fill mode for Legacy MatDescr.
typedef enum aclsparseFillMode_t {
    ACL_SPARSE_FILL_MODE_LOWER = 0,
    ACL_SPARSE_FILL_MODE_UPPER
} aclsparseFillMode_t;

// This type indicates the index type for representing the sparse matrix indices.
typedef enum aclsparseIndexType_t {
    ACL_SPARSE_INDEX_32I = 0,  // 32-bit signed integer [0, 2^31 - 1]
    ACL_SPARSE_INDEX_64I       // 64-bit signed integer [0, 2^63 - 1]
                               // Per-operator I64 support:
                               //   SpSV  — fully supported; rowPtr (ptrType) and colInd (idxType) may
                               //           be mixed: allowed (I32,I32), (I64,I32), (I64,I64).
                               //           (I32,I64) is rejected (no practical use case).
                               //   SpMV  — I64 not yet supported (returns NOT_SUPPORTED).
                               //   SpMM  — I64 supported for rowOffsets (ptrType); colInd (idxType) remains I32 only.
} aclsparseIndexType_t;

typedef enum aclsparseDenseToSparseAlg_t {
    ACL_SPARSE_DENSETOSPARSE_ALG_DEFAULT = 0
} aclsparseDenseToSparseAlg_t;

// This type indicates if the base of the matrix indices is zero or one.
typedef enum aclsparseIndexBase_t {
    ACL_SPARSE_INDEX_BASE_ZERO = 0,  // The base index is zero (C compatibility).
    ACL_SPARSE_INDEX_BASE_ONE        // The base index is one (Fortran compatibility).
} aclsparseIndexBase_t;

// This type indicates the format of the sparse matrix.
typedef enum aclsparseFormat_t {
    ACL_SPARSE_FORMAT_COO = 0,
    // The matrix is stored in Coordinate (COO) format organized in Structure of Arrays (SoA) layout
    ACL_SPARSE_FORMAT_CSR,
    // The matrix is stored in Compressed Sparse Row (CSR) format
    ACL_SPARSE_FORMAT_CSC,
    // The matrix is stored in Compressed Sparse Column (CSC) format
    ACL_SPARSE_FORMAT_BLOCKED_ELL,
    // The matrix is stored in Blocked-Ellpack (Blocked-ELL) format
    ACL_SPARSE_FORMAT_SLICED_ELL,
    // The matrix is stored in Sliced-Ellpack (Sliced-ELL) format
    ACL_SPARSE_FORMAT_BSR
    // The matrix is stored in Block Sparse Row (BSR) format
} aclsparseFormat_t;

/**
 * @brief 创建一个稠密向量。
 *
 * @param dnVecDescr IN/OUT, HOST, 稀疏向量描述符。
 * @param size IN, HOST, 向量的大小。
 * @param values IN, DEVICE, 向量的值。
 * @param valueType IN, HOST, 值的数据类型。
 * @return aclsparseStatus_t 返回稀疏操作的状态。
 */
aclsparseStatus_t aclsparseCreateDnVec(aclsparseDnVecDescr_t *dnVecDescr, int64_t size, void *values, aclDataType valueType);

/**
 * @brief 创建只读(const)稠密向量描述符。数据指针为 const。
 */
aclsparseStatus_t aclsparseCreateConstDnVec(aclsparseConstDnVecDescr_t *dnVecDescr, int64_t size,
    const void *values, aclDataType valueType);

/**
 * @brief 销毁稀疏向量描述符。
 *
 * @param dnVecDescr IN, HOST, 要销毁的稀疏向量描述符。
 * @return aclsparseStatus_t 返回执行状态，成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseDestroyDnVec(aclsparseConstDnVecDescr_t dnVecDescr);

// ============================================================================
// Sparse Vector APIs
// ============================================================================

aclsparseStatus_t aclsparseCreateSpVec(aclsparseSpVecDescr_t *spVecDescr, int64_t size,
    int64_t nnz, void *indices, void *values, aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstSpVec(aclsparseConstSpVecDescr_t *spVecDescr, int64_t size,
    int64_t nnz, const void *indices, const void *values, aclsparseIndexType_t idxType,
    aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseDestroySpVec(aclsparseConstSpVecDescr_t spVecDescr);

aclsparseStatus_t aclsparseSpVecGet(aclsparseSpVecDescr_t spVecDescr, int64_t *size,
    int64_t *nnz, void **indices, void **values, aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);

aclsparseStatus_t aclsparseConstSpVecGet(aclsparseConstSpVecDescr_t spVecDescr, int64_t *size,
    int64_t *nnz, const void **indices, const void **values, aclsparseIndexType_t *idxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);

aclsparseStatus_t aclsparseSpVecGetIndexBase(aclsparseConstSpVecDescr_t spVecDescr,
    aclsparseIndexBase_t *idxBase);

aclsparseStatus_t aclsparseSpVecGetValues(aclsparseSpVecDescr_t spVecDescr, void **values);

aclsparseStatus_t aclsparseConstSpVecGetValues(aclsparseConstSpVecDescr_t spVecDescr,
    const void **values);

aclsparseStatus_t aclsparseSpVecSetValues(aclsparseSpVecDescr_t spVecDescr, void *values);

// ============================================================================
// Generic API: aclsparseGather
// ============================================================================

/**
 * @brief Gather elements from dense vector Y into sparse vector X values.
 *
 * X.values[i] = Y[X.indices[i] - idxBase] for i = 0 .. nnz-1.
 *
 * @param handle IN, HOST, aclsparse handle.
 * @param vecY   IN, HOST, const dense vector descriptor.
 * @param vecX   IN/OUT, HOST, sparse vector descriptor.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseGather(
    aclsparseHandle_t handle,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseSpVecDescr_t vecX);

/**
 * @brief Gather 裸指针快路径（跳过 DnVec/SpVec 描述符建销）。
 *
 * 与 aclsparseGather 算法/精度完全等价，但调用方直接传入设备指针，避免每次调用
 * 的描述符创建/销毁开销（用于 Python/ATen 等高频、µs 级小 gather 场景，降低
 * 端到端 host 开销）。indices 固定 int32，idxBase 仅 ZERO/ONE。
 *
 * @param handle   IN, HOST, aclsparse handle。
 * @param y_ptr    IN, DEVICE, 稠密源向量 Y 的设备指针（只读）。
 * @param y_len    IN, HOST,  Y 的元素个数。
 * @param valueType IN, HOST, 值的数据类型（f16/bf16/f32/c64）。
 * @param idx_ptr  IN, DEVICE, int32 索引向量设备指针（只读）。
 * @param out_ptr  OUT, DEVICE, 输出 X.values 设备指针（可写，nnz 个元素）。
 * @param nnz      IN, HOST,  输出元素个数。
 * @param idxBase  IN, HOST,  索引基（ZERO=0 / ONE=1）。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseGatherRaw(
    aclsparseHandle_t handle,
    const void *y_ptr, int64_t y_len, aclDataType valueType,
    const void *idx_ptr, void *out_ptr, int64_t nnz,
    aclsparseIndexBase_t idxBase);

/**
 * @brief 获取稠密向量描述符的全部字段。
 *
 * @param dnVecDescr IN, HOST, 稠密向量描述符。
 * @param size OUT, HOST, 向量的大小。
 * @param values OUT, DEVICE, 向量的值指针。
 * @param valueType OUT, HOST, 值的数据类型。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseDnVecGet(aclsparseDnVecDescr_t dnVecDescr, int64_t *size,
                                    void **values, aclDataType *valueType);

/**
 * @brief 获取只读(const)稠密向量描述符的全部字段。
 */
aclsparseStatus_t aclsparseConstDnVecGet(aclsparseConstDnVecDescr_t dnVecDescr, int64_t *size,
                                         const void **values, aclDataType *valueType);

/**
 * @brief 获取稠密向量描述符的 values 指针。
 *
 * @param dnVecDescr IN, HOST, 稠密向量描述符。
 * @param values OUT, DEVICE, 向量的值指针。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseDnVecGetValues(aclsparseDnVecDescr_t dnVecDescr, void **values);

/**
 * @brief 获取只读(const)稠密向量描述符的 values 指针。
 */
aclsparseStatus_t aclsparseConstDnVecGetValues(aclsparseConstDnVecDescr_t dnVecDescr,
                                               const void **values);

/**
 * @brief 设置稠密向量描述符的 values 指针。
 *
 * @param dnVecDescr IN, HOST, 稠密向量描述符。
 * @param values IN, DEVICE, 向量的值指针。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseDnVecSetValues(aclsparseDnVecDescr_t dnVecDescr, void *values);

/**
 * @brief 设置稠密向量的元素步长。
 *
 * @param dnVecDescr IN, HOST, 稠密向量描述符。
 * @param stride IN, HOST, 相邻逻辑元素之间的物理元素间隔，必须大于0。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseDnVecSetStride(aclsparseDnVecDescr_t dnVecDescr, int64_t stride);

/**
 * @brief 获取稠密向量的元素步长。
 */
aclsparseStatus_t aclsparseDnVecGetStride(aclsparseConstDnVecDescr_t dnVecDescr, int64_t *stride);

/**
 * @brief 创建稀疏矩阵的CSR格式。
 *
 * 该函数用于创建一个稀疏矩阵的CSR（Compressed Sparse Row）格式。
 * CSR格式通过三个数组来表示稀疏矩阵：行偏移量数组、列索引数组和非零元素数组。
 *
 * @param spMatDescr IN/OUT, HOST, 稀疏矩阵描述符。
 *                      用于描述稀疏矩阵的属性和数据。
 * @param rows IN, HOST, 矩阵的行数，必须为非负数。
 * @param cols IN, HOST, 矩阵的列数，必须为非负数。
 * @param nnz IN, HOST, 矩阵的非零元素个数，必须为非负数。
 * @param csrRowOffsets IN, DEVICE, 指向CSR格式行偏移量数组的指针。
 *                         该数组的长度为rows + 1，用于表示每行的非零元素起始位置。
 * @param csrColInd IN, DEVICE, 指向CSR格式列索引数组的指针。
 *                       该数组的长度为nnz，用于表示每个非零元素的列索引。
 * @param csrValues IN, DEVICE, 指向CSR格式非零元素数组的指针。
 *                     该数组的长度为nnz，用于存储矩阵的非零元素值。
 * @param csrRowOffsetsType IN, HOST, 行偏移量数组的数据类型，支持 I32 / I64。
 * @param csrColIndType IN, HOST, 列索引数组的数据类型，支持 I32 / I64；
 *                          可独立于 csrRowOffsetsType 选择。
 * @param idxBase IN, HOST, 索引的基值，可以是0或1。
 *                        0表示索引从0开始，1表示索引从1开始。
 * @param valueType IN, HOST, 非零元素的数据类型。
 * @return aclsparseStatus_t 返回执行状态，成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseCreateCsr(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *csrRowOffsets, void *csrColInd, void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

/**
 * @brief 创建只读(const)CSR 稀疏矩阵描述符。
 */
aclsparseStatus_t aclsparseCreateConstCsr(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *csrRowOffsets, const void *csrColInd, const void *csrValues, aclsparseIndexType_t csrRowOffsetsType,
    aclsparseIndexType_t csrColIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

/**
 * @brief 创建一个CSC（Compressed Sparse Column）稀疏矩阵。
 *
 * @note 该接口可创建CSC稀疏矩阵描述符，可供DenseToSparse等支持CSC格式的接口使用。
 *       SpMV当前仅支持CSR格式，传入CSC描述符将返回ACL_SPARSE_STATUS_NOT_SUPPORTED；
 *       SpMM当前仅支持CSR格式，传入CSC描述符将返回ACL_SPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED。
 *
 * @param spMatDescr IN, HOST, 稀疏矩阵描述符，用于存储稀疏矩阵的元数据。
 * @param rows IN, HOST, 矩阵的行数。
 * @param cols IN, HOST, 矩阵的列数。
 * @param nnz IN, HOST, 矩阵中非零元素的数量。
 * @param cscColOffsets IN, DEVICE, 指向列偏移数组的指针，用于存储每列的起始位置。
 * @param cscRowInd IN, DEVICE, 指向行索引数组的指针，用于存储非零元素的行索引。
 * @param cscValues IN, DEVICE, 指向非零元素值数组的指针。
 * @param cscColOffsetsType IN, HOST, 列偏移数组的数据类型。
 * @param cscRowIndType IN, HOST, 行索引数组的数据类型。
 * @param idxBase IN, HOST, 索引的基础值，通常为0或1。
 * @param valueType IN, HOST, 非零元素值的数据类型。
 * @return aclsparseStatus_t 返回执行状态，成功则为ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseCreateCsc(aclsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    void *cscColOffsets, void *cscRowInd, void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

/**
 * @brief 创建只读(const)CSC 稀疏矩阵描述符。
 *
 * @note 当前版本暂未支持，调用返回 ACL_SPARSE_STATUS_NOT_SUPPORTED。
 */
aclsparseStatus_t aclsparseCreateConstCsc(aclsparseConstSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz,
    const void *cscColOffsets, const void *cscRowInd, const void *cscValues, aclsparseIndexType_t cscColOffsetsType,
    aclsparseIndexType_t cscRowIndType, aclsparseIndexBase_t idxBase, aclDataType valueType);

aclsparseStatus_t aclsparseCreateBlockedEll(
    aclsparseSpMatDescr_t *descr, int64_t rows, int64_t cols,
    int64_t ellBlockSize, int64_t ellCols, void *ellColInd, void *ellValue,
    aclsparseIndexType_t indexType, aclsparseIndexBase_t indexBase,
    aclDataType valueType);

aclsparseStatus_t aclsparseCsrSetPointers(aclsparseSpMatDescr_t descr,
    void *rowOffsets, void *colIndices, void *values);
/**
 * @brief 设置CSR三个一维数组的元素步长。
 *
 * 步长按元素计数，rowOffsetsStride、colIndStride和valuesStride均必须大于0。
 */
aclsparseStatus_t aclsparseCsrSetStrides(aclsparseSpMatDescr_t descr,
    int64_t rowOffsetsStride, int64_t colIndStride, int64_t valuesStride);
aclsparseStatus_t aclsparseCsrGetStrides(aclsparseConstSpMatDescr_t descr,
    int64_t *rowOffsetsStride, int64_t *colIndStride, int64_t *valuesStride);
aclsparseStatus_t aclsparseCscSetPointers(aclsparseSpMatDescr_t descr,
    void *colOffsets, void *rowIndices, void *values);
aclsparseStatus_t aclsparseCooSetPointers(aclsparseSpMatDescr_t descr,
    void *rowIndices, void *colIndices, void *values);

/**
 * @brief 销毁稀疏矩阵对象。
 *
 * @param spMatDescr IN, HOST, 稀疏矩阵描述符，用于标识要销毁的稀疏矩阵对象。
 * @return aclsparseStatus_t 返回执行状态，若成功返回ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseDestroySpMat(aclsparseConstSpMatDescr_t spMatDescr);

// ============================================================================
// Sparse Matrix Accessor APIs
// ============================================================================

aclsparseStatus_t aclsparseSpMatGetFormat(aclsparseConstSpMatDescr_t spMatDescr,
    aclsparseFormat_t *format);

aclsparseStatus_t aclsparseSpMatGetValues(aclsparseSpMatDescr_t spMatDescr, void **values);

aclsparseStatus_t aclsparseConstSpMatGetValues(aclsparseConstSpMatDescr_t spMatDescr,
    const void **values);

aclsparseStatus_t aclsparseSpMatSetValues(aclsparseSpMatDescr_t spMatDescr, void *values);

aclsparseStatus_t aclsparseSpMatGetSize(aclsparseConstSpMatDescr_t spMatDescr,
    int64_t *rows, int64_t *cols, int64_t *nnz);

/**
 * @brief 获取稀疏矩阵向量乘法（SpMV）所需的缓冲区大小
 *
 * @param handle IN, HOST, aclsparse 句柄
 * @param opA IN, HOST, 稀疏矩阵操作类型
 * @param alpha IN, HOST/DEVICE, 标量 alpha 指针
 * @param matA IN, HOST, 稀疏矩阵描述符
 * @param vecX IN, HOST, 稠密向量 x 描述符
 * @param beta IN, HOST/DEVICE, 标量 beta 指针
 * @param vecY IN, HOST, 稠密向量 y 描述符
 * @param computeType IN, HOST, 计算精度类型
 * @param alg IN, HOST, SpMV 算法类型
 * @param bufferSize OUT, HOST, 输出所需缓冲区大小
 * @return aclsparseStatus_t 状态码
 */
aclsparseStatus_t aclsparseSpMVGetBufferSize(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
                                                aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
                                                aclsparseSpMVAlg_t alg, size_t *bufferSize);

/**
 * @brief 对稀疏矩阵进行预处理，以便在后续的 SpMV 计算中使用。
 *
 * 可选调用；对同一 sparsity pattern 多次执行 SpMV 时可加速后续计算。
 * 调用后将 buffer 标记为 matA 的 active buffer，后续 SpMV 可复用预处理结果。
 *
 * @param handle IN, HOST, aclsparse 句柄
 * @param opA IN, HOST, 稀疏矩阵操作类型
 * @param alpha IN, HOST/DEVICE, 标量 alpha 指针
 * @param matA IN, HOST, 稀疏矩阵描述符
 * @param vecX IN, HOST, 稠密向量 x 描述符
 * @param beta IN, HOST/DEVICE, 标量 beta 指针
 * @param vecY IN, HOST, 稠密向量 y 描述符
 * @param computeType IN, HOST, 计算精度类型
 * @param alg IN, HOST, SpMV 算法类型
 * @param externalBuffer IN, DEVICE, 工作缓冲区
 * @return aclsparseStatus_t 状态码
 */
aclsparseStatus_t aclsparseSpMVPreprocess(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY,
    aclDataType computeType, aclsparseSpMVAlg_t alg, void *externalBuffer);

/**
 * @brief 稀疏矩阵向量乘法（SpMV）计算入口
 *
 * @param handle IN, HOST, aclsparse 句柄
 * @param opA IN, HOST, 稀疏矩阵操作类型
 * @param alpha IN, HOST/DEVICE, 标量 alpha 指针，不可为空。
 * @param matA IN, HOST, 稀疏矩阵描述符
 * @param vecX IN, HOST, 稠密向量 x 描述符
 * @param beta IN, HOST/DEVICE, 标量 beta 指针，不可为空。
 * @param vecY IN, HOST, 稠密向量 y 描述符
 * @param computeType IN, HOST, 计算精度类型
 * @param alg IN, HOST, SpMV 算法类型
 * @param externalBuffer IN, DEVICE, 工作缓冲区
 * @return aclsparseStatus_t 状态码
 */
aclsparseStatus_t aclsparseSpMV(aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
                                aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX, const void *beta, aclsparseDnVecDescr_t vecY, aclDataType computeType,
                                aclsparseSpMVAlg_t alg, void *externalBuffer);

/**
 * @brief 创建稠密矩阵描述符。
 *
 * 用于描述 SpMM 中的 B / C 稠密矩阵。
 *
 * @param dnMatDescr IN/OUT, HOST，输出的稠密矩阵描述符。
 * @param rows       IN, HOST，矩阵的行数，允许为 0，不允许为负数。
 * @param cols       IN, HOST，矩阵的列数，允许为 0，不允许为负数。
 * @param ld         IN, HOST，leading dimension，必须大于 0。行主序时需 >= cols；
 *                   列主序时需 >= rows；该约束同样适用于零维矩阵。
 * @param values     IN, DEVICE，矩阵数据指针，必须为非空，包括零维矩阵。
 * @param valueType  IN, HOST，元素数据类型。支持 ACL_FLOAT / ACL_FLOAT16 / ACL_INT8
 *                   （用于 B 矩阵）；C 矩阵在 INT8 路径下为 ACL_INT32。
 * @param order      IN, HOST，布局：ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseCreateDnMat(aclsparseDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, void *values,
    aclDataType valueType, aclsparseOrder_t order);

/**
 * @brief 创建只读(const)稠密矩阵描述符。
 *
 * rows/cols、ld、values、valueType 和 order 的契约与 aclsparseCreateDnMat 相同。
 */
aclsparseStatus_t aclsparseCreateConstDnMat(aclsparseConstDnMatDescr_t *dnMatDescr,
    int64_t rows, int64_t cols, int64_t ld, const void *values,
    aclDataType valueType, aclsparseOrder_t order);

/**
 * @brief 销毁稠密矩阵描述符。
 */
aclsparseStatus_t aclsparseDestroyDnMat(aclsparseConstDnMatDescr_t dnMatDescr);

/**
 * @brief 获取稠密矩阵描述符的全部字段。
 *
 * @param dnMatDescr  IN, HOST, 稠密矩阵描述符。
 * @param rows        OUT, HOST, 矩阵的行数。
 * @param cols        OUT, HOST, 矩阵的列数。
 * @param ld          OUT, HOST, leading dimension。
 * @param values      OUT, DEVICE, 矩阵数据指针。
 * @param valueType   OUT, HOST, 元素数据类型。
 * @param order       OUT, HOST, 内存布局。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseDnMatGet(aclsparseDnMatDescr_t dnMatDescr, int64_t *rows, int64_t *cols,
                                    int64_t *ld, void **values, aclDataType *valueType,
                                    aclsparseOrder_t *order);

/**
 * @brief 获取只读(const)稠密矩阵描述符的全部字段。
 */
aclsparseStatus_t aclsparseConstDnMatGet(aclsparseConstDnMatDescr_t dnMatDescr, int64_t *rows,
                                         int64_t *cols, int64_t *ld, const void **values,
                                         aclDataType *valueType, aclsparseOrder_t *order);

/**
 * @brief 获取稠密矩阵描述符的 values 指针。
 *
 * @param dnMatDescr IN, HOST, 稠密矩阵描述符。
 * @param values     OUT, DEVICE, 矩阵数据指针。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseDnMatGetValues(aclsparseDnMatDescr_t dnMatDescr, void **values);

/**
 * @brief 获取只读(const)稠密矩阵描述符的 values 指针。
 */
aclsparseStatus_t aclsparseConstDnMatGetValues(aclsparseConstDnMatDescr_t dnMatDescr,
                                               const void **values);

/**
 * @brief 设置稠密矩阵描述符的 values 指针。
 *
 * @param dnMatDescr IN, HOST, 稠密矩阵描述符。
 * @param values     IN, DEVICE, 矩阵数据指针。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseDnMatSetValues(aclsparseDnMatDescr_t dnMatDescr, void *values);

aclsparseStatus_t aclsparseDenseToSparseGetBufferSize(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
    size_t *bufferSize);
aclsparseStatus_t aclsparseDenseToSparseAnalysis(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
    void *buffer);
aclsparseStatus_t aclsparseDenseToSparseConvert(
    aclsparseHandle_t handle, aclsparseConstDnMatDescr_t matA,
    aclsparseSpMatDescr_t matB, aclsparseDenseToSparseAlg_t alg,
    void *buffer);

// ============================================================================
// SparseToDense Generic API
// ============================================================================
//  B = A，将稀疏矩阵 matA（CSR/CSC/COO）转换为稠密矩阵 matB。
//  matA 与 matB 的行列数、值类型必须一致；matB 需在调用前分配好 device 内存。
//  非零元位置写入值，其余位置由库内部清零。
//  对标 cuSPARSE cusparseSparseToDense / cusparseSparseToDense_bufferSize。

typedef enum aclsparseSparseToDenseAlg_t {
    ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT = 0,
} aclsparseSparseToDenseAlg_t;

/**
 * @brief 查询 SparseToDense 转换所需的 workspace 字节数。
 *
 * @param handle     IN, HOST, aclsparse 句柄，不可为空。
 * @param matA       IN, HOST 描述符；数据在 Device，稀疏矩阵 A（CSR/CSC/COO 格式），不可为空。
 * @param matB       IN, HOST 描述符；数据在 Device，输出稠密矩阵 B，不可为空。
 * @param alg        IN, HOST, 算法选择，仅支持 ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT。
 * @param bufferSize OUT, HOST, workspace 字节数，当前恒返回 0。
 * @return aclsparseStatus_t 返回执行状态，成功返回 ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseSparseToDense_bufferSize(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    size_t *bufferSize);

/**
 * @brief 执行 SparseToDense 转换，将稀疏矩阵 matA 写入稠密矩阵 matB。
 *
 * 按 handle 绑定的 stream 异步执行；调用方需在读取结果前同步该 stream。
 * matB 的 Device 内存由算子内部通过 aclrtMemset 清零后，非零元位置写入对应值。
 * CSR/CSC/COO 三格式的描述符字段含义参见 README。
 *
 * @param handle IN, HOST, aclsparse 句柄，不可为空。
 * @param matA   IN, HOST 描述符；数据在 Device，稀疏矩阵 A（CSR/CSC/COO 格式），不可为空。
 * @param matB   IN, HOST 描述符；数据在 Device，输出稠密矩阵 B，不可为空。
 * @param alg    IN, HOST, 算法选择，仅支持 ACL_SPARSE_SPARSETODENSE_ALG_DEFAULT。
 * @param buffer IN, DEVICE, workspace；bufferSize == 0 时可传 nullptr。
 * @return aclsparseStatus_t 返回执行状态，成功返回 ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseSparseToDense(
    aclsparseHandle_t handle,
    aclsparseConstSpMatDescr_t matA,
    aclsparseDnMatDescr_t matB,
    aclsparseSparseToDenseAlg_t alg,
    void *buffer);

/**
 * @brief 计算 SpMM 所需 workspace 字节数。
 */
aclsparseStatus_t aclsparseSpMMGetBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, size_t *size);

/**
 * @brief 对稀疏矩阵进行预处理，加速后续 SpMM 计算。
 */
aclsparseStatus_t aclsparseSpMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);

/**
 * @brief 稀疏矩阵-稠密矩阵乘法：C = alpha * op(A) * op(B) + beta * C
 */
aclsparseStatus_t aclsparseSpMM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA, aclsparseConstDnMatDescr_t matB,
    const void *beta, aclsparseDnMatDescr_t matC, aclDataType computeType,
    aclsparseSpMMAlg_t alg, void *buffer);

/**
 * @brief 创建 Cube SpMM 稀疏矩阵描述符（BCSR 专用）。
 *
 * 该描述符只保存稀疏矩阵的元信息以及预处理后的 BCSR 输出缓冲区；
 * COO 原始输入不保存在描述符中，而是在 aclsparseCubeSpmmPreprocess 时直接传入。
 *
 * @param descr     输出描述符。
 * @param rows      稀疏矩阵行数 M。
 * @param cols      稀疏矩阵列数 K。
 * @param nnz       COO 非零元个数。
 * @param blockM    行方向块大小。
 * @param blockK    列方向块大小。
 * @param numCores  使用的 AICore 数。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseCreateCubeSpmmMat(aclsparseCubeSpmmMatDescr_t *descr,
    int64_t rows, int64_t cols, int64_t nnz,
    int64_t blockM, int64_t blockK, int32_t numCores);

/**
 * @brief 销毁 Cube SpMM 稀疏矩阵描述符。
 */
aclsparseStatus_t aclsparseDestroyCubeSpmmMat(aclsparseConstCubeSpmmMatDescr_t descr);

/**
 * @brief 计算 Cube SpMM 所需 workspace 字节数。
 *
 * 当前实现不需要 device workspace：tiling 作为 kernel 启动参数随 <<<>>> 下发，
 * 本接口固定返回 0，aclsparseCubeSpmm 的 buffer 可传 nullptr。
 */
aclsparseStatus_t aclsparseCubeSpmmGetBufferSize(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    size_t *size);

/**
 * @brief 对 Cube SpMM 输入进行预处理（COO -> Cube-BCSR）。
 *
 * 只负责稀疏矩阵 A 的格式转换、列凝聚与负载均衡，不处理稠密 B。
 * 预处理结果保存在 matA 描述符内部；workspace 不被本接口使用。
 *
 * @param cooRows   设备 COO 行索引（int32）。
 * @param cooCols   设备 COO 列索引（int32）。
 * @param cooVals   设备 COO 值（float16，按 uint16_t 传入）。
 */
aclsparseStatus_t aclsparseCubeSpmmPreprocess(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseCubeSpmmMatDescr_t matA,
    const int32_t *cooRows, const int32_t *cooCols, const uint16_t *cooVals,
    aclDataType computeType);

/**
 * @brief 对 Cube SpMM 的稠密 B 做 padding 对齐。
 *
 * 根据 L0B / CopyInB stride 对齐要求，对稠密 B 的 N 维度做 padding，输出
 * nPadOut 与 bPadOut。该函数与稀疏 A 的预处理解耦，允许在 A 不变而 B
 * 变化时单独调用。
 *
 * @param bRows     稠密 B 的行数（应等于 matA->cols）。
 * @param bCols     稠密 B 的列数（原始 N）。
 * @param bLd       稠密 B 的 leading dimension。
 * @param bValues   稠密 B 的主机内存指针（fp16），由本接口负责 padding 后 H2D 拷贝到 *bPadOut。
 * @param bType     稠密 B 的数据类型，必须为 ACL_FLOAT16。
 * @param bOrder    稠密 B 的存储顺序，必须为 ACL_SPARSE_ORDER_ROW。
 * @param nPadOut   输出 padding 后的 N 大小。
 * @param bPadOut   输出 padding 后的 B 设备内存指针（需由调用者 aclrtFree）。
 */
aclsparseStatus_t aclsparseCubeSpmmPadDenseMatrixB(
    aclsparseHandle_t handle,
    int64_t bRows, int64_t bCols, int64_t bLd, const void *bValues,
    aclDataType bType, aclsparseOrder_t bOrder,
    int64_t *nPadOut, void **bPadOut);

/**
 * @brief Cube SpMM 计算：C = alpha * A(BCSR, fp16) * B(fp16) + beta * C。
 *
 * 仅支持 opA = NON_TRANSPOSE, opB = NON_TRANSPOSE；B/C 的 N 需已经过 padding 对齐。
 * 每次调用都会按当前 B/C 维度重建 tiling，并作为 kernel 启动参数随 <<<>>> 下发；
 * 当前实现不使用 workspace，buffer 可传 nullptr。
 */
aclsparseStatus_t aclsparseCubeSpmm(
    aclsparseHandle_t handle,
    const void *alpha, aclsparseConstCubeSpmmMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, const void *beta,
    aclsparseDnMatDescr_t matC, aclDataType computeType,
    void *buffer);

// ============================================================================
// SDDMM (Sampled Dense-Dense Matrix Multiplication) Generic API
// ============================================================================
//  C_out = (alpha * X * Y^T + beta * C) ∘ spy(C)
//  X: m×k dense, Y: n×k dense, C: m×n sparse (CSR), values updated in-place.

// Algorithm enum for SDDMM.
typedef enum aclsparseSDDMMAlg_t {
    ACL_SPARSE_SDDMM_ALG_DEFAULT = 0,
} aclsparseSDDMMAlg_t;

/**
 * @brief 计算 SDDMM 所需 workspace 字节数。
 */
aclsparseStatus_t aclsparseSDDMMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, size_t *size);

/**
 * @brief 对稀疏矩阵进行预处理，加速后续 SDDMM 计算。
 */
aclsparseStatus_t aclsparseSDDMMPreprocess(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);

/**
 * @brief 采样稠密-稠密矩阵乘法：C = (alpha * X * Y^T + beta * C) ∘ spy(C)
 *
 * @note 线程安全：本接口内部使用 pattern cache 加速重复调用，该 cache 非线程安全。
 *       多 stream 并发调用时，调用者须保证串行访问，或使用独立的 matC 描述符。
 */
aclsparseStatus_t aclsparseSDDMM(
    aclsparseHandle_t handle, aclsparseOperation_t opX, aclsparseOperation_t opY,
    const void *alpha, aclsparseConstDnMatDescr_t matX, aclsparseConstDnMatDescr_t matY,
    const void *beta, aclsparseSpMatDescr_t matC, aclDataType computeType,
    aclsparseSDDMMAlg_t alg, void *buffer);

// ============================================================================
// Generic SpSM (Sparse Triangular Solve with Multiple Right-Hand Sides)
// 对标 cuSPARSE Generic API cusparseSpSM。三阶段: BufferSize → Analysis → Solve。
// 求解 op(A) * X = alpha * op(B), A 为稀疏三角矩阵 (UNIT/NON_UNIT diagonal, m×m)。
// 支持 CSR/CSC/COO 稀疏格式及 FP32/complex64 数据类型。
// A2/A3 contract and known limitations: sparse/spsm/arch22/spsm_host.cpp interface comments.
// The Host does not detect destroyed handles or matrix descriptors, nor arrays
// shorter than the lengths described by their descriptors.
// ============================================================================

// SpSM 算法枚举。
typedef enum aclsparseSpSMAlg_t {
    ACL_SPARSE_SPSM_ALG_DEFAULT = 0
} aclsparseSpSMAlg_t;

/** Numeric update selection; A2/A3 takes a full nnz array in the original entry order. */
typedef enum aclsparseSpSMUpdate_t {
    ACL_SPARSE_SPSM_UPDATE_GENERAL = 0,
    ACL_SPARSE_SPSM_UPDATE_DIAGONAL = 1
} aclsparseSpSMUpdate_t;

/**
 * @brief Update numerical values on the Analysis-bound stream and workspace.
 * @note A2/A3: newValues holds nnz(A) values of the Analysis compute type in the
 *       storage order of matA's value array (before sorting or duplicate
 *       summation). It must be 4-byte aligned Device memory on the current
 *       device and must not overlap the workspace. For opA=H pass A's values.
 * @note GENERAL replaces all values. DIAGONAL reads the same full array but takes
 *       only entries with row == col; other entries keep the plan's current
 *       values, not the contents of matA's value array.
 * @note Handle, pointer mode, stream, device, descriptors and their metadata must
 *       match Analysis and remain alive. newValues, A index arrays and workspace
 *       must stay valid until the stream completes. Structural changes require
 *       Analysis again. A Host-side rejection enqueues nothing. A2/A3 waits for
 *       numerical validation; failed Device validation invalidates this plan.
 * @note arch35: GENERAL takes nnz entries; DIAGONAL takes m diagonal entries.
 *       Failed numerical validation preserves the previous plan on arch35.
 *       UNIT ignores supplied diagonal values.
 */
aclsparseStatus_t aclsparseSpSMUpdateMatrix(aclsparseHandle_t handle,
    aclsparseSpSMDescr_t spsmDescr, const void *newValues, aclsparseSpSMUpdate_t updatePart);


/**
 * @brief 创建 SpSM 描述符 (跨三阶段共享)。
 */
aclsparseStatus_t aclsparseSpSMCreateDescr(aclsparseSpSMDescr_t *spsmDescr);

/**
 * @brief 销毁 SpSM 描述符。
 * @note Does not wait for enqueued tasks. Caller-owned inputs, outputs, Device-mode alpha
 *       and workspace used by those tasks must stay valid until stream completion.
 */
aclsparseStatus_t aclsparseSpSMDestroyDescr(aclsparseSpSMDescr_t spsmDescr);

/**
 * @brief 查询 SpSM 所需 workspace 字节数。数学公式: op(A) * X = alpha * op(B)。
 * @note A2/A3 supports CSR/CSC/COO, FP32/complex64 and N/T/H for both operands.
 *       arch35 also supports CSR/CSC/COO, FP32/complex64 and N/T/H.
 *       Zero dimensions are accepted on A2/A3 but rejected on arch35.
 * @note A2/A3: B/C values may be NULL. The size depends on the current device;
 *       query and analyze on the same device. On failure *bufferSize is unchanged.
 */
aclsparseStatus_t aclsparseSpSMBufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, size_t *bufferSize);

/**
 * @brief Analyze SpSM and bind descriptors, stream and external workspace.
 *        A2/A3 enqueues Device normalization; Solve reuses that representation.
 *        数学公式: op(A) * X = alpha * op(B)。
 * @note Operand support depends on architecture; see aclsparseSpSMBufferSize.
 * @note A2/A3 binds handle, pointer mode, stream, device, descriptors, their
 *       metadata and workspace; any change requires Analysis again. workspace
 *       must be 64-byte aligned Device memory on the current device, at least
 *       bufferSize bytes, and disjoint from A, B and C. A index and value arrays
 *       must be 4-byte aligned Device memory. alpha placement is validated but
 *       its value is not read.
 * @note B/C values may be NULL and set later with aclsparseDnMatSetValues.
 *       Non-NULL values are fixed by Analysis; NULL values are fixed by the first
 *       successful Solve. Other values need Analysis again.
 * @note UNIT ignores stored diagonal entries and uses one. NON_UNIT sums
 *       duplicate diagonal entries.
 * @note A Host-side rejection enqueues nothing and keeps any previous plan
 *       unchanged; its resources must remain valid if it is used again. Success
 *       includes completion of Device numerical validation.
 * @note arch35 can use handle workspace when buffer is NULL. Analysis waits for
 *       Device validation and reads back status, not matrix arrays.
 */
aclsparseStatus_t aclsparseSpSMAnalysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr, void *buffer);

/**
 * @brief SpSM 求解阶段: 异步执行三角求解 op(A) * C = alpha * op(B)。
 *        复用 Analysis 绑定的 active buffer。禁止 host 侧 stream 同步。
 * @note Operand support depends on architecture; see aclsparseSpSMBufferSize.
 * @note A2/A3: Host-mode alpha is copied during this call. Device-mode alpha must
 *       be 4-byte aligned on the current device, disjoint from the workspace and
 *       valid until the stream completes. When m and rhs are nonzero, B/C values
 *       must be 4-byte aligned Device memory; C may equal B exactly but must not
 *       partially overlap B or overlap A. B, C, A arrays and workspace must stay
 *       valid until the stream completes. A Host-side rejection enqueues nothing
 *       and keeps the plan unchanged. Success means enqueued only.
 */
aclsparseStatus_t aclsparseSpSM(
    aclsparseHandle_t handle, aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB, aclsparseDnMatDescr_t matC,
    aclDataType computeType, aclsparseSpSMAlg_t alg,
    aclsparseSpSMDescr_t spsmDescr);

/**
 * @brief 创建 ops-sparse handle
 *
 * 在堆上分配一个 ops-sparse handle，并通过 @p handle 输出给调用者。
 *
 * @param handle 输出参数，用于接收创建的 handle 指针。调用前 *handle
 *               必须为 nullptr，否则视为已存在以防止内存泄漏。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE *handle 非空
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseCreate(aclsparseHandle_t *handle);

/**
 * @brief 销毁 ops-sparse handle
 *
 * 释放由 aclsparseCreate 创建的 handle 所占有的全部资源。
 *
 * @param handle 要销毁的 handle
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclsparseStatus_t aclsparseDestroy(aclsparseHandle_t handle);

/**
 * @brief 设置 handle 的 stream
 * @param handle aclsparse handle（void*）
 * @param stream 要设置的 stream，nullptr 表示使用默认 stream
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 */
aclsparseStatus_t aclsparseSetStream(aclsparseHandle_t handle, aclrtStream stream);

/**
 * @brief 获取 handle 的 stream
 * @param handle aclsparse handle（void*）
 * @param stream 输出参数，返回当前 stream
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE stream 输出参数为空
 */
aclsparseStatus_t aclsparseGetStream(aclsparseHandle_t handle, aclrtStream *stream);

/**
 * @brief 设置用户 workspace
 *
 * 允许用户提供预分配的设备内存作为 workspace，避免算子内部频繁 malloc/free。
 * 库仅借用用户内存，不获取所有权。
 *
 * 采用 grow-only 策略：当新 workspaceSize 不大于当前 user workspace size 时，
 * 保持原有 workspace 不变（仍返回 SUCCESS）。如需替换为更小的 workspace，
 * 请先传入 nullptr 切回默认 workspace，再重新设置。
 *
 * @param handle aclsparse handle
 * @param workspace 用户分配的设备内存指针，nullptr 表示切回默认 workspace
 * @param workspaceSize workspace 大小（字节）
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE workspaceSize 为 0（当 workspace 非空时）
 */
aclsparseStatus_t aclsparseSetWorkspace(aclsparseHandle_t handle, void *workspace, size_t workspaceSize);

/**
 * @brief 获取当前活跃 workspace
 *
 * 返回当前使用的 workspace 指针和大小（可能是默认 workspace 或用户 workspace）。
 *
 * @param handle aclsparse handle
 * @param workspace 输出参数，返回当前 workspace 指针
 * @param workspaceSize 输出参数，返回当前 workspace 大小
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE 输出参数为空
 */
aclsparseStatus_t aclsparseGetWorkspace(aclsparseHandle_t handle, void **workspace, size_t *workspaceSize);

/**
 * @brief 获取 ops-sparse 版本号
 *
 * 版本号编码为 MAJOR * 10000 + MINOR * 100 + PATCH，如 10000 表示 1.0.0。
 *
 * @param handle handle指针，可为NULL
 * @param version 输出参数，接收版本号
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_INVALID_VALUE 参数无效
 */
aclsparseStatus_t aclsparseGetVersion(aclsparseHandle_t handle, int *version);

// ============================================================================
// Pointer Mode
// ============================================================================

typedef enum aclsparsePointerMode_t {
    ACL_SPARSE_POINTER_MODE_HOST = 0,
    ACL_SPARSE_POINTER_MODE_DEVICE = 1
} aclsparsePointerMode_t;

aclsparseStatus_t aclsparseSetPointerMode(aclsparseHandle_t handle, aclsparsePointerMode_t mode);
aclsparseStatus_t aclsparseGetPointerMode(aclsparseHandle_t handle, aclsparsePointerMode_t *mode);

// ============================================================================
// Legacy API: MatDescr management
// ============================================================================

aclsparseStatus_t aclsparseCreateMatDescr(aclsparseMatDescr_t *descr);
aclsparseStatus_t aclsparseDestroyMatDescr(aclsparseMatDescr_t descr);
void aclsparseSetMatType(aclsparseMatDescr_t descr, aclsparseMatrixType_t type);
void aclsparseSetMatIndexBase(aclsparseMatDescr_t descr, aclsparseIndexBase_t base);
void aclsparseSetMatDiagType(aclsparseMatDescr_t descr, aclsparseDiagType_t diagType);
void aclsparseSetMatFillMode(aclsparseMatDescr_t descr, aclsparseFillMode_t fillMode);
aclsparseMatrixType_t aclsparseGetMatType(aclsparseMatDescr_t descr);
aclsparseIndexBase_t aclsparseGetMatIndexBase(aclsparseMatDescr_t descr);
aclsparseDiagType_t aclsparseGetMatDiagType(aclsparseMatDescr_t descr);
aclsparseFillMode_t aclsparseGetMatFillMode(aclsparseMatDescr_t descr);

// ============================================================================
// Legacy API: aclsparseSnnz
// ============================================================================

/**
 * @brief Count the number of nonzero elements per row or column in a dense matrix.
 *
 * @param handle             IN, HOST, aclsparse handle.
 * @param dirA               IN, HOST, direction: ROW or COLUMN.
 * @param m                  IN, HOST, number of rows.
 * @param n                  IN, HOST, number of columns.
 * @param descrA             IN, HOST, matrix descriptor.
 * @param A                  IN, DEVICE, dense matrix data (column-major).
 * @param lda                IN, HOST, leading dimension (>= max(1, m)).
 * @param nnzPerRowColumn    OUT, DEVICE, array of size m (ROW) or n (COLUMN).
 * @param nnzTotalDevHostPtr OUT, DEVICE/HOST, total nonzero count.
 *                           Memory location depends on aclsparseSetPointerMode:
 *                           HOST mode -> host memory, DEVICE mode -> device memory.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSnnz(
    aclsparseHandle_t handle,
    aclsparseDirection_t dirA,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const float *A, int lda,
    int *nnzPerRowColumn,
    int *nnzTotalDevHostPtr);

// ============================================================================
// Legacy API: aclsparseScsrgeam2 — C = alpha * A + beta * B  (CSR, FP32)
// ============================================================================

/**
 * @brief Query workspace size for aclsparseScsrgeam2.
 *
 * @param handle               IN, HOST, aclsparse handle.
 * @param m                    IN, HOST, number of rows of A, B, C.
 * @param n                    IN, HOST, number of columns of A, B, C.
 * @param alpha                IN, HOST/DEVICE, pointer to scalar alpha.
 * @param descrA               IN, HOST, matrix A descriptor.
 * @param nnzA                 IN, HOST, number of nonzeros in A.
 * @param csrSortedValA        IN, DEVICE, A nonzero values (length nnzA).
 * @param csrSortedRowPtrA     IN, DEVICE, A row pointers (length m+1).
 * @param csrSortedColIndA     IN, DEVICE, A column indices (length nnzA).
 * @param beta                 IN, HOST/DEVICE, pointer to scalar beta.
 * @param descrB               IN, HOST, matrix B descriptor.
 * @param nnzB                 IN, HOST, number of nonzeros in B.
 * @param csrSortedValB        IN, DEVICE, B nonzero values (length nnzB).
 * @param csrSortedRowPtrB     IN, DEVICE, B row pointers (length m+1).
 * @param csrSortedColIndB     IN, DEVICE, B column indices (length nnzB).
 * @param descrC               IN, HOST, matrix C descriptor.
 * @param csrSortedValC        IN, DEVICE, C nonzero values (may be NULL).
 * @param csrSortedRowPtrC     IN, DEVICE, C row pointers (may be NULL).
 * @param csrSortedColIndC     IN, DEVICE, C column indices (may be NULL).
 * @param pBufferSizeInBytes   OUT, HOST, workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseScsrgeam2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, const float *alpha, const aclsparseMatDescr_t descrA, int nnzA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const aclsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes);

/**
 * @brief Compute the structure of C (rowPtrC and nnzC).
 *
 * Kernel 1: count the number of nonzeros per row in the union A ∪ B,
 * then Kernel 3: perform an exclusive prefix sum on the device side.
 *
 * @param handle                IN, HOST, aclsparse handle.
 * @param m                     IN, HOST, number of rows.
 * @param n                     IN, HOST, number of columns.
 * @param descrA                IN, HOST, matrix A descriptor.
 * @param nnzA                  IN, HOST, number of nonzeros in A.
 * @param csrSortedRowPtrA      IN, DEVICE, A row pointers (length m+1).
 * @param csrSortedColIndA      IN, DEVICE, A column indices (length nnzA).
 * @param descrB                IN, HOST, matrix B descriptor.
 * @param nnzB                  IN, HOST, number of nonzeros in B.
 * @param csrSortedRowPtrB      IN, DEVICE, B row pointers (length m+1).
 * @param csrSortedColIndB      IN, DEVICE, B column indices (length nnzB).
 * @param descrC                IN, HOST, matrix C descriptor.
 * @param csrSortedRowPtrC      OUT, DEVICE, C row pointers (length m+1).
 * @param nnzTotalDevHostPtr    OUT, DEVICE/HOST, total nonzero count of C.
 * @param workspace             IN, DEVICE, workspace (>= (m+1) * sizeof(int)).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcsrgeam2Nnz(
    aclsparseHandle_t handle, int m, int n, const aclsparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const aclsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr,
    void *workspace);

/**
 * @brief Compute column indices and values of C: C = alpha * A + beta * B.
 *
 * Kernel 2: for each row, perform an ordered merge of column indices from
 * A and B, writing the resulting column indices and values to C.
 *
 * @param handle               IN, HOST, aclsparse handle.
 * @param m                    IN, HOST, number of rows.
 * @param n                    IN, HOST, number of columns.
 * @param alpha                IN, HOST/DEVICE, pointer to scalar alpha.
 * @param descrA               IN, HOST, matrix A descriptor.
 * @param nnzA                 IN, HOST, number of nonzeros in A.
 * @param csrSortedValA        IN, DEVICE, A nonzero values (length nnzA).
 * @param csrSortedRowPtrA     IN, DEVICE, A row pointers (length m+1).
 * @param csrSortedColIndA     IN, DEVICE, A column indices (length nnzA).
 * @param beta                 IN, HOST/DEVICE, pointer to scalar beta.
 * @param descrB               IN, HOST, matrix B descriptor.
 * @param nnzB                 IN, HOST, number of nonzeros in B.
 * @param csrSortedValB        IN, DEVICE, B nonzero values (length nnzB).
 * @param csrSortedRowPtrB     IN, DEVICE, B row pointers (length m+1).
 * @param csrSortedColIndB     IN, DEVICE, B column indices (length nnzB).
 * @param descrC               IN, HOST, matrix C descriptor.
 * @param csrSortedValC        OUT, DEVICE, C nonzero values (length nnzC).
 * @param csrSortedRowPtrC     IN, DEVICE, C row pointers (length m+1).
 * @param csrSortedColIndC     OUT, DEVICE, C column indices (length nnzC).
 * @param pBuffer              IN, DEVICE, workspace (from bufferSizeExt).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseScsrgeam2(
    aclsparseHandle_t handle, int m, int n, const float *alpha, const aclsparseMatDescr_t descrA, int nnzA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const aclsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const aclsparseMatDescr_t descrC, float *csrSortedValC,
    int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer);
/**
 * @brief 稀疏向量-稠密向量点积：result = sum(x_values[i] * y[x_indices[i]])
 *
 * @param handle IN, HOST, 稀疏计算句柄。
 * @param op IN, HOST, 操作类型（仅支持 ACL_SPARSE_OP_NON_TRANSPOSE）。
 * @param x IN, HOST, 稀疏向量描述符（索引必须按升序排列）。
 * @param y IN, HOST, 稠密向量描述符。
 * @param result IN/OUT, DEVICE, 指向设备端float结果的指针（1个float元素）。
 * @param computeType IN, HOST, 累加/输出类型，仅支持 ACL_FLOAT（与 valueType 相互独立）。
 *                    valueType（x/y 元素类型）可为 ACL_FLOAT 或 ACL_FLOAT16，核函数按 valueType 分发。
 * @return aclsparseStatus_t 返回执行状态。
 */
aclsparseStatus_t aclsparseSpvv(aclsparseHandle_t handle, aclsparseOperation_t op,
    aclsparseConstSpVecDescr_t x, aclsparseConstDnVecDescr_t y,
    void *result, aclDataType computeType);

// ============================================================================
// Legacy API: aclsparseSgtsvInterleavedBatch
// ============================================================================

/**
 * @brief Solve a batch of tridiagonal linear systems with interleaved data layout.
 *
 * Solves A^(k) * x^(k) = b^(k) for k = 0, ..., batchCount-1 using the Thomas
 * algorithm (algo=0). Each A^(k) is an m x m tridiagonal matrix defined by
 * lower diagonal dl, main diagonal d, and upper diagonal du.
 * Data layout: row-major interleaved, array[row * batchCount + batch].
 * dl[0][*] must be 0, du[m-1][*] must be 0 (user responsibility).
 *
 * The Thomas algorithm is a simplified LU decomposition without pivoting and
 * is numerically stable only for diagonally-dominant or well-conditioned
 * tridiagonal systems. No singularity guards are applied: if any diagonal
 * element becomes zero during forward sweep or backward substitution,
 * IEEE-754 division by zero produces Inf/NaN in the output. The caller
 * must ensure input matrices are well-conditioned; otherwise the solution
 * for that batch is undefined.
 *
 * @param handle        IN, HOST, aclsparse handle.
 * @param algo          IN, HOST, algorithm selector (0 = Thomas).
 * @param m             IN, HOST, system size (rows = cols).
 * @param dl            IN/OUT, DEVICE, sub-diagonal [m * batchCount].
 * @param d             IN/OUT, DEVICE, main diagonal [m * batchCount].
 * @param du            IN/OUT, DEVICE, super-diagonal [m * batchCount].
 * @param x             IN/OUT, DEVICE, right-hand side b on input, solution x on output [m * batchCount].
 * @param batchCount    IN, HOST, number of systems.
 * @param pBuffer       IN, DEVICE, workspace buffer (128-byte aligned).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsvInterleavedBatch(
    aclsparseHandle_t handle, int algo, int m,
    float *dl, float *d, float *du, float *x,
    int batchCount, void *pBuffer);

/**
 * @brief Query workspace buffer size for aclsparseSgtsvInterleavedBatch.
 *
 * @param handle              IN, HOST, aclsparse handle.
 * @param algo                IN, HOST, algorithm selector (0 = Thomas).
 * @param m                   IN, HOST, system size (m >= 1).
 * @param dl                  IN, DEVICE, sub-diagonal (query ignores pointer; may be NULL).
 * @param d                   IN, DEVICE, main diagonal (query ignores pointer; may be NULL).
 * @param du                  IN, DEVICE, super-diagonal (query ignores pointer; may be NULL).
 * @param x                   IN, DEVICE, right-hand side (query ignores pointer; may be NULL).
 * @param batchCount          IN, HOST, number of systems (>= 1).
 * @param pBufferSizeInBytes  OUT, HOST, required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsvInterleavedBatch_bufferSizeExt(
    aclsparseHandle_t handle, int algo, int m,
    const float *dl, const float *d, const float *du, const float *x,
    int batchCount, size_t *pBufferSizeInBytes);
// Legacy API: aclsparseSgtsv2StridedBatch — Tridiagonal system solve (FP32)
// ============================================================================

/**
 * @brief Query workspace size for aclsparseSgtsv2StridedBatch.
 *
 * Supports two paths:
 *   - m <= 2048 (pure UB path): returns 0 (pBuffer is a contract placeholder).
 *   - 2048 < m <= 2^30 (outer GM-tiled CR path): returns non-zero workspace size,
 *     used to stage outer-level coefficients and saved intermediate rows in GM.
 *   - m > 2^30: returns ACL_SPARSE_STATUS_NOT_SUPPORTED, bufferSize=0.
 *
 * @param handle          IN, HOST, aclsparse handle.
 * @param m               IN, HOST, dimension of each tridiagonal system (>= 3).
 * @param dl              IN, DEVICE, lower diagonal (length batchCount * batchStride).
 * @param d               IN, DEVICE, main diagonal.
 * @param du              IN, DEVICE, upper diagonal.
 * @param x               IN, DEVICE, right-hand side / solution.
 * @param batchCount      IN, HOST, number of batched systems.
 * @param batchStride     IN, HOST, element stride between batches (>= ceil(m_pad/8)*8, batchStride % 8 == 0).
 * @param bufferSizeInBytes OUT, HOST, workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsv2StridedBatch_bufferSizeExt(
    aclsparseHandle_t handle, int m,
    const float *dl, const float *d, const float *du,
    const float *x, int batchCount, int batchStride,
    size_t *bufferSizeInBytes);

/**
 * @brief Solve batched tridiagonal systems T_i * x_i = b_i using CR algorithm.
 *
 * Each system is defined by lower diagonal dl, main diagonal d, upper diagonal du,
 * and right-hand side x (which is overwritten with the solution). All arrays are
 * strided by batchStride elements between consecutive batches.
 *
 * Constraints:
 *   - 3 <= m <= 2^30 (m <= 2048 pure UB path, 2048 < m <= 2^30 outer GM-tiled CR path;
 *     m > 2^30 returns ACL_SPARSE_STATUS_NOT_SUPPORTED)
 *   - batchStride >= ceil(m_pad/8)*8 and batchStride % 8 == 0 (DataCopy 32B alignment)
 *   - dl[0] = 0 and du[m-1] = 0 (user-guaranteed boundary conditions)
 *   - singular systems (zero pivot during reduction) yield Inf/NaN in x
 *     (no-pivoting algorithm, no error is reported)
 *   - m <= 2048: pBuffer may be NULL (pure UB path)
 *   - 2048 < m <= 2^30: pBuffer must be non-NULL, 128B-aligned, size >= bufferSizeExt
 *     (query the required size via aclsparseSgtsv2StridedBatch_bufferSizeExt)
 *
 * @param handle          IN, HOST, aclsparse handle.
 * @param m               IN, HOST, dimension of each tridiagonal system (3 <= m <= 2^30).
 * @param dl              IN, DEVICE, lower diagonal.
 * @param d               IN, DEVICE, main diagonal.
 * @param du              IN, DEVICE, upper diagonal.
 * @param x               IN/OUT, DEVICE, right-hand side on input, solution on output.
 * @param batchCount      IN, HOST, number of batched systems (>= 0).
 * @param batchStride     IN, HOST, element stride between batches (batchStride % 8 == 0, >= ceil(m_pad/8)*8).
 * @param pBuffer         IN, DEVICE, workspace (may be NULL when bufferSizeExt returns 0).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsv2StridedBatch(
    aclsparseHandle_t handle, int m,
    const float *dl, const float *d, const float *du,
    float *x, int batchCount, int batchStride,
    void *pBuffer);

// ============================================================================
// Legacy API: aclsparseSgtsv2Nopivot
// ============================================================================

/**
 * @brief Solve a tridiagonal linear system without pivoting (Thomas algorithm).
 *
 * Solves A * X = B where A is an m x m tridiagonal matrix defined by
 * sub-diagonal dl, main diagonal d, and super-diagonal du.
 * B is an m x n right-hand side matrix (column-major, leading dimension ldb).
 * X is the solution, stored in B on output (in-place).
 *
 * Uses the Thomas algorithm (forward sweep + backward substitution), a simplified
 * LU decomposition without pivoting for tridiagonal systems. The algorithm is
 * numerically stable for diagonally-dominant or well-conditioned tridiagonal
 * systems, matching cuSPARSE gtsv2_nopivot behavior.
 *
 * No pivoting is performed.
 *
 * Constraints (user responsibility, NOT checked at runtime):
 *   - dl[0] == 0 (first element of sub-diagonal must be 0)
 *   - du[m-1] == 0 (last element of super-diagonal must be 0)
 *   - Matrix must be non-singular
 *
 * @param handle        IN, HOST, aclsparse handle.
 * @param m             IN, HOST, system size (rows = cols), m >= 3.
 * @param n             IN, HOST, number of right-hand sides, n >= 1.
 * @param dl            IN, DEVICE, sub-diagonal [m], dl[0] must be 0.
 * @param d             IN, DEVICE, main diagonal [m].
 * @param du            IN, DEVICE, super-diagonal [m], du[m-1] must be 0.
 * @param B             IN/OUT, DEVICE, right-hand side on input, solution on output [ldb * n].
 * @param ldb           IN, HOST, leading dimension of B, ldb >= max(1, m).
 * @param pBuffer       IN, DEVICE, workspace buffer (128-byte aligned), size from bufferSizeExt.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsv2Nopivot(
    aclsparseHandle_t handle, int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer);

/**
 * @brief Query workspace buffer size for aclsparseSgtsv2Nopivot.
 *
 * @param handle              IN, HOST, aclsparse handle.
 * @param m                   IN, HOST, system size (m >= 3).
 * @param n                IN, HOST, number of right-hand sides (>= 1).
 * @param dl                  IN, DEVICE, sub-diagonal [m], dl[0] must be 0.
 * @param d                   IN, DEVICE, main diagonal [m].
 * @param du                  IN, DEVICE, super-diagonal [m], du[m-1] must be 0.
 * @param B                   IN, DEVICE, right-hand side [ldb * n].
 * @param ldb                 IN, HOST, leading dimension of B (>= max(1, m)).
 * @param pBufferSizeInBytes  OUT, HOST, required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsv2Nopivot_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const float *dl, const float *d, const float *du,
    const float *B, int ldb, size_t *pBufferSizeInBytes);

// ============================================================================
// Legacy API: aclsparseSgtsv2
// ============================================================================

/**
 * @brief Solve a tridiagonal linear system with partial pivoting (Thomas + pivoting).
 *
 * Solves A * X = B where A is an m x m tridiagonal matrix defined by
 * sub-diagonal dl, main diagonal d, and super-diagonal du.
 * B is an m x n right-hand side matrix (column-major, leading dimension ldb).
 * X is the solution, stored in B on output (in-place).
 *
 * Uses the Thomas algorithm with partial pivoting.
 * Partial pivoting compares |d[i-1]| vs |dl[i]|
 * at each forward elimination step, swapping rows when |dl[i]| > |d[i-1]|
 * for numerical stability.
 *
 * Constraints (user responsibility, NOT checked at runtime):
 *   - dl[0] == 0, du[m-1] == 0
 *   - Matrix must be non-singular (zero pivot → Inf/NaN propagation, no error)
 *
 * @param handle        IN, HOST, aclsparse handle.
 * @param m             IN, HOST, system size (m >= 3 when n > 0; n=0 skips m validation).
 * @param n             IN, HOST, number of right-hand sides (n >= 0; n=0 early return).
 * @param dl            IN, DEVICE, sub-diagonal [m], dl[0] must be 0.
 * @param d             IN, DEVICE, main diagonal [m].
 * @param du            IN, DEVICE, super-diagonal [m], du[m-1] must be 0.
 * @param B             IN/OUT, DEVICE, RHS → solution [ldb * n] (column-major).
 * @param ldb           IN, HOST, leading dimension of B (ldb >= m).
 * @param pBuffer       IN, DEVICE, workspace (128-byte aligned), size from bufferSizeExt.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsv2(
    aclsparseHandle_t handle, int m, int n,
    const float *dl, const float *d, const float *du,
    float *B, int ldb, void *pBuffer);

/**
 * @brief Query workspace buffer size for aclsparseSgtsv2.
 *
 * @param handle              IN, HOST, aclsparse handle.
 * @param m                   IN, HOST, system size (m >= 3 when n > 0; n=0 skips m validation).
 * @param n                   IN, HOST, number of right-hand sides (n >= 0; n=0 returns 0).
 * @param dl                  IN, DEVICE, sub-diagonal [m], dl[0] must be 0.
 * @param d                   IN, DEVICE, main diagonal [m].
 * @param du                  IN, DEVICE, super-diagonal [m], du[m-1] must be 0.
 * @param B                   IN, DEVICE, right-hand side [ldb * n] (may be NULL for query).
 * @param ldb                 IN, HOST, leading dimension of B (>= max(1, m)).
 * @param pBufferSizeInBytes  OUT, HOST, required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgtsv2_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n,
    const float *dl, const float *d, const float *du,
    const float *B, int ldb, size_t *pBufferSizeInBytes);

// ============================================================================
// Legacy API: aclsparseSgpsvInterleavedBatch
// ============================================================================

/**
 * @brief Solve a batch of pentadiagonal linear systems with interleaved data layout.
 *
 * Solves A^(k) * x^(k) = b^(k) for k = 0, ..., batchCount-1 using QR
 * factorization with Givens rotations (algo=0). Each A^(k) is an m x m
 * pentadiagonal matrix defined by five diagonals: ds (distance-2 sub),
 * dl (sub), d (main), du (super), dw (distance-2 super).
 * Data layout: row-major interleaved, array[row * batchCount + batch].
 * Boundary conditions (user responsibility):
 *   ds[0][*] = ds[1][*] = 0, dl[0][*] = 0,
 *   du[m-1][*] = 0, dw[m-2][*] = dw[m-1][*] = 0.
 *
 * The QR algorithm applies a sequence of Givens rotations to eliminate the
 * two sub-diagonals (ds and dl), yielding an upper triangular band matrix R
 * with fill-in creating 4th and 5th upper diagonals (d4' = R[i,i+3] and
 * d5' = R[i,i+4]). The system is then solved by back-substitution. The algorithm is numerically stable for
 * general pentadiagonal systems, including non-diagonally-dominant matrices.
 *
 * Singular/near-singular input: no singularity guards are applied. If the
 * matrix is singular or near-singular, IEEE-754 division by zero produces
 * Inf/NaN in the output. The caller must ensure input matrices are
 * well-conditioned; otherwise the solution for that batch is undefined.
 *
 * @param handle        IN, HOST, aclsparse handle.
 * @param algo          IN, HOST, algorithm selector (0 = QR/Givens rotation).
 * @param m             IN, HOST, system size (rows = cols), m >= 1.
 * @param ds            IN, DEVICE, distance-2 sub-diagonal [m * batchCount].
 * @param dl            IN, DEVICE, sub-diagonal [m * batchCount].
 * @param d             IN, DEVICE, main diagonal [m * batchCount].
 * @param du            IN, DEVICE, super-diagonal [m * batchCount].
 * @param dw            IN, DEVICE, distance-2 super-diagonal [m * batchCount].
 * @param x             IN/OUT, DEVICE, right-hand side b on input, solution x on output [m * batchCount].
 * @param batchCount    IN, HOST, number of systems, batchCount >= 1.
 * @param pBuffer       IN, DEVICE, workspace buffer (128-byte aligned), size from bufferSizeExt.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgpsvInterleavedBatch(
    aclsparseHandle_t handle, int algo, int m,
    float *ds, float *dl, float *d, float *du, float *dw, float *x,
    int batchCount, void *pBuffer);

/**
 * @brief Query workspace buffer size for aclsparseSgpsvInterleavedBatch.
 *
 * @param handle              IN, HOST, aclsparse handle.
 * @param algo                IN, HOST, algorithm selector (0 = QR/Givens rotation).
 * @param m                   IN, HOST, system size (m >= 1).
 * @param ds                  IN, DEVICE, distance-2 sub-diagonal (query ignores pointer; may be NULL).
 * @param dl                  IN, DEVICE, sub-diagonal (query ignores pointer; may be NULL).
 * @param d                   IN, DEVICE, main diagonal (query ignores pointer; may be NULL).
 * @param du                  IN, DEVICE, super-diagonal (query ignores pointer; may be NULL).
 * @param dw                  IN, DEVICE, distance-2 super-diagonal (query ignores pointer; may be NULL).
 * @param x                   IN, DEVICE, right-hand side (query ignores pointer; may be NULL).
 * @param batchCount          IN, HOST, number of systems (>= 1).
 * @param pBufferSizeInBytes  OUT, HOST, required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseSgpsvInterleavedBatch_bufferSizeExt(
    aclsparseHandle_t handle, int algo, int m,
    const float *ds, const float *dl, const float *d, const float *du, const float *dw, const float *x,
    int batchCount, size_t *pBufferSizeInBytes);

// ============================================================================
// Legacy API: aclsparseCsr2cscEx2 — CSR to CSC format conversion
// ============================================================================

// Action type for format conversion: whether to copy values or only structure.
typedef enum aclsparseAction_t {
    ACL_SPARSE_ACTION_SYMBOLIC = 0,  // Only compute output structure (colPtr, rowInd).
    ACL_SPARSE_ACTION_NUMERIC        // Compute structure and copy values.
} aclsparseAction_t;

// Algorithm selector for CSR to CSC conversion.
typedef enum aclsparseCsr2CscAlg_t {
    ACL_SPARSE_CSR2CSC_ALG_DEFAULT = 0,
    ACL_SPARSE_CSR2CSC_ALG1
} aclsparseCsr2CscAlg_t;

/**
 * @brief Query workspace size for aclsparseCsr2cscEx2.
 *
 * @param handle       IN, HOST, aclsparse handle.
 * @param m            IN, HOST, number of rows (CSR) / columns (CSC).
 * @param n            IN, HOST, number of columns (CSR) / rows (CSC).
 * @param nnz          IN, HOST, number of nonzero elements.
 * @param csrVal       IN, DEVICE, CSR nonzero values (length nnz).
 * @param csrRowPtr    IN, DEVICE, CSR row pointers (length m+1).
 * @param csrColInd    IN, DEVICE, CSR column indices (length nnz).
 * @param cscVal       IN, DEVICE, CSC nonzero values (may be NULL for query).
 * @param cscColPtr    IN, DEVICE, CSC column pointers (may be NULL for query).
 * @param cscRowInd    IN, DEVICE, CSC row indices (may be NULL for query).
 * @param valType      IN, HOST, data type of nonzero values.
 * @param copyValues   IN, HOST, SYMBOLIC (structure only) or NUMERIC (structure + values).
 * @param idxBase      IN, HOST, index base (0-based or 1-based).
 * @param alg          IN, HOST, algorithm selector (ALG_DEFAULT).
 * @param bufferSize   OUT, HOST, required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseCsr2cscEx2_bufferSize(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg,
    size_t *bufferSize);

/**
 * @brief Convert a sparse matrix from CSR format to CSC format.
 *
 * Performs the format conversion: given CSR (csrVal, csrRowPtr, csrColInd),
 * produce CSC (cscVal, cscColPtr, cscRowInd). When copyValues is SYMBOLIC,
 * only cscColPtr and cscRowInd are computed; cscVal is left unmodified.
 * When copyValues is NUMERIC, all three output arrays are populated.
 *
 * @param handle       IN, HOST, aclsparse handle.
 * @param m            IN, HOST, number of rows (CSR) / columns (CSC).
 * @param n            IN, HOST, number of columns (CSR) / rows (CSC).
 * @param nnz          IN, HOST, number of nonzero elements.
 * @param csrVal       IN, DEVICE, CSR nonzero values (length nnz).
 * @param csrRowPtr    IN, DEVICE, CSR row pointers (length m+1).
 * @param csrColInd    IN, DEVICE, CSR column indices (length nnz).
 * @param cscVal       OUT, DEVICE, CSC nonzero values (length nnz).
 * @param cscColPtr    OUT, DEVICE, CSC column pointers (length n+1).
 * @param cscRowInd    OUT, DEVICE, CSC row indices (length nnz).
 * @param valType      IN, HOST, data type of nonzero values.
 * @param copyValues   IN, HOST, SYMBOLIC (structure only) or NUMERIC (structure + values).
 * @param idxBase      IN, HOST, index base (0-based or 1-based).
 * @param alg          IN, HOST, algorithm selector (ALG_DEFAULT).
 * @param buffer       IN, DEVICE, workspace (size from bufferSize query).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseCsr2cscEx2(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd,
    aclDataType valType, aclsparseAction_t copyValues,
    aclsparseIndexBase_t idxBase, aclsparseCsr2CscAlg_t alg,
    void *buffer);

// ============================================================================
// Legacy API: aclsparseGebsr2gebsc — GEBSR to GEBSC format conversion
// ============================================================================
// Converts General Block Sparse Row (GEBSR) format to General Block Sparse
// Column (GEBSC) format. Equivalent to csr2csc when each block is regarded
// as a scalar: the block sparsity pattern is transposed (block (i,j) -> (j,i)).
//
// Block value copy modes (determined by rowBlockDimC/colBlockDimC vs
// rowBlockDimA/colBlockDimA):
//   - Direct copy:  rC==rA && cC==cA — block elements copied verbatim
//   - Block transpose: rC==cA && cC==rA — block elements transposed
//
// dirA specifies the in-block memory layout (ROW=row-major, COLUMN=column-major)
// for both input and output blocks (layout does not change).

/**
 * @brief Query workspace size for aclsparseGebsr2gebsc.
 * @param handle           IN, HOST, aclsparse handle.
 * @param mb               IN, HOST, number of block rows.
 * @param nb               IN, HOST, number of block columns.
 * @param nnzb             IN, HOST, number of nonzero blocks.
 * @param bsrValA          IN, DEVICE, GEBSR nonzero block values (may be NULL for query).
 * @param bsrRowPtrA       IN, DEVICE, GEBSR block row pointers (length mb+1).
 * @param bsrColIndA       IN, DEVICE, GEBSR block column indices (length nnzb).
 * @param rowBlockDimA     IN, HOST, row dimension of input blocks.
 * @param colBlockDimA     IN, HOST, column dimension of input blocks.
 * @param dirA             IN, HOST, in-block memory layout (ROW or COLUMN).
 * @param valType          IN, HOST, data type of nonzero block values.
 * @param pBufferSizeInBytes OUT, HOST, required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseGebsr2gebsc_bufferSize(
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    aclsparseDirection_t dirA,
    aclDataType valType,
    size_t *pBufferSizeInBytes);

/**
 * @brief Convert GEBSR to GEBSC.
 * @param handle           IN, HOST, aclsparse handle.
 * @param mb               IN, HOST, number of block rows.
 * @param nb               IN, HOST, number of block columns.
 * @param nnzb             IN, HOST, number of nonzero blocks.
 * @param bsrValA          IN, DEVICE, GEBSR nonzero block values.
 * @param bsrRowPtrA       IN, DEVICE, GEBSR block row pointers (length mb+1).
 * @param bsrColIndA       IN, DEVICE, GEBSR block column indices (length nnzb).
 * @param rowBlockDimA     IN, HOST, row dimension of input blocks.
 * @param colBlockDimA     IN, HOST, column dimension of input blocks.
 * @param bscVal           OUT, DEVICE, GEBSC nonzero block values.
 * @param bscColPtr        OUT, DEVICE, GEBSC block column pointers (length nb+1).
 * @param bscRowInd        OUT, DEVICE, GEBSC block row indices (length nnzb).
 * @param rowBlockDimC     IN, HOST, row dimension of output blocks.
 * @param colBlockDimC     IN, HOST, column dimension of output blocks.
 * @param copyValues       IN, HOST, SYMBOLIC (structure only) or NUMERIC (structure + values).
 * @param idxBase          IN, HOST, index base (0-based or 1-based).
 * @param dirA             IN, HOST, in-block memory layout (ROW or COLUMN).
 * @param valType          IN, HOST, data type of nonzero block values.
 * @param pBuffer          IN, DEVICE, workspace (size from bufferSize query).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseGebsr2gebsc(
    aclsparseHandle_t handle, int mb, int nb, int nnzb,
    const void *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
    int rowBlockDimA, int colBlockDimA,
    void *bscVal, int *bscColPtr, int *bscRowInd,
    int rowBlockDimC, int colBlockDimC,
    aclsparseAction_t copyValues, aclsparseIndexBase_t idxBase,
    aclsparseDirection_t dirA,
    aclDataType valType,
    void *pBuffer);

// ============================================================================
// Legacy API: aclsparseXcoosort — COO format in-place stable sort (dual-key)
// ============================================================================

/**
 * @brief Query workspace size for aclsparseXcoosortByRow / ByColumn.
 *
 * @param handle               IN,  HOST,  aclsparse handle.
 * @param m                    IN,  HOST,  number of rows (not used by sort, must be positive).
 * @param n                    IN,  HOST,  number of columns (not used by sort, must be positive).
 * @param nnz                  IN,  HOST,  number of nonzero elements.
 * @param cooRowsA             IN,  DEVICE, COO row indices (length nnz); may be NULL when nnz==0.
 * @param cooColsA             IN,  DEVICE, COO column indices (length nnz); may be NULL when nnz==0.
 * @param pBufferSizeInBytes   OUT, HOST,  required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcoosort_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const int *cooRowsA, const int *cooColsA,
    size_t *pBufferSizeInBytes);

/**
 * @brief Stable-sort COO arrays by row (dual-key: row ascending, then col ascending).
 *
 * In-place sort. P is both input (caller presets 0:1:(nnz-1)) and output (sort
 * permutation). pBuffer has no alignment requirement (ascend950). m/n are not
 * used by the sort (only validated positive).
 *
 * @param handle    IN,  HOST,    aclsparse handle.
 * @param m         IN,  HOST,    number of rows (must be positive).
 * @param n         IN,  HOST,    number of columns (must be positive).
 * @param nnz       IN,  HOST,    number of nonzero elements.
 * @param cooRowsA  IN/OUT, DEVICE, row indices (sorted ascending on output).
 * @param cooColsA  IN/OUT, DEVICE, column indices (reordered to follow cooRowsA).
 * @param P         IN/OUT, DEVICE, permutation (length nnz); sortedVal[i] = origVal(P[i]).
 * @param pBuffer   IN,  DEVICE,  workspace (from bufferSizeExt), no alignment required.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcoosortByRow(
    aclsparseHandle_t handle, int m, int n, int nnz,
    int *cooRowsA, int *cooColsA, int *P, void *pBuffer);

/**
 * @brief Stable-sort COO arrays by column (dual-key: col ascending, then row ascending).
 *        Same semantics as aclsparseXcoosortByRow but primary key is the column index.
 */
aclsparseStatus_t aclsparseXcoosortByColumn(
    aclsparseHandle_t handle, int m, int n, int nnz,
    int *cooRowsA, int *cooColsA, int *P, void *pBuffer);
// Generic API: aclsparseSpMVOp — 7 函数
// ============================================================================

/**
 * @brief 获取 SpMVOp 所需 workspace 大小（字节数）。
 *        ALG1: 返回 0；ALG2: 返回 reorder 表 + bin_edge 表 + 对齐开销。
 */
aclsparseStatus_t aclsparseSpMVOp_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ,
    aclDataType computeType,
    aclsparseSpMVOpAlg_t alg,
    size_t *bufferSize);

/**
 * @brief 创建 SpMVOp 内部描述符并执行预处理。
 *        ALG2: 在 device 上通过 kernel 异步完成排序和 bin_edge 负载均衡切分。
 *        buffer 由调用者在 bufferSize 后分配，保持有效直到 descr 销毁。
 */
aclsparseStatus_t aclsparseSpMVOp_createDescr(
    aclsparseHandle_t handle, aclsparseSpMVOpDescr_t *descr, aclsparseOperation_t opA, aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnVecDescr_t vecX, aclsparseConstDnVecDescr_t vecY, aclsparseDnVecDescr_t vecZ,
    aclDataType computeType, aclsparseSpMVOpAlg_t alg, void *buffer);

/**
 * @brief 销毁 SpMVOp 描述符并释放其管理的 host 资源。
 *        descr 为 nullptr 时直接返回 SUCCESS（幂等语义）。
 */
aclsparseStatus_t aclsparseSpMVOp_destroyDescr(aclsparseSpMVOpDescr_t descr);

/**
 * @brief 基于 descr 创建执行计划。
 *        NPU 侧 identity epilogue: epilogueLTOBuffer 必须为 NULL，否则返回 NOT_SUPPORTED。
 */
aclsparseStatus_t aclsparseSpMVOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMVOpDescr_t descr,        // 非 const：对标 cuSPARSE，createPlan 可修改 descr
    aclsparseSpMVOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize);

/**
 * @brief 销毁 SpMVOp 执行计划。
 *        plan 为 nullptr 时直接返回 SUCCESS（幂等语义）。
 */
aclsparseStatus_t aclsparseSpMVOp_destroyPlan(aclsparseSpMVOpPlan_t plan);

/**
 * @brief 将 epilogue 模块中 __constant__ 变量设置为指定数据。
 *        NPU 侧 identity epilogue 不使用此接口，直接返回 SUCCESS。
 */
aclsparseStatus_t aclsparseSpMVOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMVOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize);

/**
 * @brief 执行稀疏矩阵向量乘法：Z = alpha * A * X + beta * Y
 *        使用确定性累加（Priest 双重补偿求和）保证 bit-wise 可重复。
 */
aclsparseStatus_t aclsparseSpMVOp(
    aclsparseHandle_t handle,
    aclsparseSpMVOpPlan_t plan,
    const void *alpha,
    const void *beta,
    aclsparseConstDnVecDescr_t vecX,
    aclsparseConstDnVecDescr_t vecY,
    aclsparseDnVecDescr_t vecZ);

// ============================================================================
// Generic API: aclsparseSpMMOp — 7 函数
// ============================================================================
//  稀疏矩阵-稠密矩阵乘法（descriptor/plan 生命周期模式）。
//  数学公式: C = alpha * op(A) * op(B) + beta * C
//    A: CSR(m×k)，B: dense(k×n 或 n×k)，C: dense(m×n，in-place)
//  生命周期: bufferSize → createDescr → createPlan → [setGlobalUserData]
//            → execute(可重复) → destroyPlan → destroyDescr。
//  matA 在 createDescr 阶段绑定（CSR pattern 固定用于 plan 生命周期）；
//  matB/matC 在 execute 阶段传入（可跨多次 execute 更换，n/ldb/ldc/order 可变）。
//  对标 cuSPARSE Generic SpMM 语义 + 仓内 spmv_op 7 函数结构。

// SpMMOp 算法枚举（对标 Generic API 中 SpMMOp 的算法选项）
// 注意：新增枚举值须追加到末尾，保持既有枚举数值不变（ABI 兼容性）。
typedef enum aclsparseSpMMOpAlg_t {
    ACL_SPARSE_SPMMOP_ALG_DEFAULT = 0,  // 路由到 ALG1, 无 workspace
    ACL_SPARSE_SPMMOP_ALG1,             // host-side 静态行切分, 无 workspace, 支持 csrValues 原地更新
    ACL_SPARSE_SPMMOP_ALG2,             // host-side merge sort (std::stable_sort) + bin_edge 负载均衡, 需 workspace
    ACL_SPARSE_SPMMOP_ALG1_HIGH_PRECISION  // FP32 Kahan 补偿求和，仅 FP32 生效；workspace 语义同 ALG1
} aclsparseSpMMOpAlg_t;

// SpMMOp opaque types
struct aclsparseSpMMOpDescr;
struct aclsparseSpMMOpPlan;

typedef struct aclsparseSpMMOpDescr* aclsparseSpMMOpDescr_t;
typedef struct aclsparseSpMMOpDescr const *aclsparseConstSpMMOpDescr_t;
typedef struct aclsparseSpMMOpPlan* aclsparseSpMMOpPlan_t;

/**
 * @brief 获取 SpMMOp 所需 workspace 大小（字节数）。
 *        ALG1/DEFAULT: 返回 0；ALG2: 返回 reorder 表 + bin_edge 表 + 对齐开销。
 */
aclsparseStatus_t aclsparseSpMMOp_bufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    size_t *bufferSize);

/**
 * @brief 创建 SpMMOp 内部描述符并执行预处理。
 *        ALG2: 在 host 侧同步完成排序（std::stable_sort）和 bin_edge 负载均衡切分。
 *        buffer 由调用者在 bufferSize 后分配，保持有效直到 descr 销毁。
 *        matA 在此阶段绑定（CSR pattern 固定用于 plan 生命周期）。
 */
aclsparseStatus_t aclsparseSpMMOp_createDescr(
    aclsparseHandle_t handle,
    aclsparseSpMMOpDescr_t *descr,
    aclsparseOperation_t opA,
    aclsparseOperation_t opB,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpMMOpAlg_t alg,
    void *buffer);

/**
 * @brief 销毁 SpMMOp 描述符并释放其管理的 host 资源。
 *        descr 为 nullptr 时直接返回 SUCCESS（幂等语义）。
 */
aclsparseStatus_t aclsparseSpMMOp_destroyDescr(aclsparseSpMMOpDescr_t descr);

/**
 * @brief 基于 descr 创建执行计划。
 *        NPU 侧 identity epilogue: epilogueLTOBuffer 必须为 NULL，否则返回 NOT_SUPPORTED。
 */
aclsparseStatus_t aclsparseSpMMOp_createPlan(
    aclsparseHandle_t handle,
    aclsparseSpMMOpDescr_t descr,        // 非 const：对标 cuSPARSE，createPlan 可修改 descr
    aclsparseSpMMOpPlan_t *plan,
    const void *epilogueLTOBuffer,
    size_t epilogueLTOBufferSize);

/**
 * @brief 销毁 SpMMOp 执行计划。
 *        plan 为 nullptr 时直接返回 SUCCESS（幂等语义）。
 */
aclsparseStatus_t aclsparseSpMMOp_destroyPlan(aclsparseSpMMOpPlan_t plan);

/**
 * @brief 将 epilogue 模块中 __constant__ 变量设置为指定数据。
 *        NPU 侧 identity epilogue 不使用此接口，直接返回 SUCCESS。
 */
aclsparseStatus_t aclsparseSpMMOp_setGlobalUserData(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const char *epilogueDataName,
    void *epilogueData,
    size_t epilogueDataSize);

/**
 * @brief 执行稀疏矩阵-稠密矩阵乘法：C = alpha * op(A) * op(B) + beta * C
 *        matB/matC 在 execute 阶段传入（可跨多次 execute 更换）。
 *        支持 FP32 / FP16（A/B/C dtype 一致，FP32 累加）。
 *        支持 opB ∈ {NON_TRANSPOSE, TRANSPOSE}，B/C 独立支持 ROW/COL 主序。
 *        异步执行，同步由调用方通过 stream 负责。
 */
aclsparseStatus_t aclsparseSpMMOp(
    aclsparseHandle_t handle,
    aclsparseSpMMOpPlan_t plan,
    const void *alpha,
    const void *beta,
    aclsparseConstDnMatDescr_t matB,
    aclsparseDnMatDescr_t matC);

// ============================================================================
// Generic API: aclsparseScatter
// ============================================================================

/**
 * @brief 将稀疏向量 X 的非零值散布到稠密向量 Y 的对应位置。
 *
 * 执行操作：Y[X.indices[i]] = X.values[i]，i ∈ [0, X.nnz)。
 * - idxBase = ONE 时，indices 基于 1，内部访问 Y 时减 1 调整。
 * - 索引重复时 last-write-wins（行为不确定，与 cuSPARSE 一致）。
 * - 索引未排序允许。
 * - 支持 valueType ∈ {INT8, FP16, BF16, FP32, COMPLEX64}（vecX.values 与 vecY dtype 须一致，
 *   complex64 为 2×fp32 交错存储）；idxType 仅 I32；idxBase ∈ {0, 1}。
 * - nnz=0 时成功返回且不修改 vecY；非法参数（空句柄/描述符、非法 idxType/idxBase/
 *   dtype、nnz>0 且 size=0、未声明 alias 等）返回确定参数错误。
 * - 异步执行，同步由调用方通过 stream 负责。
 *
 * @param handle IN, HOST, aclsparse 库上下文句柄。
 * @param vecX   IN, HOST(descr)/DEVICE(data), 稀疏向量描述符（只读）。
 * @param vecY   IN/OUT, HOST(descr)/DEVICE(data), 稠密向量描述符，其 values 被 X 散布写入。
 * @return aclsparseStatus_t 成功返回 ACL_SPARSE_STATUS_SUCCESS。
 */
aclsparseStatus_t aclsparseScatter(
    aclsparseHandle_t handle,
    aclsparseConstSpVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY);

// ============================================================================
// Legacy API: aclsparseXcsrsort
// ============================================================================

/**
 * @brief 查询 aclsparseXcsrsort 所需的 Device workspace 大小。
 *
 * @param handle             IN, HOST, aclsparse handle。
 * @param m                  IN, HOST, 矩阵行数。
 * @param n                  IN, HOST, 矩阵列数。
 * @param nnz                IN, HOST, 非零元素数量。
 * @param csrRowPtr          IN, DEVICE, CSR 行指针，长度 m+1。
 * @param csrColInd          IN, DEVICE, CSR 列索引，长度 nnz。
 * @param pBufferSizeInBytes OUT, HOST, workspace 大小（字节）。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcsrsort_bufferSizeExt(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int *csrRowPtr,
    const int *csrColInd,
    size_t *pBufferSizeInBytes);

/**
 * @brief 对 CSR 每行的列索引执行原地稳定排序，并输出排列 P。
 *
 * @param handle     IN, HOST, aclsparse handle，提供执行 stream。
 * @param m          IN, HOST, 矩阵行数。
 * @param n          IN, HOST, 矩阵列数。
 * @param nnz        IN, HOST, 非零元素数量。
 * @param descrA     IN, HOST, Legacy MatDescr；使用 indexBase，其他属性忽略。
 * @param csrRowPtr  IN, DEVICE, CSR 行指针，长度 m+1，排序期间不变。
 * @param csrColInd  IN/OUT, DEVICE, 长度 nnz；输出时每行内部升序。
 * @param P          IN/OUT, DEVICE, 长度 nnz；与 csrColInd 使用相同的稳定排列。
 * @param pBuffer    IN, DEVICE, 由 bufferSizeExt 查询并由调用方分配的 workspace。
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcsrsort(
    aclsparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtr,
    int *csrColInd,
    int *P,
    void *pBuffer);

// ============================================================================
// Generic API: aclsparseSpSV — Sparse Triangular Solve (Vector)
// ============================================================================

typedef enum aclsparseSpSVAlg_t {
    ACL_SPARSE_SPSV_ALG_DEFAULT = 0
} aclsparseSpSVAlg_t;

typedef enum aclsparseSpSVUpdate_t {
    ACL_SPARSE_SPSV_UPDATE_GENERAL = 0,
    ACL_SPARSE_SPSV_UPDATE_DIAGONAL
} aclsparseSpSVUpdate_t;

struct aclsparseSpSVDescr;
typedef struct aclsparseSpSVDescr* aclsparseSpSVDescr_t;

typedef enum aclsparseSpMatAttribute_t {
    ACL_SPARSE_SPMAT_FILL_MODE = 0,
    ACL_SPARSE_SPMAT_DIAG_TYPE
} aclsparseSpMatAttribute_t;

aclsparseStatus_t aclsparseSpSV_createDescr(aclsparseSpSVDescr_t *spsvDescr);
aclsparseStatus_t aclsparseSpSV_destroyDescr(aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_bufferSize(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, size_t *bufferSize);

aclsparseStatus_t aclsparseSpSV_analysis(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr, void *externalBuffer);

aclsparseStatus_t aclsparseSpSV_solve(
    aclsparseHandle_t handle, aclsparseOperation_t opA, const void *alpha,
    aclsparseConstSpMatDescr_t matA, aclsparseConstDnVecDescr_t vecX,
    aclsparseDnVecDescr_t vecY, aclDataType computeType,
    aclsparseSpSVAlg_t alg, aclsparseSpSVDescr_t spsvDescr);

aclsparseStatus_t aclsparseSpSV_updateMatrix(
    aclsparseHandle_t handle, aclsparseSpSVDescr_t spsvDescr,
    void *newValues, aclsparseSpSVUpdate_t updatePart);

aclsparseStatus_t aclsparseSpMatSetAttribute(
    aclsparseSpMatDescr_t spMatDescr, aclsparseSpMatAttribute_t attribute,
    const void *data, size_t dataSize);

aclsparseStatus_t aclsparseSpMatGetAttribute(
    aclsparseConstSpMatDescr_t spMatDescr, aclsparseSpMatAttribute_t attribute,
    void *data, size_t dataSize);

// ============================================================================
// Sparse Matrix Creation: COO / SLICED_ELL
// ============================================================================

aclsparseStatus_t aclsparseCreateCoo(aclsparseSpMatDescr_t *spMatDescr,
    int64_t rows, int64_t cols, int64_t nnz,
    void *cooRowInd, void *cooColInd, void *cooValues,
    aclsparseIndexType_t cooIdxType, aclsparseIndexBase_t idxBase,
    aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstCoo(aclsparseConstSpMatDescr_t *spMatDescr,
    int64_t rows, int64_t cols, int64_t nnz,
    const void *cooRowInd, const void *cooColInd, const void *cooValues,
    aclsparseIndexType_t cooIdxType, aclsparseIndexBase_t idxBase,
    aclDataType valueType);

aclsparseStatus_t aclsparseCooGet(aclsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz,
    void **cooRowInd, void **cooColInd, void **cooValues, aclsparseIndexType_t *cooIdxType,
    aclsparseIndexBase_t *idxBase, aclDataType *valueType);

aclsparseStatus_t aclsparseConstCooGet(aclsparseConstSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols,
    int64_t *nnz, const void **cooRowInd, const void **cooColInd, const void **cooValues,
    aclsparseIndexType_t *cooIdxType, aclsparseIndexBase_t *idxBase, aclDataType *valueType);

aclsparseStatus_t aclsparseCreateSlicedEll(aclsparseSpMatDescr_t *spMatDescr,
    int64_t rows, int64_t cols, int64_t nnz, int64_t sliceNnz, int64_t numSlices,
    void *sellSlicePtr, void *sellColInd, void *sellValues,
    aclsparseIndexType_t sellIdxType, aclsparseIndexBase_t idxBase,
    aclDataType valueType);

aclsparseStatus_t aclsparseCreateConstSlicedEll(aclsparseConstSpMatDescr_t *spMatDescr,
    int64_t rows, int64_t cols, int64_t nnz, int64_t sliceNnz, int64_t numSlices,
    const void *sellSlicePtr, const void *sellColInd, const void *sellValues,
    aclsparseIndexType_t sellIdxType, aclsparseIndexBase_t idxBase,
    aclDataType valueType);

// ============================================================================
// Legacy API: aclsparseXcoo2csr — COO row indices to CSR row pointers
// ============================================================================

/**
 * @brief Convert COO row indices to CSR row pointers.
 *
 * Given a sorted COO row index array cooRowInd[0..nnz-1], compute the CSR
 * row pointer array csrRowPtr[0..m] via counting + prefix sum.
 *
 * This is a type-agnostic (X-prefix) format conversion helper that operates
 * only on int32 index arrays. No floating-point data is involved.
 *
 * @param handle     IN, HOST, aclsparse handle.
 * @param cooRowInd  IN, DEVICE, COO row index array (length nnz, sorted non-decreasing).
 * @param nnz        IN, HOST, number of nonzero elements (>= 0).
 * @param m          IN, HOST, number of rows (>= 0).
 * @param csrRowPtr  OUT, DEVICE, CSR row pointer array (length m+1).
 * @param idxBase    IN, HOST, index base (ZERO or ONE).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcoo2csr(
    aclsparseHandle_t handle,
    const int *cooRowInd,
    int nnz,
    int m,
    int *csrRowPtr,
    aclsparseIndexBase_t idxBase);

// ============================================================================
// Legacy API: aclsparseXcscsort — CSC format in-place stable sort (single-key)
// ============================================================================

/**
 * @brief Query workspace size for aclsparseXcscsort.
 *
 * @param handle               IN,  HOST,  aclsparse handle.
 * @param m                    IN,  HOST,  number of rows.
 * @param n                    IN,  HOST,  number of columns.
 * @param nnz                  IN,  HOST,  number of nonzero elements.
 * @param cscColPtr            IN,  DEVICE, CSC column pointers (length n+1).
 * @param cscRowInd            IN,  DEVICE, CSC row indices (length nnz); may be NULL when nnz==0.
 * @param pBufferSizeInBytes   OUT, HOST,  required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcscsort_bufferSizeExt(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const int *cscColPtr, const int *cscRowInd,
    size_t *pBufferSizeInBytes);

/**
 * @brief Stable-sort CSC arrays by column (single-key: row index ascending within each column).
 *
 * In-place sort. P is both input (caller presets 0:1:(nnz-1)) and output (sort
 * permutation). pBuffer must be 128-byte aligned. cscColPtr is not modified.
 *
 * @param handle    IN,  HOST,    aclsparse handle.
 * @param m         IN,  HOST,    number of rows.
 * @param n         IN,  HOST,    number of columns.
 * @param nnz       IN,  HOST,    number of nonzero elements.
 * @param descrA    IN,  HOST,    matrix descriptor (provides indexBase).
 * @param cscColPtr IN,  DEVICE,  CSC column pointers (length n+1, not modified).
 * @param cscRowInd IN/OUT, DEVICE, row indices (sorted ascending within each column on output).
 * @param P         IN/OUT, DEVICE, permutation (length nnz); sortedVal[i] = origVal(P[i]).
 * @param pBuffer   IN,  DEVICE,  workspace (from bufferSizeExt), must be 128-byte aligned.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcscsort(
    aclsparseHandle_t handle, int m, int n, int nnz,
    const aclsparseMatDescr_t descrA,
    const int *cscColPtr, int *cscRowInd, int *P, void *pBuffer);

// ============================================================================
// Legacy API: aclsparseXcsr2coo — CSR row-pointer to COO row-index conversion
// ============================================================================

/**
 * @brief Convert CSR compressed row offset array to COO row index array.
 *
 * Expands csrRowPtr[m+1] into cooRowInd[nnz] where each element maps to its
 * originating row. The same function can perform CSC→COO conversion by passing
 * colPtr as csrRowPtr and column count as m. INT32 index only, no workspace.
 *
 * @param handle     IN,  HOST,    aclsparse handle.
 * @param csrRowPtr  IN,  DEVICE, CSR row offset array (length m+1), ascending.
 * @param nnz        IN,  HOST,    number of nonzero elements (>= 0).
 * @param m          IN,  HOST,    number of rows (>= 0). When m==0, nnz must be 0.
 * @param cooRowInd  OUT, DEVICE, COO row index array (length nnz).
 * @param idxBase    IN,  HOST,    index base: ACL_SPARSE_INDEX_BASE_ZERO or ONE.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcsr2coo(aclsparseHandle_t handle,
                                   const int32_t *csrRowPtr,
                                   int64_t nnz,
                                   int64_t m,
                                   int32_t *cooRowInd,
                                    aclsparseIndexBase_t idxBase);

// ============================================================================
// Legacy API: aclsparseCsr2gebsr — CSR → GEBSR 格式转换
// ============================================================================

// --- Step 1: bufferSize (4 精度版本) ---

/**
 * @brief Query workspace size for CSR→GEBSR conversion (float32).
 *
 * @param handle            IN,  HOST,    aclsparse handle.
 * @param dir               IN,  HOST,    block-internal layout (ROW or COLUMN).
 * @param m                 IN,  HOST,    number of rows (>= 0).
 * @param n                 IN,  HOST,    number of columns (>= 0).
 * @param descrA            IN,  HOST,    matrix descriptor (type=GENERAL, indexBase=ZERO/ONE).
 * @param csrValA           IN,  DEVICE,  CSR values (not read by bufferSize, may be nullptr).
 * @param csrRowPtrA        IN,  DEVICE,  CSR row offsets (length m+1).
 * @param csrColIndA        IN,  DEVICE,  CSR column indices (not read by bufferSize).
 * @param rowBlockDim       IN,  HOST,    block row dimension (> 0).
 * @param colBlockDim       IN,  HOST,    block column dimension (> 0).
 * @param pBufferSizeInBytes OUT, HOST,   required workspace size in bytes.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseScsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes);

/**
 * @brief Query workspace size for CSR→GEBSR conversion (float16).
 *        Parameters identical to aclsparseScsr2gebsr_bufferSize except csrValA type.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseHcsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes);

/**
 * @brief Query workspace size for CSR→GEBSR conversion (bfloat16).
 *        Parameters identical to aclsparseScsr2gebsr_bufferSize except csrValA type.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseBhcsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes);

/**
 * @brief Query workspace size for CSR→GEBSR conversion (int32).
 *        Parameters identical to aclsparseScsr2gebsr_bufferSize except csrValA type.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseIcsr2gebsr_bufferSize(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n, const aclsparseMatDescr_t descrA,
    const int *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    int rowBlockDim, int colBlockDim, size_t *pBufferSizeInBytes);

// --- Step 2: Nnz (类型无关) ---

/**
 * @brief Compute bsrRowPtrC and nonzero block count (nnzb) for CSR→GEBSR.
 *
 * Asynchronous (on stream). Caller must sync before calling Convert.
 * nnzTotalDevHostPtr is host pointer (HOST mode) or device pointer (DEVICE mode).
 *
 * @param handle             IN,  HOST,    aclsparse handle.
 * @param dir                IN,  HOST,    block-internal layout (ROW or COLUMN).
 * @param m                  IN,  HOST,    number of rows (>= 0).
 * @param n                  IN,  HOST,    number of columns (>= 0).
 * @param descrA             IN,  HOST,    input matrix descriptor.
 * @param csrRowPtrA         IN,  DEVICE,  CSR row offsets (length m+1).
 * @param csrColIndA         IN,  DEVICE,  CSR column indices (may be nullptr if nnz=0).
 * @param descrC             IN,  HOST,    output matrix descriptor.
 * @param bsrRowPtrC         OUT, DEVICE,  block row offsets (length mb+1).
 * @param rowBlockDim        IN,  HOST,    block row dimension (> 0).
 * @param colBlockDim        IN,  HOST,    block column dimension (> 0).
 * @param nnzTotalDevHostPtr OUT, HOST/DEVICE, nonzero block count.
 * @param pBuffer            IN,  DEVICE,  workspace (from bufferSize).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseXcsr2gebsrNnz(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    int *bsrRowPtrC,
    int rowBlockDim, int colBlockDim,
    int *nnzTotalDevHostPtr,
    void *pBuffer);

// --- Step 3: Convert (4 精度版本) ---

/**
 * @brief Execute CSR→GEBSR conversion, filling bsrColIndC and bsrValC (float32).
 *
 * Depends on prior Nnz completion on stream. Caller must sync before calling.
 *
 * @param handle       IN,  HOST,    aclsparse handle.
 * @param dir          IN,  HOST,    block-internal layout (ROW or COLUMN).
 * @param m            IN,  HOST,    number of rows (>= 0).
 * @param n            IN,  HOST,    number of columns (>= 0).
 * @param descrA       IN,  HOST,    input matrix descriptor.
 * @param csrValA      IN,  DEVICE,  CSR values (length nnz).
 * @param csrRowPtrA   IN,  DEVICE,  CSR row offsets (length m+1).
 * @param csrColIndA   IN,  DEVICE,  CSR column indices (length nnz).
 * @param descrC       IN,  HOST,    output matrix descriptor.
 * @param bsrValC      OUT, DEVICE,  GEBSR block values (length nnzb*rbd*cbd).
 * @param bsrRowPtrC   IN/OUT, DEVICE, block row offsets (length mb+1, from Nnz).
 * @param bsrColIndC   OUT, DEVICE,  block column indices (length nnzb).
 * @param rowBlockDim  IN,  HOST,    block row dimension (> 0).
 * @param colBlockDim  IN,  HOST,    block column dimension (> 0).
 * @param pBuffer      IN,  DEVICE,  workspace (from bufferSize).
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseScsr2gebsr(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    float *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void *pBuffer);

/**
 * @brief Execute CSR→GEBSR conversion (float16).
 *        Parameters identical to aclsparseScsr2gebsr except csrValA/bsrValC types.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseHcsr2gebsr(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    void *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void *pBuffer);

/**
 * @brief Execute CSR→GEBSR conversion (bfloat16).
 *        Parameters identical to aclsparseScsr2gebsr except csrValA/bsrValC types.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseBhcsr2gebsr(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const void *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    void *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void *pBuffer);

/**
 * @brief Execute CSR→GEBSR conversion (int32).
 *        Parameters identical to aclsparseScsr2gebsr except csrValA/bsrValC types.
 * @return aclsparseStatus_t
 */
aclsparseStatus_t aclsparseIcsr2gebsr(
    aclsparseHandle_t handle, aclsparseDirection_t dir,
    int m, int n,
    const aclsparseMatDescr_t descrA,
    const int *csrValA, const int *csrRowPtrA, const int *csrColIndA,
    const aclsparseMatDescr_t descrC,
    int *bsrValC, int *bsrRowPtrC, int *bsrColIndC,
    int rowBlockDim, int colBlockDim,
    void *pBuffer);

/**
 * @brief Create/destroy a reusable descriptor for multi-stage CSR SpGEMM.
 */
aclsparseStatus_t aclsparseSpGEMMCreateDescr(aclsparseSpGEMMDescr_t *descr);
aclsparseStatus_t aclsparseSpGEMMDestroyDescr(aclsparseSpGEMMDescr_t descr);

/**
 * @brief Stage 1: query/execute intermediate-product work estimation.
 *
 * Call with externalBuffer1 == nullptr to query bufferSize1, then call again
 * with a device buffer of at least that size. The execution call records the
 * exact intermediate product count in spgemmDescr.
 */
aclsparseStatus_t aclsparseSpGEMMWorkEstimation(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1);

/** @brief Return the exact product count after WorkEstimation execution. */
aclsparseStatus_t aclsparseSpGEMMGetNumProducts(
    aclsparseSpGEMMDescr_t spgemmDescr, int64_t *numProds);

/**
 * @brief Stage 2: estimate auxiliary and compute workspace sizes.
 *
 * ALG2/ALG3 consume chunkFraction in (0, 1]. The deterministic A5
 * implementation reports a zero-byte auxiliary buffer and a full exact
 * compute buffer; externalBuffer3 is reserved for source compatibility.
 */
aclsparseStatus_t aclsparseSpGEMMEstimateMemory(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    float chunkFraction,
    size_t *bufferSize3, void *externalBuffer3,
    size_t *bufferSize2);

/**
 * @brief Stage 3: query/execute deterministic CSR structure and value compute.
 *
 * The execution call writes matC row offsets and updates the descriptor nnz.
 * Call aclsparseCsrSetPointers after it to attach output column/value storage.
 */
aclsparseStatus_t aclsparseSpGEMMCompute(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2);

/** @brief Stage 4: asynchronously compact computed columns/values into matC. */
aclsparseStatus_t aclsparseSpGEMMCopy(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    aclsparseSpGEMMDescr_t spgemmDescr);

/**
 * @brief 保留旧版 C 输入有效标记接口以兼容已有调用方。
 *
 * 新多阶段接口根据 beta 和 matC 指针直接判定 C 输入；该接口保留调用语义并记录标记，
 * 已有源码无需删除调用。
 */
aclsparseStatus_t aclsparseSpGEMMSetCInValid(
    aclsparseSpGEMMDescr_t spgemmDescr, int32_t cInDataValid);

/** @brief 兼容接口：查询旧三阶段 SpGEMM 共用 workspace 大小。 */
aclsparseStatus_t aclsparseSpGEMMGetBufferSize(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    size_t *size);

/** @brief 兼容接口：执行旧三阶段 SpGEMM 的结构预处理。 */
aclsparseStatus_t aclsparseSpGEMMPreprocess(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    void *buffer);

/** @brief 兼容接口：执行旧三阶段 SpGEMM 的数值计算。 */
aclsparseStatus_t aclsparseSpGEMM(
    aclsparseHandle_t handle,
    aclsparseOperation_t opA, aclsparseOperation_t opB,
    const void *alpha,
    aclsparseConstSpMatDescr_t matA,
    aclsparseConstSpMatDescr_t matB,
    const void *beta,
    aclsparseSpMatDescr_t matC,
    aclDataType computeType,
    aclsparseSpGEMMAlg_t alg,
    void *buffer);

#ifdef __cplusplus
}
#endif

#endif
