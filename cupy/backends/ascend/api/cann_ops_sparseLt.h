/**
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

#ifndef CANN_OPS_SPARSELT_H_
#define CANN_OPS_SPARSELT_H_

#include "cann_ops_sparse.h"

// aclsparseLtHandle_t: opaque handle for the aclsparseLt library context.
struct aclsparseLtContext;
typedef struct aclsparseLtContext* aclsparseLtHandle_t;

// aclsparseLtConstHandle_t: const pointer to handle (NVIDIA cusparseLtHandle_t*).
// All public APIs accept the handle as this type per cuSPARSELt convention.
typedef const aclsparseLtHandle_t* aclsparseLtConstHandle_t;

/* ========== Descriptor Types ========== */

// aclsparseLtMatDescriptor_t: opaque pointer for a matrix descriptor (dense / structured).
struct aclsparseLtMatDescriptor;
typedef struct aclsparseLtMatDescriptor* aclsparseLtMatDescriptor_t;

// aclsparseLtMatmulDescriptor_t: opaque pointer for a matmul descriptor.
struct aclsparseLtMatmulDescriptor;
typedef struct aclsparseLtMatmulDescriptor* aclsparseLtMatmulDescriptor_t;

// aclsparseLtMatmulAlgSelection_t: opaque pointer for a matmul algorithm selection descriptor.
struct aclsparseLtMatmulAlgSelection;
typedef struct aclsparseLtMatmulAlgSelection* aclsparseLtMatmulAlgSelection_t;

// aclsparseLtMatmulPlan_t: opaque pointer for a matmul execution plan.
struct aclsparseLtMatmulPlan;
typedef struct aclsparseLtMatmulPlan* aclsparseLtMatmulPlan_t;

typedef const struct aclsparseLtMatDescriptor* aclsparseLtConstMatDescriptor_t;
typedef const struct aclsparseLtMatmulDescriptor* aclsparseLtConstMatmulDescriptor_t;
typedef const struct aclsparseLtMatmulAlgSelection* aclsparseLtConstMatmulAlgSelection_t;
typedef const struct aclsparseLtMatmulPlan* aclsparseLtConstMatmulPlan_t;

// Structured sparsity mode.
typedef enum aclsparseLtSparsity_t {
    ACL_SPARSE_LT_SPARSITY_50_PERCENT = 0,  // 2:4 structured sparsity
    ACL_SPARSE_LT_SPARSITY_NONE = 1,        // dense (PR extension, identifies dense matrix)
} aclsparseLtSparsity_t;

// Compute precision
typedef enum aclsparseComputeType_t {
    ACL_SPARSE_COMPUTE_16F = 0,
    ACL_SPARSE_COMPUTE_32F,
    ACL_SPARSE_COMPUTE_32I
} aclsparseComputeType_t;

// Matmul algorithm mode (aligned with cusparseLtMatmulAlg_t).
typedef enum aclsparseLtMatmulAlg_t {
    ACL_SPARSE_LT_MATMUL_ALG_DEFAULT = 0   // default algorithm
} aclsparseLtMatmulAlg_t;

/* ========== Prune algorithm (cuSPARSELt PruneAlg_t) ==========
 * TILE and STRIP modes both implemented; see prune_host.cpp for details.
 */
typedef enum aclsparseLtPruneAlg_t {
    ACLSPARSELT_PRUNE_SPMMA_TILE = 0,
    ACLSPARSELT_PRUNE_SPMMA_STRIP = 1,
} aclsparseLtPruneAlg_t;

/* ========== Split-K mode (cuSPARSELt SplitKMode_t) ==========
 * ONE_KERNEL(0):   fused __mix__(1,2) kernel, GEMM+reduction in single launch.
 *                  splitK is absorbed (effectiveSplitK=1, K-loop full range).
 *                  Note: splitK does NOT increase tile parallelism in this mode.
 * TWO_KERNELS(1):  matmul kernel + epilogue kernel, splitK increases parallelism.
 */
typedef enum aclsparseLtSplitKMode_t {
    ACLSPARSELT_SPLIT_K_MODE_ONE_KERNEL = 0,
    ACLSPARSELT_SPLIT_K_MODE_TWO_KERNELS = 1,
} aclsparseLtSplitKMode_t;

/* ========== Library property type (cuSPARSELt libraryPropertyType) ==========
 * Used by aclsparseLtGetProperty to query individual version components.
 */
typedef enum aclsparseLtLibraryPropertyType_t {
    ACLSPARSELT_MAJOR_VERSION = 0,
    ACLSPARSELT_MINOR_VERSION = 1,
    ACLSPARSELT_PATCH_LEVEL = 2,
} aclsparseLtLibraryPropertyType_t;

/* ========== Matmul algorithm attributes ========== */
typedef enum aclsparseLtMatmulAlgAttribute_t {
    ACLSPARSELT_MATMUL_ALG_CONFIG_ID = 0,       // algorithm config id (0 / 1)
    ACLSPARSELT_MATMUL_SEARCH_ITERATIONS = 1,    // search iterations
    ACLSPARSELT_MATMUL_SPLIT_K = 2,              // split-k factor [1, K]
    ACLSPARSELT_MATMUL_SPLIT_K_MODE = 3,         // split-k mode
    // [OPT-P3] Aligned with cuSPARSELt: additional alg attributes.
    ACLSPARSELT_MATMUL_ALG_CONFIG_MAX_ID = 4,   // read-only: returns max algConfigId+1
    ACLSPARSELT_MATMUL_SPLIT_K_BUFFERS = 5,     // accepted, stored; TWO_KERNELS uses max (splitK-1)
} aclsparseLtMatmulAlgAttribute_t;

/* ========== Matmul descriptor attributes (vector scaling + bias + activation) ==========
 * Aligned with cuSPARSELt cusparseLtMatmulDescAttribute_t.
 * When ALPHA_VECTOR_SCALING is enabled, alpha passed to aclsparseLtMatmul
 * is a device pointer to a float array of length M (per-row scaling).
 * BETA_VECTOR_SCALING implies ALPHA_VECTOR_SCALING (setting beta vector
 * auto-enables alpha vector).
 *
 * Bias + activation attributes (values 2~8) extend the epilogue chain:
 *   D = alpha * A * B + beta * C + bias, then apply activation (ReLU/GeLU).
 * ReLU and GeLU are mutually exclusive; enabling one while the other is
 * already enabled returns ACL_SPARSE_STATUS_INVALID_VALUE (setting to 0
 * disables and does not trigger the check).
 */
typedef enum aclsparseLtMatmulDescAttribute_t {
    ACLSPARSELT_MATMUL_ALPHA_VECTOR_SCALING = 0,
    ACLSPARSELT_MATMUL_BETA_VECTOR_SCALING = 1,
    ACLSPARSELT_MATMUL_BIAS_POINTER = 2,                    ///< void*: bias device pointer (NULL=no bias)
    ACLSPARSELT_MATMUL_BIAS_STRIDE = 3,                     ///< int64_t: batch stride for bias (0=broadcast)
    ACLSPARSELT_MATMUL_ACTIVATION_RELU = 4,                 ///< int: 0=disable, nonzero=enable ReLU
    ACLSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND = 5,      ///< float: ReLU upper bound (default FLT_MAX)
    ACLSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD = 6,       ///< float: ReLU threshold (default 0.0f)
    ACLSPARSELT_MATMUL_ACTIVATION_GELU = 7,                 ///< int: 0=disable, nonzero=enable GeLU
    ACLSPARSELT_MATMUL_ACTIVATION_GELU_SCALING = 8,         ///< float: GeLU scaling factor (default 1.0f)
} aclsparseLtMatmulDescAttribute_t;

/* ========== Mat descriptor attributes (batch) ==========
 * Aligned with cuSPARSELt cusparseLtMatDescAttribute_t.
 * NUM_BATCHES sets the batch count (default 1, non-batch).
 * BATCH_STRIDE sets the stride between consecutive batches in elements.
 * All four matrices (A/B/C/D) in a matmul descriptor must share the same
 * numBatches; SetAttribute returns INVALID_VALUE on mismatch.
 */
typedef enum aclsparseLtMatDescAttribute_t {
    ACLSPARSELT_MAT_NUM_BATCHES = 0,    ///< int32_t: batch count (default 1)
    ACLSPARSELT_MAT_BATCH_STRIDE = 1,   ///< int64_t: batch stride in elements
} aclsparseLtMatDescAttribute_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Library Management Functions ========== */

/**
 * @brief 初始化 aclsparseLt 库句柄。
 *
 * 在主机端分配轻量级硬件资源，创建 aclsparseLt 库上下文。
 * 必须在调用任何其他 aclsparseLt 函数之前调用。
 * 库上下文绑定到当前 NPU 设备，多设备使用需为每个设备创建独立 handle。
 *
 * @param handle OUT, HOST, aclsparseLt 库句柄输出参数。调用前 *handle 必须为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE *handle 非空（防止覆盖已有句柄导致泄漏）
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtInit(aclsparseLtHandle_t* handle);

/**
 * @brief 释放 aclsparseLt 库句柄占用的全部资源。
 *
 * 与特定 handle 关联的最后一次调用。调用后该 handle 不可再使用。
 *
 * @param handle IN, HOST, 指向要销毁的 aclsparseLt 库句柄的指针。
 *               若 handle 指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *               若 handle 指向的句柄值为 nullptr 则视为空操作，返回 SUCCESS。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 */
aclsparseStatus_t aclsparseLtDestroy(const aclsparseLtHandle_t* handle);

/**
 * @brief 返回状态码对应的枚举名字符串。
 *
 * @param status IN, HOST, 要转换的状态码。
 * @return const char* 指向枚举名字符串的指针（如 "ACL_SPARSE_STATUS_SUCCESS"）。
 *         未识别的状态码返回 "unrecognized error code"。
 */
const char* aclsparseLtGetErrorName(aclsparseStatus_t status);

/**
 * @brief 返回状态码对应的描述性字符串。
 *
 * @param status IN, HOST, 要转换的状态码。
 * @return const char* 指向描述性字符串的指针。
 *         未识别的状态码返回 "unrecognized error code"。
 */
const char* aclsparseLtGetErrorString(aclsparseStatus_t status);

/* ========== Matrix Descriptor Management ========== */

/**
 * @brief 初始化稠密矩阵描述符。
 *
 * 在主机端分配并填充矩阵描述符结构体，记录稠密矩阵的形状、布局、数据类型等元信息。
 *
 * @param handle     IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param matDescr   OUT, HOST, 矩阵描述符输出。调用前 *matDescr 须为 nullptr。
 * @param rows       IN,  HOST, 行数（须 > 0 且为对应 valueType 对齐倍数的整数倍）。
 * @param cols       IN,  HOST, 列数（须 > 0 且为对应 valueType 对齐倍数的整数倍）。
 * @param ld         IN,  HOST, leading dimension。COL 序须 >= rows，ROW 序须 >= cols；
 *                   须为对应 valueType 对齐倍数的整数倍。
 * @param alignment  IN,  HOST, 内存对齐字节数，须为 16 的倍数且非 0。
 * @param valueType  IN,  HOST, 矩阵数据存储类型。
 * @param order      IN,  HOST, 内存布局（ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matDescr 为空、*matDescr 非空、维度/对齐/布局非法
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED valueType 不在支持列表内
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtDenseDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order);

/**
 * @brief 初始化结构化稀疏矩阵描述符（2:4 结构化稀疏）。
 *
 * 与 aclsparseLtDenseDescriptorInit 类似，但描述符标记为 structured，
 * 对齐倍数采用 structured 档位，并记录稀疏模式。
 *
 * @param handle     IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param matDescr   OUT, HOST, 矩阵描述符输出。调用前 *matDescr 须为 nullptr。
 * @param rows       IN,  HOST, 行数（须 > 0 且为 structured 档位对齐倍数的整数倍）。
 * @param cols       IN,  HOST, 列数（须 > 0 且为 structured 档位对齐倍数的整数倍）。
 * @param ld         IN,  HOST, leading dimension。COL 序须 >= rows，ROW 序须 >= cols；
 *                   须为 structured 档位对齐倍数的整数倍。
 * @param alignment  IN,  HOST, 内存对齐字节数，须为 16 的倍数且非 0。
 * @param valueType  IN,  HOST, 矩阵数据存储类型。
 * @param order      IN,  HOST, 内存布局（ACL_SPARSE_ORDER_ROW / ACL_SPARSE_ORDER_COL）。
 * @param sparsity   IN,  HOST, 稀疏模式（当前仅支持 ACL_SPARSE_LT_SPARSITY_50_PERCENT）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matDescr 为空、*matDescr 非空、维度/对齐/布局/sparsity 非法
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED valueType 不在支持列表内
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtStructuredDescriptorInit(
    const aclsparseLtHandle_t*       handle,
    aclsparseLtMatDescriptor_t*      matDescr,
    int64_t                          rows,
    int64_t                          cols,
    int64_t                          ld,
    uint32_t                         alignment,
    aclDataType                      valueType,
    aclsparseOrder_t                 order,
    aclsparseLtSparsity_t            sparsity);

/**
 * @brief 销毁矩阵描述符，释放其占用的主机内存。
 *
 * @param matDescr INOUT, HOST, 指向要销毁的矩阵描述符的指针。
 *                 若指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *                 若 *matDescr 为 nullptr 则视为空操作，返回 SUCCESS。
 *                 销毁成功后 *matDescr 被置为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR matDescr 指针为空
 */
aclsparseStatus_t aclsparseLtMatDescriptorDestroy(aclsparseLtMatDescriptor_t* matDescr);

/**
 * @brief 设置矩阵描述符属性（对齐 cuSPARSELt cusparseLtMatDescSetAttribute）。
 *
 * 设置 batch 相关属性：NUM_BATCHES（batch 数量，默认 1）、BATCH_STRIDE
 * （batch 间步长，元素数为单位）。需在 PlanInit 之前调用。
 * 一致性校验：同一 matmul descriptor 中的 A/B/C/D 四个矩阵的 numBatches
 * 须一致，否则返回 ACL_SPARSE_STATUS_INVALID_VALUE。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param matDescr INOUT, HOST, 矩阵描述符。
 * @param matAttribute IN, HOST, 要设置的属性枚举。
 * @param data IN, HOST, 属性值数据指针。
 * @param dataSize IN, HOST, data 缓冲区大小（字节），须等于对应枚举的 sizeof。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matDescr/data 为空、dataSize 不匹配、
 *                                         numBatches < 1 或一致性校验失败
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED matAttribute 不在支持范围内
 */
aclsparseStatus_t aclsparseLtMatDescSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t matAttribute,
    const void* data,
    size_t dataSize);

/**
 * @brief 查询矩阵描述符属性当前值（对齐 cuSPARSELt cusparseLtMatDescGetAttribute）。
 *
 * 查询 NUM_BATCHES / BATCH_STRIDE 的当前值。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param matDescr IN, HOST, 矩阵描述符（const 只读）。
 * @param matAttribute IN, HOST, 要查询的属性枚举。
 * @param data OUT, HOST, 属性值输出缓冲区。
 * @param dataSize IN, HOST, data 缓冲区大小（字节），须等于对应枚举的 sizeof。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matDescr/data 为空或 dataSize 不匹配
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED matAttribute 不在支持范围内
 */
aclsparseStatus_t aclsparseLtMatDescGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatDescriptor_t* matDescr,
    aclsparseLtMatDescAttribute_t matAttribute,
    void* data,
    size_t dataSize);

/* ========== Matmul Descriptor Management ========== */

/**
 * @brief 初始化矩阵乘法描述符。
 *
 * 在主机端分配并填充 matmul 描述符结构体，记录 matmul 运算的完整元信息：
 * 操作类型（opA/opB）、四个矩阵描述符引用（matA/matB/matC/matD）、计算精度。
 * matA/matB/matC/matD 为非所有权引用，Destroy 时不会销毁它们，用户须自行释放。
 *
 * 约束：
 * - matA 与 matB 中有且仅有一个为 structured 描述符。
 * - matC 与 matD 须具有相同的 ld 和 order。
 * - INT8/FP8/FP4 类型下（以 matA.valueType 为准），opA/opB 须满足与 orderA/orderB 的组合约束。
 * - matC/matD 的 rows、cols 均 <= 2097120。
 *
 * @param handle       IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param matmulDescr  OUT, HOST, matmul 描述符输出。调用前 *matmulDescr 须为 nullptr。
 * @param opA          IN,  HOST, 作用于矩阵 A 的操作（NON_TRANSPOSE / TRANSPOSE）。
 * @param opB          IN,  HOST, 作用于矩阵 B 的操作。
 * @param matA         IN,  HOST, 矩阵 A 描述符（structured 或 dense），不可为 nullptr。
 * @param matB         IN,  HOST, 矩阵 B 描述符（structured 或 dense），不可为 nullptr。
 * @param matC         IN,  HOST, 矩阵 C 描述符（dense），不可为 nullptr。
 * @param matD         IN,  HOST, 矩阵 D 描述符（dense），不可为 nullptr。
 * @param computeType  IN,  HOST, 计算精度（16F / 32F / 32I）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matmulDescr 为空、*matmulDescr 非空、mat 描述符为空、
 *                                         opA/opB 非法、结构化位置/C/D 一致性/操作布局组合/维度约束违反
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED computeType 不在支持列表内
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtMatmulDescriptorInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulDescriptor_t*         matmulDescr,
    aclsparseOperation_t                   opA,
    aclsparseOperation_t                   opB,
    const aclsparseLtMatDescriptor_t*      matA,
    const aclsparseLtMatDescriptor_t*      matB,
    const aclsparseLtMatDescriptor_t*      matC,
    const aclsparseLtMatDescriptor_t*      matD,
    aclsparseComputeType_t                 computeType);

/**
 * @brief 销毁 matmul 描述符，释放其占用的主机内存。
 *
 * 注意：本接口不销毁 matA/matB/matC/matD 描述符（非所有权引用），
 * 用户须自行调用 aclsparseLtMatDescriptorDestroy 释放它们。
 *
 * @param matmulDescr INOUT, HOST, 指向要销毁的 matmul 描述符的指针。
 *                   若指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *                   若 *matmulDescr 为 nullptr 则视为空操作，返回 SUCCESS。
 *                   销毁成功后 *matmulDescr 被置为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR matmulDescr 指针为空
 */
aclsparseStatus_t aclsparseLtMatmulDescriptorDestroy(aclsparseLtMatmulDescriptor_t* matmulDescr);

/* ========== Matmul Algorithm Selection & Plan Management ========== */

/**
 * @brief 初始化 matmul 算法选择描述符。
 *
 * 在主机端分配并填充算法选择描述符结构体，记录算法模式及其关联的 matmul 描述符引用。
 * matmulDescr 为非所有权引用，Destroy 时不会销毁它，用户须自行释放。
 *
 * @param handle        IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param algSelection  OUT, HOST, 算法选择描述符输出。调用前 *algSelection 须为 nullptr。
 * @param matmulDescr   IN,  HOST, matmul 描述符引用，不可为 nullptr 且须已初始化（*matmulDescr 非 nullptr）。
 * @param alg           IN,  HOST, 算法模式（当前仅支持 ACL_SPARSE_LT_MATMUL_ALG_DEFAULT）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE algSelection 为空、*algSelection 非空、matmulDescr 为空或未初始化、alg 非法
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtMatmulAlgSelectionInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulAlgSelection_t*       algSelection,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    aclsparseLtMatmulAlg_t                 alg);

/**
 * @brief 销毁 matmul 算法选择描述符，释放其占用的主机内存。
 *
 * 注意：本接口不销毁 matmulDescr（非所有权引用），用户须自行调用 aclsparseLtMatmulDescriptorDestroy 释放。
 *
 * @param algSelection INOUT, HOST, 指向要销毁的算法选择描述符的指针。
 *                     若指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *                     若 *algSelection 为 nullptr 则视为空操作，返回 SUCCESS。
 *                     销毁成功后 *algSelection 被置为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR algSelection 指针为空
 */
aclsparseStatus_t aclsparseLtMatmulAlgSelectionDestroy(aclsparseLtMatmulAlgSelection_t* algSelection);

/**
 * @brief 初始化 matmul 执行计划。
 *
 * 在主机端分配并填充 plan 结构体，绑定 matmul 描述符与算法选择描述符，作为后续 matmul 执行的规划对象。
 * matmulDescr 与 algSelection 均为非所有权引用，Destroy 时不会销毁它们，用户须自行释放。
 *
 * @param handle        IN,  HOST, aclsparseLt 库句柄，不可为 nullptr。
 * @param plan          OUT, HOST, 执行计划输出。调用前 *plan 须为 nullptr。
 * @param matmulDescr   IN,  HOST, matmul 描述符引用，不可为 nullptr 且须已初始化（*matmulDescr 非 nullptr）。
 * @param algSelection  IN,  HOST, 算法选择描述符引用，不可为 nullptr 且须已初始化（*algSelection 非 nullptr）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 指针为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE plan 为空、*plan 非空、matmulDescr 或 algSelection 为空或未初始化
 *         ACL_SPARSE_STATUS_ALLOC_FAILED 内存分配失败
 */
aclsparseStatus_t aclsparseLtMatmulPlanInit(
    const aclsparseLtHandle_t*             handle,
    aclsparseLtMatmulPlan_t*               plan,
    const aclsparseLtMatmulDescriptor_t*   matmulDescr,
    const aclsparseLtMatmulAlgSelection_t* algSelection);

/**
 * @brief 销毁 matmul 执行计划，释放其占用的主机内存。
 *
 * 注意：本接口不销毁 matmulDescr 和 algSelection（非所有权引用），用户须自行调用对应 Destroy 释放。
 * 生命周期约束：plan 须在 algSelection 与 matmul 描述符之前销毁。
 *
 * @param plan INOUT, HOST, 指向要销毁的执行计划的指针。
 *             若指针本身为 nullptr 则返回 HANDLE_IS_NULLPTR；
 *             若 *plan 为 nullptr 则视为空操作，返回 SUCCESS。
 *             销毁成功后 *plan 被置为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR plan 指针为空
 */
aclsparseStatus_t aclsparseLtMatmulPlanDestroy(aclsparseLtMatmulPlan_t* plan);

/* ========== 2:4 structured sparsity prune ==========
 */

/**
 * @brief 对稠密矩阵执行 2:4 结构化稀疏剪枝（对齐 cuSPARSELt cusparseLtSpMMAPrune）。
 *
 * 直接接收 Matmul 描述符，从中读取矩阵 A 的维度（m、k）、数据类型、order 与 opA
 * 派生剪枝参数。算子异步启动，内部不执行 stream 同步，调用方如需读取结果须自行同步。
 *
 * @param handle       IN,  HOST, aclsparseLt 库句柄的 const 指针，不可为 nullptr。
 * @param matmulDescr  IN,  HOST, Matmul 操作描述符，算子从中读取矩阵 A 的维度/数据类型/order/opA。
 * @param d_in         IN,  DEVICE, 待剪枝的稠密矩阵 A 的指针，须 16 字节对齐，不可为 nullptr。
 * @param d_out        OUT, DEVICE, 剪枝结果 A_pruned 的指针，须 16 字节对齐，不可为 nullptr。支持 in-place（d_in == d_out）。
 * @param pruneAlg     IN,  HOST, 剪枝算法（ACLSPARSELT_PRUNE_SPMMA_STRIP / ACLSPARSELT_PRUNE_SPMMA_TILE）。
 * @param stream       IN,  HOST, ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为 nullptr
 *         ACL_SPARSE_STATUS_INVALID_VALUE matmulDescr/matA/d_in/d_out 为 nullptr 或未 16 字节对齐
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED pruneAlg 非 STRIP/TILE、数据类型非 FP32/FP16/BF16/INT8、K 维度超过 UB 容量
 */
aclsparseStatus_t aclsparseLtSpMMAPrune(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    void* d_out,
    aclsparseLtPruneAlg_t pruneAlg,
    aclrtStream stream);

/**
 * @brief 校验矩阵是否满足 2:4 结构化稀疏条件（对齐 cuSPARSELt cusparseLtSpMMAPruneCheck）。
 *
 * 逐分组统计非零元素个数（±0 按零、NaN/非规格数按非零，位级判零），与保留上限比较：
 * 全部分组满足条件时 d_valid=0（correct），任一分组超限时 d_valid=1（wrong）。
 * 校验对象与分组方向由 Matmul 描述符中稀疏侧（matA/matB 须恰有一个 isStructured）
 * 的 order 与 op 联合决定，与 aclsparseLtSpMMAPrune 的剪枝方向推导一致。
 * 支持多 batch（经 aclsparseLtMatDescSetAttribute 设置稀疏侧
 * ACLSPARSELT_MAT_NUM_BATCHES / ACLSPARSELT_MAT_BATCH_STRIDE）：d_valid 为全部
 * batch 判定结果的按位或——任一 batch 存在违规分组 → 1，全部 batch 合法 → 0；
 * numBatches=1（默认）时为单 batch 判定。
 * 经 aclsparseLtSpMMAPrune 剪枝产出的数据保证通过本检查。
 * 算子异步启动，内部不执行 stream 同步，调用方如需读取 d_valid 须自行同步。
 *
 * @param handle       IN,  HOST, aclsparseLt 库句柄的 const 指针，不可为 nullptr。
 * @param matmulDescr  IN,  HOST, Matmul 操作描述符，matA/matB 须恰有一个结构化稀疏侧。
 * @param d_in         IN,  DEVICE, 待校验矩阵的指针，行主序存储，须 32 字节对齐（实现要求：
 *                    d_in 作为 DataCopyPad 的 GM 源地址，硬件要求 32 字节对齐，严于官方 16 字节惯例），
 *                    不可为 nullptr。
 * @param d_valid      OUT, DEVICE, 校验结果 int 标量（0 满足条件，1 不满足），不可为 nullptr。
 * @param stream       IN,  HOST, ACL 流，算子在此流上异步执行，可为 nullptr（表示使用默认流）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为 nullptr
 *         ACL_SPARSE_STATUS_INVALID_VALUE matmulDescr/matA/matB/d_in/d_valid 为 nullptr、
 *                d_in 未 32 字节对齐、维度/ld 非法、批量参数非法（numBatches<1，或
 *                numBatches>1 时 batchStride≤0 / 非 32B 对齐）或扫描跨度溢出
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED matA/matB 无唯一稀疏侧、
 *                数据类型非 FP32/FP16/BF16/INT8
 *         ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES 平台核数/UB 资源不足
 *         ACL_SPARSE_STATUS_EXECUTION_FAILED d_valid 预清零或 kernel 启动失败
 */
aclsparseStatus_t aclsparseLtSpMMAPruneCheck(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    const void* d_in,
    int* d_valid,
    aclrtStream stream);

/**
 * @brief 查询结构化稀疏压缩的输出和临时区大小（Ascend950）。
 *
 * @param handle             IN,  HOST, 有效的库句柄，不可为 nullptr。
 * @param plan               IN,  HOST, 有效 Plan，A/B 中恰有一个结构化稀疏矩阵。
 * @param compressedSize     OUT, HOST, 压缩结果的精确字节数，指针不可为 nullptr。
 * @param compressBufferSize OUT, HOST, 临时区字节数，当前为 0，指针不可为 nullptr。
 * @return ACL_SPARSE_STATUS_SUCCESS 查询成功；失败时两个输出值保持不变。
 * @see sparseLt/spmma_compress/README.md 中的格式、约束和完整返回码说明。
 */
aclsparseStatus_t aclsparseLtSpMMACompressedSize(
    const aclsparseLtHandle_t* handle, const aclsparseLtMatmulPlan_t* plan,
    size_t* compressedSize, size_t* compressBufferSize);

/**
 * @brief 按 Plan 描述的布局异步压缩结构化稀疏矩阵（Ascend950）。
 *
 * 输入须满足 FP32 的 1:2 或 FP16/BF16/INT8 的 2:4 稀疏条件，选中值的位模式保持不变。
 * 输入输出首址须满足描述符的对齐要求（至少 16 字节）。读取结果或释放相关对象前须同步 stream。
 *
 * @param handle              IN,    HOST, 有效的库句柄，不可为 nullptr。
 * @param plan                IN,    HOST, 有效 Plan，A/B 中恰有一个结构化稀疏矩阵。
 * @param d_dense             IN,    DEVICE, 满足稀疏条件的输入矩阵，不可为 nullptr。
 * @param d_compressed        OUT,   DEVICE, 输出缓冲区，不可为空或与输入重叠，容量至少为查询所得大小。
 * @param d_compressed_buffer INOUT, DEVICE, 压缩临时区，当前忽略，可为 nullptr。
 * @param stream              IN,    HOST, 执行流，可为 nullptr（默认流）。
 * @return ACL_SPARSE_STATUS_SUCCESS Host 校验和启动调用成功。
 * @see sparseLt/spmma_compress/README.md 中的格式、约束和完整返回码说明。
 */
aclsparseStatus_t aclsparseLtSpMMACompress(
    const aclsparseLtHandle_t* handle, const aclsparseLtMatmulPlan_t* plan,
    const void* d_dense, void* d_compressed, void* d_compressed_buffer,
    aclrtStream stream);

/* ========== PR Extension APIs (cuSPARSELt-style) ==========
 * Data pointers are NOT bound at descriptor init time — they are passed
 * explicitly to Matmul/Prune at execution time.
 */

/**
 * @brief 返回 aclsparseLt 库版本号（对齐 cuSPARSELt cusparseLtGetVersion）。
 *
 * 版本编码格式：MAJOR*10000 + MINOR*100 + PATCH（如 0.1.0 -> 100）。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄（版本为编译期常量，handle 仅做合法性校验）。
 * @param version OUT, HOST, 版本号输出。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_INVALID_VALUE version 为空
 */
aclsparseStatus_t aclsparseLtGetVersion(aclsparseLtConstHandle_t handle, int* version);

/**
 * @brief 返回库属性（对齐 cuSPARSELt cusparseLtGetProperty）。
 *
 * 根据 propertyType 返回版本号的 MAJOR / MINOR / PATCH 分量到 int* value。
 *
 * @param propertyType IN, HOST, 要查询的属性类型（MAJOR_VERSION / MINOR_VERSION / PATCH_LEVEL）。
 * @param value OUT, HOST, 属性值输出。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_INVALID_VALUE value 为空或 propertyType 非法
 */
aclsparseStatus_t aclsparseLtGetProperty(aclsparseLtLibraryPropertyType_t propertyType, int* value);

/**
 * @brief 设置 matmul 描述符属性（对齐 cuSPARSELt cusparseLtMatmulDescSetAttribute）。
 *
 * 设置向量缩放属性（ALPHA_VECTOR_SCALING / BETA_VECTOR_SCALING），
 * bias 属性（BIAS_POINTER / BIAS_STRIDE），以及 activation 属性
 * （ACTIVATION_RELU / RELU_UPPERBOUND / RELU_THRESHOLD / ACTIVATION_GELU / GELU_SCALING）。
 * 设置 BETA_VECTOR_SCALING=true 时隐含 ALPHA_VECTOR_SCALING=true（cuSPARSELt 语义）。
 * ReLU 与 GeLU 互斥：启用一种时若另一种已启用，返回 INVALID_VALUE。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param matmulDescr INOUT, HOST, matmul 描述符。
 * @param matmulAttribute IN, HOST, 要设置的属性枚举。
 * @param data IN, HOST, 属性值数据指针。
 * @param dataSize IN, HOST, data 缓冲区大小（字节），须等于对应枚举的 sizeof。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matmulDescr/data 为空或 dataSize 不匹配
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED matmulAttribute 不在支持范围内
 */
aclsparseStatus_t aclsparseLtMatmulDescSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t matmulAttribute,
    const void* data,
    size_t dataSize);

/**
 * @brief 查询 matmul 描述符属性当前值（对齐 cuSPARSELt cusparseLtMatmulDescGetAttribute）。
 *
 * 查询向量缩放属性（ALPHA_VECTOR_SCALING / BETA_VECTOR_SCALING），
 * bias 属性（BIAS_POINTER / BIAS_STRIDE），以及 activation 属性
 * （ACTIVATION_RELU / RELU_UPPERBOUND / RELU_THRESHOLD / ACTIVATION_GELU / GELU_SCALING）的当前值。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param matmulDescr IN, HOST, matmul 描述符（const 只读）。
 * @param matmulAttribute IN, HOST, 要查询的属性枚举。
 * @param data OUT, HOST, 属性值输出缓冲区，写入对应类型的值。
 * @param dataSize IN, HOST, data 缓冲区大小（字节），须等于对应枚举的 sizeof。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE matmulDescr/data 为空或 dataSize 不匹配
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED matmulAttribute 不在支持范围内
 */
aclsparseStatus_t aclsparseLtMatmulDescGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulDescriptor_t* matmulDescr,
    aclsparseLtMatmulDescAttribute_t matmulAttribute,
    void* data,
    size_t dataSize);

/**
 * @brief 设置算法选择属性（对齐 cuSPARSELt cusparseLtMatmulAlgSetAttribute）。
 *
 * 设置 algConfigId / splitK / searchIterations / splitKMode / splitKBuffers 等算法属性。
 * ALG_CONFIG_MAX_ID 为只读属性，不可设置。须在 PlanInit 之前调用。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param algSelection INOUT, HOST, 算法选择描述符。
 * @param attr IN, HOST, 要设置的属性枚举。
 * @param attrValue IN, HOST, 属性值数据指针（均为 int32_t）。
 * @param attrValueSize IN, HOST, attrValue 缓冲区大小（字节），须等于 sizeof(int32_t)。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE algSelection/attrValue 为空、size 不匹配或属性值非法
 *         ACL_SPARSE_STATUS_NOT_INITIALIZED algSelection 未初始化（已销毁）
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED attr 不被支持（如 ALG_CONFIG_MAX_ID 为只读）
 */
aclsparseStatus_t aclsparseLtMatmulAlgSetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    const void* attrValue,
    size_t attrValueSize);

/**
 * @brief 查询算法选择属性当前值（对齐 cuSPARSELt cusparseLtMatmulAlgGetAttribute）。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param algSelection IN, HOST, 算法选择描述符。
 * @param attr IN, HOST, 要查询的属性枚举。
 * @param attrValue OUT, HOST, 属性值输出缓冲区。
 * @param attrValueSize IN, HOST, attrValue 缓冲区大小（字节）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE algSelection/attrValue 为空或 size 不匹配
 *         ACL_SPARSE_STATUS_NOT_INITIALIZED algSelection 未初始化（已销毁）
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED attr 不被支持（如 ALG_CONFIG_MAX_ID 为只读）
 */
aclsparseStatus_t aclsparseLtMatmulAlgGetAttribute(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulAlgSelection_t* algSelection,
    aclsparseLtMatmulAlgAttribute_t attr,
    void* attrValue,
    size_t attrValueSize);

/* ========== Workspace & execution ==========
 */

/**
 * @brief 查询 matmul 执行所需的 workspace 大小（对齐 cuSPARSELt cusparseLtMatmulGetWorkspace）。
 *
 * 根据 PlanInit 阶段计算的 tiling 参数返回 workspace 字节数。调用方据此分配 Device 内存，
 * 再将指针传递给 aclsparseLtMatmul。workspaceSize 为 0 时无需分配。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param plan IN, HOST, 执行计划。
 * @param workspaceSize OUT, HOST, workspace 字节数输出。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE plan/workspaceSize 为空
 */
aclsparseStatus_t aclsparseLtMatmulGetWorkspace(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan, size_t* workspaceSize);

/**
 * @brief 执行结构化稀疏矩阵乘法（对齐 cuSPARSELt cusparseLtMatmul）。
 *
 * 计算 D = alpha * A_pruned * B + beta * C。A 须为已剪枝的矩阵（通过 aclsparseLtSpMMAPrune 产出）。
 * 当 matA 为 null 时，回退到 workspace 中的 A_pruned 区域（须已由 SpMMAPrune 填充）。
 * alpha/beta 可为标量（float 指针）或设备端向量（启用 ALPHA_VECTOR_SCALING 时为 float[M] 数组指针）。
 * 该接口异步执行，调用方须通过 aclrtSynchronizeStream 同步后读取 D。
 *
 * @param handle IN, HOST, aclsparseLt 库句柄的 const 指针。
 * @param plan IN, HOST, 执行计划。
 * @param alpha IN, HOST/DEVICE, alpha 缩放因子（标量时为 float 指针，向量缩放时为 Device float[M] 指针）。
 * @param matA IN, DEVICE, 矩阵 A 数据指针（剪枝后）。可为 null（回退到 workspace A_pruned 区域）。
 * @param matB IN, DEVICE, 矩阵 B 数据指针。
 * @param beta IN, HOST/DEVICE, beta 缩放因子（标量时为 float 指针，向量缩放时为 Device float[M] 指针）。
 * @param matC IN, DEVICE, 矩阵 C 数据指针（累加项）。
 * @param matD OUT, DEVICE, 矩阵 D 数据指针（输出，须预分配）。
 * @param workspace IN, DEVICE, workspace 指针（大小由 GetWorkspaceSize 查询）。
 * @param streams IN, HOST, ACL 流指针数组。
 * @param numStreams IN, HOST, 流数量（当前仅使用 streams[0]，须 > 0）。
 * @return ACL_SPARSE_STATUS_SUCCESS 成功
 *         ACL_SPARSE_STATUS_HANDLE_IS_NULLPTR handle 为空
 *         ACL_SPARSE_STATUS_INVALID_VALUE plan/streams/matB/matC/matD/workspace 为空或参数非法
 *         ACL_SPARSE_STATUS_INSUFFICIENT_RESOURCES workspace 为空但计划需要
 *         ACL_SPARSE_STATUS_NOT_SUPPORTED dataType 不支持
 *         ACL_SPARSE_STATUS_EXECUTION_FAILED kernel 执行或 aclrt 调用失败
 */
aclsparseStatus_t aclsparseLtMatmul(
    aclsparseLtConstHandle_t handle,
    aclsparseLtConstMatmulPlan_t* plan,
    const void* alpha,
    const void* matA, const void* matB,
    const void* beta,
    const void* matC, void* matD,
    void* workspace,
    aclrtStream* streams, int32_t numStreams);

#ifdef __cplusplus
}
#endif

#endif // CANN_OPS_SPARSELT_H_
