from libc.stdint cimport intptr_t, int64_t

cdef extern from "../../cupy_cudss.h" nogil:
    cpdef enum cudssDataType_t:
        CUDSS_DATA_TYPE_UNSET
        CUDSS_R_32F
        CUDSS_R_64F
        CUDSS_C_32F
        CUDSS_C_64F
        CUDSS_R_64F_64F
        CUDSS_R_32I
        CUDSS_R_64I

    cpdef enum cudssStatus_t:
        CUDSS_STATUS_SUCCESS
        CUDSS_STATUS_NOT_INITIALIZED
        CUDSS_STATUS_ALLOC_FAILED
        CUDSS_STATUS_INVALID_VALUE
        CUDSS_STATUS_NOT_SUPPORTED
        CUDSS_STATUS_EXECUTION_FAILED
        CUDSS_STATUS_INTERNAL_ERROR
        CUDSS_STATUS_IR_FAILED

    cpdef enum cudssConfigParam_t:
        CUDSS_CONFIG_REORDERING_ALG
        CUDSS_CONFIG_FACTORIZATION_ALG
        CUDSS_CONFIG_SOLVE_ALG
        CUDSS_CONFIG_PIVOT_EPSILON_ALG
        CUDSS_CONFIG_MATCHING_ALG
        CUDSS_CONFIG_SOLVE_MODE
        CUDSS_CONFIG_IR_N_STEPS
        CUDSS_CONFIG_IR_TOL
        CUDSS_CONFIG_PIVOT_TYPE
        CUDSS_CONFIG_PIVOT_THRESHOLD
        CUDSS_CONFIG_PIVOT_EPSILON
        CUDSS_CONFIG_MAX_LU_NNZ
        CUDSS_CONFIG_HYBRID_MEMORY_MODE
        CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT
        CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY
        CUDSS_CONFIG_HOST_NTHREADS
        CUDSS_CONFIG_HYBRID_EXECUTE_MODE
        CUDSS_CONFIG_ND_NLEVELS
        CUDSS_CONFIG_ND_UBFACTOR
        CUDSS_CONFIG_UBATCH_SIZE
        CUDSS_CONFIG_UBATCH_INDEX
        CUDSS_CONFIG_USE_SUPERPANELS
        CUDSS_CONFIG_DEVICE_COUNT
        CUDSS_CONFIG_DEVICE_INDICES
        CUDSS_CONFIG_SCHUR_MODE
        CUDSS_CONFIG_DETERMINISTIC_MODE

    cpdef enum cudssDataParam_t:
        CUDSS_DATA_INFO
        CUDSS_DATA_LU_NNZ
        CUDSS_DATA_NPIVOTS
        CUDSS_DATA_INERTIA
        CUDSS_DATA_PERM_REORDER_ROW
        CUDSS_DATA_PERM_REORDER_COL
        CUDSS_DATA_PERM_ROW
        CUDSS_DATA_PERM_COL
        CUDSS_DATA_PERM_MATCHING
        CUDSS_DATA_DIAG
        CUDSS_DATA_SCALE_ROW
        CUDSS_DATA_SCALE_COL
        CUDSS_DATA_USER_PERM
        CUDSS_DATA_ND_PARTITION_TREE
        CUDSS_DATA_USER_ND_PARTITION_TREE
        CUDSS_DATA_MEMORY_ESTIMATES
        CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN
        CUDSS_DATA_FLOPS
        CUDSS_DATA_COMM_DEVICE
        CUDSS_DATA_COMM_HOST
        CUDSS_DATA_NSUPERPANELS
        CUDSS_DATA_USER_SCHUR_INDICES
        CUDSS_DATA_SCHUR_SHAPE
        CUDSS_DATA_SCHUR_MATRIX
        CUDSS_DATA_USER_HOST_INTERRUPT
        CUDSS_DATA_IR_N_STEPS
        CUDSS_DATA_UBATCH_MASK

    cpdef enum cudssPhase_t:
        CUDSS_PHASE_REORDERING
        CUDSS_PHASE_SYMBOLIC_FACTORIZATION
        CUDSS_PHASE_ANALYSIS
        CUDSS_PHASE_FACTORIZATION
        CUDSS_PHASE_REFACTORIZATION
        CUDSS_PHASE_SOLVE_FWD_PERM
        CUDSS_PHASE_SOLVE_FWD
        CUDSS_PHASE_SOLVE_DIAG
        CUDSS_PHASE_SOLVE_BWD
        CUDSS_PHASE_SOLVE_BWD_PERM
        CUDSS_PHASE_SOLVE_REFINEMENT
        CUDSS_PHASE_SOLVE

    cpdef enum cudssMatrixFormat_t:
        CUDSS_MFORMAT_DENSE
        CUDSS_MFORMAT_CSR
        CUDSS_MFORMAT_BATCH
        CUDSS_MFORMAT_DISTRIBUTED

    cpdef enum cudssMatrixType_t:
        CUDSS_MTYPE_GENERAL
        CUDSS_MTYPE_SYMMETRIC
        CUDSS_MTYPE_HERMITIAN
        CUDSS_MTYPE_SPD
        CUDSS_MTYPE_HPD

    cpdef enum cudssMatrixViewType_t:
        CUDSS_MVIEW_FULL
        CUDSS_MVIEW_LOWER
        CUDSS_MVIEW_UPPER

    cpdef enum cudssIndexBase_t:
        CUDSS_BASE_ZERO
        CUDSS_BASE_ONE

    cpdef enum cudssLayout_t:
        CUDSS_LAYOUT_COL_MAJOR
        CUDSS_LAYOUT_ROW_MAJOR

    cpdef enum cudssReorderingAlg_t:
        CUDSS_REORDERING_ALG_DEFAULT
        CUDSS_REORDERING_ALG_BTF_COLAMD
        CUDSS_REORDERING_ALG_COLAMD
        CUDSS_REORDERING_ALG_AMD
        CUDSS_REORDERING_ALG_NESTED_DISSECTION
        CUDSS_REORDERING_ALG_NONE

    cpdef enum cudssFactorizationAlg_t:
        CUDSS_FACTORIZATION_ALG_DEFAULT
        CUDSS_FACTORIZATION_ALG_GENERAL
        CUDSS_FACTORIZATION_ALG_MULTIBLOCK

    cpdef enum cudssSolveAlg_t:
        CUDSS_SOLVE_ALG_DEFAULT
        CUDSS_SOLVE_ALG_GENERAL

    cpdef enum cudssPivotEpsilonAlg_t:
        CUDSS_PIVOT_EPSILON_ALG_DEFAULT
        CUDSS_PIVOT_EPSILON_ALG_SCALED
        CUDSS_PIVOT_EPSILON_ALG_STATIC

    cpdef enum cudssMatchingAlg_t:
        CUDSS_MATCHING_ALG_NONE
        CUDSS_MATCHING_ALG_AUTO
        CUDSS_MATCHING_ALG_MAX_DIAG_COUNT
        CUDSS_MATCHING_ALG_MAX_MIN_DIAG
        CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT
        CUDSS_MATCHING_ALG_MAX_DIAG_SUM
        CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT

    cpdef enum cudssPivotType_t:
        CUDSS_PIVOT_AUTO
        CUDSS_PIVOT_NONE
        CUDSS_PIVOT_GLOBAL_COL
        CUDSS_PIVOT_GLOBAL_ROW
        CUDSS_PIVOT_DIAGONAL
        CUDSS_PIVOT_LOCAL_BLOCK
        CUDSS_PIVOT_BUNCH_KAUFMAN

    cpdef enum cudssOpType_t:
        CUDSS_SUM
        CUDSS_MAX
        CUDSS_MIN

cpdef check_status(int status)
cpdef create()
cpdef createMg(int device_count, intptr_t device_indices)
cpdef destroy(intptr_t handle)
cpdef getProperty(int propertyType)
cpdef setStream(intptr_t handle, intptr_t stream)
cpdef setMgStreams(intptr_t handle, intptr_t streams, int stream_count)
cpdef setDeviceMemHandler(intptr_t handle, intptr_t handler)
cpdef getDeviceMemHandler(intptr_t handle, intptr_t handler)
cpdef setCommLayer(intptr_t handle, filename)
cpdef setThreadingLayer(intptr_t handle, filename)
cpdef loggerSetCallback(intptr_t callback)
cpdef loggerSetFile(intptr_t file)
cpdef loggerOpenFile(filename)
cpdef loggerSetLevel(int level)
cpdef loggerSetMask(int mask)
cpdef loggerForceDisable()
cpdef configCreate()
cpdef configDestroy(intptr_t config)
cpdef configSet(
        intptr_t config,
        cudssConfigParam_t param,
        intptr_t value,
        size_t sizeInBytes)
cpdef configGet(
        intptr_t config,
        cudssConfigParam_t param,
        intptr_t value,
        size_t sizeInBytes)
cpdef dataCreate(intptr_t handle)
cpdef dataDestroy(intptr_t handle, intptr_t data)
cpdef dataSet(
        intptr_t handle,
        intptr_t data,
        cudssDataParam_t param,
        intptr_t value,
        size_t sizeInBytes)
cpdef dataGet(
        intptr_t handle,
        intptr_t data,
        cudssDataParam_t param,
        intptr_t value,
        size_t sizeInBytes)
cpdef execute(
        intptr_t handle,
        int phase,
        intptr_t config,
        intptr_t data,
        intptr_t matrix,
        intptr_t solution,
        intptr_t rhs)
cpdef matrixCreateDn(
        int64_t nrows,
        int64_t ncols,
        int64_t ld,
        intptr_t values,
        cudssDataType_t valueType,
        cudssLayout_t layout)
cpdef matrixCreateBatchDn(
        int64_t batchCount,
        intptr_t nrows,
        intptr_t ncols,
        intptr_t ld,
        intptr_t values,
        cudssDataType_t integerType,
        cudssDataType_t valueType,
        cudssLayout_t layout)
cpdef matrixCreateCsr(
        int64_t nrows,
        int64_t ncols,
        int64_t nnz,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values,
        cudssDataType_t offsetType,
        cudssDataType_t indexType,
        cudssDataType_t valueType,
        cudssMatrixType_t mtype,
        cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase)
cpdef matrixCreateBatchCsr(
        int64_t batchCount,
        intptr_t nrows,
        intptr_t ncols,
        intptr_t nnz,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values,
        cudssDataType_t offsetType,
        cudssDataType_t indexType,
        cudssDataType_t valueType,
        cudssMatrixType_t mtype,
        cudssMatrixViewType_t mview,
        cudssIndexBase_t indexBase)
cpdef matrixDestroy(intptr_t matrix)
cpdef matrixSetValues(intptr_t matrix, intptr_t values)
cpdef matrixSetBatchValues(intptr_t matrix, intptr_t values)
cpdef matrixSetCsrPointers(
        intptr_t matrix,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values)
cpdef matrixSetBatchCsrPointers(
        intptr_t matrix,
        intptr_t rowStart,
        intptr_t rowEnd,
        intptr_t colIndices,
        intptr_t values)
cpdef matrixSetDistributionRow1d(
        intptr_t matrix,
        int64_t first_row,
        int64_t last_row)
cpdef matrixGetDn(intptr_t matrix)
cpdef matrixGetBatchDn(intptr_t matrix)
cpdef matrixGetCsr(intptr_t matrix)
cpdef matrixGetBatchCsr(intptr_t matrix)
cpdef matrixGetDistributionRow1d(intptr_t matrix)
cpdef matrixGetFormat(intptr_t matrix)
