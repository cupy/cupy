// cuDSS is unavailable on ROCm/HIP. These stubs support compilation only.
#ifndef INCLUDE_GUARD_CUPY_HIP_CUDSS_H
#define INCLUDE_GUARD_CUPY_HIP_CUDSS_H

#define CUDSS_VERSION -1

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern "C" {
typedef void* cudssHandle_t;
typedef void* cudssConfig_t;
typedef void* cudssData_t;
typedef void* cudssMatrix_t;
typedef void* cudaStream_t;
typedef int libraryPropertyType;

typedef enum cudssDataType_t {
    CUDSS_DATA_TYPE_UNSET = 1024,
    CUDSS_R_32F = 0, CUDSS_R_64F = 1,
    CUDSS_C_32F = 4, CUDSS_C_64F = 5,
    CUDSS_R_64F_64F = 1026,
    CUDSS_R_32I = 10, CUDSS_R_64I = 24
} cudssDataType_t;
typedef enum cudssStatus_t {
    CUDSS_STATUS_SUCCESS,
    CUDSS_STATUS_NOT_INITIALIZED,
    CUDSS_STATUS_ALLOC_FAILED,
    CUDSS_STATUS_INVALID_VALUE,
    CUDSS_STATUS_NOT_SUPPORTED,
    CUDSS_STATUS_EXECUTION_FAILED,
    CUDSS_STATUS_INTERNAL_ERROR,
    CUDSS_STATUS_IR_FAILED
} cudssStatus_t;
typedef enum cudssConfigParam_t {
    CUDSS_CONFIG_REORDERING_ALG,
    CUDSS_CONFIG_FACTORIZATION_ALG,
    CUDSS_CONFIG_SOLVE_ALG,
    CUDSS_CONFIG_MATCHING_ALG,      
    CUDSS_CONFIG_SOLVE_MODE,        
    CUDSS_CONFIG_IR_N_STEPS,
    CUDSS_CONFIG_IR_TOL,
    CUDSS_CONFIG_PIVOT_TYPE,
    CUDSS_CONFIG_PIVOT_THRESHOLD,
    CUDSS_CONFIG_PIVOT_EPSILON,
    CUDSS_CONFIG_MAX_LU_NNZ,              
    CUDSS_CONFIG_HYBRID_MEMORY_MODE,             
    CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
    CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY,
    CUDSS_CONFIG_HOST_NTHREADS,
    CUDSS_CONFIG_HYBRID_EXECUTE_MODE,     
    CUDSS_CONFIG_PIVOT_EPSILON_ALG,
    CUDSS_CONFIG_ND_NLEVELS,
    CUDSS_CONFIG_UBATCH_SIZE, 
    CUDSS_CONFIG_UBATCH_INDEX,
    CUDSS_CONFIG_USE_SUPERPANELS, 
    CUDSS_CONFIG_DEVICE_COUNT,
    CUDSS_CONFIG_DEVICE_INDICES,
    CUDSS_CONFIG_SCHUR_MODE,
    CUDSS_CONFIG_DETERMINISTIC_MODE,
    CUDSS_CONFIG_ND_UBFACTOR  
} cudssConfigParam_t;
typedef enum cudssDataParam_t {
    CUDSS_DATA_INFO,                 
    CUDSS_DATA_LU_NNZ,               
    CUDSS_DATA_NPIVOTS,              
    CUDSS_DATA_INERTIA,              
    CUDSS_DATA_PERM_REORDER_ROW,     
    CUDSS_DATA_PERM_REORDER_COL,     
    CUDSS_DATA_PERM_ROW,             
    CUDSS_DATA_PERM_COL,             
    CUDSS_DATA_DIAG,                 
    CUDSS_DATA_USER_PERM,            
    CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
    CUDSS_DATA_COMM_DEVICE,          
    CUDSS_DATA_COMM_HOST,            
    CUDSS_DATA_MEMORY_ESTIMATES,
    CUDSS_DATA_PERM_MATCHING,        
    CUDSS_DATA_SCALE_ROW,            
    CUDSS_DATA_SCALE_COL,            
    CUDSS_DATA_NSUPERPANELS,         
    CUDSS_DATA_USER_SCHUR_INDICES,   
    CUDSS_DATA_SCHUR_SHAPE,          
    CUDSS_DATA_SCHUR_MATRIX,         
    
    CUDSS_DATA_USER_ND_PARTITION_TREE,
    CUDSS_DATA_ND_PARTITION_TREE,     
    CUDSS_DATA_USER_HOST_INTERRUPT,
    CUDSS_DATA_IR_N_STEPS,           
    CUDSS_DATA_UBATCH_MASK,          
    CUDSS_DATA_FLOPS                 
} cudssDataParam_t;
typedef enum cudssPhase_t {
    CUDSS_PHASE_REORDERING             = 1 << 0,
    CUDSS_PHASE_SYMBOLIC_FACTORIZATION = 1 << 1,
    CUDSS_PHASE_ANALYSIS               = CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
    CUDSS_PHASE_FACTORIZATION          = 1 << 2,
    CUDSS_PHASE_REFACTORIZATION        = 1 << 3,
    CUDSS_PHASE_SOLVE_FWD_PERM         = 1 << 4,
    CUDSS_PHASE_SOLVE_FWD              = 1 << 5,
    CUDSS_PHASE_SOLVE_DIAG             = 1 << 6,
    CUDSS_PHASE_SOLVE_BWD              = 1 << 7,
    CUDSS_PHASE_SOLVE_BWD_PERM         = 1 << 8,
    CUDSS_PHASE_SOLVE_REFINEMENT       = 1 << 9,
    CUDSS_PHASE_SOLVE                  = CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD | CUDSS_PHASE_SOLVE_DIAG | CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM | CUDSS_PHASE_SOLVE_REFINEMENT
} cudssPhase_t;
typedef enum cudssMatrixFormat_t {
  CUDSS_MFORMAT_DENSE       = 1,
  CUDSS_MFORMAT_CSR         = 2,
  CUDSS_MFORMAT_BATCH       = 4,
  CUDSS_MFORMAT_DISTRIBUTED = 8
} cudssMatrixFormat_t;
typedef enum cudssMatrixType_t {
    CUDSS_MTYPE_GENERAL,
    CUDSS_MTYPE_SYMMETRIC,
    CUDSS_MTYPE_HERMITIAN,
    CUDSS_MTYPE_SPD,
    CUDSS_MTYPE_HPD
} cudssMatrixType_t;
typedef enum cudssMatrixViewType_t {
    CUDSS_MVIEW_FULL,
    CUDSS_MVIEW_LOWER,
    CUDSS_MVIEW_UPPER
} cudssMatrixViewType_t;
typedef enum cudssIndexBase_t {
    CUDSS_BASE_ZERO,
    CUDSS_BASE_ONE
} cudssIndexBase_t;
typedef enum cudssLayout_t {
    CUDSS_LAYOUT_COL_MAJOR,
    CUDSS_LAYOUT_ROW_MAJOR
} cudssLayout_t;
typedef enum cudssReorderingAlg_t {
    CUDSS_REORDERING_ALG_DEFAULT,
    CUDSS_REORDERING_ALG_BTF_COLAMD,
    CUDSS_REORDERING_ALG_COLAMD,
    CUDSS_REORDERING_ALG_AMD,
    CUDSS_REORDERING_ALG_NESTED_DISSECTION,
    CUDSS_REORDERING_ALG_NONE   
} cudssReorderingAlg_t;
typedef enum cudssFactorizationAlg_t {
    CUDSS_FACTORIZATION_ALG_DEFAULT,
    CUDSS_FACTORIZATION_ALG_MULTIBLOCK,
    CUDSS_FACTORIZATION_ALG_GENERAL
} cudssFactorizationAlg_t;
typedef enum cudssSolveAlg_t {
    CUDSS_SOLVE_ALG_DEFAULT,
    CUDSS_SOLVE_ALG_GENERAL
} cudssSolveAlg_t;
typedef enum cudssPivotEpsilonAlg_t {
    CUDSS_PIVOT_EPSILON_ALG_DEFAULT,
    CUDSS_PIVOT_EPSILON_ALG_SCALED,
    CUDSS_PIVOT_EPSILON_ALG_STATIC,
} cudssPivotEpsilonAlg_t;
typedef enum cudssMatchingAlg_t {
    CUDSS_MATCHING_ALG_NONE = 0,        
    CUDSS_MATCHING_ALG_MAX_DIAG_COUNT = 1,
    CUDSS_MATCHING_ALG_MAX_MIN_DIAG = 2,
    CUDSS_MATCHING_ALG_MAX_MIN_DIAG_ALT = 3,
    CUDSS_MATCHING_ALG_MAX_DIAG_SUM = 4,
    CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT = 5,
    CUDSS_MATCHING_ALG_AUTO = 6,        
} cudssMatchingAlg_t;
typedef enum cudssPivotType_t {
    CUDSS_PIVOT_AUTO,          
    CUDSS_PIVOT_NONE,          
    CUDSS_PIVOT_GLOBAL_COL,    
    CUDSS_PIVOT_GLOBAL_ROW,    
    CUDSS_PIVOT_DIAGONAL,      
    CUDSS_PIVOT_LOCAL_BLOCK,   
    CUDSS_PIVOT_BUNCH_KAUFMAN  
} cudssPivotType_t;
typedef enum cudssOpType_t {
    CUDSS_SUM,
    CUDSS_MAX,
    CUDSS_MIN
} cudssOpType_t;
typedef struct {} cudssDeviceMemHandler_t;
typedef void (*cudssLoggerCallback_t)(int, const char*, const char*);

inline cudssStatus_t cudssCreate(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssCreateMg(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssDestroy(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssGetProperty(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssSetStream(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssSetMgStreams(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssSetDeviceMemHandler(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssGetDeviceMemHandler(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssSetCommLayer(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssSetThreadingLayer(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssLoggerSetCallback(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssLoggerSetFile(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssLoggerOpenFile(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssLoggerSetLevel(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssLoggerSetMask(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssLoggerForceDisable(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssConfigCreate(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssConfigDestroy(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssConfigSet(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssConfigGet(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssDataCreate(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssDataDestroy(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssDataSet(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssDataGet(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssExecute(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixCreateDn(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixCreateBatchDn(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixCreateCsr(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixCreateBatchCsr(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixDestroy(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixSetValues(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixSetBatchValues(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixSetCsrPointers(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixSetBatchCsrPointers(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixSetDistributionRow1d(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixGetDn(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixGetBatchDn(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixGetCsr(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixGetBatchCsr(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixGetDistributionRow1d(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

inline cudssStatus_t cudssMatrixGetFormat(...) {
    return CUDSS_STATUS_NOT_SUPPORTED;
}

}  // extern "C"

#endif  // INCLUDE_GUARD_CUPY_HIP_CUDSS_H
