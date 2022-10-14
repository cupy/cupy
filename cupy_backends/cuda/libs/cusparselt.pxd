from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t

###############################################################################
# Enum
###############################################################################

cpdef enum:
    # cusparseLtSparsity_t
    CUSPARSELT_SPARSITY_50_PERCENT = 0

    # cusparseLtMatDescAttribute_t
    CUSPARSELT_MAT_NUM_BATCHES = 0   # READ/WRITE
    CUSPARSELT_MAT_BATCH_STRIDE = 1  # READ/WRITE

    # cusparseComputeType
    CUSPARSE_COMPUTE_16F = 0
    CUSPARSE_COMPUTE_32I = 1
    CUSPARSE_COMPUTE_TF32 = 2
    CUSPARSE_COMPUTE_TF32_FAST = 3

    # cusparseLtMatmulDescAttribute_t
    CUSPARSELT_MATMUL_ACTIVATION_RELU = 0             # READ/WRITE
    CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND = 1  # READ/WRITE
    CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD = 2   # READ/WRITE
    CUSPARSELT_MATMUL_ACTIVATION_GELU = 3             # READ/WRITE
    CUSPARSELT_MATMUL_BIAS_STRIDE = 4                 # READ/WRITE
    CUSPARSELT_MATMUL_BIAS_POINTER = 5                # READ/WRITE

    # cusparseLtMatmulAlg_t
    CUSPARSELT_MATMUL_ALG_DEFAULT = 0

    # cusparseLtMatmulAlgAttribute_t
    CUSPARSELT_MATMUL_ALG_CONFIG_ID = 0      # NOQA, READ/WRITE
    CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID = 1  # NOQA, READ-ONLY
    CUSPARSELT_MATMUL_SEARCH_ITERATIONS = 2  # NOQA, READ/WRITE

    # cusparseLtPruneAlg_t
    CUSPARSELT_PRUNE_SPMMA_TILE = 0
    CUSPARSELT_PRUNE_SPMMA_STRIP = 1
