from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t, intptr_t

###############################################################################
# Enum
###############################################################################

cpdef enum:
    # cusparseLtSparsity_t
    CUSPARSELT_SPARSITY_50_PERCENT = 0

    # cusparseComputeType
    CUSPARSE_COMPUTE_16F = 0
    CUSPARSE_COMPUTE_32I = 1

    # cusparseLtMatmulAlg_t
    CUSPARSELT_MATMUL_ALG_DEFAULT = 0

    # cusparseLtMatmulAlgAttribute_t
    CUSPARSELT_MATMUL_ALG_CONFIG_ID = 0      # NOQA, READ/WRITE
    CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID = 1  # NOQA, READ-ONLY
    CUSPARSELT_MATMUL_SEARCH_ITERATIONS = 2  # NOQA, READ/WRITE

    # cusparseLtPruneAlg_t
    CUSPARSELT_PRUNE_SPMMA_TILE = 0
    CUSPARSELT_PRUNE_SPMMA_STRIP = 1
