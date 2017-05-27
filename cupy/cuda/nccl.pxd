"""
Wrapper for NCCL: Optimized primiteive for collective multi-GPU communication
"""
cpdef enum:
    NCCL_SUM = 0
    NCCL_PROD = 1
    NCCL_MAX = 2
    NCCL_MIN = 3

    NCCL_CHAR = 0
    NCCL_INT = 1
    NCCL_HALF = 2
    NCCL_FLOAT = 3
    NCCL_DOUBLE = 4
    NCCL_INT64 = 5
    NCCL_UINT64 = 6
