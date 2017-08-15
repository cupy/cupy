"""
Wrapper for NCCL: Optimized primiteive for collective multi-GPU communication
"""
cpdef enum:
    NCCL_SUM = 0
    NCCL_PROD = 1
    NCCL_MAX = 2
    NCCL_MIN = 3

    NCCL_INT8 = 0
    NCCL_CHAR = 0
    NCCL_UINT8 = 1
    NCCL_INT32 = 2
    NCCL_INT = 2
    NCCL_UINT32 = 3
    NCCL_INT64 = 4
    NCCL_UINT64 = 5
    NCCL_FLOAT16 = 6
    NCCL_HALF = 6
    NCCL_FLOAT32 = 7
    NCCL_FLOAT = 7
    NCCL_FLOAT64 = 8
    NCCL_DOUBLE = 8
