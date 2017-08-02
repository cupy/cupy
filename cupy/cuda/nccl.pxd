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

    NCCL2_INT8 = 0
    NCCL2_CHAR = 0
    NCCL2_UINT8 = 1
    NCCL2_INT32 = 2
    NCCL2_INT = 2
    NCCL2_UINT32 = 3
    NCCL2_INT64 = 4
    NCCL2_UINT64 = 5
    NCCL2_FLOAT16 = 6
    NCCL2_HALF = 6
    NCCL2_FLOAT32 = 7
    NCCL2_FLOAT = 7
    NCCL2_FLOAT64 = 8
    NCCL2_DOUBLE = 8
