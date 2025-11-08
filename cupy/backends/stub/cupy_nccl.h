// This file is a stub header file of nccl for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_NCCL_H
#define INCLUDE_GUARD_STUB_CUPY_NCCL_H

#define NCCL_MAJOR 0
#define NCCL_MINOR 0
#define NCCL_PATCH 0

extern "C" {

typedef struct ncclComm* ncclComm_t;

enum {
    NCCL_UNIQUE_ID_BYTES = 128
};
typedef struct {
    char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

typedef enum {
    ncclSuccess
} ncclResult_t;

typedef enum {} ncclRedOp_t;

typedef enum {
    ncclChar       = 0,
    ncclInt        = 1,
    ncclHalf       = 2,
    ncclFloat      = 3,
    ncclDouble     = 4,
    ncclInt64      = 5,
    ncclUint64     = 6,
    nccl_NUM_TYPES = 7 } ncclDataType_t;

const char* ncclGetErrorString(...) {
    return "";
}

ncclResult_t  ncclCommGetAsyncError(...) {
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(...) {
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(...) {
    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(...) {
    return ncclSuccess;
}

ncclResult_t ncclGroupStart(...) {
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(...) {
    return ncclSuccess;
}

void ncclCommDestroy(...) {
}

void ncclCommAbort(...) {
}

ncclResult_t ncclCommCuDevice(...) {
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(...) {
    return ncclSuccess;
}

ncclResult_t ncclCommCount(...) {
    return ncclSuccess;
}

ncclResult_t ncclAllReduce(...) {
    return ncclSuccess;
}

ncclResult_t ncclReduce(...) {
    return ncclSuccess;
}

ncclResult_t ncclBroadcast(...) {
    return ncclSuccess;
}

ncclResult_t ncclBcast(...) {
    return ncclSuccess;
}

ncclResult_t ncclReduceScatter(...) {
    return ncclSuccess;
}

ncclResult_t ncclAllGather(...) {
    return ncclSuccess;
}

ncclResult_t ncclSend(...) {
    return ncclSuccess;
}

ncclResult_t ncclRecv(...) {
    return ncclSuccess;
}

typedef struct CUstream_st *cudaStream_t;

}  // extern "C"

#endif  // INCLUDE_GUARD_STUB_CUPY_NCCL_H
