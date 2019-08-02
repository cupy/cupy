// This file is a stub header file of nccl for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_NCCL_H
#define INCLUDE_GUARD_CUPY_NCCL_H

#ifndef CUPY_NO_CUDA

#include <nccl.h>

#ifndef NCCL_MAJOR
#ifndef CUDA_HAS_HALF
#define ncclHalf ((ncclDataType_t)2)
#endif
#endif

#else // #ifndef CUPY_NO_CUDA

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

typedef struct CUstream_st *cudaStream_t;

}  // extern "C"

#endif // #ifndef CUPY_NO_CUDA

#ifndef NCCL_MAJOR
#define NCCL_MAJOR 1
#define NCCL_MINOR 0
#define NCCL_PATCH 0
#endif

#ifndef NCCL_VERSION_CODE
#define NCCL_VERSION_CODE (NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH)
#endif


#if (NCCL_VERSION_CODE >= 2000)

ncclDataType_t _get_proper_datatype(ncclDataType_t datatype) {
    return datatype;
}

#else // #if (NCCL_VERSION_CODE >= 2000)

#define NCCL_CHAR_V1 ncclChar
#define NCCL_INT_V1 ncclInt
#define NCCL_HALF_V1 ncclHalf
#define NCCL_FLOAT_V1 ncclFloat
#define NCCL_DOUBLE_v1 ncclDouble
#define NCCL_INT64_v1 ncclInt64
#define NCCL_UINT64_v1 ncclUint64
#define NCCL_INVALID_TYPE_V1 nccl_NUM_TYPES

static const ncclDataType_t TYPE2TYPE_V1[] = {
    NCCL_CHAR_V1,         // ncclInt8, ncclChar
    NCCL_INVALID_TYPE_V1, // ncclUint8
    NCCL_INT_V1,          // ncclInt32, ncclInt
    NCCL_INVALID_TYPE_V1, // ncclUint32
    NCCL_INT64_v1,        // ncclInt64
    NCCL_UINT64_v1,       // ncclUint64
    NCCL_HALF_V1,         // ncclFloat16, ncclHalf
    NCCL_FLOAT_V1,        // ncclFloat32, ncclFloat
    NCCL_DOUBLE_v1        // ncclFloat64, ncclDouble
};

ncclDataType_t _get_proper_datatype(ncclDataType_t datatype) {
    return TYPE2TYPE_V1[datatype];
}

#endif // #if (NCCL_VERSION_CODE < 2000)

#if (NCCL_VERSION_CODE < 2200)
// New function in 2.2
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
			   ncclDataType_t datatype, int root, ncclComm_t comm,
			   cudaStream_t stream) {
    return ncclSuccess;
}
#endif // #if (NCCL_VERSION_CODE < 2200)

#if (NCCL_VERSION_CODE < 2304)

ncclResult_t ncclGetVersion(int *version) {
    *version = 0;
    return ncclSuccess;
}

#endif // #if (NCCL_VERSION_CODE < 2304)

#ifndef CUPY_NO_CUDA
#if (NCCL_VERSION_CODE < 2000)
ncclResult_t ncclGroupStart() {
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
    return ncclSuccess;
}
#endif // #if (NCCL_VERSION_CODE < 2200)
#endif // #ifndef CUPY_NO_CUDA

ncclResult_t _ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                            cudaStream_t stream) {
    ncclDataType_t _datatype = _get_proper_datatype(datatype);
    return ncclAllReduce(sendbuff, recvbuff, count, _datatype, op, comm, stream);
}


ncclResult_t _ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
                         cudaStream_t stream) {
    ncclDataType_t _datatype = _get_proper_datatype(datatype);
    return ncclReduce(sendbuff, recvbuff, count, _datatype, op, root, comm, stream);
}


ncclResult_t _ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
			    ncclDataType_t datatype, int root, ncclComm_t comm,
			    cudaStream_t stream) {
    ncclDataType_t _datatype = _get_proper_datatype(datatype);
    return ncclBroadcast(sendbuff, recvbuff, count, _datatype, root, comm,  stream);
}


ncclResult_t _ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
                        ncclComm_t comm, cudaStream_t stream) {
    ncclDataType_t _datatype = _get_proper_datatype(datatype);
    return ncclBcast(buff, count, _datatype, root, comm,  stream);
}


ncclResult_t _ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                cudaStream_t stream) {
    ncclDataType_t _datatype = _get_proper_datatype(datatype);
    return ncclReduceScatter(sendbuff, recvbuff, recvcount, _datatype, op, comm, stream);
}


ncclResult_t _ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                            ncclDataType_t datatype, ncclComm_t comm,
                            cudaStream_t stream) {
    ncclDataType_t _datatype = _get_proper_datatype(datatype);
#if (NCCL_VERSION_CODE >= 2000)
    return ncclAllGather(sendbuff, recvbuff, sendcount, _datatype, comm, stream);
#else
    return ncclAllGather(sendbuff, sendcount, _datatype, recvbuff, comm, stream);
#endif // #if (NCCL_VERSION_CODE < 2000)
}

#if (NCCL_VERSION_CODE < 2400)
// New functions in 2.4
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  return ncclSuccess;
}

void ncclCommAbort(ncclComm_t comm) {
}
#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_NCCL_H
