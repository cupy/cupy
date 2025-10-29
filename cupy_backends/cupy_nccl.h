#ifndef INCLUDE_GUARD_CUPY_NCCL_H
#define INCLUDE_GUARD_CUPY_NCCL_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "cuda/cupy_nccl.h"

#elif defined(CUPY_USE_HIP)

#include "hip/cupy_rccl.h"

#else // #ifndef CUPY_NO_CUDA

#include "stub/cupy_nccl.h"

#endif

#ifndef NCCL_MAJOR
#define NCCL_MAJOR 1
#define NCCL_MINOR 0
#define NCCL_PATCH 0
#endif

#ifndef NCCL_VERSION
#define NCCL_VERSION(X, Y, Z)                                                  \
  (((X) <= 2 && (Y) <= 8) ? (X)*1000 + (Y)*100 + (Z)                           \
                          : (X)*10000 + (Y)*100 + (Z))
#endif

#ifndef NCCL_VERSION_CODE
#define NCCL_VERSION_CODE NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH)
#endif


#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 0, 0))

ncclDataType_t _get_proper_datatype(ncclDataType_t datatype) {
    return datatype;
}

#else // #if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 0, 0))

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

#ifndef CUPY_NO_CUDA
ncclResult_t ncclGroupStart() {
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
    return ncclSuccess;
}
#endif // #ifndef CUPY_NO_CUDA
#endif // #if (NCCL_VERSION_CODE < NCCL_VERSION(2, 0, 0))

#if (NCCL_VERSION_CODE < NCCL_VERSION(2, 2, 0))
// New function in 2.2
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
			   ncclDataType_t datatype, int root, ncclComm_t comm,
			   cudaStream_t stream) {
    return ncclSuccess;
}
#endif // #if (NCCL_VERSION_CODE < NCCL_VERSION(2, 2, 0))

#if (NCCL_VERSION_CODE < NCCL_VERSION(2, 3, 4))

ncclResult_t ncclGetVersion(int *version) {
    *version = 0;
    return ncclSuccess;
}

#endif // #if (NCCL_VERSION_CODE < NCCL_VERSION(2, 3, 4))

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
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 0, 0))
    return ncclAllGather(sendbuff, recvbuff, sendcount, _datatype, comm, stream);
#else
    return ncclAllGather(sendbuff, sendcount, _datatype, recvbuff, comm, stream);
#endif // #if (NCCL_VERSION_CODE < NCCL_VERSION(2, 0, 0))
}

#if (NCCL_VERSION_CODE < NCCL_VERSION(2, 4, 0))
// New functions in 2.4
#define UNUSED(x) ((void)x)

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  UNUSED(comm);
  UNUSED(asyncError);
  return ncclSuccess;
}

void ncclCommAbort(ncclComm_t comm) {
  UNUSED(comm);
}
#endif

#if (NCCL_VERSION_CODE < NCCL_VERSION(2, 7, 0))
// New functions in 2.7
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    return ncclSuccess;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
    return ncclSuccess;
}
#endif

#if (NCCL_VERSION_CODE < NCCL_VERSION(2, 18, 0))
// mock `ncclConfig_t` with our own struct
typedef struct {
    // expose more fields if necessary
    int splitShare;
} _ncclConfig_t;

static inline _ncclConfig_t ncclConfigInitializer() {
    return _ncclConfig_t{0};
}

ncclResult_t _ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm,
                            _ncclConfig_t *config) {
    return ncclSuccess;
}

ncclResult_t _ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                                     int rank, _ncclConfig_t *config) {
    return ncclSuccess;
}

#else // #if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0))
// reuse the original struct
typedef ncclConfig_t _ncclConfig_t;

static inline _ncclConfig_t ncclConfigInitializer() {
    return (_ncclConfig_t)NCCL_CONFIG_INITIALIZER;
}

ncclResult_t _ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm,
                            _ncclConfig_t *config) {
    return ncclCommSplit(comm, color, key, newcomm, config);
}

ncclResult_t _ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                                     int rank, _ncclConfig_t *config) {
    return ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}
#endif

#endif // #ifndef INCLUDE_GUARD_CUPY_NCCL_H
