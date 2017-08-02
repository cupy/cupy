# distutils: language = c++

"""
Wrapper for NCCL: Optimized primiteive for collective multi-GPU communication
"""
cimport cython

from cupy.cuda cimport driver

cdef extern from "cupy_nccl.h":
    ctypedef struct ncclComm:
        pass
    ctypedef ncclComm* ncclComm_t
    cdef enum:
        NCCL_UNIQUE_ID_BYTES = 128
    ctypedef struct ncclUniqueId:
        char internal[NCCL_UNIQUE_ID_BYTES]
    ctypedef enum ncclResult_t:
        ncclSuccess
    ctypedef enum ncclRedOp_t:
        pass
    ctypedef enum ncclDataType_t:
        pass

    const char* ncclGetErrorString(ncclResult_t result)
    ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)
    ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev,
                                  ncclUniqueId commId, int rank)
    void ncclCommDestroy(ncclComm_t comm)
    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device)
    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank)
    ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, driver.Stream stream)
    ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuf, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op, int root,
                             ncclComm_t comm, driver.Stream stream)
    ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                            int root, ncclComm_t comm, driver.Stream stream)

    int NCCL_VERSION


cdef dict ERROR1 = {
    0: 'NCCL_ERROR_SUCCESS',
    1: 'NCCL_ERROR_UNHANDLED_CUDA_ERROR',
    2: 'NCCL_ERROR_SYSTEM_ERROR',
    3: 'NCCL_ERROR_INTERNAL_ERROR',
    4: 'NCCL_ERROR_INVALID_DEVICE_POINTER',
    5: 'NCCL_ERROR_INVALID_RANK',
    6: 'NCCL_ERROR_UNSUPPORTED_DEVICE_COUNT',
    7: 'NCCL_ERROR_DEVICE_NOT_FOUND',
    8: 'NCCL_ERROR_INVALID_DEVICE_INDEX',
    9: 'NCCL_ERROR_LIB_WRAPPER_NOT_SET',
    10: 'NCCL_ERROR_CUDA_MALLOC_FAILED',
    11: 'NCCL_ERROR_RANK_MISMATCH',
    12: 'NCCL_ERROR_INVALID_ARGUMENT',
    13: 'NCCL_ERROR_INVALID_TYPE',
    14: 'NCCL_ERROR_INVALID_OPERATION',
}

cdef dict ERROR2 = {
    0: 'NCCL_ERROR_SUCCESS',
    1: 'NCCL_ERROR_UNHANDLED_CUDA_ERROR',
    2: 'NCCL_ERROR_SYSTEM_ERROR',
    3: 'NCCL_ERROR_INTERNAL_ERROR',
    4: 'NCCL_ERROR_INVALID_ARGUMENT',
    5: 'NCCL_ERROR_INVALID_USAGE',
}


class Nccl1Error(RuntimeError):

    def __init__(self, int status):
        self.status = status
        cdef msg = ncclGetErrorString(<ncclResult_t>status)
        super(Nccl1Error, self).__init__(
            '%s: %s' % (ERROR1[status], msg.decode()))


class Nccl2Error(RuntimeError):

    def __init__(self, int status):
        self.status = status
        cdef msg = ncclGetErrorString(<ncclResult_t>status)
        super(Nccl2Error, self).__init__(
            '%s: %s' % (ERROR2[status], msg.decode()))


@cython.profile(False)
cpdef inline check_status(ncclResult_t status):
    if status != ncclSuccess:
        if NCCL_VERSION < 2000:
            raise Nccl1Error(status)
        else:
            raise Nccl2Error(status)


def get_version():
    return NCCL_VERSION


def get_unique_id():
    cdef ncclUniqueId uniqueId
    status = ncclGetUniqueId(&uniqueId)
    check_status(status)
    ret = tuple([<char>uniqueId.internal[i]
                 for i in range(NCCL_UNIQUE_ID_BYTES)])
    return ret


cdef class NcclCommunicator:

    cdef:
        ncclComm_t _comm

    def __init__(self, int ndev, tuple commId, int rank):
        cdef ncclUniqueId _uniqueId
        self._comm = <ncclComm_t>0
        assert len(commId) == NCCL_UNIQUE_ID_BYTES
        for i in range(NCCL_UNIQUE_ID_BYTES):
            _uniqueId.internal[i] = commId[i]
        status = ncclCommInitRank(&self._comm, ndev, _uniqueId, rank)
        check_status(status)

    def __dealloc__(self):
        if self._comm:
            ncclCommDestroy(self._comm)

    def device_id(self):
        cdef int device_id
        status = ncclCommCuDevice(self._comm, &device_id)
        check_status(status)
        return device_id

    def rank_id(self):
        cdef int rank_id
        status = ncclCommUserRank(self._comm, &rank_id)
        check_status(status)
        return rank_id

    def allReduce(self, size_t sendbuf, size_t recvbuf,
                  size_t count, int datatype, int op, size_t stream):
        if NCCL_VERSION >= 2000:
            status = ncclAllReduce(<void*>sendbuf, <void*>recvbuf, count,
                                   <ncclDataType_t>datatype, <ncclRedOp_t>op,
                                   self._comm, <driver.Stream>stream)
        else:
            status = ncclAllReduce(<void*>sendbuf, <void*>recvbuf, <int>count,
                                   <ncclDataType_t>datatype, <ncclRedOp_t>op,
                                   self._comm, <driver.Stream>stream)
        check_status(status)

    def reduce(self, size_t sendbuf, size_t recvbuf,
               size_t count, int datatype, int op, int root, size_t stream):
        if NCCL_VERSION >= 2000:
            status = ncclReduce(<void*> sendbuf, <void*> recvbuf, count,
                                <ncclDataType_t> datatype, <ncclRedOp_t> op, root,
                                self._comm, <driver.Stream> stream)
        else:
            status = ncclReduce(<void*> sendbuf, <void*> recvbuf, <int> count,
                                <ncclDataType_t> datatype, <ncclRedOp_t> op, root,
                                self._comm, <driver.Stream> stream)
        check_status(status)

    def bcast(self, size_t buff, int count, int datatype,
              int root, size_t stream):
        if NCCL_VERSION >= 2000:
            status = ncclBcast(<void*> buff, count,
                               <ncclDataType_t> datatype, root,
                               self._comm, <driver.Stream> stream)
        else:
            status = ncclBcast(<void*> buff, <int> count,
                               <ncclDataType_t> datatype, root,
                               self._comm, <driver.Stream> stream)
        check_status(status)
