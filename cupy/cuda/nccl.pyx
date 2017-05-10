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
    ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, driver.Stream stream)
    ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuf, int count,
                             ncclDataType_t datatype, ncclRedOp_t op, int root,
                             ncclComm_t comm, driver.Stream stream)
    ncclResult_t  ncclBcast(void* buff, int count, ncclDataType_t datatype,
                            int root, ncclComm_t comm, driver.Stream stream)


cdef dict STATUS = {
    0: 'NCCL_STATUS_SUCCESS',
    1: 'NCCL_STATUS_UNHANDLED_CUDA_ERROR',
    2: 'NCCL_STATUS_SYSTEM_ERROR',
    3: 'NCCL_STATUS_INTERNAL_ERROR',
    4: 'NCCL_STATUS_INVALID_DEVICE_POINTER',
    5: 'NCCL_STATUS_INVALID_RANK',
    6: 'NCCL_STATUS_UNSUPPORTED_DEVICE_COUNT',
    7: 'NCCL_STATUS_DEVICE_NOT_FOUND',
    8: 'NCCL_STATUS_INVALID_DEVICE_INDEX',
    9: 'NCCL_STATUS_LIB_WRAPPER_NOT_SET',
    10: 'NCCL_STATUS_CUDA_MALLOC_FAILED',
    11: 'NCCL_STATUS_RANK_MISMATCH',
    12: 'NCCL_STATUS_INVALID_ARGUMENT',
    13: 'NCCL_STATUS_INVALID_TYPE',
    14: 'NCCL_STATUS_INVALID_OPERATION',
}


class NcclError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        cdef msg = ncclGetErrorString(<ncclResult_t>status)
        super(NcclError, self).__init__(
            '%s: %s' % (STATUS[status], msg.decode()))


@cython.profile(False)
cpdef inline check_status(ncclResult_t status):
    if status != ncclSuccess:
        raise NcclError(status)


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
                  int count, int datatype, int op, size_t stream):
        status = ncclAllReduce(<void*>sendbuf, <void*>recvbuf, count,
                               <ncclDataType_t>datatype, <ncclRedOp_t>op,
                               self._comm, <driver.Stream>stream)
        check_status(status)

    def reduce(self, size_t sendbuf, size_t recvbuf,
               int count, int datatype, int op, int root, size_t stream):
        status = ncclReduce(<void*> sendbuf, <void*> recvbuf, count,
                            <ncclDataType_t> datatype, <ncclRedOp_t> op, root,
                            self._comm, <driver.Stream> stream)
        check_status(status)

    def bcast(self, size_t buff, int count, int datatype,
              int root, size_t stream):
        status = ncclBcast(<void*> buff, count,
                           <ncclDataType_t> datatype, root,
                           self._comm, <driver.Stream> stream)
        check_status(status)
