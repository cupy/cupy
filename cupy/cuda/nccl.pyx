# distutils: language = c++

"""
Wrapper for NCCL: Optimized primiteive for collective multi-GPU communication
"""
cimport cython  # NOQA

from cupy.cuda cimport driver
from cupy.cuda cimport runtime

cdef extern from 'cupy_nccl.h':
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
    ncclResult_t ncclGetVersion(int* version)
    ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,
                                       ncclResult_t *asyncError) nogil
    ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)
    ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev,
                                  ncclUniqueId commId, int rank)
    ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist)
    void ncclCommDestroy(ncclComm_t comm)
    void ncclCommAbort(ncclComm_t comm)
    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device)
    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank)
    ncclResult_t _ncclAllReduce(const void* sendbuff, void* recvbuff,
                                size_t count,
                                ncclDataType_t datatype, ncclRedOp_t op,
                                ncclComm_t comm, driver.Stream stream) nogil
    ncclResult_t _ncclReduce(const void* sendbuff, void* recvbuf, size_t count,
                             ncclDataType_t datatype, ncclRedOp_t op, int root,
                             ncclComm_t comm, driver.Stream stream) nogil
    ncclResult_t _ncclBroadcast(const void* sendbuff, void* recvbuff,
                                size_t count, ncclDataType_t datatype,
                                int root, ncclComm_t comm,
                                driver.Stream stream) nogil
    ncclResult_t _ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                            int root, ncclComm_t comm,
                            driver.Stream stream) nogil
    ncclResult_t _ncclReduceScatter(const void* sendbuff,
                                    void* recvbuff, size_t recvcount,
                                    ncclDataType_t datatype, ncclRedOp_t op,
                                    ncclComm_t comm,
                                    driver.Stream stream) nogil
    ncclResult_t _ncclAllGather(const void* sendbuff, void* recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclComm_t comm, driver.Stream stream) nogil

    # Build-time version
    int NCCL_VERSION_CODE

from cpython cimport array
import array

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


class NcclError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        cdef msg = ncclGetErrorString(<ncclResult_t>status)
        if NCCL_VERSION_CODE < 2000:
            s = ERROR1[status]
        else:
            s = ERROR2[status]
        super(NcclError, self).__init__(
            '%s: %s' % (s, msg.decode()))


@cython.profile(False)
cpdef inline check_status(ncclResult_t status):
    if status != ncclSuccess:
        raise NcclError(status)


def get_build_version():
    return NCCL_VERSION_CODE


def get_version():
    """Returns the runtime version of NCCL.

    This function will return 0 when built with NCCL version earlier than
    2.3.4, which does not support ``ncclGetVersion`` API.
    """
    cdef int version
    status = ncclGetVersion(&version)
    check_status(status)
    return version


def get_unique_id():
    cdef ncclUniqueId uniqueId
    status = ncclGetUniqueId(&uniqueId)
    check_status(status)
    ret = tuple([<char>uniqueId.internal[i]
                 for i in range(NCCL_UNIQUE_ID_BYTES)])
    return ret


def _bytesize(datatype):
    bytesize = {NCCL_INT8: 1,
                NCCL_UINT8: 1,
                NCCL_INT32: 4,
                NCCL_UINT32: 4,
                NCCL_INT64: 8,
                NCCL_UINT64: 8,
                NCCL_FLOAT16: 2,
                NCCL_FLOAT32: 4,
                NCCL_FLOAT64: 8}
    if datatype not in bytesize:
        raise ValueError('Unknow datatype {}'.format(datatype))
    return bytesize[datatype]


cdef class NcclCommunicator:

    cdef:
        ncclComm_t _comm

    def __cinit__(self):
        self._comm = <ncclComm_t>0

    def __init__(self, int ndev, tuple commId, int rank):
        cdef ncclUniqueId _uniqueId
        assert len(commId) == NCCL_UNIQUE_ID_BYTES
        for i in range(NCCL_UNIQUE_ID_BYTES):
            _uniqueId.internal[i] = commId[i]
        status = ncclCommInitRank(&self._comm, ndev, _uniqueId, rank)
        check_status(status)

    def __dealloc__(self):
        self.destroy()

    @staticmethod
    def initAll(int ndev, list devlist=None):
        """ Initialize NCCL communicator in a single process.

        Args:
            ndev (int): Number of GPUs to be used.
            devlist (None or list): A list of integers that label the GPUs to be
                used. The default is None, meaning that the first ndev GPUs will
                be chosen.

        Returns:
            NcclCommunicator: An NcclCommunicator instance.

        .. note::
            This method is to be used to create an NCCL communicator in a single
            process like this:

            .. code-block:: python

                from cupy.cuda import nccl
                # Use GPU #0, #2, and #3
                comm = nccl.NcclCommunicator.initAll(3, [0, 2, 3])

            In a multi-process setup, use the default initializer instead.
        """
        cdef array.array devices
        cdef int * devices_ptr
        # Call to __new__ bypasses __init__ constructor
        cdef NcclCommunicator NcclComm = \
            NcclCommunicator.__new__(NcclCommunicator)
        if devlist is not None:
            assert len(devlist) == ndev
            devices = array.array('i', devlist)
            devices_ptr = devices.data.as_ints
        else:
            devices_ptr = NULL
        status = ncclCommInitAll(&NcclComm._comm, ndev, devices_ptr)
        check_status(status)
        return NcclComm

    cpdef destroy(self):
        if self._comm:
            ncclCommDestroy(self._comm)
            self._comm = <ncclComm_t>0

    cpdef abort(self):
        if NCCL_VERSION_CODE < 2400:
            raise RuntimeError('ncclCommAbort is not available'
                               ' in this version')
        if self._comm:
            ncclCommAbort(self._comm)
            self._comm = <ncclComm_t>0

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
        with nogil:
            status = _ncclAllReduce(<void*>sendbuf, <void*>recvbuf,
                                    count, <ncclDataType_t>datatype,
                                    <ncclRedOp_t>op, self._comm,
                                    <driver.Stream>stream)
        check_status(status)

    def reduce(self, size_t sendbuf, size_t recvbuf,
               size_t count, int datatype, int op, int root, size_t stream):
        with nogil:
            status = _ncclReduce(<void*>sendbuf, <void*>recvbuf,
                                 count, <ncclDataType_t>datatype,
                                 <ncclRedOp_t>op, root, self._comm,
                                 <driver.Stream>stream)
        check_status(status)

    def broadcast(self, size_t sendbuff, size_t recvbuff, int count,
                  int datatype, int root, size_t stream):
        if NCCL_VERSION_CODE < 2200:
            # ncclBroadcast is not available in NCCL 2.1 or older.
            if self.rank_id() == root and sendbuff != recvbuff:
                runtime.memcpyAsync(recvbuff, sendbuff,
                                    count * _bytesize(datatype),
                                    runtime.memcpyDeviceToDevice, stream)
            self.bcast(recvbuff, count, datatype, root, stream)
            return
        with nogil:
            status = _ncclBroadcast(<const void*>sendbuff, <void*>recvbuff,
                                    count, <ncclDataType_t>datatype, root,
                                    self._comm, <driver.Stream>stream)
        check_status(status)

    def bcast(self, size_t buff, int count, int datatype,
              int root, size_t stream):
        with nogil:
            status = _ncclBcast(<void*>buff, count,
                                <ncclDataType_t>datatype, root,
                                self._comm, <driver.Stream>stream)
        check_status(status)

    def reduceScatter(self, size_t sendbuf, size_t recvbuf,
                      size_t recvcount, int datatype, int op, size_t stream):
        with nogil:
            status = _ncclReduceScatter(<void*>sendbuf, <void*>recvbuf,
                                        recvcount, <ncclDataType_t>datatype,
                                        <ncclRedOp_t>op, self._comm,
                                        <driver.Stream>stream)
        check_status(status)

    def allGather(self, size_t sendbuf, size_t recvbuf, size_t count,
                  int datatype, size_t stream):
        with nogil:
            status = _ncclAllGather(<void*>sendbuf, <void*>recvbuf,
                                    count, <ncclDataType_t>datatype,
                                    self._comm, <driver.Stream>stream)
        check_status(status)

    def check_async_error(self):
        if NCCL_VERSION_CODE < 2400:
            raise RuntimeError('ncclCommGetAsyncError is not available'
                               ' in this version')
        cdef ncclResult_t asyncError = ncclSuccess
        # Releasing GIL as the function *might* block in future and
        # this won't be a hot code path. At least in NCCL 2.4 it does
        # not block so far.
        with nogil:
            result = ncclCommGetAsyncError(self._comm, &asyncError)
        check_status(asyncError)
        check_status(result)
