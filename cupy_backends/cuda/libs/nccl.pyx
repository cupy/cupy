# distutils: language = c++

"""
Wrapper for NCCL: Optimized primiteive for collective multi-GPU communication
"""
cimport cython  # NOQA

from libc.stdint cimport intptr_t
from libcpp cimport vector

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime

cdef extern from '../../cupy_nccl.h':
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
    ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev,
                                 const int* devlist)
    ncclResult_t ncclGroupStart() nogil
    ncclResult_t ncclGroupEnd() nogil
    void ncclCommDestroy(ncclComm_t comm)
    void ncclCommAbort(ncclComm_t comm)
    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device)
    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank)
    ncclResult_t ncclCommCount(const ncclComm_t comm, int* count)
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
    ncclResult_t ncclSend(const void* sendbuff, size_t count,
                          ncclDataType_t datatype, int peer, ncclComm_t comm,
                          driver.Stream stream) nogil
    ncclResult_t ncclRecv(void* recvbuff, size_t count,
                          ncclDataType_t datatype, int peer, ncclComm_t comm,
                          driver.Stream stream) nogil

    # Build-time version
    int NCCL_VERSION_CODE


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


available = True


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

    def __reduce__(self):
        return (type(self), (self.status,))


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


cpdef groupStart():
    """Start a group of NCCL calls. Must be paired with :func:`groupEnd()`.

    .. note::
        This method is useful when the ``NcclCommunicator`` instances are
        created via :meth:`~.NcclCommunicator.initAll`. A typical usage pattern
        is like this:

        .. code-block:: python

            comms = cupy.cuda.nccl.NcclCommunicator.initAll(n, dev_list)
            # ... do some preparation work
            cupy.cuda.nccl.groupStart()
            for rank, comm in enumerate(comms):
                # ... make some collective calls ...
            cupy.cuda.nccl.groupEnd()

        Other use cases include fusing several NCCL calls into one, and
        point-to-point communications using :meth:`~.NcclCommunicator.send` and
        :meth:`~.NcclCommunicator.recv` (with NCCL 2.7+).

    .. seealso:: `ncclGroupStart`_, `Group Calls`_

    .. _ncclGroupStart:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupstart

    .. _Group calls:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
    """
    if NCCL_VERSION_CODE < 2000:
        raise RuntimeError('ncclGroupStart is not available in this version')
    with nogil:
        status = ncclGroupStart()
    check_status(status)


cpdef groupEnd():
    """End a group of NCCL calls. Must be paired with :func:`groupStart()`.

    .. note::
        This method is useful when the ``NcclCommunicator`` instances are
        created via :meth:`~.NcclCommunicator.initAll`. A typical usage pattern
        is like this:

        .. code-block:: python

            comms = cupy.cuda.nccl.NcclCommunicator.initAll(n, dev_list)
            # ... do some preparation work
            cupy.cuda.nccl.groupStart()
            for rank, comm in enumerate(comms):
                # ... make some collective calls ...
            cupy.cuda.nccl.groupEnd()

        Other use cases include fusing several NCCL calls into one, and
        point-to-point communications using :meth:`~.NcclCommunicator.send` and
        :meth:`~.NcclCommunicator.recv` (with NCCL 2.7+).

    .. seealso:: `ncclGroupEnd`_, `Group Calls`_

    .. _ncclGroupEnd:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#ncclgroupend

    .. _Group calls:
        https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html
    """
    if NCCL_VERSION_CODE < 2000:
        raise RuntimeError('ncclGroupEnd is not available in this version')
    with nogil:
        status = ncclGroupEnd()
    check_status(status)


cdef class NcclCommunicator:
    """ Initialize an NCCL communicator for one device controlled by one
    process.

    Args:
        ndev (int): Total number of GPUs to be used.
        commId (tuple): The unique ID returned by :func:`get_unique_id`.
        rank (int): The rank of the GPU managed by the current process.

    Returns:
        NcclCommunicator: An ``NcclCommunicator`` instance.

    .. note::
        This method is for creating an NCCL communicator in a multi-process
        environment, typically managed by MPI or ``multiprocessing``. For
        controlling multiple devices by one process, use :meth:`initAll`
        instead.

    .. seealso:: `ncclCommInitRank`_

    .. _ncclCommInitRank:
        https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/api/comms.html#ncclcomminitrank
    """  # noqa

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
    def initAll(devices):
        """ Initialize NCCL communicators for multiple devices in a single
        process.

        Args:
            devices (int or list of int): The number of GPUs or a list of GPUs
                to be used. For the former case, the first ``devices`` GPUs
                will be used.

        Returns:
            list: A list of ``NcclCommunicator`` instances.

        .. note::
            This method is for creating a group of NCCL communicators, each
            controlling one device, in a single process like this:

            .. code-block:: python

                from cupy.cuda import nccl
                # Use 3 GPUs: #0, #2, and #3
                comms = nccl.NcclCommunicator.initAll([0, 2, 3])
                assert len(comms) == 3

            In a multi-process setup, use the default initializer instead.

        .. seealso:: `ncclCommInitAll`_

        .. _ncclCommInitAll:
            https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/api/comms.html#ncclcomminitall
        """  # noqa
        cdef int i, ndev
        cdef list comms = [], devlist = []
        cdef NcclCommunicator comm

        if isinstance(devices, list):
            ndev = len(devices)
            devlist = devices
        elif isinstance(devices, int):
            ndev = devices
            for i in range(ndev):
                devlist.append(i)
        else:
            raise ValueError("\"devices\" should be an int or a list of int.")

        for i in range(ndev):
            # Call to __new__ bypasses __init__ constructor
            # these are just placeholders
            comm = NcclCommunicator.__new__(NcclCommunicator)
            comms.append(comm)
        NcclCommunicator._initAll(comms, ndev, devlist)

        return comms

    @staticmethod
    def _initAll(list comms, int ndev, list devlist=None):
        # A helper function which does not return is favorable for subclassing
        cdef vector.vector[int] devices
        cdef vector.vector[ncclComm_t] ncclComms
        cdef NcclCommunicator comm
        cdef int* devices_ptr

        if devlist is not None:
            assert len(devlist) == ndev
            for i in range(ndev):
                devices.push_back(devlist[i])
            devices_ptr = devices.data()
        else:
            devices_ptr = NULL
        for i in range(ndev):
            ncclComms.push_back(<ncclComm_t>0)
        status = ncclCommInitAll(ncclComms.data(), ndev, devices_ptr)
        check_status(status)
        # overwrite the _comm attribute in existing NcclCommunicator instances
        for i in range(ndev):
            comm = comms[i]
            comm._comm = ncclComms[i]

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

    def size(self):
        cdef int ranks
        status = ncclCommCount(self._comm, &ranks)
        check_status(status)
        return ranks

    def allReduce(self, intptr_t sendbuf, intptr_t recvbuf,
                  size_t count, int datatype, int op, intptr_t stream):
        with nogil:
            status = _ncclAllReduce(<void*>sendbuf, <void*>recvbuf,
                                    count, <ncclDataType_t>datatype,
                                    <ncclRedOp_t>op, self._comm,
                                    <driver.Stream>stream)
        check_status(status)

    def reduce(self, intptr_t sendbuf, intptr_t recvbuf,
               size_t count, int datatype, int op, int root, intptr_t stream):
        with nogil:
            status = _ncclReduce(<void*>sendbuf, <void*>recvbuf,
                                 count, <ncclDataType_t>datatype,
                                 <ncclRedOp_t>op, root, self._comm,
                                 <driver.Stream>stream)
        check_status(status)

    def broadcast(self, intptr_t sendbuff, intptr_t recvbuff, int count,
                  int datatype, int root, intptr_t stream):
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

    def bcast(self, intptr_t buff, int count, int datatype,
              int root, intptr_t stream):
        with nogil:
            status = _ncclBcast(<void*>buff, count,
                                <ncclDataType_t>datatype, root,
                                self._comm, <driver.Stream>stream)
        check_status(status)

    def reduceScatter(self, intptr_t sendbuf, intptr_t recvbuf,
                      size_t recvcount, int datatype, int op, intptr_t stream):
        with nogil:
            status = _ncclReduceScatter(<void*>sendbuf, <void*>recvbuf,
                                        recvcount, <ncclDataType_t>datatype,
                                        <ncclRedOp_t>op, self._comm,
                                        <driver.Stream>stream)
        check_status(status)

    def allGather(self, intptr_t sendbuf, intptr_t recvbuf, size_t count,
                  int datatype, intptr_t stream):
        with nogil:
            status = _ncclAllGather(<void*>sendbuf, <void*>recvbuf,
                                    count, <ncclDataType_t>datatype,
                                    self._comm, <driver.Stream>stream)
        check_status(status)

    def send(self, intptr_t sendbuf, size_t count, int datatype, int peer,
             intptr_t stream):
        if NCCL_VERSION_CODE < 2700:
            raise RuntimeError('ncclSend is not available in this version')
        with nogil:
            status = ncclSend(<void*>sendbuf, count, <ncclDataType_t>datatype,
                              peer, self._comm, <driver.Stream>stream)
        check_status(status)

    def recv(self, intptr_t recvbuf, size_t count, int datatype, int peer,
             intptr_t stream):
        if NCCL_VERSION_CODE < 2700:
            raise RuntimeError('ncclRecv is not available in this version')
        with nogil:
            status = ncclRecv(<void*>recvbuf, count, <ncclDataType_t>datatype,
                              peer, self._comm, <driver.Stream>stream)
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
