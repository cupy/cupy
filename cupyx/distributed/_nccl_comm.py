import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend


if nccl.available:
    # types are not compliant with windows on long/int32 issue
    # but nccl does not support windows so we don't care
    _nccl_dtypes = {'b': nccl.NCCL_INT8,
                    'B': nccl.NCCL_UINT8,
                    'i': nccl.NCCL_INT32,
                    'I': nccl.NCCL_UINT32,
                    'l': nccl.NCCL_INT64,
                    'L': nccl.NCCL_UINT64,
                    'q': nccl.NCCL_INT64,
                    'Q': nccl.NCCL_UINT64,
                    'e': nccl.NCCL_FLOAT16,
                    'f': nccl.NCCL_FLOAT32,
                    'd': nccl.NCCL_FLOAT64,
                    # Size of array will be doubled
                    'F': nccl.NCCL_FLOAT32,
                    'D': nccl.NCCL_FLOAT64}

    _nccl_ops = {'sum': nccl.NCCL_SUM,
                 'prod': nccl.NCCL_PROD,
                 'max': nccl.NCCL_MAX,
                 'min': nccl.NCCL_MIN}
else:
    _nccl_dtypes = {}

    _nccl_ops = {}


class NCCLBackend(_Backend):
    """Interface that uses NVIDIA's NCCL to perform communications.

    Args:
        n_devices (int): Total number of devices that will be used in the
            distributed execution.
        rank (int): Unique id of the GPU that the communicator is associated to
            its value needs to be `0 <= rank < n_devices`.
        host (str, optional): host address for the process rendezvous on
            initialization. Defaults to `"127.0.0.1"`.
        port (int, optional): port used for the process rendezvous on
            initialization. Defaults to `13333`.
    """

    def __init__(self, n_devices, rank,
                 host=_store._DEFAULT_HOST, port=_store._DEFAULT_PORT):
        super().__init__(n_devices, rank, host, port)
        if rank == 0:
            self._store.run(host, port)
            nccl_id = nccl.get_unique_id()
            # get_unique_id return negative values due to cython issues
            # with bytes && c strings. We shift them by 128 to
            # avoid issues
            nccl_id = bytes([b + 128 for b in nccl_id])
            self._store_proxy['nccl_id'] = nccl_id
            self._store_proxy.barrier()
        else:
            self._store_proxy.barrier()
            nccl_id = self._store_proxy['nccl_id']
        # Initialize devices
        nccl_id = tuple([int(b) - 128 for b in nccl_id])
        self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)

    def _check_contiguous(self, array):
        if not array.flags.c_contiguous and not array.flags.f_contiguous:
            raise RuntimeError(
                'NCCL requires arrays to be either c- or f-contiguous')

    def _get_nccl_dtype_and_count(self, array, count=None):
        dtype = array.dtype.char
        if dtype not in _nccl_dtypes:
            raise TypeError(f'Unknown dtype {array.dtype} for NCCL')
        nccl_dtype = _nccl_dtypes[dtype]
        if count is None:
            count = array.size
        if dtype in 'FD':
            return nccl_dtype, 2 * count
        return nccl_dtype, count

    def _get_stream(self, stream):
        if stream is None:
            stream = cupy.cuda.stream.get_current_stream()
        return stream.ptr

    def _get_op(self, op, dtype):
        if op not in _nccl_ops:
            raise RuntimeError(f'Unknown op {op} for NCCL')
        if dtype in 'FD' and op != nccl.NCCL_SUM:
            raise ValueError(
                'Only nccl.SUM is supported for complex arrays')
        return _nccl_ops[op]

    def all_reduce(self, in_array, out_array, op='sum', stream=None):
        """Performs an all reduce operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        op = self._get_op(op, in_array.dtype.char)
        self._comm.allReduce(
            in_array.data.ptr, out_array.data.ptr, count, dtype, op, stream)

    def reduce(self, in_array, out_array, root=0, op='sum', stream=None):
        """Performs a reduce operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
                will only be modified by the `root` process.
            root (int, optional): rank of the process that will perform the
                reduction. Defaults to `0`.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._check_contiguous(in_array)
        if self.rank == root:
            self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        op = self._get_op(op, in_array.dtype.char)
        self._comm.reduce(
            in_array.data.ptr, out_array.data.ptr,
            count, dtype, op, root, stream)

    def broadcast(self, in_out_array, root=0, stream=None):
        """Performs a broadcast operation.

        Args:
            in_out_array (cupy.ndarray): array to be sent for `root` rank.
                Other ranks will receive the broadcast data here.
            root (int, optional): rank of the process that will send the
                broadcast. Defaults to `0`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        # in_out_array for rank !=0 will be used as output
        self._check_contiguous(in_out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_out_array)
        self._comm.broadcast(
            in_out_array.data.ptr, in_out_array.data.ptr,
            count, dtype, root, stream)

    def reduce_scatter(
            self, in_array, out_array, count, op='sum', stream=None):
        """Performs a reduce scatter operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            count (int): Number of elements to send to each rank.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array, count)
        op = self._get_op(op, in_array.dtype.char)
        self._comm.reduceScatter(
            in_array.data.ptr, out_array.data.ptr, count, dtype, op, stream)

    def all_gather(self, in_array, out_array, count, stream=None):
        """Performs an all gather operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
            count (int): Number of elements to send to each rank.
            op (str): reduction operation, can be one of
                ('sum', 'prod', 'min' 'max'), arrays of complex type only
                support `'sum'`. Defaults to `'sum'`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array, count)
        self._comm.allGather(
            in_array.data.ptr, out_array.data.ptr, count, dtype, stream)

    def send(self, array, peer, stream=None):
        """Performs a send operation.

        Args:
            array (cupy.ndarray): array to be sent.
            peer (int): rank of the process `array` will be sent to.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._check_contiguous(array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(array)
        self._send(array, peer, dtype, count, stream)

    def _send(self, array, peer, dtype, count, stream=None):
        self._comm.send(array.data.ptr, count, dtype, peer, stream)

    def recv(self, out_array, peer, stream=None):
        """Performs a receive operation.

        Args:
            array (cupy.ndarray): array used to receive data.
            peer (int): rank of the process `array` will be received from.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(out_array)
        self._recv(out_array, peer, dtype, count, stream)

    def _recv(self, out_array, peer, dtype, count, stream=None):
        self._comm.recv(out_array.data.ptr, count, dtype, peer, stream)

    # TODO(ecastill) implement nccl missing calls combining the above ones
    # AlltoAll, AllGather, and similar MPI calls that can be easily implemented
    def send_recv(self, in_array, out_array, peer, stream=None):
        """Performs a send and receive operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array used to receive data.
            peer (int): rank of the process to send `in_array` and receive
                `out_array`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        idtype, icount = self._get_nccl_dtype_and_count(in_array)
        odtype, ocount = self._get_nccl_dtype_and_count(out_array)
        nccl.groupStart()
        self._send(in_array, peer, idtype, icount, stream)
        self._recv(out_array, peer, odtype, ocount, stream)
        nccl.groupEnd()

    def scatter(self, in_array, out_array, root=0, stream=None):
        """Performs a scatter operation.

        Args:
            in_array (cupy.ndarray): array to be sent. Its shape must be
                `(total_ranks, ...)`.
            out_array (cupy.ndarray): array where the result with be stored.
            root (int): rank that will send the `in_array` to other ranks.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        if in_array.shape[0] != self._n_devices:
            raise RuntimeError(
                f'scatter requires in_array to have {self._n_devices}'
                f'elements in its first dimension, found {in_array.shape}')
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        nccl.groupStart()
        if root == self.rank:
            for i in range(self._n_devices):
                array = in_array[i]
                idtype, icount = self._get_nccl_dtype_and_count(array)
                self._send(array, i, idtype, icount, stream)
        dtype, count = self._get_nccl_dtype_and_count(out_array)
        self._recv(out_array, root, dtype, count, stream)
        nccl.groupEnd()

    def gather(self, in_array, out_array, root=0, stream=None):
        """Performs a gather operation.

        Args:
            in_array (cupy.ndarray): array to be sent.
            out_array (cupy.ndarray): array where the result with be stored.
                Its shape must be `(total_ranks, ...)`.
            root (int): rank that will receive `in_array` from other ranks.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        # TODO(ecastill) out_array needs to have comm size in shape[0]
        if out_array.shape[0] != self._n_devices:
            raise RuntimeError(
                f'gather requires out_array to have {self._n_devices}'
                f'elements in its first dimension, found {out_array.shape}')
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        nccl.groupStart()
        if root == self.rank:
            for i in range(self._n_devices):
                array = out_array[i]
                odtype, ocount = self._get_nccl_dtype_and_count(array)
                self._recv(array, i, odtype, ocount, stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        self._send(in_array, root, dtype, count, stream)
        nccl.groupEnd()

    def all_to_all(self, in_array, out_array, stream=None):
        """Performs an all to all operation.

        Args:
            in_array (cupy.ndarray): array to be sent. Its shape must be
                `(total_ranks, ...)`.
            out_array (cupy.ndarray): array where the result with be stored.
                Its shape must be `(total_ranks, ...)`.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        # TODO(ecastill) out_array needs to have comm size in shape[0]
        if out_array.shape[0] != self._n_devices:
            raise RuntimeError(
                f'all_to_all requires in_array to have {self._n_devices}'
                f'elements in its first dimension, found {in_array.shape}')
        if out_array.shape[0] != self._n_devices:
            raise RuntimeError(
                f'all_to_all requires out_array to have {self._n_devices}'
                f'elements in its first dimension, found {out_array.shape}')
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        idtype, icount = self._get_nccl_dtype_and_count(in_array[0])
        odtype, ocount = self._get_nccl_dtype_and_count(out_array[0])
        # TODO check out dtypes are the same as in dtypes
        nccl.groupStart()
        for i in range(self._n_devices):
            self._send(in_array[i], i, idtype, icount, stream)
            self._recv(out_array[i], i, odtype, ocount, stream)
        nccl.groupEnd()

    def barrier(self):
        """Performs a barrier operation.

        The barrier is done in the cpu and is a explicit synchronization
        mechanism that halts the thread progression.
        """
        # implements a barrier CPU side
        # TODO allow multiple barriers to be executed
        self._store_proxy.barrier()
