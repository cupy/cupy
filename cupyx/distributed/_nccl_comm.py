import numpy
import warnings

import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse


try:
    from mpi4py import MPI
    _mpi_available = True
except ImportError:
    _mpi_available = False


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
        use_mpi(bool, optional): switch between MPI and use the included TCP
            server for initialization & synchronization. Defaults to `False`.
    """

    def __init__(self, n_devices, rank,
                 host=_store._DEFAULT_HOST, port=_store._DEFAULT_PORT,
                 use_mpi=False):
        super().__init__(n_devices, rank, host, port)
        self._use_mpi = _mpi_available and use_mpi
        if self._use_mpi:
            self._init_with_mpi(n_devices, rank)
        else:
            self._init_with_tcp_store(n_devices, rank, host, port)

    def _init_with_mpi(self, n_devices, rank):
        # MPI is used only for management purposes
        # so the rank may be different than the one specified
        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_rank = self._mpi_comm.Get_rank()
        self._mpi_comm.Barrier()
        nccl_id = None
        if self._mpi_rank == 0:
            nccl_id = nccl.get_unique_id()
        nccl_id = self._mpi_comm.bcast(nccl_id, root=0)
        # Initialize devices
        self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)

    def _init_with_tcp_store(self, n_devices, rank, host, port):
        nccl_id = None
        if rank == 0:
            self._store.run(host, port)
            nccl_id = nccl.get_unique_id()
            # get_unique_id return negative values due to cython issues
            # with bytes && c strings. We shift them by 128 to
            # make them positive and send them as bytes to the proxy store
            shifted_nccl_id = bytes([b + 128 for b in nccl_id])
            self._store_proxy['nccl_id'] = shifted_nccl_id
            self._store_proxy.barrier()
        else:
            self._store_proxy.barrier()
            nccl_id = self._store_proxy['nccl_id']
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

    def _dispatch_arg_type(self, function, args):
        comm_class = _DenseNCCLCommunicator
        if sparse.issparse(args[0]):
            comm_class = _SparseNCCLCommunicator
        getattr(comm_class, function)(self, *args)

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
        self._dispatch_arg_type(
            'all_reduce', (in_array, out_array, op, stream))

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
        self._dispatch_arg_type(
            'reduce', (in_array, out_array, root, op, stream))

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
        self._dispatch_arg_type(
            'broadcast', (in_out_array, root, stream))

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
        self._dispatch_arg_type(
            'reduce_scatter', (in_array, out_array, count, op, stream))

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
        self._dispatch_arg_type(
            'all_gather', (in_array, out_array, count, stream))

    def send(self, array, peer, stream=None):
        """Performs a send operation.

        Args:
            array (cupy.ndarray): array to be sent.
            peer (int): rank of the process `array` will be sent to.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('send', (array, peer, stream))

    def recv(self, out_array, peer, stream=None):
        """Performs a receive operation.

        Args:
            array (cupy.ndarray): array used to receive data.
            peer (int): rank of the process `array` will be received from.
            stream (cupy.cuda.Stream, optional): if supported, stream to
                perform the communication.
        """
        self._dispatch_arg_type('recv', (out_array, peer, stream))

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
        self._dispatch_arg_type(
            'send_recv', (in_array, out_array, peer, stream))

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
        self._dispatch_arg_type(
            'scatter', (in_array, out_array, root, stream))

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
        self._dispatch_arg_type(
            'gather', (in_array, out_array, root, stream))

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
        self._dispatch_arg_type(
            'all_to_all', (in_array, out_array, stream))

    def barrier(self):
        """Performs a barrier operation.

        The barrier is done in the cpu and is a explicit synchronization
        mechanism that halts the thread progression.
        """
        # implements a barrier CPU side
        # TODO allow multiple barriers to be executed
        if self._use_mpi:
            self._mpi_comm.Barrier()
        else:
            self._store_proxy.barrier()


class _DenseNCCLCommunicator:

    @classmethod
    def all_reduce(cls, comm, in_array, out_array, op='sum', stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array)
        op = comm._get_op(op, in_array.dtype.char)
        comm._comm.allReduce(
            in_array.data.ptr, out_array.data.ptr, count, dtype, op, stream)

    @classmethod
    def reduce(cls, comm, in_array, out_array, root=0, op='sum', stream=None):
        comm._check_contiguous(in_array)
        if comm.rank == root:
            comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array)
        op = comm._get_op(op, in_array.dtype.char)
        comm._comm.reduce(
            in_array.data.ptr, out_array.data.ptr,
            count, dtype, op, root, stream)

    @classmethod
    def broadcast(cls, comm, in_out_array, root=0, stream=None):
        comm._check_contiguous(in_out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_out_array)
        comm._comm.broadcast(
            in_out_array.data.ptr, in_out_array.data.ptr,
            count, dtype, root, stream)

    @classmethod
    def reduce_scatter(
            cls, comm, in_array, out_array, count, op='sum', stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array, count)
        op = comm._get_op(op, in_array.dtype.char)
        comm._comm.reduceScatter(
            in_array.data.ptr, out_array.data.ptr, count, dtype, op, stream)

    @classmethod
    def all_gather(cls, comm, in_array, out_array, count, stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array, count)
        comm._comm.allGather(
            in_array.data.ptr, out_array.data.ptr, count, dtype, stream)

    @classmethod
    def send(cls, comm, array, peer, stream=None):
        comm._check_contiguous(array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(array)
        cls._send(comm, array, peer, dtype, count, stream)

    @classmethod
    def _send(cls, comm, array, peer, dtype, count, stream=None):
        comm._comm.send(array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def recv(cls, comm, out_array, peer, stream=None):
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        dtype, count = comm._get_nccl_dtype_and_count(out_array)
        cls._recv(comm, out_array, peer, dtype, count, stream)

    @classmethod
    def _recv(cls, comm, out_array, peer, dtype, count, stream=None):
        comm._comm.recv(out_array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def send_recv(cls, comm, in_array, out_array, peer, stream=None):
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        idtype, icount = comm._get_nccl_dtype_and_count(in_array)
        odtype, ocount = comm._get_nccl_dtype_and_count(out_array)
        nccl.groupStart()
        cls._send(comm, in_array, peer, idtype, icount, stream)
        cls._recv(comm, out_array, peer, odtype, ocount, stream)
        nccl.groupEnd()

    @classmethod
    def scatter(cls, comm, in_array, out_array, root=0, stream=None):
        if in_array.shape[0] != comm._n_devices:
            raise RuntimeError(
                f'scatter requires in_array to have {comm._n_devices}'
                f'elements in its first dimension, found {in_array.shape}')
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        nccl.groupStart()
        if root == comm.rank:
            for i in range(comm._n_devices):
                array = in_array[i]
                idtype, icount = comm._get_nccl_dtype_and_count(array)
                cls._send(comm, array, i, idtype, icount, stream)
        dtype, count = comm._get_nccl_dtype_and_count(out_array)
        cls._recv(comm, out_array, root, dtype, count, stream)
        nccl.groupEnd()

    @classmethod
    def gather(cls, comm, in_array, out_array, root=0, stream=None):
        # TODO(ecastill) out_array needs to have comm size in shape[0]
        if out_array.shape[0] != comm._n_devices:
            raise RuntimeError(
                f'gather requires out_array to have {comm._n_devices}'
                f'elements in its first dimension, found {out_array.shape}')
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        nccl.groupStart()
        if root == comm.rank:
            for i in range(comm._n_devices):
                array = out_array[i]
                odtype, ocount = comm._get_nccl_dtype_and_count(array)
                cls._recv(comm, array, i, odtype, ocount, stream)
        dtype, count = comm._get_nccl_dtype_and_count(in_array)
        cls._send(comm, in_array, root, dtype, count, stream)
        nccl.groupEnd()

    @classmethod
    def all_to_all(cls, comm, in_array, out_array, stream=None):
        # TODO(ecastill) out_array needs to have comm size in shape[0]
        if out_array.shape[0] != comm._n_devices:
            raise RuntimeError(
                f'all_to_all requires in_array to have {comm._n_devices}'
                f'elements in its first dimension, found {in_array.shape}')
        if out_array.shape[0] != comm._n_devices:
            raise RuntimeError(
                f'all_to_all requires out_array to have {comm._n_devices}'
                f'elements in its first dimension, found {out_array.shape}')
        comm._check_contiguous(in_array)
        comm._check_contiguous(out_array)
        stream = comm._get_stream(stream)
        idtype, icount = comm._get_nccl_dtype_and_count(in_array[0])
        odtype, ocount = comm._get_nccl_dtype_and_count(out_array[0])
        # TODO check out dtypes are the same as in dtypes
        nccl.groupStart()
        for i in range(comm._n_devices):
            cls._send(comm, in_array[i], i, idtype, icount, stream)
            cls._recv(comm, out_array[i], i, odtype, ocount, stream)
        nccl.groupEnd()


class _SparseNCCLCommunicator:

    @classmethod
    def _get_internal_arrays(cls, array):
        if sparse.isspmatrix_coo(array):
            array.sum_duplicates()  # set it to cannonical form
            return (array.data, array.row, array.col)
        elif sparse.isspmatrix_csr(array) or sparse.isspmatrix_csc(array):
            return (array.data, array.indptr, array.indices)
        raise TypeError('NCCL is not supported for this type of sparse matrix')

    @classmethod
    def _get_shape_and_sizes(cls, arrays, shape):
        # We get the elements from the array and send them
        # so that other process can create receiving arrays for it
        # However, this exchange synchronizes the gpus
        sizes_shape = shape + tuple((a.size for a in arrays))
        return sizes_shape

    @classmethod
    def _exchange_shape_and_sizes(
            cls, comm, peer, sizes_shape, method, stream):
        if comm._use_mpi:
            # Sends the metadata for the arrays using MPI
            if method == 'send':
                sizes_shape = numpy.array(sizes_shape, dtype='q')
                comm._mpi_comm.Send(sizes_shape, dest=peer, tag=1)
                return None
            if method == 'recv':
                # Shape is a tuple of two elements, and a single scalar per
                # each array (5)
                sizes_shape = numpy.empty(5, dtype='q')
                comm._mpi_comm.Recv(sizes_shape, source=peer, tag=1)
                return sizes_shape
            else:
                raise RuntimeError('Unsupported method')
        else:
            warnings.warn(
                'Using NCCL for transferring sparse arrays metadata. This'
                ' will cause device synchronization and a huge performance'
                ' degradation. Please install MPI and `mpi4py` in order to'
                ' avoid this issue.'
            )
            if method == 'send':
                sizes_shape = cupy.array(sizes_shape, dtype='q')
                cls._send(
                    comm, sizes_shape, peer, sizes_shape.dtype, 5, stream)
                return None
            if method == 'recv':
                # Shape is a tuple of two elements, and a single scalar per
                # each array (5)
                sizes_shape = cupy.empty(5, dtype='q')
                cls._recv(
                    comm, sizes_shape, peer, sizes_shape.dtype, 5, stream)
                return cupy.asnumpy(sizes_shape)
            else:
                raise RuntimeError('Unsupported method')

    def _assign_arrays(matrix, arrays, shape):
        if sparse.isspmatrix_coo(matrix):
            matrix.data = arrays[0]
            matrix.row = arrays[1]
            matrix.col = arrays[2]
            matrix._shape = tuple(shape)
        elif sparse.isspmatrix_csr(matrix) or sparse.isspmatrix_csc(matrix):
            matrix.data = arrays[0]
            matrix.indptr = arrays[1]
            matrix.indices = arrays[2]
            matrix._shape = tuple(shape)
        else:
            raise TypeError(
                'NCCL is not supported for this type of sparse matrix')

    @classmethod
    def all_reduce(cls, comm, in_array, out_array, op='sum', stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def reduce(cls, comm, in_array, out_array, root=0, op='sum', stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def broadcast(cls, comm, in_out_array, root=0, stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def reduce_scatter(
            cls, comm, in_array, out_array, count, op='sum', stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def all_gather(cls, comm, in_array, out_array, count, stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def send(cls, comm, array, peer, stream=None):
        arrays = cls._get_internal_arrays(array)
        shape_and_sizes = cls._get_shape_and_sizes(arrays, array.shape)
        cls._exchange_shape_and_sizes(
            comm, peer, shape_and_sizes, 'send', stream)
        # Naive approach, we send each of the subarrays one by one
        for a in arrays:
            cls._send(comm, a, peer, a.dtype, a.size, stream)

    @classmethod
    def _send(cls, comm, array, peer, dtype, count, stream=None):
        dtype = array.dtype.char
        if dtype not in _nccl_dtypes:
            raise TypeError(f'Unknown dtype {array.dtype} for NCCL')
        dtype, count = comm._get_nccl_dtype_and_count(array)
        stream = comm._get_stream(stream)
        comm._comm.send(array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def recv(cls, comm, out_array, peer, stream=None):
        shape_and_sizes = cls._exchange_shape_and_sizes(
            comm, peer, (), 'recv', stream)
        # Change the array sizes in out_array to match the sent ones
        # Receive the three arrays
        # TODO(ecastill) dtype is not correct, it must match the internal
        # sparse matrix arrays dtype
        arrays = cls._get_internal_arrays(out_array)
        shape = tuple(shape_and_sizes[0:2])
        sizes = shape_and_sizes[2:]
        # TODO(use the out_array datatypes)
        arrs = [cupy.empty(s, dtype=a.dtype) for s, a in zip(sizes, arrays)]
        for a in arrs:
            cls._recv(comm, a, peer, a.dtype, a.size, stream)
        # Create a sparse matrix from the received arrays
        cls._assign_arrays(out_array, arrs, shape)

    @classmethod
    def _recv(cls, comm, out_array, peer, dtype, count, stream=None):
        dtype = dtype.char
        if dtype not in _nccl_dtypes:
            raise TypeError(f'Unknown dtype {out_array.dtype} for NCCL')
        dtype, count = comm._get_nccl_dtype_and_count(out_array)
        stream = comm._get_stream(stream)
        comm._comm.recv(out_array.data.ptr, count, dtype, peer, stream)

    @classmethod
    def send_recv(cls, comm, in_array, out_array, peer, stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def scatter(cls, comm, in_array, out_array, root=0, stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def gather(cls, comm, in_array, out_array, root=0, stream=None):
        raise RuntimeError('Method not supported for sparse matrices')

    @classmethod
    def all_to_all(cls, comm, in_array, out_array, stream=None):
        raise RuntimeError('Method not supported for sparse matrices')
