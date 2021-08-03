import cupy
from cupy.cuda import nccl


_nccl_dtypes = {cupy.int8: nccl.NCCL_INT8,
                cupy.uint8: nccl.NCCL_UINT8,
                cupy.int32: nccl.NCCL_INT32,
                cupy.uint32: nccl.NCCL_UINT32,
                cupy.int64: nccl.NCCL_INT64,
                cupy.uint64: nccl.NCCL_UINT64,
                cupy.float16: nccl.NCCL_FLOAT16,
                cupy.float32: nccl.NCCL_FLOAT32,
                cupy.float64: nccl.NCCL_FLOAT64,
                # Size of array will be doubled
                cupy.complex64: nccl.NCCL_FLOAT32,
                cupy.complex128: nccl.NCCL_FLOAT64}


_nccl_ops = {'sum': nccl.NCCL_SUM,
             'prod': nccl.NCCL_PROD,
             'max': nccl.NCCL_MAX,
             'min': nccl.NCCL_MIN}


class NCCLCommunicator:
    def __init__(self, n_devices, comm_id, rank):
        self._n_devices = n_devices
        self._comm_id = comm_id
        self.rank = rank
        self._comm = nccl.NcclCommunicator(n_devices, comm_id, rank)

    def _check_contiguous(self, array):
        if not array.flags.c_contiguous or array.flags.f_contiguous:
            raise RuntimeError('NCCL requires arrays to be contiguous')

    def _get_nccl_dtype_and_count(self, array):
        dtype = array.dtype
        if dtype not in _nccl_dtypes:
            raise TypeError(f'Unknown dtype {dtype} for NCCL')
        nccl_dtype = _nccl_dtypes[dtype]
        if dtype.kind == 'c':
            return nccl_dtype, 2 * array.size
        return nccl_dtype, array.size

    def _get_stream(self, stream):
        if stream is None:
            stream = cupy.cuda.stream.get_current_stream()
        return stream.ptr

    def _get_op(self, op):
        if op not in _nccl_ops:
            raise RuntimeError(f'Unknown op {op} for NCCL')
        return _nccl_ops[op]

    def all_reduce(self, in_array, out_array, op='sum', stream=None):
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        op = self._get_op(op)
        self._comm.allReduce(
            in_array.data.ptr, out_array.data_ptr, count, dtype, op, stream)

    def reduce(self, in_array, out_array, root=0, op='sum', stream=None):
        self._check_contiguous(in_array)
        if self.rank == root:
            self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        op = self._get_op(op)
        self._comm.reduce(
            in_array.data.ptr, out_array.data.ptr,
            count, dtype, op, root, stream)

    def broadcast(self, in_array, root=0, op='sum', stream=None):
        # in_array for root !=0 will be used as output
        self._check_contiguous(in_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        self._comm.broadcast(
            in_array.data.ptr, in_array.data.ptr, count, dtype, root, stream)

    def reduce_scatter(self, in_array, out_array, op='sum', stream=None):
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        op = self._get_op(op)
        self._comm.reduceScatter(
            in_array.data.ptr, out_array.data_ptr, count, dtype, op, stream)

    def all_gather(self, in_array, out_array, stream=None):
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        self._comm.allGather(
            in_array.data.ptr, out_array.data_ptr, count, dtype, stream)

    def send(self, array, peer, stream=None):
        self._check_contiguous(array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(array)
        self._comm.send(array.data.ptr, count, dtype, peer, stream)

    def recv(self, out_array, peer, stream=None):
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(out_array)
        self._comm.recv(out_array.data.ptr, count, dtype, peer, stream)

    # TODO(ecastill) implement nccl missing calls combining the above ones
    # AlltoAll, AllGather, and similar MPI calls that can be easily implemented
