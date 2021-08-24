import os

import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store

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


class NCCL:
    def __init__(self, n_devices, rank):
        self._n_devices = n_devices
        self.rank = rank
        host = int(os.environ.get('CUPYX_DISTRIBUTED_HOST', '127.0.0.1'))
        port = int(os.environ.get('CUPYX_DISTRIBUTED_PORT', '12345'))
        self._store_proxy = _store.TCPStoreProxy(host, port)
        if rank == 0:
            self._store = _store.TCPStore()
            self._store.run(host, port)
            nccl_id = nccl.get_unique_id()
            self._store_proxy['nccl_id'] = nccl_id
        else:
            nccl_id = self._store_proxy['nccl_id']

        # Initialize comm with the above id
        self._comm = nccl.NcclCommunicator(n_devices, nccl_id, rank)

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
        self._send(array, peer, dtype, count, stream)

    def _send(self, array, peer, dtype, count, stream=None):
        self._comm.send(array.data.ptr, dtype, count, peer, stream)

    def recv(self, out_array, peer, stream=None):
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        dtype, count = self._get_nccl_dtype_and_count(out_array)
        self._recv(out_array, peer, dtype, count, stream)

    def _recv(self, out_array, peer, dtype, count, stream=None):
        self._comm.recv(out_array.data.ptr, count, dtype, peer, stream)

    # TODO(ecastill) implement nccl missing calls combining the above ones
    # AlltoAll, AllGather, and similar MPI calls that can be easily implemented
    def send_recv(self, in_array, out_array, peer, stream=None):
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        idtype, icount = self._get_nccl_dtype_and_count(in_array)
        odtype, ocount = self._get_nccl_dtype_and_count(out_array)
        nccl.ncclGroupStart()
        self._send(in_array, peer, idtype, icount, stream)
        self._recv(out_array, peer, odtype, ocount, stream)
        nccl.ncclGroupEnd()

    def scatter(self, in_array, out_array, root=0, stream=None):
        if in_array.shape[0] != self._n_devices:
            raise RuntimeError(
                f'scatter requires in_array to have {self._n_devices}'
                f'elements in its first dimension, found {in_array.shape}')
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        nccl.ncclGroupStart()
        if root == self.rank:
            for i in range(self._n_devices):
                array = in_array[i]
                idtype, icount = self._get_nccl_dtype_and_count(out_array)
                self._send(array, i, idtype, icount, stream)
        dtype, count = self._get_nccl_dtype_and_count(out_array)
        self._recv(out_array, root, dtype, count, stream)
        nccl.ncclGroupEnd()

    def gather(self, in_array, out_array, root=0, stream=None):
        # TODO(ecastill) out_array needs to have comm size in shape[0]
        if out_array.shape[0] != self._n_devices:
            raise RuntimeError(
                f'gather requires out_array to have {self._n_devices}'
                f'elements in its first dimension, found {out_array.shape}')
        self._check_contiguous(in_array)
        self._check_contiguous(out_array)
        stream = self._get_stream(stream)
        nccl.ncclGroupStart()
        if root == self.rank:
            for i in range(self._n_devices):
                array = out_array[i]
                odtype, ocount = self._get_nccl_dtype_and_count(out_array)
                self._recv(array, i, odtype, ocount, stream)
        dtype, count = self._get_nccl_dtype_and_count(in_array)
        self._send(in_array, root, dtype, count, stream)
        nccl.ncclGroupEnd()

    def all_to_all(self, in_array, out_array, stream=None):
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
        nccl.ncclGroupStart()
        for i in range(self._n_devices):
            self._send(in_array[i], i, idtype, icount, stream)
            self._recv(out_array[i], i, odtype, ocount, stream)
        nccl.ncclGroupEnd()

    def barrier(self):
        # implements a barrier CPU side
        # TODO allow multiple barriers to be executed
        self._store_proxy.wait_until('barrier', 0)
