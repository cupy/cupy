import pickle
import unittest

import cupy
from cupy import cuda
from cupy import testing


nccl_available = cuda.nccl.available

if nccl_available:
    nccl_version = cuda.nccl.get_version()
else:
    nccl_version = -1


@unittest.skipUnless(nccl_available, 'nccl is not installed')
class TestNCCL(unittest.TestCase):

    @testing.gpu
    def test_single_proc_ring(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        assert 0 == comm.rank_id()
        comm.destroy()

    @testing.gpu
    @unittest.skipUnless(nccl_version >= 2400, 'Using old NCCL')
    def test_abort(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        comm.abort()

    @testing.gpu
    @unittest.skipUnless(nccl_version >= 2400, 'Using old NCCL')
    def test_check_async_error(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        comm.check_async_error()
        comm.destroy()

    @testing.gpu
    def test_init_all(self):
        comms = cuda.nccl.NcclCommunicator.initAll(1)
        for i, comm in enumerate(comms):
            assert i == comms[i].rank_id()
        for i, comm in enumerate(comms):
            comms[i].destroy()

    @testing.gpu
    def test_single_proc_single_dev(self):
        comms = cuda.nccl.NcclCommunicator.initAll(1)
        cuda.nccl.groupStart()
        for comm in comms:
            cuda.Device(comm.device_id()).use()
            sendbuf = cupy.arange(10)
            recvbuf = cupy.zeros_like(sendbuf)
            comm.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, 10,
                           cuda.nccl.NCCL_INT64, cuda.nccl.NCCL_SUM,
                           cuda.Stream.null.ptr)
        cuda.nccl.groupEnd()
        assert cupy.allclose(sendbuf, recvbuf)

    @testing.gpu
    def test_comm_size(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        assert 1 == comm.size()

    @testing.multi_gpu(2)
    @unittest.skipUnless(nccl_version >= 2700, 'Using old NCCL')
    def test_send_recv(self):
        devs = [0, 1]
        comms = cuda.nccl.NcclCommunicator.initAll(devs)
        cuda.nccl.groupStart()
        for comm in comms:
            dev_id = comm.device_id()
            rank = comm.rank_id()
            assert rank == dev_id

            if rank == 0:
                with cuda.Device(dev_id):
                    sendbuf = cupy.arange(10, dtype=cupy.int64)
                    comm.send(sendbuf.data.ptr, 10, cuda.nccl.NCCL_INT64,
                              1, cuda.Stream.null.ptr)
            elif rank == 1:
                with cuda.Device(dev_id):
                    recvbuf = cupy.zeros(10, dtype=cupy.int64)
                    comm.recv(recvbuf.data.ptr, 10, cuda.nccl.NCCL_INT64,
                              0, cuda.Stream.null.ptr)
        cuda.nccl.groupEnd()

        # check result
        with cuda.Device(1):
            expected = cupy.arange(10, dtype=cupy.int64)
            assert (recvbuf == expected).all()


@unittest.skipUnless(nccl_available, 'nccl is not installed')
class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = cuda.nccl.NcclError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
