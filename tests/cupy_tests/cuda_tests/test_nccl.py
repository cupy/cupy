from __future__ import annotations

import pickle
import pytest
import unittest

import cupy
from cupy import cuda
from cupy.cuda import nccl
from cupy import testing


nccl_available = nccl.available

if nccl_available:
    nccl_version_code = nccl.get_version()
else:
    nccl_version_code = -1


def nccl_version(x, y, z):
    return (
        (x * 1000 + y * 100 + z) if (x <= 2 and y <=
                                     8) else (x * 10000 + y * 100 + z)
    )


@unittest.skipUnless(nccl_available, 'nccl is not installed')
class TestNCCL(unittest.TestCase):

    def test_single_proc_ring(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        assert 0 == comm.rank_id()
        comm.destroy()

    @unittest.skipUnless(nccl_version_code >= nccl_version(2, 4, 0),
                         'Using old NCCL')
    def test_abort(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        comm.abort()

    @unittest.skipUnless(nccl_version_code >= nccl_version(2, 4, 0),
                         'Using old NCCL')
    def test_check_async_error(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        comm.check_async_error()
        comm.destroy()

    def test_init_all(self):
        comms = nccl.NcclCommunicator.initAll(1)
        for i, comm in enumerate(comms):
            assert i == comms[i].rank_id()
        for i, comm in enumerate(comms):
            comms[i].destroy()

    def test_single_proc_single_dev(self):
        comms = nccl.NcclCommunicator.initAll(1)
        nccl.groupStart()
        for comm in comms:
            cuda.Device(comm.device_id()).use()
            sendbuf = cupy.arange(10)
            recvbuf = cupy.zeros_like(sendbuf)
            comm.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, 10,
                           nccl.NCCL_INT64, nccl.NCCL_SUM,
                           cuda.Stream.null.ptr)
        nccl.groupEnd()
        assert cupy.allclose(sendbuf, recvbuf)

    def test_comm_size(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        assert 1 == comm.size()

    def test_nccl_config(self):
        config = nccl.NcclConfig()
        assert config.split_share == 0
        config = nccl.NcclConfig(split_share=1)
        assert config.split_share == 1

    @testing.multi_gpu(2)
    @unittest.skipUnless(nccl_version_code >= nccl_version(2, 7, 0),
                         'Using old NCCL')
    def test_send_recv(self):
        devs = [0, 1]
        comms = nccl.NcclCommunicator.initAll(devs)
        nccl.groupStart()
        for comm in comms:
            dev_id = comm.device_id()
            rank = comm.rank_id()
            assert rank == dev_id

            if rank == 0:
                with cuda.Device(dev_id):
                    sendbuf = cupy.arange(10, dtype=cupy.int64)
                    comm.send(sendbuf.data.ptr, 10, nccl.NCCL_INT64,
                              1, cuda.Stream.null.ptr)
            elif rank == 1:
                with cuda.Device(dev_id):
                    recvbuf = cupy.zeros(10, dtype=cupy.int64)
                    comm.recv(recvbuf.data.ptr, 10, nccl.NCCL_INT64,
                              0, cuda.Stream.null.ptr)
        nccl.groupEnd()

        # check result
        with cuda.Device(1):
            expected = cupy.arange(10, dtype=cupy.int64)
            assert (recvbuf == expected).all()


@unittest.skipUnless(nccl_available, 'nccl is not installed')
class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = nccl.NcclError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


@pytest.mark.skipif(
    nccl_version_code < nccl_version(2, 18, 1), reason='Using old NCCL'
)
class TestCommSplit:
    # use generate_config_params to avoid NcclConfig construction failure
    # on old NCCL versions, because parametrize happens before skipif check
    def generate_config_params():
        return (
            [nccl.NcclConfig(), nccl.NcclConfig(split_share=1), None]
            if nccl_version_code >= nccl_version(2, 18, 1)
            else []
        )

    @pytest.mark.parametrize("config", generate_config_params())
    def test_comm_split(self, config):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0, config)
        new_comm = comm.commSplit(color=0, key=0, config=config)
        assert new_comm is not None
        assert 1 == new_comm.size()
        new_comm.destroy()
        comm.destroy()

    def test_split_no_color(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        new_comm = comm.commSplit(color=-1, key=0)
        assert new_comm is None
        comm.destroy()

    @testing.multi_gpu(4)
    def test_4_processes_2_subcomm(self):
        import multiprocessing as mp
        id = nccl.get_unique_id()

        def worker(rank, world_size):
            cuda.Device(rank).use()
            global_comm = nccl.NcclCommunicator(
                world_size, id, rank, nccl.NcclConfig(split_share=1)
            )
            color = 0 if rank < 2 else 1
            sub_comm = global_comm.commSplit(color, rank)
            assert sub_comm is not None
            send_buf = cupy.array([rank], dtype=cupy.int8)
            recv_buf = cupy.empty_like(send_buf)
            sub_comm.allReduce(
                send_buf.data.ptr, recv_buf.data.ptr, send_buf.size,
                nccl.NCCL_INT8, nccl.NCCL_SUM, cuda.Stream.null.ptr,
            )
            cuda.Stream.null.synchronize()
            expected = (
                cupy.array([0 + 1], dtype=cupy.int8)
                if color == 0
                else cupy.array([2 + 3], dtype=cupy.int8)
            )
            testing.assert_array_equal(recv_buf, expected)
            sub_comm.destroy()
            global_comm.destroy()

        procs = [mp.Process(target=worker, args=(i, 4,)) for i in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
