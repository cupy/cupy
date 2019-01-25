import unittest

from cupy.cuda import nccl
from cupy.testing import attr


class TestNCCL(unittest.TestCase):

    @attr.gpu
    def test_single_proc_ring(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        assert 0 == comm.rank_id()
        comm.destroy()

    @attr.gpu
    @unittest.skipUnless(nccl.get_version() >= 2400, "Using old NCCL")
    def test_abort(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        comm.abort()

    @attr.gpu
    @unittest.skipUnless(nccl.get_version() >= 2400, "Using old NCCL")
    def test_check_async_error(self):
        id = nccl.get_unique_id()
        comm = nccl.NcclCommunicator(1, id, 0)
        comm.check_async_error()
        comm.destroy()
