import unittest

from cupy import cuda
from cupy.testing import attr


@unittest.skipUnless(cuda.nccl_enabled, 'nccl is not installed')
class TestNCCL(unittest.TestCase):

    @attr.gpu
    def test_single_proc_ring(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        assert 0 == comm.rank_id()
        comm.destroy()

    @attr.gpu
    @unittest.skipUnless(cuda.nccl.get_version() >= 2400, "Using old NCCL")
    def test_abort(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        comm.abort()

    @attr.gpu
    @unittest.skipUnless(cuda.nccl.get_version() >= 2400, "Using old NCCL")
    def test_check_async_error(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        comm.check_async_error()
        comm.destroy()
