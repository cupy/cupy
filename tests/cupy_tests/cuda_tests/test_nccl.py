import pickle
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
    @unittest.skipUnless(cuda.nccl_enabled and
                         cuda.nccl.get_version() >= 2400, 'Using old NCCL')
    def test_abort(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        comm.abort()

    @attr.gpu
    @unittest.skipUnless(cuda.nccl_enabled and
                         cuda.nccl.get_version() >= 2400, 'Using old NCCL')
    def test_check_async_error(self):
        id = cuda.nccl.get_unique_id()
        comm = cuda.nccl.NcclCommunicator(1, id, 0)
        comm.check_async_error()
        comm.destroy()


@unittest.skipUnless(cuda.nccl_enabled, 'nccl is not installed')
class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = cuda.nccl.NcclError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
