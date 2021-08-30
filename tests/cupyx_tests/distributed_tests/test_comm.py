import multiprocessing
import unittest

import cupy
from cupy import cuda
from cupy.cuda import nccl
from cupy import testing

from cupyx.distributed import NCCLBackend

nccl_available = nccl.available

if nccl_available:
    nccl_version = nccl.get_version()
else:
    nccl_version = -1

N_WORKERS = 2


@unittest.skipUnless(nccl_available, 'nccl is not installed')
class TestNCCLBackend:
    def _launch_workers(self, n_workers, func, args=()):
        processes = []
        # TODO catch exceptions
        for rank in range(n_workers):
            p = multiprocessing.Process(
                target=func, args=(rank, n_workers) + args)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # TODO for all dtypes
    def test_broadcast(self):
        def run_broadcast(rank, n_workers, root):
            dev = cuda.Device(rank)
            dev.use()
            comm = NCCLBackend(n_workers, rank)
            expected = cupy.arange(2 * 3 * 4, dtype='f').reshape((2, 3, 4))
            if rank == root:
                in_array = expected
            else:
                in_array = cupy.zeros((2, 3, 4), dtype='f')

            comm.broadcast(in_array, root)
            testing.assert_allclose(in_array, expected)

        self._launch_workers(N_WORKERS, run_broadcast, (0,))
        self._launch_workers(N_WORKERS, run_broadcast, (1,))

    def test_reduce(self):
        def run_reduce(rank, n_workers, root):
            dev = cuda.Device(rank)
            dev.use()
            comm = NCCLBackend(n_workers, rank)
            in_array = cupy.arange(2 * 3 * 4, dtype='f').reshape(2, 3, 4)
            out_array = cupy.zeros((2, 3, 4), dtype='f')

            comm.reduce(in_array, out_array, root)
            if rank == root:
                testing.assert_allclose(out_array, 2 * in_array)

        self._launch_workers(N_WORKERS, run_reduce, (0,))
        self._launch_workers(N_WORKERS, run_reduce, (1,))

    def test_all_reduce(self):
        def run_all_reduce(rank, n_workers):
            dev = cuda.Device(rank)
            dev.use()
            comm = NCCLBackend(n_workers, rank)
            in_array = cupy.arange(2 * 3 * 4, dtype='f').reshape(2, 3, 4)
            out_array = cupy.zeros((2, 3, 4), dtype='f')

            comm.all_reduce(in_array, out_array)
            testing.assert_allclose(out_array, 2 * in_array)

        self._launch_workers(N_WORKERS, run_all_reduce)

    def test_reduce_scatter(self):
        pass

    def test_all_gather(self):
        pass

    def test_send_rcv(self):
        pass

    def test_scatter(self):
        pass

    def test_gather(self):
        pass

    def test_all_to_all(self):
        pass
