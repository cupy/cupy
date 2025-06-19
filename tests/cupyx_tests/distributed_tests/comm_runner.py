import sys
import time
import warnings

import cupy
from cupy import cuda
from cupy.cuda import nccl
from cupy import testing

from cupyx.distributed import init_process_group
from cupyx.distributed._nccl_comm import NCCLBackend
from cupyx.distributed._store import ExceptionAwareProcess
from cupyx.scipy import sparse


nccl_available = nccl.available


N_WORKERS = 2


def _launch_workers(func, args=(), n_workers=N_WORKERS):
    processes = []
    # TODO catch exceptions
    for rank in range(n_workers):
        p = ExceptionAwareProcess(
            target=func,
            args=(rank,) + args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def broadcast(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_broadcast(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        expected = cupy.arange(2 * 3 * 4, dtype=dtype).reshape((2, 3, 4))
        if rank == root:
            in_array = expected
        else:
            in_array = cupy.zeros((2, 3, 4), dtype=dtype)
        comm.broadcast(in_array, root)
        testing.assert_allclose(in_array, expected)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_broadcast(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_broadcast(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_broadcast, (0, dtype))
        _launch_workers(run_broadcast, (1, dtype))


def reduce(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_reduce(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = cupy.arange(2 * 3 * 4, dtype='f').reshape(2, 3, 4)
        out_array = cupy.zeros((2, 3, 4), dtype='f')
        comm.reduce(in_array, out_array, root)
        if rank == root:
            testing.assert_allclose(out_array, 2 * in_array)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_reduce(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_reduce(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_reduce, (0, dtype))
        _launch_workers(run_reduce, (1, dtype))


def all_reduce(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_all_reduce(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = cupy.arange(2 * 3 * 4, dtype='f').reshape(2, 3, 4)
        out_array = cupy.zeros((2, 3, 4), dtype='f')

        comm.all_reduce(in_array, out_array)
        testing.assert_allclose(out_array, 2 * in_array)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_all_reduce(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_all_reduce, (dtype,))


def reduce_scatter(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_reduce_scatter(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = 1 + cupy.arange(
            N_WORKERS * 10, dtype='f').reshape(N_WORKERS, 10)
        out_array = cupy.zeros((10,), dtype='f')

        comm.reduce_scatter(in_array, out_array, 10)
        testing.assert_allclose(out_array, 2 * in_array[rank])

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_reduce_scatter(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_reduce_scatter, (dtype,))


def all_gather(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_all_gather(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = (rank + 1) * cupy.arange(
            N_WORKERS * 10, dtype='f').reshape(N_WORKERS, 10)
        out_array = cupy.zeros((N_WORKERS, 10), dtype='f')
        comm.all_gather(in_array, out_array, 10)
        expected = 1 + cupy.arange(N_WORKERS).reshape(N_WORKERS, 1)
        expected = expected * cupy.broadcast_to(
            cupy.arange(10, dtype='f'), (N_WORKERS, 10))
        testing.assert_allclose(out_array, expected)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_all_gather(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_all_gather, (dtype,))


def send_and_recv(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_send_and_recv(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = cupy.arange(10, dtype='f')
        out_array = cupy.zeros((10,), dtype='f')
        if rank == 0:
            comm.send(in_array, 1)
        else:
            comm.recv(out_array, 0)
            testing.assert_allclose(out_array, in_array)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_send_and_recv(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_send_and_recv, (dtype,))


def send_recv(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_send_recv(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = cupy.arange(10, dtype='f')
        for i in range(N_WORKERS):
            out_array = cupy.zeros((10,), dtype='f')
            comm.send_recv(in_array, out_array, i)
            testing.assert_allclose(out_array, in_array)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_send_recv(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_send_recv, (dtype,))


def scatter(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_scatter(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = 1 + cupy.arange(
            N_WORKERS * 10, dtype='f').reshape(N_WORKERS, 10)
        out_array = cupy.zeros((10,), dtype='f')

        comm.scatter(in_array, out_array, root)
        if rank > 0:
            testing.assert_allclose(out_array, in_array[rank])

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_scatter(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_scatter(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_scatter, (0, dtype))
        _launch_workers(run_scatter, (1, dtype))


def gather(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_gather(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = (rank + 1) * cupy.arange(10, dtype='f')
        out_array = cupy.zeros((N_WORKERS, 10), dtype='f')
        comm.gather(in_array, out_array, root)
        if rank == root:
            expected = 1 + cupy.arange(N_WORKERS).reshape(N_WORKERS, 1)
            expected = expected * cupy.broadcast_to(
                cupy.arange(10, dtype='f'), (N_WORKERS, 10))
            testing.assert_allclose(out_array, expected)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_gather(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_gather(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_gather, (0, dtype))
        _launch_workers(run_gather, (1, dtype))


def all_to_all(dtype, use_mpi=False):
    if dtype in 'hH':
        return  # nccl does not support int16

    def run_all_to_all(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = cupy.arange(
            N_WORKERS * 10, dtype='f').reshape(N_WORKERS, 10)
        out_array = cupy.zeros((N_WORKERS, 10), dtype='f')
        comm.all_to_all(in_array, out_array)
        expected = (10 * rank) + cupy.broadcast_to(
            cupy.arange(10, dtype='f'), (N_WORKERS, 10))
        testing.assert_allclose(out_array, expected)

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_all_to_all(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_all_to_all, (dtype,))


def barrier(use_mpi=False):
    def run_barrier(rank, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        comm.barrier()
        before = time.time()
        if rank == 0:
            time.sleep(2)
        comm.barrier()
        after = time.time()
        assert int(after - before) == 2

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_barrier(MPI.COMM_WORLD.Get_rank(), True)
    else:
        _launch_workers(run_barrier)


def init(use_mpi=False):
    def run_init(rank, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = init_process_group(N_WORKERS, rank, use_mpi=use_mpi)
        # Do a simple call to verify we got a valid comm
        in_array = cupy.zeros(1)
        if rank == 0:
            in_array = in_array + 1
        comm.broadcast(in_array, 0)
        testing.assert_allclose(in_array, cupy.ones(1))

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_init(MPI.COMM_WORLD.Get_rank(), dtype, True)
        run_init(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_init)


def _make_sparse(dtype):
    data = cupy.array([1, 3, 2, 5, 1, 1], dtype)
    indices = cupy.array([0, 3, 1, 3, 0, 2], 'i')
    indptr = cupy.array([0, 2, 3, 4, 6], 'i')
    return sparse.csr_matrix((data, indices, indptr), shape=(4, 4))


def _make_sparse_empty(dtype):
    data = cupy.array([0], dtype)
    indices = cupy.array([0], 'i')
    indptr = cupy.array([0], 'i')
    return sparse.csr_matrix((data, indices, indptr), shape=(0, 0))


def sparse_send_and_recv(dtype, use_mpi=False):
    def run_send_and_recv(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = _make_sparse(dtype)
        out_array = _make_sparse_empty(dtype)
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        if rank == 0:
            comm.send(in_array, 1)
        else:
            comm.recv(out_array, 0)
            testing.assert_allclose(out_array.todense(), in_array.todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_send_and_recv(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_send_and_recv, (dtype,))


def sparse_send_recv(dtype, use_mpi=False):
    def run_send_recv(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = _make_sparse(dtype)
        out_array = _make_sparse_empty(dtype)
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        if rank == 0:
            comm.send_recv(in_array, out_array, 1)
        else:
            comm.send_recv(in_array, out_array, 0)
            testing.assert_allclose(out_array.todense(), in_array.todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_send_recv(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_send_recv, (dtype,))


def sparse_broadcast(dtype, use_mpi=False):

    def run_broadcast(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        expected = _make_sparse(dtype)
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        if rank == root:
            in_array = expected
        else:
            in_array = _make_sparse_empty(dtype)
        comm.broadcast(in_array, root)
        testing.assert_allclose(in_array.todense(), expected.todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_broadcast(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_broadcast(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_broadcast, (0, dtype,))
        _launch_workers(run_broadcast, (1, dtype,))


def sparse_reduce(dtype, use_mpi=False):

    def run_reduce(rank, root, dtype, op, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = _make_sparse(dtype)
        out_array = _make_sparse_empty(dtype)
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        comm.reduce(in_array, out_array, root, op)
        if rank == root:
            if op == 'sum':
                testing.assert_allclose(
                    out_array.todense(), 2 * in_array.todense())
            else:
                testing.assert_allclose(
                    out_array.todense(),
                    cupy.matmul(in_array.todense(), in_array.todense()))

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_reduce(MPI.COMM_WORLD.Get_rank(), 0, dtype, 'sum', True)
        run_reduce(MPI.COMM_WORLD.Get_rank(), 1, dtype, 'prod', True)
    else:
        _launch_workers(run_reduce, (0, dtype, 'sum'))
        _launch_workers(run_reduce, (1, dtype, 'prod'))


def sparse_all_reduce(dtype, use_mpi=False):

    def run_all_reduce(rank, dtype, op, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = _make_sparse(dtype)
        out_array = _make_sparse_empty(dtype)
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        comm.all_reduce(in_array, out_array, op)
        if op == 'sum':
            testing.assert_allclose(
                out_array.todense(), 2 * in_array.todense())
        else:
            testing.assert_allclose(
                out_array.todense(),
                cupy.matmul(in_array.todense(), in_array.todense()))

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_all_reduce(MPI.COMM_WORLD.Get_rank(), dtype, 'sum', True)
        run_all_reduce(MPI.COMM_WORLD.Get_rank(), dtype, 'prod', True)
    else:
        _launch_workers(run_all_reduce, (dtype, 'sum'))
        _launch_workers(run_all_reduce, (dtype, 'prod'))


def sparse_scatter(dtype, use_mpi=False):

    def run_scatter(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_arrays = [_make_sparse(dtype), 2*_make_sparse(dtype)]
        out_array = _make_sparse_empty(dtype)
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        comm.scatter(in_arrays, out_array, root)
        testing.assert_allclose(
            out_array.todense(), in_arrays[rank].todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_scatter(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_scatter(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_scatter, (0, dtype))
        _launch_workers(run_scatter, (1, dtype))


def sparse_gather(dtype, use_mpi=False):

    def run_gather(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = (rank + 1) * _make_sparse(dtype)
        out_arrays = []
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        comm.gather(in_array, out_arrays, root)
        if rank == root:
            expected = [_make_sparse(dtype), 2 * _make_sparse(dtype)]
            testing.assert_allclose(
                out_arrays[0].todense(), expected[0].todense())
            testing.assert_allclose(
                out_arrays[1].todense(), expected[1].todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_gather(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_gather(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_gather, (0, dtype))
        _launch_workers(run_gather, (1, dtype))


def sparse_all_gather(dtype, use_mpi=False):

    def run_all_gather(rank, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_array = (rank + 1) * _make_sparse(dtype)
        out_arrays = []
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        comm.all_gather(in_array, out_arrays, 0)
        expected = [_make_sparse(dtype), 2 * _make_sparse(dtype)]
        testing.assert_allclose(
            out_arrays[0].todense(), expected[0].todense())
        testing.assert_allclose(
            out_arrays[1].todense(), expected[1].todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_all_gather(MPI.COMM_WORLD.Get_rank(), dtype, True)
        run_all_gather(MPI.COMM_WORLD.Get_rank(), dtype, True)
    else:
        _launch_workers(run_all_gather, (dtype,))
        _launch_workers(run_all_gather, (dtype,))


def sparse_all_to_all(dtype, use_mpi=False):

    def run_all_to_all(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_arrays = [_make_sparse(dtype), 2 * _make_sparse(dtype)]
        out_array = []
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        comm.all_to_all(in_arrays, out_array)
        testing.assert_allclose(
            out_array[0].todense(), (rank + 1) * in_arrays[0].todense())
        testing.assert_allclose(
            out_array[1].todense(), (rank + 1) * in_arrays[0].todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_all_to_all(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_all_to_all(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_all_to_all, (0, dtype))
        _launch_workers(run_all_to_all, (1, dtype))


def sparse_reduce_scatter(dtype, use_mpi=False):

    def run_reduce_scatter(rank, root, dtype, use_mpi=False):
        dev = cuda.Device(rank)
        dev.use()
        comm = NCCLBackend(N_WORKERS, rank, use_mpi=use_mpi)
        in_arrays = [(rank + 1) * _make_sparse(dtype),
                     (rank + 2) * _make_sparse(dtype)]
        out_array = _make_sparse_empty(dtype)
        warnings.filterwarnings(
            'ignore', '.*transferring sparse.*', UserWarning)
        comm.reduce_scatter(in_arrays, out_array, 2)
        target = ((rank + 1) * _make_sparse(dtype)
                  + (rank + 2) * _make_sparse(dtype))
        testing.assert_allclose(
            out_array.todense(), target.todense())

    if use_mpi:
        from mpi4py import MPI
        # This process was run with mpiexec
        run_reduce_scatter(MPI.COMM_WORLD.Get_rank(), 0, dtype, True)
        run_reduce_scatter(MPI.COMM_WORLD.Get_rank(), 1, dtype, True)
    else:
        _launch_workers(run_reduce_scatter, (0, dtype))
        _launch_workers(run_reduce_scatter, (1, dtype))


if __name__ == '__main__':
    # Run the templatized test
    func = globals()[sys.argv[1]]
    # dtype is the char representation
    use_mpi = True if sys.argv[2] == "mpi" else False
    dtype = sys.argv[3] if len(sys.argv) == 4 else None
    if dtype is not None:
        func(dtype, use_mpi)
    else:
        func(use_mpi)
