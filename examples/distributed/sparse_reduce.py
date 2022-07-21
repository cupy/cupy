#!/usr/bin/env python

# Simple "cupyx.distributed" example using sparse matrix.
# To try this script on a single node (with 2+ GPUs), run:
#    $ mpiexec -n 2 ./sparse_reduce.py


import argparse
import os

import scipy
import mpi4py

import cupy
import cupyx.distributed


def main():
    comm = mpi4py.MPI.COMM_WORLD
    workers = comm.Get_size()
    rank = comm.Get_rank()
    pid = os.getpid()

    print(f'[{pid}] Size: {workers}')
    print(f'[{pid}] Rank: {rank}')

    cupy.cuda.Device(rank).use()
    comm = cupyx.distributed.init_process_group(workers, rank, use_mpi=True)
    sm_gpu = cupyx.scipy.sparse.csr_matrix(generate(rank))
    comm.reduce(sm_gpu, sm_gpu, root=0, op='sum')

    if rank == 0:
        expected = sum([generate(n) for n in range(workers)])
        actual = sm_gpu.get()
        assert (expected != actual).nnz == 0
        print('Success!')


def generate(seed):
    return scipy.sparse.random(1000, 1000, format='csr', random_state=seed)


if __name__ == '__main__':
    main()
