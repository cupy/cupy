#!/usr/bin/env python

# Simple "cupyx.distributed" example.
# To try this script on a single node (with 2+ GPUs), either:
#   (a) Run 2 processes manually at once (TCP-backed execution)
#     $ ./simple_reduce.py -n 2 -r 0 & ./simple_reduce.py -n 2 -r 1
#   (b) Run via mpiexec (MPI4py-backed execution)
#     $ mpiexec -n 2 ./simple_reduce.py --mpi


import argparse
import os

import cupy
import cupyx.distributed


def parse_args():
    parser = argparse.ArgumentParser()

    # For TCP-backed execution
    parser.add_argument('-n', '--workers', type=int, default=-1,
                        help='number of workers')
    parser.add_argument('-r', '--rank', type=int, default=-1,
                        help='rank')

    # For MPI-backed execution
    parser.add_argument('-M', '--mpi', action='store_true', default=False,
                        help='use MPI4py')

    options = parser.parse_args()

    if options.mpi:
        import mpi4py
        comm = mpi4py.MPI.COMM_WORLD
        options.workers = comm.Get_size()
        options.rank = comm.Get_rank()
    elif options.workers == -1 or options.rank == -1:
        parser.error(
            'Number of workers and rank are mandatory when using TCP')

    return options


options = parse_args()
pid = os.getpid()
print(f'[{pid}] Size: {options.workers}')
print(f'[{pid}] Rank: {options.rank}')

cupy.cuda.Device(options.rank).use()

comm = cupyx.distributed.init_process_group(
    options.workers, options.rank, use_mpi=options.mpi)
array = cupy.empty(10)
array.fill(options.rank + 1)
comm.reduce(array, array, root=0, op='sum')
if options.rank == 0:
    print(array)  # expects array filled with 3.0
