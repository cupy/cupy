from __future__ import division

import argparse
import math
import os

import cupy as cp
import numpy as np

from utils import benchmark
from utils import load_kernel
from utils import read_code


sgemm_file = os.path.join(os.path.dirname(__file__), 'sgemm.cu')


def sgemm(A, B,
          dim_x=16, dim_y=16, blk_m=64, blk_n=64, blk_k=4,
          dim_xa=64, dim_ya=4, dim_xb=4, dim_yb=64):
    assert A.dtype == cp.float32
    assert B.dtype == cp.float32
    assert(dim_x * dim_y == dim_xa * dim_ya == dim_xb * dim_yb)

    m, k = A.shape
    k, n = B.shape

    # Inputs matrices need to be in Fortran order.
    A = cp.asfortranarray(A)
    B = cp.asfortranarray(B)

    C = cp.empty((m, n), dtype=cp.float32, order='F')

    config = {'DIM_X': dim_x, 'DIM_Y': dim_y,
              'BLK_M': blk_m, 'BLK_N': blk_n, 'BLK_K': blk_k,
              'DIM_XA': dim_xa, 'DIM_YA': dim_ya,
              'DIM_XB': dim_xb, 'DIM_YB': dim_yb,
              'THR_M': blk_m // dim_x, 'THR_N': blk_n // dim_y}
    code = read_code(sgemm_file, params=config)
    kern = load_kernel('sgemm', code)

    grid = (int(math.ceil(m / blk_m)), int(math.ceil(n / blk_n)), 1)
    block = (dim_x, dim_y, 1)
    args = (m, n, k, A, B, C)
    shared_mem = blk_k * (blk_m + 1) * 4 + blk_n * (blk_k + 1) * 4
    kern(grid, block, args=args, shared_mem=shared_mem)
    return C


def main():
    parser = argparse.ArgumentParser(
        description='SGEMM kernel call from CuPy')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='ID of GPU.')
    parser.add_argument(
        '--m', type=int, default=np.random.randint(1000, 1500))
    parser.add_argument(
        '--n', type=int, default=np.random.randint(1000, 1500))
    parser.add_argument(
        '--k', type=int, default=np.random.randint(500, 3000))
    args = parser.parse_args()

    print('m={} n={} k={}'.format(args.m, args.n, args.k))
    print('start benchmarking')
    print('')

    with cp.cuda.Device(args.gpu):
        A = cp.random.uniform(
            low=-1., high=1., size=(args.m, args.k)).astype(cp.float32)
        B = cp.random.uniform(
            low=-1., high=1., size=(args.k, args.n)).astype(cp.float32)

        # check correctness
        cp.testing.assert_array_equal(sgemm(A, B), cp.dot(A, B))

        # dry run
        for _ in range(3):
            sgemm(A, B)
        kernel_times = benchmark(sgemm, (A, B), n_run=5)

        for _ in range(3):
            cp.dot(A, B)
        cublas_times = benchmark(cp.dot, (A, B), n_run=5)

    print('=============================Result===============================')
    print('hand written kernel time {} ms'.format(np.mean(kernel_times)))
    print('cuBLAS              time {} ms'.format(np.mean(cublas_times)))


if __name__ == '__main__':
    main()
