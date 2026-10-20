# SPDX-FileCopyrightText: 2023-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""CuPy port of NVIDIA's cuDSS/simple_get_set/simple_get_set.cpp.

https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS/simple_get_set
"""
import argparse

import numpy as np

from _common import check_solution, get_data, set_config, system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'float64'],
                        default='float64')
    parser.add_argument('--index-dtype', choices=['int32', 'int64'],
                        default='int32')
    args = parser.parse_args()

    import cupy as cp
    from cupy_backends.cuda.libs import cudss

    version = '.'.join(str(cudss.getProperty(i)) for i in range(3))
    print('cuDSS version:', version)
    with system(symmetric=True, dtype=args.dtype,
                index_dtype=args.index_dtype) as problem:
        set_config(problem.config, cudss.CUDSS_CONFIG_REORDERING_ALG,
                   cudss.CUDSS_REORDERING_ALG_DEFAULT)
        set_config(problem.config, cudss.CUDSS_CONFIG_MATCHING_ALG,
                   cudss.CUDSS_MATCHING_ALG_AUTO)
        set_config(problem.config, cudss.CUDSS_CONFIG_USE_SUPERPANELS, 0)

        cudss.execute(problem.handle, cudss.CUDSS_PHASE_ANALYSIS,
                      problem.config, problem.data,
                      problem.matrix, problem.x, problem.b)
        for name in ('PERM_REORDER_ROW', 'PERM_REORDER_COL'):
            print(name, get_data(problem, getattr(cudss, 'CUDSS_DATA_' + name),
                                 5, args.index_dtype))
        matching = np.empty(1, dtype=np.int32)
        cudss.configGet(problem.config, cudss.CUDSS_CONFIG_MATCHING_ALG,
                        matching.ctypes.data, matching.nbytes)
        print('MATCHING_ALG', matching[0])
        if matching[0] != cudss.CUDSS_MATCHING_ALG_NONE:
            print('PERM_MATCHING', get_data(
                problem, cudss.CUDSS_DATA_PERM_MATCHING, 5, args.index_dtype))
        estimates = get_data(
            problem, cudss.CUDSS_DATA_MEMORY_ESTIMATES, 16, np.int64)
        print('device memory (permanent, peak):', estimates[:2])
        print('host memory (permanent, peak):', estimates[2:4])
        print('hybrid memory (device minimum, host maximum):', estimates[4:6])

        cudss.execute(problem.handle, cudss.CUDSS_PHASE_FACTORIZATION,
                      problem.config, problem.data,
                      problem.matrix, problem.x, problem.b)
        # INFO is always int32, even for a matrix with int64 indices.
        info = get_data(problem, cudss.CUDSS_DATA_INFO, 1, np.int32)[0]
        print('INFO', info)
        if info != 0:
            raise RuntimeError(f'cuDSS factorization returned INFO={info}')
        print('NPIVOTS', get_data(
            problem, cudss.CUDSS_DATA_NPIVOTS, 1, args.index_dtype))
        print('INERTIA', get_data(
            problem, cudss.CUDSS_DATA_INERTIA, 2, args.index_dtype))

        # Query the required buffer size before retrieving an int64 LU count.
        size = cudss.dataGet(problem.handle, problem.data,
                             cudss.CUDSS_DATA_LU_NNZ, 0, 0)
        lu_nnz = np.empty(size // np.dtype(np.int64).itemsize, dtype=np.int64)
        cudss.dataGet(problem.handle, problem.data, cudss.CUDSS_DATA_LU_NNZ,
                      lu_nnz.ctypes.data, lu_nnz.nbytes)
        print('LU_NNZ', lu_nnz)

        # Demonstrate retrieving factor/scaling arrays directly to the GPU.
        names = ['DIAG']
        if matching[0] in (cudss.CUDSS_MATCHING_ALG_AUTO,
                           cudss.CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT):
            names += ['SCALE_ROW', 'SCALE_COL']
        for name in names:
            buffer = cp.empty(5, dtype=args.dtype)
            cudss.dataGet(problem.handle, problem.data,
                          getattr(cudss, 'CUDSS_DATA_' + name),
                          buffer.data.ptr, buffer.nbytes)
            print(name, cp.asnumpy(buffer))

        set_config(problem.config, cudss.CUDSS_CONFIG_IR_N_STEPS, 2)
        cudss.execute(problem.handle, cudss.CUDSS_PHASE_SOLVE,
                      problem.config, problem.data,
                      problem.matrix, problem.x, problem.b)
        check_solution(problem)


if __name__ == '__main__':
    main()
