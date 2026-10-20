# SPDX-FileCopyrightText: 2023-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""CuPy port of NVIDIA's cuDSS/simple/simple.cpp.

https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS/simple
"""
from _common import check_solution, system


def main():
    from cupy_backends.cuda.libs import cudss

    # _common.system creates CuPy arrays, a stream, solver objects and matrix
    # descriptors, and releases all cuDSS resources on exit.
    with system() as problem:
        for phase in (cudss.CUDSS_PHASE_ANALYSIS,
                      cudss.CUDSS_PHASE_FACTORIZATION,
                      cudss.CUDSS_PHASE_SOLVE):
            cudss.execute(problem.handle, phase, problem.config, problem.data,
                          problem.matrix, problem.x, problem.b)
        check_solution(problem)


if __name__ == '__main__':
    main()
