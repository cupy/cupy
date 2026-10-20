# SPDX-FileCopyrightText: 2023-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""CuPy port of NVIDIA's cuDSS/simple_multithreaded_mode example.

Source: NVIDIA/CUDALibrarySamples, cuDSS/simple_multithreaded_mode.
See README.md for the original sample URL.
"""
import argparse
import os

from _common import check_solution, set_config, system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('threading_layer', nargs='?',
                        help='library path; otherwise use CUDSS_THREADING_LIB')
    parser.add_argument('--threads', type=int, default=2,
                        help='number of cuDSS CPU threads (default: 2)')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if not (args.threading_layer or os.environ.get('CUDSS_THREADING_LIB')):
        parser.error('provide a threading layer or set CUDSS_THREADING_LIB')

    from cupy_backends.cuda.libs import cudss

    def configure_handle(handle):
        # None makes cuDSS read CUDSS_THREADING_LIB, as in the CUDA example.
        cudss.setThreadingLayer(handle, args.threading_layer)

    with system(configure_handle=configure_handle) as problem:
        set_config(problem.config, cudss.CUDSS_CONFIG_HOST_NTHREADS,
                   args.threads)
        for phase in (cudss.CUDSS_PHASE_ANALYSIS,
                      cudss.CUDSS_PHASE_FACTORIZATION,
                      cudss.CUDSS_PHASE_SOLVE):
            cudss.execute(problem.handle, phase, problem.config, problem.data,
                          problem.matrix, problem.x, problem.b)
        check_solution(problem)


if __name__ == '__main__':
    main()
