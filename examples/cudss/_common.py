# SPDX-FileCopyrightText: 2023-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Shared matrix setup and resource cleanup for the cuDSS examples.

Adapted from https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuDSS.
The original CUDA samples are rewritten in Python using CuPy arrays.
"""
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace

import numpy as np


@contextmanager
def system(*, root=True, symmetric=False, dtype='float64', index_dtype='int32',
           configure_handle=None):
    """Create the original 5x5 system, retaining its buffers until cleanup.

    In MGMN mode only rank 0 owns input/output buffers. Other ranks create
    descriptors with the same global dimensions and NULL data pointers.
    """
    import cupy as cp
    from cupy_backends.cuda.libs import cudss

    if not cudss.available:
        raise RuntimeError('These examples require cuDSS 0.8.0 or later')
    value_type = {'float32': cudss.CUDSS_R_32F,
                  'float64': cudss.CUDSS_R_64F}[dtype]
    integer_type = {'int32': cudss.CUDSS_R_32I,
                    'int64': cudss.CUDSS_R_64I}[index_dtype]
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, ExitStack() as resources:
        try:
            # Store only the upper triangle of the symmetric matrix.
            if root:
                offsets = cp.array([0, 2, 4, 6, 7, 8], dtype=index_dtype)
                columns = cp.array([0, 2, 1, 2, 2, 4, 3, 4], dtype=index_dtype)
                values = cp.array([4, 1, 3, 2, 5, 1, 1, 2], dtype=dtype)
                rhs = cp.array([7, 12, 25, 4, 13], dtype=dtype)
                solution = cp.empty(5, dtype=dtype)
                pointers = [a.data.ptr for a in
                            (offsets, columns, values, rhs, solution)]
            else:
                solution = None
                pointers = [0] * 5

            handle = cudss.create()
            resources.callback(cudss.destroy, handle)
            cudss.setStream(handle, stream.ptr)
            if configure_handle is not None:
                configure_handle(handle)
            config = cudss.configCreate()
            resources.callback(cudss.configDestroy, config)
            data = cudss.dataCreate(handle)
            resources.callback(cudss.dataDestroy, handle, data)

            matrix = cudss.matrixCreateCsr(
                5, 5, 8, pointers[0], 0, pointers[1], pointers[2],
                integer_type, integer_type, value_type,
                (cudss.CUDSS_MTYPE_SYMMETRIC if symmetric
                 else cudss.CUDSS_MTYPE_SPD),
                cudss.CUDSS_MVIEW_UPPER, cudss.CUDSS_BASE_ZERO)
            resources.callback(cudss.matrixDestroy, matrix)
            b = cudss.matrixCreateDn(
                5, 1, 5, pointers[3], value_type, cudss.CUDSS_LAYOUT_COL_MAJOR)
            resources.callback(cudss.matrixDestroy, b)
            x = cudss.matrixCreateDn(
                5, 1, 5, pointers[4], value_type, cudss.CUDSS_LAYOUT_COL_MAJOR)
            resources.callback(cudss.matrixDestroy, x)
            yield SimpleNamespace(handle=handle, config=config, data=data,
                                  matrix=matrix, x=x, b=b, stream=stream,
                                  solution=solution)
        finally:
            # Finish asynchronous work before descriptors or buffers are freed.
            stream.synchronize()


def set_config(config, param, value):
    """cuDSS configuration integers/enums are host-side C ints."""
    from cupy_backends.cuda.libs import cudss

    buffer = np.array([value], dtype=np.int32)
    cudss.configSet(config, param, buffer.ctypes.data, buffer.nbytes)


def get_data(problem, param, count, dtype):
    """Retrieve solver metadata into a host NumPy array."""
    from cupy_backends.cuda.libs import cudss

    buffer = np.empty(count, dtype=dtype)
    cudss.dataGet(problem.handle, problem.data, param,
                  buffer.ctypes.data, buffer.nbytes)
    return buffer


def check_solution(problem):
    """Check the known solution x = [1, 2, 3, 4, 5]."""
    import cupy as cp

    problem.stream.synchronize()
    actual = cp.asnumpy(problem.solution)
    expected = np.arange(1, 6, dtype=actual.dtype)
    tolerance = 100 * np.finfo(actual.dtype).eps
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance)
    print('x =', actual)
    print('Example PASSED')
