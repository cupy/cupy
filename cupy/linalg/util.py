from numpy import linalg

import cupy
from cupy import core


def _assert_cupy_array(*arrays):
    for a in arrays:
        if not isinstance(a, cupy.core.ndarray):
            raise linalg.LinAlgError(
                'cupy.linalg only supports cupy.core.ndarray')


def _assert_rank2(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise linalg.LinAlgError(
                '{}-dimensional array given. Array must be '
                'two-dimensional'.format(a.ndim))


def _assert_nd_squareness(*arrays):
    for a in arrays:
        if max(a.shape[-2:]) != min(a.shape[-2:]):
            raise linalg.LinAlgError(
                'Last 2 dimensions of the array must be square')


def _synchronize_check_cusolver_dev_info(routine, dev_info):
    # `dev_info` contains a single integer, the status code of a cuSOLVER
    # routine call. It is referred to as "devInfo" in the official cuSOLVER
    # documentation.
    assert isinstance(dev_info, core.ndarray)
    assert dev_info.size == 1

    dev_info_host = dev_info.item()

    if dev_info_host != 0:
        raise linalg.LinAlgError(
            'Error reported by {} in cuSOLVER. devInfo = {}. Please refer'
            ' to the cuSOLVER documentation.'.format(
                routine.__name__, dev_info_host))


def _check_cusolver_dev_info_if_synchronization_allowed(routine, dev_info):
    if cupy.linalg._synchronize_check_cusolver_dev_info:
        _synchronize_check_cusolver_dev_info(routine, dev_info)


_tril_kernel = core.ElementwiseKernel(
    'int64 k', 'S x',
    'x = (_ind.get()[1] - _ind.get()[0] <= k) ? x : 0',
    'tril_kernel',
    reduce_dims=False
)


def _tril(x, k=0):
    _tril_kernel(k, x)
    return x


_triu_kernel = core.ElementwiseKernel(
    'int64 k', 'S x',
    'x = (_ind.get()[1] - _ind.get()[0] >= k) ? x : 0',
    'triu_kernel',
    reduce_dims=False
)


def _triu(x, k=0):
    _triu_kernel(k, x)
    return x
