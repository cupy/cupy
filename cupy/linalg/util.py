from numpy import linalg

import cupy
from cupy import core
import cupyx


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


def _check_cusolver_dev_info_if_synchronization_allowed(routine, dev_info):
    # `dev_info` contains a single integer, the status code of a cuSOLVER
    # routine call. It is referred to as "devInfo" in the official cuSOLVER
    # documentation.
    assert isinstance(dev_info, core.ndarray)
    assert dev_info.size == 1
    config_linalg = cupyx._ufunc_config.get_config_linalg()
    # Only 'ignore' and 'raise' are currently supported.
    if config_linalg == 'ignore':
        return

    assert config_linalg == 'raise'
    dev_info_host = dev_info.item()
    if dev_info_host != 0:
        raise linalg.LinAlgError(
            'Error reported by {} in cuSOLVER. devInfo = {}. Please refer'
            ' to the cuSOLVER documentation.'.format(
                routine.__name__, dev_info_host))


def _check_cublas_info_array_if_synchronization_allowed(routine, info_array):
    # `info_array` contains integers, the status codes of a cuBLAS routine
    # call. It is referrd to as "infoArray" or "devInfoArray" in the official
    # cuBLAS documentation.
    assert isinstance(info_array, core.ndarray)
    assert info_array.ndim == 1

    config_linalg = cupyx._ufunc_config.get_config_linalg()
    # Only 'ignore' and 'raise' are currently supported.
    if config_linalg == 'ignore':
        return

    assert config_linalg == 'raise'
    if (info_array != 0).any():
        raise linalg.LinAlgError(
            'Error reported by {} in cuBLAS. infoArray/devInfoArray = {}.'
            ' Please refer to the cuBLAS documentation.'.format(
                routine.__name__, info_array))


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
