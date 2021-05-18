import os

import numpy
from numpy import linalg

import cupy
import cupy._util
from cupy import _core
import cupyx


_default_precision = os.getenv('CUPY_DEFAULT_PRECISION')


def _assert_cupy_array(*arrays):
    for a in arrays:
        if not isinstance(a, cupy._core.ndarray):
            raise linalg.LinAlgError(
                'cupy.linalg only supports cupy.ndarray')


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


def linalg_common_type(*arrays, reject_float16=True):
    """Common type for linalg

    The logic is intended to be equivalent with
    `numpy.linalg.linalg._commonType`.
    The differences from `numpy.common_type` are
    - to accept ``bool_`` arrays, and
    - to reject ``float16`` arrays.

    Args:
        *arrays (ndarray): Input arrays.
        reject_float16 (bool): Flag to follow NumPy to raise TypeError for
            ``float16`` inputs.

    Returns:
        compute_dtype (dtype): The precision to be used in linalg calls.
        result_dtype (dtype): The dtype of (possibly complex) output(s).
    """
    dtypes = [arr.dtype for arr in arrays]
    if reject_float16 and 'float16' in dtypes:
        raise TypeError('float16 is unsupported in linalg')

    if _default_precision is not None:
        cupy._util.experimental('CUPY_DEFAULT_PRECISION')
        if _default_precision not in ('32', '64'):
            raise ValueError(
                'invalid CUPY_DEFAULT_PRECISION: {}'.format(
                    _default_precision))
        default = 'float' + _default_precision
    else:
        default = 'float64'
    compute_dtype = _common_type_internal(default, *dtypes)
    # No fp16 cuSOLVER routines
    if compute_dtype == 'float16':
        compute_dtype = numpy.dtype('float32')

    # numpy casts integer types to float64
    result_dtype = _common_type_internal('float64', *dtypes)

    return compute_dtype, result_dtype


def _common_type_internal(default_dtype, *dtypes):
    inexact_dtypes = [
        dtype if dtype.kind in 'fc' else default_dtype
        for dtype in dtypes]
    return numpy.result_type(*inexact_dtypes)


def _check_cusolver_dev_info_if_synchronization_allowed(routine, dev_info):
    # `dev_info` contains integers, the status code of a cuSOLVER
    # routine call. It is referred to as "infoArray" or "devInfo" in the
    # official cuSOLVER documentation.
    assert isinstance(dev_info, _core.ndarray)
    config_linalg = cupyx._ufunc_config.get_config_linalg()
    # Only 'ignore' and 'raise' are currently supported.
    if config_linalg == 'ignore':
        return

    try:
        name = routine.__name__
    except AttributeError:
        name = routine  # routine is a str

    assert config_linalg == 'raise'
    if (dev_info != 0).any():
        raise linalg.LinAlgError(
            'Error reported by {} in cuSOLVER. devInfo = {}. Please refer'
            ' to the cuSOLVER documentation.'.format(
                name, dev_info))


def _check_cublas_info_array_if_synchronization_allowed(routine, info_array):
    # `info_array` contains integers, the status codes of a cuBLAS routine
    # call. It is referrd to as "infoArray" or "devInfoArray" in the official
    # cuBLAS documentation.
    assert isinstance(info_array, _core.ndarray)
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


_tril_kernel = _core.ElementwiseKernel(
    'int64 k', 'S x',
    'x = (_ind.get()[1] - _ind.get()[0] <= k) ? x : 0',
    'tril_kernel',
    reduce_dims=False
)


def _tril(x, k=0):
    _tril_kernel(k, x)
    return x


_triu_kernel = _core.ElementwiseKernel(
    'int64 k', 'S x',
    'x = (_ind.get()[1] - _ind.get()[0] >= k) ? x : 0',
    'triu_kernel',
    reduce_dims=False
)


def _triu(x, k=0):
    _triu_kernel(k, x)
    return x
