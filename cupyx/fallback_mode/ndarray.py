"""
Wrapper around cupy.ndarray
to support fallback of methods such as `ndarray.func()`
"""

import numpy as np

import cupy as cp
from cupyx.fallback_mode import utils
from cupyx.fallback_mode import data_transfer


class ndarray:

    def __init__(self, array):
        self._array = array
        self.func = None

    def __repr__(self):
        return self._array.__repr__()

    def __getattr__(self, attr):

        self.func = getattr(cp.ndarray, attr, None)
        if not callable(self.func) and self.func is not None:
            return getattr(self._array, attr)

        if self.func is not None:
            return self._call_cupy_ndarray

        self.func = getattr(np.ndarray, attr)
        return self._call_numpy_ndarray

    def _call_cupy_ndarray(self, *args, **kwargs):

        res = self.func(self._array, *args, **kwargs)

        return utils._get_fallback_ndarray(res)

    def _call_numpy_ndarray(self, *args, **kwargs):

        numpy_args, numpy_kwargs = data_transfer._get_numpy_args(args, kwargs)

        numpy_array = cp.asnumpy(self._array)
        numpy_res = self.func(numpy_array, *numpy_args, **numpy_kwargs)

        cupy_res = data_transfer._get_cupy_result(numpy_res)

        return utils._get_fallback_ndarray(cupy_res)
