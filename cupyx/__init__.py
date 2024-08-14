from cupy._core.syncdetect import DeviceSynchronized, allow_synchronize
from cupyx import linalg, optimizing, scipy, time
from cupyx._gufunc import GeneralizedUFunc
from cupyx._pinned_array import (
    empty_like_pinned,
    empty_pinned,
    zeros_like_pinned,
    zeros_pinned,
)
from cupyx._rsqrt import rsqrt
from cupyx._runtime import get_runtime_info
from cupyx._scatter import scatter_add, scatter_max, scatter_min
from cupyx._ufunc_config import errstate, geterr, seterr


def __getattr__(key):
    if key == 'lapack':
        import cupyx.lapack
        return cupyx.lapack

    raise AttributeError(
        "module '{}' has no attribute '{}'".format(__name__, key))
