from cupyx._rsqrt import rsqrt
from cupyx._runtime import get_runtime_info
from cupyx._scatter import scatter_add
from cupyx._scatter import scatter_max
from cupyx._scatter import scatter_min

from cupyx import linalg
from cupyx import time
from cupyx import scipy
from cupyx import optimizing

from cupyx._ufunc_config import errstate
from cupyx._ufunc_config import geterr
from cupyx._ufunc_config import seterr
from cupy._core.syncdetect import allow_synchronize
from cupy._core.syncdetect import DeviceSynchronized

from cupyx._pinned_array import empty_pinned
from cupyx._pinned_array import empty_like_pinned
from cupyx._pinned_array import zeros_pinned
from cupyx._pinned_array import zeros_like_pinned

from cupyx._gufunc import GeneralizedUFunc


def __getattr__(key):
    if key == 'lapack':
        import cupyx.lapack
        return cupyx.lapack

    raise AttributeError(
        "module '{}' has no attribute '{}'".format(__name__, key))
