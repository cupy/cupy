# "NOQA" to suppress flake8 warning
from __future__ import annotations

from cupyx._rsqrt import rsqrt  # NOQA
from cupyx._runtime import get_runtime_info  # NOQA
from cupyx._scatter import scatter_add  # NOQA
from cupyx._scatter import scatter_max  # NOQA
from cupyx._scatter import scatter_min  # NOQA

from cupyx import linalg  # NOQA
from cupyx import time  # NOQA
from cupyx import scipy  # NOQA
from cupyx import optimizing  # NOQA

from cupyx._ufunc_config import errstate  # NOQA
from cupyx._ufunc_config import geterr  # NOQA
from cupyx._ufunc_config import seterr  # NOQA
from cupy._core.syncdetect import allow_synchronize  # NOQA
from cupy._core.syncdetect import DeviceSynchronized  # NOQA

from cupyx._pinned_array import empty_pinned  # NOQA
from cupyx._pinned_array import empty_like_pinned  # NOQA
from cupyx._pinned_array import zeros_pinned  # NOQA
from cupyx._pinned_array import zeros_like_pinned  # NOQA

from cupyx._gufunc import GeneralizedUFunc  # NOQA


def __getattr__(key):
    if key == 'lapack':
        import cupyx.lapack
        return cupyx.lapack

    raise AttributeError(
        "module '{}' has no attribute '{}'".format(__name__, key))


# Auto-register IPython magics only when running in IPython/Jupyter
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        from cupyx.profiler._gpu_timeit import try_register_ipython_magic
        try_register_ipython_magic()
except Exception:
    # Must never fail in a normal Python environment
    pass
