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


class _StubModule:
    """Placeholder for CUDA-only `cupyx` extension modules.

    `cupyx.cusolver` / `cupyx.cusparse` are Cython extension modules linked
    against cuSOLVER / cuSPARSE, and `cupyx.cutensor` against cuTENSOR; none
    of them is built for the Ascend backend.  Accessing them returns this
    stub, whose attribute access raises a clear ``NotImplementedError``
    instead of an import error.

    `cusolver` and `cusparse` are planned to be supported in the future via
    aclnn solver / sparse ops; `cutensor` currently has no aclnn equivalent.
    """

    def __init__(self, name):
        self._name = name

    def __getattr__(self, item):
        raise NotImplementedError(
            'cupyx.{} is not supported on the Ascend backend yet '
            '(may be supported in the future)'.format(self._name))

    def __dir__(self):
        return []


# CUDA-only submodules: import lazily so the attribute access works on the
# CUDA backend, and degrade to a stub when they are not built (Ascend).
_CUDA_ONLY_SUBMODULES = ('cusolver', 'cusparse', 'cutensor')


def __getattr__(key):
    if key == 'lapack':
        import cupyx.lapack
        return cupyx.lapack

    if key in _CUDA_ONLY_SUBMODULES:
        # Ascend: not supported yet, always return the stub (a stale,
        # previously built extension may still be importable -- do not
        # return it, it would fail later with CUDA errors).
        from cupy.backends.backend import is_ascend
        if is_ascend:
            return _StubModule(key)
        import importlib
        try:
            return importlib.import_module('cupyx.' + key)
        except ImportError:
            return _StubModule(key)

    raise AttributeError(
        "module '{}' has no attribute '{}'".format(__name__, key))
