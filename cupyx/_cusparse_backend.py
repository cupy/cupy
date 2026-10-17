"""cuSPARSE backend resolution for ``cupyx.cusparse``.

``cupyx/cusparse.py`` is upstream code: it hard-imports
``cupy_backends.cuda.libs.cusparse``, which does not exist on the Ascend
backend.  This module centralises the "which sparse library am I talking to?"
decision in one place (mirroring ``cupy/fft/_backend.py``'s ``get_cufft``):

* Ascend  -> :mod:`cupy.backends.ascend.api.acl_sparse`, the ops-sparse
  (``libops_sparse.so``) binding with a cuSPARSE-compatible surface.  The
  module is optional (ops-sparse is *not* part of the base CANN SDK), so a
  failed import must degrade to a clear ``RuntimeError`` instead of breaking
  ``import cupyx.cusparse``.
* CUDA / ROCm -> the upstream ``cupy_backends.cuda.libs.cusparse``.

``import cupyx.cusparse`` therefore works on every backend; individual calls
raise an actionable error when ops-sparse was not built in.
"""
from __future__ import annotations

import functools as _functools
import os as _os

import cupy as _cupy  # NOQA: F401  (ensures cupy/__init__ ran: backend alias)
from cupy.backends.backend.api import runtime as _runtime


#: ops-sparse is optional: set ``CUPY_ASCEND_SPARSE_STRICT=1`` to turn a
#: missing binding into an ``ImportError`` instead of a lazy ``RuntimeError``
#: (useful for debugging the build).
_STRICT = _os.environ.get('CUPY_ASCEND_SPARSE_STRICT', '') == '1'


@_functools.lru_cache(maxsize=None)
def _import_acl_sparse():
    """Import the ops-sparse binding.

    Returns:
        tuple: ``(module, None)`` on success, ``(None, exc)`` when the optional
        binding is unavailable.
    """
    try:
        from cupy.backends.ascend.api import acl_sparse
    except ImportError as exc:
        if _STRICT:
            raise
        return None, exc
    return acl_sparse, None


def is_ascend() -> bool:
    return _runtime.is_ascend


def _symbol_hint(exc):
    """Extra hint when the binding exists but ``libops_sparse.so`` lacks symbols.

    ops-sparse can be built for a single SOC and for an explicit subset of
    operators (``build.sh --ops=spmv``), so a library that loads fine on the
    machine that built it can be unusable elsewhere.  The dynamic loader then
    reports an undefined ``aclsparse*`` symbol instead of "library not found".
    """
    if exc is not None and 'undefined symbol' in str(exc):
        return (
            'libops_sparse.so was found but it does not export the symbols '
            'this binding needs: it was built for a different SOC or with a '
            'subset of operators (build.sh --ops=...). Rebuild it for the '
            'target SOC (e.g. "bash build.sh --pkg --soc=ascend910b3") and '
            'reinstall. ')
    return ''


def get_cusparse():
    """Return the module implementing the cuSPARSE API for this backend.

    Returns:
        module: ``cupy.backends.ascend.api.acl_sparse`` on Ascend,
        ``cupy_backends.cuda.libs.cusparse`` otherwise.

    Raises:
        RuntimeError: on Ascend when the optional ops-sparse binding is not
            available (ops-sparse not installed at build time, or
            ``libops_sparse.so`` missing at run time).
    """
    if is_ascend():
        module, exc = _import_acl_sparse()
        if module is None:
            raise RuntimeError(
                'cusparse functionality is not available: the optional '
                'ops-sparse (libops_sparse.so) binding '
                '"cupy.backends.ascend.api.acl_sparse" is not compiled in. '
                'ops-sparse is a separate CANN package (not part of the base '
                'SDK, requires CANN >= 9.0); install it and rebuild cupy '
                '(export CUPY_INSTALL_USE_ASCEND=1 && python setup.py '
                'develop --inplace), or point the build at an existing copy '
                'via ASCEND_OPS_SPARSE_PATH. See docs/ascend/ascend_sparse.md. '
                + _symbol_hint(exc)
                + f'Original import error: {exc}')
        return module
    from cupy_backends.cuda.libs import cusparse
    return cusparse


def get_cusparse_handle():
    """Return the sparse library handle for the current device/stream."""
    if is_ascend():
        module, exc = _import_acl_sparse()
        if module is None:
            raise RuntimeError(
                'cusparse handle is not available: the optional ops-sparse '
                'binding is missing. ' + _symbol_hint(exc)
                + f'({exc}); see docs/ascend/ascend_sparse.md.')
        return module.get_handle()
    from cupy.cuda import device as _device
    return _device.get_cusparse_handle()
