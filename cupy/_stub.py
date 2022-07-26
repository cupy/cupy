"""
Stubs
=====

APIs listed here are always made available under `cupy.*` namespace even when
`import cupy` failed (e.g., no CUDA installation found), because they are
often used in the global context.
Stubs will be overridden with real implementations after successful import.

This file must not depend on any other CuPy modules.
"""


def is_available():
    return False


def _fail(*args, **kwargs):
    import cupy
    cupy.raise_import_failure()


class _fail_class:
    __init__ = _fail


def _fail_decorator(*args, **kwargs):
    return _fail


ndarray = _fail_class
ufunc = _fail_class

fuse = _fail_decorator
memoize = _fail_decorator
