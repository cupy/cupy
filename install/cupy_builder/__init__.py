import typing as _typing

# Legacy modules
from cupy_builder import cupy_setup_build  # NOQA
from cupy_builder import install_build  # NOQA
from cupy_builder import install_utils  # NOQA

import cupy_builder._command  # NOQA
from cupy_builder._context import Context  # NOQA
from cupy_builder._preflight import preflight_check  # NOQA
from cupy_builder._modules import get_modules  # NOQA


_context: _typing.Optional[Context] = None


def initialize(context: Context) -> None:
    global _context
    _context = context


def get_context() -> Context:
    assert _context is not None, 'cupy_builder is not initialized'
    return _context
