import typing as _typing

import cupy_builder._command

# Legacy modules
from cupy_builder import cupy_setup_build, install_build, install_utils
from cupy_builder._context import Context
from cupy_builder._features import get_features
from cupy_builder._preflight import preflight_check

_context: _typing.Optional[Context] = None


def initialize(context: Context) -> None:
    global _context
    _context = context


def get_context() -> Context:
    assert _context is not None, 'cupy_builder is not initialized'
    return _context
