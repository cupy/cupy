import typing as _typing

# Legacy modules
from cupy_builder import cupy_setup_build
from cupy_builder import install_build
from cupy_builder import install_utils

import cupy_builder._command
from cupy_builder._context import Context
from cupy_builder._preflight import preflight_check
from cupy_builder._features import get_features


_context: _typing.Optional[Context] = None


def initialize(context: Context) -> None:
    global _context
    _context = context


def get_context() -> Context:
    assert _context is not None, 'cupy_builder is not initialized'
    return _context
