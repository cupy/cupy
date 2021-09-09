from cupy_builder import cupy_setup_build  # NOQA
from cupy_builder import install_build  # NOQA
from cupy_builder import install_utils  # NOQA


_source_root = None


def initialize(source_root):
    global _source_root
    _source_root = source_root


def get_source_root():
    assert _source_root is not None, 'cupy_builder is not initialized'
    return _source_root
