"""Tests for IPython %gpu_timeit magic."""
from __future__ import annotations

import pytest

import cupy


# Skip entire module if IPython is not available
pytest.importorskip('IPython')


@pytest.fixture
def ipython_shell():
    """Provide an IPython shell for testing."""
    from IPython.testing.globalipapp import get_ipython
    ip = get_ipython()

    # Load the extension
    from cupyx.profiler import load_ipython_extension
    load_ipython_extension(ip)

    # Add cupy to namespace for convenience
    ip.user_ns['cp'] = cupy

    yield ip

    # Cleanup
    for key in ['cp', 'x', 'n', 'a', 'b', 'c', 'd', 'result']:
        ip.user_ns.pop(key, None)


class TestGPUTimeitMagic:
    """Test %gpu_timeit magic functionality."""

    def test_line_magic_simple_expression(self, ipython_shell):
        """Test line magic with a simple expression."""
        result = ipython_shell.run_line_magic(
            'gpu_timeit', 'cp.array([1, 2, 3]).sum()'
        )
        # Should return None and print output
        assert result is None

    def test_line_magic_with_namespace(self, ipython_shell):
        """Test that line magic can access user namespace."""
        ipython_shell.user_ns['x'] = cupy.array([1, 2, 3, 4, 5])
        result = ipython_shell.run_line_magic('gpu_timeit', 'x.sum()')
        assert result is None

    def test_cell_magic_simple(self, ipython_shell):
        """Test cell magic with simple code."""
        cell_code = """
x = cp.random.random((100, 100))
y = x @ x.T
"""
        result = ipython_shell.run_cell_magic('gpu_timeit', '', cell_code)
        assert result is None

    def test_cell_magic_with_existing_namespace(self, ipython_shell):
        """Test cell magic can access existing variables."""
        ipython_shell.user_ns['n'] = 100
        cell_code = """
x = cp.random.random((n, n))
result = x.sum()
"""
        result = ipython_shell.run_cell_magic('gpu_timeit', '', cell_code)
        assert result is None

    def test_cell_magic_multiline(self, ipython_shell):
        """Test cell magic with multiple operations."""
        cell_code = """
a = cp.ones((50, 50))
b = cp.ones((50, 50))
c = a + b
d = c * 2
result = d.sum()
"""
        result = ipython_shell.run_cell_magic('gpu_timeit', '', cell_code)
        assert result is None

    def test_extension_registered(self, ipython_shell):
        """Test that extension loads and registers magics properly."""
        assert 'gpu_timeit' in ipython_shell.magics_manager.magics['line']
        assert 'gpu_timeit' in ipython_shell.magics_manager.magics['cell']


def test_import_without_ipython():
    """Test importing module doesn't fail without IPython."""
    # This test just verifies the module can be imported
    # The actual import happens at module level, so if we got here, it worked
    from cupyx.profiler import _timeit_magic
    assert hasattr(_timeit_magic, 'load_ipython_extension')
