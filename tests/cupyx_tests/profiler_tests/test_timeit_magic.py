"""Tests for IPython %gpu_timeit magic."""
from __future__ import annotations

import re
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


def _capture_output(ipython_shell, magic_type, line, cell=None):
    """Capture stdout from running a magic command."""
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        if magic_type == 'line':
            ipython_shell.run_line_magic('gpu_timeit', line)
        else:
            ipython_shell.run_cell_magic('gpu_timeit', line, cell)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    return output


def _verify_benchmark_output(output):
    """Verify output contains expected benchmark results."""
    # Check for presence of CPU and GPU timing info
    assert 'CPU:' in output, "Output should contain CPU timing"
    assert 'GPU' in output, "Output should contain GPU timing"
    assert 'us' in output or 'ms' in output, "Output should contain time units"

    # Verify format matches benchmark output (contains numbers)
    # Example: "CPU:    40.705 us   +/-  1.570"
    number_pattern = r'\d+\.\d+'
    assert re.search(
        number_pattern, output), "Output should contain timing numbers"


class TestGPUTimeitMagic:
    """Test %gpu_timeit magic functionality."""

    def test_line_magic_simple_expression(self, ipython_shell):
        """Test line magic with a simple expression."""
        output = _capture_output(
            ipython_shell, 'line', 'cp.array([1, 2, 3]).sum()'
        )
        _verify_benchmark_output(output)

    def test_line_magic_with_namespace(self, ipython_shell):
        """Test that line magic can access user namespace."""
        ipython_shell.user_ns['x'] = cupy.array([1, 2, 3, 4, 5])
        output = _capture_output(ipython_shell, 'line', 'x.sum()')
        _verify_benchmark_output(output)

    def test_line_magic_with_options(self, ipython_shell):
        """Test line magic with custom options."""
        output = _capture_output(
            ipython_shell, 'line', '-n 10 -w 2 cp.ones(100).sum()'
        )
        _verify_benchmark_output(output)

    def test_cell_magic_simple(self, ipython_shell):
        """Test cell magic with simple code."""
        cell_code = """
x = cp.random.random((100, 100))
y = x @ x.T
"""
        output = _capture_output(ipython_shell, 'cell', '', cell_code)
        _verify_benchmark_output(output)

    def test_cell_magic_with_existing_namespace(self, ipython_shell):
        """Test cell magic can access existing variables."""
        ipython_shell.user_ns['n'] = 100
        cell_code = """
x = cp.random.random((n, n))
result = x.sum()
"""
        output = _capture_output(ipython_shell, 'cell', '', cell_code)
        _verify_benchmark_output(output)

    def test_cell_magic_multiline(self, ipython_shell):
        """Test cell magic with multiple operations."""
        cell_code = """
a = cp.ones((50, 50))
b = cp.ones((50, 50))
c = a + b
d = c * 2
result = d.sum()
"""
        output = _capture_output(ipython_shell, 'cell', '', cell_code)
        _verify_benchmark_output(output)

    def test_cell_magic_with_options(self, ipython_shell):
        """Test cell magic with custom options."""
        cell_code = """
x = cp.ones((10, 10))
y = x.sum()
"""
        output = _capture_output(
            ipython_shell, 'cell', '-n 5 --max-duration 1', cell_code
        )
        _verify_benchmark_output(output)

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
