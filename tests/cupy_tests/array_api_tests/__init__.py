"""
Tests for the array API namespace.

Note, full compliance with the array API can be tested with the official array API test
suite https://github.com/data-apis/array-api-tests. This test suite primarily
focuses on those things that are not tested by the official test suite.
"""

import sys

import pytest

from cupy_backends.cuda.api import runtime


if sys.version_info < (3, 8):
    pytest.skip('Python array API requires Python 3.8+',
                allow_module_level=True)


# hiprtc seems to have a bug and it causes errors in some tests later. We
# temporarily skip the Python array API tests on ROCm until it is fixed.
# See #5843
if runtime.is_hip:
    pytest.skip('Python array API tests are temporarily skipped on ROCm',
                allow_module_level=True)
