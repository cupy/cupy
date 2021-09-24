"""
Tests for the array API namespace.

Note, full compliance with the array API can be tested with the official array API test
suite https://github.com/data-apis/array-api-tests. This test suite primarily
focuses on those things that are not tested by the official test suite.
"""

import sys

import pytest


if sys.version_info < (3, 8):
    pytest.skip('Python array API requires Python 3.8+',
                allow_module_level=True)
