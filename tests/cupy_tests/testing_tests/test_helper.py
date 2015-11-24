import re
import unittest

import numpy

from cupy.testing import helper


class TestContainsSignedAndUnsigned(unittest.TestCase):

    def test_include(self):
        kw = {'x': numpy.int32, 'y': numpy.uint32}
        self.assertTrue(helper._contains_signed_and_unsigned(kw))

        kw = {'x': numpy.float32, 'y': numpy.uint32}
        self.assertTrue(helper._contains_signed_and_unsigned(kw))

    def test_signed_only(self):
        kw = {'x': numpy.int32}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))

        kw = {'x': numpy.float}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))

    def test_unsigned_only(self):
        kw = {'x': numpy.uint32}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))


class TestCheckCupyNumpyError(unittest.TestCase):

    def test_both_success(self):
        with self.assertRaises(AssertionError):
            helper._check_cupy_numpy_error(self, None, None, None, None)

    def test_cupy_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        with self.assertRaisesRegexp(AssertionError, cupy_tb):
            helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                           None, None)

    def test_numpy_error(self):
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        with self.assertRaisesRegexp(AssertionError, numpy_tb):
            helper._check_cupy_numpy_error(self, None, None,
                                           numpy_error, numpy_tb)

    def test_cupy_numpy_different_error(self):
        cupy_error = TypeError()
        cupy_tb = 'xxxx'
        numpy_error = ValueError()
        numpy_tb = 'yyyy'
        # Use re.S mode to ignore new line characters
        pattern = re.compile(cupy_tb + '.*' + numpy_tb, re.S)
        with self.assertRaisesRegexp(AssertionError, pattern):
            helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                           numpy_error, numpy_tb)

    def test_same_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        # Nothing happens
        helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                       numpy_error, numpy_tb)

    def test_forbidden_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        # Use re.S mode to ignore new line characters
        pattern = re.compile(cupy_tb + '.*' + numpy_tb, re.S)
        with self.assertRaisesRegexp(AssertionError, pattern):
            helper._check_cupy_numpy_error(
                self, cupy_error, cupy_tb,
                numpy_error, numpy_tb, accept_error=False)
