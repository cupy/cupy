import os
import unittest

from cupy.cuda import compiler
from cupy import testing


@testing.gpu
class NvccNotFoundTest(unittest.TestCase):

    def setUp(self):
        self.path = os.environ.pop('PATH', '')

    def tearDown(self):
        os.environ['PATH'] = self.path

    def test_nvcc_without_command(self):
        # Check that error message includes command name `nvcc`
        with self.assertRaisesRegexp(OSError, 'nvcc'):
            compiler.nvcc('')

    def test_preprocess_without_command(self):
        # Check that error message includes command name `nvcc`
        with self.assertRaisesRegexp(OSError, 'nvcc'):
            compiler.preprocess('')


@testing.gpu
class TestNvccStderr(unittest.TestCase):

    def test(self):
        # An error message contains the file name `kern.cu`
        with self.assertRaisesRegexp(RuntimeError, 'kern.cu'):
            compiler.nvcc('a')
