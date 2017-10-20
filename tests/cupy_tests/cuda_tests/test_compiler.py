import unittest

import six

from cupy.cuda import compiler
from cupy import testing


@testing.gpu
class TestNvrtcStderr(unittest.TestCase):

    def test(self):
        # An error message contains the file name `kern.cu`
        with six.assertRaisesRegex(self, compiler.CompileException, 'kern.cu'):
            compiler.compile_using_nvrtc('a')


class TestIsValidKernelName(unittest.TestCase):

    def test_valid(self):
        self.assertTrue(compiler.is_valid_kernel_name('valid_name_1'))

    def test_empyt(self):
        self.assertFalse(compiler.is_valid_kernel_name(''))

    def test_start_with_digit(self):
        self.assertFalse(compiler.is_valid_kernel_name('0_invalid'))

    def test_new_line(self):
        self.assertFalse(compiler.is_valid_kernel_name('invalid\nname'))

    def test_symbol(self):
        self.assertFalse(compiler.is_valid_kernel_name('invalid$name'))

    def test_space(self):
        self.assertFalse(compiler.is_valid_kernel_name('invalid name'))
