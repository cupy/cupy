import unittest

import mock
import six

import cupy
from cupy.cuda import compiler
from cupy import testing


def cuda_version():
    return cupy.cuda.runtime.runtimeGetVersion()


@testing.gpu
class TestNvrtcArch(unittest.TestCase):

    def _check_get_arch(self, device_cc, expected_arch):
        with mock.patch('cupy.cuda.device.Device') as device_class:
            device_class.return_value.compute_capability = device_cc
            assert compiler._get_arch() == expected_arch

    @unittest.skipUnless(cuda_version() < 9000, 'Requires CUDA 8.x or earlier')
    def test_get_arch_cuda8(self):
        self._check_get_arch('52', 'compute_52')
        self._check_get_arch('53', 'compute_53')
        self._check_get_arch('54', 'compute_53')

    @unittest.skipUnless(9000 <= cuda_version(), 'Requires CUDA 9.x or later')
    def test_get_arch_cuda9(self):
        self._check_get_arch('71', 'compute_71')
        self._check_get_arch('72', 'compute_72')
        self._check_get_arch('73', 'compute_72')

    def _compile(self, arch):
        compiler.compile_using_nvrtc('', arch=arch)

    @unittest.skipUnless(cuda_version() < 9000, 'Requires CUDA 8.x or earlier')
    def test_compile_cuda8(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_53` as it is for Tegra.)
        self._compile('compute_52')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, 'compute_54')
        self.assertRaises(
            compiler.CompileException, self._compile, 'compute_60')

    @unittest.skipUnless(9000 <= cuda_version(), 'Requires CUDA 9.0 or later')
    def test_compile_cuda9(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_72` as it is for Tegra.)
        self._compile('compute_70')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, 'compute_73')


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
