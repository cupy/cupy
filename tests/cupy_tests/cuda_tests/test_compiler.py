import pickle
import unittest
from unittest import mock

import cupy
from cupy.cuda import compiler
from cupy import testing


def cuda_version():
    return cupy.cuda.runtime.runtimeGetVersion()


@testing.gpu
class TestNvrtcArch(unittest.TestCase):
    def setUp(self):
        cupy.clear_memo()  # _get_arch result is cached

    def _check_get_arch(self, device_cc, expected_arch):
        with mock.patch('cupy.cuda.device.Device') as device_class:
            device_class.return_value.compute_capability = device_cc
            assert compiler._get_arch() == expected_arch
        cupy.clear_memo()  # _get_arch result is cached

    @unittest.skipUnless(9000 <= cuda_version(), 'Requires CUDA 9.x or later')
    def test_get_arch_cuda9(self):
        self._check_get_arch('62', '62')  # Tegra
        self._check_get_arch('70', '70')
        self._check_get_arch('72', '72')  # Tegra

    @unittest.skipUnless(10010 <= cuda_version(),
                         'Requires CUDA 10.1 or later')
    def test_get_arch_cuda101(self):
        self._check_get_arch('75', '75')

    @unittest.skipUnless(11000 <= cuda_version(),
                         'Requires CUDA 11.0 or later')
    def test_get_arch_cuda11(self):
        self._check_get_arch('80', '80')

    def _compile(self, arch):
        compiler.compile_using_nvrtc('', arch=arch)

    @unittest.skipUnless(9000 <= cuda_version(), 'Requires CUDA 9.0 or later')
    def test_compile_cuda9(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_72` as it is for Tegra.)
        self._compile('70')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, '73')

    @unittest.skipUnless(10010 <= cuda_version() < 11000,
                         'Requires CUDA 10.1 or 10.2')
    def test_compile_cuda101(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_72` as it is for Tegra.)
        self._compile('75')

        # It should fail. (compute_80 is not supported until CUDA 11)
        self.assertRaises(
            compiler.CompileException, self._compile, '80')

    @unittest.skipUnless(11000 <= cuda_version(),
                         'Requires CUDA 11.0 or later')
    def test_compile_cuda11(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        self._compile('80')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, '83')


@testing.gpu
class TestNvrtcStderr(unittest.TestCase):

    @unittest.skipIf(cupy.cuda.runtime.is_hip,
                     'HIPRTC has different error message')
    def test1(self):
        # An error message contains the file name `kern.cu`
        with self.assertRaisesRegex(compiler.CompileException, 'kern.cu'):
            compiler.compile_using_nvrtc('a')

    @unittest.skipIf(not cupy.cuda.runtime.is_hip,
                     'NVRTC has different error message')
    def test2(self):
        with self.assertRaises(compiler.CompileException) as e:
            compiler.compile_using_nvrtc('a')
            assert "unknown type name 'a'" in e


class TestIsValidKernelName(unittest.TestCase):

    def test_valid(self):
        assert compiler.is_valid_kernel_name('valid_name_1')

    def test_empty(self):
        assert not compiler.is_valid_kernel_name('')

    def test_start_with_digit(self):
        assert not compiler.is_valid_kernel_name('0_invalid')

    def test_new_line(self):
        assert not compiler.is_valid_kernel_name('invalid\nname')

    def test_symbol(self):
        assert not compiler.is_valid_kernel_name('invalid$name')

    def test_space(self):
        assert not compiler.is_valid_kernel_name('invalid name')


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = compiler.CompileException('msg', 'fn.cu', 'fn', ('-ftz=true',))
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
