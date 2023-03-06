import pickle
import unittest
from unittest import mock

import pytest

import cupy
from cupy.cuda import compiler
from cupy import testing


def cuda_version():
    return cupy.cuda.runtime.runtimeGetVersion()


@unittest.skipIf(cupy.cuda.runtime.is_hip, 'CUDA specific tests')
class TestNvrtcArch(unittest.TestCase):
    def setUp(self):
        cupy.clear_memo()  # _get_arch result is cached

    def _check_get_arch(self, device_cc, expected_arch):
        with mock.patch('cupy.cuda.device.Device') as device_class:
            device_class.return_value.compute_capability = device_cc
            assert (
                compiler._get_cc_for_compile(compiler._get_cc()) ==
                expected_arch)
        cupy.clear_memo()  # _get_arch result is cached

    def test_get_arch(self):
        self._check_get_arch(62, 62)  # Tegra
        self._check_get_arch(70, 70)
        self._check_get_arch(72, 72)  # Tegra
        self._check_get_arch(75, 75)

    @unittest.skipUnless(10020 == cuda_version(),
                         'Requires CUDA 10.2')
    def test_get_arch_cuda102(self):
        self._check_get_arch(80, 75)

    @unittest.skipUnless(11000 == cuda_version(),
                         'Requires CUDA 11.0')
    def test_get_arch_cuda110(self):
        self._check_get_arch(80, 80)
        self._check_get_arch(86, 80)

    def _compile(self, arch):
        compiler.compile_using_nvrtc('', arch=arch)


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


class TestCompileWithCache:
    def test_compile_module_with_cache(self):
        compiler._compile_module_with_cache('__device__ void func() {}')

    def test_deprecated_compile_with_cache(self):
        with pytest.warns(UserWarning):
            compiler.compile_with_cache('__device__ void func() {}')
