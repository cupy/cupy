from distutils import ccompiler
from distutils import sysconfig
import unittest

import pytest

from . import _from_install_import


build = _from_install_import('build')


class TestCheckVersion(unittest.TestCase):

    def setUp(self):
        self.compiler = ccompiler.new_compiler()
        sysconfig.customize_compiler(self.compiler)
        self.settings = build.get_compiler_setting(False)

    @pytest.mark.gpu
    @pytest.mark.skipif(build.use_hip, reason='For CUDA environment')
    def test_check_cuda_version(self):
        with self.assertRaises(RuntimeError):
            build.get_cuda_version()
        assert build.check_cuda_version(
            self.compiler, self.settings)
        assert isinstance(build.get_cuda_version(), int)
        assert isinstance(build.get_cuda_version(True), str)

    @pytest.mark.gpu
    @pytest.mark.skipif(not build.use_hip, reason='For ROCm/HIP environment')
    def test_check_hip_version(self):
        with self.assertRaises(RuntimeError):
            build.get_hip_version()
        assert build.check_hip_version(
            self.compiler, self.settings)
        assert isinstance(build.get_hip_version(), int)
        assert isinstance(build.get_hip_version(True), str)

    @pytest.mark.gpu
    @pytest.mark.cudnn
    @pytest.mark.xfail(build.use_hip,
                       reason='ROCm/HIP DNN support is not ready')
    def test_check_cudnn_version(self):
        with self.assertRaises(RuntimeError):
            build.get_cudnn_version()
        assert build.check_cudnn_version(
            self.compiler, self.settings)
        assert isinstance(build.get_cudnn_version(), int)
        assert isinstance(build.get_cudnn_version(True), str)
