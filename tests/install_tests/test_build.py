from distutils import ccompiler
from distutils import sysconfig
import os
import unittest

import pytest

from cupy_builder import install_build as build


test_hip = bool(int(os.environ.get('CUPY_INSTALL_USE_HIP', '0')))


class TestCheckVersion(unittest.TestCase):

    def setUp(self):
        self.compiler = ccompiler.new_compiler()
        sysconfig.customize_compiler(self.compiler)
        self.settings = build.get_compiler_setting(False)

    @pytest.mark.gpu
    @pytest.mark.skipif(not test_hip, reason='For ROCm/HIP environment')
    def test_check_hip_version(self):
        with self.assertRaises(RuntimeError):
            build.get_hip_version()
        assert build.check_hip_version(
            self.compiler, self.settings)
        assert isinstance(build.get_hip_version(), int)
        assert isinstance(build.get_hip_version(True), str)

    @pytest.mark.gpu
    @pytest.mark.cudnn
    @pytest.mark.xfail(test_hip,
                       reason='ROCm/HIP DNN support is not ready')
    def test_check_cudnn_version(self):
        with self.assertRaises(RuntimeError):
            build.get_cudnn_version()
        assert build.check_cudnn_version(
            self.compiler, self.settings)
        assert isinstance(build.get_cudnn_version(), int)
        assert isinstance(build.get_cudnn_version(True), str)
