from distutils import ccompiler
from distutils import sysconfig
import unittest

import pytest

from install import build


class TestCheckVersion(unittest.TestCase):

    def setUp(self):
        self.compiler = ccompiler.new_compiler()
        sysconfig.customize_compiler(self.compiler)
        self.settings = build.get_compiler_setting(False)

    @pytest.mark.gpu
    def test_check_cuda_version(self):
        with self.assertRaises(RuntimeError):
            build.get_cuda_version()
        self.assertTrue(build.check_cuda_version(
            self.compiler, self.settings))
        self.assertIsInstance(build.get_cuda_version(), int)
        self.assertIsInstance(build.get_cuda_version(True), str)

    @pytest.mark.gpu
    @pytest.mark.cudnn
    def test_check_cudnn_version(self):
        with self.assertRaises(RuntimeError):
            build.get_cudnn_version()
        self.assertTrue(build.check_cudnn_version(
            self.compiler, self.settings))
        self.assertIsInstance(build.get_cudnn_version(), int)
        self.assertIsInstance(build.get_cudnn_version(True), str)
