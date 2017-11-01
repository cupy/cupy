from distutils import ccompiler
from distutils import sysconfig
import unittest

import pytest

from install import build


class TestCheckVersion(unittest.TestCase):

    def setUp(self):
        self.compiler = ccompiler.new_compiler()
        sysconfig.customize_compiler(self.compiler)
        self.settings = build.get_compiler_setting()

    @pytest.mark.gpu
    def test_check_cuda_version(self):
        self.assertTrue(build.check_cuda_version(
            self.compiler, self.settings))

    @pytest.mark.gpu
    @pytest.mark.cudnn
    def test_check_cudnn_version(self):
        self.assertTrue(build.check_cudnn_version(
            self.compiler, self.settings))
